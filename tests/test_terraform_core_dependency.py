"""Terraform/HCL parsing is a core capability, not an optional extra.

tree-sitter-hcl used to live behind the [terraform] extra, so a default
`uv tool install graphifyy` / `pipx install graphifyy` produced a graph with
every .tf/.hcl file missing: the extractor bailed with an error and the #1745
warning was the only signal. These tests pin the fix at the packaging layer so
a regression (the dependency sliding back into an extra) fails the suite
rather than resurfacing as a field report, and they pin the two behaviours the
promotion must not disturb: structural extraction from a default install, and
the routing of `*.tf.json` through the JSON extractor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from graphify import extract as extractmod
from graphify.extract import extract

REPO_ROOT = Path(__file__).resolve().parent.parent

MAIN_TF = """\
variable "bucket_prefix" {
  type    = string
  default = "example"
}

resource "aws_s3_bucket" "logs" {
  bucket = "example-logs"
}

module "network" {
  source = "./modules/network"
  prefix = var.bucket_prefix
}

output "logs_bucket_arn" {
  value = aws_s3_bucket.logs.arn
}
"""

# Terraform's JSON syntax, as a hand-written or generated legacy file.
LEGACY_TF_JSON = {
    "variable": {"region": {"type": "string", "default": "eu-west-1"}},
    "resource": {
        "aws_sqs_queue": {
            "events": {
                "name": "example-events",
                "tags": {"region": "${var.region}"},
            }
        }
    },
    "output": {"events_queue_arn": {"value": "${aws_sqs_queue.events.arn}"}},
}


def _pyproject() -> dict:
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


def _is_hcl(requirement: str) -> bool:
    return requirement.replace("_", "-").startswith("tree-sitter-hcl")


def _block_hcl_grammar(monkeypatch) -> None:
    """Make the grammar genuinely unavailable: import fails and find_spec says absent."""
    import importlib.util as _ilu

    monkeypatch.setitem(sys.modules, "tree_sitter_hcl", None)  # import -> ImportError
    _real_find_spec = _ilu.find_spec
    monkeypatch.setattr(
        _ilu, "find_spec",
        lambda name, *a, **k: None if name == "tree_sitter_hcl" else _real_find_spec(name, *a, **k),
    )


def _normalised(result: dict, root: Path, rename: dict[str, str] | None = None) -> str:
    """Stable JSON of an extract() result with tmp paths reduced to relative names."""
    root = root.resolve()
    rename = rename or {}

    def rel(p: str) -> str:
        name = Path(p).resolve().relative_to(root).as_posix()
        return rename.get(name, name)

    out = {
        "nodes": result["nodes"],
        "edges": result["edges"],
        "failed_sources": sorted(rel(p) for p in result.get("failed_sources", [])),
        "skipped_sources": sorted(rel(p) for p in result.get("skipped_sources", [])),
        "extracted_sources": sorted(rel(p) for p in result.get("extracted_sources", [])),
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
    }
    return json.dumps(out, sort_keys=True, indent=1)


# ── packaging ─────────────────────────────────────────────────────────────────


def test_tree_sitter_hcl_is_a_core_dependency():
    """The grammar must be in [project.dependencies], not only in an extra."""
    deps = _pyproject()["project"]["dependencies"]
    assert [d for d in deps if _is_hcl(d)], (
        "tree-sitter-hcl is missing from [project.dependencies]; a default "
        "install would silently skip every .tf/.hcl file again (#1745)"
    )


def test_tree_sitter_hcl_core_pin_has_an_upper_bound():
    """The core pin must carry an upper bound like every other grammar pin."""
    deps = _pyproject()["project"]["dependencies"]
    pins = [d for d in deps if _is_hcl(d)]
    assert pins, "tree-sitter-hcl is missing from [project.dependencies]"
    assert "<" in pins[0], f"tree-sitter-hcl core pin has no upper bound: {pins[0]!r}"


def test_terraform_extra_remains_as_an_alias():
    """`graphifyy[terraform]` must keep resolving for existing install commands."""
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "terraform" in extras, "the [terraform] extra was removed; keep it as an alias"
    assert [d for d in extras["terraform"] if _is_hcl(d)]


def test_dev_group_does_not_mask_a_missing_core_dependency():
    """A grammar listed only in the dev group would hide a core-dependency gap in CI."""
    dev = _pyproject()["dependency-groups"]["dev"]
    assert not [d for d in dev if isinstance(d, str) and _is_hcl(d)], (
        "tree-sitter-hcl is redundant in [dependency-groups].dev now that it is "
        "core, and listing it there would mask a regression of the core pin"
    )


@pytest.mark.parametrize("suffix", [".tf", ".tfvars", ".hcl"])
def test_hcl_extensions_are_not_mapped_to_an_optional_extra(suffix):
    """#1745's hint map must not send users to a now-redundant [terraform] extra."""
    assert suffix not in extractmod._EXTRA_FOR_EXTENSION, (
        f"_EXTRA_FOR_EXTENSION still maps {suffix} to an extra; the 'install "
        "graphifyy[terraform]' hint is wrong now that the grammar is core"
    )


def test_tree_sitter_hcl_imports_in_this_environment():
    """The dev environment itself must satisfy the core dependency."""
    import tree_sitter_hcl  # noqa: F401


# ── default-install extraction ────────────────────────────────────────────────


def test_terraform_corpus_produces_structural_nodes_and_edges(tmp_path, capsys):
    """A default install extracts blocks and the references between them."""
    tf = tmp_path / "main.tf"
    tf.write_text(MAIN_TF, encoding="utf-8")

    r = extract([tf], cache_root=tmp_path / "cache", parallel=False)
    err = capsys.readouterr().err

    labels = {n["label"] for n in r["nodes"]}
    assert {
        "main.tf", "var.bucket_prefix", "aws_s3_bucket.logs",
        "module.network", "output.logs_bucket_arn",
    } <= labels, labels

    by_id = {n["id"]: n["label"] for n in r["nodes"]}
    pairs = {(e["relation"], by_id.get(e["source"]), by_id.get(e["target"])) for e in r["edges"]}
    assert ("contains", "main.tf", "aws_s3_bucket.logs") in pairs
    assert ("references", "output.logs_bucket_arn", "aws_s3_bucket.logs") in pairs
    assert ("references", "module.network", "var.bucket_prefix") in pairs

    node_ids = set(by_id)
    for e in r["edges"]:
        assert e["source"] in node_ids, f"dangling edge source: {e['source']}"
        assert e["target"] in node_ids, f"dangling edge target: {e['target']}"

    assert "#1745" not in err
    assert r["failed_sources"] == []


# ── *.tf.json non-regression (guard: must hold with and without the grammar) ──


def test_tf_json_routes_to_the_json_extractor_not_terraform(tmp_path):
    """effective_suffix(x.tf.json) is .json, so dispatch is the JSON extractor."""
    from graphify.extractors.json_config import extract_json

    tf_json = tmp_path / "legacy.tf.json"
    tf_json.write_text(json.dumps(LEGACY_TF_JSON), encoding="utf-8")
    plain_json = tmp_path / "legacy_config.json"
    plain_json.write_text(json.dumps(LEGACY_TF_JSON), encoding="utf-8")

    assert extractmod._get_extractor(tf_json) is extract_json
    assert extractmod._get_extractor(tf_json) is extractmod._get_extractor(plain_json)
    assert extractmod._get_extractor(tf_json) is not extractmod.extract_terraform


@pytest.mark.parametrize("grammar", ["available", "blocked"])
def test_tf_json_never_reaches_the_terraform_extractor(tmp_path, capsys, monkeypatch, grammar):
    """extract([x.tf.json]) succeeds even if the Terraform extractor would raise."""
    if grammar == "blocked":
        _block_hcl_grammar(monkeypatch)

    def _boom(*_a, **_k):
        raise AssertionError("extract_terraform must not be called for *.tf.json")

    import graphify.extractors.terraform as tfmod

    monkeypatch.setattr(tfmod, "extract_terraform", _boom)
    monkeypatch.setattr(extractmod, "extract_terraform", _boom)
    for suffix in (".tf", ".tfvars", ".hcl"):
        monkeypatch.setitem(extractmod._DISPATCH, suffix, _boom)

    tf_json = tmp_path / "legacy.tf.json"
    tf_json.write_text(json.dumps(LEGACY_TF_JSON), encoding="utf-8")

    r = extract([tf_json], cache_root=tmp_path / "cache", parallel=False)
    err = capsys.readouterr().err

    # Golden: data-shaped JSON contributes nothing and is listed as skipped.
    assert json.loads(_normalised(r, tmp_path)) == {
        "nodes": [],
        "edges": [],
        "failed_sources": [],
        "skipped_sources": ["legacy.tf.json"],
        "extracted_sources": ["legacy.tf.json"],
        "input_tokens": 0,
        "output_tokens": 0,
    }
    assert "#1745" not in err


def test_tf_json_result_equals_the_same_content_as_plain_json(tmp_path):
    """Being named *.tf.json changes nothing about how the content is extracted."""
    tf_json = tmp_path / "legacy.tf.json"
    tf_json.write_text(json.dumps(LEGACY_TF_JSON), encoding="utf-8")
    plain_json = tmp_path / "legacy_config.json"
    plain_json.write_text(json.dumps(LEGACY_TF_JSON), encoding="utf-8")

    r_tf = extract([tf_json], cache_root=tmp_path / "cache_a", parallel=False)
    r_plain = extract([plain_json], cache_root=tmp_path / "cache_b", parallel=False)

    assert _normalised(r_tf, tmp_path, {"legacy.tf.json": "X"}) == _normalised(
        r_plain, tmp_path, {"legacy_config.json": "X"}
    )


def test_tf_json_result_is_identical_with_and_without_the_hcl_grammar(tmp_path, monkeypatch):
    """The HCL grammar being present or absent must not change the .tf.json result."""
    tf_json = tmp_path / "legacy.tf.json"
    tf_json.write_text(json.dumps(LEGACY_TF_JSON), encoding="utf-8")

    with_grammar = extract([tf_json], cache_root=tmp_path / "cache_a", parallel=False)
    _block_hcl_grammar(monkeypatch)
    without_grammar = extract([tf_json], cache_root=tmp_path / "cache_b", parallel=False)

    assert _normalised(with_grammar, tmp_path) == _normalised(without_grammar, tmp_path)


def test_tf_json_next_to_main_tf_adds_nothing_and_warns_nothing(tmp_path, capsys):
    """In a mixed directory only main.tf contributes; legacy.tf.json adds no node."""
    tf = tmp_path / "main.tf"
    tf.write_text(MAIN_TF, encoding="utf-8")
    tf_json = tmp_path / "legacy.tf.json"
    tf_json.write_text(json.dumps(LEGACY_TF_JSON), encoding="utf-8")

    only_tf = extract([tf], cache_root=tmp_path / "cache_a", parallel=False)
    capsys.readouterr()
    mixed = extract([tf, tf_json], cache_root=tmp_path / "cache_b", parallel=False)
    err = capsys.readouterr().err

    def by_id(r: dict, key: str) -> list:
        return sorted(r[key], key=lambda x: json.dumps(x, sort_keys=True))

    assert by_id(mixed, "nodes") == by_id(only_tf, "nodes")
    assert by_id(mixed, "edges") == by_id(only_tf, "edges")
    assert any(n["label"] == "aws_s3_bucket.logs" for n in mixed["nodes"])
    assert mixed["failed_sources"] == []
    assert {Path(p).name for p in mixed["skipped_sources"]} == {"legacy.tf.json"}
    assert "#1745" not in err
