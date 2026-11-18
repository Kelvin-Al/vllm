"""Render the changed feature guide and the repository-generated example."""

import argparse
import io
import os
import runpy
import tempfile
from pathlib import Path
from unittest.mock import patch

from mkdocs.commands.build import build
from mkdocs.config import load_config


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, required=True)
args = parser.parse_args()
repo = args.repo.resolve()
os.chdir(repo)
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    docs = root / "docs"
    example = Path("examples/disaggregated/disaggregated_encoder.md")
    feature = Path("features/disagg_encoder.md")
    (docs / feature.parent).mkdir(parents=True)
    (docs / feature).write_bytes((repo / "docs" / feature).read_bytes())

    def capture(name, *unused, **kwargs):
        if Path(name) != example:
            return io.StringIO()
        path = docs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("w", encoding="utf-8")

    with patch("mkdocs_gen_files.open", capture), patch("mkdocs_gen_files.set_edit_path"):
        runpy.run_path(str(repo / "docs/mkdocs/gen_files/generate_examples.py"))
    config_path = root / "mkdocs.yml"
    config_path.write_text("site_name: EPD render check\nplugins: []\ntheme: mkdocs\n", encoding="utf-8")
    config = load_config(config_file=str(config_path))
    build(config)
    site = Path(config.site_dir)
    feature_html = (site / "features/disagg_encoder/index.html").read_text(encoding="utf-8")
    example_html = (site / "examples/disaggregated/disaggregated_encoder/index.html").read_text(encoding="utf-8")
    assert "Size an E+PD deployment" in feature_html and "size_epd.py" in example_html
    assert "../../examples/disaggregated/disaggregated_encoder/" in feature_html
print("PASS: rendered feature guide, generated example, and new navigation link")
