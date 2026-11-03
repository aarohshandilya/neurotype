"""Inline engine.js into the page. Writes neurotype.html (publish) and index.html (open locally)."""
from pathlib import Path

here = Path(__file__).parent
page = (here / "template.html").read_text().replace("/*ENGINE*/", (here / "engine.js").read_text())
(here / "neurotype.html").write_text(page)
(here / "index.html").write_text('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
                                 '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' + page)
