# -*- coding: utf-8 -*-
"""HTML template utilities for report generation."""

from pathlib import Path
from string import Template

TEMPLATE_DIR = Path(__file__).parent


def load_template(name: str) -> Template:
    """Load an HTML template by filename."""
    path = TEMPLATE_DIR / name
    return Template(path.read_text(encoding="utf-8"))
