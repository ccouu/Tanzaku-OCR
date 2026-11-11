"""Collect menu or interior photos: one image per distinct Tabelog restaurant."""
from pathlib import Path
import sys

# Support both `python -m scripts.data.tabelog_scraper` and direct file execution.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.tabelog_collection import main

if __name__ == '__main__':
    raise SystemExit(main())
