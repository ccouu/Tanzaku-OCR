"""Reusable CLI for menu augmentation and restaurant-scene compositing."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.menu_augmentation import main

if __name__ == '__main__':
    main()
