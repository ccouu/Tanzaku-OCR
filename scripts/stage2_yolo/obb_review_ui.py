"""Compatibility entrypoint for src.detection.review.obb_review_ui."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("src.detection.review.obb_review_ui", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("src.detection.review.obb_review_ui")
