"""Compatibility entrypoint for src.detection.predict_initial_boxes."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("src.detection.predict_initial_boxes", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("src.detection.predict_initial_boxes")
