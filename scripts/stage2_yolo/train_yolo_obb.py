"""Compatibility entrypoint for src.detection.train_yolo_obb."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("src.detection.train_yolo_obb", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("src.detection.train_yolo_obb")
