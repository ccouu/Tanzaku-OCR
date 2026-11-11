"""One-time source and documentation layout migration."""
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / 'src/detection').exists():
    raise SystemExit('Source layout already migrated; refusing to repeat the one-time refactor')


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def move(old, new):
    source, target = ROOT / old, ROOT / new
    if target.exists():
        raise ValueError(f'Destination exists: {new}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))


# Extract shared helpers while retaining the previous import surface.
common = (ROOT / 'src/data/common.py').read_text(encoding='utf-8')
common = common.replace('ROOT = Path(__file__).resolve().parents[2]',
                        'from .paths import ROOT, resolve_path')
common = common.replace('def digest(path):\n', 'def digest(path):\n    path = resolve_path(path)\n')
write(ROOT / 'src/common/io.py', common)
write(ROOT / 'src/data/common.py',
      '"""Compatibility imports; shared helpers now live in src.common."""\n'
      'from src.common import ROOT, digest, dump_json, safe_path\n')

# Stage 2 reusable modules; legacy module objects alias implementations so
# monkeypatching/import identity and private helper imports remain compatible.
write(ROOT / 'src/detection/__init__.py', '"""Menu text-boundary detection workflows."""\n')
write(ROOT / 'src/detection/review/__init__.py', '"""Local oriented-box review tools."""\n')
legacy = ROOT / 'scripts/stage2_yolo'
for source in sorted(legacy.glob('*.py')):
    if source.name == '__init__.py':
        continue
    name = source.stem
    module = 'src.detection.' + ('review.obb_review_ui' if name == 'obb_review_ui' else name)
    destination = ROOT / (module.replace('.', '/') + '.py')
    text = source.read_text(encoding='utf-8')
    if name == 'obb_review_ui':
        text = text.replace('from .common import', 'from ..common import')
    text = text.replace('from src.data.common import', 'from src.common import')
    write(destination, text)
    write(source, f'"""Compatibility entrypoint for {module}."""\n'
          'import importlib\nimport runpy\nimport sys\n\n'
          f'if __name__ == "__main__":\n    runpy.run_module("{module}", run_name="__main__")\n'
          f'else:\n    sys.modules[__name__] = importlib.import_module("{module}")\n')
move('scripts/stage2_yolo/review.html', 'src/detection/review/review.html')

# Apply centralized paths only to source, never to historical data artifacts.
for base in ('src', 'scripts', 'tests'):
    for path in (ROOT / base).rglob('*.py'):
        text = path.read_text(encoding='utf-8')
        if base == 'src' and 'common' in path.relative_to(ROOT / base).parts:
            continue
        text = text.replace('from src.data.common import', 'from src.common import')
        # Defaults and public CLI arguments continue accepting historical paths.
        if path.name in ('synthetic_renderer.py', 'lexicon_builder.py', 'download_fonts.py',
                          'verify_stage1_dataset.py', 'finalize_menu_image_collection.py', 'merge_real_dataset.py'):
            text = text.replace("ROOT / 'data'", 'SYNTHETIC_ROOT')
            text = 'from src.common.paths import SYNTHETIC_ROOT, asset, dataset_root\n' + text
            text = text.replace('root = Path(root)', 'root = dataset_root(root)')
            text = text.replace('data_root = Path(data_root)', 'data_root = dataset_root(data_root)')
            text = text.replace('root, reviewed, output = Path(root).resolve()', 'root, reviewed, output = dataset_root(root).resolve()')
        # Shared resources are no longer assumed to sit within the synthetic bundle.
        if path.name in ('synthetic_renderer.py', 'lexicon_builder.py', 'menu_grammar.py',
                          'download_fonts.py', 'verify_stage1_dataset.py', 'finalize_menu_image_collection.py'):
            if 'import SYNTHETIC_ROOT, asset' not in text:
                text = 'from src.common.paths import asset\n' + text
            text = re.sub(r"\b(root|data_root)\s*/\s*'((?:fonts|raw|lexicon)(?:/[^']*)?)'",
                          lambda m: f"asset({m[1]}, '{m[2]}')", text)
        # Canonical path literals in active code and fixtures (not provenance data).
        replacements = {
            'data/fonts': 'data/resources/fonts',
            'data/lexicon': 'data/processed/lexicon',
            'data/augmentation_500': 'data/processed/augmentation/run_500',
            'data/augmentation': 'data/processed/augmentation/run_1100',
            'data/yolo_obb': 'data/processed/synthetic_base_v1/yolo_obb',
            'data/stage2_training_v2': 'data/datasets/detection/reviewed_v2',
            'data/stage2_training': 'data/datasets/detection/reviewed_v1',
            'data/stage2_runtime': '.cache/ultralytics',
            'data/stage2': 'data/annotations/menu_obb',
            'data/test-runs': '.cache/test-runs',
            'outputs/stage2_colab_v2': 'outputs/detection/training/colab_v2',
            'outputs/stage2_colab': 'outputs/detection/training/colab_v1',
            'outputs/stage2_local_test_100': 'outputs/detection/inference/tabelog_100',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        if path.name == 'menu_grammar.py':
            text = text.replace('root = Path(data_root)', 'root = Path(data_root)')
        write(path, text)

for name in ('test_dataset_audit.py', 'test_lexicon_and_price.py', 'test_menu_image_batch.py',
             'test_menu_augmentation.py', 'test_synthetic_renderer.py', 'test_tabelog_scraper.py'):
    move('tests/' + name, 'tests/data/' + name)
for name in ('test_stage2.py', 'test_stage2_training.py'):
    move('tests/' + name, 'tests/detection/' + name)

move('notebooks/colab_stage2_yolo_obb.ipynb', 'notebooks/stage2_detection/colab_stage2_yolo_obb.ipynb')
for source in list((ROOT / 'doc').glob('*.md')):
    group = 'stage1_data' if source.name in ('stage1_summary.md', 'menu_augmentation.md', 'tabelog_scraper_review.md') else 'stage2_detection'
    if source.name == 'japanese_menu_ocr_implementation_plan.md':
        target = 'docs/overview.md'
    else:
        target = f'docs/{group}/{source.name}'
    move(source.relative_to(ROOT).as_posix(), target)
move('WORKFLOW.md', 'docs/orchestration.md')

print('Source, tests, notebook and documentation regrouped')
