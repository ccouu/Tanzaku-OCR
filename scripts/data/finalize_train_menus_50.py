"""Verify and finalize all 50 generated training menu images and JSON sidecars."""
import hashlib
import json
from pathlib import Path
from PIL import Image

OUT_DIR = Path(r"D:\profolio\japanese_translation\data\raw\gemini_banana_mobile_menus_train_50")


def main():
    items = []
    hashes = set()
    sizes = set()
    for idx in range(1, 51):
        img_path = OUT_DIR / f"menu-{idx:03d}.jpg"
        json_path = OUT_DIR / f"menu-{idx:03d}.json"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")
        if not json_path.exists():
            raise FileNotFoundError(f"Missing sidecar: {json_path}")

        with Image.open(img_path) as im:
            im.verify()
        with Image.open(img_path) as im:
            w, h = im.size
        sha = hashlib.sha256(img_path.read_bytes()).hexdigest()
        if sha in hashes:
            raise ValueError(f"Duplicate SHA-256 detected on {img_path.name}")
        hashes.add(sha)
        sizes.add((w, h))

        meta = json.loads(json_path.read_text(encoding="utf-8"))
        meta["status"] = "completed"
        meta["size"] = [w, h]
        meta["sha256"] = sha
        json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        items.append(meta)

    (OUT_DIR / "prompts.jsonl").write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n",
        encoding="utf-8",
    )
    manifest_path = OUT_DIR / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["completed"] = len(items)
    manifest["items"] = items
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"FINALIZED_50_MENUS count={len(items)} unique_hashes={len(hashes)} sizes={sorted(sizes)}")


if __name__ == "__main__":
    main()
