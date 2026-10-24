import random
import shutil
import sys
from pathlib import Path

# Usage: python prepare_data.py "<dataset folder>" [name prefix, e.g. Tomato]
if len(sys.argv) < 2:
    sys.exit('Usage: python prepare_data.py "<dataset folder>" [prefix, e.g. Tomato]')

SOURCE = Path(sys.argv[1])
PREFIX = sys.argv[2].lower() if len(sys.argv) > 2 else ""
OUT = Path("dataset")
NUM_CLASSES = 6
IMAGES_PER_CLASS = 200
MIN_IMAGES = 50
IMAGE_TYPES = {".jpg", ".jpeg", ".png"}
SPLIT_NAMES = {"train", "training", "val", "valid", "validation", "test", "testing"}

random.seed(42)

if not SOURCE.is_dir():
    sys.exit(f"Folder not found: {SOURCE}")

# Find every folder that directly contains images. The folder name is the class name.
found = {}
for folder in SOURCE.rglob("*"):
    if not folder.is_dir() or folder.name.lower() in SPLIT_NAMES:
        continue
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_TYPES]
    if imgs:
        found.setdefault(folder.name, []).extend(imgs)

if not found:
    sys.exit("No images (.jpg, .jpeg, .png) found anywhere under that folder.")

print("Folders with images found:")
for name in sorted(found):
    print(f"  {name}: {len(found[name])} images")

names = sorted(n for n in found if n.lower().startswith(PREFIX) and len(found[n]) >= MIN_IMAGES)
if not names and PREFIX:
    print(f"\nNo class names start with '{PREFIX}'. Using all folders instead.")
    names = sorted(n for n in found if len(found[n]) >= MIN_IMAGES)
if not names:
    sys.exit(f"No folder has at least {MIN_IMAGES} images.")

chosen = names[:NUM_CLASSES]
print("\nUsing these classes:", ", ".join(chosen))

if OUT.exists():
    shutil.rmtree(OUT)

for name in chosen:
    images = found[name]
    random.shuffle(images)
    images = images[:IMAGES_PER_CLASS]

    n_train = int(len(images) * 0.70)
    n_val = int(len(images) * 0.15)
    parts = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }
    for split, files in parts.items():
        dest = OUT / split / name
        dest.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(files):
            shutil.copy2(f, dest / f"{i}_{f.name}")
    print(f"{name}: train={len(parts['train'])}, val={len(parts['val'])}, test={len(parts['test'])}")

print("\nDone. Images are in the 'dataset' folder.")