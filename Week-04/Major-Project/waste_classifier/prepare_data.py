"""Prepare the waste dataset: pick classes, remove duplicates, split into train / val / test.

Usage:  python prepare_data.py "<folder that contains (or sits above) the class folders>"
"""
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic']    # the 5 chosen waste categories
SPLITS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
NEAR_DUPLICATE_BITS = 10      # two images differing in at most this many of 256 hash bits count as duplicates
IMAGE_TYPES = {'.jpg', '.jpeg', '.png'}

HERE = Path(__file__).resolve().parent
OUT = HERE / 'dataset'
RESULTS = HERE / 'results'
random.seed(42)

if len(sys.argv) < 2:
    sys.exit('Usage: python prepare_data.py "<folder with the class folders>"')
SOURCE = Path(sys.argv[1])
if not SOURCE.is_dir():
    sys.exit(f'Folder not found: {SOURCE}')

# ---------- 1. Collect the images of the chosen classes ----------
found = defaultdict(list)
for folder in SOURCE.rglob('*'):
    if folder.is_dir() and folder.name.lower() in CLASSES:
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_TYPES:
                found[folder.name.lower()].append(p)

missing = [c for c in CLASSES if not found[c]]
if missing:
    sys.exit(f'No images found for: {missing}. Point the script at the folder that contains the class folders.')

raw_counts = {c: len(found[c]) for c in CLASSES}
print('Raw images per class:', raw_counts)


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


# ---------- 2. Remove exact duplicates (identical files) ----------
by_hash = defaultdict(list)
for cls in CLASSES:
    for p in found[cls]:
        by_hash[md5(p)].append((cls, p))

items, exact_removed, label_conflicts = [], 0, 0
for entries in by_hash.values():
    if len({c for c, _ in entries}) > 1:        # the same picture filed under two classes: unreliable, drop it
        label_conflicts += len(entries)
        continue
    items.append(entries[0])
    exact_removed += len(entries) - 1
print(f'Exact duplicates removed: {exact_removed} | pictures with conflicting labels removed: {label_conflicts}')


# ---------- 3. Remove near-duplicates (same picture, resized or re-saved) ----------
def dhash(path, size=16):
    """Difference hash: 256 booleans describing how brightness changes from left to right."""
    img = Image.open(path).convert('L').resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


readable, hashes, unreadable = [], [], 0
for cls, p in items:
    try:
        hashes.append(dhash(p))
        readable.append((cls, p))
    except Exception:
        unreadable += 1
items = readable
hashes = np.array(hashes)
print(f'Unreadable images skipped: {unreadable}')

parent = list(range(len(items)))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for i in range(len(items) - 1):
    dist = (hashes[i + 1:] != hashes[i]).sum(axis=1)
    for j in np.nonzero(dist <= NEAR_DUPLICATE_BITS)[0]:
        a, b = find(i), find(i + 1 + int(j))
        if a != b:
            parent[b] = a

groups = defaultdict(list)
for i in range(len(items)):
    groups[find(i)].append(i)

kept, near_removed, near_conflicts = [], 0, 0
for members in groups.values():
    if len({items[i][0] for i in members}) > 1:     # near-identical pictures with different labels
        near_conflicts += len(members)
        continue
    kept.append(items[members[0]])
    near_removed += len(members) - 1
print(f'Near-duplicates removed: {near_removed} | near-duplicates with conflicting labels removed: {near_conflicts}')

# ---------- 4. Split each class into train / val / test ----------
by_class = defaultdict(list)
for cls, p in kept:
    by_class[cls].append(p)

if OUT.exists():
    shutil.rmtree(OUT)
counts = {s: {} for s in SPLITS}
for cls in CLASSES:
    paths = sorted(by_class[cls])
    random.shuffle(paths)
    n_train = int(len(paths) * SPLITS['train'])
    n_val = int(len(paths) * SPLITS['val'])
    parts = {'train': paths[:n_train], 'val': paths[n_train:n_train + n_val], 'test': paths[n_train + n_val:]}
    for split, files in parts.items():
        dest = OUT / split / cls
        dest.mkdir(parents=True, exist_ok=True)
        for k, f in enumerate(files):
            shutil.copy2(f, dest / f'{cls}_{k:04d}{f.suffix.lower()}')
        counts[split][cls] = len(files)

# ---------- 5. Prove that no identical image sits in two splits ----------
split_hashes = {s: {md5(p) for p in (OUT / s).rglob('*') if p.is_file()} for s in SPLITS}
overlap = sum(len(split_hashes[a] & split_hashes[b])
              for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')])
print(f'Identical images shared between splits: {overlap}')
if overlap:
    sys.exit('Leakage found: identical images appear in more than one split.')

# ---------- 6. Save the dataset summary (used later in the evaluation report) ----------
RESULTS.mkdir(exist_ok=True)
kept_counts = {c: len(by_class[c]) for c in CLASSES}
summary = {
    'classes': CLASSES, 'raw_counts': raw_counts, 'after_dedup_counts': kept_counts, 'split_counts': counts,
    'exact_duplicates_removed': exact_removed, 'near_duplicates_removed': near_removed,
    'conflicting_labels_removed': label_conflicts + near_conflicts, 'unreadable_skipped': unreadable,
    'near_duplicate_bits': NEAR_DUPLICATE_BITS, 'shared_between_splits': overlap,
}
(RESULTS / 'dataset_summary.json').write_text(json.dumps(summary, indent=2))

lines = ['| Class | Raw images | After duplicate removal | Train | Val | Test |', '|---|---|---|---|---|---|']
for c in CLASSES:
    lines.append(f"| {c} | {raw_counts[c]} | {kept_counts[c]} | {counts['train'][c]} | {counts['val'][c]} | {counts['test'][c]} |")
totals = [sum(raw_counts.values()), sum(kept_counts.values())] + [sum(counts[s].values()) for s in SPLITS]
lines.append('| **Total** | ' + ' | '.join(f'**{t}**' for t in totals) + ' |')
(RESULTS / 'dataset_summary.md').write_text('\n'.join(lines) + '\n')
print('\n' + '\n'.join(lines))

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(CLASSES))
for k, split in enumerate(SPLITS):
    ax.bar(x + (k - 1) * 0.27, [counts[split][c] for c in CLASSES], width=0.27, label=split)
ax.set_xticks(x)
ax.set_xticklabels(CLASSES)
ax.set_ylabel('Number of images')
ax.set_title('Class distribution per split')
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / 'class_distribution.png', dpi=120)
plt.close(fig)
print('\nSaved results/dataset_summary.json, dataset_summary.md and class_distribution.png')