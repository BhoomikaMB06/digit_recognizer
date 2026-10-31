"""Write the Assignment 4 evaluation report and the project README from the saved results.

Run after prepare_data.py and train.py:    python make_report.py
Then read both files and edit anything that does not match what you see in your own graphs.
"""
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent.parent                       # the Week-04 folder
RES = HERE / 'results'
REL = '../Major-Project/waste_classifier/results'    # how the report (in Week-04/Assignment) finds the images

for need in ['metrics.json', 'dataset_summary.json', 'dataset_summary.md', 'summary.md']:
    if not (RES / need).exists():
        raise SystemExit(f'results/{need} not found. Run prepare_data.py and train.py first.')

m = json.load(open(RES / 'metrics.json'))
d = json.load(open(RES / 'dataset_summary.json'))
summary = (RES / 'summary.md').read_text(encoding='utf-8', errors='replace')
dataset_table = (RES / 'dataset_summary.md').read_text(encoding='utf-8', errors='replace').strip()

classes = m['classes']
n = len(classes)
base, imp = m['models'][0], m['models'][1]
cm = m['confusion_matrix']['improved']
imp_pc = m['per_class']['improved']
n_test = sum(d['split_counts']['test'].values())
margin = 1.96 * math.sqrt(imp['test_acc'] * (1 - imp['test_acc']) / n_test) * 100    # 95% margin, in points


def pct(x):
    return f'{x * 100:.1f}%'


def fmt(x):
    return pct(x) if isinstance(x, (int, float)) else 'n/a'


def section(prefix):
    for part in re.split(r'(?m)^## ', summary):
        if part.startswith(prefix):
            return part.split('\n', 1)[1].strip()
    return '(not found in summary.md)'


def img(keyword, alt, prefix):
    for f in sorted(RES.glob('*.png')):
        if keyword in f.name.lower():
            return f'![{alt}]({prefix}/{f.name})'
    return ''


def per_class_table(key):
    rows = ['| Class | Precision | Recall | F1 | Test images |', '|---|---|---|---|---|']
    for c in classes:
        r = m['per_class'][key][c]
        rows.append(f"| {c} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['f1'])} | {r['support']} |")
    return '\n'.join(rows)


def confusion_table():
    rows = ['| True \\ Predicted | ' + ' | '.join(classes) + ' |', '|---' * (n + 1) + '|']
    for i, c in enumerate(classes):
        rows.append(f'| **{c}** | ' + ' | '.join(str(v) for v in cm[i]) + ' |')
    return '\n'.join(rows)


def gap_sentence(name, key):
    h = m['history'][key]
    i = len(h['val_acc']) - 1
    tr, va = h['train_acc'][i], h['val_acc'][i]
    gap = (tr - va) * 100
    text = (f'At the last epoch the {name} reached {pct(tr)} training accuracy and {pct(va)} validation accuracy '
            f'(difference {gap:+.1f} points). ')
    if gap > 10:
        text += 'This large gap suggests the model is memorising the training images (overfitting).'
    elif gap > 3:
        text += 'This moderate gap shows some overfitting.'
    elif gap >= 0:
        text += 'The gap is small, so there is little sign of overfitting.'
    else:
        text += ('Training accuracy is below validation accuracy, which is normal here because the training images '
                 'are augmented and dropout is active during training.')
    return text


# ---------- facts computed from the results ----------
diff = (imp['test_acc'] - base['test_acc']) * 100
if diff > 0.5:
    compare = (f"The improved CNN scored {pct(imp['test_acc'])} on the test set against {pct(base['test_acc'])} "
               f"for the baseline, which is {diff:.1f} percentage points higher.")
elif diff < -0.5:
    compare = (f"The improved CNN ({pct(imp['test_acc'])}) did not beat the baseline ({pct(base['test_acc'])}) "
               f"in this run, so the extra complexity did not pay off here.")
else:
    compare = (f"The two models scored about the same on the test set ({pct(base['test_acc'])} baseline, "
               f"{pct(imp['test_acc'])} improved).")
noise = (f'The test set has only {n_test} images, so an accuracy of {pct(imp["test_acc"])} has a 95% margin of about '
         f'plus or minus {margin:.1f} points, and differences of a point or two between models are within noise.')

best_cls = max(classes, key=lambda c: imp_pc[c]['recall'])
worst_cls = min(classes, key=lambda c: imp_pc[c]['recall'])
pairs = sorted(((cm[i][j], classes[i], classes[j]) for i in range(n) for j in range(n) if i != j), reverse=True)
top_pairs = [p for p in pairs[:3] if p[0] > 0]
pair_text = '; '.join(f'{t} predicted as {p} ({c} times)' for c, t, p in top_pairs) or 'none'

counts = d['after_dedup_counts']
big, small = max(counts, key=counts.get), min(counts, key=counts.get)
ratio = counts[big] / counts[small]
wrong_high = re.search(r'score of 90% or more: (\d+)', summary)
wrong_high = int(wrong_high.group(1)) if wrong_high else 0
top_conf = re.search(r'(?m)^\| (?!File)[^|]+\|[^|]+\|[^|]+\| ([\d.]+%) \|', section('6. Failure examples'))
top_conf = top_conf.group(1) if top_conf else 'n/a'
if wrong_high > 0:
    score_note = (f'The model made {wrong_high} wrong predictions with a score of 90% or more, so a high score is '
                  f'not a guarantee of a correct answer.')
else:
    score_note = (f'No wrong prediction reached a score of 90% on this test set, but the most confident mistake still '
                  f'scored {top_conf}. This only describes this test set: on real photos that differ from the training '
                  f'photos the model can be confidently wrong, so a score is never a guarantee.')

model_rows = ['| Configuration | Description | Parameters | Training time | Best epoch | Validation acc | Test acc |',
              '|---|---|---|---|---|---|---|']
for label, mod in [('Baseline CNN', base), ('Improved CNN', imp)]:
    model_rows.append(f"| {label} | {mod['description']} | {mod['params']:,} | {mod['seconds'] / 60:.1f} min | "
                      f"{mod.get('best_epoch', 'n/a')} | {fmt(mod.get('val_acc'))} | {fmt(mod['test_acc'])} |")
model_table = '\n'.join(model_rows)

# ---------- Assignment 4: evaluation report ----------
report = f"""# Assignment 4: Final Model Evaluation

Project: Waste Image Classification System (Major Project, Week 4). The code, models and the Gradio app are in
[Major-Project/waste_classifier](../Major-Project/waste_classifier/).

## 1. Dataset source, class distribution and split sizes

**Source:** the Kaggle "Garbage Classification" dataset, a TrashNet-style collection of photos of single waste items
(Thung and Yang, Stanford). Check the licence on the Kaggle page; the images are not stored in this repository.
I used {n} of its 6 categories: {', '.join(classes)}. The sixth category ("trash") has far fewer images and is a
catch-all, so it was left out.

{dataset_table}

{img('distribution', 'Class distribution', REL)}

- **Class balance:** the largest class ({big}, {counts[big]} images) is {ratio:.1f} times the size of the smallest ({small}, {counts[small]} images).
- **Duplicates:** {d['exact_duplicates_removed']} exact duplicates and {d['near_duplicates_removed']} near-duplicates (same picture resized or re-saved, at most {d['near_duplicate_bits']} of 256 hash bits different) were removed, and {d['conflicting_labels_removed']} pictures that appeared under two different labels were dropped. {d['unreadable_skipped']} unreadable files were skipped.
- **Split:** each class was split 70% train, 15% validation and 15% test with a fixed random seed. After copying, every file was hashed again and **{d['shared_between_splits']} identical images appear in more than one split**, so duplicates cannot leak from training into testing.
- **Known weakness:** photos of the same object taken from different angles may still land in different splits, which can make the test score slightly optimistic.

## 2. Model configurations and results

{model_table}

{compare} {noise}

## 3. Training curves and overfitting

{img('curve', 'Training curves', REL)}

- {gap_sentence('baseline CNN', 'baseline')}
- {gap_sentence('improved CNN', 'improved')}

## 4. Per-class results (improved CNN)

{per_class_table('improved')}

The easiest class was **{best_cls}** (recall {pct(imp_pc[best_cls]['recall'])}) and the hardest was **{worst_cls}** (recall {pct(imp_pc[worst_cls]['recall'])}).

{img('confusion', 'Confusion matrix', REL)}

{confusion_table()}

Most common confusions: {pair_text}.

## 5. Failure examples

{section('6. Failure examples')}

## 6. Confidence of the mistakes

{score_note}

## 7. Suggested improvements

1. **More and more varied data for the weakest classes.** {worst_cls} has the lowest recall, and {small} is the smallest class ({counts[small]} images). Adding photos with different backgrounds, lighting and angles should help most here.
2. **Fix the most common confusions.** The biggest mistakes were: {pair_text}. Collecting extra examples of these pairs, or adding class weights so errors on the weaker class cost more, targets them directly.
3. **Stronger augmentation.** Random crops, colour jitter and rotation make the model less dependent on clean studio backgrounds.
4. **Transfer learning.** Fine-tuning a pretrained network such as ResNet18 or MobileNet usually beats a small CNN trained from scratch on a dataset this size.
5. **Group-aware splitting.** Keeping photos of the same object in one split would remove the optimism noted in section 1.
6. **Confidence threshold.** Show "not sure" when the top score is low, so uncertain predictions are not presented as answers.

## 8. Limitations and conclusion

- The dataset has clean studio-style photos with plain backgrounds. Real photos of waste (cluttered, dirty, crushed, partly hidden) will be harder, so the real-world accuracy is probably lower than the test score.
- Only {n} material classes are covered, and each photo shows one item. A photo with several items or an unknown material will still get one of the {n} labels.
- The test set is small ({n_test} images), so small differences are within noise.
- The weakest class ({worst_cls}) and the common confusions above are where more data or better augmentation would help first.

**Conclusion:** the final model reached {pct(imp['test_acc'])} test accuracy. It works well on this dataset, but it should be treated as a classroom prototype and not as a tool that is ready for real sorting.
"""

# ---------- project README ----------
app = next((f for f in ['app.py', 'gradio_app.py'] if (HERE / f).exists()), 'app.py')
readme = f"""# Waste Image Classification System

A small CNN that classifies a photo of a waste item into one of {n} classes: {', '.join(classes)}.
Built with PyTorch, with a Gradio web app for trying it on your own photos.

## Results

| Model | Parameters | Test accuracy |
|---|---|---|
| Baseline CNN | {base['params']:,} | {pct(base['test_acc'])} |
| Improved CNN | {imp['params']:,} | {pct(imp['test_acc'])} |

The weakest class is **{worst_cls}** (recall {pct(imp_pc[worst_cls]['recall'])}). Common confusions: {pair_text}.
The test set has {n_test} images, so the 95% margin on the accuracy is about plus or minus {margin:.1f} points.

{img('confusion', 'Confusion matrix', 'results')}

Full analysis: [Assignment 4 report](../../Assignment/assignment4_report.md).

## How to run
1. Download the Kaggle "Garbage Classification" dataset and put it where `prepare_data.py` expects it.
2. `python prepare_data.py` cleans the data, removes duplicates and makes the train/validation/test split.
3. `python train.py` trains both models and writes everything to `results/`.
4. `python {app}` starts the Gradio app.
5. `python make_report.py` rebuilds this README and the Assignment 4 report from the results.

## Limitations
- Trained on clean single-item photos, so cluttered real-world photos will be harder.
- Only {n} classes; anything else is forced into one of them.
- {score_note}
"""

out_dir = WEEK / 'Assignment'
out_dir.mkdir(exist_ok=True)
(out_dir / 'assignment4_report.md').write_text(report, encoding='utf-8')
(HERE / 'README.md').write_text(readme, encoding='utf-8')
print('Wrote', out_dir / 'assignment4_report.md')
print('Wrote', HERE / 'README.md')