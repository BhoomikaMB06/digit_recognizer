"""Simple Gradio demo: upload a photo of one waste item and see the predicted category and model score.

Run:  python app.py      then open http://127.0.0.1:7860
"""
import json

import gradio as gr

from common import HERE, MODEL_FILES, MODELS_DIR, load_model, predict_probs

CLASSES = json.load(open(HERE / 'classes.json'))
MODEL_CHOICES = {'Improved CNN (MobileNetV2)': 'improved', 'Baseline CNN': 'baseline'}
DEFAULT_CHOICE = list(MODEL_CHOICES)[0]
LOW_SCORE = 0.60

DESCRIPTION = (
    'Upload a photo of **one** waste item. The model chooses one of these categories: '
    + ', '.join(CLASSES) + '.\n\n'
    '**Limitations:** the models were trained on a small public dataset of single items on plain backgrounds. '
    'They can fail on cluttered scenes, dirty, crushed or mixed-material items, and on objects outside these '
    'categories. **A high model score does not guarantee the answer is correct.** '
    'This demo is for learning, not for real waste-sorting decisions.'
)

missing = [f for f in MODEL_FILES.values() if not (MODELS_DIR / f).exists()]
if missing:
    raise SystemExit(f'Model file(s) missing: {missing}. Run train.py first.')

_loaded = {}


def get_model(choice):
    kind = MODEL_CHOICES[choice]
    if kind not in _loaded:
        _loaded[kind] = load_model(kind, len(CLASSES))
    return _loaded[kind]


def classify(image, model_choice=None):
    if image is None:
        return None, 'Please upload a photo first.'
    # Gradio can send None when the model selector is empty, so fall back to the default model
    used = model_choice if model_choice in MODEL_CHOICES else DEFAULT_CHOICE
    probs = predict_probs(get_model(used), image)
    scores = {name: float(p) for name, p in zip(CLASSES, probs)}
    top = max(scores, key=scores.get)
    top_score = scores[top]
    lines = [f'**Predicted category: {top}** (model score {top_score * 100:.1f}%)', '',
             f'Model used: {used}', '']
    if top_score < LOW_SCORE:
        lines += ['**Low score:** the model is unsure, so treat this answer with extra caution.', '']
    lines.append(
        f'*The score shows how strongly the model prefers this category. It is not the chance that the answer '
        f'is correct: the model can be wrong with a high score, and it must always pick one of the '
        f'{len(CLASSES)} categories, even for objects it has never seen.*')
    return scores, '\n'.join(lines)


def find_examples():
    """One test image per class (only if the dataset folder exists on this computer)."""
    examples = []
    for cls in CLASSES:
        folder = HERE / 'dataset' / 'test' / cls
        files = sorted(folder.glob('*.*')) if folder.is_dir() else []
        if files:
            examples.append([str(files[0]), DEFAULT_CHOICE])
    return examples


demo = gr.Interface(
    fn=classify,
    inputs=[gr.Image(type='pil', label='Photo of one waste item'),
            gr.Radio(choices=list(MODEL_CHOICES), value=DEFAULT_CHOICE, label='Model')],
    outputs=[gr.Label(num_top_classes=len(CLASSES), label='Model scores'), gr.Markdown()],
    title='Waste Image Classifier',
    description=DESCRIPTION,
    examples=find_examples() or None,
    cache_examples=False,
)

if __name__ == '__main__':
    demo.launch()