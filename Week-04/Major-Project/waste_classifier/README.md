# Waste Image Classifier

A small image classifier that sorts a photo of one waste item into 5 categories: cardboard, glass, metal, paper and plastic. It compares a baseline CNN trained from scratch with an improved MobileNetV2 model (transfer learning), and includes a Gradio web demo.

## Install

    python -m pip install torch torchvision gradio pillow

## Run the demo

    python app.py

Then open http://127.0.0.1:7860. The dataset is not included in this repo (it is in .gitignore), so example images only appear if you put the dataset in the dataset/ folder.

## Results (test set, 360 images)

| Model | Validation acc | Test acc |
|---|---|---|
| Baseline CNN | 69.77% | 73.61% |
| Improved CNN (MobileNetV2) | 89.83% | 92.78% |

The full evaluation is in REPORT.md. Plots and confusion matrices are in the results folder.

## Limitations

- Trained on a small public dataset of single items on plain backgrounds, so it may fail on cluttered photos, dirty or crushed items, and mixed materials.
- It must always choose one of the 5 classes, even for objects it has never seen.
- A high score does not guarantee a correct answer: on the test set, 3 wrong predictions had a score of 90% or more.
- This is a learning project, not a tool for real waste-sorting decisions.

## Screenshots

Add your demo screenshots from results/screenshots here.