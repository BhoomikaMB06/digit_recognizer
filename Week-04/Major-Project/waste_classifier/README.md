# Waste Image Classification System

A small CNN that classifies a photo of a waste item into one of 5 classes: cardboard, glass, metal, paper, plastic.
Built with PyTorch, with a Gradio web app for trying it on your own photos.

## Results

| Model | Parameters | Test accuracy |
|---|---|---|
| Baseline CNN | 1,662,373 | 73.6% |
| Improved CNN | 2,230,277 | 92.8% |

The weakest class is **glass** (recall 86.8%). Common confusions: glass predicted as plastic (6 times); plastic predicted as metal (3 times); metal predicted as glass (3 times).
The test set has 360 images, so the 95% margin on the accuracy is about plus or minus 2.7 points.

![Confusion matrix](results/confusion_matrix_baseline.png)

Full analysis: [Assignment 4 report](../../Assignment/assignment4_report.md).

## How to run
1. Download the Kaggle "Garbage Classification" dataset and put it where `prepare_data.py` expects it.
2. `python prepare_data.py` cleans the data, removes duplicates and makes the train/validation/test split.
3. `python train.py` trains both models and writes everything to `results/`.
4. `python app.py` starts the Gradio app.
5. `python make_report.py` rebuilds this README and the Assignment 4 report from the results.

## Limitations
- Trained on clean single-item photos, so cluttered real-world photos will be harder.
- Only 5 classes; anything else is forced into one of them.
- The model made 3 wrong predictions with a score of 90% or more, so a high score is not a guarantee of a correct answer.
