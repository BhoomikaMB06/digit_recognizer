# Assignment 4: Evaluation Report

## 1. Models compared
| Model | Description | Parameters | Training time | Best epoch |
|---|---|---|---|---|
| Baseline CNN | 3 conv blocks trained from scratch, no augmentation | 1,662,373 | 5.9 min | 10 |
| Improved CNN | MobileNetV2 pretrained + fine-tuning + augmentation + dropout | 2,230,277 | 6.9 min | 10 |

Dataset: 5 classes (cardboard, glass, metal, paper, plastic). 2,390 raw images, 2,374 after duplicate removal, split into 1,660 train, 354 validation and 360 test images. The best epoch was chosen on validation accuracy, and the test set was used once, at the end.

## 2. Metrics
| Model | Validation acc | Test acc | Macro F1 (test) |
|---|---|---|---|
| Baseline CNN | 69.77% | 73.61% | 73.00% |
| Improved CNN | 89.83% | 92.78% | 92.74% |

The improved model gains 19.17 percentage points of test accuracy over the baseline.

Per-class F1 on the test set:

| Class | Baseline | Improved |
|---|---|---|
| cardboard | 82.64% | 95.93% |
| glass | 70.13% | 89.19% |
| metal | 64.76% | 92.06% |
| paper | 82.98% | 95.51% |
| plastic | 64.47% | 91.03% |

The baseline's weakest points: metal recall was only 54.84% (many metal items were missed) and plastic precision was 61.25% (many other items were called plastic). The improved model is above 86% recall for every class, and its weakest class is glass (F1 89.19%).

## 3. Confusion matrix analysis
See results/confusion_matrix_baseline.png and results/confusion_matrix_improved.png.
- Most confused class pairs (baseline): __
- Most confused class pairs (improved): __
- Why might these happen (similar color, shape, transparency or material)? __
- Did the improved model fix them? __

## 4. Failure analysis
The improved model made 26 mistakes on 360 test images. See results/failure_examples.png.
- Among the 10 most confident mistakes, glass is involved in 6 (as the true or the predicted class), and cardboard and paper are confused in both directions (cardboard_0035 and paper_0034).
- What the failure photos have in common (lighting, background, crushed or dirty items, transparent or shiny surfaces): __
- Confidence on failures: the average score of wrong predictions was 66.41%, much lower than for correct ones, but 3 wrong predictions still had a score of 90% or more (the highest was 97.12%).

## 5. Do high scores mean the answer is correct?
| Model score | Predictions | Correct | Share correct |
|---|---|---|---|
| below 50% | 13 | 8 | 61.54% |
| 50% to 80% | 68 | 55 | 80.88% |
| 80% to 95% | 94 | 87 | 92.55% |
| 95% and above | 185 | 184 | 99.46% |

Higher scores are linked to more correct answers, so the score is a useful warning signal. It is still not a guarantee: the most confident mistake was 97.12% sure.

## 6. Improvements and limitations
Improvements to try:
- More images, especially glass, metal and plastic in different lighting.
- Stronger augmentation (brightness, rotation, background changes).
- Training longer if the validation curve was still rising at epoch 10 (check results/training_curves.png).
- An "unknown / other" class, because the model must always pick one of the 5 classes.

Limitations:
- The dataset has single items on plain backgrounds, so the results may not hold for real photos.
- My cluttered phone photo test: __ (screenshot in results/screenshots).
- An object outside the 5 classes (__) was predicted as __ with a score of __%, which shows why a high score does not mean the answer is correct.
- With only 360 test images, one image changes accuracy by about 0.28 percentage points, so small differences should not be over-read.