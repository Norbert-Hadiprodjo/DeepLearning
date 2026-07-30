# Deep Learning for Automated AI-Generated Image Detection

**APS360 — Applied Fundamentals of Deep Learning · University of Toronto**
Norbert Rama Hadiprodjo

A binary image classifier that estimates the probability an image is AI-generated. A fine-tuned ResNet18 is benchmarked against a pixel-based SVM, then stress-tested on a self-collected dataset built from a personal camera and current-generation image models.

![Python](https://img.shields.io/badge/Python-3.10+-1a1a1a?style=flat-square&logo=python&logoColor=FFB000)
![PyTorch](https://img.shields.io/badge/PyTorch-1a1a1a?style=flat-square&logo=pytorch&logoColor=FFB000)
![torchvision](https://img.shields.io/badge/torchvision-ResNet18-1a1a1a?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-1a1a1a?style=flat-square&logo=scikitlearn&logoColor=FFB000)
![Colab](https://img.shields.io/badge/Google%20Colab-1a1a1a?style=flat-square&logo=googlecolab&logoColor=FFB000)

---

## Headline result

| Model | Held-out test split | Self-collected data |
| --- | --- | --- |
| SVM baseline (grayscale pixels) | 82.4% | — |
| **Fine-tuned ResNet18** | **96.02%** | **67.3%** |

The 29-point drop between columns is the actual finding of this project. Near-ceiling accuracy on in-distribution data collapses under distribution shift, which suggests the network learned dataset-specific cues alongside genuinely generalizable artifacts.

---

## Pipeline

```
Kaggle DALL·E / Midjourney dataset
 │
 ├─ Filter to valid image formats (.png/.jpg/.jpeg), drop metadata files
 ├─ Load via PIL → convert to RGB → resize to 224×224
 ├─ Stratified 70 / 15 / 15 train-val-test split
 │
 ├─ Baseline:  128×128 grayscale → flatten to 16,384-dim vector → RBF SVM
 │
 ├─ Primary:   ImageNet-pretrained ResNet18
 │             fc layer → single logit → sigmoid → P(AI-generated)
 │             BCEWithLogitsLoss, Adam @ 1e-4, batch 32, 5 epochs
 │
 └─ Evaluate:  held-out 15% split  +  self-collected out-of-distribution set
```

---

## Data

Two classes drawn from AI recognition datasets on Kaggle — real photographs, and AI-generated images from DALL·E and Midjourney. After filtering:

| Class | Label | Count |
| --- | --- | --- |
| AI-generated | `1` | 17,857 |
| Real | `0` | 3,781 |

**The 4.7:1 imbalance matters.** It is the most likely driver of the model's bias toward predicting AI, and it is worth reading the results section with this ratio in mind.

Training augmentation is deliberately conservative — `RandomHorizontalFlip` and `ColorJitter(brightness=0.1)` only. Aggressive augmentation risks destroying the high-frequency generative artifacts the model needs to detect, so validation and test transforms apply resize and tensor conversion with no augmentation at all.

---

## Models

### Baseline — RBF SVM on raw pixels

Each image is resized to 128×128, converted to grayscale, and flattened into a 16,384-dimensional intensity vector fed to an RBF-kernel `SVC`. Full-dataset training was computationally prohibitive at this feature dimensionality, so the baseline is fit on a 4,000-image sample and evaluated on 1,000.

It reaches **82.4%**, which is the interesting part: raw pixel intensities alone carry real signal for this task. That number is what the CNN has to beat to justify itself.

### Primary — Fine-tuned ResNet18

Transfer learning from ImageNet weights. The final fully connected layer is replaced with a single output neuron, and the logit is passed through a sigmoid at inference to yield P(AI-generated). Residual connections in the backbone allow effective feature extraction without training from scratch on a dataset of this size.

| Hyperparameter | Value |
| --- | --- |
| Input resolution | 224 × 224 |
| Loss | `BCEWithLogitsLoss` |
| Optimizer | Adam |
| Learning rate | `1e-4` |
| Batch size | 32 |
| Epochs | 5 |

---

## Results

### In-distribution

Final training accuracy ≈ 99.5%, validation accuracy ≈ 96.5%. Validation stays flat and high while training accuracy climbs, indicating effective learning with limited overfitting over the 5-epoch run.

**Confusion matrix (held-out 15% split):**

| | Predicted Real | Predicted AI |
| --- | --- | --- |
| **True Real** | 468 | 99 |
| **True AI** | 30 | 2,649 |

| Metric | Value |
| --- | --- |
| Precision | 0.964 |
| Recall | 0.989 |
| F1 | 0.976 |
| Specificity | 0.825 |

Recall of 0.989 against specificity of 0.825 is the asymmetry to notice. The model is excellent at catching synthetic images and noticeably worse at leaving real ones alone — 99 real photographs were flagged as AI-generated.

### Out-of-distribution

The self-collected evaluation set pairs real photographs shot on a personal camera with AI images from **GPT Image 1.5** and **Gemini 2.5 Flash Image** — both generators absent from training, which is the point.

Accuracy falls to **67.3%**. Failures cluster around lighting, texture, and composition: real photographs whose visual properties happen to resemble generative artifacts get flagged, and highly realistic modern synthetic images slip through. Training on DALL·E and Midjourney outputs does not transfer cleanly to the generator landscape that came after them.

---

## Setup

```bash
pip install torch torchvision scikit-learn opencv-python pandas matplotlib seaborn pillow
```

The notebook is written for Google Colab and mounts Drive to source `fakeV2.zip` and `real.zip`. To run elsewhere, replace the Drive mount and zip-extraction cells with local paths and point `fake_path` / `real_path` at the extracted directories. CUDA is detected automatically; the ResNet fine-tune is impractical on CPU.

Dataset: [DALL·E recognition dataset](https://www.kaggle.com/datasets/superpotato9/dalle-recognition-dataset/data) (Kaggle).

---

## Limitations

- **Class imbalance is uncorrected.** No class weighting, resampling, or threshold tuning was applied, and the 4.7:1 skew toward AI images plausibly explains much of the low specificity.
- **The baseline is trained on a subsample.** 4,000 of ~21,600 images, so 82.4% is an approximation of what a full-data SVM would achieve.
- **Generator coverage is narrow.** Training saw DALL·E and Midjourney only. The out-of-distribution result shows how thin that coverage is against newer models.
- **Short training run.** 5 epochs with no learning-rate schedule, early stopping, or hyperparameter search.
- **Accuracy is the only tracked metric during training.** With this imbalance, per-epoch precision and recall would have been more informative than the accuracy curve.

---

## Ethical considerations

The specificity gap has a direct real-world cost: this model's dominant failure mode is labelling authentic photographs as synthetic. Deployed in a content-moderation context, that means removal or demotion of legitimate media, and the burden falls on whoever took the photograph to contest it.

Training data also does not represent the full diversity of real-world imagery or subject populations, so error rates are unlikely to be uniform across contexts. And detection is a moving target — as generative models improve, a classifier frozen at this point in time degrades whether or not anyone notices.

---

## Next steps

- Address the imbalance directly — class-weighted loss, resampling, or tuning the decision threshold against a precision-recall curve rather than defaulting to 0.5
- Add frequency-domain features (FFT, DCT), which the literature reports as more generator-agnostic than pixel-space cues
- Expand training data across generators and include more recent models
- Longer training with a learning-rate schedule and early stopping on validation loss
- Add Grad-CAM to visualize which regions drive predictions, and check whether the model attends to artifacts or to background context

---

## References

Key related work informing the approach — full citation list in the project report.

- Marra et al., *Detection of GAN-Generated Fake Images Over Social Networks*, IEEE MIPR 2019
- Yu, Davis & Fritz, *Attributing Fake Images to GANs: Learning and Analyzing GAN Fingerprints*, ICCV 2019
- Rössler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images*, ICCV 2019
- Wang et al., *CNN-Generated Images Are Surprisingly Easy to Spot... for Now*, CVPR 2020
