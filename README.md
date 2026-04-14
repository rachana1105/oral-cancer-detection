# Oral Cancer Detection

A binary image classifier that distinguishes **Normal** oral tissue from **OSCC** (Oral Squamous Cell Carcinoma) using transfer learning on VGG16.

## Approach

- **Base model**: VGG16 pretrained on ImageNet (top layers removed), with a custom head — Flatten → Dense(1024, ReLU) → BatchNorm → Dense(1, sigmoid) — for binary classification.
- **Input handling**: images are resized to 224×224 before being passed through VGG16's preprocessing.
- **Training**: data augmentation (rotation, shear, zoom, flips, shifts) via `ImageDataGenerator`, with early stopping (on validation accuracy) and checkpointing to save the best-performing weights.
- **Evaluation**: accuracy, F1, precision, recall, and a confusion matrix (with sensitivity/specificity) computed on a held-out test set.

## Dataset structure expected

```
oral/
├── train/
│   ├── Normal/
│   └── OSCC/
├── val/
│   ├── Normal/
│   └── OSCC/
└── test/
    ├── Normal/
    └── OSCC/
```

## Requirements

```bash
pip install tensorflow scikit-learn matplotlib numpy pandas
```

The notebook was originally built in Google Colab and expects the dataset in Google Drive — swap the `/content/drive/...` paths for local paths to run it outside Colab.

## Notes

- Some of the core logic (the VGG16 model construction, the confusion matrix computation/plotting, and the test-set evaluation loops) is left as `# TODO` in this version rather than fully filled in.
- **This is a research/learning project, not a diagnostic tool.** It is not validated for clinical use and should not be used to make real medical decisions.

## License

MIT — see [LICENSE](LICENSE).
