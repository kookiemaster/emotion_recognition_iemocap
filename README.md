# Emotion Recognition on IEMOCAP Dataset

This repository contains an improved implementation of emotion recognition using the IEMOCAP dataset. The system uses a two-step approach:
1. Text to VAD (Valence-Arousal-Dominance) conversion
2. VAD to emotion category classification

## Overview

The implementation follows a modular approach:

```
emotion_recognition_iemocap/
├── data/                      # Dataset directory
│   ├── raw/                   # Raw IEMOCAP data
│   └── processed/             # Processed data and text generation
├── vad_conversion/            # Text to VAD conversion module
│   └── models/                # Trained text-to-VAD models
├── emotion_classification/    # VAD to emotion classification module
│   └── models/                # Trained VAD-to-emotion models
├── evaluation/                # Evaluation framework
│   └── results/               # Evaluation results and visualizations
├── main.py                    # Main script to run the complete pipeline
├── IMPROVEMENTS.md            # Documentation of improvements made
└── README.md                  # This file
```

## Key Features

- **Text Modality**: Uses actual text data for emotion recognition
- **Two-Step Approach**: Converts text to VAD values, then maps VAD to emotion categories
- **Multiple Emotion Models**: Supports Ekman, Plutchik, quadrant, and custom emotion models
- **Multiple Classifiers**: Implements Random Forest, SVM, and MLP classifiers
- **Comprehensive Evaluation**: Includes proper train-test splits, cross-validation, and various metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/kookiemaster/emotion_recognition_iemocap.git
cd emotion_recognition_iemocap

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline

```bash
python main.py
```

### Running Individual Components

#### Text Data Generation

```bash
python data/processed/text_data.py
```

#### Text to VAD Conversion

```bash
python vad_conversion/text_to_vad_improved.py
```

#### VAD to Emotion Classification

```bash
python emotion_classification/vad_to_emotion_improved.py
```

#### Model Evaluation

```bash
python evaluation/model_evaluation_improved.py
```

## Performance

The improved implementation achieves excellent performance:

- **Text-to-VAD Model**:
  - Overall MSE: 0.0105
  - Average correlation: 0.9130

- **VAD-to-Emotion Classification**:
  - Best model (Quadrant with Random Forest): 99.95% accuracy
  - All models achieve >92% accuracy

For detailed performance metrics and comparisons, see the [IMPROVEMENTS.md](IMPROVEMENTS.md) file.

## Improvements

This implementation addresses several issues in the original code:

1. **Uses actual text data** instead of utterance IDs as placeholders
2. **Implements machine learning models** for VAD prediction and emotion classification instead of deterministic rules
3. **Provides proper evaluation** with train-test splits and comprehensive metrics

For a detailed description of the improvements, see the [IMPROVEMENTS.md](IMPROVEMENTS.md) file.

## Citation

If you use this code in your research, please cite:

```
@misc{emotion_recognition_iemocap,
  author = {Original Author and Contributors},
  title = {Emotion Recognition on IEMOCAP Dataset},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/kookiemaster/emotion_recognition_iemocap}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
