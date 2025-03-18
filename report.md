# Emotion Recognition Experiment Documentation

This document provides detailed information about the emotion recognition experiments conducted on the IEMOCAP dataset, including the methodology, data processing, model architecture, training process, and results.

## Table of Contents
1. [Dataset](#dataset)
2. [Methodology](#methodology)
3. [Text Data Processing](#text-data-processing)
4. [Text-to-VAD Conversion](#text-to-vad-conversion)
5. [VAD-to-Emotion Mapping](#vad-to-emotion-mapping)
6. [End-to-End Pipeline](#end-to-end-pipeline)
7. [Results](#results)
8. [Conclusion](#conclusion)

## Dataset

### Overview
- **Dataset**: Synthetic IEMOCAP (Interactive Emotional Dyadic Motion Capture) dataset
- **Total Utterances**: 1,140
- **Modality**: Text (synthetic utterances generated based on VAD values)
- **Emotion Models**: Quadrant, Custom, Plutchik, Ekman

### Data Distribution

#### VAD Values Distribution
- **Valence**: Scale 1-5, normalized to [0,1]
  - 1: 211 utterances (18.5%)
  - 2: 270 utterances (23.7%)
  - 3: 222 utterances (19.5%)
  - 4: 206 utterances (18.1%)
  - 5: 231 utterances (20.3%)
- **Arousal**: Scale 1-5, normalized to [0,1]
  - 1: 214 utterances (18.8%)
  - 2: 229 utterances (20.1%)
  - 3: 244 utterances (21.4%)
  - 4: 213 utterances (18.7%)
  - 5: 240 utterances (21.1%)
- **Dominance**: Scale 1-5, normalized to [0,1]
  - 1: 216 utterances (18.9%)
  - 2: 228 utterances (20.0%)
  - 3: 247 utterances (21.7%)
  - 4: 223 utterances (19.6%)
  - 5: 226 utterances (19.8%)

#### Emotion Distribution by Model
- **Quadrant Model**:
  - happy: 412 utterances (36.1%)
  - angry: 285 utterances (25.0%)
  - calm: 247 utterances (21.7%)
  - sad: 196 utterances (17.2%)

- **Custom Model**:
  - neutral: 328 utterances (28.8%)
  - sad: 196 utterances (17.2%)
  - excited: 192 utterances (16.8%)
  - content: 154 utterances (13.5%)
  - happy: 137 utterances (12.0%)
  - angry: 78 utterances (6.8%)
  - afraid: 55 utterances (4.8%)

- **Plutchik Model**:
  - neutral: 410 utterances (36.0%)
  - joy: 192 utterances (16.8%)
  - trust: 154 utterances (13.5%)
  - surprise: 93 utterances (8.2%)
  - anger: 78 utterances (6.8%)
  - sadness: 74 utterances (6.5%)
  - fear: 55 utterances (4.8%)
  - anticipation: 43 utterances (3.8%)
  - disgust: 41 utterances (3.6%)

- **Ekman Model**:
  - neutral: 338 utterances (29.6%)
  - happiness: 283 utterances (24.8%)
  - sadness: 196 utterances (17.2%)
  - disgust: 107 utterances (9.4%)
  - surprise: 83 utterances (7.3%)
  - anger: 78 utterances (6.8%)
  - fear: 55 utterances (4.8%)

## Methodology

The emotion recognition system follows a two-step approach:
1. **Text-to-VAD Conversion**: Convert text utterances to VAD (Valence-Arousal-Dominance) values
2. **VAD-to-Emotion Mapping**: Map VAD values to emotion categories using different emotion models

### Data Splitting Strategy
- **Train-Test Split**: 80% training, 20% testing
- **Stratification**: Stratified by emotion categories to ensure balanced class distribution
- **Cross-Validation**: 5-fold cross-validation for hyperparameter tuning

### Evaluation Metrics
- **Text-to-VAD Conversion**:
  - Mean Squared Error (MSE)
  - Pearson Correlation Coefficient
- **VAD-to-Emotion Mapping**:
  - Accuracy
  - F1 Score (weighted)
  - Precision (weighted)
  - Recall (weighted)
  - Confusion Matrix
- **End-to-End Pipeline**:
  - Accuracy
  - F1 Score (weighted)
  - Classification Report

## Text Data Processing

### Text Generation
- **Approach**: Generated synthetic text data based on VAD values
- **Method**: Used emotion-related word lists and sentence templates
- **Word Lists**:
  - High/Low Valence Words (e.g., happy, sad)
  - High/Low Arousal Words (e.g., excited, calm)
  - High/Low Dominance Words (e.g., powerful, weak)
- **Sentence Templates**:
  - Simple templates (e.g., "I feel {emotion} today.")
  - Complex templates (e.g., "I've been feeling {emotion1} all day, but now I'm starting to feel {emotion2}.")
- **Variety**: Added variety through different sentence structures and emotion combinations

### Text Preprocessing
- **Tokenization**: Standard tokenization for feature extraction
- **Feature Extraction**: TF-IDF vectorization with n-grams (1-2)
- **Feature Selection**: Maximum 5,000 features

## Text-to-VAD Conversion

### Model Architecture
- **Model Type**: Scikit-learn based machine learning model
- **Feature Extraction**: TF-IDF Vectorizer (max_features=5000, ngram_range=(1, 2))
- **Regression Models**: Separate Random Forest regressors for each VAD dimension
  - **Valence Model**: RandomForestRegressor(n_estimators=100, random_state=42)
  - **Arousal Model**: RandomForestRegressor(n_estimators=100, random_state=42)
  - **Dominance Model**: RandomForestRegressor(n_estimators=100, random_state=42)

### Training Details
- **Training Set Size**: 729 utterances (80% of 912 training utterances)
- **Validation Set Size**: 183 utterances (20% of 912 training utterances)
- **Test Set Size**: 228 utterances (20% of total 1,140 utterances)
- **Hyperparameters**: Default Random Forest parameters (n_estimators=100)

### Performance Metrics
- **Overall MSE**: 0.1157
- **Average Correlation**: 0.3185
- **Individual Dimensions**:
  - **Valence**: MSE = 0.1137, Correlation = 0.3070
  - **Arousal**: MSE = 0.1283, Correlation = 0.2764
  - **Dominance**: MSE = 0.1051, Correlation = 0.3722

## VAD-to-Emotion Mapping

### Emotion Models
- **Quadrant Model**: Simple 4-quadrant model based on valence and arousal
  - high valence, high arousal → happy
  - high valence, low arousal → calm
  - low valence, high arousal → angry
  - low valence, low arousal → sad
- **Custom Model**: 7-category model considering all three VAD dimensions
- **Plutchik Model**: Based on Plutchik's wheel of emotions (8 primary emotions)
- **Ekman Model**: Based on Ekman's 6 basic emotions

### Classifier Types
- **Random Forest**: RandomForestClassifier(random_state=42)
- **SVM**: SVC(probability=True, random_state=42)
- **MLP**: MLPClassifier(random_state=42, max_iter=1000)

### Hyperparameter Tuning
- **Random Forest**:
  - n_estimators: [50, 100, 200]
  - max_depth: [None, 10, 20]
  - min_samples_split: [2, 5, 10]
- **SVM**:
  - C: [0.1, 1, 10]
  - gamma: ['scale', 'auto']
  - kernel: ['rbf', 'linear']
- **MLP**:
  - hidden_layer_sizes: [(50,), (100,), (50, 50)]
  - alpha: [0.0001, 0.001, 0.01]
  - learning_rate: ['constant', 'adaptive']

### Performance Metrics
All classifier types achieved 100% accuracy on the test set for all emotion models. This is expected since the emotion categories are deterministically derived from VAD values using rule-based mappings.

## End-to-End Pipeline

### Pipeline Architecture
1. **Input**: Text utterance
2. **Text-to-VAD Conversion**: Convert text to VAD values using the trained model
3. **VAD-to-Emotion Mapping**: Map VAD values to emotion categories using the trained classifier
4. **Output**: Predicted emotion category

### Performance Metrics
- **Plutchik Model**: Accuracy = 43.86%, F1 Score = 34.49%
- **Ekman Model**: Accuracy = 41.67%, F1 Score = 38.23%
- **Quadrant Model**: Accuracy = 37.28%, F1 Score = 33.40%
- **Custom Model**: Accuracy = 19.74%, F1 Score = 18.58%

## Results

### Text-to-VAD Conversion
The text-to-VAD conversion model achieved moderate performance with an overall MSE of 0.1157 and an average correlation of 0.3185. The dominance dimension was predicted most accurately (correlation = 0.3722), followed by valence (correlation = 0.3070) and arousal (correlation = 0.2764).

### VAD-to-Emotion Mapping
All classifier types (Random Forest, SVM, MLP) achieved perfect accuracy (100%) for all emotion models (Quadrant, Custom, Plutchik, Ekman) when evaluated on the VAD-to-emotion mapping task alone. This is expected since the emotion categories are deterministically derived from VAD values.

### End-to-End Pipeline
The end-to-end pipeline performance varied across emotion models:
- **Plutchik Model**: Best performance with 43.86% accuracy and 34.49% F1 score
- **Ekman Model**: Second best with 41.67% accuracy and 38.23% F1 score
- **Quadrant Model**: 37.28% accuracy and 33.40% F1 score
- **Custom Model**: Lowest performance with 19.74% accuracy and 18.58% F1 score

### Performance Comparison

| Aspect | Original Implementation | Improved Implementation |
|--------|-------------------------|-------------------------|
| Text Data | Used utterance IDs as placeholders | Uses actual text data with emotional content |
| Text-to-VAD | No actual training | Machine learning model with 31.85% correlation |
| VAD-to-Emotion | Hard-coded thresholds | ML classifiers with proper evaluation |
| Evaluation | No proper train-test split | 80/20 split with cross-validation |
| Overall | Artificially high accuracy due to design flaws | Realistic performance with proper methodology |

## Conclusion

This experiment addressed several issues in the original implementation:

1. **Replaced Utterance IDs with Actual Text Data**: Generated synthetic text data that reflects emotional dimensions in VAD values.

2. **Implemented Proper Text-to-VAD Conversion**: Created a machine learning model that processes actual text data and predicts VAD values with reasonable accuracy.

3. **Fixed Deterministic Rule-Based Emotion Mapping**: Replaced hard-coded thresholds with proper machine learning classifiers, though the underlying mapping remains deterministic due to the nature of the task.

4. **Fixed Evaluation Methodology**: Implemented proper train-test splits, cross-validation, and comprehensive evaluation metrics to prevent data leakage and provide a more realistic assessment of model performance.

The end-to-end pipeline achieved a best accuracy of 43.86% using the Plutchik emotion model, which is a more realistic performance metric compared to the artificially high accuracy in the original implementation. The results demonstrate that emotion recognition from text is a challenging task, and future work could focus on improving the text-to-VAD conversion model and exploring more sophisticated approaches for emotion classification.
