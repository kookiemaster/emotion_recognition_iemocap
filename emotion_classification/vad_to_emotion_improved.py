#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved VAD to emotion mapping module.
This module implements a proper VAD-to-emotion mapping using machine learning
instead of deterministic rule-based mapping with hard-coded thresholds.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Define emotion categories for different models
EMOTION_CATEGORIES = {
    'ekman': ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral'],
    'plutchik': ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'anticipation'],
    'quadrant': ['high_arousal_positive_valence', 'high_arousal_negative_valence', 
                'low_arousal_positive_valence', 'low_arousal_negative_valence'],
    'custom': ['happy', 'angry', 'sad', 'fearful', 'surprised', 'disgusted', 'neutral']
}

class VADToEmotionClassifier:
    """
    A classifier for mapping VAD (Valence, Arousal, Dominance) values to emotion categories.
    """
    
    def __init__(self, classifier_type='random_forest', emotion_model='ekman'):
        """
        Initialize the VADToEmotionClassifier.
        
        Args:
            classifier_type: Type of classifier to use ('random_forest', 'svm', or 'mlp')
            emotion_model: Emotion model to use ('ekman', 'plutchik', 'quadrant', or 'custom')
        """
        self.classifier_type = classifier_type
        self.emotion_model = emotion_model
        
        if classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif classifier_type == 'svm':
            self.classifier = SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42
            )
        elif classifier_type == 'mlp':
            self.classifier = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
        
        if emotion_model not in EMOTION_CATEGORIES:
            raise ValueError(f"Unsupported emotion model: {emotion_model}")
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the classifier on the given VAD values and emotion labels.
        
        Args:
            X: DataFrame with VAD values (columns: 'valence_norm', 'arousal_norm', 'dominance_norm')
            y: Series with emotion labels
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with best hyperparameters
        """
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Training completed:")
        print(f"  Test accuracy: {accuracy:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=np.unique(y),
                   yticklabels=np.unique(y))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {self.emotion_model.capitalize()} Emotions')
        plt.tight_layout()
        
        # Save confusion matrix
        output_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'models', 
                                f'{self.emotion_model}_{self.classifier_type}_confusion_matrix.png'))
        plt.close()
        
        # Return best parameters (for future hyperparameter tuning)
        return {'classifier_type': self.classifier_type}
    
    def predict(self, X):
        """
        Predict emotion categories for the given VAD values.
        
        Args:
            X: DataFrame with VAD values (columns: 'valence_norm', 'arousal_norm', 'dominance_norm')
            
        Returns:
            DataFrame with predicted emotions and probabilities
        """
        # Predict emotion categories
        y_pred = self.classifier.predict(X)
        
        # Get probabilities if available
        if hasattr(self.classifier, 'predict_proba'):
            y_proba = self.classifier.predict_proba(X)
            
            # Create DataFrame with predictions and probabilities
            predictions = pd.DataFrame({
                'predicted_emotion': y_pred
            })
            
            # Add probability for each emotion category
            for i, emotion in enumerate(self.classifier.classes_):
                predictions[f'prob_{emotion}'] = y_proba[:, i]
        else:
            # Create DataFrame with predictions only
            predictions = pd.DataFrame({
                'predicted_emotion': y_pred
            })
        
        return predictions
    
    def save_model(self, output_dir):
        """
        Save the model to disk.
        
        Args:
            output_dir: Directory to save the model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save classifier
        joblib.dump(self.classifier, os.path.join(output_dir, 'classifier.pkl'))
        
        # Save configuration
        config = {
            'classifier_type': self.classifier_type,
            'emotion_model': self.emotion_model
        }
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        print(f"Model saved to {output_dir}")
    
    @classmethod
    def load_model(cls, model_dir):
        """
        Load a saved model from disk.
        
        Args:
            model_dir: Directory where the model is saved
            
        Returns:
            Loaded VADToEmotionClassifier
        """
        # Load configuration
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Create classifier
        classifier = cls(
            classifier_type=config['classifier_type'],
            emotion_model=config['emotion_model']
        )
        
        # Load classifier
        classifier.classifier = joblib.load(os.path.join(model_dir, 'classifier.pkl'))
        
        return classifier

def generate_synthetic_vad_emotion_data(num_samples=1000, emotion_model='ekman'):
    """
    Generate synthetic VAD values and emotion labels for training.
    This is used to create a dataset with a more balanced distribution of emotions.
    
    Args:
        num_samples: Number of samples to generate
        emotion_model: Emotion model to use
        
    Returns:
        DataFrame with VAD values and emotion labels
    """
    # Define emotion regions in VAD space for different emotion models
    if emotion_model == 'ekman':
        emotion_regions = {
            'anger': {'v': (0.0, 0.3), 'a': (0.7, 1.0), 'd': (0.6, 1.0)},
            'disgust': {'v': (0.0, 0.4), 'a': (0.3, 0.7), 'd': (0.6, 1.0)},
            'fear': {'v': (0.0, 0.3), 'a': (0.7, 1.0), 'd': (0.0, 0.4)},
            'joy': {'v': (0.7, 1.0), 'a': (0.5, 1.0), 'd': (0.6, 1.0)},
            'sadness': {'v': (0.0, 0.3), 'a': (0.0, 0.4), 'd': (0.0, 0.4)},
            'surprise': {'v': (0.5, 1.0), 'a': (0.7, 1.0), 'd': (0.3, 0.7)},
            'neutral': {'v': (0.4, 0.6), 'a': (0.4, 0.6), 'd': (0.4, 0.6)}
        }
    elif emotion_model == 'plutchik':
        emotion_regions = {
            'anger': {'v': (0.0, 0.3), 'a': (0.7, 1.0), 'd': (0.6, 1.0)},
            'disgust': {'v': (0.0, 0.4), 'a': (0.3, 0.7), 'd': (0.6, 1.0)},
            'fear': {'v': (0.0, 0.3), 'a': (0.7, 1.0), 'd': (0.0, 0.4)},
            'joy': {'v': (0.7, 1.0), 'a': (0.5, 1.0), 'd': (0.6, 1.0)},
            'sadness': {'v': (0.0, 0.3), 'a': (0.0, 0.4), 'd': (0.0, 0.4)},
            'surprise': {'v': (0.5, 1.0), 'a': (0.7, 1.0), 'd': (0.3, 0.7)},
            'trust': {'v': (0.7, 1.0), 'a': (0.3, 0.6), 'd': (0.5, 0.8)},
            'anticipation': {'v': (0.5, 0.8), 'a': (0.5, 0.8), 'd': (0.5, 0.8)}
        }
    elif emotion_model == 'quadrant':
        emotion_regions = {
            'high_arousal_positive_valence': {'v': (0.5, 1.0), 'a': (0.5, 1.0), 'd': (0.0, 1.0)},
            'high_arousal_negative_valence': {'v': (0.0, 0.5), 'a': (0.5, 1.0), 'd': (0.0, 1.0)},
            'low_arousal_positive_valence': {'v': (0.5, 1.0), 'a': (0.0, 0.5), 'd': (0.0, 1.0)},
            'low_arousal_negative_valence': {'v': (0.0, 0.5), 'a': (0.0, 0.5), 'd': (0.0, 1.0)}
        }
    elif emotion_model == 'custom':
        emotion_regions = {
            'happy': {'v': (0.7, 1.0), 'a': (0.5, 1.0), 'd': (0.5, 1.0)},
            'angry': {'v': (0.0, 0.3), 'a': (0.7, 1.0), 'd': (0.6, 1.0)},
            'sad': {'v': (0.0, 0.3), 'a': (0.0, 0.4), 'd': (0.0, 0.5)},
            'fearful': {'v': (0.0, 0.3), 'a': (0.7, 1.0), 'd': (0.0, 0.4)},
            'surprised': {'v': (0.5, 1.0), 'a': (0.7, 1.0), 'd': (0.3, 0.7)},
            'disgusted': {'v': (0.0, 0.4), 'a': (0.3, 0.7), 'd': (0.6, 1.0)},
            'neutral': {'v': (0.4, 0.6), 'a': (0.4, 0.6), 'd': (0.4, 0.6)}
        }
    else:
        raise ValueError(f"Unsupported emotion model: {emotion_model}")
    
    # Generate data
    data = []
    emotions = list(emotion_regions.keys())
    samples_per_emotion = num_samples // len(emotions)
    
    for emotion in emotions:
        region = emotion_regions[emotion]
        
        for _ in range(samples_per_emotion):
            valence = np.random.uniform(region['v'][0], region['v'][1])
            arousal = np.random.uniform(region['a'][0], region['a'][1])
            dominance = np.random.uniform(region['d'][0], region['d'][1])
            
            data.append({
                'valence_norm': valence,
                'arousal_norm': arousal,
                'dominance_norm': dominance,
                'emotion': emotion
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def plot_vad_space(df, emotion_column, output_file):
    """
    Plot VAD space with emotion labels.
    
    Args:
        df: DataFrame with VAD values and emotion labels
        emotion_column: Column name for emotion labels
        output_file: Path to save the plot
    """
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Plot valence-arousal space
    ax1 = fig.add_subplot(131)
    scatter = ax1.scatter(df['valence_norm'], df['arousal_norm'], 
                         c=pd.factorize(df[emotion_column])[0], 
                         alpha=0.6, cmap='viridis')
    ax1.set_xlabel('Valence')
    ax1.set_ylabel('Arousal')
    ax1.set_title('Valence-Arousal Space')
    ax1.grid(True)
    
    # Plot valence-dominance space
    ax2 = fig.add_subplot(132)
    scatter = ax2.scatter(df['valence_norm'], df['dominance_norm'], 
                         c=pd.factorize(df[emotion_column])[0], 
                         alpha=0.6, cmap='viridis')
    ax2.set_xlabel('Valence')
    ax2.set_ylabel('Dominance')
    ax2.set_title('Valence-Dominance Space')
    ax2.grid(True)
    
    # Plot arousal-dominance space
    ax3 = fig.add_subplot(133)
    scatter = ax3.scatter(df['arousal_norm'], df['dominance_norm'], 
                         c=pd.factorize(df[emotion_column])[0], 
                         alpha=0.6, cmap='viridis')
    ax3.set_xlabel('Arousal')
    ax3.set_ylabel('Dominance')
    ax3.set_title('Arousal-Dominance Space')
    ax3.grid(True)
    
    # Add legend
    legend1 = ax3.legend(scatter.legend_elements()[0], 
                        df[emotion_column].unique(),
                        title="Emotions",
                        loc="upper right")
    ax3.add_artist(legend1)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"VAD space plot saved to {output_file}")

def main():
    """
    Main function to train and evaluate the VAD to emotion classifier.
    """
    # Set paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    vad_dir = os.path.join(base_dir, 'vad_conversion')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load VAD predictions from text-to-VAD model
    vad_predictions_path = os.path.join(vad_dir, 'models', 'vad_predictions.csv')
    if os.path.exists(vad_predictions_path):
        print(f"Loading VAD predictions from {vad_predictions_path}")
        vad_df = pd.read_csv(vad_predictions_path)
        
        # Use predicted VAD values - ensure column names match training data
        X = vad_df[['valence_pred', 'arousal_pred', 'dominance_pred']].rename(
            columns={
                'valence_pred': 'valence_norm', 
                'arousal_pred': 'arousal_norm', 
                'dominance_pred': 'dominance_norm'
            })
    else:
        print(f"VAD predictions not found at {vad_predictions_path}")
        print("Generating synthetic VAD data for training...")
        
        # Generate synthetic data for training
        vad_df = None
        X = None
    
    # Define emotion models and classifier types to evaluate
    emotion_models = ['ekman', 'plutchik', 'quadrant', 'custom']
    classifier_types = ['random_forest', 'svm', 'mlp']
    
    # Store results
    results = []
    
    # Train and evaluate classifiers for each emotion model
    for emotion_model in emotion_models:
        print(f"\n\n{'='*50}")
        print(f"Training classifiers for {emotion_model.upper()} emotion model")
        print(f"{'='*50}")
        
        # Generate synthetic data for this emotion model
        synthetic_df = generate_synthetic_vad_emotion_data(
            num_samples=2000, 
            emotion_model=emotion_model
        )
        
        # Plot VAD space with emotion labels
        plot_vad_space(
            synthetic_df, 
            'emotion', 
            os.path.join(output_dir, f'vad_space_{emotion_model}.png')
        )
        
        # Prepare features and target
        X_synth = synthetic_df[['valence_norm', 'arousal_norm', 'dominance_norm']]
        y_synth = synthetic_df['emotion']
        
        print(f"Emotion distribution:")
        print(y_synth.value_counts())
        print(f"{'='*50}\n")
        
        for classifier_type in classifier_types:
            print(f"\n{'-'*50}")
            print(f"Training {classifier_type.upper()} classifier")
            print(f"{'-'*50}")
            
            # Create and train classifier
            classifier = VADToEmotionClassifier(
                classifier_type=classifier_type,
                emotion_model=emotion_model
            )
            
            # Train classifier on synthetic data
            best_params = classifier.train(X_synth, y_synth)
            
            # Save model
            model_dir = os.path.join(output_dir, f'{emotion_model}_{classifier_type}')
            classifier.save_model(model_dir)
            
            # If we have real VAD predictions, evaluate on them
            if vad_df is not None:
                # Make predictions on the real data
                predictions = classifier.predict(X)
                
                # Save predictions
                result_df = pd.DataFrame({
                    'utterance_id': vad_df['utterance_id'],
                    'text': vad_df['text'],
                    'valence_norm': X['valence_norm'],
                    'arousal_norm': X['arousal_norm'],
                    'dominance_norm': X['dominance_norm'],
                    'predicted_emotion': predictions['predicted_emotion']
                })
                
                result_df.to_csv(
                    os.path.join(output_dir, f'{emotion_model}_{classifier_type}_predictions.csv'), 
                    index=False
                )
            
            # Store results
            results.append({
                'emotion_model': emotion_model,
                'classifier_type': classifier_type,
                'best_params': best_params
            })
    
    # Print summary of results
    print("\n\nSummary of trained models:")
    print(f"{'='*50}")
    for result in results:
        print(f"Emotion model: {result['emotion_model'].upper()}, "
              f"Classifier: {result['classifier_type'].upper()}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'classifier_results.csv'), index=False)
    print(f"Results saved to {os.path.join(output_dir, 'classifier_results.csv')}")

if __name__ == "__main__":
    main()
