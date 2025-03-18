#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text to VAD conversion model for emotion recognition using scikit-learn.
This script implements a lighter model that converts text to VAD (Valence-Arousal-Dominance) values.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import json
import pickle

class TextToVADModel:
    def __init__(self):
        """
        Initialize the Text to VAD conversion model using scikit-learn.
        """
        # TF-IDF vectorizer for text features
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        
        # Separate regressors for each VAD dimension
        self.valence_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.arousal_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.dominance_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Trained flag
        self.is_trained = False
    
    def train(self, texts, vad_values, validation_split=0.2):
        """
        Train the model on texts and their VAD values.
        
        Args:
            texts: List of text strings
            vad_values: DataFrame with columns 'valence_norm', 'activation_norm', 'dominance_norm'
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training history
        """
        # Convert to numpy arrays
        valence = vad_values['valence_norm'].values
        arousal = vad_values['activation_norm'].values
        dominance = vad_values['dominance_norm'].values
        
        # Training history
        history = {
            'val_valence_mse': [],
            'val_arousal_mse': [],
            'val_dominance_mse': [],
            'val_valence_corr': [],
            'val_arousal_corr': [],
            'val_dominance_corr': []
        }
        
        # Split data into train and validation sets
        train_texts, val_texts, train_valence, val_valence, train_arousal, val_arousal, train_dominance, val_dominance = train_test_split(
            texts, valence, arousal, dominance, test_size=validation_split, random_state=42
        )
        
        print(f"Training on {len(train_texts)} samples, validating on {len(val_texts)} samples")
        
        # Extract features from text
        print("Extracting features from text...")
        X_train = self.vectorizer.fit_transform(train_texts)
        X_val = self.vectorizer.transform(val_texts)
        
        # Train valence model
        print("Training valence model...")
        self.valence_model.fit(X_train, train_valence)
        val_valence_pred = self.valence_model.predict(X_val)
        valence_mse = mean_squared_error(val_valence, val_valence_pred)
        valence_corr, _ = pearsonr(val_valence, val_valence_pred)
        history['val_valence_mse'].append(valence_mse)
        history['val_valence_corr'].append(valence_corr)
        
        # Train arousal model
        print("Training arousal model...")
        self.arousal_model.fit(X_train, train_arousal)
        val_arousal_pred = self.arousal_model.predict(X_val)
        arousal_mse = mean_squared_error(val_arousal, val_arousal_pred)
        arousal_corr, _ = pearsonr(val_arousal, val_arousal_pred)
        history['val_arousal_mse'].append(arousal_mse)
        history['val_arousal_corr'].append(arousal_corr)
        
        # Train dominance model
        print("Training dominance model...")
        self.dominance_model.fit(X_train, train_dominance)
        val_dominance_pred = self.dominance_model.predict(X_val)
        dominance_mse = mean_squared_error(val_dominance, val_dominance_pred)
        dominance_corr, _ = pearsonr(val_dominance, val_dominance_pred)
        history['val_dominance_mse'].append(dominance_mse)
        history['val_dominance_corr'].append(dominance_corr)
        
        # Set trained flag
        self.is_trained = True
        
        # Print validation results
        print("\nValidation results:")
        print(f"Valence - MSE: {valence_mse:.4f}, Correlation: {valence_corr:.4f}")
        print(f"Arousal - MSE: {arousal_mse:.4f}, Correlation: {arousal_corr:.4f}")
        print(f"Dominance - MSE: {dominance_mse:.4f}, Correlation: {dominance_corr:.4f}")
        
        # Calculate overall metrics
        overall_mse = (valence_mse + arousal_mse + dominance_mse) / 3
        avg_corr = (valence_corr + arousal_corr + dominance_corr) / 3
        print(f"Overall MSE: {overall_mse:.4f}")
        print(f"Average correlation: {avg_corr:.4f}")
        
        return history
    
    def predict_vad(self, texts):
        """
        Predict VAD values for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            DataFrame with predicted VAD values
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Extract features from text
        X = self.vectorizer.transform(texts)
        
        # Predict VAD values
        valence_pred = self.valence_model.predict(X)
        arousal_pred = self.arousal_model.predict(X)
        dominance_pred = self.dominance_model.predict(X)
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'valence_pred': valence_pred,
            'arousal_pred': arousal_pred,
            'dominance_pred': dominance_pred
        })
        
        return df
    
    def save_model(self, output_dir):
        """
        Save the model to disk.
        
        Args:
            output_dir: Directory to save the model
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save vectorizer
        with open(os.path.join(output_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save models
        with open(os.path.join(output_dir, 'valence_model.pkl'), 'wb') as f:
            pickle.dump(self.valence_model, f)
        
        with open(os.path.join(output_dir, 'arousal_model.pkl'), 'wb') as f:
            pickle.dump(self.arousal_model, f)
        
        with open(os.path.join(output_dir, 'dominance_model.pkl'), 'wb') as f:
            pickle.dump(self.dominance_model, f)
        
        print(f"Model saved to {output_dir}")
    
    @classmethod
    def load_model(cls, model_dir):
        """
        Load a saved model from disk.
        
        Args:
            model_dir: Directory where the model is saved
            
        Returns:
            Loaded TextToVADModel
        """
        model = cls()
        
        # Load vectorizer
        with open(os.path.join(model_dir, 'vectorizer.pkl'), 'rb') as f:
            model.vectorizer = pickle.load(f)
        
        # Load models
        with open(os.path.join(model_dir, 'valence_model.pkl'), 'rb') as f:
            model.valence_model = pickle.load(f)
        
        with open(os.path.join(model_dir, 'arousal_model.pkl'), 'rb') as f:
            model.arousal_model = pickle.load(f)
        
        with open(os.path.join(model_dir, 'dominance_model.pkl'), 'rb') as f:
            model.dominance_model = pickle.load(f)
        
        model.is_trained = True
        
        return model

def plot_training_history(history, output_dir):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training history
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot MSE for each dimension
    plt.figure(figsize=(10, 6))
    plt.bar(['Valence', 'Arousal', 'Dominance'], 
            [history['val_valence_mse'][0], history['val_arousal_mse'][0], history['val_dominance_mse'][0]])
    plt.title('Validation MSE')
    plt.ylabel('MSE')
    plt.savefig(os.path.join(output_dir, 'mse.png'))
    plt.close()
    
    # Plot correlation for each dimension
    plt.figure(figsize=(10, 6))
    plt.bar(['Valence', 'Arousal', 'Dominance'], 
            [history['val_valence_corr'][0], history['val_arousal_corr'][0], history['val_dominance_corr'][0]])
    plt.title('Validation Correlation')
    plt.ylabel('Correlation')
    plt.savefig(os.path.join(output_dir, 'correlation.png'))
    plt.close()

def evaluate_model(model, texts, true_vad, output_dir):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained TextToVADModel
        texts: List of text strings
        true_vad: DataFrame with columns 'valence_norm', 'activation_norm', 'dominance_norm'
        output_dir: Directory to save the evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Predict VAD values
    predictions = model.predict_vad(texts)
    
    # Extract true values
    true_valence = true_vad['valence_norm'].values
    true_arousal = true_vad['activation_norm'].values
    true_dominance = true_vad['dominance_norm'].values
    
    # Extract predicted values
    pred_valence = predictions['valence_pred'].values
    pred_arousal = predictions['arousal_pred'].values
    pred_dominance = predictions['dominance_pred'].values
    
    # Calculate MSE for each dimension
    valence_mse = mean_squared_error(true_valence, pred_valence)
    arousal_mse = mean_squared_error(true_arousal, pred_arousal)
    dominance_mse = mean_squared_error(true_dominance, pred_dominance)
    overall_mse = (valence_mse + arousal_mse + dominance_mse) / 3
    
    # Calculate correlation for each dimension
    valence_corr, _ = pearsonr(true_valence, pred_valence)
    arousal_corr, _ = pearsonr(true_arousal, pred_arousal)
    dominance_corr, _ = pearsonr(true_dominance, pred_dominance)
    avg_corr = (valence_corr + arousal_corr + dominance_corr) / 3
    
    # Create evaluation metrics dictionary
    metrics = {
        'valence_mse': valence_mse,
        'arousal_mse': arousal_mse,
        'dominance_mse': dominance_mse,
        'overall_mse': overall_mse,
        'valence_corr': valence_corr,
        'arousal_corr': arousal_corr,
        'dominance_corr': dominance_corr,
        'avg_corr': avg_corr
    }
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot scatter plots for each dimension
    plt.figure(figsize=(18, 6))
    
    # Valence
    plt.subplot(1, 3, 1)
    plt.scatter(true_valence, pred_valence, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(f'Valence (MSE: {valence_mse:.4f}, Corr: {valence_corr:.4f})')
    plt.xlabel('True Valence')
    plt.ylabel('Predicted Valence')
    
    # Arousal
    plt.subplot(1, 3, 2)
    plt.scatter(true_arousal, pred_arousal, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(f'Arousal (MSE: {arousal_mse:.4f}, Corr: {arousal_corr:.4f})')
    plt.xlabel('True Arousal')
    plt.ylabel('Predicted Arousal')
    
    # Dominance
    plt.subplot(1, 3, 3)
    plt.scatter(true_dominance, pred_dominance, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(f'Dominance (MSE: {dominance_mse:.4f}, Corr: {dominance_corr:.4f})')
    plt.xlabel('True Dominance')
    plt.ylabel('Predicted Dominance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_plots.png'))
    plt.close()
    
    # Save predictions to file
    predictions['true_valence'] = true_valence
    predictions['true_arousal'] = true_arousal
    predictions['true_dominance'] = true_dominance
    predictions.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    return metrics

def main():
    """
    Main function to train and evaluate the Text to VAD model.
    """
    # Set paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(data_dir, 'data', 'processed')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    evaluation_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation')
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)
    
    # Load text data with VAD annotations
    text_data_path = os.path.join(processed_dir, 'iemocap_text_data.csv')
    
    if not os.path.exists(text_data_path):
        print(f"Text data file not found at {text_data_path}")
        print("Please run the preprocessing script first.")
        return
    
    text_df = pd.read_csv(text_data_path)
    
    print(f"Loaded text data with {len(text_df)} utterances")
    
    # Extract texts and VAD values
    texts = text_df['text'].tolist()
    vad_df = text_df[['valence_norm', 'activation_norm', 'dominance_norm']]
    
    # Split data into train and test sets
    train_texts, test_texts, train_vad, test_vad = train_test_split(
        texts, vad_df, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(train_texts)} utterances")
    print(f"Test set: {len(test_texts)} utterances")
    
    # Create and train model
    model = TextToVADModel()
    
    print("Training text-to-VAD model using scikit-learn...")
    history = model.train(train_texts, train_vad)
    
    # Plot training history
    plot_training_history(history, plots_dir)
    
    # Save model
    model.save_model(output_dir)
    
    # Evaluate model on test set
    print("Evaluating model on test set...")
    metrics = evaluate_model(model, test_texts, test_vad, evaluation_dir)
    
    # Print evaluation metrics
    print("\nEvaluation metrics:")
    print(f"Overall MSE: {metrics['overall_mse']:.4f}")
    print(f"Average correlation: {metrics['avg_corr']:.4f}")
    print(f"Valence - MSE: {metrics['valence_mse']:.4f}, Correlation: {metrics['valence_corr']:.4f}")
    print(f"Arousal - MSE: {metrics['arousal_mse']:.4f}, Correlation: {metrics['arousal_corr']:.4f}")
    print(f"Dominance - MSE: {metrics['dominance_mse']:.4f}, Correlation: {metrics['dominance_corr']:.4f}")
    
    # Example of how to use the model for prediction
    print("\nExample predictions:")
    sample_texts = [
        "I feel happy today.",
        "I am sad about what happened.",
        "This situation makes me feel angry.",
        "I'm excited about the upcoming event.",
        "I feel calm and relaxed."
    ]
    
    predictions = model.predict_vad(sample_texts)
    
    for i, text in enumerate(sample_texts):
        valence = predictions.iloc[i]['valence_pred']
        arousal = predictions.iloc[i]['arousal_pred']
        dominance = predictions.iloc[i]['dominance_pred']
        
        print(f"Text: {text}")
        print(f"Predicted VAD: V={valence:.2f}, A={arousal:.2f}, D={dominance:.2f}")
        print()

if __name__ == "__main__":
    main()
