#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved text to VAD conversion module.
This module implements a proper text-to-VAD conversion using NLP techniques
instead of using utterance IDs as placeholders.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt

class TextToVADModel:
    """
    A model for converting text to VAD (Valence, Arousal, Dominance) values.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the TextToVADModel.
        
        Args:
            model_type: Type of model to use ('random_forest' or other options in the future)
        """
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.85,
            ngram_range=(1, 2)
        )
        
        if model_type == 'random_forest':
            self.valence_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                random_state=42
            )
            self.arousal_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                random_state=42
            )
            self.dominance_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, texts, vad_values, test_size=0.2, random_state=42):
        """
        Train the model on the given texts and VAD values.
        
        Args:
            texts: List of text strings
            vad_values: DataFrame with columns 'valence_norm', 'arousal_norm', 'dominance_norm'
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training history
        """
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            vad_values,
            test_size=test_size,
            random_state=random_state
        )
        
        # Transform text to features
        X_train_features = self.vectorizer.fit_transform(X_train)
        X_test_features = self.vectorizer.transform(X_test)
        
        # Train valence model
        self.valence_model.fit(X_train_features, y_train['valence_norm'])
        valence_train_pred = self.valence_model.predict(X_train_features)
        valence_test_pred = self.valence_model.predict(X_test_features)
        valence_train_mse = mean_squared_error(y_train['valence_norm'], valence_train_pred)
        valence_test_mse = mean_squared_error(y_test['valence_norm'], valence_test_pred)
        
        # Train arousal model
        self.arousal_model.fit(X_train_features, y_train['arousal_norm'])
        arousal_train_pred = self.arousal_model.predict(X_train_features)
        arousal_test_pred = self.arousal_model.predict(X_test_features)
        arousal_train_mse = mean_squared_error(y_train['arousal_norm'], arousal_train_pred)
        arousal_test_mse = mean_squared_error(y_test['arousal_norm'], arousal_test_pred)
        
        # Train dominance model
        self.dominance_model.fit(X_train_features, y_train['dominance_norm'])
        dominance_train_pred = self.dominance_model.predict(X_train_features)
        dominance_test_pred = self.dominance_model.predict(X_test_features)
        dominance_train_mse = mean_squared_error(y_train['dominance_norm'], dominance_train_pred)
        dominance_test_mse = mean_squared_error(y_test['dominance_norm'], dominance_test_pred)
        
        # Calculate overall MSE
        train_mse = (valence_train_mse + arousal_train_mse + dominance_train_mse) / 3
        test_mse = (valence_test_mse + arousal_test_mse + dominance_test_mse) / 3
        
        # Store training history
        history = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'valence_train_mse': valence_train_mse,
            'valence_test_mse': valence_test_mse,
            'arousal_train_mse': arousal_train_mse,
            'arousal_test_mse': arousal_test_mse,
            'dominance_train_mse': dominance_train_mse,
            'dominance_test_mse': dominance_test_mse,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        print(f"Training completed:")
        print(f"  Train MSE: {train_mse:.4f}")
        print(f"  Test MSE: {test_mse:.4f}")
        print(f"  Valence - Train MSE: {valence_train_mse:.4f}, Test MSE: {valence_test_mse:.4f}")
        print(f"  Arousal - Train MSE: {arousal_train_mse:.4f}, Test MSE: {arousal_test_mse:.4f}")
        print(f"  Dominance - Train MSE: {dominance_train_mse:.4f}, Test MSE: {dominance_test_mse:.4f}")
        
        return history
    
    def predict_vad(self, texts):
        """
        Predict VAD values for the given texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            DataFrame with predicted VAD values
        """
        # Transform text to features
        X_features = self.vectorizer.transform(texts)
        
        # Predict VAD values
        valence_pred = self.valence_model.predict(X_features)
        arousal_pred = self.arousal_model.predict(X_features)
        dominance_pred = self.dominance_model.predict(X_features)
        
        # Create DataFrame with predictions
        predictions = pd.DataFrame({
            'valence_norm': valence_pred,
            'arousal_norm': arousal_pred,
            'dominance_norm': dominance_pred
        })
        
        return predictions
    
    def save_model(self, output_dir):
        """
        Save the model to disk.
        
        Args:
            output_dir: Directory to save the model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model configuration
        config = {
            'model_type': self.model_type
        }
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        # Save vectorizer
        joblib.dump(self.vectorizer, os.path.join(output_dir, 'vectorizer.pkl'))
        
        # Save models
        joblib.dump(self.valence_model, os.path.join(output_dir, 'valence_model.pkl'))
        joblib.dump(self.arousal_model, os.path.join(output_dir, 'arousal_model.pkl'))
        joblib.dump(self.dominance_model, os.path.join(output_dir, 'dominance_model.pkl'))
        
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
        # Load configuration
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Create model
        model = cls(model_type=config['model_type'])
        
        # Load vectorizer
        model.vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.pkl'))
        
        # Load models
        model.valence_model = joblib.load(os.path.join(model_dir, 'valence_model.pkl'))
        model.arousal_model = joblib.load(os.path.join(model_dir, 'arousal_model.pkl'))
        model.dominance_model = joblib.load(os.path.join(model_dir, 'dominance_model.pkl'))
        
        return model

def plot_training_history(history, output_file):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training history
        output_file: Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Plot MSE
    plt.subplot(1, 2, 1)
    plt.bar(['Train', 'Test'], [history['train_mse'], history['test_mse']])
    plt.ylabel('Mean Squared Error')
    plt.title('Overall MSE')
    
    # Plot MSE by dimension
    plt.subplot(1, 2, 2)
    dimensions = ['Valence', 'Arousal', 'Dominance']
    train_mse = [history['valence_train_mse'], history['arousal_train_mse'], history['dominance_train_mse']]
    test_mse = [history['valence_test_mse'], history['arousal_test_mse'], history['dominance_test_mse']]
    
    x = np.arange(len(dimensions))
    width = 0.35
    
    plt.bar(x - width/2, train_mse, width, label='Train')
    plt.bar(x + width/2, test_mse, width, label='Test')
    
    plt.xlabel('Dimension')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE by Dimension')
    plt.xticks(x, dimensions)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Training history plot saved to {output_file}")

def main():
    """
    Main function to train and evaluate the Text to VAD model.
    """
    # Set paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    processed_dir = os.path.join(data_dir, 'processed')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load text dataset
    text_dataset_path = os.path.join(processed_dir, 'text_dataset.csv')
    if not os.path.exists(text_dataset_path):
        print(f"Error: Text dataset not found at {text_dataset_path}")
        print("Please run the text extraction script first.")
        return
    
    text_df = pd.read_csv(text_dataset_path)
    
    print("Creating a text-to-VAD model using actual text data...")
    print(f"Number of utterances: {len(text_df)}")
    
    # Prepare data
    texts = text_df['text'].tolist()
    vad_values = text_df[['valence_norm', 'arousal_norm', 'dominance_norm']]
    
    # Create and train model
    model = TextToVADModel()
    history = model.train(texts, vad_values)
    
    # Plot training history
    plot_training_history(history, os.path.join(output_dir, 'training_history.png'))
    
    # Save model
    model.save_model(output_dir)
    
    # Example of how to use the model for prediction
    sample_texts = texts[:5]
    predictions = model.predict_vad(sample_texts)
    
    print("\nSample predictions:")
    for i, (text, pred) in enumerate(zip(sample_texts, predictions.itertuples())):
        print(f"Text: {text}")
        print(f"Predicted VAD: V={pred.valence_norm:.4f}, A={pred.arousal_norm:.4f}, D={pred.dominance_norm:.4f}")
        print(f"Actual VAD: V={vad_values.iloc[i]['valence_norm']:.4f}, A={vad_values.iloc[i]['arousal_norm']:.4f}, D={vad_values.iloc[i]['dominance_norm']:.4f}")
        print()
    
    # Save predictions for all data
    all_predictions = model.predict_vad(texts)
    result_df = pd.DataFrame({
        'utterance_id': text_df['utterance_id'],
        'text': texts,
        'valence_actual': vad_values['valence_norm'],
        'arousal_actual': vad_values['arousal_norm'],
        'dominance_actual': vad_values['dominance_norm'],
        'valence_pred': all_predictions['valence_norm'],
        'arousal_pred': all_predictions['arousal_norm'],
        'dominance_pred': all_predictions['dominance_norm']
    })
    
    result_df.to_csv(os.path.join(output_dir, 'vad_predictions.csv'), index=False)
    print(f"All predictions saved to {os.path.join(output_dir, 'vad_predictions.csv')}")

if __name__ == "__main__":
    main()
