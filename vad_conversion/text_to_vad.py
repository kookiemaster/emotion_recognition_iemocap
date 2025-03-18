#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text to VAD conversion model for emotion recognition.
This script implements a BERT-based model that converts text to VAD (Valence-Arousal-Dominance) values.
"""

import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from scipy.stats import pearsonr

class BERTTextToVADModel:
    def __init__(self, model_name="distilbert-base-uncased"):
        """
        Initialize the BERT-based Text to VAD conversion model.
        
        Args:
            model_name: Name of the pre-trained transformer model to use
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Linear layers for VAD prediction
        self.valence_layer = torch.nn.Linear(768, 1).to(self.device)
        self.arousal_layer = torch.nn.Linear(768, 1).to(self.device)
        self.dominance_layer = torch.nn.Linear(768, 1).to(self.device)
        
        # Optimizer
        self.optimizer = AdamW([
            {'params': self.model.parameters(), 'lr': 1e-5},
            {'params': self.valence_layer.parameters(), 'lr': 1e-4},
            {'params': self.arousal_layer.parameters(), 'lr': 1e-4},
            {'params': self.dominance_layer.parameters(), 'lr': 1e-4}
        ])
        
        # Loss function
        self.criterion = torch.nn.MSELoss()
    
    def get_text_embeddings(self, texts):
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tensor of embeddings
        """
        # Tokenize texts
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embedding
        
        return embeddings
    
    def predict_vad(self, texts):
        """
        Predict VAD values for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            DataFrame with predicted VAD values
        """
        self.model.eval()
        
        # Process in batches to handle large datasets
        batch_size = 32
        all_valence = []
        all_arousal = []
        all_dominance = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Get embeddings
            with torch.no_grad():
                embeddings = self.get_text_embeddings(batch_texts)
                
                # Predict VAD values
                valence = self.valence_layer(embeddings).squeeze().cpu().numpy()
                arousal = self.arousal_layer(embeddings).squeeze().cpu().numpy()
                dominance = self.dominance_layer(embeddings).squeeze().cpu().numpy()
                
                # Handle single item case
                if len(batch_texts) == 1:
                    valence = np.array([valence])
                    arousal = np.array([arousal])
                    dominance = np.array([dominance])
                
                all_valence.extend(valence)
                all_arousal.extend(arousal)
                all_dominance.extend(dominance)
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'valence_pred': all_valence,
            'arousal_pred': all_arousal,
            'dominance_pred': all_dominance
        })
        
        return df
    
    def train(self, texts, vad_values, epochs=10, batch_size=16, validation_split=0.2):
        """
        Train the model on texts and their VAD values.
        
        Args:
            texts: List of text strings
            vad_values: DataFrame with columns 'valence_norm', 'activation_norm', 'dominance_norm'
            epochs: Number of training epochs
            batch_size: Batch size for training
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
            'val_loss': [],
            'train_loss': [],
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
        
        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Train
            self.model.train()
            train_loss = 0
            
            # Create batches
            indices = np.arange(len(train_texts))
            np.random.shuffle(indices)
            
            # Process in batches
            for start_idx in tqdm(range(0, len(train_texts), batch_size), desc="Training"):
                batch_indices = indices[start_idx:start_idx+batch_size]
                batch_texts = [train_texts[i] for i in batch_indices]
                batch_valence = torch.tensor(train_valence[batch_indices], dtype=torch.float32).to(self.device)
                batch_arousal = torch.tensor(train_arousal[batch_indices], dtype=torch.float32).to(self.device)
                batch_dominance = torch.tensor(train_dominance[batch_indices], dtype=torch.float32).to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Get embeddings
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
                
                # Predict VAD values
                pred_valence = self.valence_layer(embeddings).squeeze()
                pred_arousal = self.arousal_layer(embeddings).squeeze()
                pred_dominance = self.dominance_layer(embeddings).squeeze()
                
                # Calculate loss
                loss_valence = self.criterion(pred_valence, batch_valence)
                loss_arousal = self.criterion(pred_arousal, batch_arousal)
                loss_dominance = self.criterion(pred_dominance, batch_dominance)
                loss = loss_valence + loss_arousal + loss_dominance
                
                # Backpropagation
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * len(batch_indices)
            
            train_loss /= len(train_texts)
            history['train_loss'].append(train_loss)
            
            # Validate
            self.model.eval()
            val_loss = 0
            all_val_valence = []
            all_val_arousal = []
            all_val_dominance = []
            all_pred_valence = []
            all_pred_arousal = []
            all_pred_dominance = []
            
            with torch.no_grad():
                for start_idx in tqdm(range(0, len(val_texts), batch_size), desc="Validating"):
                    batch_texts = val_texts[start_idx:start_idx+batch_size]
                    batch_valence = torch.tensor(val_valence[start_idx:start_idx+batch_size], dtype=torch.float32).to(self.device)
                    batch_arousal = torch.tensor(val_arousal[start_idx:start_idx+batch_size], dtype=torch.float32).to(self.device)
                    batch_dominance = torch.tensor(val_dominance[start_idx:start_idx+batch_size], dtype=torch.float32).to(self.device)
                    
                    # Get embeddings
                    inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]
                    
                    # Predict VAD values
                    pred_valence = self.valence_layer(embeddings).squeeze()
                    pred_arousal = self.arousal_layer(embeddings).squeeze()
                    pred_dominance = self.dominance_layer(embeddings).squeeze()
                    
                    # Handle single item case
                    if len(batch_texts) == 1:
                        pred_valence = pred_valence.unsqueeze(0)
                        pred_arousal = pred_arousal.unsqueeze(0)
                        pred_dominance = pred_dominance.unsqueeze(0)
                    
                    # Calculate loss
                    loss_valence = self.criterion(pred_valence, batch_valence)
                    loss_arousal = self.criterion(pred_arousal, batch_arousal)
                    loss_dominance = self.criterion(pred_dominance, batch_dominance)
                    loss = loss_valence + loss_arousal + loss_dominance
                    
                    val_loss += loss.item() * len(batch_texts)
                    
                    # Store predictions and true values
                    all_val_valence.extend(batch_valence.cpu().numpy())
                    all_val_arousal.extend(batch_arousal.cpu().numpy())
                    all_val_dominance.extend(batch_dominance.cpu().numpy())
                    all_pred_valence.extend(pred_valence.cpu().numpy())
                    all_pred_arousal.extend(pred_arousal.cpu().numpy())
                    all_pred_dominance.extend(pred_dominance.cpu().numpy())
            
            val_loss /= len(val_texts)
            history['val_loss'].append(val_loss)
            
            # Calculate MSE for each dimension
            valence_mse = mean_squared_error(all_val_valence, all_pred_valence)
            arousal_mse = mean_squared_error(all_val_arousal, all_pred_arousal)
            dominance_mse = mean_squared_error(all_val_dominance, all_pred_dominance)
            
            # Calculate correlation for each dimension
            valence_corr, _ = pearsonr(all_val_valence, all_pred_valence)
            arousal_corr, _ = pearsonr(all_val_arousal, all_pred_arousal)
            dominance_corr, _ = pearsonr(all_val_dominance, all_pred_dominance)
            
            history['val_valence_mse'].append(valence_mse)
            history['val_arousal_mse'].append(arousal_mse)
            history['val_dominance_mse'].append(dominance_mse)
            history['val_valence_corr'].append(valence_corr)
            history['val_arousal_corr'].append(arousal_corr)
            history['val_dominance_corr'].append(dominance_corr)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val MSE - Valence: {valence_mse:.4f}, Arousal: {arousal_mse:.4f}, Dominance: {dominance_mse:.4f}")
            print(f"Val Corr - Valence: {valence_corr:.4f}, Arousal: {arousal_corr:.4f}, Dominance: {dominance_corr:.4f}")
        
        return history
    
    def save_model(self, output_dir):
        """
        Save the model to disk.
        
        Args:
            output_dir: Directory to save the model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model configuration
        config = {
            'model_name': self.model_name
        }
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        # Save model weights
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'model.pt'))
        torch.save(self.valence_layer.state_dict(), os.path.join(output_dir, 'valence_layer.pt'))
        torch.save(self.arousal_layer.state_dict(), os.path.join(output_dir, 'arousal_layer.pt'))
        torch.save(self.dominance_layer.state_dict(), os.path.join(output_dir, 'dominance_layer.pt'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
    
    @classmethod
    def load_model(cls, model_dir):
        """
        Load a saved model from disk.
        
        Args:
            model_dir: Directory where the model is saved
            
        Returns:
            Loaded BERTTextToVADModel
        """
        # Load configuration
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Create model
        model = cls(model_name=config['model_name'])
        
        # Load weights
        model.model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt')))
        model.valence_layer.load_state_dict(torch.load(os.path.join(model_dir, 'valence_layer.pt')))
        model.arousal_layer.load_state_dict(torch.load(os.path.join(model_dir, 'arousal_layer.pt')))
        model.dominance_layer.load_state_dict(torch.load(os.path.join(model_dir, 'dominance_layer.pt')))
        
        return model

def plot_training_history(history, output_dir):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training history
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()
    
    # Plot MSE for each dimension
    plt.figure(figsize=(10, 6))
    plt.plot(history['val_valence_mse'], label='Valence MSE')
    plt.plot(history['val_arousal_mse'], label='Arousal MSE')
    plt.plot(history['val_dominance_mse'], label='Dominance MSE')
    plt.title('Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'mse.png'))
    plt.close()
    
    # Plot correlation for each dimension
    plt.figure(figsize=(10, 6))
    plt.plot(history['val_valence_corr'], label='Valence Correlation')
    plt.plot(history['val_arousal_corr'], label='Arousal Correlation')
    plt.plot(history['val_dominance_corr'], label='Dominance Correlation')
    plt.title('Validation Correlation')
    plt.xlabel('Epoch')
    plt.ylabel('Correlation')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'correlation.png'))
    plt.close()

def evaluate_model(model, texts, true_vad, output_dir):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained BERTTextToVADModel
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
    Main function to train and evaluate the BERT-based Text to VAD model.
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
    model = BERTTextToVADModel()
    
    print("Training BERT-based text-to-VAD model...")
    history = model.train(train_texts, train_vad, epochs=5, batch_size=16)
    
    # Plot training history
    plot_training_history(history, plots_dir)
    
    # Save model
    model.save_model(output_dir)
    print(f"Model saved to {output_dir}")
    
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
