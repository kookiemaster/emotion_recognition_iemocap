#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT-based text to VAD conversion module.
This module implements a proper text-to-VAD conversion using BERT for text processing.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm

class TextVADDataset(Dataset):
    """Dataset for text to VAD conversion."""
    
    def __init__(self, texts, vad_values, tokenizer, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text strings
            vad_values: DataFrame with columns 'valence_norm', 'arousal_norm', 'dominance_norm'
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.vad_values = vad_values
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        vad = np.array([
            self.vad_values.iloc[idx]['valence_norm'],
            self.vad_values.iloc[idx]['arousal_norm'],
            self.vad_values.iloc[idx]['dominance_norm']
        ], dtype=np.float32)
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Add VAD values
        encoding['vad'] = torch.tensor(vad, dtype=torch.float32)
        
        return encoding

class TextToVADBert(nn.Module):
    """BERT-based model for text to VAD conversion."""
    
    def __init__(self, bert_model_name='bert-base-uncased'):
        """
        Initialize the model.
        
        Args:
            bert_model_name: Name of the pre-trained BERT model
        """
        super(TextToVADBert, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.vad_predictor = nn.Linear(self.bert.config.hidden_size, 3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            
        Returns:
            VAD predictions
        """
        # Process text through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Predict VAD values
        vad_predictions = self.sigmoid(self.vad_predictor(pooled_output))
        
        return vad_predictions

class TextToVADModelBert:
    """
    A model for converting text to VAD (Valence, Arousal, Dominance) values using BERT.
    """
    
    def __init__(self, bert_model_name='bert-base-uncased', device=None):
        """
        Initialize the TextToVADModelBert.
        
        Args:
            bert_model_name: Name of the pre-trained BERT model
            device: Device to use for training and inference
        """
        self.bert_model_name = bert_model_name
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model = TextToVADBert(bert_model_name).to(self.device)
    
    def train(self, texts, vad_values, epochs=5, batch_size=16, learning_rate=2e-5, test_size=0.2, random_state=42):
        """
        Train the model on the given texts and VAD values.
        
        Args:
            texts: List of text strings
            vad_values: DataFrame with columns 'valence_norm', 'arousal_norm', 'dominance_norm'
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training history
        """
        # Split data into training and testing sets
        train_texts, test_texts, train_vad, test_vad = train_test_split(
            texts,
            vad_values,
            test_size=test_size,
            random_state=random_state
        )
        
        # Create datasets and dataloaders
        train_dataset = TextVADDataset(train_texts, train_vad, self.tokenizer)
        test_dataset = TextVADDataset(test_texts, test_vad, self.tokenizer)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_valence_mse': [],
            'val_arousal_mse': [],
            'val_dominance_mse': []
        }
        
        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Training
            self.model.train()
            train_loss = 0
            
            for batch in tqdm(train_dataloader, desc="Training"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                vad_targets = batch['vad'].to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                vad_predictions = self.model(input_ids, attention_mask)
                
                # Calculate loss
                loss = criterion(vad_predictions, vad_targets)
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update parameters
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_dataloader)
            history['train_loss'].append(train_loss)
            
            # Evaluation
            self.model.eval()
            val_loss = 0
            all_val_vad = []
            all_pred_vad = []
            
            with torch.no_grad():
                for batch in tqdm(test_dataloader, desc="Evaluation"):
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    vad_targets = batch['vad'].to(self.device)
                    
                    # Forward pass
                    vad_predictions = self.model(input_ids, attention_mask)
                    
                    # Calculate loss
                    loss = criterion(vad_predictions, vad_targets)
                    val_loss += loss.item()
                    
                    # Store predictions and targets
                    all_val_vad.append(vad_targets.cpu().numpy())
                    all_pred_vad.append(vad_predictions.cpu().numpy())
            
            val_loss /= len(test_dataloader)
            history['val_loss'].append(val_loss)
            
            # Calculate MSE for each dimension
            all_val_vad = np.vstack(all_val_vad)
            all_pred_vad = np.vstack(all_pred_vad)
            
            valence_mse = mean_squared_error(all_val_vad[:, 0], all_pred_vad[:, 0])
            arousal_mse = mean_squared_error(all_val_vad[:, 1], all_pred_vad[:, 1])
            dominance_mse = mean_squared_error(all_val_vad[:, 2], all_pred_vad[:, 2])
            
            history['val_valence_mse'].append(valence_mse)
            history['val_arousal_mse'].append(arousal_mse)
            history['val_dominance_mse'].append(dominance_mse)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val MSE - Valence: {valence_mse:.4f}, Arousal: {arousal_mse:.4f}, Dominance: {dominance_mse:.4f}")
        
        return history
    
    def predict_vad(self, texts, batch_size=16):
        """
        Predict VAD values for the given texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for prediction
            
        Returns:
            DataFrame with predicted VAD values
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Create dataset and dataloader
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Predict in batches
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                # Get batch
                batch_input_ids = encodings['input_ids'][i:i+batch_size].to(self.device)
                batch_attention_mask = encodings['attention_mask'][i:i+batch_size].to(self.device)
                
                # Forward pass
                batch_predictions = self.model(batch_input_ids, batch_attention_mask)
                
                # Store predictions
                all_predictions.append(batch_predictions.cpu().numpy())
        
        # Combine predictions
        all_predictions = np.vstack(all_predictions)
        
        # Create DataFrame
        predictions = pd.DataFrame({
            'valence_norm': all_predictions[:, 0],
            'arousal_norm': all_predictions[:, 1],
            'dominance_norm': all_predictions[:, 2]
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
            'bert_model_name': self.bert_model_name
        }
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        # Save model weights
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'model.pt'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
    
    @classmethod
    def load_model(cls, model_dir, device=None):
        """
        Load a saved model from disk.
        
        Args:
            model_dir: Directory where the model is saved
            device: Device to use for inference
            
        Returns:
            Loaded TextToVADModelBert
        """
        # Load configuration
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Create model
        model = cls(bert_model_name=config['bert_model_name'], device=device)
        
        # Load model weights
        model.model.load_state_dict(torch.load(
            os.path.join(model_dir, 'model.pt'),
            map_location=model.device
        ))
        
        return model

def plot_training_history(history, output_file):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training history
        output_file: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot MSE
    plt.subplot(2, 1, 2)
    plt.plot(history['val_valence_mse'], label='Valence MSE')
    plt.plot(history['val_arousal_mse'], label='Arousal MSE')
    plt.plot(history['val_dominance_mse'], label='Dominance MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Validation MSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Training history plot saved to {output_file}")

def main():
    """
    Main function to train and evaluate the BERT-based Text to VAD model.
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
    
    print("Creating a BERT-based text-to-VAD model using synthetic text data...")
    print(f"Number of utterances: {len(text_df)}")
    
    # Prepare data
    texts = text_df['text'].tolist()
    vad_values = text_df[['valence_norm', 'arousal_norm', 'dominance_norm']]
    
    # Create and train model
    model = TextToVADModelBert()
    history = model.train(texts, vad_values, epochs=3, batch_size=16)
    
    # Plot training history
    plot_training_history(history, os.path.join(output_dir, 'bert_training_history.png'))
    
    # Save model
    model.save_model(os.path.join(output_dir, 'bert_model'))
    
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
    
    result_df.to_csv(os.path.join(output_dir, 'bert_vad_predictions.csv'), index=False)
    print(f"All predictions saved to {os.path.join(output_dir, 'bert_vad_predictions.csv')}")

if __name__ == "__main__":
    main()
