#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end pipeline for emotion recognition.
This script implements the complete pipeline from text to emotion prediction.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import joblib
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from vad_conversion.text_to_vad_sklearn import TextToVADModel
from emotion_classification.vad_to_emotion_classifier import VADToEmotionClassifier

def load_models(text_to_vad_model_dir, vad_to_emotion_model_dir):
    """
    Load the trained models.
    
    Args:
        text_to_vad_model_dir: Directory containing the text-to-VAD model
        vad_to_emotion_model_dir: Directory containing the VAD-to-emotion model
        
    Returns:
        Tuple of (text_to_vad_model, vad_to_emotion_model)
    """
    # Load text-to-VAD model
    text_to_vad_model = TextToVADModel.load_model(text_to_vad_model_dir)
    
    # Load VAD-to-emotion model
    vad_to_emotion_model = VADToEmotionClassifier.load_model(vad_to_emotion_model_dir)
    
    return text_to_vad_model, vad_to_emotion_model

def predict_emotions_from_text(texts, text_to_vad_model, vad_to_emotion_model):
    """
    Predict emotions from text using the complete pipeline.
    
    Args:
        texts: List of text strings
        text_to_vad_model: Trained text-to-VAD model
        vad_to_emotion_model: Trained VAD-to-emotion model
        
    Returns:
        DataFrame with predicted emotions and VAD values
    """
    # Predict VAD values from text
    vad_predictions = text_to_vad_model.predict_vad(texts)
    
    # Extract VAD values
    X_vad = vad_predictions[['valence_pred', 'arousal_pred', 'dominance_pred']]
    
    # Rename columns to match expected input for VAD-to-emotion model
    X_vad.columns = ['valence_norm', 'activation_norm', 'dominance_norm']
    
    # Predict emotions from VAD values
    emotion_predictions = vad_to_emotion_model.predict(X_vad)
    
    # Combine results
    results = pd.DataFrame({
        'text': texts,
        'valence': X_vad['valence_norm'],
        'arousal': X_vad['activation_norm'],
        'dominance': X_vad['dominance_norm'],
        'predicted_emotion': emotion_predictions['predicted_emotion']
    })
    
    # Add emotion probabilities
    for col in emotion_predictions.columns:
        if col.startswith('prob_'):
            results[col] = emotion_predictions[col]
    
    return results

def evaluate_pipeline(test_data, text_to_vad_model, vad_to_emotion_model, emotion_column):
    """
    Evaluate the complete pipeline on test data.
    
    Args:
        test_data: DataFrame with text and true emotion labels
        text_to_vad_model: Trained text-to-VAD model
        vad_to_emotion_model: Trained VAD-to-emotion model
        emotion_column: Column name for true emotion labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Extract texts and true emotions
    texts = test_data['text'].tolist()
    true_emotions = test_data[emotion_column].tolist()
    
    # Predict emotions
    predictions = predict_emotions_from_text(texts, text_to_vad_model, vad_to_emotion_model)
    predicted_emotions = predictions['predicted_emotion'].tolist()
    
    # Calculate metrics
    accuracy = accuracy_score(true_emotions, predicted_emotions)
    f1 = f1_score(true_emotions, predicted_emotions, average='weighted')
    
    # Create classification report
    report = classification_report(true_emotions, predicted_emotions, output_dict=True)
    
    # Create confusion matrix
    cm = confusion_matrix(true_emotions, predicted_emotions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=np.unique(true_emotions),
               yticklabels=np.unique(true_emotions))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - End-to-End Pipeline')
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_end_to_end.png'))
    plt.close()
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': report
    }
    
    return metrics, predictions

def main():
    """
    Main function to run the end-to-end pipeline.
    """
    # Set paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, 'data', 'processed')
    text_to_vad_model_dir = os.path.join(project_dir, 'vad_conversion', 'models')
    vad_to_emotion_model_dir = os.path.join(project_dir, 'emotion_classification', 'models', 'quadrant_random_forest')
    output_dir = os.path.join(project_dir, 'evaluation', 'results')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Running end-to-end emotion recognition pipeline...")
    
    # Load text data
    text_data_path = os.path.join(data_dir, 'iemocap_text_data.csv')
    if not os.path.exists(text_data_path):
        print(f"Text data file not found at {text_data_path}")
        return
    
    text_df = pd.read_csv(text_data_path)
    print(f"Loaded text data with {len(text_df)} utterances")
    
    # Load emotion labels
    emotion_data_path = os.path.join(project_dir, 'emotion_classification', 'vad_with_emotions.csv')
    if not os.path.exists(emotion_data_path):
        print(f"Emotion data file not found at {emotion_data_path}")
        return
    
    emotion_df = pd.read_csv(emotion_data_path)
    print(f"Loaded emotion data with {len(emotion_df)} utterances")
    
    # Merge text and emotion data
    merged_df = pd.merge(text_df, emotion_df, on='utterance_id')
    print(f"Merged data has {len(merged_df)} utterances")
    
    # Split data into train and test sets
    train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42)
    print(f"Training set: {len(train_df)} utterances")
    print(f"Test set: {len(test_df)} utterances")
    
    # Load models
    print("Loading models...")
    text_to_vad_model, vad_to_emotion_model = load_models(
        text_to_vad_model_dir, vad_to_emotion_model_dir
    )
    
    # Evaluate pipeline for each emotion model
    emotion_models = ['emotion_quadrant', 'emotion_custom', 'emotion_plutchik', 'emotion_ekman']
    
    results = []
    
    for emotion_model in emotion_models:
        print(f"\nEvaluating pipeline with {emotion_model} emotion model...")
        
        # Update VAD-to-emotion model
        model_dir = os.path.join(project_dir, 'emotion_classification', 'models', 
                                f"{emotion_model.replace('emotion_', '')}_{vad_to_emotion_model.classifier_type}")
        vad_to_emotion_model = VADToEmotionClassifier.load_model(model_dir)
        
        # Evaluate pipeline
        metrics, predictions = evaluate_pipeline(test_df, text_to_vad_model, vad_to_emotion_model, emotion_model)
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 score: {metrics['f1_score']:.4f}")
        print("Classification report:")
        print(classification_report(test_df[emotion_model], predictions['predicted_emotion']))
        
        # Save predictions
        predictions.to_csv(os.path.join(output_dir, f'predictions_{emotion_model}.csv'), index=False)
        
        # Save metrics
        with open(os.path.join(output_dir, f'metrics_{emotion_model}.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Store results
        results.append({
            'emotion_model': emotion_model,
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score']
        })
    
    # Print summary of results
    print("\nSummary of results:")
    print("=" * 50)
    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"Emotion model: {result['emotion_model']}, "
              f"Accuracy: {result['accuracy']:.4f}, "
              f"F1 score: {result['f1_score']:.4f}")
    
    # Find best model
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\nBest model: {best_result['emotion_model']} "
          f"(Accuracy: {best_result['accuracy']:.4f}, "
          f"F1 score: {best_result['f1_score']:.4f})")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'pipeline_results.csv'), index=False)
    print(f"Results saved to {os.path.join(output_dir, 'pipeline_results.csv')}")
    
    # Example predictions
    print("\nExample predictions:")
    sample_texts = [
        "I feel happy today.",
        "I am sad about what happened.",
        "This situation makes me feel angry.",
        "I'm excited about the upcoming event.",
        "I feel calm and relaxed."
    ]
    
    # Use the best model for example predictions
    best_model_name = best_result['emotion_model'].replace('emotion_', '')
    best_model_dir = os.path.join(project_dir, 'emotion_classification', 'models', 
                                f"{best_model_name}_random_forest")
    best_vad_to_emotion_model = VADToEmotionClassifier.load_model(best_model_dir)
    
    example_predictions = predict_emotions_from_text(
        sample_texts, text_to_vad_model, best_vad_to_emotion_model
    )
    
    for i, text in enumerate(sample_texts):
        valence = example_predictions.iloc[i]['valence']
        arousal = example_predictions.iloc[i]['arousal']
        dominance = example_predictions.iloc[i]['dominance']
        emotion = example_predictions.iloc[i]['predicted_emotion']
        
        print(f"Text: {text}")
        print(f"Predicted VAD: V={valence:.2f}, A={arousal:.2f}, D={dominance:.2f}")
        print(f"Predicted emotion: {emotion}")
        print()

if __name__ == "__main__":
    main()
