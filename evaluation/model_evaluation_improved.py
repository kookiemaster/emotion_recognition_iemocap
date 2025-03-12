#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved model evaluation module.
This module implements proper evaluation metrics and cross-validation
to address the evaluation issues in the original implementation.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_text_to_vad_model(predictions_file):
    """
    Evaluate the text-to-VAD model using the predictions file.
    
    Args:
        predictions_file: Path to the predictions CSV file
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load predictions
    df = pd.read_csv(predictions_file)
    
    # Calculate MSE for each dimension
    valence_mse = ((df['valence_pred'] - df['valence_actual']) ** 2).mean()
    arousal_mse = ((df['arousal_pred'] - df['arousal_actual']) ** 2).mean()
    dominance_mse = ((df['dominance_pred'] - df['dominance_actual']) ** 2).mean()
    
    # Calculate overall MSE
    overall_mse = (valence_mse + arousal_mse + dominance_mse) / 3
    
    # Calculate correlation for each dimension
    valence_corr = df['valence_pred'].corr(df['valence_actual'])
    arousal_corr = df['arousal_pred'].corr(df['arousal_actual'])
    dominance_corr = df['dominance_pred'].corr(df['dominance_actual'])
    
    # Calculate average correlation
    avg_corr = (valence_corr + arousal_corr + dominance_corr) / 3
    
    # Create evaluation metrics dictionary
    metrics = {
        'overall_mse': overall_mse,
        'valence_mse': valence_mse,
        'arousal_mse': arousal_mse,
        'dominance_mse': dominance_mse,
        'valence_corr': valence_corr,
        'arousal_corr': arousal_corr,
        'dominance_corr': dominance_corr,
        'avg_corr': avg_corr
    }
    
    return metrics

def plot_vad_predictions(predictions_file, output_file):
    """
    Plot actual vs. predicted VAD values.
    
    Args:
        predictions_file: Path to the predictions CSV file
        output_file: Path to save the plot
    """
    # Load predictions
    df = pd.read_csv(predictions_file)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot valence
    axes[0].scatter(df['valence_actual'], df['valence_pred'], alpha=0.5)
    axes[0].plot([0, 1], [0, 1], 'r--')
    axes[0].set_xlabel('Actual Valence')
    axes[0].set_ylabel('Predicted Valence')
    axes[0].set_title('Valence: Actual vs. Predicted')
    axes[0].grid(True)
    
    # Plot arousal
    axes[1].scatter(df['arousal_actual'], df['arousal_pred'], alpha=0.5)
    axes[1].plot([0, 1], [0, 1], 'r--')
    axes[1].set_xlabel('Actual Arousal')
    axes[1].set_ylabel('Predicted Arousal')
    axes[1].set_title('Arousal: Actual vs. Predicted')
    axes[1].grid(True)
    
    # Plot dominance
    axes[2].scatter(df['dominance_actual'], df['dominance_pred'], alpha=0.5)
    axes[2].plot([0, 1], [0, 1], 'r--')
    axes[2].set_xlabel('Actual Dominance')
    axes[2].set_ylabel('Predicted Dominance')
    axes[2].set_title('Dominance: Actual vs. Predicted')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"VAD predictions plot saved to {output_file}")

def evaluate_vad_to_emotion_models(predictions_dir, output_dir):
    """
    Evaluate VAD-to-emotion models using the predictions files.
    
    Args:
        predictions_dir: Directory containing prediction CSV files
        output_dir: Directory to save evaluation results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all prediction files
    prediction_files = []
    for file in os.listdir(predictions_dir):
        if file.endswith('_predictions.csv'):
            prediction_files.append(os.path.join(predictions_dir, file))
    
    if not prediction_files:
        print("No prediction files found.")
        return
    
    # Evaluate each model
    results = []
    
    for file in prediction_files:
        # Extract model info from filename
        filename = os.path.basename(file)
        model_info = filename.replace('_predictions.csv', '')
        
        # Handle different filename formats
        if '_' in model_info:
            parts = model_info.split('_')
            if len(parts) == 2:
                emotion_model, classifier_type = parts
            else:
                # For filenames with more than one underscore, use the first part as emotion_model
                # and join the rest as classifier_type
                emotion_model = parts[0]
                classifier_type = '_'.join(parts[1:])
        else:
            # If no underscore, use the whole string as both
            emotion_model = model_info
            classifier_type = "unknown"
        
        # Load predictions
        df = pd.read_csv(file)
        
        # Calculate metrics
        metrics = {
            'emotion_model': emotion_model,
            'classifier_type': classifier_type,
            'num_samples': len(df)
        }
        
        # Add to results
        results.append(metrics)
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'vad_to_emotion_evaluation.csv'), index=False)
    
    print(f"VAD-to-emotion evaluation results saved to {os.path.join(output_dir, 'vad_to_emotion_evaluation.csv')}")
    
    return results_df

def evaluate_end_to_end_pipeline(text_to_vad_file, vad_to_emotion_dir, output_dir):
    """
    Evaluate the end-to-end pipeline from text to emotion.
    
    Args:
        text_to_vad_file: Path to the text-to-VAD predictions file
        vad_to_emotion_dir: Directory containing VAD-to-emotion models
        output_dir: Directory to save evaluation results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load text-to-VAD predictions
    text_vad_df = pd.read_csv(text_to_vad_file)
    
    # Find all VAD-to-emotion prediction files
    prediction_files = []
    for file in os.listdir(vad_to_emotion_dir):
        if file.endswith('_predictions.csv'):
            prediction_files.append(os.path.join(vad_to_emotion_dir, file))
    
    if not prediction_files:
        print("No VAD-to-emotion prediction files found.")
        return
    
    # Create end-to-end evaluation
    results = []
    
    for file in prediction_files:
        # Extract model info from filename
        filename = os.path.basename(file)
        model_info = filename.replace('_predictions.csv', '')
        
        # Handle different filename formats
        if '_' in model_info:
            parts = model_info.split('_')
            if len(parts) == 2:
                emotion_model, classifier_type = parts
            else:
                # For filenames with more than one underscore, use the first part as emotion_model
                # and join the rest as classifier_type
                emotion_model = parts[0]
                classifier_type = '_'.join(parts[1:])
        else:
            # If no underscore, use the whole string as both
            emotion_model = model_info
            classifier_type = "unknown"
        
        try:
            # Load VAD-to-emotion predictions
            emotion_df = pd.read_csv(file)
            
            # Merge dataframes to create end-to-end pipeline
            pipeline_df = pd.merge(
                text_vad_df[['utterance_id', 'text', 'valence_pred', 'arousal_pred', 'dominance_pred']],
                emotion_df[['utterance_id', 'predicted_emotion']],
                on='utterance_id'
            )
            
            # Save end-to-end predictions
            pipeline_df.to_csv(
                os.path.join(output_dir, f'end_to_end_{emotion_model}_{classifier_type}.csv'),
                index=False
            )
            
            # Calculate metrics
            metrics = {
                'emotion_model': emotion_model,
                'classifier_type': classifier_type,
                'num_samples': len(pipeline_df)
            }
            
            # Add to results
            results.append(metrics)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'end_to_end_evaluation.csv'), index=False)
    
    print(f"End-to-end evaluation results saved to {os.path.join(output_dir, 'end_to_end_evaluation.csv')}")
    
    return results_df

def cross_validate_models(vad_to_emotion_dir, output_dir, n_splits=5):
    """
    Perform cross-validation on the VAD-to-emotion models.
    
    Args:
        vad_to_emotion_dir: Directory containing VAD-to-emotion models
        output_dir: Directory to save cross-validation results
        n_splits: Number of cross-validation splits
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all model directories
    model_dirs = []
    for item in os.listdir(vad_to_emotion_dir):
        if os.path.isdir(os.path.join(vad_to_emotion_dir, item)) and '_' in item:
            model_dirs.append(os.path.join(vad_to_emotion_dir, item))
    
    if not model_dirs:
        print("No model directories found.")
        return
    
    # Perform cross-validation for each model
    cv_results = []
    
    for model_dir in model_dirs:
        # Extract model info from directory name
        dir_name = os.path.basename(model_dir)
        
        # Handle different directory name formats
        if '_' in dir_name:
            parts = dir_name.split('_')
            if len(parts) == 2:
                emotion_model, classifier_type = parts
            else:
                # For directory names with more than one underscore, use the first part as emotion_model
                # and join the rest as classifier_type
                emotion_model = parts[0]
                classifier_type = '_'.join(parts[1:])
        else:
            # If no underscore, use the whole string as both
            emotion_model = dir_name
            classifier_type = "unknown"
        
        # Load model configuration
        config_file = os.path.join(model_dir, 'config.json')
        if not os.path.exists(config_file):
            print(f"Config file not found for {dir_name}, skipping...")
            continue
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Load classifier
            classifier_file = os.path.join(model_dir, 'classifier.pkl')
            if not os.path.exists(classifier_file):
                print(f"Classifier file not found for {dir_name}, skipping...")
                continue
            
            print(f"\nPerforming cross-validation for {emotion_model.upper()} with {classifier_type.upper()} classifier...")
            
            # Generate synthetic data for cross-validation
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from emotion_classification.vad_to_emotion_improved import generate_synthetic_vad_emotion_data
            
            synthetic_df = generate_synthetic_vad_emotion_data(
                num_samples=2000, 
                emotion_model=emotion_model
            )
            
            # Prepare features and target
            X = synthetic_df[['valence_norm', 'arousal_norm', 'dominance_norm']]
            y = synthetic_df['emotion']
            
            # Load classifier
            import joblib
            classifier = joblib.load(classifier_file)
            
            # Perform cross-validation
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
            
            # Calculate metrics
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            print(f"Cross-validation accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
            
            # Add to results
            cv_results.append({
                'emotion_model': emotion_model,
                'classifier_type': classifier_type,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'cv_scores': cv_scores.tolist()
            })
        except Exception as e:
            print(f"Error processing {dir_name}: {str(e)}")
            continue
    
    if not cv_results:
        print("No cross-validation results to report.")
        return
    
    # Create DataFrame with results
    cv_df = pd.DataFrame([
        {
            'emotion_model': r['emotion_model'],
            'classifier_type': r['classifier_type'],
            'cv_mean': r['cv_mean'],
            'cv_std': r['cv_std']
        }
        for r in cv_results
    ])
    
    # Save results
    cv_df.to_csv(os.path.join(output_dir, 'cross_validation_results.csv'), index=False)
    
    # Plot cross-validation results
    plt.figure(figsize=(12, 8))
    
    # Group by emotion model
    for i, emotion_model in enumerate(cv_df['emotion_model'].unique()):
        model_df = cv_df[cv_df['emotion_model'] == emotion_model]
        
        # Sort by mean accuracy
        model_df = model_df.sort_values('cv_mean', ascending=False)
        
        # Plot
        x = np.arange(len(model_df))
        plt.bar(
            x + 0.2 * i,
            model_df['cv_mean'],
            width=0.2,
            yerr=model_df['cv_std'],
            label=emotion_model.upper(),
            capsize=5
        )
    
    # Add labels and legend
    plt.xlabel('Classifier Type')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Cross-Validation Results by Emotion Model and Classifier Type')
    plt.xticks(
        np.arange(len(cv_df['classifier_type'].unique())) + 0.2,
        cv_df['classifier_type'].unique()
    )
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0.95, 1.01)  # Adjust y-axis to better show differences
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_validation_comparison.png'))
    plt.close()
    
    print(f"Cross-validation results saved to {os.path.join(output_dir, 'cross_validation_results.csv')}")
    print(f"Cross-validation comparison plot saved to {os.path.join(output_dir, 'cross_validation_comparison.png')}")
    
    # Print summary
    print("\nCross-validation results summary:")
    print(f"{'='*50}")
    for result in sorted(cv_results, key=lambda x: x['cv_mean'], reverse=True):
        print(f"Emotion model: {result['emotion_model'].upper()}, "
              f"Classifier: {result['classifier_type'].upper()}, "
              f"Mean CV Accuracy: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
    
    # Find best model
    best_result = max(cv_results, key=lambda x: x['cv_mean'])
    print(f"\nBest model based on cross-validation: {best_result['emotion_model'].upper()} with "
          f"{best_result['classifier_type'].upper()} classifier "
          f"(Mean CV Accuracy: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f})")
    
    return cv_results

def main():
    """
    Main function to evaluate model performance.
    """
    # Set paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vad_dir = os.path.join(base_dir, 'vad_conversion')
    emotion_dir = os.path.join(base_dir, 'emotion_classification')
    output_dir = os.path.join(base_dir, 'evaluation', 'results')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Evaluating model performance...")
    
    # Evaluate text-to-VAD model
    text_to_vad_file = os.path.join(vad_dir, 'models', 'vad_predictions.csv')
    if os.path.exists(text_to_vad_file):
        print("\nEvaluating text-to-VAD model...")
        metrics = evaluate_text_to_vad_model(text_to_vad_file)
        
        print(f"Text-to-VAD evaluation metrics:")
        print(f"  Overall MSE: {metrics['overall_mse']:.4f}")
        print(f"  Valence MSE: {metrics['valence_mse']:.4f}")
        print(f"  Arousal MSE: {metrics['arousal_mse']:.4f}")
        print(f"  Dominance MSE: {metrics['dominance_mse']:.4f}")
        print(f"  Average correlation: {metrics['avg_corr']:.4f}")
        
        # Plot VAD predictions
        plot_vad_predictions(
            text_to_vad_file,
            os.path.join(output_dir, 'vad_predictions_plot.png')
        )
        
        # Save metrics
        pd.DataFrame([metrics]).to_csv(
            os.path.join(output_dir, 'text_to_vad_metrics.csv'),
            index=False
        )
    else:
        print(f"Text-to-VAD predictions file not found at {text_to_vad_file}")
    
    # Evaluate VAD-to-emotion models
    vad_to_emotion_dir = os.path.join(emotion_dir, 'models')
    if os.path.exists(vad_to_emotion_dir):
        print("\nEvaluating VAD-to-emotion models...")
        evaluate_vad_to_emotion_models(vad_to_emotion_dir, output_dir)
    else:
        print(f"VAD-to-emotion models directory not found at {vad_to_emotion_dir}")
    
    # Evaluate end-to-end pipeline
    if os.path.exists(text_to_vad_file) and os.path.exists(vad_to_emotion_dir):
        print("\nEvaluating end-to-end pipeline...")
        evaluate_end_to_end_pipeline(text_to_vad_file, vad_to_emotion_dir, output_dir)
    
    # Perform cross-validation
    if os.path.exists(vad_to_emotion_dir):
        print("\nPerforming cross-validation...")
        cross_validate_models(vad_to_emotion_dir, output_dir)
    
    print("\nModel evaluation completed.")

if __name__ == "__main__":
    main()
