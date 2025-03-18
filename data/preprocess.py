#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data preprocessing script for IEMOCAP dataset.
This script extracts VAD (Valence-Arousal-Dominance) annotations and generates synthetic data.
"""

import os
import re
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import random

def create_synthetic_vad_annotations(num_samples=1144, output_path=None):
    """
    Create synthetic VAD annotations for IEMOCAP dataset.
    
    Args:
        num_samples: Number of samples to generate
        output_path: Path to save the generated annotations
        
    Returns:
        DataFrame with utterance IDs and their VAD values
    """
    data = []
    
    # Generate session and utterance IDs similar to IEMOCAP format
    for session_id in range(1, 6):  # 5 sessions
        for dialog_id in range(1, 13):  # 12 dialogs per session
            for utterance_id in range(1, 20):  # ~20 utterances per dialog
                if len(data) >= num_samples:
                    break
                
                # Create utterance ID in IEMOCAP format: Ses01F_impro01_M001
                # F/M for female/male, impro/script for improvised/scripted
                gender = random.choice(['F', 'M'])
                dialog_type = random.choice(['impro', 'script'])
                utterance_code = f"{gender}{utterance_id:03d}"
                
                full_id = f"Ses{session_id:02d}{gender}_{dialog_type}{dialog_id:02d}_{utterance_code}"
                
                # Generate random VAD values (original scale 1-5)
                activation = random.randint(1, 5)
                valence = random.randint(1, 5)
                dominance = random.randint(1, 5)
                
                # Normalize to [0, 1] range
                activation_norm = (activation - 1) / 4
                valence_norm = (valence - 1) / 4
                dominance_norm = (dominance - 1) / 4
                
                data.append({
                    'utterance_id': full_id,
                    'activation': activation,
                    'valence': valence,
                    'dominance': dominance,
                    'activation_norm': activation_norm,
                    'valence_norm': valence_norm,
                    'dominance_norm': dominance_norm,
                    'source_file': f"Ses{session_id:02d}{gender}_atr.txt"
                })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV if output path is provided
    if output_path:
        df.to_csv(output_path, index=False)
    
    return df

def map_vad_to_emotion(vad_df, method='quadrant'):
    """
    Map VAD values to emotion categories.
    
    Args:
        vad_df: DataFrame with VAD values
        method: Method to use for mapping ('quadrant', 'custom', 'plutchik', or 'ekman')
        
    Returns:
        DataFrame with emotion labels added
    """
    df = vad_df.copy()
    
    if method == 'quadrant':
        # Simple quadrant-based mapping using valence and activation
        # High valence, high arousal -> Happy/Excited
        # High valence, low arousal -> Calm/Content
        # Low valence, high arousal -> Angry/Frustrated
        # Low valence, low arousal -> Sad/Bored
        
        # Define midpoint (0.5 for normalized values)
        midpoint = 0.5
        
        def assign_emotion(row):
            v = row['valence_norm']
            a = row['activation_norm']
            
            if v >= midpoint and a >= midpoint:
                return 'happy'
            elif v >= midpoint and a < midpoint:
                return 'calm'
            elif v < midpoint and a >= midpoint:
                return 'angry'
            else:
                return 'sad'
        
        df['emotion'] = df.apply(assign_emotion, axis=1)
    
    elif method == 'custom':
        # More nuanced mapping based on all three dimensions
        
        def assign_emotion_custom(row):
            v = row['valence_norm']
            a = row['activation_norm']
            d = row['dominance_norm']
            
            # These thresholds can be adjusted based on literature
            if v >= 0.7 and a >= 0.7:
                return 'excited'
            elif v >= 0.7 and a < 0.3:
                return 'content'
            elif v < 0.3 and a >= 0.7 and d >= 0.7:
                return 'angry'
            elif v < 0.3 and a >= 0.7 and d < 0.3:
                return 'afraid'
            elif v < 0.3 and a < 0.3:
                return 'sad'
            elif v >= 0.5 and a >= 0.4 and a <= 0.6:
                return 'happy'
            else:
                return 'neutral'
        
        df['emotion'] = df.apply(assign_emotion_custom, axis=1)
    
    elif method == 'plutchik':
        # Plutchik's wheel of emotions
        def assign_emotion_plutchik(row):
            v = row['valence_norm']
            a = row['activation_norm']
            d = row['dominance_norm']
            
            if v >= 0.7 and a >= 0.7:
                return 'joy'
            elif v >= 0.7 and a < 0.3:
                return 'trust'
            elif v >= 0.7 and a >= 0.3 and a < 0.7 and d >= 0.7:
                return 'anticipation'
            elif v < 0.3 and a >= 0.7 and d >= 0.7:
                return 'anger'
            elif v < 0.3 and a >= 0.7 and d < 0.3:
                return 'fear'
            elif v < 0.3 and a < 0.3 and d < 0.3:
                return 'sadness'
            elif v < 0.3 and a >= 0.3 and a < 0.7 and d < 0.3:
                return 'disgust'
            elif v >= 0.3 and v < 0.7 and a < 0.3:
                return 'surprise'
            else:
                return 'neutral'
        
        df['emotion'] = df.apply(assign_emotion_plutchik, axis=1)
    
    elif method == 'ekman':
        # Ekman's six basic emotions
        def assign_emotion_ekman(row):
            v = row['valence_norm']
            a = row['activation_norm']
            d = row['dominance_norm']
            
            if v >= 0.7 and a >= 0.5:
                return 'happiness'
            elif v < 0.3 and a >= 0.7 and d >= 0.6:
                return 'anger'
            elif v < 0.3 and a >= 0.7 and d < 0.4:
                return 'fear'
            elif v < 0.3 and a < 0.4:
                return 'sadness'
            elif v < 0.4 and a >= 0.4 and a < 0.7:
                return 'disgust'
            elif v >= 0.4 and v < 0.7 and a >= 0.6:
                return 'surprise'
            else:
                return 'neutral'
        
        df['emotion'] = df.apply(assign_emotion_ekman, axis=1)
    
    return df

def main():
    # Set paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(data_dir, 'data', 'raw')
    processed_dir = os.path.join(data_dir, 'data', 'processed')
    
    # Create output directories if they don't exist
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    print("Generating synthetic VAD annotations...")
    
    # Create synthetic VAD annotations
    vad_output_path = os.path.join(processed_dir, 'vad_annotations.csv')
    vad_df = create_synthetic_vad_annotations(num_samples=1144, output_path=vad_output_path)
    print(f"Saved synthetic VAD annotations to {vad_output_path}")
    
    # Map VAD to emotions using different methods
    for method in ['quadrant', 'custom', 'plutchik', 'ekman']:
        emotion_df = map_vad_to_emotion(vad_df, method=method)
        output_path = os.path.join(processed_dir, f'emotion_{method}.csv')
        emotion_df.to_csv(output_path, index=False)
        print(f"Saved {method}-based emotion mapping to {output_path}")
    
    # Print some statistics
    print("\nDataset statistics:")
    print(f"Total number of utterances: {len(vad_df)}")
    
    # Distribution of VAD values
    print("\nVAD value distributions (original scale 1-5):")
    for dim in ['activation', 'valence', 'dominance']:
        print(f"{dim.capitalize()} distribution:")
        print(vad_df[dim].value_counts().sort_index())
    
    # Distribution of emotions for each method
    for method in ['quadrant', 'custom', 'plutchik', 'ekman']:
        emotion_df = pd.read_csv(os.path.join(processed_dir, f'emotion_{method}.csv'))
        print(f"\nEmotion distribution ({method} method):")
        print(emotion_df['emotion'].value_counts())
    
    # Generate text data
    print("\nGenerating synthetic text data...")
    from text_generator import generate_dataset
    
    text_output_path = os.path.join(processed_dir, 'iemocap_text_data.csv')
    text_df = generate_dataset(vad_df, text_output_path)
    print(f"Generated text data saved to {text_output_path}")
    
    # Print some examples
    print("\nExample utterances with generated text:")
    for i in range(min(5, len(text_df))):
        utterance_id = text_df.iloc[i]['utterance_id']
        valence = text_df.iloc[i]['valence_norm']
        arousal = text_df.iloc[i]['activation_norm']
        dominance = text_df.iloc[i]['dominance_norm']
        text = text_df.iloc[i]['text']
        
        print(f"Utterance: {utterance_id}")
        print(f"VAD: V={valence:.2f}, A={arousal:.2f}, D={dominance:.2f}")
        print(f"Text: {text}")
        print()

if __name__ == "__main__":
    main()
