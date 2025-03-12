#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text data extraction and generation for the IEMOCAP dataset.
This script extracts utterance IDs from the VAD annotation files and generates
synthetic text data based on the VAD values.
"""

import os
import re
import csv
import random
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Define emotion-related word lists based on VAD dimensions
VALENCE_WORDS = {
    'high': ['happy', 'joyful', 'pleased', 'delighted', 'excited', 'glad', 'satisfied', 'cheerful', 'content', 'thrilled'],
    'medium': ['okay', 'fine', 'neutral', 'moderate', 'average', 'fair', 'decent', 'reasonable', 'acceptable', 'standard'],
    'low': ['sad', 'unhappy', 'disappointed', 'depressed', 'miserable', 'gloomy', 'upset', 'distressed', 'sorrowful', 'down']
}

AROUSAL_WORDS = {
    'high': ['excited', 'energetic', 'alert', 'stimulated', 'aroused', 'active', 'intense', 'agitated', 'frenzied', 'hyper'],
    'medium': ['awake', 'attentive', 'engaged', 'interested', 'involved', 'focused', 'aware', 'present', 'mindful', 'conscious'],
    'low': ['calm', 'relaxed', 'peaceful', 'tranquil', 'serene', 'quiet', 'still', 'placid', 'sleepy', 'drowsy']
}

DOMINANCE_WORDS = {
    'high': ['powerful', 'dominant', 'controlling', 'influential', 'commanding', 'authoritative', 'strong', 'confident', 'assertive', 'bold'],
    'medium': ['capable', 'competent', 'adequate', 'sufficient', 'balanced', 'moderate', 'fair', 'reasonable', 'standard', 'normal'],
    'low': ['submissive', 'weak', 'helpless', 'vulnerable', 'powerless', 'dependent', 'inferior', 'intimidated', 'insecure', 'uncertain']
}

# Common sentence templates for generating synthetic text
SENTENCE_TEMPLATES = [
    "I feel {valence} and {arousal} right now, {dominance} to handle this situation.",
    "This is {valence} news, making me feel {arousal} and {dominance}.",
    "I'm {valence} about what happened, {arousal} by the events, and feeling {dominance}.",
    "The situation makes me {valence}, {arousal}, and {dominance}.",
    "I'm experiencing a {valence} emotion, feeling {arousal} and {dominance}.",
    "That's {valence}! I'm {arousal} and {dominance} about it.",
    "How {valence} and {arousal} this is, making me feel {dominance}.",
    "This {valence} event has me feeling {arousal} and {dominance}.",
    "I can't believe how {valence} this is, I'm {arousal} and {dominance}.",
    "What a {valence} day, I feel {arousal} and {dominance}."
]

# Question templates for generating synthetic text
QUESTION_TEMPLATES = [
    "Why do I feel so {valence} and {arousal} about this?",
    "How can something so {valence} make me feel this {arousal} and {dominance}?",
    "Do you feel {valence} and {arousal} about this too?",
    "Isn't it {valence} how {arousal} and {dominance} this situation is?",
    "Can you believe how {valence} and {arousal} this is?"
]

def extract_utterance_ids(atr_file_path):
    """
    Extract utterance IDs from an attribute file.
    
    Args:
        atr_file_path: Path to the attribute file
        
    Returns:
        List of utterance IDs
    """
    utterance_ids = []
    
    with open(atr_file_path, 'r') as f:
        for line in f:
            # Extract utterance ID (e.g., Ses01F_script01_1_F000)
            match = re.match(r'^(Ses\d+[FM]_\w+_[FM]\d+)', line)
            if match:
                utterance_ids.append(match.group(1))
    
    return utterance_ids

def extract_vad_values(atr_file_path):
    """
    Extract VAD values from an attribute file.
    
    Args:
        atr_file_path: Path to the attribute file
        
    Returns:
        DataFrame with utterance IDs and VAD values
    """
    data = []
    
    with open(atr_file_path, 'r') as f:
        for line in f:
            # Extract utterance ID and VAD values
            match = re.match(r'^(Ses\d+[FM]_\w+_[FM]\d+)\s+:act\s+(\d+);\s+:val\s+(\d+);\s+:dom\s+(\d+);', line)
            if match:
                utterance_id = match.group(1)
                arousal = int(match.group(2))
                valence = int(match.group(3))
                dominance = int(match.group(4))
                
                data.append({
                    'utterance_id': utterance_id,
                    'valence': valence,
                    'arousal': arousal,
                    'dominance': dominance
                })
    
    return pd.DataFrame(data)

def normalize_vad_values(df):
    """
    Normalize VAD values to [0, 1] range.
    
    Args:
        df: DataFrame with VAD values
        
    Returns:
        DataFrame with normalized VAD values
    """
    # Assuming VAD values are in range [1, 5]
    df['valence_norm'] = (df['valence'] - 1) / 4
    df['arousal_norm'] = (df['arousal'] - 1) / 4
    df['dominance_norm'] = (df['dominance'] - 1) / 4
    
    return df

def get_vad_category(value):
    """
    Categorize a VAD value as high, medium, or low.
    
    Args:
        value: Normalized VAD value [0, 1]
        
    Returns:
        Category string ('high', 'medium', or 'low')
    """
    if value >= 0.7:
        return 'high'
    elif value <= 0.3:
        return 'low'
    else:
        return 'medium'

def generate_text_from_vad(valence, arousal, dominance):
    """
    Generate synthetic text based on VAD values.
    
    Args:
        valence: Normalized valence value [0, 1]
        arousal: Normalized arousal value [0, 1]
        dominance: Normalized dominance value [0, 1]
        
    Returns:
        Generated text
    """
    # Categorize VAD values
    valence_cat = get_vad_category(valence)
    arousal_cat = get_vad_category(arousal)
    dominance_cat = get_vad_category(dominance)
    
    # Select random words for each VAD dimension
    valence_word = random.choice(VALENCE_WORDS[valence_cat])
    arousal_word = random.choice(AROUSAL_WORDS[arousal_cat])
    dominance_word = random.choice(DOMINANCE_WORDS[dominance_cat])
    
    # Select a random template (80% statements, 20% questions)
    if random.random() < 0.8:
        template = random.choice(SENTENCE_TEMPLATES)
    else:
        template = random.choice(QUESTION_TEMPLATES)
    
    # Fill in the template
    text = template.format(
        valence=valence_word,
        arousal=arousal_word,
        dominance=dominance_word
    )
    
    return text

def expand_text_with_synonyms(text, expansion_factor=0.3):
    """
    Expand text by replacing some words with their synonyms.
    
    Args:
        text: Input text
        expansion_factor: Probability of replacing a word with a synonym
        
    Returns:
        Expanded text
    """
    words = word_tokenize(text)
    result = []
    
    for word in words:
        # Only replace content words with some probability
        if len(word) > 3 and random.random() < expansion_factor:
            # Get synonyms
            synonyms = []
            for syn in wn.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word and synonym not in synonyms:
                        synonyms.append(synonym)
            
            # Replace with a random synonym if available
            if synonyms:
                result.append(random.choice(synonyms))
            else:
                result.append(word)
        else:
            result.append(word)
    
    return ' '.join(result)

def create_text_dataset(vad_df):
    """
    Create a dataset with synthetic text based on VAD values.
    
    Args:
        vad_df: DataFrame with utterance IDs and VAD values
        
    Returns:
        DataFrame with utterance IDs, VAD values, and synthetic text
    """
    df = vad_df.copy()
    
    # Generate text for each utterance
    texts = []
    for _, row in df.iterrows():
        text = generate_text_from_vad(
            row['valence_norm'],
            row['arousal_norm'],
            row['dominance_norm']
        )
        
        # Expand text with synonyms for variety
        if random.random() < 0.5:
            text = expand_text_with_synonyms(text)
            
        texts.append(text)
    
    df['text'] = texts
    
    return df

def process_all_attribute_files(attribute_dir, output_file):
    """
    Process all attribute files in a directory and create a text dataset.
    
    Args:
        attribute_dir: Directory containing attribute files
        output_file: Path to save the output CSV file
    """
    all_data = []
    
    # Find all attribute files
    for file in os.listdir(attribute_dir):
        if file.endswith('_atr.txt'):
            file_path = os.path.join(attribute_dir, file)
            
            # Extract VAD values
            vad_df = extract_vad_values(file_path)
            
            # Normalize VAD values
            vad_df = normalize_vad_values(vad_df)
            
            # Add to all data
            all_data.append(vad_df)
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Create text dataset
        text_df = create_text_dataset(combined_df)
        
        # Save to CSV
        text_df.to_csv(output_file, index=False)
        
        print(f"Created text dataset with {len(text_df)} utterances")
        print(f"Saved to {output_file}")
        
        return text_df
    else:
        print("No attribute files found")
        return None

def main():
    """
    Main function to create a text dataset from IEMOCAP attribute files.
    """
    # Set paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    attribute_dir = os.path.join(base_dir, 'processed', 'EmoEvaluation', 'Attribute')
    output_dir = os.path.join(base_dir, 'processed')
    output_file = os.path.join(output_dir, 'text_dataset.csv')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating text dataset from IEMOCAP attribute files...")
    
    # Process all attribute files
    text_df = process_all_attribute_files(attribute_dir, output_file)
    
    if text_df is not None:
        # Print some examples
        print("\nExample utterances with synthetic text:")
        for _, row in text_df.sample(5).iterrows():
            print(f"Utterance ID: {row['utterance_id']}")
            print(f"VAD: V={row['valence']}, A={row['arousal']}, D={row['dominance']}")
            print(f"Text: {row['text']}")
            print()

if __name__ == "__main__":
    main()
