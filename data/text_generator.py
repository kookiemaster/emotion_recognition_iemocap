#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text generator for IEMOCAP dataset.
This script generates synthetic text data based on VAD (Valence-Arousal-Dominance) values.
"""

import os
import pandas as pd
import numpy as np
import random
from pathlib import Path
import re

# Define emotion-related word lists for different VAD levels
HIGH_VALENCE_WORDS = [
    "happy", "joyful", "delighted", "excited", "thrilled", "pleased", "content", "satisfied",
    "cheerful", "ecstatic", "elated", "glad", "grateful", "wonderful", "fantastic", "amazing",
    "excellent", "terrific", "great", "good", "positive", "lovely", "beautiful", "awesome"
]

LOW_VALENCE_WORDS = [
    "sad", "unhappy", "miserable", "depressed", "gloomy", "disappointed", "upset", "distressed",
    "sorrowful", "heartbroken", "dejected", "downcast", "dismal", "terrible", "awful", "horrible",
    "dreadful", "bad", "negative", "unpleasant", "painful", "tragic", "unfortunate", "disheartening"
]

HIGH_AROUSAL_WORDS = [
    "energetic", "active", "alert", "lively", "vigorous", "dynamic", "stimulated", "aroused",
    "excited", "agitated", "frenzied", "hyper", "intense", "passionate", "wild", "frantic",
    "enthusiastic", "eager", "animated", "spirited", "thrilled", "exhilarated", "electrified", "charged"
]

LOW_AROUSAL_WORDS = [
    "calm", "relaxed", "peaceful", "tranquil", "serene", "quiet", "still", "placid",
    "gentle", "mellow", "subdued", "passive", "inactive", "sleepy", "tired", "drowsy",
    "lethargic", "sluggish", "dull", "bored", "uninterested", "apathetic", "indifferent", "detached"
]

HIGH_DOMINANCE_WORDS = [
    "powerful", "strong", "dominant", "controlling", "influential", "commanding", "authoritative", "confident",
    "assertive", "decisive", "determined", "forceful", "bold", "brave", "courageous", "fearless",
    "mighty", "potent", "capable", "competent", "effective", "efficient", "masterful", "skilled"
]

LOW_DOMINANCE_WORDS = [
    "weak", "powerless", "submissive", "vulnerable", "helpless", "dependent", "subordinate", "inferior",
    "insecure", "uncertain", "hesitant", "timid", "shy", "afraid", "fearful", "anxious",
    "doubtful", "unsure", "incapable", "incompetent", "ineffective", "inefficient", "unskilled", "inadequate"
]

# Define sentence templates for different emotional contexts
SENTENCE_TEMPLATES = [
    "I feel {emotion} today.",
    "This situation makes me feel {emotion}.",
    "I am {emotion} about what happened.",
    "The news made me {emotion}.",
    "I'm {emotion} because of the recent events.",
    "My day has been {emotion}.",
    "That experience was {emotion}.",
    "I've been feeling {emotion} lately.",
    "The conversation left me feeling {emotion}.",
    "This outcome is {emotion} for me.",
    "I can't help but feel {emotion} right now.",
    "Everything seems {emotion} at the moment.",
    "The atmosphere feels {emotion}.",
    "I'm in a {emotion} mood.",
    "The results made everyone {emotion}.",
    "The situation is {emotion} for all of us.",
    "We were all {emotion} after hearing the news.",
    "The team is {emotion} about the project.",
    "The meeting was {emotion} and productive.",
    "The feedback was {emotion} and helpful."
]

# Define more complex sentence templates for variety
COMPLEX_TEMPLATES = [
    "I've been feeling {emotion1} all day, but now I'm starting to feel {emotion2}.",
    "Although the situation seems {emotion1}, I believe it will become {emotion2} soon.",
    "The project started out {emotion1}, but ended up being quite {emotion2}.",
    "Despite feeling {emotion1} about the challenge, I approached it with a {emotion2} attitude.",
    "The movie had a {emotion1} beginning but a {emotion2} ending.",
    "What started as a {emotion1} conversation quickly turned {emotion2}.",
    "I was {emotion1} at first, but after thinking about it, I feel {emotion2}.",
    "The team was {emotion1} about the deadline, but {emotion2} about the quality of work.",
    "The feedback was mostly {emotion1}, with some {emotion2} comments mixed in.",
    "While the presentation was {emotion1}, the audience response was {emotion2}."
]

def get_emotion_word(valence, arousal, dominance):
    """
    Get an emotion-related word based on VAD values.
    
    Args:
        valence: Normalized valence value [0, 1]
        arousal: Normalized arousal value [0, 1]
        dominance: Normalized dominance value [0, 1]
        
    Returns:
        An emotion-related word
    """
    # Determine which word lists to use based on VAD values
    valence_words = HIGH_VALENCE_WORDS if valence >= 0.5 else LOW_VALENCE_WORDS
    arousal_words = HIGH_AROUSAL_WORDS if arousal >= 0.5 else LOW_AROUSAL_WORDS
    dominance_words = HIGH_DOMINANCE_WORDS if dominance >= 0.5 else LOW_DOMINANCE_WORDS
    
    # Weighted selection based on VAD values
    v_weight = valence if valence >= 0.5 else (1 - valence)
    a_weight = arousal if arousal >= 0.5 else (1 - arousal)
    d_weight = dominance if dominance >= 0.5 else (1 - dominance)
    
    weights = [v_weight, a_weight, d_weight]
    weights = [w/sum(weights) for w in weights]
    
    word_lists = [valence_words, arousal_words, dominance_words]
    selected_list = random.choices(word_lists, weights=weights, k=1)[0]
    
    return random.choice(selected_list)

def generate_text_for_vad(valence, arousal, dominance):
    """
    Generate synthetic text based on VAD values.
    
    Args:
        valence: Normalized valence value [0, 1]
        arousal: Normalized arousal value [0, 1]
        dominance: Normalized dominance value [0, 1]
        
    Returns:
        Generated text
    """
    # Decide whether to use a simple or complex template
    if random.random() < 0.7:  # 70% simple, 30% complex
        template = random.choice(SENTENCE_TEMPLATES)
        emotion_word = get_emotion_word(valence, arousal, dominance)
        text = template.format(emotion=emotion_word)
    else:
        template = random.choice(COMPLEX_TEMPLATES)
        emotion_word1 = get_emotion_word(valence, arousal, dominance)
        
        # For the second emotion, slightly vary the VAD values
        var_range = 0.2
        v2 = max(0, min(1, valence + random.uniform(-var_range, var_range)))
        a2 = max(0, min(1, arousal + random.uniform(-var_range, var_range)))
        d2 = max(0, min(1, dominance + random.uniform(-var_range, var_range)))
        
        emotion_word2 = get_emotion_word(v2, a2, d2)
        text = template.format(emotion1=emotion_word1, emotion2=emotion_word2)
    
    return text

def generate_dataset(vad_df, output_path):
    """
    Generate synthetic text data for each utterance in the VAD dataframe.
    
    Args:
        vad_df: DataFrame with VAD values
        output_path: Path to save the generated dataset
        
    Returns:
        DataFrame with utterance IDs, VAD values, and generated text
    """
    # Create a copy of the dataframe
    df = vad_df.copy()
    
    # Generate text for each utterance
    texts = []
    for _, row in df.iterrows():
        valence = row['valence_norm']
        arousal = row['activation_norm']
        dominance = row['dominance_norm']
        
        text = generate_text_for_vad(valence, arousal, dominance)
        texts.append(text)
    
    # Add text to dataframe
    df['text'] = texts
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return df

def main():
    """
    Main function to generate synthetic text data for IEMOCAP dataset.
    """
    # Set paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(data_dir, 'data', 'processed')
    output_dir = os.path.join(data_dir, 'data', 'processed')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load VAD annotations
    vad_path = os.path.join(processed_dir, 'vad_annotations.csv')
    
    if not os.path.exists(vad_path):
        print(f"VAD annotations file not found at {vad_path}")
        print("Please run the preprocessing script first.")
        return
    
    vad_df = pd.read_csv(vad_path)
    
    print(f"Generating synthetic text data for {len(vad_df)} utterances...")
    
    # Generate text data
    output_path = os.path.join(output_dir, 'iemocap_text_data.csv')
    df = generate_dataset(vad_df, output_path)
    
    print(f"Generated text data saved to {output_path}")
    print(f"Total utterances with text: {len(df)}")
    
    # Print some examples
    print("\nExample utterances with generated text:")
    for i in range(min(5, len(df))):
        utterance_id = df.iloc[i]['utterance_id']
        valence = df.iloc[i]['valence_norm']
        arousal = df.iloc[i]['activation_norm']
        dominance = df.iloc[i]['dominance_norm']
        text = df.iloc[i]['text']
        
        print(f"Utterance: {utterance_id}")
        print(f"VAD: V={valence:.2f}, A={arousal:.2f}, D={dominance:.2f}")
        print(f"Text: {text}")
        print()

if __name__ == "__main__":
    main()
