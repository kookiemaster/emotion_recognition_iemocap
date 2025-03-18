#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for emotion recognition on IEMOCAP dataset.
This script runs the complete pipeline:
1. Generate text data from VAD annotations
2. Convert text to VAD values using BERT
3. Classify VAD values to emotion categories
4. Evaluate the models
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse

def main():
    """
    Main function to run the complete emotion recognition pipeline.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Emotion recognition on IEMOCAP dataset')
    parser.add_argument('--skip-text-generation', action='store_true', help='Skip text data generation')
    parser.add_argument('--skip-text-to-vad', action='store_true', help='Skip text to VAD conversion')
    parser.add_argument('--skip-vad-to-emotion', action='store_true', help='Skip VAD to emotion classification')
    parser.add_argument('--skip-evaluation', action='store_true', help='Skip model evaluation')
    args = parser.parse_args()
    
    # Set paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    processed_dir = os.path.join(data_dir, 'processed')
    
    # Step 1: Generate text data from VAD annotations
    if not args.skip_text_generation:
        print("\n" + "="*50)
        print("Step 1: Generating text data from VAD annotations")
        print("="*50)
        
        # Import text data generation module
        sys.path.append(os.path.join(data_dir, 'processed'))
        from text_data import main as text_data_main
        
        # Run text data generation
        text_data_main()
    
    # Step 2: Convert text to VAD values using BERT
    if not args.skip_text_to_vad:
        print("\n" + "="*50)
        print("Step 2: Converting text to VAD values using BERT")
        print("="*50)
        
        # Import text to VAD conversion module
        sys.path.append(os.path.join(base_dir, 'vad_conversion'))
        from text_to_vad_bert import main as text_to_vad_main
        
        # Run text to VAD conversion
        text_to_vad_main()
    
    # Step 3: Classify VAD values to emotion categories
    if not args.skip_vad_to_emotion:
        print("\n" + "="*50)
        print("Step 3: Classifying VAD values to emotion categories")
        print("="*50)
        
        # Import VAD to emotion classification module
        sys.path.append(os.path.join(base_dir, 'emotion_classification'))
        from vad_to_emotion_improved import main as vad_to_emotion_main
        
        # Run VAD to emotion classification
        vad_to_emotion_main()
    
    # Step 4: Evaluate the models
    if not args.skip_evaluation:
        print("\n" + "="*50)
        print("Step 4: Evaluating the models")
        print("="*50)
        
        # Import model evaluation module
        sys.path.append(os.path.join(base_dir, 'evaluation'))
        from model_evaluation_improved import main as evaluation_main
        
        # Run model evaluation
        evaluation_main()
    
    print("\n" + "="*50)
    print("Emotion recognition pipeline completed")
    print("="*50)

if __name__ == "__main__":
    main()
