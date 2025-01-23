import json
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statistics
import time

def analyze_asm_data():
    start = time.time()
    
    # Initialize statistics
    length_counter = Counter()
    total_length = 0
    total_words = 0
    
    # Process each Assamese dataset file
    files = [
        'data/raw/asm_valid.json'
    ]
    
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                pair = json.loads(line.strip())
                # Analyze source word length
                src_len = len(pair['native word'])
                length_counter[src_len] += 1
                total_length += src_len
                total_words += 1
                
                # Analyze target word length
                tgt_len = len(pair['english word'])
                length_counter[tgt_len] += 1
                total_length += tgt_len
                total_words += 1
    
    # Calculate statistics
    length_dist = OrderedDict(sorted(length_counter.items()))
    avg_length = total_length / total_words
    
    # Print results
    print(f"Total words analyzed: {total_words}")
    print(f"Average word length: {avg_length:.2f} characters")
    print("\nLength distribution:")
    for length, count in length_dist.items():
        print(f"{length} chars: {count} words")
    
    # Generate plot
    plt.figure(figsize=(12, 6))
    plt.bar(length_dist.keys(), length_dist.values())
    plt.title('Assamese Word Length Distribution')
    plt.xlabel('Word Length (characters)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('data/corpus_preprocessing/Analysis/asm_length_dist.png')
    
    end = time.time()
    print(f'\nRuntime: {end - start:.2f} seconds')

if __name__ == '__main__':
    analyze_asm_data()
