import json
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statistics
import time
import os
from typing import Dict, List, Set

def get_language_files(lang_code: str) -> List[str]:
    """Get all dataset files for a given language code."""
    return [
        f'data/raw/{lang_code}_{split}.json' 
        for split in ['train', 'test', 'valid']
        if os.path.exists(f'data/raw/{lang_code}_{split}.json')
    ]

def analyze_character_sets(pairs: List[Dict]) -> Dict:
    """Analyze character sets in native and English words."""
    native_chars = set()
    english_chars = set()
    
    for pair in pairs:
        native_chars.update(pair['native word'])
        english_chars.update(pair['english word'])
    
    return {
        'native_charset': sorted(list(native_chars)),
        'english_charset': sorted(list(english_chars)),
        'native_charset_size': len(native_chars),
        'english_charset_size': len(english_chars)
    }

def analyze_source_distribution(pairs: List[Dict]) -> Dict:
    """Analyze distribution of data sources."""
    source_counter = Counter()
    for pair in pairs:
        source_counter[pair['source']] += 1
    return dict(source_counter)

def analyze_score_distribution(pairs: List[Dict]) -> Dict:
    """Analyze distribution of confidence scores."""
    # Filter out None/null scores
    scores = [pair['score'] for pair in pairs if pair.get('score') is not None]
    
    if not scores:
        return {
            'min_score': None,
            'max_score': None,
            'mean_score': None,
            'median_score': None,
            'total_scores': 0,
            'missing_scores': len(pairs)
        }
    
    return {
        'min_score': min(scores),
        'max_score': max(scores),
        'mean_score': statistics.mean(scores),
        'median_score': statistics.median(scores),
        'total_scores': len(scores),
        'missing_scores': len(pairs) - len(scores)
    }

def analyze_length_distribution(pairs: List[Dict]) -> Dict:
    """Analyze word length distributions."""
    native_lengths = Counter()
    english_lengths = Counter()
    total_native_len = 0
    total_english_len = 0
    missing_fields = {'native': 0, 'english': 0}
    
    for pair in pairs:
        native_word = pair.get('native word')
        english_word = pair.get('english word')
        
        if native_word:
            native_len = len(native_word)
            native_lengths[native_len] += 1
            total_native_len += native_len
        else:
            missing_fields['native'] += 1
            
        if english_word:
            english_len = len(english_word)
            english_lengths[english_len] += 1
            total_english_len += english_len
        else:
            missing_fields['english'] += 1
    
    valid_pairs = len(pairs) - max(missing_fields['native'], missing_fields['english'])
    
    return {
        'native_length_dist': dict(OrderedDict(sorted(native_lengths.items()))),
        'english_length_dist': dict(OrderedDict(sorted(english_lengths.items()))),
        'avg_native_length': total_native_len / (len(pairs) - missing_fields['native']) if len(pairs) > missing_fields['native'] else 0,
        'avg_english_length': total_english_len / (len(pairs) - missing_fields['english']) if len(pairs) > missing_fields['english'] else 0,
        'total_words': valid_pairs,
        'total_pairs': len(pairs),
        'missing_fields': missing_fields
    }

def analyze_language_data(lang_code: str):
    """Analyze all data files for a given language."""
    start = time.time()
    print(f"\nAnalyzing {lang_code.upper()} dataset...")
    
    # Get all files for this language
    files = get_language_files(lang_code)
    if not files:
        print(f"No files found for language code: {lang_code}")
        return
    
    # Process each split
    split_stats = {}
    for file in files:
        split = file.split('_')[-1].split('.')[0]  # Extract split name (train/test/valid)
        pairs = []
        
        print(f"\nProcessing {split} split...")
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                pairs.append(json.loads(line.strip()))
        
        # Collect statistics for this split
        split_stats[split] = {
            'length_stats': analyze_length_distribution(pairs),
            'char_stats': analyze_character_sets(pairs),
            'source_stats': analyze_source_distribution(pairs),
            'score_stats': analyze_score_distribution(pairs)
        }
        
        # Print summary for this split
        print(f"\n{split.capitalize()} Split Statistics:")
        length_stats = split_stats[split]['length_stats']
        score_stats = split_stats[split]['score_stats']
        
        print(f"\n{split.capitalize()} Split Statistics:")
        print(f"Total pairs analyzed: {length_stats['total_pairs']}")
        print(f"Valid word pairs: {length_stats['total_words']}")
        if length_stats['missing_fields']['native'] > 0 or length_stats['missing_fields']['english'] > 0:
            print(f"Missing fields - Native: {length_stats['missing_fields']['native']}, "
                  f"English: {length_stats['missing_fields']['english']}")
        
        print(f"\nAverage lengths - Native: {length_stats['avg_native_length']:.2f}, "
              f"English: {length_stats['avg_english_length']:.2f}")
        print(f"Character set sizes - Native: {split_stats[split]['char_stats']['native_charset_size']}, "
              f"English: {split_stats[split]['char_stats']['english_charset_size']}")
        
        print("\nScore Statistics:")
        if score_stats['total_scores'] > 0:
            print(f"Score range: {score_stats['min_score']:.3f} to {score_stats['max_score']:.3f}")
            print(f"Mean score: {score_stats['mean_score']:.3f}")
            print(f"Median score: {score_stats['median_score']:.3f}")
        if score_stats['missing_scores'] > 0:
            print(f"Missing scores: {score_stats['missing_scores']}")
        
        print("\nSource distribution:")
        for source, count in split_stats[split]['source_stats'].items():
            print(f"{source}: {count}")
        
        # Generate length distribution plot
        plt.figure(figsize=(12, 6))
        native_dist = split_stats[split]['length_stats']['native_length_dist']
        english_dist = split_stats[split]['length_stats']['english_length_dist']
        
        x = np.arange(max(max(native_dist.keys()), max(english_dist.keys())) + 1)
        width = 0.35
        
        plt.bar(x - width/2, [native_dist.get(i, 0) for i in x], width, label='Native')
        plt.bar(x + width/2, [english_dist.get(i, 0) for i in x], width, label='English')
        
        plt.title(f'{lang_code.upper()} {split.capitalize()} Word Length Distribution')
        plt.xlabel('Word Length (characters)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        
        # Create directory if it doesn't exist
        os.makedirs('data/corpus_preprocessing/Analysis/plots', exist_ok=True)
        plt.savefig(f'data/corpus_preprocessing/Analysis/plots/{lang_code}_{split}_length_dist.png')
        plt.close()
    
    # Save detailed statistics to JSON
    os.makedirs('data/corpus_preprocessing/Analysis/stats', exist_ok=True)
    with open(f'data/corpus_preprocessing/Analysis/stats/{lang_code}_stats.json', 'w', encoding='utf-8') as f:
        json.dump(split_stats, f, indent=2, ensure_ascii=False)
    
    end = time.time()
    print(f'\nRuntime: {end - start:.2f} seconds')

def main():
    """Process all language datasets."""
    # List of all language codes
    languages = [
        'asm', 'ben', 'brx', 'guj', 'hin', 'kan', 'kas', 'kok', 'mai', 'mal',
        'mni', 'mar', 'nep', 'ori', 'pan', 'san', 'sid', 'tam', 'tel', 'urd'
    ]
    
    for lang in languages:
        analyze_language_data(lang)

if __name__ == '__main__':
    main()
