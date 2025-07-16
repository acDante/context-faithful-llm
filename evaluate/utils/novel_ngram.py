from collections import Counter
import re

def find_ngrams(input_list, n):
    """Extract n-grams from input list."""
    if len(input_list) < n:
        return []
    return list(zip(*[input_list[i:] for i in range(n)]))

def tokenize_and_normalize(text, case=False):
    """
    Tokenizes, lowercases, and removes punctuation from a string.
    
    Args:
        text (str): The input string.
        case (bool): If True, preserves original case. Default is False (lowercase).
    
    Returns:
        list: A list of clean tokens.
    """
    # Use regex to find all word sequences, which handles punctuation better.
    # \b matches word boundaries. \w+ matches one or more word characters.
    tokens = re.findall(r'\b\w+\b', text)
    if not case:
        return [token.lower() for token in tokens]
    return tokens

def n_gram_percent(summary, text, n_gram_max):
    """
    Calculate the percentage of novel n-grams in a summary.
    Novel n-grams are those that appear in the summary but not in the original text.
    
    Args:
        summary (str): The summary text.
        text (str): The original source text.
        n_gram_max (int): Maximum n-gram size to analyze (e.g., 3 for 1, 2, and 3-grams).
    
    Returns:
        dict: A dictionary where keys are "percentage_novel_n-gram" and
              values are the novelty scores.
    """
    if not summary.strip() or not text.strip():
        return {}
    
    # Tokenize and normalize text ONCE before the loop.
    tokenized_summary = tokenize_and_normalize(summary)
    tokenized_text = tokenize_and_normalize(text)
    
    novelty_scores = {}
    
    for n in range(1, n_gram_max + 1):
        # Generate n-grams for both source and summary
        text_ngrams = find_ngrams(tokenized_text, n)
        summary_ngrams = find_ngrams(tokenized_summary, n)
        
        # If summary is too short to form n-grams of size n, we can't calculate.
        if not summary_ngrams:
            continue
            
        # Use sets to find unique n-grams
        text_ngrams_set = set(text_ngrams)
        summary_ngrams_set = set(summary_ngrams)
        
        # The denominator is the number of unique n-grams in the summary.
        # This prevents division by zero.
        num_unique_summary_ngrams = len(summary_ngrams_set)
        
        # Novel n-grams are in the summary but not in the source text.
        novel_ngrams = summary_ngrams_set.difference(text_ngrams_set)
        
        # Calculate the novelty score
        score = len(novel_ngrams) / num_unique_summary_ngrams
        novelty_scores[f"percentage_novel_{n}-gram"] = score
        
    return novelty_scores

def normalize(tokens, case=False):
    """
    Lowercases and turns tokens into distinct words.
    """
    return [str(t).lower() if not case else str(t) for t in tokens]

def n_gram_novelty(summary, text, n_gram):
    """
    Calculate n-gram statistics for summary evaluation. (Note: this implementation has some issues with punctuation)
    
    Args:
        summary (str): The summary text
        text (str): The original text
        n_gram (int): Maximum n-gram size to analyze
    
    Returns:
        dict: Dictionary containing percentage statistics
    """
    if not summary.strip() or not text.strip():
        return {}
    
    tokenized_summary = normalize(summary.split())
    tokenized_text = normalize(text.split())
    score_dict = {}
    
    for i in range(1, n_gram + 1):
        input_ngrams = find_ngrams(tokenized_text, i)
        summ_ngrams = find_ngrams(tokenized_summary, i)
        
        if not summ_ngrams:  # Skip if no n-grams can be formed
            continue
            
        input_ngrams_set = set(input_ngrams)
        summ_ngrams_set = set(summ_ngrams)
        intersect = summ_ngrams_set.intersection(input_ngrams_set)
        
        # Novel n-grams: n-grams in summary not found in original text
        novel_ngrams = summ_ngrams_set - input_ngrams_set
        score_dict[f"percentage_novel_{i}-gram"] = len(novel_ngrams) / len(summ_ngrams_set)
        
    return score_dict

if __name__ == "__main__":
    # Test case
    original_text = "The quick brown fox jumps over the lazy dog. The fox is very quick and brown."
    summary_text = "The quick fox jumps over the dog. The fox is fast."

    results = n_gram_percent(summary_text, original_text, 3)

    for metric, value in results.items():
        print(f"{metric}: {value:.3f}")