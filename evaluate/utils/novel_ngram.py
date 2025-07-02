from collections import Counter

def find_ngrams(input_list, n):
    """Extract n-grams from input list."""
    if len(input_list) < n:
        return []
    return list(zip(*[input_list[i:] for i in range(n)]))

def normalize(tokens, case=False):
    """
    Lowercases and turns tokens into distinct words.
    """
    return [str(t).lower() if not case else str(t) for t in tokens]

def n_gram_percent(summary, text, n_gram):
    """
    Calculate n-gram statistics for summary evaluation.
    
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