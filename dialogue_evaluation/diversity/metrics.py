import random
from nltk.tokenize import word_tokenize
import nltk
from collections import defaultdict
import numpy as np
from self_bleu import SelfBleuOriginal

def _truncate_response_set(response_set, max_token_length):
    truncated_response_set = []
    token_count = 0
    for line in response_set:
        words = line.split()
        if token_count + len(words) <= max_token_length:
            truncated_response_set.append(line)
            token_count += len(words)
        else:
            remaining_space = max_token_length - token_count
            words_to_append = words[:remaining_space]
            truncated_response_set.append(' '.join(words_to_append))
            token_count += len(words_to_append)
            break
    
    # Check if total token count is less than max_token_length
    if token_count < max_token_length:
        # ANSI escape code for yellow, then reset to default
        print("\033[93mWarning: The response did not reach the maximum token limit.\033[0m")

    return truncated_response_set


# IMPORTANT:
# This uses word-level tokenization. This is because:
#   1. Tevet's implementation uses word-level tokenization.
#   2. The original paper (Li et al., 2016) says "distinct-1 and distinct-2 are respectively the number of distinct uni-gram and bi-grams divided by total number of generated words."
# However, Bao et al. (2019) uses character-level n-grams, which represents there is no agreement on the tokenization method for Distinct-n metric.

class DistinctNgrams:
    """Class to calculate Distinct-n metric as per Tevet's implementation."""

    def __init__(self, n=3, tokenizer_name='split', use_lower_case=True):
        self.n = n
        self.tokenizer_name = tokenizer_name
        self.use_lower_case = use_lower_case

    def _tokenize(self, line):
        """Tokenizes the input line based on the specified tokenizer_name.

        Args:
        line (str): The input line to _tokenize.

        Returns:
        list: A list of tokens from the input line.
        """
        if self.use_lower_case:
            line = line.lower() # this is because M2M is lowercased dataset, thus for fair comparison we need to lowercase the other datasets as well.
        if self.tokenizer_name == 'split':
            return [e for e in line.replace('.','').replace('\n','').split(' ') if e != ''] # The same as Tevet's implementation
        elif self.tokenizer_name == 'nltk':
            return word_tokenize(line)
        else:
            raise ValueError("Invalid tokenizer_name. Valid options are 'split' and 'nltk'.")

    def lines_to_ngrams(self, lines):
        ngram_lists = []
        for line in lines:
            words = self._tokenize(line)
            ngrams = [tuple(words[i:i+self.n]) for i in range(len(words)-self.n+1)]
            ngram_lists.append(ngrams)
        return ngram_lists

    def _normalized_unique_ngrams(self, ngram_lists):
        ngrams = [item for sublist in ngram_lists for item in sublist]  # flatten
        return len(set(ngrams)) / len(ngrams) if len(ngrams) > 0 else 0.

    def calculate_distinct_n(self, response_set):
        return self._normalized_unique_ngrams(self.lines_to_ngrams(response_set))

    def calculate_truncated_distinct_n(self, response_set, max_token_length, iterations, return_avg, seed=42):
        distinct_n_values = []
        random.seed(seed)
        for _ in range(iterations):
            shuffled_response_set = random.sample(response_set, len(response_set))
            truncated_response_set = _truncate_response_set(shuffled_response_set, max_token_length)
            distinct_n = self.calculate_distinct_n(truncated_response_set)
            distinct_n_values.append(distinct_n)

        if return_avg:
            return sum(distinct_n_values) / iterations
        else:
            return distinct_n_values

# Entropy-n (Ent-n)
# - Ent-n is proposed in "Generating Informative and Diverse Conversational Responses via Adversarial Information Maximization"
# - The implementation is from https://github.com/dreasysnail/converse_GAN/blob/ff2388274f985b3693bd38f0623b3626593676d0/utils.py#L433
# - div_score is distinct-n; confirmed that div_score is the same as `DistinctNgrams` if we use the same tokenization method.

class EntropyNgrams:

    def __init__(self, n=4, use_lower_case=True):
        self.n = n
        self.use_lower_case = use_lower_case

    def _preprocess(self, s):
        if self.use_lower_case:
            s = s.lower()
        return s

    def _cal_entropy(self, generated):
        """Original implementation, thus wording is different from DistinctNgrams (e.g., `generated` -> `response_set`).
        """
        generated = [self._preprocess(g) for g in generated]
        etp_score = [0.0,0.0,0.0,0.0]
        # div_score = [0.0,0.0,0.0,0.0]
        counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
        for gg in generated:
            g = gg.rstrip('2').split()
            for n in range(4):
                for idx in range(len(g)-n):
                    ngram = ' '.join(g[idx:idx+n+1])
                    counter[n][ngram] += 1
        for n in range(4):
            total = sum(counter[n].values()) +1e-10
            for v in counter[n].values():
                etp_score[n] += - (v+0.0) /total * (np.log(v+0.0) - np.log(total))
            # div_score[n] = (len(counter[n].values())+0.0) /total
        return etp_score #, div_score 
    
    def calculate_entropy_n(self, response_set):
        return self._cal_entropy(response_set)[self.n-1]

    def calculate_truncated_entropy_n(self, response_set, max_token_length, iterations, return_avg, seed=42):
        entropy_n_values = []
        random.seed(seed)
        for _ in range(iterations):
            shuffled_response_set = random.sample(response_set, len(response_set))
            truncated_response_set = _truncate_response_set(shuffled_response_set, max_token_length)
            entropy_n = self.calculate_entropy_n(truncated_response_set)
            entropy_n_values.append(entropy_n)
        
        if return_avg:
            return sum(entropy_n_values) / iterations
        else:
            return entropy_n_values


class SelfBLEU:
    
    def __init__(self, n=4, use_lower_case=True):
        """
        Args:
          - n: calculates UP TO n-gram and output geometric mean)
        """
        self.self_bleu_metric = SelfBleuOriginal(gram=n)
        self.use_lower_case = use_lower_case

    def _preprocess(self, s):
        if self.use_lower_case:
            s = s.lower()
        return s
    
    def calculate_self_bleu(self, response_set):
        """This should not be used independently because original SelfBLEU code has sentence cut-off (Truncation).
        
        """
        response_set = [self._preprocess(g) for g in response_set]
        
        if len(response_set) == 1:
            return 0.0
        score = self.self_bleu_metric.get_score(response_set)

        return score
    
    def calculate_truncated_self_bleu(self, response_set, max_token_length, iterations, return_avg, seed=42):
        self_bleu_values = []
        random.seed(seed)
        for _ in range(iterations):
            shuffled_response_set = random.sample(response_set, len(response_set))
            truncated_response_set = _truncate_response_set(shuffled_response_set, max_token_length)
            self_bleu = self.calculate_self_bleu(truncated_response_set)
            self_bleu_values.append(self_bleu)
        
        if return_avg:
            return sum(self_bleu_values) / iterations
        else:
            return self_bleu_values