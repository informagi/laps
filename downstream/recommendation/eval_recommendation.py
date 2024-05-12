import numpy as np
from tqdm import tqdm
import re
import copy
import pandas as pd
import matplotlib.pyplot as plt

def calculate_prf(tp, fp, fn):
    assert type(tp) == int, f"tp should be int. tp: {tp}"
    assert type(fp) == int, f"fp should be int. fp: {fp}"
    assert type(fn) == int, f"fn should be int. fn: {fn}"
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    f = 2 * p * r / (p + r) if p + r > 0 else 0

    return p, r, f

def calculate_overall_prf_scores(counts_list, flag_print):
    """Calculate overall precision, recall, and f1 scores from multiple target_session_len
    """    
    prf_scores_overall = {'pred': {'p': None, 'r': None, 'f': None}, 'baseline': {'p': None, 'r': None, 'f': None}, 'gold': {'p': None, 'r': None, 'f': None}}

    counts = {
        'pred': {'tp': 0, 'fp': 0, 'fn': 0},
        'baseline': {'tp': 0, 'fp': 0, 'fn': 0},
        'gold': {'tp': 0, 'fp': 0, 'fn': 0}
    }

    for counts_dict in counts_list:
        for category in ['pred', 'baseline', 'gold']:
            for tp_fp_fn in ['tp', 'fp', 'fn']:
                for session_ind in range(len(counts_dict[category][tp_fp_fn])):
                    for data_point_ind in range(len(counts_dict[category][tp_fp_fn][session_ind])):
                        counts[category][tp_fp_fn] += counts_dict[category][tp_fp_fn][session_ind][data_point_ind]

    if flag_print: print(f"\n========== Overall PRF scores ==========")
    for category in ['pred', 'baseline', 'gold']:
        tp = counts[category]['tp']
        fp = counts[category]['fp']
        fn = counts[category]['fn']
        p, r, f = calculate_prf(tp, fp, fn)
        prf_scores_overall[category]['p'] = p
        prf_scores_overall[category]['r'] = r
        prf_scores_overall[category]['f'] = f
        if flag_print: print(f"-{category[:4]} - P: {p:.3f}, R: {r:.3f}, F1: {f:.3f}")
    if flag_print: print(f"\n========================================")

    return prf_scores_overall


class EvalRecommendation:
    def __init__(self, test_recommendation, target_session_len):
        self.url2text = self.get_url2text()
        self.scores = {
            'num_preferences_from_each_session': {'pred': [], 'baseline': [], 'gold': []}, # only focus on intersection of preferences from each session and pred/baseline/gold
            'preference_usage_pct': {'pred': [], 'baseline': [], 'gold': []}, # percentage of how many preferences are used from each session (recall-ish metric)
        }
        self.counts = {
            'pred': {'tp': [], 'fp': [], 'fn': []},
            'baseline': {'tp': [], 'fp': [], 'fn': []},
            'gold': {'tp': [], 'fp': [], 'fn': []}
        }
        self.prf_scores = {category: {'p': [], 'r': [], 'f': []} for category in ['pred', 'baseline', 'gold']}
        self.prf_scores_overall = None # {'pred': {'p': None, 'r': None, 'f': None}, 'baseline': {'p': None, 'r': None, 'f': None}, 'gold': {'p': None, 'r': None, 'f': None}}
        self.ratios = {category: [] for category in ['pred', 'baseline', 'gold']}
        self.target_session_len = target_session_len
        self.test_recommendation = copy.deepcopy(test_recommendation)
        # remove session_ind != target_session_len-1
        if target_session_len != None:
            self.test_recommendation = [rec for rec in self.test_recommendation if rec['session_ind'] == self.target_session_len-1]
        else:
            print('target_session_len is None. Not filtering out session_ind.')
        self.avg_preference_recall_by_session = {category: [] for category in ['pred', 'baseline', 'gold']}



    def strip_and_remove_unnecessary_spaces_etc(self, text: str) -> str:
        assert type(text) == str, f'text should be str. text: {text}'
        text = text.strip()
        # remove \n
        text = text.replace('\n', ' ')
        # remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # remove non-ascii characters
        text = text.encode("ascii", errors="ignore").decode()
        return text

    def get_url2text(self):
        url_df_recipe = pd.read_csv('./data/url_list_recipe.csv') # recipe urls
        url_df_movie = pd.read_csv('./data/url_list_movie.csv') # movie urls
        url_df = pd.concat([url_df_recipe, url_df_movie], ignore_index=True)
        url2text = {row['url']: row['text'] for _, row in url_df.iterrows() if (row['text'] and type(row['text']) == str)}
        url2text = {url: self.strip_and_remove_unnecessary_spaces_etc(text) for url, text in url2text.items()}
        return url2text

    def append_content_from_url(self, _gold: str) -> str:
        # assert _gold should be lowered beforehand
        assert _gold == _gold.lower(), '_gold should be lowered beforehand'

        # lower url2text keys and values
        url2text_lower = {url.lower(): text.lower() for url, text in self.url2text.items()}

        gold_urls = set(re.findall(r'(https?://\S+)', _gold))
        # _gold = _gold + ' ' + ' '.join([url2text[url] for url in gold_urls if url in url2text])
        for url in gold_urls:
            if url in url2text_lower:
                # print(f'Found url: {url}. Appending the content: {url2text[url][:100]}...')
                _gold = _gold + ' ' + url2text_lower[url]
        return _gold

    def get_flattened_preferences(self, preferences_for_each_session, session_ind, use_lower):
        preferences_each_session_flat = [None for _ in range(len(preferences_for_each_session))]
        for i in range(len(preferences_for_each_session)):
            if use_lower:
                preferences_each_session_flat[i] = set(pref.lower() for _, prefs in preferences_for_each_session[i] for pref in prefs)
            else:
                preferences_each_session_flat[i] = set(pref for _, prefs in preferences_for_each_session[i] for pref in prefs)

        considered_preferences = set(v for i in range(session_ind + 1) for v in preferences_each_session_flat[i])

        return preferences_each_session_flat, considered_preferences


    def preprocess_text_longest_match(self, text: str, considered_preferences: set) -> set:
        # text and considered_preferences should be lowered beforehand
        assert text == text.lower(), 'text should be lowered beforehand'
        assert considered_preferences == set(pref.lower() for pref in considered_preferences), 'considered_preferences should be lowered beforehand'

        # Sort the preferences by length (longest first)
        sorted_preferences = sorted(considered_preferences, key=len, reverse=True)

        # Initialize a set to hold the matched words
        matched_words = set()

        # Iterate over the sorted preferences
        for preference in sorted_preferences:
            # Create a regex pattern with word boundaries
            pattern = r'\b' + re.escape(preference) + r'\b'

            # Check if the preference (as a whole word) is in the text using regex
            if re.search(pattern, text):
                # Before adding, ensure that no longer preference already includes this one
                if not any(preference in longer_pref for longer_pref in matched_words):
                    matched_words.add(preference)

        return matched_words


    def calc_metrics(self, flag_skip_without_url=True, flag_print=False):

        def print_(*args, **kwargs):
            if flag_print:
                print(*args, **kwargs)

        # for i in tqdm(range(len(self.test_recommendation))):
        for i in range(len(self.test_recommendation)):
            # calculate precision, recall, and f1 between pred vs. preferences (prev, curr, respectively)
            pred = self.test_recommendation[i]['pred_text_proposed'].lower()
            base = self.test_recommendation[i]['pred_text_baseline'].lower()
            gold = self.test_recommendation[i]['reference'].lower()

            self.test_recommendation[i]['reference_with_web_content'] = self.append_content_from_url(gold)
            gold = self.test_recommendation[i]['reference_with_web_content']

            # NOTE: This should be no more needed since `session_ind != self.target_session_len-1` 
            # is already filtered out in __init__()
            session_ind = self.test_recommendation[i]['session_ind']
            if session_ind != self.target_session_len-1:
                continue

            # If gold does not contain URL, continue
            if flag_skip_without_url and not re.findall(r'(https?://\S+)', gold):
                print_(f'skip i: {i}', end=', ')
                continue
            print_(f"i: {i}", end=', ')
                
            preferences_each_session_flat, considered_preferences = self.get_flattened_preferences(self.test_recommendation[i]['preferences_for_each_session'], session_ind, use_lower=True)

            pred_preferences = self.preprocess_text_longest_match(pred, considered_preferences)
            base_preferences = self.preprocess_text_longest_match(base, considered_preferences)
            gold_preferences = self.preprocess_text_longest_match(gold, considered_preferences)

            num_preferences_from_each_session = {
                'pred':[None for _ in range(len(preferences_each_session_flat))], 
                'baseline':[None for _ in range(len(preferences_each_session_flat))], 
                'gold':[None for _ in range(len(preferences_each_session_flat))]
                }


            # Inside the loop
            for k in num_preferences_from_each_session.keys(): # k: pred, baseline, gold
                pred_or_gold_or_base = set()
                if k == 'pred':
                    pred_or_gold_or_base = pred_preferences
                elif k == 'baseline':
                    pred_or_gold_or_base = base_preferences
                elif k == 'gold':
                    pred_or_gold_or_base = gold_preferences
                else:
                    raise ValueError(f'k should be either pred, baseline, or gold. k: {k}')

                counts_for_current_key = [] # len should be the same as len(preferences_each_session_flat) i.e., number of sessions
                assert self.target_session_len == len(preferences_each_session_flat), f'target_session_len should be the same as len(preferences_each_session_flat). target_session_len: {self.target_session_len}, len(preferences_each_session_flat): {len(preferences_each_session_flat)}'
                for session_ind in range(self.target_session_len):
                    num_preferences_from_each_session[k][session_ind] = len(pred_or_gold_or_base & preferences_each_session_flat[session_ind])
                    tp = len(pred_or_gold_or_base & preferences_each_session_flat[session_ind] & gold_preferences)
                    fp = len(pred_or_gold_or_base & preferences_each_session_flat[session_ind]) - tp
                    fn = len(gold_preferences & preferences_each_session_flat[session_ind]) - tp

                    counts_for_current_key.append({'tp': tp, 'fp': fp, 'fn': fn})

                self.counts[k]['tp'].append([counts_for_current_key[j]['tp'] for j in range(self.target_session_len)])
                self.counts[k]['fp'].append([counts_for_current_key[j]['fp'] for j in range(self.target_session_len)])
                self.counts[k]['fn'].append([counts_for_current_key[j]['fn'] for j in range(self.target_session_len)])

            # calculate 'preference_usage_pct'
            # i.e., calculate recall for each session
            for k in ['pred', 'baseline', 'gold']:
                # for each session
                pref_use_pct = []
                for i in range(len(preferences_each_session_flat)):
                    # get the total number of preferences for each session (you can use preferences_each_session_flat)
                    total_preferences = len(preferences_each_session_flat[i])
                    # num_preferences for the given session
                    num_preferences = num_preferences_from_each_session[k][i]
                    # pct = num_preferences for the given session / total number of preferences for the given session
                    pct = num_preferences / total_preferences if total_preferences > 0 else 0
                    pref_use_pct.append(pct)
                assert len(pref_use_pct) == len(preferences_each_session_flat), f'len(pref_use_pct) should be the same as len(preferences_each_session_flat). len(pref_use_pct): {len(pref_use_pct)}, len(preferences_each_session_flat): {len(preferences_each_session_flat)}'
                self.scores['preference_usage_pct'][k].append(pref_use_pct)

            self.scores['num_preferences_from_each_session']['pred'].append(num_preferences_from_each_session['pred'])
            self.scores['num_preferences_from_each_session']['baseline'].append(num_preferences_from_each_session['baseline'])
            self.scores['num_preferences_from_each_session']['gold'].append(num_preferences_from_each_session['gold'])

    def calculate_prf_scores(self, flag_print=False):
        for category in ['pred', 'baseline', 'gold']:
            sample_size = len(self.counts[category]['tp']) # not necessarily tp; can be fp or fn. this number is smaller than tqdm loop since if focus is on only the last session.
            if flag_print: print(f"sample_size: {sample_size}")
            for session in range(self.target_session_len):
                tp = sum(self.counts[category]['tp'][sample_ind][session] for sample_ind in range(sample_size))
                fp = sum(self.counts[category]['fp'][sample_ind][session] for sample_ind in range(sample_size))
                fn = sum(self.counts[category]['fn'][sample_ind][session] for sample_ind in range(sample_size))
                p, r, f = calculate_prf(tp, fp, fn)
                self.prf_scores[category]['p'].append(p)
                self.prf_scores[category]['r'].append(r)
                self.prf_scores[category]['f'].append(f)

        if flag_print:
            for category in ['pred', 'baseline', 'gold']:
                if flag_print: print(f"\nPRF scores for each session for {category}:")
                for session in range(self.target_session_len): 
                    if flag_print: print(f"Session {session} - P: {self.prf_scores[category]['p'][session]:.3f}, R: {self.prf_scores[category]['r'][session]:.3f}, F1: {self.prf_scores[category]['f'][session]:.3f}")

                if flag_print: print(f"\nAvg. PRF scores for {category} - P: {np.mean(self.prf_scores[category]['p']):.3f}, R: {np.mean(self.prf_scores[category]['r']):.3f}, F1: {np.mean(self.prf_scores[category]['f']):.3f}")
        if self.prf_scores_overall==None:
            self.prf_scores_overall = calculate_overall_prf_scores([self.counts], flag_print) # why []? --> because calculate_overall_prf_scores() takes a list of counts_list. This is originally designed to take multiple counts_list (e.g., for different target session ind)
        else:
            raise ValueError(f'self.prf_scores_overall should be None. self.prf_scores_overall: {self.prf_scores_overall}')

    def display_f1_scores(self):
        # Plotting the F1 Scores
        categories = ['pred', 'baseline']
        metrics = ['f']
        labels = ['F1 Score']
        colors = {'pred': ['deepskyblue'], 'baseline': ['darkgray']}
        markers = {'f': 'o'}  # 'o' for circle

        plt.figure(figsize=(3, 3))  # Adjusted figure size to make it square
        for category in categories:
            for metric, label, color in zip(metrics, labels, colors[category]):
                scores = self.prf_scores[category][metric]
                line, = plt.plot(range(self.target_session_len), scores, marker=markers[metric], color=color, label=category)
                for i, score in enumerate(scores):
                    plt.text(i, score, f'{score:.2f}', ha='center', va='bottom')

        session_labels = [str(i+1) for i in range(self.target_session_len)]  # Create labels ['1', '2', '3']

        plt.xlabel('Session of Preference Provenance')
        plt.ylabel('F1 Score')

        plt.xticks(range(self.target_session_len), session_labels)
        plt.ylim(0, 0.5)  # Set y-axis range from 0 to 0.5
        plt.legend()
        plt.title(f'Performance by Sessions of Preference Provenance (Current Session = {self.target_session_len})')
        plt.tight_layout()  # Adjust layout to fit in smaller figure size
        plt.show()

    def calculate_ratios(self, flag_print=True):
        for category in ['pred', 'baseline', 'gold']:
            total_preferences = sum([sum(session_counts) for session_counts in self.scores['num_preferences_from_each_session'][category]])
            for session in range(self.target_session_len):
                session_total = sum([session_counts[session] for session_counts in self.scores['num_preferences_from_each_session'][category]])
                session_ratio = session_total / total_preferences if total_preferences > 0 else 0
                self.ratios[category].append(session_ratio)

        if flag_print:
            for category in ['pred', 'baseline', 'gold']:
                print(f"\nRatios for {category}:")
                for session in range(self.target_session_len):
                    print(f"Session {session}: {self.ratios[category][session]:.3f}")

    def display_ratios(self):
        # Plotting the Ratios
        categories = ['pred', 'baseline', 'gold']
        session_indices = np.arange(self.target_session_len)  # Session 1, 2, 3
        bar_width = 0.2

        plt.figure(figsize=(10, 6))
        for i, category in enumerate(categories):
            bars = plt.bar(session_indices + i * bar_width, self.ratios[category], width=bar_width, label=category)
            # Add annotations to each bar
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), ha='center', va='bottom')

        plt.xlabel('Session')
        plt.ylabel('Ratio')
        plt.xticks(session_indices + bar_width, [f'Session {i+1}' for i in range(self.target_session_len)])
        plt.legend()
        plt.title('Preference Provenance Ratio by Session: Where the current turn\'s preferences come from')
        plt.show()
        

    def calculate_preference_recall_for_each_session(self):
        # using self.avg_preference_recall_by_session instead of avg_preference_usage
        for category in ['pred', 'baseline', 'gold']:
            for session in range(self.target_session_len):
                sum_per_session = 0
                data_size = len(self.scores['preference_usage_pct'][category])
                for session_scores in self.scores['preference_usage_pct'][category]:
                    sum_per_session += session_scores[session]
                self.avg_preference_recall_by_session[category].append(sum_per_session / data_size)
            assert len(self.avg_preference_recall_by_session[category]) == self.target_session_len, f'len(self.avg_preference_recall_by_session[category]) should be the same as self.target_session_len. len(self.avg_preference_recall_by_session[category]): {len(self.avg_preference_recall_by_session[category])}, self.target_session_len: {self.target_session_len}'


    def display_preference_recall_for_each_session(self):
        # Plotting the Preference Usage (Recall)
        categories = ['pred', 'baseline', 'gold']
        session_indices = np.arange(self.target_session_len)
        bar_width = 0.2

        plt.figure(figsize=(10, 6))
        for category in categories:
            line, = plt.plot(session_indices + 1, self.avg_preference_recall_by_session[category], marker='o', label=category)
            # Add annotations to each point
            for x, y in zip(session_indices + 1, self.avg_preference_recall_by_session[category]):
                plt.text(x, y, round(y, 3), ha='right', va='bottom')

        plt.xlabel('Session')
        plt.ylabel('Average Preference Usage (Recall)')
        plt.xticks(session_indices + 1)
        plt.legend()
        plt.title('Preference Usage by Session (Recall): How many preferences mentioned in each session are used in the current turn')
        plt.show()


def display_prf_scores(prf_scores, target_session_len, topic, y_lim=(0.0, 1.0), font_size=12, metrics=["f"]):
    """This function displays specified PRF scores over different seed values."""

    valid_metrics = {'p': 'Precision', 'r': 'Recall', 'f': 'F1'}
    categories = ['pred', 'baseline']
    colors = {'pred': 'deepskyblue', 'baseline': 'darkgray'}
    markers = {'p': 's', 'r': '^', 'f': 'o'}
    legend_labels = {'pred': 'Pref. Prompt', 'baseline': 'Standard'}

    # Validation of metrics input
    if not all(metric in valid_metrics for metric in metrics):
        raise ValueError("Invalid metric specified. Valid options are 'p', 'r', 'f'.")

    session_labels = ['1st', '2nd', '3rd'] if target_session_len == 3 else [f'{i+1}th' for i in range(target_session_len)]

    for metric in metrics:
        plt.figure(figsize=(5, 3))
        plt.grid(color='lightgray', linestyle='-', linewidth=1.0, axis='y')

        for category in categories:
            scores = prf_scores[category][metric][:target_session_len]
            plt.plot(range(target_session_len), scores, marker=markers[metric], color=colors[category], label=f'{legend_labels[category]} - {valid_metrics[metric]}')

        plt.xlabel('Session of Preference Disclosure')
        plt.ylabel(f'{valid_metrics[metric]} Score')
        plt.xticks(range(target_session_len), session_labels)
        plt.ylim(y_lim[0], y_lim[1])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f'{valid_metrics[metric]} by Session ({topic.capitalize()})')
        plt.tight_layout()
        plt.show()

def display_f1_scores_from_given_prf_lost_in_the_middle_style(prf_scores, target_session_len, topic, y_lim=(0.0, 1.0), font_size=12):
    """Updated function to display averaged F1 scores with original proposed styling.
    
    Args:
        prf_scores (dict): PRF scores by category and metric.
        target_session_len (int): Length of the target session.
        topic (str): The topic of the data (e.g., 'recipe', 'movie').
        y_lim (tuple): Y-axis limits.
        font_size (int): Font size for all text elements.
    """
    # Revert to original proposed plot styling
    # colors = {'pred': '#a79bcf', 'baseline': '#cf9a9a'}
    colors = {'pred': 'deepskyblue', 'baseline': 'darkgray'}
    markers = {'pred': 'o', 'baseline': 'D'}
    legend_labels = {'pred': 'Pref. Prompt', 'baseline': 'Standard'}
    line_styles = {'pred': '-', 'baseline': '--'}

    # Start plotting
    plt.figure(figsize=(6.5, 3.5))  # Adjusted figure size to make it square
    # plt.gca().set_facecolor('#eaeaf3')  # Light grayish purple background
    plt.gca().set_facecolor('#f0f0f0')  # Light gray background
    plt.grid(color='white', linestyle='-', linewidth=1)

    # Loop through categories and plot
    for category in ['pred', 'baseline']:
        scores = prf_scores[category]['f'][:target_session_len]
        plt.plot(range(target_session_len), scores, marker=markers[category],
                 linestyle=line_styles[category], color=colors[category], label=legend_labels[category])

    # Set session labels
    session_labels = [f'{i+1}th' for i in range(target_session_len)]
    plt.xticks(range(target_session_len), session_labels, fontsize=font_size)

    # Set axis labels and title
    plt.xlabel('Session of Preference Disclosure', fontsize=font_size)
    plt.ylabel('F1', fontsize=font_size)
    plt.ylim(y_lim)

    # Set legend
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=font_size)
    # plt.legend(fontsize=font_size)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=font_size)

    # Set title
    plt.title('Preference Utilization by Session (Recipe)', fontsize=font_size)

    # Remove axis spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Tick labels
    plt.tick_params(axis='both', which='both', length=2, color='gray', labelsize=font_size)

    # Adjust layout
    plt.tight_layout()
    plt.show()
