import os
import json
import pandas as pd

# main part
class UniEvalScore:

    def __init__(self, dialogues):
        self.dialogues = dialogues # set None if you want to use read_scores
        self.scores = None
    
    def read_scores(self, path: str):
        assert self.dialogues is None, 'dialogues must be None'
        if not path.endswith('.json'):
            raise ValueError('path must be a json file')
        if not os.path.exists(path):
            raise ValueError('path does not exist')
        with open(path, 'r') as f:
            scores = json.load(f)
        self.scores = scores
        return scores

    # def calc(self, task: str ='dialogue'):

    #     def create_unieval_lists(dialogue_):
    #        """To use this function, you need to install the UniEval package.
    #        See: https://github.com/maszhongming/UniEval
    #        """
    #         src_list = []
    #         context_list = []
    #         output_list = []
    #         roles = []
    #         turn_nums = []
    #         texts = []  # <-- Added
    #         for idx, turn in enumerate(dialogue_):
    #             context = " \n ".join([t['text'] for t in dialogue_[:idx]])
    #             src_list.append(context)
    #             output_list.append(turn['text'])
    #             context_list.append('')
    #             roles.append(turn['role'])
    #             turn_nums.append(idx)
    #             texts.append(turn['text']) 
    #         return src_list, context_list, output_list, roles, turn_nums, texts

    #     all_eval_scores = []

    #     for dialogue in self.dialogues:
    #         src_list, context_list, output_list, roles, turn_nums, texts = create_unieval_lists(dialogue)
            
    #         # Prepare data for pre-trained evaluators
    #         data = convert_to_json(output_list=output_list, src_list=src_list, context_list=context_list) # Function from UniEval package
            
    #         # Evaluate Quality Metrics
    #         evaluator = get_evaluator(task)
    #         eval_scores_raw = evaluator.evaluate(data, print_result=False) # Function from UniEval package
            
    #         # Re-structure and add role, turn_num, and text to eval scores
    #         dialogue_scores = [{'role': roles[idx], 'turn_num': turn_nums[idx], 'scores': score, 'text': texts[idx]}
    #                            for idx, score in enumerate(eval_scores_raw)]
            
    #         all_eval_scores.append(dialogue_scores)

    #     self.scores = {'all_scores': all_eval_scores}
    #     return self.scores

    def get_avg(self):
        """
        Returns:
            avg_scores (dict): {'assistant': {'metric1': 0.5, 'metric2': 0.6, ...}, 'user': {'metric1': 0.5, 'metric2': 0.6, ...}}
        """
        scores = self.calc() if self.scores is None else self.scores
        avg_scores = {'assistant': {}, 'user': {}}

        # Flatten all_scores for calculating the averages
        flattened_scores = [score for dialogue_scores in scores['all_scores'] for score in dialogue_scores]

        for role in ['assistant', 'user']:
            role_scores = [score for score in flattened_scores if score['role'] == role]

            # If there's no score, just continue to the next iteration
            if not role_scores:
                continue

            for metric in role_scores[0]['scores'].keys():
                avg_scores[role][metric] = sum([score['scores'][metric] for score in role_scores]) / len(role_scores)

        return avg_scores
    
    # def save_all_scores(self, path: str):
    #     """
    #     Args:
    #         path (str): path to save the scores
    #     """
    #     # save json
    #     with open(path, 'w') as f:
    #         json.dump(self.scores, f)




def get_metrics_from_score_path(score_path, significant_digits=3):
    # use UniEvalScore.read_scores to read the score
    uni_eval = UniEvalScore(dialogues=None)
    uni_eval.read_scores(score_path)
    avg_scores = uni_eval.get_avg()
    metrics_dict = {
        ('UniEval (NAT)', 'Assistant'): [f"{avg_scores['assistant']['naturalness']:.{significant_digits}f}"],
        ('UniEval (NAT)', 'User'): [f"{avg_scores['user']['naturalness']:.{significant_digits}f}"],
        ('UniEval (COH)', 'Assistant'): [f"{avg_scores['assistant']['coherence']:.{significant_digits}f}"],
        ('UniEval (COH)', 'User'): [f"{avg_scores['user']['coherence']:.{significant_digits}f}"],
        ('UniEval (UND)', 'Assistant'): [f"{avg_scores['assistant']['understandability']:.{significant_digits}f}"],
        ('UniEval (UND)', 'User'): [f"{avg_scores['user']['understandability']:.{significant_digits}f}"],
    }
    return pd.DataFrame(metrics_dict)



def add_avg_cols(df):
    """
    Function to add average score columns for each metric and a final average score column in a multi-level header DataFrame.

    Args:
    df (pd.DataFrame): The input DataFrame with multi-level headers.

    Returns:
    pd.DataFrame: A new DataFrame with added average score columns and a final average column.
    """
    # Copying the DataFrame to avoid modifying the original
    new_df = df.copy()

    # Extracting the unique metrics from the first level of the multi-level header
    metrics = set([col[0] for col in df.columns])

    # Lists to hold all user and assistant scores for final average calculation
    all_user_scores = []
    all_assistant_scores = []

    # Calculating and adding the average score columns for each metric
    for metric in metrics:
        assistant_scores = []
        user_scores = []

        # Iterating through each row to calculate averages
        for _, row in new_df.iterrows():
            try:
                assistant_score = row[(metric, "Assistant")]
                user_score = row[(metric, "User")]
            except KeyError as e:
                print(f"KeyError encountered: {e}")
                continue

            assistant_scores.append(float(assistant_score))
            user_scores.append(float(user_score))

            # Adding scores to the final average lists
            all_assistant_scores.append(float(assistant_score))
            all_user_scores.append(float(user_score))

        # Calculating the average for each metric
        avg_scores = [(a + u) / 2 for a, u in zip(assistant_scores, user_scores)]

        # Adding the average column to the DataFrame
        new_df[(metric, 'Avg.')] = avg_scores

    # Reordering the columns to User, Assistant, Avg.
    new_columns = []
    for metric in metrics:
        new_columns.extend([(metric, 'User'), (metric, 'Assistant'), (metric, 'Avg.')])

    new_df = new_df.reindex(new_columns, axis=1)

    # Calculating and adding the row-wise final average
    final_avg_scores = []
    for _, row in new_df.iterrows():
        avg_scores = [row[(metric, 'Avg.')] for metric in metrics]
        row_final_avg = sum(avg_scores) / len(avg_scores)
        final_avg_scores.append(row_final_avg)

    new_df[('Final Average', '')] = final_avg_scores

    # Convert all values to str
    new_df = new_df.applymap(str)
    # Convert all to 3 dicimal places
    new_df = new_df.applymap(lambda x: x[:5])

    return new_df


# def highlight_higher_value(result_df):
    
#     def apply_highlighting(x, best_val):
#         return f"<b>{x}</b>" if float(x) == best_val else x

#     for col in result_df.columns:
#         val_series = result_df[col].apply(float)
#         best_val = max(val_series)  # Always higher is better
#         result_df[col] = result_df[col].apply(lambda x: apply_highlighting(x, best_val))

#     return result_df