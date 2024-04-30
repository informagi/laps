
import os
from dotenv import load_dotenv, find_dotenv
import json
import re
from functools import wraps
from datetime import datetime
import random
from colorama import Fore, Style
import argparse

import openai_api
import response
import preference_prompts
import config
import steps
import utils
import preference_extraction
from system_bot import SystemBot
from tqdm import tqdm

DELIMITER = config.DELIMITER
VERBOSE = config.VERBOSE

DEBUG = False
USER_GPT = True

OUTPUT_AS_JSONL = True
OUTPUT_DIR = './data/output'
PATH_TO_DATA = './data'


##################################################
############### EXAMPLE CONVERSATION #############
##################################################




if __name__ == '__main__':
    # Set example drop rate and flag_alternative
    parser = argparse.ArgumentParser()

    # OLD
    parser.add_argument('--example_drop_rate', type=float, default=0, help='Rate of dropping R-examples')
    parser.add_argument('--flag_alternative', type=bool, default=False, help='Whether to use alternative example or not')

    # For LLM-LLM experiments
    parser.add_argument('--seed', type=int, default=None, help='Seed for random preferences')
    parser.add_argument('--use_random_preferences', type=str, default=None, help='Whether to use random preferences or not')
    parser.add_argument('--topic', type=str, default=None, help='Topic: recipe or movie')

    args = parser.parse_args()
    # check --use_random_preferences
    if args.use_random_preferences is not None:
        assert args.use_random_preferences in ['True', 'False'], "Invalid use_random_preferences"
        args.use_random_preferences = True if args.use_random_preferences == 'True' else False
        
    example_drop_rate = args.example_drop_rate
    flag_alternative = args.flag_alternative
    print(f"example_drop_rate: {example_drop_rate} | flag_alternative: {flag_alternative}")
else: # if imported
    example_drop_rate = 0
    flag_alternative = False

if config.USE_ADVICE_MODE:
    response_class = response.Guidance()
else:
    raise ValueError("Only advice (guidance) mode is supported.")

class Conversation:
    def __init__(self, conv_id, topic, initial_assistant_utterance=None):
        self.topic = topic
        # conversation_history includes the current conversation
        self.conversation_history = [{
            "role": "assistant",
            "text": initial_assistant_utterance
        }] if initial_assistant_utterance else []
        self.session_turn_numbers = []
        self.conv_id = conv_id
        # for recipe topic, the conv_id == prolific_id, however, for movie topic, the conv_id is e.g., 'Movie-Guidance--PROLIFIC-ID'
        self.prolific_id = conv_id if topic == 'recipe' else conv_id.split('--')[-1]

        if topic == 'recipe':
            self.preference_property_dict = preference_prompts.RECIPE_PREFERENCE_PROPERTY_DICT
        elif topic == 'movie':
            self.preference_property_dict = preference_prompts.MOVIE_PREFERENCE_PROPERTY_DICT
        else:
            raise ValueError("Invalid topic: {topic}.")

    def _check_duplication(self, current_turn_dc):
        if not self.conversation_history:
            return False

        last_turn_dc = self.conversation_history[-1]
        is_duplicate = (
            last_turn_dc['role'] == current_turn_dc['role'] and
            last_turn_dc['text'] == current_turn_dc['text']
        )
        return is_duplicate


    def set_current_utterance(self, json_str: str):
        current_turn_dc = utils._to_dict(json_str)
        if self._check_duplication(current_turn_dc):
            print('Warning: Duplicate assistant utterance')
        else:
            self.conversation_history.append(current_turn_dc)


    def save_rewritten_utterance(self, original_utterance, rewritten_utterance, i_turn):
        """Saves the rewritten assistant utterance if it's different from the original one."""

        # Check if the original and rewritten utterances are different
        if original_utterance != rewritten_utterance:
            file_path = os.path.join(PATH_TO_DATA, 'rewritten_utterances', self.prolific_id, self.topic, 'rewritten_utterances.json')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Read the existing data
            rewritten_data = []
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    rewritten_data = json.load(f)

            # Add the rewritten utterance along with the turn number
            rewritten_data.append({
                "original": original_utterance,
                "rewritten": rewritten_utterance,
                "turn_number": i_turn
            })

            # Write back to the file
            with open(file_path, 'w') as f:
                json.dump(rewritten_data, f, indent=4)



    def replace_most_recent_assistant_utterance(self, rewritten_assistant_utterance):
        """If the assistant utterance is rewritten, replace the most recent assistant utterance with the rewritten one."""

        # Reverse iterate through the conversation history
        for i in reversed(range(len(self.conversation_history))):
            # Check if the role of the current conversation turn is 'assistant'
            if self.conversation_history[i]['role'] == 'assistant':
                # Save the original utterance
                original_utterance = self.conversation_history[i]['text']

                # Replace the assistant's utterance and break the loop
                self.conversation_history[i]['text'] = rewritten_assistant_utterance

                # Call the method to save the rewritten utterance if it's different
                self.save_rewritten_utterance(original_utterance, rewritten_assistant_utterance, i)
                break

    # def get_current_turn(self, output_keys=['role', 'text']):
    #     # return self.get_recent_n_turns(1)
    #     return self.get_recent_n_turns(1, output_keys=output_keys)

    def get_hist_with_current_utterance(self, output_keys=['role', 'text'], flag_only_current_session=True, flag_only_user=False):
        if flag_only_current_session and self.session_turn_numbers:
            start_of_session = self.session_turn_numbers[-1]
            conversation_history = self.conversation_history[start_of_session:]
            offset = start_of_session
        else:
            conversation_history = self.conversation_history
            offset = 0

        if flag_only_user:
            conversation_history = [turn for turn in conversation_history if turn['role'] == 'user']

        return '\n'.join(f"Turn {idx+1+offset:02d} {json.dumps({key: elem[key] for key in output_keys})}" for idx, elem in enumerate(conversation_history))

    def get_hist_wo_current_utterance(self, output_keys=['role', 'text'], flag_only_current_session=True, flag_only_user=False):
        full_history = self.get_hist_with_current_utterance(output_keys, flag_only_current_session, flag_only_user)
        if full_history:
            return '\n'.join(full_history.split('\n')[:-1])  # Remove last turn
        else:
            return ''

    def print_all_conv_for_verbose(self, output_keys=['role', 'text', 'step']):
        ret = '\n'.join(f"Turn {idx+1:02d} {json.dumps({key: elem[key] for key in output_keys})}" for idx, elem in enumerate(self.conversation_history))
        # print in cyan
        print('\033[96m' + ret + '\033[0m')

    def save_all_conv_to_file(self, example_drop_rate=None, seed=None, topic=None, use_random_preferences=None, out_dir=OUTPUT_DIR, output_keys=['role', 'text', 'step'], dt_string=None, use_advice_mode=config.USE_ADVICE_MODE):

        if use_advice_mode:
            assert topic is not None, "topic must be specified when use_advice_mode is True"
            assert seed is not None, "seed must be specified when use_advice_mode is True"
            assert use_random_preferences is not None, "use_random_preferences must be specified when use_advice_mode is True"
        else:
            assert example_drop_rate is not None, "example_drop_rate must be specified when use_advice_mode is False"

        if dt_string is None:
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        
        # Ensure the output directory exists
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if use_advice_mode:
            file_name = f"{dt_string}_{topic}_seed{seed}_use_random_preferences{str(use_random_preferences)}.txt"
        else:
            # ansi yellow warning
            print('\033[93m' + "WARNING: The file name is very old; including example_drop_rate and flag_alternative in the file name. Please check if this is still valid." + '\033[0m')
            file_name = f"{dt_string}_conv_drop{example_drop_rate}_alt{str(flag_alternative)}.txt"
        file_path = os.path.join(out_dir, file_name)
        with open(file_path, 'w') as f:
            ret = '\n'.join(f"Turn {idx+1:02d} {json.dumps({key: elem[key] for key in output_keys})}" for idx, elem in enumerate(self.conversation_history))
            f.write(ret)

    def get_ptkb_str(self, drop_short_term=True):
        ptkb = self.get_ptkb_combined()
        formatted_strings = []
        for key, values in ptkb.items():
            if len(values) == 0:
                continue
            if drop_short_term and key in self.preference_property_dict and self.preference_property_dict[key]['duration'] == 'short-term':
                continue
            value_str = '; '.join(values)
            formatted_strings.append(f"- {key}: {value_str}")
        return "\n".join(formatted_strings)
    
    def get_ptkb_combined(self):
        # read all existing personal turing knowledge bases (ptkb) from the file
        ptkb_each_session = self.read_ptkb()
        ptkb_each_session = [ptkb for ptkb in ptkb_each_session if ptkb is not None] # remove None to avoid None not iterable error

        # combine all existing ptkbs into a single one
        # for each key, we add the list of values and remove duplicates by converting the list to a set and back to a list
        return {key: list(set([elem for session_ptkb in ptkb_each_session for elem in session_ptkb.get(key, [])])) for key in set().union(*ptkb_each_session)}


    # @try_n_times_decorator(n=3)
    def update_ptkb(self):
        new_pt = preference_extraction.extract(self.get_hist_with_current_utterance(), self.topic)
        self.save_ptkb_by_appending_to_existing_json(new_pt)

    def save_ptkb_by_appending_to_existing_json(self, new_pt: dict):
        path = os.path.join(PATH_TO_DATA, 'ptkb', self.prolific_id, self.topic, 'ptkb.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        ptkb = self.read_ptkb()

        ptkb.append(new_pt)
        
        with open(path, 'w') as f:
            print(f'\033[94m' + f'Saving ptkb to {path}' + '\033[0m')
            json.dump(ptkb, f, indent=4)

    def save_conv_history(self, output_keys=['role', 'text', 'step']):
        path = os.path.join(PATH_TO_DATA, 'conv', self.prolific_id, self.topic, 'conv.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w') as f:
            json.dump([{key: turn[key] for key in output_keys} for turn in self.conversation_history], f, indent=4)


    def read_ptkb(self):
        path = os.path.join(PATH_TO_DATA, 'ptkb', self.prolific_id, self.topic, 'ptkb.json')

        if os.path.isfile(path):
            with open(path, 'r') as f:
                ptkb = json.load(f)
        else:
            ptkb = []
        
        return ptkb


    def start_new_session(self, flag_update_ptkb) -> None:
        if flag_update_ptkb:
            self.update_ptkb()
        self.session_turn_numbers.append(len(self.conversation_history))

    def utterance_duplication(self, response: str, flag_only_current_session=True) -> bool:
        """Check if the utterance is a duplicate of a previous utterance.
        Implement `response not in conv.get_hist_wo_current_utterance()` in more detail.
        """
        duplicate_threshold = 0.2 # 20% of the characters in the response are duplicates of a previous utterance
        n = int(len(response) * duplicate_threshold)

        # response is json string; convert to dict
        response = utils._to_dict(response)['text']
        
        # if there is an exact match for more than n characters, then it is a duplicate
        _response = response.lower()
        _hist_wo_current_utterance = self.get_hist_wo_current_utterance(flag_only_current_session=flag_only_current_session).lower()

        # slide a window of size n over the response and check if each substring exists in history
        for i in range(len(_response) - n + 1):
            substring = _response[i:i+n]
            if substring in _hist_wo_current_utterance:
                print('\033[93m' + f"WARNING: The substring '{substring}' is a duplicate of a previous utterance. n={n}" + '\033[0m')
                return True

        return False


if DEBUG:
    conv = Conversation()




class RecipeChat:
    """Not resticted to recipe, but keep this name to avoid discrepancy with the chats.pkl file used in llm_server.py"""
    def __init__(self, conv_id, topic, initial_assistant_utterance=None):
        assert topic in ['recipe', 'movie'], f"Invalid topic: {topic}"
        self.conv = Conversation(conv_id, topic, initial_assistant_utterance=initial_assistant_utterance) if initial_assistant_utterance else Conversation(conv_id, topic)

        steps_session1 = [
            {"name": "greeting", "max_turn": 4}, # NOTE: This number also includes pre-defined assistant utterances from TaskMAD.
            {"name": "preference_elicitation_must", "max_turn": 4},
            {"name": "preference_elicitation_should", "max_turn": 4},
            {"name": "preference_elicitation_could", "max_turn": 4},
            {"name": "recommendation", "max_turn": 6},
            {"name": "recommendation_followup", "max_turn": 2},
            {"name": "closing", "max_turn": 2}
            ]
        steps_session2 = [
            {"name": "greeting_session2", "max_turn": 4},
            {"name": "preference_elicitation_session2", "max_turn": 4},
            {"name": "recommendation_session2", "max_turn": 6},
            {"name": "recommendation_followup_session2", "max_turn": 2},
            {"name": "closing_session2", "max_turn": 2}
            ]
        steps_session3 = [
            {"name": "greeting_session3", "max_turn": 4},
            {"name": "preference_elicitation_session3", "max_turn": 4},
            {"name": "recommendation_session3", "max_turn": 6},
            {"name": "recommendation_followup_session3", "max_turn": 2},
            {"name": "closing_session3", "max_turn": 2}
            ]


        self.start_steps_of_subsequent_sessions = [steps_session2[0]['name'], steps_session3[0]['name']]
        self.steps = steps_session1 + steps_session2 + steps_session3
        self.step_index = 0
        self.turn = 0
        self.session_index = 0
        self.role = 'assistant' # 'assistant' or 'system'
        self.flag_task_end = False

        self.system_bot = SystemBot()

        self.steps_session1 = steps_session1  # Used for debugging
        self.stepindex2turncount = {step_index: 0 for step_index in range(len(self.steps))}  # Store the turn count for each step index. This is used for step completion detection by using max_turn.

        # topic specific
        if topic == 'recipe':
            self.meal_types = ["dinner", "breakfast", "lunch"]
        elif topic == 'movie':
            self.meal_types = ["movie_session_1", "movie_session_2", "movie_session_3"]
        self.topic = topic
            
    def set_user_utterance_assistant(self, user_raw_utterance):
        user_utterance = json.dumps({"role": "user", "text": user_raw_utterance, "step": self.steps[self.step_index]['name']})
        self.conv.set_current_utterance(user_utterance)
        self.turn = len(self.conv.conversation_history)
        self.stepindex2turncount[self.step_index] = self.turn - sum(v for k, v in self.stepindex2turncount.items() if k < self.step_index)  # Update the turn count for the current step index

    def set_user_utterance_system(self, user_raw_utterance):
        self.system_bot.set_user_utterance_system(user_raw_utterance)

    def set_user_utterance(self, user_raw_utterance):
        if self.role == 'assistant':
            self.set_user_utterance_assistant(user_raw_utterance)
        elif self.role == 'system':
            self.set_user_utterance_system(user_raw_utterance)

    def replace_most_recent_assistant_utterance(self, rewritten_assistant_utterance):
        """If the assistant utterance is rewritten, replace the most recent assistant utterance with the rewritten one."""
        self.conv.replace_most_recent_assistant_utterance(rewritten_assistant_utterance)

    def system_response(self) -> str:
        user_utterance = self.system_bot.get_the_last_user_utterance_string() # user utterance might be None if there is no system-related user utterance. 
        system_response = self.system_bot.respond(user_utterance)  # json string
        if self.system_bot.check_finish_system_chat():
            if not self.system_bot.check_continue_session():  # User decided to end the session
                self.flag_task_end = True
            self.role = 'assistant' # Initialize the role to 'assistant' when the system chat is finished.
        return system_response

    def system_response_complete(self) -> str:
        """This function is called when the task is completed."""
        system_response = self.system_bot.final_system_response()
        return system_response
        

    def assistant_response(self) -> str:
        asst_response_json = response_class.response_generation(self.steps[self.step_index]['name'], self.conv, self.topic)
        self.conv.set_current_utterance(asst_response_json)
        self.turn = len(self.conv.conversation_history)
        self.stepindex2turncount[self.step_index] = self.turn - sum(v for k, v in self.stepindex2turncount.items() if k < self.step_index)  # Update the turn count for the current step index
        return asst_response_json


    def get_response(self):
        if self.flag_task_end:
            end_response = json.dumps({"role": "system", "text": "SYSTEM: The chat task has completed. Thanks for participating in the study."})
            return end_response, self.step_index
            
        if self.role == 'system' and self.step_index < len(self.steps)-1: # if not the final step
            response = self.system_response()
            return response, self.step_index

        if steps.step_completion_detection(self.steps[self.step_index]['name'], self.conv, self.topic) or (self.stepindex2turncount[self.step_index] >= self.steps[self.step_index]['max_turn']):
            # Check completion of the task or not
            if self.step_index >= len(self.steps)-1:
                if self.role != 'system': # The final step should be completed by the system
                    self.role = 'system'
                    # first update ptkb
                    self.conv.update_ptkb()
                    self.system_bot.reset()
                    return self.system_response_complete(), self.step_index
                else:
                    print("Complete!")
                    self.flag_task_end = True
                    return None, None
            else:
                self.update_step()

        response = None
        if self.role == 'assistant':
            response = self.assistant_response()
        elif self.role == 'system':
            response = self.system_response()
        else:
            response = "Error: Invalid role"

        # IMPORTANT: self.role and 'role' in response is not necessarily the same. This is because the role in response is updated/prepared for the next turn.
        return response, self.step_index


    def update_step(self):
        self.step_index += 1
        if self.steps[self.step_index]['name'] in self.start_steps_of_subsequent_sessions:
            self.start_new_session()

    def start_new_session(self):
        self.conv.start_new_session(flag_update_ptkb=True)
        self.session_index += 1
        self.role = 'system'
        self.system_bot.reset()  # Reset the system bot when starting a new session

