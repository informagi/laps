import json
import os
from pathlib import Path
import re
import sys

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = os.path.join(PROJECT_DIR, 'existing_datasets')

# sys.path.append(os.path.join(PROJECT_DIR, 'analysis'))
# import display_conversation as disp_conv

class DialogueDataset():

    def get_messages(self, *args, **kwargs):
        all_dialogues = self.get_dialogues(*args, **kwargs)

        all_messages = [text for dialogue in all_dialogues for text in [entry['text'] for entry in dialogue]]
        assistant_messages = [text for dialogue in all_dialogues for text in [entry['text'] for entry in dialogue if entry['role'] == 'assistant']]
        user_messages = [text for dialogue in all_dialogues for text in [entry['text'] for entry in dialogue if entry['role'] == 'user']]

        return all_messages, assistant_messages, user_messages


class CCPEM(DialogueDataset):

    def __init__(self):
        self.file_path = os.path.join(DATASET_DIR, 'ccpe/data.json')

    def get_dialogues(self):
        """
        Return:
            - all_dialogues: list of all dialogues, where each dialogue is the list of dict; i.e., [[{"role": role, "text": text}, ...], ...]
        """

        # Initialize list to store all dialogues
        all_dialogues = []

        # Load the CCPE-M dataset
        with open(self.file_path, 'r') as f:
            data = json.load(f)

        # Loop through each conversation in the dataset
        for conversation in data:
            utterances = conversation['utterances']

            # Initialize list to store the dialogue
            dialogue = []

            # Loop through each utterance in the conversation
            for utterance in utterances:
                text = utterance['text']
                raw_speaker = utterance['speaker']

                # Convert raw_speaker to speaker
                if raw_speaker == 'ASSISTANT':
                    speaker = 'assistant'
                elif raw_speaker == 'USER':
                    speaker = 'user'
                else:
                    raise ValueError(f"Invalid speaker: {raw_speaker}")

                # Add entry to the dialogue
                dialogue.append({"role": speaker, "text": text})
            
            # Add dialogue to all_dialogues
            all_dialogues.append(dialogue)

        return all_dialogues



class MGShopDial(DialogueDataset):

    def __init__(self):
        self.file_path = os.path.join(DATASET_DIR, 'MG-ShopDial/MGShopDial/MGShopDial.json')

    def get_dialogues(self):
        """
        Return:
            - all_dialogues: list of all dialogues, where each dialogue is the list of dict; i.e., [[{"role": role, "text": text}, ...], ...]
        """

        # Initialize list to store all dialogues
        all_dialogues = []

        # Load the MGShopDial dataset
        with open(self.file_path, 'r') as f:
            data = json.load(f)

        # Loop through each conversation in the dataset
        for conversation in data:
            utterances = conversation['utterances']

            # Initialize list to store the dialogue
            dialogue = []

            # Loop through each utterance in the conversation
            for utterance in utterances:
                text = utterance['utterance']
                raw_participant = utterance['participant']

                # Convert raw_participant to role
                if raw_participant == 'Wizard':
                    role = 'assistant'
                elif raw_participant == 'Client':
                    role = 'user'
                else:
                    raise ValueError(f"Invalid participant: {raw_participant}")

                # Add entry to the dialogue
                dialogue.append({"role": role, "text": text})

            # Add dialogue to all_dialogues
            all_dialogues.append(dialogue)

        return all_dialogues
        

class M2M(DialogueDataset):

    def __init__(self):
        self.file_path = os.path.join(DATASET_DIR, 'simulated-dialogue/sim-R/train.json')

    def get_dialogues(self):
        """
        Return:
            - all_dialogues: list of all dialogues, where each dialogue is the list of dict; i.e., [[{"role": role, "text": text}, ...], ...]
        """
        # Initialize list to store all dialogues
        all_dialogues = []
        
        # Load the M2M dataset
        with open(self.file_path, 'r') as f:
            data = json.load(f)

        # Loop through each conversation in the dataset
        for conversation in data:
            turns = conversation['turns']
            
            # Initialize list to store the dialogue
            dialogue = []

            # Loop through each turn in the conversation
            for i, turn in enumerate(turns):
                # Get user and system utterances
                user_utterance = turn.get('user_utterance', {}).get('text', '')
                system_utterance = turn.get('system_utterance', {}).get('text', '')
                
                if user_utterance:
                    dialogue.append({"role": "user", "text": user_utterance})
                if system_utterance:
                    dialogue.append({"role": "assistant", "text": system_utterance})
            
            # Add dialogue to all_dialogues
            all_dialogues.append(dialogue)
        
        return all_dialogues

class MWOZ(DialogueDataset):

    def __init__(self):
        self.dir_path = os.path.join(DATASET_DIR, 'multiwoz/data/MultiWOZ_2.2/train')

    def get_dialogues(self, flag_use_only_restaurant):
        """
        Return:
            - all_dialogues: list of all dialogues, where each dialogue is the list of dict; i.e., [{"role": role, "text": text}, ...]
        """
        # Initialize list to store all dialogues
        all_dialogues = []
        
        # Loop through all files in the directory
        for file_name in os.listdir(self.dir_path):
            # Check if the file is a JSON file
            if file_name.endswith(".json"):
                # Get the full file path
                file_path = os.path.join(self.dir_path, file_name)
                
                # Load the data from the file
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Loop through each dialogue in the dataset
                for dialogue in data:
                    # If flag_use_only_restaurant is set to True, skip dialogues that don't involve the restaurant service
                    if flag_use_only_restaurant and "restaurant" not in dialogue['services']:
                        continue
                    
                    # Initialize list to store the dialogue
                    current_dialogue = []

                    # Loop through each turn in the dialogue
                    for turn in dialogue['turns']:
                        # Determine the role based on the speaker
                        role = 'assistant' if turn['speaker'] == 'SYSTEM' else 'user'
                        text = turn['utterance']

                        # Add entry to the current dialogue
                        current_dialogue.append({"role": role, "text": text})

                    # Add current_dialogue to all_dialogues
                    all_dialogues.append(current_dialogue)
        
        return all_dialogues

class Taskmaster(DialogueDataset):

    def __init__(self):
        # Change the file path according to your directory structure
        self.file_path_self = os.path.join(DATASET_DIR, 'Taskmaster/TM-1-2019/self-dialogs.json')  # self-dialogs
        self.file_path_woz = os.path.join(DATASET_DIR, 'Taskmaster/TM-1-2019/woz-dialogs.json')  # spoken

    def get_dialogues(self, flag_use_only_restaurant, self_or_spoken):
        """
        Return:
            - all_dialogues: list of all dialogues, where each dialogue is the list of dict; i.e., [[{"role": role, "text": text}, ...], ...]
        """
        assert self_or_spoken in ['self', 'spoken'], "self_or_spoken must be either 'self' or 'spoken'"

        # Initialize list to store all dialogues
        all_dialogues = []

        # Load the Taskmaster dataset
        data = json.load(open(self.file_path_self, 'r')) if self_or_spoken == 'self' else json.load(open(self.file_path_woz, 'r'))

        # Loop through each conversation in the dataset
        for conversation in data:
            instruction_id = conversation['instruction_id']

            # If flag_use_only_restaurant is True, skip dialogues that don't involve restaurant
            if flag_use_only_restaurant and "restaurant" not in instruction_id:
                continue

            utterances = conversation['utterances']

            # Initialize list to store the dialogue
            dialogue = []

            # Loop through each utterance in the conversation
            for utterance in utterances:
                text = utterance['text']
                role = utterance['speaker'].lower()

                # Add entry to the dialogue
                dialogue.append({"role": role, "text": text})

            # Add dialogue to all_dialogues
            all_dialogues.append(dialogue)

        return all_dialogues


class Baize(DialogueDataset):

    def __init__(self):
        # Mapping from topic to the corresponding file name
        self.topic_to_filename = {
            'alpaca': 'alpaca_chat_data.json',
            'medical': 'medical_chat_data.json',
            'quora': 'quora_chat_data.json',
            'stackoverflow': 'stackoverflow_chat_data.json'
        }

    @staticmethod
    def convert_role(role):
        """Convert raw roles to standardized roles"""
        if role == "AI":
            return "assistant"
        elif role == "Human":
            return "user"
        else:
            raise ValueError(f"Invalid role: {role}")

    def separate_into_turns(self, text):
        """
        Separate the conversation text into turns based on the specified role patterns.
        Return a list of dictionaries with 'role' and 'text'.
        """
        turns = []
        split_patterns = [r'(\n\[\|AI\|\]|\n\[\|Human\|\])']
        split_data = re.split('|'.join(split_patterns), text)

        role = None
        for entry in split_data:
            if entry is None or not entry.strip():  # Skip None or empty/whitespace-only entries
                continue

            if entry == "\n[|AI|]":
                role = "AI"
            elif entry == "\n[|Human|]":
                role = "Human"
            else:
                if role:
                    readable_role = self.convert_role(role)
                    turns.append({"role": readable_role, "text": entry.strip()})

        return turns


    def get_dialogues(self, topic):
        """
        Return:
            - all_dialogues: list of all dialogues, where each dialogue is the list of dict; i.e., [[{"role": role, "text": text}, ...], ...]
        """
        # Ensure the topic provided is valid
        valid_topics = list(self.topic_to_filename.keys())
        if topic not in valid_topics:
            raise ValueError(f"Invalid topic. Choose from: {', '.join(valid_topics)}")

        file_path = os.path.join(DATASET_DIR, f"baize-chatbot/data/{self.topic_to_filename[topic]}")

        # Initialize list to store all dialogues
        all_dialogues = []

        # Load the Baize dataset for the given topic
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Loop through each conversation in the dataset
        for conversation in data:
            input_text = conversation['input']
            dialogue = self.separate_into_turns(input_text)
            
            # Add dialogue to all_dialogues
            all_dialogues.append(dialogue)

        return all_dialogues


class SGD(DialogueDataset):
    def __init__(self):
        self.train_dir_path = os.path.join(DATASET_DIR, 'dstc8-schema-guided-dialogue/train')
        self.dev_dir_path = os.path.join(DATASET_DIR, 'dstc8-schema-guided-dialogue/dev')
        self.test_dir_path = os.path.join(DATASET_DIR, 'dstc8-schema-guided-dialogue/test')

    def get_dialogues(self, flag_use_only_restaurant):
        all_dialogues = []
        for dir_path in [self.train_dir_path, self.dev_dir_path, self.test_dir_path]:
            if not os.path.exists(dir_path):
                raise ValueError(f"Directory {dir_path} doesn't exist.")
            for file_name in os.listdir(dir_path):
                if file_name.endswith(".json"):
                    file_path = os.path.join(dir_path, file_name)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    for dialogue in data:
                        if "services" not in dialogue:
                            continue  # Skip dialogues without 'services' key
                        if flag_use_only_restaurant and not any(service in ["Restaurants_1", "Restaurants_2"] for service in dialogue["services"]):
                            continue
                        current_dialogue = []
                        for turn in dialogue['turns']:
                            # role = 'assistant' if turn['speaker'] == 'SYSTEM' else 'user'
                            if turn['speaker'] == 'SYSTEM':
                                role = 'assistant'
                            elif turn['speaker'] == 'USER':
                                role = 'user'
                            else:
                                raise ValueError(f"Invalid speaker: {turn['speaker']}")
                            text = turn['utterance']
                            current_dialogue.append({"role": role, "text": text})
                        all_dialogues.append(current_dialogue)
        return all_dialogues


class OurSelfDial(DialogueDataset):

    def __init__(self):
        self.file_path = os.path.join(PROJECT_DIR, 'analysis/firebase/data/processed_dialogues')

    def get_dialogues(self, topic):
        """
        Return:
            - all_dialogues: list of all dialogues, where each dialogue is the list of dict; i.e., [[{"role": role, "text": text}, ...], ...]
        """
        # Ensure the topic provided is valid
        valid_topics = ['movie', 'music', 'recipe', 'travel']
        if topic not in valid_topics:
            raise ValueError(f"Invalid topic. Choose from: {', '.join(valid_topics)}")

        # Update the path based on the topic
        topic_path = os.path.join(self.file_path, topic)

        # Initialize list to store all dialogues
        all_dialogues = []

        # Get list of all files in the directory
        file_list = os.listdir(topic_path)

        # Loop through each file in the directory
        for file_name in file_list:
            # Check if file is a .json file
            if file_name.endswith('.json'):
                with open(os.path.join(topic_path, file_name), 'r') as f:
                    data = json.load(f)

                    # Initialize list to store the dialogue
                    dialogue = []

                    # Loop through each entry (message) in the data
                    for entry in data:
                        raw_role = entry['role']
                        text = entry['utterance']

                        # Convert raw_role to role
                        if raw_role == 'ASSISTANT':
                            role = 'assistant'
                        elif raw_role == 'USER':
                            role = 'user'
                        else:
                            raise ValueError(f"Invalid role: {raw_role}")

                        # Add entry to the dialogue
                        dialogue.append({"role": role, "text": text})
                    
                    # Add dialogue to all_dialogues
                    all_dialogues.append(dialogue)
        
        return all_dialogues


class OurLLM2(DialogueDataset):
    """Our LLM-LLM setting"""
    def __init__(self, file_names):
        self.file_paths = [os.path.join(PROJECT_DIR, 'assistant/data/output', file_name) for file_name in file_names]

    def _extract_json(self, line):
        """
        Extract the JSON part from a line of the format:
        Turn ## {JSON_content}
        """
        match = re.search(r'\{.*\}', line)
        if match:
            return json.loads(match.group(0))
        return None
    
    def _new_session(self, step_current, step_previous):
        # if previous step has "closing" as a substring, and current step has "greeting" as a substring, then it's a new session.
        if (step_previous and "closing" in step_previous) and (step_current and "greeting" in step_current):
            return True
        else:
            return False

    def get_dialogues(self):
        """
        Return:
            - all_dialogues: list of all dialogues, where each dialogue is the list of dict; i.e., [[{"role": role, "text": text}, ...], ...]
        """
        all_dialogues = []

        # Loop through each file in the provided filenames
        for file_name in self.file_paths:
            with open(file_name, 'r') as f:
                lines = f.readlines()
                
                dialogue = []
                step_current = None
                step_previous = None
                # Extract the JSON content from each line and categorize the messages
                for line in lines:
                    json_content = self._extract_json(line)
                    if json_content:
                        message = json_content['text']
                        if message.lower() == "done":
                            continue
                        role = json_content['role']
                        step_current = json_content['step']
                        if self._new_session(step_current, step_previous) and len(dialogue) > 0:
                            all_dialogues.append(dialogue)
                            dialogue = []
                        dialogue.append({"role": role, "text": message})
                        step_previous = step_current

                all_dialogues.append(dialogue)
        
        return all_dialogues


class OursPublished(DialogueDataset):

    def __init__(self, topic):
        if topic not in ['recipe', 'movie']:
            raise ValueError("Invalid topic. Please choose 'recipe' or 'movie'.")

        self.file_path = os.path.join(PROJECT_DIR, f'dataset/{topic}_dataset.json')

    def get_dialogues(self):
        """
        Return:
            - all_dialogues: list of all dialogues, where each dialogue is a list of dict; i.e., [[{"role": role, "text": text}, ...], ...]
        """

        # Initialize list to store all dialogues
        all_dialogues = []

        # Load the dataset
        with open(self.file_path, 'r') as f:
            data = json.load(f)

        # Loop through each session group in the dataset
        for session_group in data:
            sessions = session_group['sessions']

            # Loop through each session in the session group
            for session in sessions:
                dialogue = session['dialogue']

                # Process each message in the dialogue
                processed_dialogue = [{"role": message['role'].lower(), "text": message['message']} for message in dialogue]
                
                # Add processed dialogue to all_dialogues
                all_dialogues.append(processed_dialogue)

        return all_dialogues



class PersonaChatGen(DialogueDataset):

    def __init__(self, only_food_drink=False):
        self.file_path = os.path.join(DATASET_DIR, 'PersonaChatGenData/Chat/train_both_original_no_cands.txt')
        self.only_food_drink = only_food_drink

    def _filter_conversations_by_keywords(self, conversations, keywords):
        filtered_conversations = []
        for conv in conversations:
            if any(keyword in persona.lower() for persona in conv['persona_you'] + conv['persona_partner'] for keyword in keywords):
                filtered_conversations.append(conv)
        return filtered_conversations

    def get_dialogues(self):
        # Parse the conversations
        conversations = self.parse_conversation(self.file_path)

        # If only_food_drink is True, filter the conversations
        if self.only_food_drink:
            keywords = ['drink', 'food']
            conversations = self._filter_conversations_by_keywords(conversations, keywords)

        # Extract dialogues from the filtered conversations
        all_dialogues = []
        for conv in conversations:
            dialogue = []
            for turn in conv['turns']:
                dialogue.append({'role': turn['role'], 'text': turn['text']})
            all_dialogues.append(dialogue)

        return all_dialogues

    def parse_conversation(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        conversations = []
        current_conversation = {'persona_you': [], 'persona_partner': [], 'turns': []}
        previous_line_number = 0

        for line in lines:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                line_number, content = parts
                line_number = int(line_number)

                if line_number == 1 and previous_line_number != 0:
                    conversations.append(current_conversation)
                    current_conversation = {'persona_you': [], 'persona_partner': [], 'turns': []}

                if content.startswith('your persona:'):
                    current_conversation['persona_you'].append(content.replace('your persona:', '').strip())
                elif content.startswith('partner\'s persona:'):
                    current_conversation['persona_partner'].append(content.replace('partner\'s persona:', '').strip())
                else:
                    assistant_turn, user_turn = content.split('\t')
                    current_conversation['turns'].append({'role': 'assistant', 'text': assistant_turn})
                    current_conversation['turns'].append({'role': 'user', 'text': user_turn})

                previous_line_number = line_number

        conversations.append(current_conversation)

        return conversations