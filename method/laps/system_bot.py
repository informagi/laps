import json
import config

SURVEY_URL = config.SURVEY_URL

class SystemBot:
    def __init__(self):
        self.states = ["form_confirmation", "done_confirmation", "next_session_query"]
        self.state_index = 0
        self.next_session_answer = None  # Used to track whether user wants to continue the session
        self.conversation_history = []  # This should store both user and system utterances.

    def set_user_utterance_system(self, user_raw_utterance):
        user_utterance = json.dumps({"role": "user", "text": user_raw_utterance})
        self.conversation_history.append(user_utterance)

    def set_system_utterance_system(self, system_raw_utterance):
        system_utterance = json.dumps({"role": "system", "text": system_raw_utterance})
        self.conversation_history.append(system_utterance)

    def get_the_last_user_utterance_string(self):
        if self.conversation_history:  # Check if the list is not empty
            # find the last user utterance
            for i in range(len(self.conversation_history)-1, -1, -1):
                dc = json.loads(self.conversation_history[i])
                if dc['role'] == "user":
                    return dc['text']
        else:
            print("WARNING: The conversation history is empty. Returning None.")
            return None

    def respond(self, user_utterance, flag_print=True, flag_json_output=True):
        if self.states[self.state_index] == "form_confirmation":
            self.state_index += 1
            response = f"SYSTEM: Thank you for participating in this study. We would like to confirm whether the preferences you provided in the form are correctly registered. Please click the link below: {SURVEY_URL} The completion code is shown after answering the questionnaire. Type 'done' once you have finished the questionnaire, and to continue the task."
        elif self.states[self.state_index] == "done_confirmation":
            if user_utterance.lower().strip() == "done":
                self.state_index += 1
                response = "SYSTEM: Thanks for confirming the preferences. Do you want to also participate in the next session? (yes/no)"
            else:
                response = "SYSTEM: Incorrect response. Please enter 'done' once you have finished the questionnaire."
        elif self.states[self.state_index] == "next_session_query":
            user_utterance = user_utterance.lower().strip()
            if user_utterance in ["yes", "yes."]:
                self.state_index += 1
                self.next_session_answer = True
                response = "SYSTEM: Thanks. Let's move to the next part of the study. Greet the assistant and start the conversation from the user interface."
            elif user_utterance in ["no", "no."]:
                self.state_index += 1
                response = "SYSTEM: You have successfully completed the session. Thanks for participating in the study."
            else:
                response = "SYSTEM: Please answer with yes/no."
        
        self.set_system_utterance_system(response)
        
        if flag_json_output:
            response = json.dumps({"role": "system", "text": response})

        if flag_print:
            print("\033[96m" + response + "\033[0m")
            
        return response


    def final_system_response(self, flag_json_output=True):
        """Only used for the final system response of the whole conversation."""
        response = f"SYSTEM: You have successfully completed the all sessions. First, please coplete the questionnaire form at {SURVEY_URL} and obtain the completion code. Then, return to the Prolific page and complete the task. Thanks for participating in the study."
        self.set_system_utterance_system(response)
        if flag_json_output:
            response = json.dumps({"role": "system", "text": response})
        return response

        

    def reset(self):
        self.state_index = 0
        self.next_session_answer = None
    
    def check_finish_system_chat(self):
        if self.state_index == len(self.states):
            return True
        else:
            return False

    def check_continue_session(self):
        return self.next_session_answer
