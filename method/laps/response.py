
import json
import openai_api
import re
import preference_prompts
import config
import utils
import random

DELIMITER = config.DELIMITER
VERBOSE = config.VERBOSE
GUIDANCE_DELETE_SUFFIX = 'Delete this guidance and compose your utterance in this text box.'


class Guidance:
    """Generate guidance for the assistant on how to write a response. (i.e., 'option 3' discussed on 2023-08-25)    
    """
    def __init__(self):
        self.prompt_per_topic = {
            'recipe': {
                'task_setting': 'breakfast, lunch, or dinner',
                'preference_elicitation': 'food preferences and cooking habits'
            },
            'movie': {
                'task_setting': 'viewing movies at home, while travelling, or on other occasions, as well as any specific requests',
                'preference_elicitation': 'movie preferences'
            }
        }
            

    def postprocess(self, advice_dict: dict, step, topic) -> dict:
        """
        - Convert `{"advice_to_assistant": <guidance>}` to `{"role": "assistant", "text": <text>}` to make it compatible with Candidate.
        - Prepend "(Guidance:" and append ")" to the guidance.
        """
        assert type(advice_dict) == dict, f"advice_dict must be dict, but {type(advice_dict)} is given."
        guidance = advice_dict["advice_to_assistant"]
        if step in ["recommendation", "recommendation_session2", "recommendation_session3"]:
            guidance = f"{guidance} Checkboxes: You should (1) search the web and provide the {topic}'s URL, (2) explain the reason for your recommendation, and (3) ask the user for feedback with its reason."
        guidance = f"(Guidance: {guidance} {GUIDANCE_DELETE_SUFFIX})"
        return {"role": "assistant", "text": guidance}

    def is_response_valid(self, json_part, conv):
        anything_else_phrases = ['anything else', 'additional assistance', 'further assistance']
        ret = True
        if re.search(r'https?://', json_part):
            print('\033[91m' + f"WARNING: The generated response contains URLs." + '\033[0m')
            ret = False
        if ("[" in json_part or "<" in json_part):
            print('\033[91m' + f"WARNING: The generated response contains brackets." + '\033[0m')
            ret = False
        if conv.utterance_duplication(json_part):
            print('\033[91m' + f"WARNING: The generated response is a duplicate of the previous utterance." + '\033[0m')
            ret = False
        if 'link' in json_part.lower():
            print('\033[91m' + f"WARNING: The generated response contains 'link'." + '\033[0m')
            ret = False
        if not ("The assistant" in json_part):
            print('\033[91m' + f"WARNING: The generated response does not contain 'The assistant'." + '\033[0m')
            ret = False
        if any([phrase in json_part.lower() for phrase in anything_else_phrases]):
            print('\033[91m' + f"WARNING: The generated response contains 'anything else' type phrases." + '\033[0m')
            ret = False
        if ('The assistant response could include ...' in json_part) or ('The assistant response could include...' in json_part): # whitespace difference
            print('\033[91m' + f"WARNING: The generated response contains dots (...) in the guidance, which is not as intended." + '\033[0m')
            ret = False
        
        if ret == False:
            print('\033[91m' + f'json_part: {json_part}' + '\033[0m')
        return ret

    def response_generation(self, step, conv, topic, flag_include_step_key=True) -> str:
        print('\033[94m' + "======= Guidance generation =======\n" + '\033[0m')
        # prompt_task_setting_str = self.prompt_task_setting_dict[topic]
        prompt_task_setting_str = self.prompt_per_topic[topic]['task_setting']
        prompt_preference_elicitation_str = self.prompt_per_topic[topic]['preference_elicitation']

        response_generation_prompt = lambda step_instructions, history, ptkb, cot, topic: f"""
        You are an advisor, who supports {topic} recommendation assistants to compose the response to the user.
        {step_instructions}

        User's personal preferences:
        {ptkb}

        Conversation:
        {history}
        
        {DELIMITER}
        {cot}
        """.replace('    ', '')

        cot_default = f"""
        Step 1: Explain what the assistant is required to do according to the instructions.
        Step 2: Identify the last user turn number.
        Step 3: Explain the intent of the last user utterance.
        Step 4: What should the assistant's reply be?
        Step 5: Compose very short guidance for the human assistant on how to write a response to the user. The text must start with "The assistant response could include..." NEVER include an example.
        Step 6: Output step 5 result in JSON format {{"advice_to_assistant": "The assistant response could include..."}}.
        Ensure that you distinctly label and delineate Steps 1, 2, 3, 4, 5, and 6. Let's think step by step:
        """.replace('    ', '')

        greeting_step_instructions = f"""
        The assistant collects if the user is interested in {prompt_task_setting_str}.

        NOTE:
        - NEVER generate/recommend {topic}s in this step, focus on greeting in this step.
        - NEVER answer the {topic} preference questions in this step. Instead, focus on greeting and knowing the user's interest in {prompt_task_setting_str}.
        """.replace('    ', '')

        preference_elicitation_step_instructions = lambda preference_name_bullets, flag_at_least_one: f"""
        In this step, the assistant collects information about the user's {prompt_preference_elicitation_str}.
        - Always check the user's preferences and the conversation to avoid asking the same question twice.
        - Ask WHY the user has their preferences in a natural way.
        - Select the best preference(s) to ask next based on the conversation history.
        - NEVER generate/recommend {topic}s in this step but must focus on collecting as much information as possible.

        Continue until:
        - You collect {'at least one' if flag_at_least_one else 'ALL'} of the following preferences:

        PREFERENCES TO COLLECT:
        {preference_name_bullets}

        {DELIMITER}
        """.replace('    ', '')

        cot_elicitation = f"""
        Step 1: Identify the last user turn number.
        Step 2: Explain the intent of the last user utterance.
        Step 3: Which preference(s) should the assistant ask next given the conversation history?
        Step 4: Compose very short guidance for the human assistant on how to write a response to the user. The text must start with "The assistant response could include..." NEVER include an example.
        Step 5: Output step 5 result in JSON format {{"advice_to_assistant": "The assistant response could include..."}}.
        Ensure that you distinctly label and delineate Steps 1, 2, 3, 4, and 5. Let's think step by step:
        """

        recommendation_step_instructions = f"""
        In this step, the assistant recommends the movie based on the user's personal preferences.

        NOTE:
        - The assistant's response should always be concise and short; there is no need to explain the details of the {topic}.
        - The assistant must explain why the assistant is recommending the {topic}.
        """.replace('    ', '')

        cot_recommendation = f"""
        Step 1: Identify the last user turn number.
        Step 2: Explain the intent of the last user utterance.
        Step 3: What should the assistant's reply be?
        Step 4: Compose short guidance for the human assistant on how to write a response to the user. The text must start with "The assistant response could include..." NEVER include an example. NEVER mention specific movie names. Must include "When making recommendations, if necessary, effectively utilize the user's preferences, such as..."
        Step 5: Convert the step 6 result in JSON format {{"advice_to_assistant": "The assistant response could include... When making recommendations, if necessary, effectively utilize the collected user's preferences, such as..."}}.
        Ensure that you distinctly label and delineate Steps 1, 2, 3, 4, and 5. Let's think step by step: 
        """.replace('    ', '')

        followup_step_instructions = f"""
        In this step, the assistant understands WHY the user wants to try or dislikes the {topic}.

        Continue until:
        - The user shares WHY they want to try or dislike the {topic}.
        - The assistant asks follow-up questions to get more information about the user's preferences.
        """.replace('    ', '')

        cot_followup = f"""
        Step 1: Identify the last user turn number.
        Step 2: Explain the intent of the last user utterance.
        Step 3: How to ask WHY the user wants to try or dislike the recommendation?
        Step 4: Compose very short guidance for the human assistant on how to write a response to the user. The text must start with "The assistant response could include..." NEVER include an example.
        Step 5: Output step 5 result in JSON format {{"advice_to_assistant": "The assistant response could include..."}}.
        Ensure that you distinctly label and delineate Steps 1, 2, 3, 4, and 5. Let's think step by step:
        """.replace('    ', '')

        closing_step_instructions = f"""
        In this step, the assistant concludes the conversation.
        - Express gratitude to the user.
        - End the conversation.

        NOTE:
        - Do NOT ask any additional questions; focus on closing the conversation.
        - Do NOT ask questions like "Is there anything else I can help you with?"
        """.replace('    ', '')

        preference_elicitation_step_instructions_subsequent_sessions = lambda preference_name_bullets, flag_at_least_one: f"""
        In this step, the assistant collects information about the user's {prompt_preference_elicitation_str}.
        - Always check the user's personal preferences and the conversation to avoid asking the same question twice.
        - Ask WHY the user has their preferences in a natural way.
        - Select the best preference(s) to ask next based on the conversation history.
        - NEVER generate/recommend {topic}s in this step but must focus on collecting as much information as possible.

        NOTE:
        - Collect ONLY the preferences not previously asked:

        PREFERENCES TO COLLECT:
        {preference_name_bullets}
        """.replace('    ', '')

        if topic == "recipe":
            pref_prompt_must = preference_prompts.RECIPE_MUST
            pref_prompt_should = preference_prompts.RECIPE_SHOULD
            pref_prompt_could = preference_prompts.RECIPE_COULD
            pref_prompt_subsequent_sessions = preference_prompts.RECIPE_SUBSEQUENT_SESSIONS
        elif topic == "movie":
            pref_prompt_must = preference_prompts.MOVIE_MUST
            pref_prompt_should = preference_prompts.MOVIE_SHOULD
            pref_prompt_could = preference_prompts.MOVIE_COULD
            pref_prompt_subsequent_sessions = preference_prompts.MOVIE_SUBSEQUENT_SESSIONS

        # Observed that for movie, LLMs are more susceptible to the order of the preferences; thus, shuffle the order them.
        def shuffle_pref_prompt(pref_prompt):
            pref_prompt_list = pref_prompt.split('\n')
            random.shuffle(pref_prompt_list)
            return '\n'.join(pref_prompt_list)
        
        pref_prompt_must = shuffle_pref_prompt(pref_prompt_must)
        pref_prompt_should = shuffle_pref_prompt(pref_prompt_should)
        pref_prompt_could = shuffle_pref_prompt(pref_prompt_could)
        pref_prompt_subsequent_sessions = shuffle_pref_prompt(pref_prompt_subsequent_sessions)

        # The first, second, third sessions correspond to the different contexts, i.e., breakfast, lunch, dinner
        # In the first session, the assistant ask the preferences in depth, but in the subsequent sessions, the assistant only asks the preferences that they have not asked before
        if step in ["greeting", "greeting_session2", "greeting_session3"]:
            _prompt = response_generation_prompt(greeting_step_instructions, conv.get_hist_with_current_utterance(), "", cot_default, topic)
        elif step == "preference_elicitation_must":
            _preference_elicitation_step_instructions = preference_elicitation_step_instructions(pref_prompt_must, flag_at_least_one=False)
            _prompt = response_generation_prompt(_preference_elicitation_step_instructions, conv.get_hist_with_current_utterance(), conv.get_ptkb_str(), cot_elicitation, topic)
        elif step == "preference_elicitation_should":
            _preference_elicitation_step_instructions = preference_elicitation_step_instructions(pref_prompt_should, flag_at_least_one=True)
            _prompt = response_generation_prompt(_preference_elicitation_step_instructions, conv.get_hist_with_current_utterance(), conv.get_ptkb_str(), cot_elicitation, topic)
        elif step == "preference_elicitation_could":
            _preference_elicitation_step_instructions = preference_elicitation_step_instructions(pref_prompt_could, flag_at_least_one=True)
            _prompt = response_generation_prompt(_preference_elicitation_step_instructions, conv.get_hist_with_current_utterance(), conv.get_ptkb_str(), cot_elicitation, topic)
        elif step in ["preference_elicitation_session2", "preference_elicitation_session3"]:
            _preference_elicitation_step_instructions = preference_elicitation_step_instructions_subsequent_sessions(pref_prompt_subsequent_sessions, flag_at_least_one=True)
            _prompt = response_generation_prompt(_preference_elicitation_step_instructions, conv.get_hist_with_current_utterance(), conv.get_ptkb_str(), cot_default, topic)
        elif step in ["recommendation", "recommendation_session2", "recommendation_session3",]:
            # recipe_no_str = step.split('_')[0] # first, second, third
            _prompt = response_generation_prompt(recommendation_step_instructions, conv.get_hist_with_current_utterance(), conv.get_ptkb_str(), cot_recommendation, topic) # use special cot for recommendation
        elif step in ["recommendation_followup", "recommendation_followup_session2", "recommendation_followup_session3"]:
            # recipe_no_str = step.split('_')[0] # first, second, third
            _prompt = response_generation_prompt(followup_step_instructions, conv.get_hist_with_current_utterance(), conv.get_ptkb_str(), cot_followup, topic)
        elif step in ["closing", "closing_session2", "closing_session3"]:
            _prompt = response_generation_prompt(closing_step_instructions, conv.get_hist_with_current_utterance(), conv.get_ptkb_str(), cot_default, topic)
        else:
            raise ValueError(f"Invalid step: {step}")


        n_max_trial = 5
        _backup_response = None # backup response to avoid crucial errors such as json parsing / non-json response
        json_part = None
        for i in range(n_max_trial):

            # from the second attempt, set the temperature [0,1] and frequency_penalty [0,2] randomly
            # (temperature, frequency_penalty) = (1.0, 0.5) if i == 0 else (random.uniform(0, 1), random.uniform(0, 2))
            (temperature, frequency_penalty) = (0.6, 0.0) # frequency_penalty is 0.0 to make the model follow instructions strictly (e.g., it does not make sense if output False just because there is "False" value shown in the previous reasoning step)
            response = openai_api.get_completion_from_prompt(_prompt, verbose=VERBOSE, temperature=temperature, frequency_penalty=frequency_penalty)
            try:
                extracted_dict = utils.extract_json(response)
                if extracted_dict is not None:
                    extracted_dict = self.postprocess(extracted_dict, step, topic) # convert `{"advice_to_assistant": <guidance>}` to `{"role": "assistant", "text": <text>}` to make it compatible with Candidate
                    json_part = json.dumps({**extracted_dict, 'step': step}) if flag_include_step_key else json.dumps(extracted_dict) # add step key to the AI response
                else:
                    print(f"WARNING: GPT-3 generates text instead of JSON or invalid JSON. Retrying {i+1}/{n_max_trial}...")
                    continue
            except:
                print('\033[91m' + f"WARNING: JSON parsing error. Retrying {i+1}/{n_max_trial}..." + '\033[0m')
                continue
            if self.is_response_valid(json_part, conv):
                break
            else:
                print('\033[91m' + f"WARNING: The generated response does not meet the criteria, such as containing URLs, etc. Retrying {i+1}/{n_max_trial}..." + '\033[0m')
                _backup_response = json_part
                continue

        return json_part if utils.check_json_validity(json_part) else _backup_response # return the backup response if the response is not valid JSON


