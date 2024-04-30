import openai_api
import preference_prompts
import config
import re
import utils
DELIMITER = config.DELIMITER
VERBOSE = config.VERBOSE

def _extract_json_part(new_pt):
    """Extract the last PT from the user's response
    E.g., "Output: [first] (some text) [second] (more text)" --> "[second]"
    """
    new_pt = re.sub(r'\n+', ' ', new_pt).strip()
    matches = re.findall(r'(\{.*?\})', new_pt) # The latter part handle the case '{"text": "The user has no allergies", "preference_name": "allergies", "turn_number": 2}'
    new_pt = matches[-1] if matches else ''

    return new_pt

def _ptkb_preprocess(new_pt_dc: dict, topic):
    # E.g., {"allergies": "No allergies", ...} --> {"allergies": ["No allergies"], ...}
    return {key: [new_pt_dc[key]] if type(new_pt_dc[key]) == str else new_pt_dc[key] for key in new_pt_dc}

def _ptkb_key_error_check(new_pt_dc: dict, topic):
    if topic == 'recipe':
        preference_property_dict = preference_prompts.RECIPE_PREFERENCE_PROPERTY_DICT
    elif topic == 'movie':
        preference_property_dict = preference_prompts.MOVIE_PREFERENCE_PROPERTY_DICT
    else:
        raise ValueError("Invalid topic: {topic}.")
        
    categories = preference_property_dict.keys()
    for key in new_pt_dc:
        assert key in categories, f"Invalid preference category: {key}"

def extract(hist_with_current_utterance, topic):
    print('\033[94m' + "======= Update PTKB =======\n" + '\033[0m')
    # preference_list_for_extraction
    if topic == 'recipe':
        preference_list_for_extraction = preference_prompts.RECIPE_PREFERENCE_LIST_FOR_EXTRACTION
    elif topic == 'movie':
        preference_list_for_extraction = preference_prompts.MOVIE_PREFERENCE_LIST_FOR_EXTRACTION
    else:
        raise ValueError("Invalid topic: {topic}.")

    prompt_ptkb_extraction = lambda recent_conversation, preference_list_for_extraction: f"""
    You are a personal preferences extractor. Your goal is to extract the user's personal information from the conversation. 
    Each "personal preference" is a summary of the user's personal information. It is a list of JSON objects,
    - key: preference category.
    - value (list): preference information.

    Preference categories:
    ```
    {preference_list_for_extraction}
    ```

    Example output:
    ```
    {{"allergies": ["No allergies"], "diet_requirements": ["vegan", ], ...}}
    ```

    {DELIMITER}

    Conversation:
    {recent_conversation}

    {DELIMITER}

    Step 1: Identify all turns where the user shares their preferences; identify as many as possible.
    Step 2: For each identified turn, extract user preferences.
    Step 3: Output in JSON format following the example above (i.e., `{{...}}`).
    Ensure that you distinctly label and delineate Steps 1, 2, and 3. Let's think step by step: 
    """.replace('    ', '')

    _prompt = prompt_ptkb_extraction(hist_with_current_utterance, preference_list_for_extraction)
    
    n_max_trial = 5
    last_valid_new_pt = None
    for i in range(n_max_trial):
        # Step 1: Obtain a response from the OpenAI API and extract the JSON part from the response
        new_pt = openai_api.get_completion_from_prompt(_prompt, model="gpt-4-1106-preview", temperature=0.6, frequency_penalty=0.0, verbose=VERBOSE).strip()
        new_pt = _extract_json_part(new_pt)
        try:
            # Step 2: Convert new_pt from a string to a dictionary and preprocess it to the required format
            new_pt = utils._to_dict(new_pt)
            new_pt = _ptkb_preprocess(new_pt, topic)

            # Step 3: Store the currently processed valid response
            # Note: `_ptkb_key_error_check` is less important than `_ptkb_preprocess` 
            #   because it checks the validity of the keys in the response
            #   while `_to_dict` and `_ptkb_preprocess` checks the validity of JSON format.
            last_valid_new_pt = new_pt

            # Step 4: Check if the keys in the response are valid
            _ptkb_key_error_check(new_pt, topic)

            return new_pt
        except Exception as e:
            print("\033[91m" + f"Invalid response. Try again. ({i+1}/{n_max_trial})\n{new_pt}" + "\033[0m")
            print('Error message:', e)

    # Step 5: Return the last valid processed response if the maximum number of trials is reached without success
    return last_valid_new_pt