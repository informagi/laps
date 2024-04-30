import preference_prompts
import config
import openai_api
import utils


DELIMITER = config.DELIMITER
VERBOSE = config.VERBOSE

# @try_n_times_decorator(n=3)
def step_completion_detection(step, conv, topic):
    if topic == 'recipe':
        prompt_must = preference_prompts.RECIPE_MUST
        prompt_should = preference_prompts.RECIPE_SHOULD
        prompt_could = preference_prompts.RECIPE_COULD
    elif topic == 'movie':
        prompt_must = preference_prompts.MOVIE_MUST
        prompt_should = preference_prompts.MOVIE_SHOULD
        prompt_could = preference_prompts.MOVIE_COULD
    else:
        raise ValueError("Invalid topic: {topic}.")

    print('\033[94m' + "======= Step Completion Detection =======\n" + '\033[0m')

    step_completion_detection_prompt = lambda history, step_completion_criteria: f"""

    {DELIMITER}
    Conversation:
    {history}
    {DELIMITER}

    {step_completion_criteria}
    The answer should be JSON, either {{"final_decision": true}} or {{"final_decision": false}}.
    Step 1: Summarize what you should perform.
    Step 2: Examine each criterion.
    Step 3: Final decision in JSON format.
    Ensure that you distinctly label and delineate Steps 1, 2, and 3. Let's think step by step: 
    """.replace('    ', '')

    greeting_step_completion_criteria_recipe = f"""
    Does the user clearly shared whether they are interested in breakfast, lunch, or dinner?
    """.replace('    ', '').strip()

    greeting_step_completion_criteria_movie = f"""
    Does the user clearly shared their movie viewing occasion?
    """.replace('    ', '').strip()

    preference_elicitation_must_step_completion_criteria = f"""
    ```
    {prompt_must}
    ```
    Have all of the preferences above been confirmed by the assistant?
    """.replace('    ', '').strip()

    preference_elicitation_should_step_completion_criteria = f"""
    ```
    {prompt_should}
    ```
    Is at least one of these preferences shared by the user?
    """.replace('    ', '')

    preference_elicitation_could_step_completion_criteria = f"""
    ```
    {prompt_could}
    ```
    (C1) Has the user shared any of the preferences listed above?
    (C2) Has the assistant collected why the user has the preference?
    If both (C1) and (C2) are true, return true.
    """.replace('    ', '')

    recommendation_step_completion_criteria = f"""
    (C) If the user is satisfied with the recommendation, return true. Otherwise, return false.
    """.replace('    ', '')

    recommendation_followup_step_completion_criteria = f"""
    (C1) The user shared the reason WHY they like or dislike the recommendation.
    (C2) The assistant asks follow-up questions and user shared more information about their preferences.
    If both (C1) and (C2) are true, return true.
    """.replace('    ', '')

    closing_step_completion_criteria = f"""
    (C1) The assistant has expressed gratitude to the user.
    (C2) The assistant has ended the conversation.
    If both (C1) and (C2) are true, return true.
    """.replace('    ', '')

    hist = conv.get_hist_with_current_utterance()

    # before calling the API, check if conversation history is empty --> if so, return False
    if len(hist) == 0:
        return False

    if step in ["greeting", "greeting_session2", "greeting_session3"]:
        if topic == 'recipe':
            greeting_step_completion_criteria = greeting_step_completion_criteria_recipe
        elif topic == 'movie':
            greeting_step_completion_criteria = greeting_step_completion_criteria_movie
        _prompt = step_completion_detection_prompt(hist, step_completion_criteria=greeting_step_completion_criteria)
    elif step == "preference_elicitation_must":
        _prompt = step_completion_detection_prompt(hist, step_completion_criteria=preference_elicitation_must_step_completion_criteria)
    elif step == "preference_elicitation_should":
        _prompt = step_completion_detection_prompt(hist, step_completion_criteria=preference_elicitation_should_step_completion_criteria)
    elif step in ["preference_elicitation_could", "preference_elicitation_session2", "preference_elicitation_session3"]:
        _prompt = step_completion_detection_prompt(hist, step_completion_criteria=preference_elicitation_could_step_completion_criteria)
    elif step in ["recommendation", "recommendation_session2", "recommendation_session3"]:
        # recipe_no_str = step.split('_')[0] # first, second, third
        _prompt = step_completion_detection_prompt(hist, step_completion_criteria=recommendation_step_completion_criteria)
    elif step in ["recommendation_followup", "recommendation_followup_session2", "recommendation_followup_session3"]:
        # recipe_no_str = step.split('_')[0] # first, second, third
        _prompt = step_completion_detection_prompt(hist, step_completion_criteria=recommendation_followup_step_completion_criteria)
    elif step in ["closing", "closing_session2", "closing_session3"]:
        _prompt = step_completion_detection_prompt(hist, step_completion_criteria=closing_step_completion_criteria)
    else:
        raise ValueError("Invalid step: {step}.")

    n_max_trial = 5
    for i in range(n_max_trial):
        tf_str = openai_api.get_completion_from_prompt(_prompt, temperature=0.6, frequency_penalty=0.0, verbose=VERBOSE).strip() # either 'True' or 'False' in str format will be returned
        json_ = utils.extract_json(tf_str)
        if json_ is not None and 'final_decision' in json_:
            return json_['final_decision']
        else:
            print('\033[91m' + f"WARNING: Invalid JSON. Retrying {i+1}/{n_max_trial}..." + '\033[0m')
