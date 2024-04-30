Recommendation Prompts
======================

We use Llama-2 7b for the recommendation task with zero-shot prompting.



# Memory method

## Prompt code snippet
```py
# Memory prompting method
def get_prompt_with_preferences(worker_id, session_ind, turn_num_end, topic):
    """
    Args:
    - worker_id: str, worker id
    - session_ind: int, index of the session (0-indexed)
    - turn_num_end: int, the last turn number to include in the dialogue
    - topic: str, the topic of the recommendation. either 'movie' or 'recipe'
    
    Return:
    - prompt: str, the prompt for the assistant to generate the next turn

    Functions:
    - get_dialogue(): returns the dialogue from LAPS dataset for the given worker_id and session_ind
    - get_dialogue_string(): returns the dialogue as a string. The format is shown in the sample prompt below.
    """
    dialogue = get_dialogue(worker_id, session_ind, turn_num_end)
    dialogue_string = get_dialogue_string(dialogue)
    preference_string = get_preferences_string(worker_id, session_ind)

    prompt = f"""
    You are a professional {topic} recommendation assistant.
    In this step, you recommend a {topic} based on the user's personal preferences and the conversation history.
    You must explain the reason for your recommendation.

    User's personal preferences:
    {preference_string}

    Conversation history:
    {dialogue_string}

    Write the next assistant turn to recommend a {topic}:
    """.replace('        ', '').strip()

    return prompt
```

## Sample
```
You are a professional movie recommendation assistant.
In this step, you recommend a movie based on the user's personal preferences and the conversation history.
You must explain the reason for your recommendation in detail.

User's personal preferences:
- genre_preferences_like: romantic comedy; mystery
- actor_like: Tom Hanks; Tom Hanks

Conversation history:
User: Hello
Asst: Welcome can i ask the reason for your visit today and how I can help
User: Hi i would like an option of a family friendly movie to watch with young children whilst travelling so needs to be downloaded
Asst: is there a theme or a specific interest that you think would be helpful to consider in choices
User: something that is a cartoon light and fun to watch
Asst: Is this film to be linked towards a seasonal event? what is your preferred platform to watch on
User: no special seasonal event but Disney plus as its for travel so will not have access to internet

Write the next assistant turn to recommend a movie:
```

# Standard method

## Prompt code snippet

```py
# Standard prompting method
def get_prompt_without_preferences(worker_id, session_ind, turn_num_end, topic):
    dialogue_string = ''
    # for all previous sessions including the current session (note ind is 0-indexed)
    for s_ind in range(session_ind + 1):
        dialogue = get_dialogue(worker_id, s_ind, turn_num_end) if s_ind == session_ind else get_dialogue(worker_id, s_ind, None)
        dialogue_string += get_dialogue_string(dialogue) + '\n'
    
    prompt = f"""
    You are a professional {topic} recommendation assistant.
    In this step, you recommend a {topic} based on the conversation history.
    You must explain the reason for your recommendation.

    Conversation history:
    {dialogue_string}

    Write the next assistant turn to recommend a {topic}:
    """.replace('        ', '').strip()

    return prompt
```

## Sample
```
You are a professional movie recommendation assistant.
In this step, you recommend a movie based on the user's conversation history.
You must explain the reason for your recommendation.

Conversation history:
Asst: Hello there! Allow me to be your personal movie advisor, recommending films you'll love.
User: Hello looking for a film to watch alone
Asst: Do you have any age restrictions to be aware of and which movie genre do you feel like watching
User: I am an adult and would prefer something light to watch such as romantic comedy with good ratings
Asst: Do you have any preferred actors that you enjoy watching
User: I enjoy films with Tom Hanks
Asst: what services or subscriptions do you have access to in order to watch films of your choice
User: i have access to Sky, Disney plus, Paramount Plus, Amazon and Netflix
Asst: here are some options with Tom Hanks Forrest Gump https://www.imdb.com/title/tt0109830/?ref_=nm_knf_c_3  Big https://www.imdb.com/title/tt0094737/?ref_=nm_knf_c_2  You've got mail https://www.imdb.com/title/tt0128853/?ref_=nm_flmg_c_59_act
User: These are all excellent options with my favourite actor and the genre i would like
Asst: Great pleased that these options give you the actor that you requested and are the type of film you had wanted to watch
User: Thank you these are good choices to watch on my own or with family
Asst: Pleased to have offered a service you have found to be of use. Enjoy watching and look forward to help again the future
User: thank you goodbye
User: done
Asst: Hello do you have any new requests for film options
User: Hello yes would like Tom Hanks films but a more serious nature
Asst: Is there any specific genres within the films that you would wish to see such as criminal or war themes
User: Perhaps a mystery type with some criminal element
Asst: https://www.imdb.com/title/tt0382625/?ref_=nm_flmg_t_45_act  The Da VInci code is likely to meet your expectations of a film which has Tom Hanks and has an element of mystery
User: Oh thank you this is a really different type of film from Tom Hanks. It is a film that makes you think
Asst: Is there anything else that helps you to know that this film is suitable for your preferences
User: It has relatively good ratings and is a good critic choice
Asst: Thank you for your time and feedback to improve the outcomes for users in the future. Enjoy your film thank you
User: Thank you
User: Hello
Asst: Welcome can i ask the reason for your visit today and how I can help
User: Hi i would like an option of a family friendly movie to watch with young children whilst travelling so needs to be downloaded
Asst: is there a theme or a specific interest that you think would be helpful to consider in choices
User: something that is a cartoon light and fun to watch
Asst: Is this film to be linked towards a seasonal event? what is your preferred platform to watch on
User: no special seasonal event but Disney plus as its for travel so will not have access to internet


Write the next assistant turn to recommend a movie:
```