from flask import Flask, request, jsonify
from assistant import RecipeChat
from firebase_manager import FirebaseManager
import json
import pickle
import os
import time
import logger_settings
import config

app = Flask(__name__)
fb_manager = FirebaseManager()

PATH_TO_CHATS_PKL = './data/chats.pkl'
os.makedirs(os.path.dirname(PATH_TO_CHATS_PKL), exist_ok=True)

TOPIC = config.TOPIC

# Attempt to load chats from a pickle file
try:
    with open(PATH_TO_CHATS_PKL, 'rb') as file:
        chats = pickle.load(file)
except (FileNotFoundError, EOFError):
    chats = {}

last_request_time = {}
WAIT_TIME = 5

logger = logger_settings.get_logger('browserRefresh')

def get_recent_assistant_utterance(conv_id):
    """Get the most recent assistant utterance from Firebase."""
    try:
        return fb_manager.get_recent_assistant_message(conv_id)
    except Exception as e:
        print(e)
        return None            

def handle_browser_refresh(conv_id, user_utterance, chats, current_time):
    global last_request_time
    #  This means no response is generated yet since conv_id is not in chats
    if conv_id not in chats:
        logger.info('No response generated yet for conversation ID: %s', conv_id)
        return None
    
    chat = chats[conv_id]

    if chat.role == 'system':
        # Ignore system messages since it does not use OpenAI API
        return None
    
    # Handle *frequent* refresh
    if (conv_id in last_request_time) and (current_time - last_request_time[conv_id] < WAIT_TIME):
        # NOTE: you should ignore 'system' since it is just yes&no and could be within WAIT_TIME
        time.sleep(WAIT_TIME)
        logger.info('Request ignored due to frequent refresh for conversation ID: %s', conv_id)
        return jsonify(status='error', message='Request ignored due to frequent refresh', data=None)

    last_dialogue = chat.conv.conversation_history[-1]
    second_last_dialogue = chat.conv.conversation_history[-2] if len(chat.conv.conversation_history) > 1 else None

    # If there is a duplicate user utterance, regenerate the response
    if last_dialogue['role'] == 'user' and user_utterance == last_dialogue['text']:
        response_json, step_index = chat.get_response()
        response_data = json.loads(response_json) if response_json else None
        assert response_data is not None, "The response generation should not return None"
        assert 'role' in response_data and 'text' in response_data, "Response data must have 'role' and 'text' keys"
        logger.info('Regenerated response for duplicate user utterance for conversation ID: %s', conv_id)
        return jsonify(status='success', message='Processed successfully', data={
            'role': response_data['role'],
            'message': response_data['text'],
            'stepNo': step_index
        })

    # If the last response was not sent to the client, return the last assistant response
    elif second_last_dialogue and user_utterance == second_last_dialogue['text'] and second_last_dialogue['role'] == 'user':
        logger.info('Returning last assistant response for conversation ID: %s', conv_id)
        return jsonify(status='success', message='Processed successfully', data={
            'role': last_dialogue['role'],
            'message': last_dialogue['text'],
            'stepNo': chat.step_index
        })

    # If no conditions are met, return None indicating that the normal process should continue
    return None


@app.route('/assistant', methods=['POST'])
def assistant():
    global chats
    global last_request_time
    try:
        data = request.json
        conv_id = data['conversationID']

        # Fetching latest user and assistant message from Firebase
        user_utterance = fb_manager.get_recent_user_message(conv_id)

        # Error handling: user refreshes the page
        current_time = time.time()
        refresh_response = handle_browser_refresh(conv_id, user_utterance, chats, current_time)
        if refresh_response: # If refresh_response is not None, it means we have a response to return due to a browser refresh
            return refresh_response
        # if not, then continue the normal process...
        last_request_time[conv_id] = current_time

        if user_utterance is None:
            if conv_id in chats: 
                return jsonify(status='error', message='No user message was found in Firebase even though this is not the first turn.', data=None)
            else: # initial turn start with a greeting from assistant
                user_utterance = ""

        # Handling assistant's turn & creating instance of RecipeChat
        recent_assistant_utterance = get_recent_assistant_utterance(conv_id)
        if conv_id not in chats: # first turn
            chats[conv_id] = RecipeChat(conv_id, TOPIC, initial_assistant_utterance=recent_assistant_utterance) # create instance of RecipeChat with initial assistant utterance
        else: # not first turn; replace most recent assistant utterance in case crowd worker rewrote it
            if recent_assistant_utterance:
                chats[conv_id].replace_most_recent_assistant_utterance(recent_assistant_utterance)

        chat = chats[conv_id]
        chat.set_user_utterance(user_utterance)  # Set user message fetched from Firebase

        # Main: generate assistant response
        asst_response_json, step_index = chat.get_response()

        # Extracting text and role from JSON string
        dc = json.loads(asst_response_json)
        asst_response_text = dc['text']
        role = dc['role']

        response = {
            "status": "success",
            "message": "Processed successfully",
            "data": {
                "role": role,  # role can be "assistant" or "system"
                "message": asst_response_text,
                "stepNo": step_index
            }
        }

        # Save chats to a pickle file
        with open(PATH_TO_CHATS_PKL, 'wb') as file:
            pickle.dump(chats, file)

        return jsonify(response)

    except Exception as e:
        logger.exception(f'Error in assistant(): {str(e)} for conversation ID: {conv_id} - llm_server.py')
        return jsonify(status='error', message=str(e), data=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007, debug=True)
