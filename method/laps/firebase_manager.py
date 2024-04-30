import argparse
import datetime
import json
import os
import logging
import config
import logger_settings

import firebase_admin
from firebase_admin import firestore

PROJECT_DIR = config.PROJECT_DIR

# https://cloud.google.com/python/docs/reference/firestore

class FirebaseManager:
    def __init__(self):
        # set GOOGLE_APPLICATION_CREDENTIALS env var to path to .json private key file and 
        # this should automatically work
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'your_firebase_api_key.json'
        try:
            firebase_admin.get_app()
        except ValueError as e:
            firebase_admin.initialize_app()

        self.db = firestore.client()

        # top level collection structure/naming
        self.collection_name = 'TaskMAD/WizardOfOz/conversations'

        # get a reference to the above collection
        self.c = self.db.collection(self.collection_name)

        # Configure logging
        self.logger = logger_settings.get_logger('FirebaseManager')

    def _log_to_file(self, message):
        # Log the message to the file
        self.logger.info(message)

    def _get_recent_message(self, conversation_id, filter_func):
        # Get a reference to the conversation document
        d = self.c.document(conversation_id).collection('messages')  # Assuming 'messages' is your sub-collection

        # Query for the most recent message
        query = d.order_by('timestamp', direction=firestore.Query.DESCENDING)

        # Log the attempt to retrieve a message
        self._log_to_file(f"Attempting to retrieve messages for conversation_id: {conversation_id}")

        # Execute the query
        results = query.stream()

        # Iterate over the messages
        for result in results:
            message = result.to_dict()
            if filter_func(message):
                self._log_to_file(f"Message retrieved for conversation_id: {conversation_id}")
                return message['interaction_text']
        
        self._log_to_file(f"No message found for conversation_id: {conversation_id}")
        return None  # No message was found

    def warn_if_id_unexpected(self, user_id):
        # user_id should be lowercased
        if not (user_id.startswith("assistant") or user_id.startswith("user") or user_id.startswith("system")):
            warning_msg = f"Warning: user_id {user_id} does not start with 'assistant', 'user', or 'system'."
            self._log_to_file(warning_msg)
            print(warning_msg)

    def get_recent_user_message(self, conversation_id):
        def filter_func(message):
            user_id = message["user_id"].lower()
            text = message["interaction_text"].lower()
            self.warn_if_id_unexpected(user_id)
            return not user_id.startswith("assistant") and not text.startswith("user moved to section") and not text.endswith("joined the chat.")

        return self._get_recent_message(conversation_id, filter_func)

    def get_recent_assistant_message(self, conversation_id):
        def filter_func(message):
            user_id = message["user_id"].lower()
            self.warn_if_id_unexpected(user_id)
            return user_id.startswith("assistant") and len(message['interaction_text']) > 0

        return self._get_recent_message(conversation_id, filter_func)