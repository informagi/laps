
pref_map_recipe_flan = {
    'allergies': 'What is the user\'s allergies?',
    'diet_requirements': 'What is the user\'s diet requirements?',
    'skill_level': 'What is the user\'s cooking skill level?',
    'time_availability': 'How much time does the user have for cooking?',
    'dish_like': 'What type of dish does the user like?',
    'dish_dislike': 'What type of dish does the user dislike?',
    'cuisine_like': 'What cuisine does the user like?',
    'cuisine_dislike': 'What cuisine does the user dislike?',
    'ingredient_like': 'What ingredient does the user like?',
    'ingredient_dislike': 'What ingredient does the user dislike?',
    'seasonal_ingredients': 'What seasonal ingredients does the user like?',
    'companions': 'Who will the user be dining with?',
    'companion_preferences_like': 'What do the user\'s companions like to eat?',
    'companion_preferences_dislike': 'What do the user\'s companions dislike to eat?',
    'health_goals': 'What is the user\'s health goals?',
    'occasion_or_theme': 'What is the occasion or theme for the meal?',
    'ingredient_availability': 'What is the availability of ingredients?'
}

pref_map_movie_flan = {
    'content_restrictions': 'What are the user\'s content restrictions?',
    'genre_preferences_like': 'What are the user\'s preferred movie genres?',
    'genre_preferences_dislike': 'What are the user\'s disliked movie genres?',
    'frequently_watched': 'What types of movies does the user frequently watch?',
    'viewing_companions': 'Who is the user watching the movie with?',
    'cultural_preferences_like': 'What are the user\'s current cultural or regional film preferences?',
    'cultural_preferences_dislike': 'What are the user\'s disliked cultural or regional film preferences?',
    'actor_like': 'What specific actors or directors does the user enjoy?',
    'actor_dislike': 'What specific actors or directors does the user avoid?',
    'movie_like': 'What specific movies does the user enjoy?',
    'movie_dislike': 'What specific movies does the user avoid?',
    'seasonal_themes': 'What are the user\'s preferences for seasonal or timely themes in movies?',
    'companion_preferences_like': 'What are the movie preferences that the user\'s companions like?',
    'companion_preferences_dislike': 'What are the movie preferences that the user\'s companions dislike?',
    'special_occasion': 'What is the special occasion or theme for which the user wants a movie recommendation?',
    'release_period_preferences': 'What are the user\'s preferences for movies from certain periods, e.g., classics, modern era?',
    'users_mood': 'What is the current mood of the user which may influence the type of movie they wish to watch?',
    'platform_availability': 'What are the user\'s available streaming platforms?'
}

def get_dialogue_string(dialogue):
    dialogue_string = ""
    role_map = {'Assistant': 'Asst', 'User': 'User'}
    for d in dialogue:
        dialogue_string += f"{role_map[d['role']]}: {d['message']}\n"
    return dialogue_string.strip()


if __name__ == '__main__':

    dialogue_string = 'foo' # use get_dialogue_string(dialogue) to get the dialogue string. Use the corresponding dialogue in the LAPS dataset for the input `dialogue`.
    topic = 'recipe' # set this to 'recipe' or 'movie'
    preference_category = 'allergies' # set this to the preference category you want to extract

    if topic == 'recipe':
        pref_map = pref_map_recipe_flan
    elif topic == 'movie':
        pref_map = pref_map_movie_flan
    question_string = pref_map[preference_category]

    input_instruction_template = f"""
    Please answer the following question based on the given dialogue.
    ####
    {dialogue_string}
    ####
    Q: {question_string}
    """.replace('        ', '').strip()

    print(input_instruction_template)