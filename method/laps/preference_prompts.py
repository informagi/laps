
RECIPE_MUST = """
- `allergies`: User's food allergies.
- `diet_requirements`: User's dietary requirements, e.g., vegetarian, vegan, gluten-free, low carb diets, or any other dietary specifics.
""".strip()

RECIPE_SHOULD = """
- `skill_level`: User's culinary proficiency, e.g., beginner, intermediate, or advanced cook.
- `time_availability`: User's time availability for cooking. (short-term)
""".strip()

RECIPE_COULD = """
- `weekly_dishes`: Types of dishes the user frequently cooks each week.
- `cuisine_preferences`: User's current cuisine preferences, e.g., Italian, Mexican, Chinese, or any other cuisine type.
- `ingredient_preferences`: Specific flavors, ingredients, or spices the user enjoys or dislikes.
- `seasonal_ingredients`: Seasonal ingredients.
- `companions`: Whether the user is cooking for themselves or others, e.g., son, daughter, spouse, etc.
- `companion_preferences`: Food preferences of the user's companions.
- `health_goals`: User's health goals or dietary restrictions.
- `occasion_or_theme`: Special occasion or theme for which the user needs a recipe.
- `ingredient_availability`: Specific ingredients the user currently has on hand.
""".strip()

RECIPE_SUBSEQUENT_SESSIONS = """
- `time_availability`: User's time availability for cooking. (short-term)
- `weekly_dishes`: Types of dishes the user frequently cooks each week.
- `cuisine_preferences`: User's current cuisine preferences, e.g., Italian, Mexican, Chinese, or any other cuisine type.
- `ingredient_preferences`: Specific flavors, ingredients, or spices the user enjoys or dislikes.
- `seasonal_ingredients`: Seasonal ingredients.
- `companions`: Whether the user is cooking for themselves or others, e.g., son, daughter, spouse, etc.
- `companion_preferences`: Food preferences of the user's companions.
- `health_goals`: User's health goals or dietary restrictions.
- `occasion_or_theme`: Special occasion or theme for which the user needs a recipe.
- `ingredient_availability`: Specific ingredients the user currently has on hand.
""".strip()


# This explicitly separate likes and dislikes
RECIPE_PREFERENCE_LIST_FOR_EXTRACTION = """
- `allergies`
- `diet_requirements`
- `skill_level`
- `time_availability`
- `dish_like`
- `dish_dislike`
- `cuisine_like`
- `cuisine_dislike`
- `ingredient_like`
- `ingredient_dislike`
- `seasonal_ingredients`
- `companions`
- `companion_preferences_like`
- `companion_preferences_dislike`
- `health_goals`
- `occasion_or_theme`
- `ingredient_availability`
"""

# This is for both preference extraction and removing the short-term preferences in prompts in recommendation
RECIPE_PREFERENCE_PROPERTY_DICT = {
    'allergies': {'duration': 'long-term', 'importance': 'must'},
    'diet_requirements': {'duration': 'long-term', 'importance': 'must'},
    'skill_level': {'duration': 'long-term', 'importance': 'should'},
    'time_availability': {'duration': 'short-term', 'importance': 'should'},
    'dish_like': {'duration': 'long-term', 'importance': 'could'},
    'dish_dislike': {'duration': 'long-term', 'importance': 'could'},
    'cuisine_like': {'duration': 'long-term', 'importance': 'could'},
    'cuisine_dislike': {'duration': 'long-term', 'importance': 'could'},
    'ingredient_like': {'duration': 'long-term', 'importance': 'could'},
    'ingredient_dislike': {'duration': 'long-term', 'importance': 'could'},
    'seasonal_ingredients': {'duration': 'long-term', 'importance': 'could'},
    'companions': {'duration': 'long-term', 'importance': 'could'},
    'companion_preferences_like': {'duration': 'long-term', 'importance': 'could'},
    'companion_preferences_dislike': {'duration': 'long-term', 'importance': 'could'},
    'health_goals': {'duration': 'long-term', 'importance': 'could'},
    'occasion_or_theme': {'duration': 'short-term', 'importance': 'could'},
    'ingredient_availability': {'duration': 'short-term', 'importance': 'could'},
}

# =====================================
# ===== MOVIE Preferences =============
# =====================================

MOVIE_MUST = """
- `content_restrictions`: User's content restrictions that they do not want to see.
""".strip()

MOVIE_SHOULD = """
- `genre_preferences`: User's preferred movie genres.
- `viewing_companions`: Whether the user is watching alone or with others.
""".strip()

MOVIE_COULD = """
- `frequently_watched`: Types of movies the user frequently watches and why.
- `cultural_preferences`: User's current cultural or regional film preferences.
- `actor_preferences`: Specific actors or directors the user enjoys or avoids.
- `seasonal_themes`: Seasonal or timely themes in movies.
- `companion_preferences`: Movie preferences of the users companions.
- `special_occasion`: Special occasion or theme for which the user wants a movie recommendation.
- `release_period_preferences`: User's preferences for movies from certain periods, e.g., classics, modern era.
- `users_mood`: The current mood of the user which may influence the type of movie they wish to watch.
- `platform_availability`: User's available streaming platforms.
""".strip()

MOVIE_SUBSEQUENT_SESSIONS = """
- `viewing_companions`: Whether the user is watching alone or with others.
- `companion_preferences`: Movie preferences of the users companions.
- `frequently_watched`: Types of movies the user frequently watches and why.
- `cultural_preferences`: User's current cultural or regional film preferences.
- `actor_preferences`: Specific actors or directors the user enjoys or avoids.
- `seasonal_themes`: Seasonal or timely themes in movies.
- `special_occasion`: Special occasion or theme for which the user wants a movie recommendation.
- `release_period_preferences`: User's preferences for movies from certain periods, e.g., classics, modern era.
- `users_mood`: The current mood of the user which may influence the type of movie they wish to watch.
- `platform_availability`: User's available streaming platforms.
""".strip()

# This explicitly separate likes and dislikes
# NOTE: For recipe, I excluded the description of the preference category. This is because, it seems to be higher performance and also can reduce the number of input tokens.
# However, for movie, I noticed that huge extraction quality drop if descriptions are excluded. Thus, here I include them.
# Sideeffect: The GPT-4 less likely to follow the output json format... (which didn't happen at all without the descriptions)
# --> modify CoT to handle this
MOVIE_PREFERENCE_LIST_FOR_EXTRACTION = """
- `content_restrictions`: User's content restrictions.
- `genre_preferences_like`: User's preferred movie genres.
- `genre_preferences_dislike`: User's disliked movie genres.
- `frequently_watched`: Types of movies the user frequently watches.
- `viewing_companions`: Whether the user is watching alone or with others.
- `cultural_preferences_like`: User's current cultural or regional film preferences.
- `cultural_preferences_dislike`: User's disliked cultural or regional film preferences.
- `actor_like`: Specific actors or directors the user enjoys.
- `actor_dislike`: Specific actors or directors the user avoids.
- `movie_like`: Specific movies the user enjoys.
- `movie_dislike`: Specific movies the user avoids.
- `seasonal_themes`: User's preferences for seasonal or timely themes in movies.
- `companion_preferences_like`: Movie preferences of the users companions.
- `companion_preferences_dislike`: Movie preferences of the users companions.
- `special_occasion`: Special occasion or theme for which the user wants a movie recommendation.
- `release_period_preferences`: User's preferences for movies from certain periods, e.g., classics, modern era.
- `users_mood`: The current mood of the user which may influence the type of movie they wish to watch.
- `platform_availability`: User's available streaming platforms.
"""


MOVIE_PREFERENCE_PROPERTY_DICT = {
    'content_restrictions': {'duration': 'long-term', 'importance': 'must'},
    'genre_preferences_like': {'duration': 'long-term', 'importance': 'must'},
    'genre_preferences_dislike': {'duration': 'long-term', 'importance': 'must'},
    'frequently_watched': {'duration': 'long-term', 'importance': 'could'},
    'viewing_companions': {'duration': 'short-term', 'importance': 'could'},
    'cultural_preferences_like': {'duration': 'long-term', 'importance': 'could'},
    'cultural_preferences_dislike': {'duration': 'long-term', 'importance': 'could'},
    'actor_like': {'duration': 'long-term', 'importance': 'could'},
    'actor_dislike': {'duration': 'long-term', 'importance': 'could'},
    'movie_like': {'duration': 'long-term', 'importance': 'could'},
    'movie_dislike': {'duration': 'long-term', 'importance': 'could'},
    'seasonal_themes': {'duration': 'long-term', 'importance': 'could'},
    'companion_preferences_like': {'duration': 'long-term', 'importance': 'could'},
    'companion_preferences_dislike': {'duration': 'long-term', 'importance': 'could'},
    'special_occasion': {'duration': 'short-term', 'importance': 'could'},
    'release_period_preferences': {'duration': 'long-term', 'importance': 'could'},
    'users_mood': {'duration': 'short-term', 'importance': 'should'},
    'platform_availability': {'duration': 'short-term', 'importance': 'should'},
}
