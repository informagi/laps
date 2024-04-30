from pathlib import Path

DELIMITER = "####"
VERBOSE = True
USE_ADVICE_MODE = True # if true, use Guidance class in response.py. (only True is supported now)
SURVEY_URL = 'https://your-survey-url' # set your survey URL here.
PROJECT_DIR = Path(__file__).resolve().parents[1]
TOPIC = 'movie' # 'recipe' or 'movie'