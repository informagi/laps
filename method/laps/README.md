

## Before you start

Confirm that TaskMAD for LAPS is running; this LLM server is a backend server for TaskMAD for LAPS.
For how to run TaskMAD for LAPS, see [here](https://github.com/grill-lab/TaskMAD/tree/radboud_branch).


## Step 1: Make your conda environment

```bash
cd ./method/laps
conda create --name laps_server python=3.8.16
conda activate laps_server
pip install -r requirements.txt
```

## Step 2: Set up API keys


### OpenAI API

```bash
export OPENAI_API_KEY='your_openai_api_key'
```


### Firebase
Fill out the file `your_firebase_api_key.json`.


## Step 3: Run the server

### Before your run: confirm the config

Open `config.py` and check:
- If `TOPIC` (either "recipe" or "movie") is correctly set.
- If `SURVEY_URL` is correctly set if you use quetionnaire.

### Run the server

```bash
python llm_server.py
```

Then you get a guidance on TaskMAD for LAPS.