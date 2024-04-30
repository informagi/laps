import time
import concurrent.futures
import openai
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

def get_completion_from_prompt(prompt, model="gpt-3.5-turbo-1106", temperature=None, frequency_penalty=None, remove_newlines=True, verbose=True, request_timeout=30):
    if verbose: print('\033[92m' + prompt + '\033[0m')

    # if `gpt-4` in model then double the request timeout
    if 'gpt-4' in model:
        request_timeout = request_timeout * 2 
    
    attempts = 0
    while attempts < 5:
        print('\033[95m' + f"Attempt {attempts + 1} with temperature={temperature} and frequency_penalty={frequency_penalty}" + '\033[0m')
        time.sleep(10*attempts) # Wait 10 seconds between attempts, then 20, then 30, etc.
        try:
            response = openai.ChatCompletion.create(model=model, messages=[{'role': 'system', 'content': prompt}], temperature=temperature, frequency_penalty=frequency_penalty, request_timeout=request_timeout)
            if verbose: print('\033[94m' + response.choices[0].message["content"] + '\033[0m')
            return response.choices[0].message["content"].replace('\n', ' ') if remove_newlines else response.choices[0].message["content"]
        except (concurrent.futures.TimeoutError, openai.error.RateLimitError, openai.error.ServiceUnavailableError) as e:
            print('\033[91m' + f"Error occurred: {str(e)}. model={model}, request_timeout={request_timeout} retrying..." + '\033[0m')
        except Exception as e: # E.g., Bad gateway
            print('\033[91m' + f"An unexpected error occurred: {str(e)}. model={model}, request_timeout={request_timeout} retrying..." + '\033[0m')
        finally:
            attempts += 1

    print('\033[91m' + f"Failed after 5 attempts. Last exception: {str(e)}" + '\033[0m')
    return None
