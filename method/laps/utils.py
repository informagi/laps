import re
import json
import ast

def extract_json(string) -> dict:
    if string is None:
        return None
    
    pattern = r"\{.*?\}"
    match = re.findall(pattern, string)

    if match:
        json_string = match[-1]  # select the last match
        try:
            extracted_json = json.loads(json_string)
            return extracted_json
        except json.JSONDecodeError:
            # try literal_eval
            try:
                _json = ast.literal_eval(json_string.replace("'", '"'))
                return _json
            except:
                print("No JSON string found")
    else:
        print("No JSON string found")
    
    return None


# Currently not used
def try_n_times_decorator(n=3):
    def outer_wrapper(func):
        @wraps(func)
        def inner_wrapper(*args, **kwargs):
            for i in range(n):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f'{i+1}/{n} attempts. Exception caught: {e}. Retrying...')
            # If the function still fails after 'n' attempts, re-raise the last exception.
            raise
        return inner_wrapper
    return outer_wrapper


def _to_dict(json_str: str) -> dict:
    assert type(json_str) == str
    try:
        json_ = ast.literal_eval(json_str)
    except: # mainly json.decoder.JSONDecodeError
        try: # error case: {...'turn_number": 33}
            json_ = ast.literal_eval(json_str.replace("'", '"'))
        except:
            try: # error case: {...'The user doesn't like spicy food or coriander', ...}
                json_ = json.loads(json_str.replace("'", '"').replace('"t', "'t"))
            except Exception as e:
                print(f"WARNING: Invalid JSON string: {json_str}. Error: {e}")
                json_ = extract_json(json_str) # _to_dict expects json-formatted string but if it's not, try to extract_json.
                if json_ is None:
                    raise ValueError(f"Invalid JSON string: {json_str}")
    return json_

# It seems that currently we don't use this function...?
def _to_json_to_str(json_str: str):
    assert type(json_str) == str
    json_ = _to_dict(json_str)
    return json.dumps(json_)

def check_json_validity(json_str: str):
    try:
        json_ = _to_dict(json_str)
        return True
    except:
        return False