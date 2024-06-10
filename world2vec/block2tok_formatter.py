import difflib
import json
from fuzzywuzzy import process
import re 

completed_json_path = "completed.json" 
all_json_path = "all.json" 
state_delims = ['=', ',']
reg_delims = [',']

def split_by_multiple_delimiters(text, delimiters):
    pattern = '|'.join(map(re.escape, delimiters))
    return re.split(pattern, text)


def custom_matcher(query, choices):
    query = query.replace(" ", "")
    query_words = set(split_by_multiple_delimiters(query, reg_delims))
    
    scores = {}
    for choice in choices:
        choice_words = set(split_by_multiple_delimiters(choice, state_delims))
        common = query_words.intersection(choice_words)
        scores[choice] = len(common)
    best_match = max(scores, key=scores.get)
    return best_match


with open(completed_json_path, 'r') as file:
    completed_json = json.load(file)

with open(all_json_path, 'r') as file:
    all_json = json.load(file)

found = False

for block_name in completed_json:
    if (block_name == "brick_stairs"): 
        found = True
    
    if found: 
        if isinstance(completed_json[block_name], dict):  
            if block_name in all_json and 'variants' in all_json[block_name]:
                all_states = list(all_json[block_name]['variants'].keys())
                updated_states = {}
                for state in completed_json[block_name]:
                    if state.strip() == "":
                        updated_states[state] = completed_json[block_name][state]
                    else:      
                        closest_state = custom_matcher(state, all_states)
                        if (block_name == "brick_stairs"): 
                            print("for " + str(state) + " " + str(closest_state) + " is closest match")
                          
                        updated_states[closest_state] = completed_json[block_name][state]
                    
                completed_json[block_name] = updated_states

updated_json_path = 'block2tok.json'
with open(updated_json_path, 'w') as file:
    json.dump(completed_json, file, indent=4)

