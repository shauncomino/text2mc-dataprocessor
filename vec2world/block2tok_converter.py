import json

def reverse_block2tok(data, outer_key=""):
    reversed_dict = {}
    for key, value in data.items():

        # Handle nested JSON object
        if isinstance(value, dict):
            nested_reversed = reverse_block2tok(value, outer_key=key)

            for nested_key, nested_value in nested_reversed.items():
                block_index = nested_key.split(":")[0]
                nested_key = nested_key.lstrip(f"{block_index}:")
                reversed_dict[block_index] = f"{nested_key}[{nested_value}]"

        # Handle simple key-value swap
        else:
            new_key = f"{value}:{outer_key}" if outer_key else value
            reversed_dict[new_key] = key
    
    return reversed_dict

with open("../world2vec/tok2block.json", "w") as f:
    block2tok_file = open("block2tok.json")
    data = json.load(block2tok_file)
    data_reversed = reverse_block2tok(data)
    json.dump(data_reversed, f, ensure_ascii=False, indent=4)
