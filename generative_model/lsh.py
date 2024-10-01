import lshashpy3 as lshash
import numpy as np
import os
import json
import random

# Pass embeddings as list, and create hash, saving it to file_name
def create_hash(embeddings, num_bits, num_tables, file_name):
    lsh = LSHash(hash_size=num_bits, input_dim=32, num_hashtables=num_tables,
    storage_config={ 'dict': None },
    matrices_filename=file_name.split('.')[0] + '_weights.npz',
    hashtable_filename=file_name,
    overwrite=True)

    for e in range(0, len(embeddings)):
        lsh.index(embeddings[e], extra_data=e)
    
    lsh.save()

# Read embeddings into a list
def prepare_embedding_matrix():
    block2embedding = None
    block2embedding_file_path = r'embeddings/embeddings.json'
    with open(block2embedding_file_path, 'r') as j:
        block2embedding = json.loads(j.read())
    block2tok = None
    block2tok_file_path = r'../world2vec/block2tok.json'
    with open(block2tok_file_path, 'r') as j:
        block2tok = json.loads(j.read())
    # Convert block names to tokens (integers)
    # Build tok2embedding mapping
    tok2embedding_int = {}
    for block_name, embedding in block2embedding.items():
        token_str = block2tok.get(block_name)
        if token_str is not None:
            token = int(token_str)
            tok2embedding_int[token] = np.array(embedding, dtype=np.float32)
    
    # Get the air token and embedding
    air_block_name = 'minecraft:air'
    air_token_str = block2tok.get(air_block_name)
    if air_token_str is None:
        raise ValueError('minecraft:air block not found in block2tok mapping')
    air_token = int(air_token_str)
    air_embedding = block2embedding.get(air_block_name)
    if air_embedding is None:
        raise ValueError('minecraft:air block not found in block2embedding mapping')
    air_embedding = np.array(air_embedding, dtype=np.float32)

    # Ensure that the air token and embedding are included
    tok2embedding_int[air_token] = air_embedding

    # Collect all tokens
    tokens = list(tok2embedding_int.keys())

    # Ensure tokens are non-negative integers
    if min(tokens) < 0:
        raise ValueError("All block tokens must be non-negative integers.")

    token_set = set(tokens)
    embedding_dim = len(air_embedding)

    # Find the maximum token value to determine the size of the lookup arrays
    max_token = max(tokens + [3714])  # Include unknown block token

    # Build the embedding matrix
    embedding_matrix = np.zeros((max_token + 1, embedding_dim), dtype=np.float32)
    for token in tokens:
        embedding_matrix[token] = tok2embedding_int[token]
    # Set the embedding for unknown blocks to the air embedding
    embedding_matrix[3714] = air_embedding  # Token 3714 is the unknown block

    # Build the lookup array mapping tokens to indices in the embedding matrix
    lookup_array = np.full((max_token + 1,), air_token, dtype=np.int32)
    for token in tokens:
        lookup_array[token] = token  # Map token to its own index

    # Map unknown block token to 3714
    lookup_array[3714] = 3714

    return lookup_array, embedding_matrix

lookup_array, embedding_matrix = prepare_embedding_matrix()
create_hash(embedding_matrix, 10, 1, 'test1.npz')
# Below is for testing the hash tables; you may want to comment out the above line if you are just testing
# test1: hash_size=10, num_hashtables=1
lsh = LSHash(hash_size=10, input_dim=32, num_hashtables=1,
    storage_config={ 'dict': None },
    matrices_filename='test1_weights.npz',
    hashtable_filename='test1.npz',
    overwrite=True)

test_blocks = []
test_embeddings = []
for i in range(0, 20):
    block_token = random.randint(0, 3716)
    while block_token in test_blocks:
        block_token = random.randint(0, 3716)
    test_blocks.append(block_token)
    test_embeddings.append(embedding_matrix[block_token])
# First, test if the exact embeddings map back to the original tokens
print("Test 1: Exact embeddings")
results_1 = []
blocks_searched = 1
for e in test_embeddings:
    anns = lsh.query(e, distance_func='euclidean')
    results_1.append(anns)
    print("Block %d: %d, ANNs: %d" % (blocks_searched, test_blocks[blocks_searched - 1], anns))
    blocks_searched += 1
successes = 0
score = 20
for r in range(0, len(results_1)):
    if test_blocks[r] in results_1[r]:
        successes += 1
        score -= results_1[r].index(test_blocks[r]) # A higher score means the hash table is more reliable
print("Exact embeddings: %d/20 blocks correctly mapped, score: %d" % (successes, score))
# Now, check slightly modified embeddings to see if they can still be mapped back to the correct tokens
print("Test 2: Modified embeddings")
for e in range(0, len(test_embeddings)):
    for i in range(0, len(test_embeddings[e])):
        coin_flip = random.randint(0, 1)
        if coin_flip == 0:
            test_embeddings[e][i] = test_embeddings[e][i] + 0.001
        else:
            test_embeddings[e][i] = test_embeddings[e][i] - 0.001
results_2 = []
blocks_searched = 1
for e in test_embeddings:
    anns = lsh.query(e, distance_func='euclidean')
    results_2.append(anns)
    print("Block %d: %d, ANNs: %d" % (blocks_searched, test_blocks[blocks_searched - 1], anns))
    blocks_searched += 1
successes = 0
score = 20
for r in range(0, len(results_2)):
    if test_blocks[r] in results_2[r]:
        successes += 1
        score -= results_2[r].index(test_blocks[r]) # A higher score means the hash table is more reliable
print("Modified embeddings: %d/20 blocks correctly mapped, score: %d" % (successes, score))