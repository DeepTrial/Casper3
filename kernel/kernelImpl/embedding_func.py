import numpy as np

def sinusoidal_position_embedding(shape_list: list, base = 100000) -> np.ndarray:
    batch_size, nums_head, max_len, output_dim = shape_list
    
    positions = np.expand_dims(np.arange(0, max_len, dtype = np.float32), axis = -1)
    ids = np.arange(0, output_dim // 2, dtype = np.float32)
    theta = np.pow(base, -2.0 * ids / output_dim)
    
    embeddings = positions * theta # (max_len, output_dim // 2)
    embeddings = np.stack([np.sin(embeddings), np.cos(embeddings)], axis=-1)  # (max_len, output_dim // 2, 2)
    
    embeddings = np.expand_dims(embeddings, axis=(0,1))
    embeddings = np.repeat(embeddings, repeats = batch_size, axis = 0)
    embeddings = np.repeat(embeddings, repeats = nums_head, axis = 1)  # (bs, num_heads, max_len, output_dim//2 ,2)
    
    embeddings = np.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
    return embeddings