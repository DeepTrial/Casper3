import numpy as np

from .kernelImpl.embedding_func import sinusoidal_position_embedding

class RotaryPositionalEmbeddings():
    def __init__(self, pow_base = 100000) -> None:
        self.pow_base = pow_base 
    
    def forward(self, input: np.ndarray) -> np.ndarray:
       
        pos_emb = sinusoidal_position_embedding(input.shape, self.pow_base)
        pos_cos_emb = np.repeat(pos_emb[..., 1::2], repeats = 2, axis = -1)
        pos_sin_emb = np.repeat(pos_emb[..., ::2], repeats = 2, axis = -1)
        
        input_temp = np.stack([-input[..., 1::2], input[..., ::2]], axis = -1)
        input_temp = input_temp.reshape(input.shape)
        
        return input * pos_cos_emb + input_temp * pos_sin_emb