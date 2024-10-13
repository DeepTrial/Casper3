import numpy as np

from .kernelImpl.reduce_func import ReduceImpl

class ReduceMean:
    def __init__(self, keepdims: int = 1, noop_with_empty_axes: int = 0, intermediate_dtype = np.float32):
        self.impl = ReduceImpl(mode = 'mean', reduce_dtype=intermediate_dtype)
        self.keepdims = keepdims
        self.noop_with_empty_axes = noop_with_empty_axes
    
    def forward(self, input: np.ndarray, axes:tuple = ()) -> np.ndarray:
        return self.impl.forwardImpl(input, axes, self.keepdims, self.noop_with_empty_axes)

