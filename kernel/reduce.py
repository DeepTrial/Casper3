import numpy as np

from .kernelImpl.reduce_func import ReduceImpl

class ReduceMean:
    def __init__(self, keepdims: int = 1, noop_with_empty_axes: int = 0, _intermediate_dtype = np.float32):
        self.impl = ReduceImpl(mode = 'mean', reduce_dtype=_intermediate_dtype)
        self.keepdims = keepdims
        self.noop_with_empty_axes = noop_with_empty_axes
    
    def forward(self, data: np.ndarray, axes:tuple = ()) -> np.ndarray:
        return self.impl.forwardImpl(data, axes, self.keepdims, self.noop_with_empty_axes)

class ReduceSum:
    def __init__(self, axes: list, keepdims: int = 1, _intermediate_dtype = np.float32):
        self.impl = ReduceImpl(mode = 'sum', reduce_dtype=_intermediate_dtype)
        self.axes = axes
        self.keepdims = keepdims
        self.noop_with_empty_axes = 0
    
    def forward(self, data: np.ndarray) -> np.ndarray:
        return self.impl.forwardImpl(data, self.axes, self.keepdims, self.noop_with_empty_axes)

class ReduceMax:
    def __init__(self, keepdims: int = 1, noop_with_empty_axes: int = 0, _intermediate_dtype = np.float32):
        self.impl = ReduceImpl(mode = 'max', reduce_dtype=_intermediate_dtype)
        self.keepdims = keepdims
        self.noop_with_empty_axes = noop_with_empty_axes
    
    def forward(self, data: np.ndarray, axes:tuple = ()) -> np.ndarray:
        return self.impl.forwardImpl(data, axes, self.keepdims, self.noop_with_empty_axes)


class ReduceMin:
    def __init__(self, keepdims: int = 1, noop_with_empty_axes: int = 0, _intermediate_dtype = np.float32):
        self.impl = ReduceImpl(mode = 'min', reduce_dtype=_intermediate_dtype)
        self.keepdims = keepdims
        self.noop_with_empty_axes = noop_with_empty_axes
    
    def forward(self, data: np.ndarray, axes:tuple = ()) -> np.ndarray:
        return self.impl.forwardImpl(data, axes, self.keepdims, self.noop_with_empty_axes)