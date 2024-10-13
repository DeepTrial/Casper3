import copy
import numpy as np

class ReduceImpl:
    def __init__(self, mode: str, reduce_dtype = np.float32):
        self.raw_shape = None 
        self.support_reduce_func = {
            'min':  self.reduce_min_impl,
            'max':  self.reduce_max_impl,
            'mean': self.reduce_mean_impl,
            'sum':  self.reduce_sum_impl
        }
        self.reduce_func = self.support_reduce_func.get(mode.lower(), None)
        assert self.reduce_func != None, f"Reduce Mode {mode} has not supported yet!"
        
        self.reduce_dtype = reduce_dtype
    
    def forwardImpl(self, data: np.ndarray, axes: tuple, keepdims: int = 1 , noop_with_empty_axes: int = 0) -> np.ndarray:
        self.raw_shape = data.shape 
        normalized_axes, reduced_shape  = self.__convert_axes(data.ndim, list(axes), noop_with_empty_axes)
        axes_reduce_ret = copy.deepcopy(data)
        
        for axes_value in normalized_axes:
            axes_reduce_ret = self.reduce_func(np.swapaxes(axes_reduce_ret, axes_value, 0))
            axes_reduce_ret = np.swapaxes(axes_reduce_ret, 0, axes_value)
        axes_reduce_ret = np.reshape(axes_reduce_ret, reduced_shape)
        
        if not keepdims:
            axes_reduce_ret = np.squeeze(axes_reduce_ret, axis = axes)
        return axes_reduce_ret
    
    def __convert_axes(self, data_rank:int, axes: tuple, noop_with_empty_axes: int = 0) -> list[int]:
        norm_axes = []
        reduced_shape = copy.deepcopy(list(self.raw_shape))
        if list(axes) == [] and noop_with_empty_axes == 0:
            # reudce all axes 
            axes = list(range(data_rank))
            
        for axes_value in axes:
            if axes_value < 0:
                norm_axes.append(axes_value + data_rank)
            else:
                norm_axes.append(axes_value)
            reduced_shape[norm_axes[-1]] = 1
        
        return norm_axes, reduced_shape

    def reduce_min_impl(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def reduce_max_impl(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reduce_mean_impl(self, input: np.ndarray) -> np.ndarray:
        inter_sum = self.reduce_sum_impl(input)
        inter_sum = inter_sum / input.shape[0]
        return inter_sum.astype(self.reduce_dtype)

    def reduce_sum_impl(self, input: np.ndarray) -> np.ndarray:
        # record ret shape
        output_shape = list(input.shape)
        output_shape[0] = 1 # reduce on axis 0
        
        # merge non-reduce axis
        input = input.reshape(input.shape[0], -1)
        # impl reduce
        inter_sum = np.zeros([1, input.shape[1]], dtype = self.reduce_dtype)
        for reduce_axes in range(input.shape[0]):
            inter_sum[0,:] += input[reduce_axes, :].astype(self.reduce_dtype)
            
        return inter_sum.reshape(output_shape)