import os
import sys
import numpy as np 
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from kernel.reduce_interface import ReduceMean

np.random.seed(123456)


class TestReduceMean:    
    def base_test_func(self, test_args):
        data = test_args['data']
        axes = test_args['axes']
        keep_dim = test_args['keepdims']
        dtype = test_args['dtype']
        noop_with_empty_axes = test_args['noop_with_empty_axes']
        
        ret = ReduceMean(
            keepdims=keep_dim, 
            noop_with_empty_axes=noop_with_empty_axes,
            intermediate_dtype=dtype).forward(data, axes = axes)
        golden = torch.mean(torch.from_numpy(data), keepdim=keep_dim>0, dim = axes).numpy()
        
        return np.absolute(ret - golden)

    def test_case_1(self):
        args = {
            'data':np.random.uniform(low=-10,high=10, size=(1,1024,3584)).astype(np.float16),
            'axes': (2,),
            'keepdims': 1,
            'dtype':np.float64,
            'noop_with_empty_axes':0
        }
        diff = self.base_test_func(args)
        max_diff = np.max(diff)
        avg_diff = np.mean(diff)
        print(max_diff, avg_diff)
    

if __name__=="__main__":
    obj = TestReduceMean()
    obj.test_case_1()