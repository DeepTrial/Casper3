import os
import sys
import numpy as np 
import torch
from torchtune.modules import RotaryPositionalEmbeddings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from kernel.pe_interface import RotaryPositionalEmbeddings as RoPE

np.random.seed(123456)


class TestPositionalEmbeddings:    
    def base_test_func(self, test_args): 
        base = test_args['base']
        batch_size = test_args['batch_size']
        head = test_args['head']
        max_seq_len = test_args['max_seq_len']
        dim = test_args['dim']
        
        # bs, n_h, s, h_d
        data = np.random.uniform(low=-10, high=10, size = (batch_size, head, max_seq_len, dim))
        
        ret = RoPE(pow_base = base).forward(data)
        golden = RotaryPositionalEmbeddings(dim=dim, max_seq_len=max_seq_len, base=base)(torch.from_numpy(data.transpose(0,2,1,3)))
        
        return ret, golden.numpy().transpose(0,2,1,3)

    def test_rope_case_1(self):
        args = {
            'batch_size': 1,
            'head': 28,
            'max_seq_len': 4096,
            'dim': 128,
            'base': 100000
        }
        ret, golden = self.base_test_func(args)
        compare_ret = np.isclose(ret, golden, rtol=1e-5, atol=1e-8, equal_nan=True)
        fail_rate = 1 - compare_ret.sum() / np.prod(compare_ret.shape)
        assert fail_rate < 0.1

if __name__=="__main__":
    obj = TestPositionalEmbeddings()
    obj.test_rope_case_1()