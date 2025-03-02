import os
import sys
import numpy as np 
import torch
import torch.nn as nn
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from kernel.deconv import ConvTranspose2d

np.random.seed(123456)

class TestDeconv:    
    def base_test_func(self, input, test_args):
        # 创建测试输入
        X_np = input
        X_torch = torch.tensor(X_np.copy())
        
        # 初始化两种实现
        np_conv = ConvTranspose2d(**test_args)
        torch_conv = nn.ConvTranspose2d(**test_args)

        # 同步权重
        with torch.no_grad():
            torch_conv.weight.data = torch.from_numpy(np_conv.weight)
            if test_args['bias']:
                torch_conv.bias.data = torch.from_numpy(np_conv.bias)

        # forward 
        triggle_time = time.time()
        np_output = np_conv.forward(X_np)
        torch_output = torch_conv(X_torch).detach().numpy()
        # 精度验证
        print("Finished in: {:.2f} sec".format(time.time() - triggle_time))
        print(f"Max ABS Error:  {np.max(np.abs(np_output - torch_output))}")
        # print(f"Mean ABS Error: {np.mean(np.abs((np_output - torch_output)/torch_output))}") 
        
        print(np_output.shape)
        np_output = np_output.transpose(0,2,3,1)
        bias = [-386.66250576,-447.13125711,-347.29375518,-214.29375319, -186.95000279,-293.44375437,-356.05000531,-340.31250507,-491.65625733,-495.66875739,-554.46875826,-568.75000848,-457.62500682,-361.00625538,-419.68750625,-436.0562565]
        new_bias = [-43.7625,-96.13125,-118.693756,-138.09375,-72.65,-141.04375,-165.55,-111.7125,-224.95624,-190.86876,-211.56876,-187.75,-38.525,-94.30625,-114.887505,-93.15625]
        a = np_output.flatten()[4] * 0.00625
        b = np_output.flatten()[5] * 0.00625
        c = np_output.flatten()[6] * 0.00625
        d = np_output.flatten()[7] * 0.00625
        print(a + new_bias[4], b+new_bias[5],c+new_bias[6],d+new_bias[7])
        # print(np_output.flatten()[:12])
        
    
    def test_case_1(self):
        args = {
            'in_channels': 16,
            'out_channels': 16,
            'kernel_size': 2,
            'stride': 2,
            'padding': 0,
            'output_padding': 0,
            'dilation': 1,
            'groups': 1,
            'bias': False
        }
        X_data = np.ones((1,16,640,320)).astype(np.float32) * 150
        # X_f = 0.25 * (X_data - 127)
        
        # self.base_test_func(X_f, args)
        
        self.base_test_func(X_data, args)
if __name__ == "__main__":
    case = TestDeconv()
    case.test_case_1()