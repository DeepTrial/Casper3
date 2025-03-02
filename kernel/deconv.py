import numpy as np
from numba import njit

class ConvTranspose2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, dilation=1, bias=True):
        # 初始化参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.output_padding = (output_padding, output_padding) if isinstance(output_padding, int) else output_padding
        self.groups = groups
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

        # 验证参数有效性
        assert in_channels % groups == 0, "in_channels必须能被groups整除"
        assert out_channels % groups == 0, "out_channels必须能被groups整除"

        # 初始化权重
        self.weight = np.random.randn(in_channels, out_channels // groups, self.kernel_size[0], self.kernel_size[1]).astype(np.float32)
        
        for i in range(self.weight.shape[1]):
            self.weight[:,i,0,0] = (3+i)%15
            self.weight[:,i,0,1] = (-2+i)%15
            self.weight[:,i,1,0] = (2+i)%15
            self.weight[:,i,1,1] = (-3+i)%15
        
        # self.weight = self.weight * 0.025
        bias_val = np.asarray([-906,-7253,-8831,-9903,2600,-6311,-8200,2446,-13641,-6155,-7435,-1592,-6164,-13057,-14318,-8809], dtype = np.float32)
        x_bias = - np.sum(self.weight.transpose(1,0,2,3)[:,:,0,0] * 127, axis = 1) * 0.00625
        x_bias = x_bias.flatten() + bias_val.flatten() * 0.00625
        for i in x_bias:
            print(i, end = ",")
        print()

        
        if bias:
            self.bias = np.random.randn(out_channels).astype(np.float32)
            # self.bias = np.asarray([-906,-7253,-8831,-9903,2600,-6311,-8200,2446,-13641,-6155,-7435,-1592,-6164,-13057,-14318,-8809], dtype = np.float32)
            # self.bias = self.bias * 0.025 * 0.25
        else:
            self.bias = np.zeros(out_channels).astype(np.float32)
            
    @njit
    def forwardImpl(X, weight, bias, stride, padding, dilation, kernel_size, output_padding, groups):
         # 输入维度处理
        X = np.asarray(X)
        N, C_in, H_in, W_in = X.shape
        in_channels = C_in 
        out_channels = bias.shape[0]
        
        assert C_in == in_channels, "输入通道数不匹配"

        # 计算输出尺寸
        H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
        W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1

        # 初始化输出
        output = np.zeros((N, out_channels, H_out, W_out), dtype=X.dtype)

        # 遍历每个样本
        for n in range(N):
            # 处理每个组
            for g in range(groups):
                # 当前组的输入输出通道范围
                in_group_channels = in_channels // groups
                out_group_channels = out_channels // groups
                c_in_start = g * in_group_channels
                c_in_end = c_in_start + in_group_channels
                c_out_start = g * out_group_channels
                c_out_end = c_out_start + out_group_channels

                # walk through in/out channel
                for c_in in range(c_in_start, c_in_end):
                    for c_out in range(c_out_start, c_out_end):
                        kernel = weight[c_in, c_out - c_out_start, :, :]
                        # walk through input
                        for h_in in range(H_in):
                            for w_in in range(W_in):
                                
                                h_out_center = h_in * stride[0] - padding[0]
                                w_out_center = w_in * stride[1] - padding[1]

                                # 遍历卷积核元素
                                for kh in range(kernel_size[0]):
                                    for kw in range(kernel_size[1]):
                                        # 计算实际位置（考虑膨胀）
                                        h_out = h_out_center + kh * dilation[0]
                                        w_out = w_out_center + kw * dilation[1]

                                        # 检查是否越界
                                        if 0 <= h_out < H_out and 0 <= w_out < W_out:
                                            # 累加计算结果
                                            output[n, c_out, h_out, w_out] += X[n, c_in, h_in, w_in] * kernel[kh, kw]

        # 添加偏置
        if bias is not None:
            output += bias.reshape(1, -1, 1, 1).astype(output.dtype)

        return output
    
    
    def forward(self, X):
        output = ConvTranspose2d.forwardImpl(X, 
                                             self.weight, 
                                             self.bias, 
                                             self.stride,
                                             self.padding, 
                                             self.dilation, 
                                             self.kernel_size, 
                                             self.output_padding,
                                             self.groups)
       

        return output
