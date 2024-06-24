from __future__ import print_function
import torch.nn as nn
import torch
from torch.nn import functional as F
from utils.funcs import complex_mul_taps, complex_conv_transpose, to_complex, channel_estimate, signal_est, to_real, pilot_est
from nets.attention_module import AF_Module
import numpy

class attention_Net_DNN(nn.Module):
    def __init__(self, if_relu): # it only gets paramters from other network's parameters
        super(attention_Net_DNN, self).__init__()
        if if_relu:
            self.activ = nn.ReLU()
        else:
            self.activ = nn.Tanh()
        self.tanh = nn.Tanh()
        # self.af_module1 = AF_Module(1)
        # self.af_module2 = AF_Module(1)
        # self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        # self.dense1 = nn.Linear(2, 1, device = 'cuda:0')
        # self.dense2 = nn.Linear(1, 1, device = 'cuda:0')
        self.activ1 = nn.ReLU()
        self.activ2 = nn.Sigmoid()
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling

    def forward(self, x, var, if_bias, h, device, noise_dist, if_RTN, true_snr, noise_var):
        pilot = torch.tensor([-0.5,0.5,0.5,-0.5,-0.5,-0.5,0.5,0.5], dtype=torch.float)
        pilot = pilot.view(1, -1)
        pilot_complex = to_complex(pilot)
        # print('initial size of x:', x.size())
        n = torch.zeros(16, 8)
        for noise_batch_ind in range(x.shape[0]):
            n[noise_batch_ind] = noise_dist.sample()
        n = n.type(torch.FloatTensor)

        n = to_complex(n)
        if true_snr:
            for channel_idx in range(n.shape[0]):
                for noise_idx in range(n.shape[1]):
                    n[channel_idx][noise_idx] = (n[channel_idx][noise_idx] / abs(n[channel_idx][noise_idx])) * noise_var
        n = to_real(n)

        h_est, snr = pilot_est(h, n, pilot, pilot_complex)
        # print('initial size of x:')
        # print(x.size())
        idx_init = 0
        if if_bias:
            gap = 2
        else:
            gap = 1
        idx = idx_init
        while idx < 8:
            if idx > idx_init: # no activation from the beginning
                if idx == gap * 2+idx_init: # after last layer of encoder
                    # print('pass')
                    pass
                else:
                    # print('activation')
                    # print('idx:',idx)
                    # print('check: idx == gap * 2+idx_init', idx == gap * 2+idx_init)
                    x = self.activ(x)

                    m = self.global_pool(x).view(1, 1).to(device)  # Shape: (batch_size, channels)
        
                        # Concatenate m with snr
                    snr = snr.view(1, 1).to(device)  # Ensure snr is of shape (batch_size, 1) and on the correct device
                    m = torch.cat([m, snr], dim=1) # Shape: (batch_size, channels + 1)
                        
                    if idx < gap * 2+idx_init:
                        w5, b5 = var[8], var[9]  # weight and bias
                        w6, b6 = var[10], var[11]  # weight and bias
                    elif idx > gap * 2+idx_init:
                        w5, b5 = var[12], var[13]
                        w6, b6 = var[14], var[15]
                    # Fully Connected Layers
                    m = F.linear(m,w5,b5)  # Shape: (batch_size, ch_num // 16)
                    m = self.activ1(m)
                    m = F.linear(m,w6,b6)
                    m = self.activ2(m)  # Shape: (batch_size, channels)
                        
                    # Multiply inputs with m
                    x = x * m  # Element-wise multiplication

            if idx == idx_init:
                # print('linear1')
                if if_bias:
                    w1, b1 = var[idx], var[idx + 1] # weight and bias
                    x = F.linear(x, w1, b1)
                    idx += 2
                else:
                    w1 = var[idx] # weight
                    x = F.linear(x, w1)
                    idx += 1
            elif idx == gap * 1+idx_init:
                # print('linear2')
                if if_bias:
                    w2, b2 = var[idx], var[idx + 1]  # weight and bias
                    x = F.linear(x, w2, b2)
                    idx += 2
                else:
                    w2 = var[idx]  # weight and bias
                    x = F.linear(x, w2)
                    idx += 1
            elif idx == gap * 2+idx_init:
                # print('channel')
                # print(x.shape)
                #### now we need to normalize and then pass the channel
                x_norm = torch.norm(x, dim=1)
                x_norm = x_norm.unsqueeze(1)
                x = pow(x.shape[1], 0.5) * pow(0.5, 0.5) * x / x_norm

                # x_complex = to_complex(x)

                x = complex_mul_taps(h, x)
                x = x.to(device)
                # print('x',x)
                # noise
                n=n.to(device)
                x = x + n

                # x = to_complex(x)
                # h_est = channel_estimate(x_complex, x, h.shape[0]//2)

                # x = signal_est(h_est, x)

                # # print('x_est', x_est[0,:])
                
                # x = to_real(x)
                x = x.to(device)
                # print('x_est',x)

                # print('x', x[0,:])

                if if_RTN:
                    if if_bias:
                        w_rtn_1, b_rtn_1 = var[idx], var[idx+1]
                        h_inv = F.linear(x, w_rtn_1, b_rtn_1)
                        h_inv = self.tanh(h_inv)
                        w_rtn_2, b_rtn_2 = var[idx+2], var[idx + 3]
                        h_inv = F.linear(h_inv, w_rtn_2, b_rtn_2)
                        h_inv = self.tanh(h_inv)
                        w_rtn_3, b_rtn_3 = var[idx + 4], var[idx + 5]
                        h_inv = F.linear(h_inv, w_rtn_3, b_rtn_3)
                        rtn_gap = 6
                    else:
                        w_rtn_1 = var[idx]
                        h_inv = F.linear(x, w_rtn_1)
                        h_inv = self.tanh(h_inv)
                        w_rtn_2 = var[idx+1]
                        h_inv = F.linear(h_inv, w_rtn_2)
                        h_inv = self.tanh(h_inv)
                        w_rtn_3 = var[idx+2]
                        h_inv = F.linear(h_inv, w_rtn_3)
                        rtn_gap = 3
                    x = complex_conv_transpose(h_inv, x)
                    x = x.to(device)
                else:
                    rtn_gap = 0
                ############## from now, demodulator
                if if_bias:
                    # print('linear3')
                    w3, b3 = var[idx+ rtn_gap], var[idx + rtn_gap + 1]  # weight and bias
                    x = F.linear(x, w3, b3)
                    idx += (2 + rtn_gap)
                else: 
                    w3 = var[idx + rtn_gap]  # weight
                    x = F.linear(x, w3)
                    idx += (1 + rtn_gap)
            elif idx == gap * 3+rtn_gap+idx_init:
                # print('linear4')
                if if_bias:
                    w4, b4 = var[idx], var[idx + 1]  # weight and bias
                    x = F.linear(x, w4, b4)
                    idx += 2
                else:
                    w4 = var[idx]  # weight
                    x = F.linear(x, w4)
                    idx += 1
        return x

def attention_dnn(**kwargs):
    net = attention_Net_DNN(**kwargs)
    return net