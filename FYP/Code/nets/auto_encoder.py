from __future__ import print_function
import torch
import torch.nn as nn
from utils.funcs import complex_mul_taps, complex_conv_transpose, to_complex, channel_estimate, signal_est, to_real


class basic_DNN(nn.Module):
    def __init__(self, M, num_neurons_encoder, n, n_inv_filter, num_neurons_decoder, if_bias, if_relu, if_RTN):
        super(basic_DNN, self).__init__()
        self.enc_fc1 = nn.Linear(M, num_neurons_encoder, bias=if_bias)
        self.enc_fc2 = nn.Linear(num_neurons_encoder, n, bias=if_bias)

        ### norm, nothing to train
        ### channel, nothing to train

        num_inv_filter = 2 * n_inv_filter
        if if_RTN:
            self.rtn_1 = nn.Linear(n, n, bias=if_bias)
            self.rtn_2 = nn.Linear(n, n, bias=if_bias)
            self.rtn_3 = nn.Linear(n, num_inv_filter, bias=if_bias)
        else:
            pass

        self.dec_fc1 = nn.Linear(n, num_neurons_decoder, bias=if_bias)
        self.dec_fc2 = nn.Linear(num_neurons_decoder, M, bias=if_bias)
        if if_relu:
            self.activ = nn.ReLU()
        else:
            self.activ = nn.Tanh()
        self.tanh = nn.Tanh()
    def forward(self, x, h, noise_dist, device, if_RTN):
        # print(x)
        # print(x.size())
        x = self.enc_fc1(x)
        x = self.activ(x)
        x = self.enc_fc2(x)
        # normalize
        x_norm = torch.norm(x, dim=1)
        x_norm = x_norm.unsqueeze(1)
        x = pow(x.shape[1], 0.5) * pow(0.5, 0.5) * x / x_norm  # since each has ^2 norm as 0.5 -> complex 1
        # channel
        # print('Before channel')
        # print(x.size())
        # print(x)
        x_complex = to_complex(x)

        x = complex_mul_taps(h, x)
        # noise
        n = torch.zeros(x.shape[0], x.shape[1])
        for noise_batch_ind in range(x.shape[0]):
            n[noise_batch_ind] = noise_dist.sample()
        n = n.type(torch.FloatTensor)
        
        x = x + n # noise insertion
        # print(x.size())
        x = to_complex(x)
        h_est = channel_estimate(x_complex, x, h.shape[0]//2)

        x = signal_est(h_est, x)

        x = to_real(x)
        x = x.to(device)


        # RTN
        if if_RTN:
            h_inv = self.rtn_1(x)
            h_inv = self.tanh(h_inv)
            h_inv = self.rtn_2(h_inv)
            h_inv = self.tanh(h_inv)
            h_inv = self.rtn_3(h_inv) # no activation for the final rtn (linear activation without weights)
            x = complex_conv_transpose(h_inv, x)
            x = x.to(device)
        else:
            pass
        x = self.dec_fc1(x)
        x = self.activ(x)
        x = self.dec_fc2(x) # softmax taken at loss function
        return x

def dnn(**kwargs):
    net = basic_DNN(**kwargs)
    return net
