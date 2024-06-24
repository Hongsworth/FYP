from __future__ import print_function
import torch
import torch.nn as nn
from utils.funcs import complex_mul_taps, complex_conv_transpose, to_complex, channel_estimate, signal_est, to_real, pilot_est
from nets.attention_module import AF_Module


class att_basic_DNN(nn.Module):
    def __init__(self, M, num_neurons_encoder, n, n_inv_filter, num_neurons_decoder, if_bias, if_relu, if_RTN):
        super(att_basic_DNN, self).__init__()
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

        # self.af_module = AF_Module(1)
        # AF module in encoder
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.en_dense1 = nn.Linear(2, 1, device = 'cuda:0')
        self.en_dense2 = nn.Linear(1, 1, device = 'cuda:0')

        # AF module in decoder
        self.de_dense1 = nn.Linear(2, 1, device = 'cuda:0')
        self.de_dense2 = nn.Linear(1, 1, device = 'cuda:0')

        self.activ1 = nn.ReLU()
        self.activ2 = nn.Sigmoid()

    def forward(self, x, h, noise_dist, device, if_RTN, true_snr, noise_var):
        pilot = torch.tensor([-0.5,0.5,0.5,-0.5,-0.5,-0.5,0.5,0.5], dtype=torch.float)
        pilot = pilot.view(1, -1)
        pilot_complex = to_complex(pilot)
        n = torch.zeros(x.shape[0], 8)
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
        # print(x)
        # print(x.size())
        x = self.enc_fc1(x)
        x = self.activ(x)
        # x = self.af_module(x, snr, device)

        m = self.global_pool(x).view(1, 1).to(device)  # Shape: (batch_size, channels)
        
        # Concatenate m with snr
        snr = snr.view(1, 1).to(device)  # Ensure snr is of shape (batch_size, 1) and on the correct device
        m = torch.cat([m, snr], dim=1) # Shape: (batch_size, channels + 1)
        
        # Fully Connected Layers
        m = self.en_dense1(m)  # Shape: (batch_size, ch_num // 16)
        m = self.activ1(m)
        m = self.en_dense2(m)
        m = self.activ2(m)  # Shape: (batch_size, channels)
        
        # Multiply inputs with m
        x = x * m


        x = self.enc_fc2(x)
        # normalize
        x_norm = torch.norm(x, dim=1)
        x_norm = x_norm.unsqueeze(1)
        x = pow(x.shape[1], 0.5) * pow(0.5, 0.5) * x / x_norm  # since each has ^2 norm as 0.5 -> complex 1
        # channel
        # print('Before channel')
        # print(x.size())
        # print(x)
        # x_complex = to_complex(x)

        x = complex_mul_taps(h, x)
        x = x.to(device)

        n=n.to(device)
        # noise
        x = x + n # noise insertion
        # print(x.size())

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
        # x = self.af_module(x, snr, device)

        m = self.global_pool(x).view(1, 1).to(device)  # Shape: (batch_size, channels)
        
        # Concatenate m with snr
        snr = snr.view(1, 1).to(device)  # Ensure snr is of shape (batch_size, 1) and on the correct device
        m = torch.cat([m, snr], dim=1) # Shape: (batch_size, channels + 1)
        
        # Fully Connected Layers
        m = self.de_dense1(m)  # Shape: (batch_size, ch_num // 16)
        m = self.activ1(m)
        m = self.de_dense2(m)
        m = self.activ2(m)  # Shape: (batch_size, channels)
        
        # Multiply inputs with m
        x = x * m

        x = self.dec_fc2(x) # softmax taken at loss function
        return x

def att_dnn(**kwargs):
    net = att_basic_DNN(**kwargs)
    return net
