import torch
import torch.nn as nn
import torch.nn.functional as F

class AF_Module(nn.Module):
    def __init__(self, input_channels, name_prefix='af_module'):
        super(AF_Module, self).__init__()
        self.name_prefix = name_prefix
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.dense1 = nn.Linear(2, 1, device = 'cuda:0')
        self.dense2 = nn.Linear(1, 1, device = 'cuda:0')
        self.activ1 = nn.ReLU()
        self.activ2 = nn.Sigmoid()

    def forward(self, inputs, snr, device):
        batch_size, ch_num = 1,1
        # Global Average Pooling
        m = self.global_pool(inputs).view(batch_size, ch_num).to(device)  # Shape: (batch_size, channels)
        
        # Concatenate m with snr
        snr = snr.view(batch_size, 1).to(device)  # Ensure snr is of shape (batch_size, 1) and on the correct device
        m = torch.cat([m, snr], dim=1) # Shape: (batch_size, channels + 1)
        
        # Fully Connected Layers
        m = self.dense1(m)  # Shape: (batch_size, ch_num // 16)
        m = self.activ1(m)
        m = self.dense2(m)
        m = self.activ2(m)  # Shape: (batch_size, channels)
        
        # Multiply inputs with m
        m = inputs * m  # Element-wise multiplication
        return m
        