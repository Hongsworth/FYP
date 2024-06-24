import torch

def complex_mul(h, x): # h fixed on batch, x has multiple batch
    if len(h.shape) == 1:
        # h is same over all messages (if estimated h, it is averaged)
        y = torch.zeros(x.shape[0], 2, dtype=torch.float)
        y[:, 0] = x[:, 0] * h[0] - x[:, 1] * h[1]
        y[:, 1] = x[:, 0] * h[1] + x[:, 1] * h[0]
    elif len(h.shape) == 2:
        # h_estimated is not averaged
        assert x.shape[0] == h.shape[0]
        y = torch.zeros(x.shape[0], 2, dtype=torch.float)
        y[:, 0] = x[:, 0] * h[:, 0] - x[:, 1] * h[:, 1]
        y[:, 1] = x[:, 0] * h[:, 1] + x[:, 1] * h[:, 0]
    else:
        print('h shape length need to be either 1 or 2')
        raise NotImplementedError
    return y


def complex_mul_taps(h, x_tensor):
    if len(h.shape) == 1:
        L = h.shape[0] // 2  # length/2 of channel vector means number of taps
    elif len(h.shape) == 2:
        L = h.shape[1] // 2  # length/2 of channel vector means number of taps
    else:
        print('h shape length need to be either 1 or 2')
        raise NotImplementedError
    y = torch.zeros(x_tensor.shape[0], x_tensor.shape[1], dtype=torch.float)
    assert x_tensor.shape[1] % 2 == 0
    for ind_channel_use in range(x_tensor.shape[1]//2):
        for ind_conv in range(min(L, ind_channel_use+1)):
            if len(h.shape) == 1:
                y[:, (ind_channel_use) * 2:(ind_channel_use + 1) * 2] += complex_mul(h[2*ind_conv:2*(ind_conv+1)], x_tensor[:, (ind_channel_use-ind_conv)*2:(ind_channel_use-ind_conv+1)*2])
            else:
                y[:, (ind_channel_use) * 2:(ind_channel_use + 1) * 2] += complex_mul(
                    h[:, 2 * ind_conv:2 * (ind_conv + 1)],
                    x_tensor[:, (ind_channel_use - ind_conv) * 2:(ind_channel_use - ind_conv + 1) * 2])

    return y

def complex_conv_transpose(h_trans, y_tensor): # takes the role of inverse filtering
    assert len(y_tensor.shape) == 2 # batch
    assert y_tensor.shape[1] % 2 == 0
    assert h_trans.shape[0] % 2 == 0
    if len(h_trans.shape) == 1:
        L = h_trans.shape[0] // 2
    elif len(h_trans.shape) == 2:
        L = h_trans.shape[1] // 2
    else:
        print('h shape length need to be either 1 or 2')

    deconv_y = torch.zeros(y_tensor.shape[0], y_tensor.shape[1] + 2*(L-1), dtype=torch.float)
    for ind_y in range(y_tensor.shape[1]//2):
        ind_y_deconv = ind_y + (L-1)
        for ind_conv in range(L):
            if len(h_trans.shape) == 1:
                deconv_y[:, 2*(ind_y_deconv - ind_conv):2*(ind_y_deconv - ind_conv+1)] += complex_mul(h_trans[2*ind_conv:2*(ind_conv+1)] , y_tensor[:,2*ind_y:2*(ind_y+1)])
            else:
                deconv_y[:, 2 * (ind_y_deconv - ind_conv):2 * (ind_y_deconv - ind_conv + 1)] += complex_mul(
                    h_trans[:, 2 * ind_conv:2 * (ind_conv + 1)], y_tensor[:, 2 * ind_y:2 * (ind_y + 1)])
    return deconv_y[:, 2*(L-1):]


def to_complex(x_tensor):
    if len(x_tensor.shape) == 2:
        x_complex = torch.zeros(x_tensor.shape[0], x_tensor.shape[1]//2, dtype=torch.cfloat, )
        for ind_complex in range(x_tensor.shape[1]//2):
            x_complex[:, ind_complex] = x_tensor[:, 2*ind_complex] + 1j * x_tensor[:, 2*ind_complex+1]
    elif len(x_tensor.shape) == 1:
        x_complex = torch.zeros(x_tensor.shape[0]//2, dtype=torch.cfloat)
        for ind_complex in range(x_tensor.shape[0]//2):
            x_complex[ind_complex] = x_tensor[2*ind_complex] + 1j * x_tensor[2*ind_complex+1]
    return x_complex

def channel_estimate(x_tensor, y_tensor, taps):
    # print(x_tensor.shape[1], y_tensor.shape[1])
    pilot = torch.zeros([(x_tensor.shape[1]-2), taps],dtype = torch.cfloat, requires_grad=False)
    # print(pilot.shape)
    for i in range(x_tensor.shape[1]-2):
        pilot[i,0] = x_tensor[0,i+2]
        pilot[i,1] = x_tensor[0,i+1]
        pilot[i,2] = x_tensor[0,i]

    h_est_complex = torch.matmul(torch.linalg.pinv(pilot), y_tensor[0,2:])
    h_est_complex = h_est_complex.detach()

    return h_est_complex

def signal_est(h_est, y_tensor, Snr=15):
    batch_size, signal_length = y_tensor.shape
    num_taps = h_est.shape[0]

    # Create the convolution matrix H using torch.linalg.toeplitz
    H = torch.zeros((signal_length, signal_length), dtype=torch.cfloat, device=y_tensor.device)
    for i in range(y_tensor.shape[1]):
        for j in range(h_est.shape[0]):
            H[i,i-j] = h_est[j]
    # Snr_lin = 10 ** (Snr / 10)
    # snr_ratio = 1 / Snr_lin

    # H_H = torch.conj(H).T
    # I = torch.eye(signal_length, dtype=torch.cfloat, device=y_tensor.device)
    # H_MMSE = torch.linalg.inv(H_H @ H + snr_ratio * I) @ H_H
    H_pinv = torch.linalg.pinv(H)

    x_est = torch.zeros(batch_size, signal_length, dtype=torch.cfloat, device=y_tensor.device)
    x_est = torch.matmul(y_tensor, H_pinv.T)  # batch matrix multiplication
    # For MMSE, use:
    # x_est = torch.matmul(y_tensor, H_MMSE.T)  # batch matrix multiplication

    return x_est

def to_real(x_tensor):
    x_real = torch.zeros(x_tensor.shape[0], x_tensor.shape[1]*2, dtype=torch.float)
    for ind_complex in range(x_tensor.shape[1]):
        x_real[:, 2*ind_complex] = x_tensor[:, ind_complex].real
        x_real[:, 2*ind_complex+1] = x_tensor[:, ind_complex].imag
    return x_real

def pilot_est(channel, noise, x, x_complex):   

    y = complex_mul_taps(channel, x)
    y = y + noise[0,:]
    y = to_complex(y)
    h_est = channel_estimate(x_complex, y, channel.shape[0]//2)
    h_est = to_complex(channel)

    x_est = signal_est(h_est, y)
    x_est_r = to_real(x_est)
    # y_est = complex_mul_taps(channel, x_est_r)
    y_est = complex_mul_taps(channel, x)
    y_est = to_complex(y_est)

    y_diff = y - y_est

    P_signal = torch.mean(torch.abs(x_complex)**2)
    P_noise = torch.mean(torch.abs(y_diff)**2)
    SNR = P_signal / P_noise
    SNR_dB = 10 * torch.log10(SNR)

    return h_est, SNR_dB

    
