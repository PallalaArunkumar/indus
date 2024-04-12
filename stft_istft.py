
import torch
import torchaudio

def stft_lay(x):
    sig = x
    window_length = 320
    hop_length = 160

    time_frame = (sig.shape[2]-window_length)//hop_length + 1


    window_coeff = torch.hann_window(window_length)


    stft_spec = torch.zeros((sig.shape[0],257,time_frame),dtype=torch.complex64).to(x.device)

    c=0
    for i in range(0,sig.shape[2]-160,160):
        ip_sig = sig[:,:,i:i+320]*window_coeff.to(x.device)
        ip_sig = ip_sig.squeeze(1).clone().detach()


        each_frame_frequencies = torch.fft.rfft(ip_sig,n=512)
        stft_spec[:,:,c] = each_frame_frequencies
        c+=1


    real = torch.real(stft_spec)
    imag = torch.imag(stft_spec)

    stacked_tensor = torch.stack([real,imag],dim=1)

    return stacked_tensor,stft_spec


def overlap_and_add(x, hop_len=160):

    window_coeff = torch.hann_window(320)

    sig_len = (x.shape[1] - 1)* hop_len + x.shape[2]    # x shape is (batch ,time,freq)
    signal = torch.zeros((x.shape[0],sig_len)).to(x.device)
    for i in range(x.shape[1]):

        signal[:, i*hop_len:i*hop_len+x.shape[2]] += x[:,i,:]# * window_coeff
    return signal


def istft_lay(x,op_freq=512): # x shape batch,freq,time
    #print('x shape in istft',x.shape)
    irfft_x = torch.fft.irfft(x.permute(0,2,1),op_freq)
    #print(irfft_x.shape)

    signal = overlap_and_add(irfft_x[:,:,:320].to(x.device))#irfft_x batch,time,freq

    return signal
