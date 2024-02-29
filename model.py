import torch
from torch import nn
from modules import ConvSC, Inception
from PatchTST import *
def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]

class Encoder(nn.Module):
    def __init__(self,C_in, C_hid, N_S):
        super(Encoder,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )
    
    def forward(self,x):# B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1


class Decoder(nn.Module):
    def __init__(self,C_hid, C_out, N_S):
        super(Decoder,self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
    
    def forward(self, hid, enc1=None):
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y



class SimVP(nn.Module):
    def __init__(self, shape_in, hid_S=64, hid_T=256, N_S=2, N_T=4, incep_ker=[3,5,3,11], groups=8):
        super(SimVP, self).__init__()
        print(f'shape_in size: {shape_in}')
        T, C, H, W = shape_in
        self.enc = Encoder(C, hid_S, N_S)
        # self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups)
        self.patchtst = PatchTST(hid_S,T, 2, 2)
        self.dec = Decoder(hid_S, C, N_S)


    def forward(self, x_raw):
        
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape
       
        z = embed.view(B*H_*W_, C_, T)
       
        patchtst_out = self.patchtst(z)
    
        patchtst_out = patchtst_out.reshape(B*T, C_, H_, W_) #
        
        Y = self.dec(patchtst_out, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y
