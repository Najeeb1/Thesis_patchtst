#Implementation of PatchTST

import torch
from .utils import RevIN
from .encoder import TSTiEncoder

#Implementation of head layer

class Flatten_Head(torch.nn.Module):
    def __init__(self, dmodel, patchnum, target_window, head_dropout=0):
        super().__init__()
        # self.n_vars = n_vars
        self.flatten = torch.nn.Flatten(start_dim=2)
        self.linear = torch.nn.Linear(dmodel * patchnum, target_window) 
        self.dropout = torch.nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)                               # x: [bs x nvars * d_model * patch_num]   instead should do # x: [bs x nvars x d_model * patch_num] 
        x = self.linear(x)                                # x: [bs x target_window]
        x = self.dropout(x)
        return x
        
class PatchTST(torch.nn.Module):
    # context_window = T
    def __init__(self, c_in, context_window, patch_len, stride, max_seq_len=1024, 
                 n_layers=3, d_model=16, n_heads=4, d_k=None, d_v=None,
                 d_ff=128, attn_dropout=0.0, dropout=0.3, key_padding_mask='auto',
                 padding_var=None, attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
                 head_dropout = 0.0, padding_patch = "end",
                 revin = True, affine = False, subtract_last = False,
                 verbose=False, target_idx=-1, **kwargs):
        super().__init__()

        # self.revin = revin
        # if revin:
        #     self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last, target_idx=target_idx)

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)

        if padding_patch == "end":
            self.padding_patch_layer = torch.nn.ReplicationPad1d((0, stride))
            patch_num += 1

        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                verbose=verbose, **kwargs)
        
        self.head_nf = d_model * patch_num
        self.n_vars = c_in

     
        self.head = Flatten_Head(d_model, patch_num, 10, head_dropout=head_dropout)
        
    def forward(self, z):                                                                   # z: [bs x seq_len × nvars]
        # instance norm
        # if self.revin:                                                        
        #     z = self.revin_layer(z, 'norm')
        #     z = z.permute(0,2,1)                                                            # z: [bs x nvars × seq_len]
            
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z = self.backbone(z)       # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)                                                                    # z: [bs x target_window] 
        
        # denorm
        # if self.revin:                                                        
        #     z = self.revin_layer(z, 'denorm')
        return z
    
    