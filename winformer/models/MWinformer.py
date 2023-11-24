from math import ceil

import torch
from einops import rearrange
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding,DataEmbedding
from layers.MWinformer_TCFeatureLayer import TFeatureLayer#,CFeatureLayer

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        # self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        # x = self.flatten(x)
        x = self.linear(x)
        # x = self.dropout(x)
        return x
class FlattenHead2(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        # self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Conv1d(nf,nf,kernel_size=3,stride=1,padding=1)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        # x = self.flatten(x)
        x = self.linear(x)
        # x = self.dropout(x)
        return x

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=4, stride=2):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        padding = stride

        self.pad_in_len = ceil(1.0 * self.seq_len / patch_len) * patch_len #embed 之后的长度
        self.pad_out_len = int((configs.seq_len - patch_len) / stride + 2)
        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding,self.enc_in,self.pad_out_len, configs.dropout)
        # self.patch_embedding =DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout)
        self.Tlayers = nn.ModuleList()
        for i in range(1):
            self.Tlayers.append(TFeatureLayer(configs, self.pad_out_len,configs.d_model,configs.n_heads,
                                  configs.d_ff,configs.e_layers,configs.dropout,False,seg_num=configs.seg_num)
                               )
        self.Clayers = nn.ModuleList()
        for i in range(1):
            self.Clayers.append(TFeatureLayer(configs,configs.enc_in ,configs.pred_len,configs.n_heads,
                                          configs.pred_len,1,configs.dropout,False,seg_num=configs.seg_num)
                               )
        # Prediction Head
        self.head_nf = configs.d_model*self.pad_out_len#//2**(configs.e_layers)
        # self.head_nf = self.seq_len#//2**(configs.e_layers)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head1=FlattenHead(configs.enc_in,self.head_nf,configs.pred_len)
            self.head2= FlattenHead2(configs.enc_in, configs.enc_in, configs.enc_in)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        # enc_out = self.patch_embedding(x_enc,x_mark_enc)
        B,N,D=enc_out.shape
        # enc_out = rearrange(enc_out, '(b v) seg_num d_model -> b v seg_num d_model', v = n_vars)
        # Encoder
        # z: [bs * nvars x
        # x d_model]
        # enc_out, attns = self.encoder(enc_out)
        for layer in self.Tlayers:
            enc_out,attns = layer(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out=enc_out.reshape(B//n_vars,n_vars,N,enc_out.shape[-1])
        enc_out = rearrange(enc_out, 'b v seg_num d_model -> b v (d_model seg_num)')

        out1=self.head1(enc_out)

        for layer in self.Clayers:
            out2,attns = layer(out1)
        out2=self.head2(out2).transpose(2,1)
        # z: [bs x nvars x d_model x patch_num]

        # Decoder
        dec_out =out1.transpose(2,1) # z: [bs x nvars x target_window]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out1 = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        dec_out =out2# z: [bs x nvars x target_window]
        # dec_out = dec_out.permute(0, 2, 1)
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out2 = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out1,dec_out2


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out1,dec_out2 = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out1[:, -self.pred_len:, :],dec_out2[:, -self.pred_len:, :]   # [B, L, D]

        return None


    def flops(self):
        flops = 0
        flops += self.patch_embedding.flops()

        for layer in self.Tlayers:
            flops+=layer.flops()*self.enc_in
        for layer in self.Clayers:
            flops += layer.flops()
        flops += self.head_nf*self.pred_len
        flops += self.pred_len*self.pred_len
        return flops

