import torch
import torch.nn as nn
from einops import rearrange, repeat
from layers.SelfAttention_Family import AttentionLayer,FullAttention,CrossAttentionLayer,CrossAttention


def window_partition(x, seg_num):

    B_, seqlen, d_model = x.shape
    c=seqlen//seg_num
    # pad_num = int(seqlen - seg_num*(seqlen//seg_num))
    pad_num = int((seqlen//seg_num+1)*seg_num - seqlen)
    if pad_num != 0:
        x = torch.cat((x, x[:, -pad_num:, :]), dim=-2)
    seqlen += pad_num
    x = x.view(B_, seg_num, seqlen//seg_num,  d_model)
    # windows = x.contiguous().view(-1, window_size, d_model)
    return x, pad_num


def window_reverse(x, padding):

    B,seg_num,seqlen,d_model = x.shape
    L = seg_num*seqlen-padding
    x = x.contiguous().view(B, seg_num*seqlen, -1)
    x = x[:,:L,:]
    return x

class SegMerging(nn.Module):
    '''
    responsible for the patch fusion problem, regradless of winsize padding problem
    '''
    def __init__(self, d_model, scale, win_size, seg_num,norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.scale = scale   #融合窗口的含义
        self.seg_num = seg_num
        if seg_num > self.win_size:
            self.linear_trans = nn.Linear(scale * d_model, d_model)
        else:
            self.linear_trans = nn.Linear( d_model, d_model)
        self.norm = norm_layer(scale * d_model)
    def forward(self, x):
        batch_size, ts_d, seg_num, d_model = x.shape
        #首先需要判断是否需要合并
        if seg_num > self.win_size:
            pad_num = seg_num % self.scale
            if pad_num != 0:
                pad_num = self.scale - pad_num
                x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)
            seg_to_merge = []
            for i in range(self.scale):
                seg_to_merge.append(x[:, :, i::self.scale, :])
            x = torch.cat(seg_to_merge, -1)
            x = self.norm(x)
        #之后进行维度扩张
        x = self.linear_trans(x)

        return x







class TFeatureLayer(nn.Module):
    def __init__(self, configs, seqlen, d_model, n_heads, d_ff, depth, dropout, contrac, \
                 seg_num=2, factor=10):
        super(TFeatureLayer, self).__init__()

        self.seg_num = seg_num
        self.seqlen = seqlen
        pad_num = int((seqlen // seg_num + 1) * seg_num - seqlen)
        self.sseqlen = (seqlen + pad_num) //seg_num
        self.encode_layers = nn.ModuleList()
        self.downconv_layers = nn.ModuleList()
        self.contrac=contrac
        for ii in range(int(depth)):
            if contrac:
                self.encode_layers.append(MWinAttentionLayer(configs,  self.sseqlen,self.seg_num, factor, d_model//(ii+1), n_heads, \
                                                             d_ff//(ii+1), dropout))
                # self.encode_layers.append(    AttentionLayer(
                #         FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                #                       output_attention=configs.output_attention), configs.d_model, configs.n_heads))
            else:
                self.encode_layers.append(MWinAttentionLayer(configs,  self.sseqlen,self.seg_num, factor, d_model, n_heads, \
                                                             d_ff, dropout))
                # self.encode_layers.append(    AttentionLayer(
                #         FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                #                       output_attention=configs.output_attention), configs.d_model, configs.n_heads))
        for ii in range(int(depth)):
            self.downconv_layers.append(ConvLayer(self.seqlen,d_model//(ii+1)))
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        batch_size, seg_num, d_model = x.shape
        #划分window
        # x=rearrange(x,'b ts_d seqlen d_model -> b ts_d (seqlen d_model)')

        #注意力
        for layer,down in zip(self.encode_layers,self.downconv_layers):
            x, pad_num = window_partition(x, self.seg_num)
            x = layer(x)
        #合并window
            x = window_reverse(x,pad_num)
            if self.contrac:
                x=down(x)
        # x = rearrange(x, '(b ts_d) seg_num d_model -> b ts_d seg_num d_model',ts_d=ts_d)
        return x, None
    def flops(self):
        flops = 0
        for layer in self.encode_layers:
            flops += layer.flops()
        return flops


# class CFeatureLayer(nn.Module):
#     def __init__(self, configs, seqlen, d_model, n_heads, d_ff, depth, dropout, \
#                  seg_num=2, factor=10):
#         super(CFeatureLayer, self).__init__()
#         self.seg_num = seg_num
#         self.seqlen = seqlen
#         self.encode_layers = nn.ModuleList()
#         for ii in range(int(depth)):
#             self.encode_layers.append(MWinAttentionLayer(configs,  self.seqlen, factor, d_model, n_heads, \
#                                                              d_ff, dropout))
#
#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         batch_size,  seg_num, d_model = x.shape
#         x,pad_num = window_partition(x,self.seg_num)
#         for layer in self.encode_layers:
#             x=layer(x)
#         x = window_reverse(x,pad_num)
#         return x, None


class ConvLayer(nn.Module):
    def __init__(self, c_in,d):
        super(ConvLayer, self).__init__()
        # self.downConv = nn.Conv1d(in_channels=c_in,
        #                           out_channels=c_in,
        #                           kernel_size=3,
        #                           stride=2,
        #                           padding=1,
        #                           padding_mode='circular')
        self.downlinear = nn.Linear(in_features=d,out_features=d//2)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x = self.downConv(x)
        x = self.downlinear(x)
        x = self.norm(x)
        # x = self.activation(x)
        # x = self.maxPool(x)
        # x = x.transpose(1, 2)
        return x

class MWinAttentionLayer(nn.Module):
    def __init__(self, configs,
                  win_size, win_num,factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(MWinAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.d_ff = d_ff
        self.winsize=win_size
        self.winnum = win_num
        self.enc_in = configs.enc_in
        self.head = n_heads
        self.win_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.cross_attention = CrossAttentionLayer(CrossAttention(False,win_size, d_model//n_heads,False,configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)




        # self.MLP1 = nn.Sequential(nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1),
        #                           nn.GELU(),
        #                           nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1))
        # self.MLP2 = nn.Sequential(nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1),
        #                           nn.GELU(),
        #                           nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1))
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))


    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Window Attention: Directly apply MSA to each dimension
        B_,win_num,win_size,d_model = x.shape
        x1_in = rearrange(x,'B win_num win_size d_model -> (B win_num) win_size d_model')
        x1_out, attn = self.win_attention(
            x1_in, x1_in, x1_in, attn_mask=None, tau=None, delta=None
        )
        x1_out = x1_in + self.dropout(x1_out)
        x1_out = self.norm1(x1_out)
        x1_out = x1_out + self.dropout(self.MLP1(x1_out))
        # x1_out = x1_out + self.dropout(self.MLP1(x1_out.transpose(2,1)).transpose(2,1))
        x2_in = self.norm2(x1_out)

        # Mem Windwo Attention:
        x2_in = rearrange(x2_in,'(B win_num) win_size d_model -> B win_num win_size d_model',win_num=win_num)
        b,n,w,d=x2_in.shape
        x2_list = []
        if win_num==1:
            return x2_in
        elif win_num==2:
            mem = x2_in[:, 0, :, :]
            y = x2_in[:,1,:,:]
            mem_,attn,y_,_ = self.cross_attention(mem,mem,y,y,)
            # mem = self.GRUNet(y, mem)
            x2_list.append(((mem+mem_)/2).unsqueeze(1))
            x2_list.append(((y+y_)/2).unsqueeze(1))
        else:
            mem = x2_in[:, 0, :, :]
            x2_list.append(mem.unsqueeze(1))
            # x2_list.append(mem)
            for ii in range(1,win_num):
                y = x2_in[:,ii,:,:]
                y_,attn,mem_,_ = self.cross_attention(mem,mem,y,y,)
                mem=(mem+mem_)/2
                # mem=(mem_)
                y=(y+y_)/2
                # y=(y_)
                x2_list.append(y.unsqueeze(1))
        x2_out = torch.cat([out for out in x2_list], dim=1)
        x2_out = x2_in + self.dropout(x2_out)
        x2_out = self.norm3(x2_out)

        x2_out = x2_out + self.dropout(self.MLP2(x2_out))
        # x2_out = x2_out + self.dropout(self.MLP2(x2_out.view(-1,w,d).transpose(2,1)).transpose(2,1).reshape(b,n,w,d))
        final_out = self.norm4(x2_out)
        return final_out


    def flops(self):
        flops = 0
        N = self.winsize
        #W - MSA
        flops += self.winnum * self.win_attention.flops(N)
        # mlp
        flops += 2 * N * self.d_model * self.d_ff
        # norm2
        flops += self.d_model * N* 2
        # W-MCA
        if self.winnum == 1:
            return  flops
        if self.winnum==2:
            flops += self.cross_attention.flops(N)
        else:
            flops += self.winnum * self.cross_attention.flops(N)
        # mlp
        flops += 2 * N * self.d_model * self.d_ff
        # norm2
        flops += self.d_model * N * 2


        return flops




