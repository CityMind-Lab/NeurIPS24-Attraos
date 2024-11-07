import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN
from layers.Non_Para_Embed import Embeddings
import math

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        ### add ###
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.Embedding = Embeddings(configs)
        self.emb = configs.emb_type
        self.patch_len = configs.patch_len

        # Encoder
        if configs.emb_type == 'ori':
            self.evo_linear = nn.Linear(configs.seq_len, configs.pred_len)
            if self.task_name == 'classification':
                self.projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)
        elif configs.emb_type == 'PSR':
            patch_num = self.Embedding.num_patches
            pred_PSR_len = self.pred_len - (configs.PSR_dim-1) * configs.PSR_delay
            pred_patch_num = math.ceil((pred_PSR_len - configs.patch_len) / configs.stride) + 1
            self.evo_linear = nn.Linear(patch_num, pred_patch_num)
            if self.task_name == 'classification':
                self.evo_linear = nn.Linear(patch_num, patch_num)
                self.projection = nn.Linear(configs.enc_in * patch_num * configs.patch_len * configs.PSR_dim, configs.num_class)
        elif configs.emb_type == 'Ndiff':
            patch_num = self.Embedding.num_patches
            pred_patch_num = math.ceil((self.pred_len - configs.patch_len) / configs.stride) + 1
            self.evo_linear = nn.Linear(patch_num, pred_patch_num)
            if self.task_name == 'classification':
                self.evo_linear = nn.Linear(patch_num, patch_num)
                self.projection = nn.Linear(configs.enc_in * patch_num * configs.patch_len * (configs.N_diff+1), configs.num_class)

        self.rev = RevIN(configs.c_out)

        ### add classification
        

    def forecasting(self, x, y, x1, y1):
        B, T, C = x.shape
        x = self.rev(x, 'norm')

        # Embedding
        x = self.Embedding.System_reconstruction(x) # [*, P, D]
        # print(x.shape)

        # Encoder
        x = self.evo_linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        # print(x.shape)
        
        # Decoder
        x = self.Embedding.inverse(x, B=B, T=T, C=C)
        # print(x.shape)

        pred = self.rev(x, 'denorm')  # [B, T, C]
        return pred
    
    # def forecasting(self, x, y, x1, y1):
    #     B, T, C = x.shape
    #     x = self.rev(x, 'norm')
    #     P = T//self.patch_len

    #     x = x.reshape(B, P, -1) # (B, P, *)


    #     # Encoder
    #     x = self.evo_linear(x.permute(0, 2, 1)).permute(0, 2, 1)
    #     # print(x.shape)
        
    #     # Decoder
    #     x = x.reshape(B, T, C)

    #     pred = self.rev(x, 'denorm')  # [B, T, C]
    #     return pred
    
    

    def classification_original(self, x, x_mark_enc, x_dec, x_mark_dec):
        bs, _, n_vars = x.shape
        x = self.rev(x, 'norm')

        # Embedding
        if self.emb == 'ori':
            x = x
        else:
            x = self.Embedding.System_reconstruction(x) # [*, P, D]

        # Encoder (whether)
        x = self.evo_linear(x.permute(0, 2, 1)).permute(0, 2, 1) # [*, P, D]

        # Decoder
        output = self.Embedding.Linear_classification(x, bs, n_vars, self.projection)

        return output
    
    def classification(self, x, x_mark_enc, x_dec, x_mark_dec):
        bs, _, n_vars = x.shape
        # x = self.rev(x, 'norm')

        # Embedding
        if self.emb == 'ori':
            x = x
        else:
            x = self.Embedding.System_reconstruction(x) # [*, P, D]

        # Encoder (whether)
        x = self.evo_linear(x.permute(0, 2, 1)).permute(0, 2, 1) # [*, P, D]

        # Decoder
        output = self.Embedding.Linear_classification(x, bs, n_vars, self.projection)

        return output
    
    def classification_map(self, x ,x_mark_enc, x_dec, x_mark_dec):

        x = self.rev(x, 'norm')

        x = self.dropout(x)

        output = x.reshape(x.shape[0], -1)

        output = self.projection(output)

        return output

    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask = None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecasting(x_enc,x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.forecasting(x_enc,x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.forecasting(x_enc,x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc,x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # [B, N]
        return None