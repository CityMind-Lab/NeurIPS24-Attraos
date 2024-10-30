import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy import linalg as la
from scipy import special as ss
from layers.Attraos_Blocks import HiPPO_LegT_Fast, MDMU, sparseKernelFT1d
from typing import List, Tuple
from torch import Tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):

    """
    when L=0, PSR_type=pass
    Attraos will become FiLM without MOE.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.task_name = configs.task_name
        self.PSR_dim = configs.PSR_dim
        self.PSR_type = configs.PSR_type
        self.PSR_delay = configs.PSR_delay
        self.PSR_enc_len = (configs.seq_len - (configs.PSR_dim - 1) * configs.PSR_delay)//8
        self.PSR_dec_len = (configs.pred_len - (configs.PSR_dim - 1) * configs.PSR_delay)//8
        self.pred_len = configs.pred_len
        self.level = configs.level
        self.modes = configs.modes
        self.order = configs.order
        if configs.PSR_type == "merged" or configs.PSR_type == "merged_seq":
            self.MOE_dim = nn.Linear(
                configs.PSR_dim * configs.enc_in, configs.M_PSR_dim
            )
            self.obs = nn.Linear(configs.M_PSR_dim, configs.c_out)

        self.MDMU = MDMU(
            level=3,
            k=self.order,
            modes=configs.modes,
            base="learned_1",
            length=self.PSR_enc_len,
            pred=self.PSR_dec_len
        )
        self.K = sparseKernelFT1d(k=self.order, alpha=self.modes, c=self.PSR_dim*8)
        self.if_patch = configs.Attraos_patch


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x_enc /= stdev

        B, T, C = x_enc.shape
        enc_PSR = PSR(x_enc, self.PSR_dim, self.PSR_delay, self.PSR_type)  # [BC, T, D]
        if self.if_patch:
            enc_PSR = enc_PSR.unfold(dimension=-2, size=8, step=8) # [BC, P, D, len]
            enc_PSR = enc_PSR.reshape(B*C, -1, self.PSR_dim*8) # [BC, P, D*len]

        # get finest memory:
        m = self.MDMU.win0(enc_PSR.transpose(1, 2), fast=True)  # [T, BC, D, k]
        m = m.permute(1, 0, 2, 3)  # [BC, T, D, k]

        # up projection and evolute by K:
        List_ms = torch.jit.annotate(List[Tensor], [])  # for s
        for i in range(self.level):
            m2 = torch.cat(
                [
                    m[:, ::2, :, :],
                    m[:, 1::2, :, :],
                ],
                -1,
            )
            mc = torch.matmul(m2, self.MDMU.W_down_c)
            ms = torch.matmul(m2, self.MDMU.W_down_s)
            m = mc
            List_ms += [self.K(ms)]
        # last mc
        m = self.K(m)

        # back projection:
        for i in range(self.level - 1, -1, -1):
            b, n, d, k = m.shape  # [BC, T, D, k]
            m = torch.cat((m, List_ms[i]), -1)
            ml = torch.matmul(m, self.MDMU.W_up_0)
            mr = torch.matmul(m, self.MDMU.W_up_1)
            m = torch.zeros(b, n * 2, d, k, device=mc.device)
            m[..., ::2, :, :] = ml
            m[..., 1::2, :, :] = mr
            # mc = m  # [BC, T, D, k]

        if self.PSR_enc_len >= self.PSR_dec_len:
            m = m[:, self.PSR_dec_len - 1, :, :]  # [BC, D, k]
        else:
            m = m[:, -1, :, :]
        dec_PSR = m @ self.MDMU.win0.C  # [BC, D, H]
        dec_out = dec_PSR.permute(0, 2, 1) # [BC, H, D]

        if self.if_patch:
            dec_out = dec_out.reshape(B*C, self.PSR_dec_len, self.PSR_dim, 8)
            dec_out = dec_out.permute(0, 1, 3, 2).reshape(B*C, -1, self.PSR_dim)

        dec_out = inverse_PSR(
            dec_out, self.PSR_dim, self.PSR_delay, B, C, self.PSR_type
        )

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev
        dec_out = dec_out + means
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        return None


def PSR(input_data, embedding_dim, delay, mode="indep"):
    batch_size, seq_length, input_channels = input_data.shape

    if mode == "pass":
        return input_data.permute(0, 2, 1).reshape(
            batch_size * input_channels, seq_length, 1
        )

    device = input_data.device
    len_embedded = seq_length - (embedding_dim - 1) * delay
    embedded_data = torch.zeros(
        batch_size, len_embedded, embedding_dim, input_channels, device=device
    )

    for i in range(embedding_dim):
        start_idx = i * delay
        end_idx = start_idx + len_embedded
        embedded_data[:, :, i, :] = input_data[:, start_idx:end_idx, :]

    if mode == "merged_seq":
        embedded_data = embedded_data.permute(0, 1, 3, 2).reshape(
            batch_size, len_embedded, -1
        )
    elif mode == "merged":
        embedded_data = embedded_data.reshape(batch_size, len_embedded, -1)
    else:  # independent
        embedded_data = embedded_data.permute(0, 3, 1, 2).reshape(
            batch_size * input_channels, len_embedded, embedding_dim
        )  # [BC, T, D]
    return embedded_data


def inverse_PSR(
    embedded_data, embedding_dim, delay, batch_size, input_channels, mode="merged_seq"
):
    if mode == "pass":
        return input_data.permute(0, 2, 1).reshape(
            batch_size * input_channels, seq_length, 1
        )
    device = embedded_data.device
    len_embedded = embedded_data.shape[1]
    seq_length = len_embedded + (embedding_dim - 1) * delay

    if mode == "merged_seq":
        embedded_data = embedded_data.reshape(
            batch_size, len_embedded, input_channels, embedding_dim
        ).permute(0, 1, 3, 2)

    elif mode == "merged":
        embedded_data = embedded_data.reshape(
            batch_size, len_embedded, embedding_dim, input_channels
        )
    else:  # independent
        embedded_data = embedded_data.reshape(
            batch_size, input_channels, len_embedded, embedding_dim
        ).permute(0, 2, 3, 1)

    input_data = torch.zeros(batch_size, seq_length, input_channels, device=device)

    for i in range(embedding_dim):
        start_idx = i * delay
        end_idx = start_idx + len_embedded
        input_data[:, start_idx:end_idx, :] = embedded_data[:, :, i, :]

    return input_data


if __name__ == "__main__":

    class Configs(object):
        seq_len = 336
        pred_len = 96
        enc_in = 7
        dec_in = 7
        d_model = 16
        embed = "timeF"
        dropout = 0.1
        freq = "h"
        factor = 1
        n_heads = 8
        c_out = 7
        activation = "gelu"
        PSR_dim = 1
        PSR_type = "indep"
        PSR_delay = 12
        poly_order = 64
        level = 3
        task_name = "long_term_forecast"
        time_mapping = False

    configs = Configs()
    # model = Model(configs).to(device)
    model = Model(configs)

    enc = torch.randn([16, configs.seq_len, configs.enc_in])
    out = model.forward(enc, None, None, None)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("model size", count_parameters(model) / (1024 * 1024))
    print("input shape", enc.shape)
    print("output shape", out[0].shape)

