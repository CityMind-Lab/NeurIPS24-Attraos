import torch
from layers.poly_matrics import get_filter
from layers.op import transition
from layers import unroll
import numpy as np
from typing import List
from torch import Tensor
import math
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from scipy import linalg as la
from scipy import special as ss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HiPPO_LegT_Fast(nn.Module):
    def __init__(self, N, dt=1.0, discretization="bilinear", pred=96):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super().__init__()
        self.N = N
        max_len = int(1 / dt)
        A, B = transition("legt", N)
        C = np.ones((1, N))
        D = np.zeros((1,))
        # dt, discretization options
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)
        B = B.squeeze(-1)

        self.A = torch.Tensor(A).to(device)
        self.A_stack = self.A.unsqueeze(0).repeat(max_len, 1, 1)
        self.A_stack = nn.Parameter(self.A_stack)
        # self.A_stack = nn.Parameter(torch.randn(max_len, N, N)* 1e-2)

        self.B = torch.Tensor(B).to(device)
        self.B = nn.Parameter(self.B)
        # self.B = nn.Parameter(torch.randn(N)* 1e-2)
        vals = np.arange(0.0, 1.0, min(dt,1/pred))
        eval_matrix = torch.Tensor(
            ss.eval_legendre(np.arange(N)[:, None], - 2 * vals + 1).T
        ).to(device)
        self.C = nn.Parameter(eval_matrix[-pred :, :].T)
        # self.eval_matrix = nn.Parameter(torch.randn(max_len, N)* 1e-2)

    def forward(self, inputs, fast=False):  # torch.Size([128, 1, 1]) -
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        c = torch.zeros(inputs.shape[:-1] + tuple([self.N])).to(
            device
        )  # torch.Size([1, 256])
        if fast:
            B, C, L = inputs.shape
            u = inputs.unsqueeze(-1) * self.B
            u = u.permute(2, 0, 1, 3).reshape(L, B * C, -1)
            # u --> (length, ..., N)
            result = unroll.variable_unroll_matrix(self.A_stack, u)
            return result.reshape(L, B, C, -1)
        else:
            cs = []
            for f in inputs.permute([-1, 0, 1]):
                f = f.unsqueeze(-1)
                # f: [1,1]
                new = f @ self.B.unsqueeze(0)  # [B, D, H, 256]
                c = F.linear(c, self.A) + new
                # c = [1,256] * [256,256] + [1, 256]
                cs.append(c)
            return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        a = (self.C @ c.unsqueeze(-1)).squeeze(-1)
        return a


def compl_mul1d(x, weights):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", x, weights)


class sparseKernelFT1d(nn.Module):
    def __init__(self, k, alpha, c=1, nl=1, initializer=None, **kwargs):
        super(sparseKernelFT1d, self).__init__()

        self.modes1 = alpha
        self.scale = 1 / (c * k * c * k)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.cfloat)
        )
        self.weights1.requires_grad = True
        self.k = k

    def forward(self, x):
        B, N, c, k = x.shape  # (B, N, c, k)

        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        x_fft = torch.fft.rfft(x)
        # Multiply relevant Fourier modes
        l = min(self.modes1, N // 2 + 1)
        out_ft = torch.zeros(B, c * k, N // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :l] = compl_mul1d(x_fft[:, :, :l], self.weights1[:, :, :l])

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=N)
        x = x.permute(0, 2, 1).view(B, N, c, k)
        return x


class MDMU(nn.Module):
    def __init__(
        self, level=3, k=32, phase_dim=1, modes=64, base="legendre", length=336, pred=96
    ):
        super().__init__()

        self.order = k
        self.level = level
        self.length = length
        if base == "learned_tri":
            H0 = torch.tril(torch.randn(k, k) * 1e-2)
            H1 = H0 * 1
            for i in range(k):
                for j in range(k):
                    if (i + j) % 2 != 0:
                        H1[i, j] *= -1
            H0r = H0 * 1
            H1r = H1 * 1
            # 上三角
            G0 = torch.triu(torch.randn(k, k) * 1e-2)
            G1 = G0 * 1
            for i in range(k):
                for j in range(k):
                    if (i + j) % 2 == 0:
                        G1[i, j] *= -1
            G0r = G0 * 1
            G1r = G1 * 1

            G0 = nn.Parameter(G0)
            G1 = nn.Parameter(G1)
            H0 = nn.Parameter(H0)
            H1 = nn.Parameter(H1)

            H0r = nn.Parameter(H0r)
            G0r = nn.Parameter(G0r)
            H1r = nn.Parameter(H1r)
            G1r = nn.Parameter(G1r)

        elif base == "learned_1":
            H0 = nn.Parameter(torch.randn(k, k) * 1e-2)
            H1 = nn.Parameter(torch.randn(k, k) * 1e-2)
            G0 = nn.Parameter(torch.randn(k, k) * 1e-2)
            G1 = nn.Parameter(torch.randn(k, k) * 1e-2)
            H0r = nn.Parameter(torch.randn(k, k) * 1e-2)
            G0r = nn.Parameter(torch.randn(k, k) * 1e-2)
            H1r = nn.Parameter(torch.randn(k, k) * 1e-2)
            G1r = nn.Parameter(torch.randn(k, k) * 1e-2)

        elif base == "learned_2":
            H0 = nn.Parameter(torch.randn(k, k) * 1e-2)
            H1 = nn.Parameter(torch.randn(k, k) * 1e-2)
            G0 = nn.Parameter(torch.randn(k, k) * 1e-2)
            G1 = nn.Parameter(torch.randn(k, k) * 1e-2)
            H0r = H0
            G0r = G0
            H1r = H1
            G1r = G1

        self.register_buffer("W_down_c", torch.cat((H0.T, H1.T)))
        self.register_buffer("W_down_s", torch.cat((G0.T, G1.T)))
        self.register_buffer("W_up_0", torch.cat((H0r, G0r)))
        self.register_buffer("W_up_1", torch.cat((H1r, G1r)))
        self.win0 = HiPPO_LegT_Fast(N=k, dt=1.0 / length, pred=pred)

    def forward(self, x):
        pass

