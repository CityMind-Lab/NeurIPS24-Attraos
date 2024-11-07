import math

import torch
import torch.nn.functional as F

"""

An implementation of the parallel scan operation in PyTorch (Blelloch version).
Please see docs/pscan.ipynb for a detailed explanation of what happens here.

"""

def npo2(len):
    """
    Returns the next power of 2 above len
    """

    return 2 ** math.ceil(math.log2(len))

def pad_npo2(X):
    """
    Pads input length dim to the next power of 2

    Args:
        X : (B, L, D, N)

    Returns:
        Y : (B, npo2(L), D, N)
    """

    len_npo2 = npo2(X.size(1))
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
    return F.pad(X, pad_tuple, "constant", 0)

# class PScan_H(torch.autograd.Function):
#     @staticmethod
#     def pscan(A, X, H):
#         # A : (B, D, L, N)
#         # X : (B, D, L, N)
#         # H : (B, D, L, N) hierarchical projection matrix

#         B, D, L, _ = A.size()
#         num_steps = int(math.log2(L))

#         Aa = A
#         Xa = X
#         Ha = H
#         for _ in range(num_steps-2):
#             T = Xa.size(2)
#             Aa = Aa.view(B, D, T//2, 2, -1)
#             Xa = Xa.view(B, D, T//2, 2, -1)
#             Ha = Ha.view(B, D, T//2, 2, -1)
            
#             # Modify the scan operation to include hierarchical projection
#             Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
#             Xa[:, :, :, 1] = Ha[:, :, :, 1].mul(Xa[:, :, :, 1]) # Apply H here
#             Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

#             Aa = Aa[:, :, :, 1]
#             Xa = Xa[:, :, :, 1]
#             Ha = Ha[:, :, :, 1]  # Update H for next level

#         # Handle final nodes left
#         if Xa.size(2) == 4:
#             Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
#             Xa[:, :, 1] = Ha[:, :, 1].mul(Xa[:, :, 1]) # Apply H
#             Aa[:, :, 1].mul_(Aa[:, :, 0])

#             Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
#             Xa[:, :, 3] = Ha[:, :, 3].mul(Xa[:, :, 3]) # Apply H
#         elif Xa.size(2) == 2:
#             Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
#             Xa[:, :, 1] = Ha[:, :, 1].mul(Xa[:, :, 1]) # Apply H
#             return
#         else:
#             return

#     @staticmethod
#     def pscan_rev(A, X, H):
#         B, D, L, _ = A.size()
#         num_steps = int(math.log2(L))

#         Aa = A
#         Xa = X
#         Ha = H
#         for _ in range(num_steps-2):
#             T = Xa.size(2)
#             Aa = Aa.view(B, D, T//2, 2, -1)
#             Xa = Xa.view(B, D, T//2, 2, -1)
#             Ha = Ha.view(B, D, T//2, 2, -1)
            
#             Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
#             Xa[:, :, :, 0] = Ha[:, :, :, 0].mul(Xa[:, :, :, 0]) # Apply H
#             Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

#             Aa = Aa[:, :, :, 0]
#             Xa = Xa[:, :, :, 0]
#             Ha = Ha[:, :, :, 0]

#         if Xa.size(2) == 4:
#             Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
#             Xa[:, :, 2] = Ha[:, :, 2].mul(Xa[:, :, 2]) # Apply H
#             Aa[:, :, 2].mul_(Aa[:, :, 3])

#             Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))))
#             Xa[:, :, 0] = Ha[:, :, 0].mul(Xa[:, :, 0]) # Apply H
#         elif Xa.size(2) == 2:
#             Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
#             Xa[:, :, 0] = Ha[:, :, 0].mul(Xa[:, :, 0]) # Apply H
#             return
#         else:
#             return

#     @staticmethod
#     def forward(ctx, A_in, X_in, H):
#         L = X_in.size(1)

#         if L == npo2(L):
#             A = A_in.clone()
#             X = X_in.clone()
#             H = H.clone()
#         else:
#             A = pad_npo2(A_in)
#             X = pad_npo2(X_in)
#             H = pad_npo2(H)

#         A = A.transpose(2, 1)
#         X = X.transpose(2, 1)
#         H = H.transpose(2, 1)

#         PScan_H.pscan(A, X, H)

#         ctx.save_for_backward(A_in, X, H)
        
#         return X.transpose(2, 1)[:, :L]
    
#     @staticmethod
#     def backward(ctx, grad_output_in):
#         A_in, X, H = ctx.saved_tensors

#         L = grad_output_in.size(1)

#         if L == npo2(L):
#             grad_output = grad_output_in.clone()
#         else:
#             grad_output = pad_npo2(grad_output_in)
#             A_in = pad_npo2(A_in)
#             H = pad_npo2(H)

#         grad_output = grad_output.transpose(2, 1)
#         A_in = A_in.transpose(2, 1)
#         H = H.transpose(2, 1)
        
#         A = torch.nn.functional.pad(A_in[:, :, 1:], (0, 0, 0, 1))

#         PScan_H.pscan_rev(A, grad_output, H1, H2)


#         Q = torch.zeros_like(X)
#         Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])
#         gradH = torch.zeros_like(Q).transpose(2, 1)[:, :L]

#         return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]
    


class PScan_H(torch.autograd.Function):
    @staticmethod
    def pscan(A, X, H):
        # A : (B, D, L, N)
        # X : (B, D, L, N)
        # H : (B, D, L, N) hierarchical projection matrix

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        Ha = H
        for _ in range(num_steps-2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
            Ha = Ha.view(B, D, T//2, 2, -1)
            
            # Modify the scan operation to include hierarchical projection
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Xa[:, :, :, 1] = Ha[:, :, :, 1].mul(Xa[:, :, :, 1]) # Apply H here
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]
            Ha = Ha[:, :, :, 1]  # Update H for next level

        # Handle final nodes left
        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Xa[:, :, 1] = Ha[:, :, 1].mul(Xa[:, :, 1]) # Apply H
            Aa[:, :, 1].mul_(Aa[:, :, 0])

            Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
            Xa[:, :, 3] = Ha[:, :, 3].mul(Xa[:, :, 3]) # Apply H
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Xa[:, :, 1] = Ha[:, :, 1].mul(Xa[:, :, 1]) # Apply H
            return
        else:
            return

    @staticmethod
    def pscan_rev(A, X, H):
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        Ha = H
        for _ in range(num_steps-2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
            Ha = Ha.view(B, D, T//2, 2, -1)
            
            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Xa[:, :, :, 0] = Ha[:, :, :, 0].mul(Xa[:, :, :, 0]) # Apply H
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]
            Ha = Ha[:, :, :, 0]

        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Xa[:, :, 2] = Ha[:, :, 2].mul(Xa[:, :, 2]) # Apply H
            Aa[:, :, 2].mul_(Aa[:, :, 3])

            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))))
            Xa[:, :, 0] = Ha[:, :, 0].mul(Xa[:, :, 0]) # Apply H
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            Xa[:, :, 0] = Ha[:, :, 0].mul(Xa[:, :, 0]) # Apply H
            return
        else:
            return

    @staticmethod
    def forward(ctx, A_in, X_in, H):
        L = X_in.size(1)

        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
            H = H.clone()
        else:
            A = pad_npo2(A_in)
            X = pad_npo2(X_in)
            H = pad_npo2(H)

        A = A.transpose(2, 1)
        X = X.transpose(2, 1)
        H = H.transpose(2, 1)

        PScan_H.pscan(A, X, H)

        ctx.save_for_backward(A_in, X, H)
        
        return X.transpose(2, 1)[:, :L]
    
    @staticmethod
    def backward(ctx, grad_output_in):
        A_in, X, H = ctx.saved_tensors

        L = grad_output_in.size(1) #6


        if L == npo2(L):
            grad_output = grad_output_in.clone()
        else:
            grad_output = pad_npo2(grad_output_in)
            A_in = pad_npo2(A_in)
            H = pad_npo2(H)

        grad_output = grad_output.transpose(2, 1)
        A_in = A_in.transpose(2, 1)
        H = H.transpose(2, 1) # (224, 8, 64, 64)
        
        A = torch.nn.functional.pad(A_in[:, :, 1:], (0, 0, 0, 1))

        PScan_H.pscan_rev(A, grad_output, H)


        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])
        # gradH = torch.zeros_like(Q).transpose(2, 1)[:, :L]

        """print(Q.shape)
        print(Q.transpose(2, 1)[:, :L].shape)
        torch.Size([224, 64, 8, 64])
        torch.Size([224, 6, 64, 64])"""

        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L], H[:, :L]

    
pscan_H = PScan_H.apply
