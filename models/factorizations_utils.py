import torch
import torch._lowrank
from typing import Tuple, Optional


def svd_decomposition(
    A: torch.Tensor,
    randomized: bool,
    num_ranks: int,
    num_oversampling: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if A.ndim != 2:
        raise ValueError(f"Expected 2D Matrix, but got {A.ndim}.")

    if randomized is False:
        U, S, VT = torch.linalg.svd(A, full_matrices=False)
    elif randomized is True:
        U, S, V = torch.svd_lowrank(A, num_ranks + num_oversampling)
        # https://pytorch.org/docs/stable/_modules/torch/_lowrank.html#svd_lowrank
        VT = V.mH
    else:
        raise ValueError(f"`randomized` {randomized} not supported")

    S_sqrt = torch.sqrt(S)
    L1 = U * S_sqrt.unsqueeze(dim=0)
    L2 = VT * S_sqrt.unsqueeze(dim=1)
    L1k = L1[:, :num_ranks]
    L2k = L2[:num_ranks, :]
    return L1k, L2k


def weighted_svd_decomposition(
    A: torch.Tensor,
    W: torch.Tensor,
    heuristic: Optional[str],
    randomized: bool,
    num_ranks: int,
    num_oversampling: int = 5,
    normalize: bool = False,
    reduce_before_sqrt: bool = True,  # seems to be better empirically
) -> Tuple[torch.Tensor, torch.Tensor]:
    if A.ndim != 2 or W.ndim != 2:
        raise ValueError(f"Expected 2D Matrix, but got {A.ndim} and {W.ndim}.")
    if A.shape != W.shape:
        raise ValueError(f"Expected A.shape == W.shape, but got {A.shape} and {W.shape}.")

    if heuristic is None:
        heuristic = "two-sided"
    if heuristic not in ["none", "row", "col", "two-sided"]:
        raise ValueError(f"Heuristic {heuristic} not supported.")

    if normalize is True:
        W = W / torch.linalg.norm(W, ord="fro")

    if reduce_before_sqrt is True:
        # (A.shape[0], 1)
        W1 = torch.sqrt(torch.mean(W, dim=1, keepdim=True))
        # (1, A.shape[1])
        W2 = torch.sqrt(torch.mean(W, dim=0, keepdim=True))
    else:
        # (A.shape[0], 1)
        W1 = torch.mean(torch.sqrt(W), dim=1, keepdim=True)
        # (1, A.shape[1])
        W2 = torch.mean(torch.sqrt(W), dim=0, keepdim=True)

    if heuristic == "none":
        A_tilde = A
    elif heuristic == "row":
        A_tilde = W1 * A
    elif heuristic == "col":
        A_tilde = A * W2
    elif heuristic == "two-sided":
        A_tilde = W1 * A * W2
    else:
        raise ValueError

    L1_tilde, L2_tilde = svd_decomposition(
        A_tilde,
        randomized=randomized,
        num_ranks=num_ranks,
        num_oversampling=num_oversampling)

    if heuristic == "none":
        L1 = L1_tilde
        L2 = L2_tilde
    elif heuristic == "row":
        L1 = L1_tilde / W1
        L2 = L2_tilde
    elif heuristic == "col":
        L1 = L1_tilde
        L2 = L2_tilde / W2
    elif heuristic == "two-sided":
        L1 = L1_tilde / W1
        L2 = L2_tilde / W2
    else:
        raise ValueError

    return L1, L2
