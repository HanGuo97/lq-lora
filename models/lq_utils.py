import torch
import numpy as np
import gurobipy as gp
import scipy.optimize._milp as scipy_milp
import scipy.optimize._optimize as scipy_optimize
import scipy.optimize._constraints as scipy_constraints
from tqdm.auto import tqdm
from typing import Tuple, List, Optional

from models import misc_utils
from models import quantization_utils
from models import quantization_utils_2
from models.factorizations_utils import (
    svd_decomposition,
    weighted_svd_decomposition)
from models.quantization_utils import (
    QuantConfig,
    estimate_storage_from_config)


def maybe_sparsify_or_quantize(
    A: torch.Tensor,
    qconfig: Optional[QuantConfig],
    fake: bool = False,
) -> torch.Tensor:
    if qconfig is None:
        return A
    if fake is True:
        misc_utils.swarn(f"Fake quantization", fg="red")
        return quantization_utils.quantize(
            A,
            method="blockwise-nf",
            qconfig=qconfig)
    else:
        misc_utils.swarn(f"Real quantization", fg="red")
        return quantization_utils_2.quantize(
            A,
            method="blockwise-nf",
            qconfig=qconfig)


def lowrank_quantized_sparse_decomposition(
    A: torch.Tensor,
    num_ranks: int,
    num_iterations: int = 100,
    num_oversampling: int = 10,
    randomized: bool = True,
    qconfig: Optional[QuantConfig] = None,
    W: Optional[torch.Tensor] = None,
    heuristic: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

    misc_utils.swarn(
        f"Applying Robust PCA:\n"
        f"- num_ranks={num_ranks}\n"
        f"- num_iterations={num_iterations}\n"
        f"- num_oversampling={num_oversampling}\n"
        f"- randomized={randomized}\n"
        f"- qconfig={qconfig}\n"
        f"- W={W is not None}\n"
        f"- heuristic={heuristic}",
        fg="yellow")

    if len(A.shape) != 2:
        raise ValueError
    if num_iterations < 1:
        raise ValueError

    last_error = A.new_tensor(float("inf"))
    Q = torch.zeros_like(A)
    for _ in range(num_iterations):
        if W is None:
            L1, L2 = svd_decomposition(
                A - Q,
                randomized=randomized,
                num_ranks=num_ranks,
                num_oversampling=num_oversampling)
        else:
            L1, L2 = weighted_svd_decomposition(
                A - Q,
                W=W,
                heuristic=heuristic,
                randomized=randomized,
                num_ranks=num_ranks,
                num_oversampling=num_oversampling)

        Q = maybe_sparsify_or_quantize(
            A - L1 @ L2,
            qconfig=qconfig)

        A_ = L1 @ L2 + Q
        if W is None:
            error = torch.linalg.norm(A - A_, ord="fro")
        else:
            error = torch.linalg.norm(torch.sqrt(W) * (A - A_), ord="fro")
        # (TODO) return the last outputs, instead of the
        # current outputs since this is slightly worse.
        # print(t, last_error, error, error > last_error)
        if error > last_error:
            break
        last_error = error

    return L1, L2, Q, None


def lowrank_quantized_sparse_decomposition_maybe_cast(
    A: torch.Tensor,
    num_ranks: int,
    num_iterations: int = 100,
    num_oversampling: int = 10,
    randomized: bool = True,
    qconfig: Optional[QuantConfig] = None,
    W: Optional[torch.Tensor] = None,
    heuristic: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

    if A.dtype in [torch.float16]:
        old_dtype = A.dtype
        A = A.to(dtype=torch.float32)
    else:
        old_dtype = None

    if W is not None:
        W = W.to(dtype=A.dtype, device=A.device)

    L1, L2, Q, unused = lowrank_quantized_sparse_decomposition(
        A=A,
        num_ranks=num_ranks,
        num_iterations=num_iterations,
        num_oversampling=num_oversampling,
        randomized=randomized,
        qconfig=qconfig,
        W=W,
        heuristic=heuristic)

    if old_dtype is not None:
        L1 = L1.to(dtype=old_dtype)
        L2 = L2.to(dtype=old_dtype)
        Q = Q.to(dtype=old_dtype)

    return L1, L2, Q, unused


@torch.no_grad()
def prepare_data_for_qconfig_assignments(
    parameters: List[torch.Tensor],
    qconfigs: List[QuantConfig],
    num_ranks: Optional[int],
    inputs: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    costs = torch.zeros(
        len(parameters),
        len(qconfigs))
    weights = torch.zeros(
        len(parameters),
        len(qconfigs))

    for i0, param in enumerate(tqdm(parameters)):
        for i1, qconfig in enumerate(qconfigs):
            A = param.cuda()
            if inputs is None:
                W = None
            else:
                W = inputs[i0].to(device=A.device)

            if num_ranks is None:
                A_ = maybe_sparsify_or_quantize(
                    A=A,
                    qconfig=qconfig)
            else:
                L1, L2, Q, _ = lowrank_quantized_sparse_decomposition_maybe_cast(
                    A=A,
                    num_ranks=num_ranks,
                    qconfig=qconfig,
                    W=W,
                    heuristic="two-sided")
                A_ = L1 @ L2 + Q

            errors = A - A_
            if W is not None:
                errors = torch.sqrt(W) * errors

            costs[i0, i1] = torch.linalg.norm(errors, ord="fro") ** 2
            weights[i0, i1] = estimate_storage_from_config(A, qconfig=qconfig)

    return costs, weights


def compute_qconfig_assignments(
    budget: float,
    costs: torch.Tensor,
    weights: torch.Tensor,
    num_chunks: int,
) -> Tuple[float, torch.Tensor]:
    costs_np = (
        costs
        .numpy(force=True)
        .reshape(costs.shape[0], -1))
    weights_np = (
        weights
        .numpy(force=True)
        .reshape(weights.shape[0], -1))
    costs_list = np.split(
        costs_np,
        indices_or_sections=num_chunks,
        axis=0)
    weights_list = np.split(
        weights_np,
        indices_or_sections=num_chunks,
        axis=0)

    results = []
    for _costs, _weights in zip(costs_list, weights_list):
        result = mip_solve(
            budget=budget / float(num_chunks),
            costs=_costs,
            weights=_weights,
            backend="grurobi")
        results.append(result)

    assignments_cost = sum([r.fun for r in results])
    assignments = np.concatenate([r.x for r in results], axis=0)
    assignments = assignments.reshape(costs.shape)
    return assignments_cost, torch.from_numpy(assignments)


def mip_solve(
    budget: float,
    costs: np.ndarray,
    weights: np.ndarray,
    backend: str,
) -> scipy_optimize.OptimizeResult:
    if backend not in ["scipy", "grurobi"]:
        raise ValueError(f"Unknown backend: {backend}")

    N = costs.shape[0]
    coefficients = costs.reshape(-1)
    A_upperbound = weights.reshape(1, -1)
    A_equality = np.zeros_like(
        weights,
        shape=(N,) + weights.shape)
    A_equality[
        np.arange(N),
        np.arange(N), :] = 1.
    A_equality = A_equality.reshape(N, -1)

    if backend == "scipy":
        constraints = [
            scipy_constraints.LinearConstraint(
                A=A_upperbound,
                lb=0,
                ub=budget),
            scipy_constraints.LinearConstraint(
                A=A_equality,
                lb=1,
                ub=1),
        ]
        integrality = np.ones_like(coefficients)
        bounds = scipy_constraints.Bounds(0, 1)
        return scipy_milp.milp(
            c=coefficients,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds)

    # https://www.gurobi.com/documentation/10.0/quickstart_mac/cs_example_matrix1_py.html
    # https://stackoverflow.com/questions/64247609/how-to-use-mipgap-and-timelimit-from-gurobi-in-python
    if backend == "grurobi":
        grurobi_model = gp.Model()
        grurobi_model.setParam(
            paramname="Timelimit",
            newval=60)  # type: ignore
        x = grurobi_model.addMVar(
            shape=coefficients.shape,
            vtype=gp.GRB.BINARY,
            name="x")
        grurobi_model.setObjective(
            coefficients @ x,
            gp.GRB.MINIMIZE)
        grurobi_model.addConstr(
            (A_upperbound @ x) <= budget,
            name="upperbounds")
        grurobi_model.addConstr(
            (A_equality @ x) == 1.,
            name="equalities")
        grurobi_model.optimize()
        return scipy_optimize.OptimizeResult(
            x=x.X,
            fun=grurobi_model.ObjVal)

    raise ValueError
