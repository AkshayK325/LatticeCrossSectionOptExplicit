
import torch
import numpy as np
import scipy.sparse as sp

class DifferentiableSparseSolve(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A_sparse, b, solve_fn):
        """
        Solve A x = b using a provided SciPy solve function.

        A_sparse : torch.sparse_coo_tensor (n x n)
        b        : torch.Tensor (n,)
        solve_fn : function(A_csr, b) -> numpy array
        """

        # --- Convert torch sparse COO -> SciPy CSR ---
        idx = A_sparse.indices().cpu().numpy()
        vals = A_sparse.values().detach().cpu().numpy()
        n = A_sparse.size(0)

        A_csr = sp.csr_matrix((vals, (idx[0], idx[1])), shape=(n, n))

        # --- Solve A x = b ---
        x_np = solve_fn(A_csr, b.detach().cpu().numpy())

        # Some solvers return (x, info)
        if not isinstance(x_np, np.ndarray):
            x_np = x_np[0]

        x = torch.tensor(x_np, dtype=b.dtype, device=b.device)

        # Save stuff for backward
        ctx.save_for_backward(x, b, A_sparse.values())
        ctx.A_csr = A_csr
        ctx.solve_fn = solve_fn
        ctx.idx = idx

        return x

    @staticmethod
    def backward(ctx, grad_x):
        """
        grad_x : dL/dx
        """
        #    Ensure device/dtype consistency and build sparse grad_A
        device = grad_x.device
        dtype_all = grad_x.dtype
        
        x, b, A_vals_torch = ctx.saved_tensors
        A = ctx.A_csr
        solve_fn = ctx.solve_fn
        idx = ctx.idx

        grad_x_np = grad_x.detach().cpu().numpy()

        # --- Solve adjoint system A^T v = grad_x ---
        v_np = solve_fn(A.T, grad_x_np)
        if not isinstance(v_np, np.ndarray):
            v_np = v_np[0]

        v = torch.tensor(v_np, dtype=dtype_all, device=device)

        # --- Gradient wrt b ---
        grad_b = v.clone()

        # --- Gradient wrt sparse values of A ---
        # For COO entry A[i,j], dL/dA_ij = - v[i] * x[j]
        x_np = x.detach().cpu().numpy()

        grad_vals_np = -(v_np[idx[0]] * x_np[idx[1]])
        grad_vals = torch.tensor(grad_vals_np, dtype=dtype_all, device=device)
        # Convert numpy indices to a tensor on the correct device
        idx_tensor = torch.tensor(idx, device=device, dtype=torch.long)

        # Ensure gradient values are on the correct device/dtype
        grad_vals = grad_vals.to(device=device, dtype=dtype_all)

        # Ensure shape is a tuple of ints (A is a SciPy CSR matrix)
        shape = tuple(int(s) for s in A.shape)

        # Recreate/coalesce sparse gradient on the proper device/dtype
        grad_A = torch.sparse_coo_tensor(idx_tensor, grad_vals, shape, device=device, dtype=dtype_all).coalesce()
        
        return grad_A, grad_b, None

