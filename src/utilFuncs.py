import torch
import numpy as np
import random
from mmapy import mmasub
from scipy.optimize import minimize, Bounds, NonlinearConstraint # minimize is used for testing
from scipy.interpolate import BSpline
from types import SimpleNamespace
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# sys.path.append(os.path.realpath('./src/'))
#--------------------------#
def set_seed(manualSeed):
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.manual_seed(manualSeed)
  torch.cuda.manual_seed(manualSeed)
  torch.cuda.manual_seed_all(manualSeed)
  np.random.seed(manualSeed)
  random.seed(manualSeed)
#--------------------------#
def curve_setup(cp, t, curve='points'):
    """
    Calculates points on a Bézier or Spline curve of arbitrary degree.

    Args:
        cp: A tensor of control points (shape (n+1, d)), where n is the 
            degree of the curve and d is the dimension of the points 
            (e.g., d=1 for a 2D curve represented as (x,y) pairs, d=1 for your case).
        t: A tensor of parameter values (shape (m,)), where m is the number 
           of points to calculate on the curve.  t should be in the range [0, 1].

    Returns:
        A tensor of points on the Bézier curve (shape (m, d)).
    """
    if curve == 'points':
        curve_points = cp.clone().T
    elif curve == 'spline':
        p = 2 # degree of the curve
        n = cp.shape[1] # number of control points
        U = np.zeros(p+n+1)
        U[p:n+1] = np.linspace(0,1,n-p+1)
        U[-p::] = 1
        N = np.zeros((n,t.shape[0]))
        I =  np.eye(n)
        for i in range(n):
            N[i,:] = BSpline(U,I[i],p)(t.cpu().detach().numpy())
        N = torch.from_numpy(N).float().to(cp.device)
        curve_points = N.T @ cp.T
    elif curve == 'bezier':
        n = cp.shape[1] - 1  # Degree of the Bézier curve
        m = t.shape[0]  # Number of points to calculate
        if n == 4:
            b = torch.tensor([1,4,6,4,1.]).to(cp.device)
        elif n == 5:
            b = torch.tensor([1,5,10,10,5,1]).to(cp.device)

        # Calculate Bernstein basis polynomials
        bernstein_matrix = torch.zeros((m, n + 1)).to(cp.device)
        for i in range(n + 1):
            bernstein_matrix[:, i] =  b[i]* (t ** i) * ((1 - t) ** (n - i))
        
        # Calculate points on the Bézier curve
        curve_points = bernstein_matrix @ cp.T

    return curve_points.T
#--------------------------#
def sparse_submatrix(A, rows, cols):
    """
    Extract submatrix A[rows][:, cols] from a PyTorch COO sparse tensor.
    Keeps autograd enabled for A.values().
    """

    assert A.layout == torch.sparse_coo, "convert CSR to COO first"

    idx = A.indices()
    vals = A.values()

    # Mask entries that survive the row/column selection.
    row_mask = torch.isin(idx[0], rows)
    col_mask = torch.isin(idx[1], cols)
    mask = row_mask & col_mask        # (nnz,) boolean

    # Filter indices and values
    sub_idx = idx[:, mask]
    sub_vals = vals[mask]             # gradients preserved

    # Remap rows and cols to new numbering 0..len(rows)-1, 0..len(cols)-1
    # Use torch.searchsorted (GPU-safe)
    rows_sorted, row_perm = torch.sort(rows)
    cols_sorted, col_perm = torch.sort(cols)

    new_rows = torch.searchsorted(rows_sorted, sub_idx[0])
    new_cols = torch.searchsorted(cols_sorted, sub_idx[1])

    new_idx = torch.stack([new_rows, new_cols])

    return torch.sparse_coo_tensor(
        new_idx, sub_vals,
        size=(len(rows), len(cols)),
        device=A.device,
        dtype=A.dtype
    ).coalesce()

def custom_minimize(objCall, x0, bounds,method='GCMMA',constraintCall=None, options=None,callback=None): # type: ignore
    """
    Optimize using the MMA or GCMMA (default) method.
    
    Parameters:
    - objCall: Callable for the objective function. Should return:
        - Objective value: shape (1,)
        - Gradient: shape (n,)
    - x0: Initial guess for the optimization (shape (n,))
    - bounds: scipy.optimize.Bounds object for variable bounds.
    - constraintCall: Callable for the constraint function. Should return:
        - Constraint values: shape (m,1)
        - Gradient: shape (m, n)
    - options: Dictionary for optimization options (e.g., maxiter, kkttol, etc.).
    - callback: Callable function which takes x as input every iter, can be used for plot and early stop.
    
    Returns:
    - xval: Optimized variable values (shape (n,))
    - f0val: Final objective value (shape (1,))
    - func_evals: Number of function evaluations.
    """
    n = len(x0)
    if constraintCall== None: # size of 1 constraint
        def constraintCall(x):
            c = 1e-6*np.ones((1,1))
            dc = 1e-12*(x[np.newaxis]*0+1.)
            return c,dc
            
    if bounds == None:
        bounds = Bounds([0.]*n,[1.]*n) # scipy bounds # type: ignore
            
    # Handle options
    default_options = {
    'maxiter': 200,
    'miniter':10,
    'kkttol': 1e-5,
    'disp': False,
    'maxfun': 1000,
    'move_limit':0.1
    }
    
    # Use defaults for missing options
    if options is None:
        options = {}  # Initialize empty if none provided
    
    # Check for each key in the options; if missing, use default value
    options['maxiter'] = options['maxiter'] if 'maxiter' in options else default_options['maxiter']
    options['miniter'] = options['miniter'] if 'miniter' in options else default_options['miniter']
    options['kkttol'] = options['kkttol'] if 'kkttol' in options else default_options['kkttol']
    options['disp'] = options['disp'] if 'disp' in options else default_options['disp']
    options['maxfun'] = options['maxfun'] if 'maxfun' in options else default_options['maxfun']
    options['move_limit'] = options['move_limit'] if 'move_limit' in options else default_options['move_limit']

    if method == 'MMA':
        print('Running '+method+' optimizer from mmapy')
        x,fun,fun_eval = optimizeMMA(objCall, x0, bounds, constraintCall, options=options,callback=callback)
    elif method == 'Adam':
        print('Running '+method+' optimizer from torch.optim')
        x,fun,fun_eval = optimizeAdam(objCall, x0, bounds, constraintCall, options=options,callback=callback)
    else:
        print('Running '+method+' optimizer from scipy.optimize')
        options['method'] = method
        x,fun,fun_eval = optimizeScipy(objCall, x0, bounds, constraintCall, options=options, callback=callback)
        
    result = SimpleNamespace()                
    result.x = x
    result.fun = fun
    result.fun_eval = fun_eval
    
    return result
    
def compute_move(outer_iter, max_iter, move_min=1e-3, move_max=1.0, decay_rate=5.0, wiggle=0.5):
    """
    Returns a damped oscillatory move size that:
    - Starts at move_max
    - Decays toward move_min
    - Oscillates like a damped spring

    Parameters:
    - outer_iter: current iteration
    - max_iter: maximum number of iterations
    - move_min: lower limit of move (e.g., 1e-3)
    - move_max: upper limit of move (e.g., 1.0)
    - decay_rate: exponential decay rate (higher = faster damping)
    - wiggle: amplitude of oscillation (0 = no oscillation)

    Returns:
    - move: float in [move_min, move_max]
    """
    envelope = np.exp(-decay_rate * outer_iter / max_iter)  # damping from 1 to ~0
    oscillation = (1 + wiggle * np.cos(2 * np.pi * outer_iter / 10))  # between [1 - wiggle, 1 + wiggle]
    scaled = envelope * oscillation
    move = move_min + (move_max - move_min) * scaled
    return move
    
def optimizeMMA(objCall, x0, bounds, constraintCall, options,callback):
        """
        Optimize using the MMA method.
        
        Parameters:
        - objCall: Callable for the objective function. Should return:
            - Objective value: shape (1,)
            - Gradient: shape (n,)
        - x0: Initial guess for the optimization (shape (n,))
        - bounds: scipy.optimize.Bounds object for variable bounds.
        - constraintCall: Callable for the constraint function. Should return:
            - Constraint values: shape (m,1)
            - Gradient: shape (m, n)
        - options: Dictionary for optimization options (e.g., maxiter, kkttol, etc.).
        
        Returns:
        - xval: Optimized variable values (shape (n,))
        - f0val: Final objective value (shape (1,))
        - func_evals: Number of function evaluations.
        """
        
        # unpack options
        max_iter = options['maxiter'] 
        min_iter = options['miniter']
        kkttol = options['kkttol'] 
        move_limit = options['move_limit'] 
        disp = options['disp']
        max_fun = options['maxfun']
        func_evals = 0

        # Initial values
        f0val_init, df0dx_init = objCall(x0)
        fval, dfdx = constraintCall(x0)
        func_evals += 1

        n = len(x0)
        m = np.shape(fval)[0]
    
        xmin = bounds.lb.reshape((-1, 1))
        xmax = bounds.ub.reshape((-1, 1))
        
        xval = x0.copy()#[np.newaxis].T
        xold1 = xval.copy()
        xold2 = xval.copy()
        
        c = 1000 * np.ones((m, 1))
        d = np.ones((m, 1))
        a0 = 1
        a = np.zeros((m, 1))
        move = move_limit
        scale_grad = 100.0
        
        outer_iter = 0
        kktnorm = np.array([kkttol + 10])
        change = kktnorm.copy()
        change4 = kktnorm.copy()
        
    
        f0valnew, df0dxnew = f0val_init.copy(), df0dx_init.copy()
        fvalnew, dfdxnew = fval.copy(), dfdx.copy()
        
        F_best = np.zeros(max_iter)
        F_best[0] = f0valnew 
        xbest = xval.copy()
        
        
        updateConTol = -1e-5
        # The main iteration loop
        for outer_iter in range(1,max_iter):
            
            move = compute_move(outer_iter, max_iter*0.5, move_max=0.5, decay_rate=min_iter*1.0)
            
            # if outer_iter < 6:
            #     updateConTol = np.max((fvalnew)) * 0.25
            # else:
            # updateConTol = -1e-5
                       
            for _ in range(2):
                # Solve the MMA subproblem
                xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub( # type: ignore
                    m, n, outer_iter, xval, xmin, xmax, xold1, xold2, f0valnew[np.newaxis], df0dxnew, fvalnew, dfdxnew, xmin, xmax, a0, a, c, d, move)

                # Update the variables for the next iteration
                xold2 = xold1.copy()
                xold1 = xval.copy()
                xval = xmma.copy()
        
            # Compute the objective and constraint values at the new point
            fvalnew,dfdxnew = constraintCall(xval.reshape(-1))
            f0valnew,df0dxnew = objCall(xval.reshape(-1))
            func_evals += 1  # Increment the function evaluation count

            f0val, df0dx = f0valnew.copy(), df0dxnew.copy()
            fval, dfdx = fvalnew.copy(), dfdxnew.copy()
            
            xbest = xval.copy()
            F_best[outer_iter] = f0val.copy()
            fval_best = fval.copy()
     
            change = abs(F_best[outer_iter] - F_best[outer_iter - 1])
            
            # change = max(fval)
            
            if disp:
                print(f"Iter {outer_iter}: f0 = {f0val.item():.4g}, max_con: {max(fval).item():.4g} Function evals = {func_evals}, n = {len(xbest)}")
            if callback != None:
                callback(xbest.reshape(-1))
                
            # Check for termination condition
            if ((change <= kkttol or func_evals >= max_fun) and (outer_iter >= min_iter) and (fval<kkttol).all()):
                print("Local minima found")
                break    
                
        return xbest.reshape(-1), f0val, func_evals
        
def optimizeAdam(objCall, x0, bounds, constraintCall, options, callback=None):
    """
    Optimize using Adam from PyTorch, with bounds and constraint penalties.

    Parameters:
    - objCall: Callable returning (objective, gradient), both as NumPy arrays.
    - x0: Initial guess (NumPy array).
    - bounds: scipy.optimize.Bounds object.
    - constraintCall: Callable returning (constraints, gradients) in NumPy.
    - options: Dictionary with keys:
        - 'maxiter': max iterations
        - 'lr': learning rate
        - 'kkttol': tolerance
        - 'penalty': penalty weight for constraint violation
        - 'disp': bool for display
    - callback: Callable with signature callback(x) called after each iteration.

    Returns:
    - xval: Final optimized variables
    - f0val: Final objective value
    - func_evals: Number of evaluations
    """

    max_iter = options.get('maxiter', 500)
    lr = options.get('lr', 1e-1)
    kkttol = options.get('kkttol', 1e-4)
    penalty_max = options.get('penalty', 100.0)
    disp = options.get('disp', True)
    max_fun = options.get('maxfun', 10000)

    func_evals = 0

    # Bounds
    xmin = bounds.lb
    xmax = bounds.ub

    # Make x a torch parameter with gradients
    x = torch.tensor(x0, requires_grad=True).double()
    optimizer = torch.optim.Adam([x], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(max_iter), eta_min=1e-8)

    best_x = x0.copy()
    best_f = np.inf

    mu = -1e-4
    Lambda = 1.0
    for it in range(max_iter):
        optimizer.zero_grad()
        # penalty = np.abs(1-np.sin(np.pi/300 + (it+3)*np.pi/10))

        # Get obj value and grad (from numpy)
        f0val_np, df0dx_np = objCall(x.detach().numpy())
        fval_np, dfdx_np = constraintCall(x.detach().numpy())
        func_evals += 1

        # Convert gradients to torch
        df0dx = torch.tensor(df0dx_np).double()
        dfdx = torch.tensor(dfdx_np).double()
        fval = torch.tensor(fval_np).double().flatten()

        # # Final loss with penalty
        # loss = torch.tensor(f0val_np).double() +  mu * torch.log(-fval) #torch.pow(fval,2)

        # # Backward using manually set gradients
        # x.grad = df0dx + mu*dfdx.reshape(-1) * (1.0/fval) # -1/fval * -dfdx
        
        # Lambda = 10*it/max_iter
        loss = torch.tensor(f0val_np).double() +  Lambda * torch.maximum(fval*0,fval) #torch.pow(fval,2)

        if fval > 0:
            x.grad = df0dx + Lambda * dfdx.reshape(-1)
        else:
            x.grad = df0dx
        
        optimizer.step()
        scheduler.step()

        # Project onto bounds
        with torch.no_grad():
            x[:] = torch.clamp(x, min= torch.from_numpy(xmin), max= torch.from_numpy(xmax))

        # Save best feasible solution
        if np.all(fval_np <= 0) and f0val_np < best_f:
            best_x = x.detach().numpy().copy()
            best_f = f0val_np

        # Verbose output
        if disp:
            print(f"Iter {it+1}: f0 = {f0val_np.item():.4g}, con = {fval.item():.4g}, grad norm = {x.grad.norm().item():.4g}")

        # Callback
        if it % 10 == 0:
            if callback is not None:
                callback(x.detach().numpy())

        # Termination
        if x.grad.norm().item() <= kkttol or func_evals >= max_fun:
            break

    return best_x, best_f, func_evals


def optimizeScipy(objCall, x0, bounds, constraintCall, options, callback=None):
    """
    Optimize using SciPy's gradient-based constrained optimizers: SLSQP or trust-constr.

    Parameters:
    - objCall: Callable -> Returns (objective_value, gradient)
    - x0: Initial guess
    - bounds: scipy.optimize.Bounds
    - constraintCall: Callable -> Returns (constraint_values, constraint_jacobian)
    - options: Dict with optimization settings, e.g., {'method': 'SLSQP', 'maxiter': 100}
    - callback: Optional user-defined callback

    Returns:
    - xval: Optimized variable values
    - fval: Final objective value
    - func_evals: Number of function evaluations
    """
    
    method = options.get('method', 'SLSQP')
    disp = options.get('disp', True)
    max_iter = options.get('maxiter', 100)
    tol = options.get('tol', 1e-6)

    # Objective function wrapper
    def fun(x):
        fval, grad = objCall(x)
        return fval, grad

    # Constraint formatting
    con_vals, con_jac = constraintCall(x0)
    m = con_vals.shape[0] if len(con_vals.shape) > 0 else 1

    if method == 'trust-constr':
        # For trust-constr: use NonlinearConstraint
        def constraint_fun(x): # type: ignore
            cval, _ = constraintCall(x)
            return cval.flatten()

        def constraint_jac(x):
            _, cjac = constraintCall(x)
            return cjac  # Shape (m, n)

        constraints = NonlinearConstraint(
            fun=constraint_fun,
            jac=constraint_jac, # type: ignore
            hess='2-point',  # Could be 'cs' if complex-step is possible
            lb=-np.inf * np.ones(m),
            ub=np.zeros(m),  # inequality constraints: c(x) <= 0
        )
    else:
        # For SLSQP: dictionary-based inequality constraints
        def constraint_fun(x):
            cval, _ = constraintCall(x)
            return -cval.flatten()  # SLSQP requires c(x) >= 0

        def constraint_jac(x):
            _, cjac = constraintCall(x)
            return -cjac

        constraints = {
            'type': 'ineq',
            'fun': constraint_fun,
            'jac': constraint_jac
        }

    result = minimize(
        fun=fun,
        x0=x0,
        jac=True,
        bounds=bounds,
        constraints=constraints,
        method=method,
        callback=callback,
        options={
            'disp': disp,
            'maxiter': max_iter,
            'gtol': tol if method == 'trust-constr' else None,
            'ftol': tol if method == 'SLSQP' else None,
        }
    )

    final_fval = objCall(result.x)[0]
    return result.x, final_fval, result.nfev
    

def test1():
    def obj_rosenbrock(x):
        """
        Rosenbrock objective function: f(x) = sum(100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2)
        Gradient: computed analytically for Rosenbrock
        """
        f0 = np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        # Gradient calculation
        grad = np.zeros_like(x)
        grad[1:-1] = 200 * (x[1:-1] - x[:-2]**2) - 400 * (x[2:] - x[1:-1]**2) * x[1:-1] + 2 * (x[1:-1] - 1)
        grad[0] = -400 * (x[1] - x[0]**2) * x[0] + 2 * (x[0] - 1)
        grad[-1] = 200 * (x[-1] - x[-2]**2)
        
        return np.array([f0]), grad  # (1,) and (n,) shapes
    
    default_options = {
            'maxiter': 100,
            'kkttol': 1e-5,
            'move_limit': 0.1, # ignored by gcmma
            'disp': True,
            'maxfun': 1000
            }
    n=2
    xval=np.zeros(n)
    x,fun,l = optimizeGCMMA(obj_rosenbrock,xval,bounds=Bounds([0.]*n,[10.]*n),constraintCall=None,options=default_options) # type: ignore
    # Output the results
    print("GCMMA Result")
    print("Optimal value:", fun)
    print("Optimal solution:", x)    
    x,fun,l = optimizeMMA(obj_rosenbrock,xval,bounds=Bounds([0.]*n,[10.]*n),constraintCall=None,options=default_options) # type: ignore
    # Output the results
    print("MMA Result")
    print("Optimal value:", fun)
    print("Optimal solution:", x)
    # print("Success:", result.success)
    # print("Message:", result.message)
    
    # Perform optimization using L-BFGS-B
    result = minimize(
        fun=obj_rosenbrock,  # Objective function
        x0=xval,  # Initial guess
        jac=True,  # Gradient of the objective
        bounds=Bounds([0.]*n,[10.]*n),  # Variable bounds # type: ignore
        method='TNC'  # Optimization method
    )
    # Output the results
    print("Scipy Result")
    print("Optimal value:", result.fun)
    print("Optimal solution:", result.x)
    print("Success:", result.success)
    print("Message:", result.message)
    
def test2():
    # Define the quadratic objective function: f(x) = 1/2 * x^T Q x + c^T x
    def objectiveQuad(x):
        # Define the quadratic objective components Q and c
        Q = np.array([[2, 0], [0, 2]])  # Positive definite matrix
        c = np.array([-2, -5])  # Linear coefficients in the objective
        obj_val = 0.5 * np.dot(x.T, np.dot(Q, x)) + np.dot(c.T, x)
        grad = np.dot(Q, x) + c
        return obj_val.reshape(-1), grad.reshape(-1)
    
    # Define a quadratic constraint: x^T A x - b <= 0 and
    # Define a linear constraint: G x - h <= 0
    def quadratic_constraint(x):
      # Define the quadratic constraint components A and b
        A = np.array([[1, 0], [0, 1]])  # Identity matrix for constraint
        b = 1.  # The quadratic constraint threshold
    
        # Define the linear constraint components G and h
        G = np.array([[1, 1]])  # Coefficients of the linear constraint
        h = 1.  # Right-hand side of the linear inequality
    
        con_val = np.array([np.dot(x.T, np.dot(A, x)) - b]) # Shape (1,)
        con_grad = (2 * np.dot(A, x))[np.newaxis]  # Shape (1, n)
        
        con_val2 = np.dot(G, x) - h # Shape (1,)
        con_val = np.concatenate((con_val[np.newaxis],con_val2[np.newaxis])) # Shape (m, 1)
        con_grad = np.concatenate((con_grad,G),axis=0) # Shape (m, n)
        return con_val, con_grad
    
    default_options = {
            'maxiter': 100,
            'kkttol': 1e-5,
            'move_limit': 0.1, # ignored by gcmma
            'disp': True,
            'maxfun': 1000
            }
            
    n=2
    xval=np.zeros(n)
    x,fun,l = optimizeGCMMA(objectiveQuad,xval,bounds=Bounds([0.]*n,[10.]*n),constraintCall=quadratic_constraint,options=default_options) # type: ignore
    # Output the results
    print("GCMMA Result")
    print("Optimal value:", fun)
    print("Optimal solution:", x)       
    x,fun,l = optimizeMMA(objectiveQuad,xval,bounds=Bounds([0.]*n,[10.]*n),constraintCall=quadratic_constraint,options=default_options) # type: ignore
    # Output the results
    print("MMA Result")
    print("Optimal value:", fun)
    print("Optimal solution:", x)       
    
    # Wrapper for constraints for scipy
    def constraint_func(x):
        con_val, _ = quadratic_constraint(x)
        return con_val  # Return the constraint values
    
    # Gradient of the constraints for scipy
    def constraint_grad(x):
        _, con_grad = quadratic_constraint(x)
        return con_grad.flatten()  
      
    # Define bounds
    bounds = [(0, 10), (0, 10)]  # Bounds for each variable
    
    # Define constraints in the format required by scipy
    constraints = {
        'type': 'ineq',  # Inequality constraint
        'fun': constraint_func,  # Constraint function
        'jac': constraint_grad  # Gradient of the constraints
    }
    
    # Perform optimization using L-BFGS-B
    result = minimize(
        fun=objectiveQuad,  # Objective function
        x0=xval,  # Initial guess
        jac=True,  # Gradient of the objective
        bounds=bounds,  # Variable bounds
        constraints=constraints,  # Constraints
        method='TNC'  # Optimization method
    )
    
    # Output the results
    print("Scipy Result")
    print("Optimal value:", result.fun)
    print("Optimal solution:", result.x)
    print("Success:", result.success)
    print("Message:", result.message)


# test1()
# test2()