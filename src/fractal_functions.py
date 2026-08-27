import numpy as np 
def pointwise_fractal(z, fractal_dd):
    p = fractal_dd["partition"]
    z_arr = np.asarray(z)

    if np.any(z_arr < p[0]) or np.any(z_arr > p[-1]):
        raise ValueError(
            f"Point {z} outside interpolation domain [{p[0]}, {p[-1]}]"
        )

    return np.interp(z, p, fractal_dd["values"])


def alpha_fractalize_first_derivative(f1, g1, a, b, n_sub, in_alpha, n_iter, dict=True):

    X = np.linspace(a, b, n_sub + 1)

    Y = f1(X)
    G = g1(X)

    N = len(X) - 1

    if np.isscalar(in_alpha):
        alpha = np.full(N, in_alpha, dtype=float)
    else:
        alpha = np.asarray(in_alpha, dtype=float)

    if len(alpha) != N:
        raise ValueError("alpha must have length len(X)-1.")

    x0 = X[0]
    xN = X[-1]

    # affine coefficients
    A = np.zeros(N)
    B = np.zeros(N)

    for i in range(N):
        A[i] = (X[i+1] - X[i])/(xN - x0)
        B[i] = (xN*X[i] - x0*X[i+1])/(xN - x0)

    def L_i(x, i):
        return A[i]*x + B[i]

    # Initial approximation
    current_points = np.column_stack((X, Y))

    # RB iterations
    for _ in range(n_iter):

        new_points = []

        for i in range(N):

            x_old = current_points[:,0]
            y_old = current_points[:,1]

            x_new = L_i(x_old, i)

            # The RB operator for the first derivative
            y_new = (
                f1(x_new)
                + (alpha[i]/A[i]) *
                  (y_old - g1(x_old))
            )

            new_points.extend(np.column_stack((x_new, y_new)))

        current_points = np.asarray(new_points)

    # sort
    order = np.argsort(current_points[:,0])
    current_points = current_points[order]

    # remove duplicates
    _, idx = np.unique(
        np.round(current_points[:,0],14),
        return_index=True
    )

    current_points = current_points[np.sort(idx)]
    
    if dict == True:
        return {
                "partition": current_points[:,0],
                "values": current_points[:,1],
            }
    else:
        return current_points[:,0], current_points[:,1]


    


def alpha_fractalize_second_derivative( f2, g2, a, b, n_sub, in_alpha, n_iter, dict= True):

    X = np.linspace(a, b, n_sub + 1)

    Y = f2(X)
    G = g2(X)

    N = len(X) - 1

    if np.isscalar(in_alpha):
        alpha = np.full(N, in_alpha, dtype=float)
    else:
        alpha = np.asarray(in_alpha, dtype=float)

    if len(alpha) != N:
        raise ValueError("alpha must have length len(X)-1.")

    x0 = X[0]
    xN = X[-1]

    # affine coefficients
    A = np.zeros(N)
    B = np.zeros(N)

    for i in range(N):
        A[i] = (X[i+1] - X[i])/(xN - x0)
        B[i] = (xN*X[i] - x0*X[i+1])/(xN - x0)

    def L_i(x, i):
        return A[i]*x + B[i]

    # Initial approximation
    current_points = np.column_stack((X, Y))

    # RB iterations
    for _ in range(n_iter):

        new_points = []

        for i in range(N):

            x_old = current_points[:,0]
            y_old = current_points[:,1]

            x_new = L_i(x_old, i)

            y_new = (
                f2(x_new)
                + (alpha[i]/A[i]**2) *
                  (y_old - g2(x_old))
            )

            new_points.extend(np.column_stack((x_new, y_new)))

        current_points = np.asarray(new_points)

    # sort
    order = np.argsort(current_points[:,0])
    current_points = current_points[order]

    # remove duplicates
    _, idx = np.unique(
        np.round(current_points[:,0],14),
        return_index=True
    )

    current_points = current_points[np.sort(idx)]
    if dict == True:
        return {
                "partition": current_points[:,0],
                "values": current_points[:,1],
            }
    else:
        return current_points[:,0], current_points[:,1]


#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------

def alpha_fractalize(f, g, a, b, n_sub, in_alpha, n_iter, dict=True):
    X = np.linspace(a, b, n_sub + 1)
    Y = f(X)
    G = g(X)

    if not np.isclose(G[0], Y[0], atol=1e-10) or not np.isclose(G[-1], Y[-1], atol=1e-10):
        print("Boundary condition mismatch!")
        print(f"|g({a}) - f({a})| = {np.abs(G[0] - Y[0]):.2e}")
        print(f"|g({b}) - f({b})| = {np.abs(G[-1] - Y[-1]):.2e}")
        raise ValueError("The boundary conditions of g must match those of f.")

    N = len(X) - 1  
    if np.isscalar(in_alpha):
        alpha = np.full(N, in_alpha, dtype=float)
    else:
        alpha = np.asarray(in_alpha, dtype=float)

    if len(alpha) != N:
        raise ValueError("alpha must have length len(X)-1.")

    # if type(in_alpha) is float:
    #     alpha = np.full(N, in_alpha)
    # else:
    #     alpha = in_alpha

    
    alpha = np.asarray(alpha, dtype=float)
    
    if len(alpha) != N:
        raise ValueError("alpha must have length len(X)-1.")

    x0, xN = X[0], X[-1]

    # Compute affine map coefficients
    a = np.zeros(N)
    b = np.zeros(N)

    for i in range(N):
        xn0 = X[i]
        xn1 = X[i + 1]

        a[i] = (xn1 - xn0) / (xN - x0)
        b[i] = (xN * xn0 - x0 * xn1) / (xN - x0)


    def L_i(x,i):
        return a[i] * x + b[i]


# =========================================================================
    # STAGE 0: Initial nodal points
    # Math: { (x_j, f(x_j)) : j = 0, 1, ..., N }
    # Total points at this stage = N + 1
    # =========================================================================
    current_points = []
    for j in range(len(X)):
        current_points.append((X[j], Y[j]))

    # =========================================================================
    # STAGE k: The Induction Loop
    # Math: Repeated application of the self-referential equation up to level K
    # =========================================================================
    for k in range(1, n_iter + 1):
        
        next_stage_points = []
        
        # 1. Loop over every subinterval i = 0, 1, ..., N-1
        #    (Corresponds to math index i = 1, 2, ..., N)
        for i in range(N):
            
            # 2. Loop over every point (x_old, y_old) from stage k-1
            #    At k=1, these are the N+1 base points (j = 0, 1, ..., N).
            #    At k=2, these are the points from L_m(x_j), and so on.
            for pt in current_points:
                x_old = pt[0]  # This is x_j (or L_sigma(x_j) at deeper levels)
                y_old = pt[1]  # This is f^alpha(x_old) from the previous stage
                
                # 3. Apply the spatial contraction: L_i(x)
                x_new = L_i(x_old, i)
                
                # 4. Apply the Read-Bajraktarevic operator:
                #    f(L_i(x)) + alpha_i * [ f^alpha(x) - g(x) ]
                y_new = f(x_new) + alpha[i] * (y_old - g(x_old))
                
                # 5. Add the newly generated point to our stage k collection
                next_stage_points.append((x_new, y_new))
        
        # The set of points for stage k is complete. 
        # Size is now exactly N * (size of previous stage), reaching N^k * (N + 1).
        current_points = next_stage_points

    # Convert the list of tuples back into a NumPy array for sorting
    current_points = np.array(current_points)

    # =========================================================================
    # POST-PROCESSING: Deduplication
    # Math: L_i(x_N) and L_{i+1}(x_0) map to the exact same boundary x-coordinate.
    # We remove these duplicate boundary evaluations to get the true graph.
    # =========================================================================
    
    # Sort points from left to right across the domain [a, b]
    order = np.argsort(current_points[:, 0])
    current_points = current_points[order]

    # Keep only unique x-coordinates (rounding to 14 decimals avoids float precision errors)
    _, idx = np.unique(np.round(current_points[:, 0], 14), return_index=True)
    current_points = current_points[np.sort(idx)]

    if dict == True:
        return {
                "partition": current_points[:,0],
                "values": current_points[:,1],
            }
    else:
        return current_points[:,0], current_points[:,1]




