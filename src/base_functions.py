import numpy as np 
from .activations import ( relu, d_relu, dd_relu,
                          squared_relu, d_squared_relu, dd_squared_relu,
                          cubic_relu, d_cubic_relu, dd_cubic_relu )



#=====================================================================
# H5 Hermite Interpolation for ReLU and its derivatives
#=====================================================================
def relu_H5(z, x1=-2, xN=2):
    z = np.asarray(z, dtype=float)
    dx = xN - x1

    relu1   = relu(x1)
    reluN   = relu(xN)
    relu1d  = d_relu(x1)
    reluNd  = d_relu(xN)
    relu1dd = dd_relu(x1)
    reluNdd = dd_relu(xN)

    h1 = (reluN - relu1 - relu1d*dx - 0.5*relu1dd*dx**2) / dx**3
    h2 = (3*(relu1 - reluN) + 2*(relu1d + 0.5*reluNd)*dx + 0.5*relu1dd*dx**2) / dx**4
    h3 = (6*(reluN - relu1) - 3*(relu1d + reluNd)*dx + 0.5*(reluNdd - relu1dd)*dx**2) / dx**5

    dz = z - x1
    return (relu1
            + relu1d*dz
            + 0.5*relu1dd*dz**2
            + h1*dz**3
            + h2*dz**3*(z - xN)
            + h3*dz**3*(z - xN)**2)


def relu_H5_d(z, x1=-2, xN=2):
    z = np.asarray(z, dtype=float)
    dx = xN - x1

    relu1   = relu(x1)
    reluN   = relu(xN)
    relu1d  = d_relu(x1)
    reluNd  = d_relu(xN)
    relu1dd = dd_relu(x1)
    reluNdd = dd_relu(xN)

    h1 = (reluN - relu1 - relu1d*dx - 0.5*relu1dd*dx**2) / dx**3
    h2 = (3*(relu1 - reluN) + 2*(relu1d + 0.5*reluNd)*dx + 0.5*relu1dd*dx**2) / dx**4
    h3 = (6*(reluN - relu1) - 3*(relu1d + reluNd)*dx + 0.5*(reluNdd - relu1dd)*dx**2) / dx**5

    dz = z - x1
    w = z - xN

    return (
        relu1d
        + relu1dd*dz
        + 3.0*h1*dz**2
        + h2*(3.0*dz**2*w + dz**3)
        + h3*(3.0*dz**2*w**2 + 2.0*dz**3*w)
    )


def relu_H5_dd(z, x1=-2, xN=2):
    z = np.asarray(z, dtype=float)
    dx = xN - x1

    relu1   = relu(x1)
    reluN   = relu(xN)
    relu1d  = d_relu(x1)
    reluNd  = d_relu(xN)
    relu1dd = dd_relu(x1)
    reluNdd = dd_relu(xN)

    h1 = (reluN - relu1 - relu1d*dx - 0.5*relu1dd*dx**2) / dx**3
    h2 = (3*(relu1 - reluN) + 2*(relu1d + 0.5*reluNd)*dx + 0.5*relu1dd*dx**2) / dx**4
    h3 = (6*(reluN - relu1) - 3*(relu1d + reluNd)*dx + 0.5*(reluNdd - relu1dd)*dx**2) / dx**5

    dz = z - x1
    w = z - xN

    return (
        relu1dd
        + 6.0*h1*dz
        + h2*(6.0*dz*w + 6.0*dz**2)
        + h3*(6.0*dz*w**2 + 12.0*dz**2*w + 2.0*dz**3)
    )


#=====================================================================
# H5 Hermite Interpolation for Squared ReLU and its derivatives
#=====================================================================

def squared_relu_H5(z, x1=-2, xN=2):
    z = np.asarray(z, dtype=float)
    dx = xN - x1

    sq_relu1   = squared_relu(x1)
    sq_reluN   = squared_relu(xN)
    sq_relu1d  = d_squared_relu(x1)
    sq_reluNd  = d_squared_relu(xN)
    sq_relu1dd = dd_squared_relu(x1)
    sq_reluNdd = dd_squared_relu(xN)

    h1 = (sq_reluN - sq_relu1 - sq_relu1d*dx - 0.5*sq_relu1dd*dx**2) / dx**3
    h2 = (3*(sq_relu1 - sq_reluN) + 2*(sq_relu1d + 0.5*sq_reluNd)*dx + 0.5*sq_relu1dd*dx**2) / dx**4
    h3 = (6*(sq_reluN - sq_relu1) - 3*(sq_relu1d + sq_reluNd)*dx + 0.5*(sq_reluNdd - sq_relu1dd)*dx**2) / dx**5

    dz = z - x1
    return (sq_relu1
            + sq_relu1d*dz
            + 0.5*sq_relu1dd*dz**2
            + h1*dz**3
            + h2*dz**3*(z - xN)
            + h3*dz**3*(z - xN)**2)


def squared_relu_H5_d(z, x1=-2, xN=2):
    z = np.asarray(z, dtype=float)
    dx = xN - x1

    sq_relu1   = squared_relu(x1)
    sq_reluN   = squared_relu(xN)
    sq_relu1d  = d_squared_relu(x1)
    sq_reluNd  = d_squared_relu(xN)
    sq_relu1dd = dd_squared_relu(x1)
    sq_reluNdd = dd_squared_relu(xN)

    h1 = (sq_reluN - sq_relu1 - sq_relu1d*dx - 0.5*sq_relu1dd*dx**2) / dx**3
    h2 = (3*(sq_relu1 - sq_reluN) + 2*(sq_relu1d + 0.5*sq_reluNd)*dx + 0.5*sq_relu1dd*dx**2) / dx**4
    h3 = (6*(sq_reluN - sq_relu1) - 3*(sq_relu1d + sq_reluNd)*dx + 0.5*(sq_reluNdd - sq_relu1dd)*dx**2) / dx**5

    dz = z - x1
    w = z - xN

    return (
        sq_relu1d
        + sq_relu1dd*dz
        + 3.0*h1*dz**2
        + h2*(3.0*dz**2*w + dz**3)
        + h3*(3.0*dz**2*w**2 + 2.0*dz**3*w)
    )


def squared_relu_H5_dd(z, x1=-2, xN=2):
    z = np.asarray(z, dtype=float)
    dx = xN - x1

    sq_relu1   = squared_relu(x1)
    sq_reluN   = squared_relu(xN)
    sq_relu1d  = d_squared_relu(x1)
    sq_reluNd  = d_squared_relu(xN)
    sq_relu1dd = dd_squared_relu(x1)
    sq_reluNdd = dd_squared_relu(xN)

    h1 = (sq_reluN - sq_relu1 - sq_relu1d*dx - 0.5*sq_relu1dd*dx**2) / dx**3
    h2 = (3*(sq_relu1 - sq_reluN) + 2*(sq_relu1d + 0.5*sq_reluNd)*dx + 0.5*sq_relu1dd*dx**2) / dx**4
    h3 = (6*(sq_reluN - sq_relu1) - 3*(sq_relu1d + sq_reluNd)*dx + 0.5*(sq_reluNdd - sq_relu1dd)*dx**2) / dx**5

    dz = z - x1
    w = z - xN

    return (
        sq_relu1dd
        + 6.0*h1*dz
        + h2*(6.0*dz*w + 6.0*dz**2)
        + h3*(6.0*dz*w**2 + 12.0*dz**2*w + 2.0*dz**3)
    )


#=====================================================================
# H5 Hermite Interpolation for Cubic ReLU and its derivatives
#===================================================================== 

def cubic_relu_H5(z, x1=-2, xN=2):
    z = np.asarray(z, dtype=float)
    dx = xN - x1

    cu_relu1   = cubic_relu(x1)
    cu_reluN   = cubic_relu(xN)
    cu_relu1d  = d_cubic_relu(x1)
    cu_reluNd  = d_cubic_relu(xN)
    cu_relu1dd = dd_cubic_relu(x1)
    cu_reluNdd = dd_cubic_relu(xN)

    h1 = (cu_reluN - cu_relu1 - cu_relu1d*dx - 0.5*cu_relu1dd*dx**2) / dx**3
    h2 = (3*(cu_relu1 - cu_reluN) + 2*(cu_relu1d + 0.5*cu_reluNd)*dx + 0.5*cu_relu1dd*dx**2) / dx**4
    h3 = (6*(cu_reluN - cu_relu1) - 3*(cu_relu1d + cu_reluNd)*dx + 0.5*(cu_reluNdd - cu_relu1dd)*dx**2) / dx**5

    dz = z - x1
    return (cu_relu1
            + cu_relu1d*dz
            + 0.5*cu_relu1dd*dz**2
            + h1*dz**3
            + h2*dz**3*(z - xN)
            + h3*dz**3*(z - xN)**2)

def cubic_relu_H5_d(z, x1=-2, xN=2):
    z = np.asarray(z, dtype=float)
    dx = xN - x1

    cu_relu1   = cubic_relu(x1)
    cu_reluN   = cubic_relu(xN)
    cu_relu1d  = d_cubic_relu(x1)
    cu_reluNd  = d_cubic_relu(xN)
    cu_relu1dd = dd_cubic_relu(x1)
    cu_reluNdd = dd_cubic_relu(xN)

    h1 = (cu_reluN - cu_relu1 - cu_relu1d*dx - 0.5*cu_relu1dd*dx**2) / dx**3
    h2 = (3*(cu_relu1 - cu_reluN) + 2*(cu_relu1d + 0.5*cu_reluNd)*dx + 0.5*cu_relu1dd*dx**2) / dx**4
    h3 = (6*(cu_reluN - cu_relu1) - 3*(cu_relu1d + cu_reluNd)*dx + 0.5*(cu_reluNdd - cu_relu1dd)*dx**2) / dx**5

    dz = z - x1
    w = z - xN

    return (
        cu_relu1d
        + cu_relu1dd*dz
        + 3.0*h1*dz**2
        + h2*(3.0*dz**2*w + dz**3)
        + h3*(3.0*dz**2*w**2 + 2.0*dz**3*w)
    )

def cubic_relu_H5_dd(z, x1=-2, xN=2):
    z = np.asarray(z, dtype=float)
    dx = xN - x1

    cu_relu1   = cubic_relu(x1)
    cu_reluN   = cubic_relu(xN)
    cu_relu1d  = d_cubic_relu(x1)
    cu_reluNd  = d_cubic_relu(xN)
    cu_relu1dd = dd_cubic_relu(x1)
    cu_reluNdd = dd_cubic_relu(xN)

    h1 = (cu_reluN - cu_relu1 - cu_relu1d*dx - 0.5*cu_relu1dd*dx**2) / dx**3
    h2 = (3*(cu_relu1 - cu_reluN) + 2*(cu_relu1d + 0.5*cu_reluNd)*dx + 0.5*cu_relu1dd*dx**2) / dx**4
    h3 = (6*(cu_reluN - cu_relu1) - 3*(cu_relu1d + cu_reluNd)*dx + 0.5*(cu_reluNdd - cu_relu1dd)*dx**2) / dx**5

    dz = z - x1
    w = z - xN

    return (
        cu_relu1dd
        + 6.0*h1*dz
        + h2*(6.0*dz*w + 6.0*dz**2)
        + h3*(6.0*dz*w**2 + 12.0*dz**2*w + 2.0*dz**3)
    )
