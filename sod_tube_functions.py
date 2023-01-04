"""
Created on Wed Mar 23 14:15:59 2022

@author: Sasha Bakker
"""
import numpy as np
import matplotlib.pyplot as plt

def construct_F(U, gamma):
    """
    Parameters
    ----------
    U : 2D array
        Component U of the vector form for nonlinear hyperbolic
        conservation laws.
    gamma : float
        Ratio of specific heats

    Returns
    -------
    F : 2D array
        Component F(U) of the vector form for nonlinear hyperbolic
        conservation laws.
    """
    
    rho = U[0, :]
    m = U[1, :]
    e = U[2, :]
    p = compute_pressure(gamma, rho, m, e)
    
    f0 = m
    f1 = m**2 / rho + p
    f2 = (m / rho) * (e + p)
    F = np.array([f0, f1, f2])
    
    return F
    

def construct_system(p0, r0, u0, gamma):
    """
    Parameters
    ----------
    p0 : 1D array
        Initial pressures p
    r0 : 1D array
        Initial densities ρ
    u0 : 1D array
        Initial velocities u

    Returns
    -------
    U : 2D array
        Initial component U of the vector form for nonlinear hyperbolic
        conservation laws.
    F : 2D array
        Component F(U) of the vector form for nonlinear hyperbolic
        conservation laws.
    """
    
    # Construct U
    rho = r0
    m = rho * u0
    eps = p0 / ((gamma - 1) * rho)
    e = rho * eps + 0.5 * rho * u0**2
    U = np.array([rho, m, e])

    # Construct F(U)
    f0 = m
    f1 = m**2 / rho + p0
    f2 = (m / rho) * (e + p0)
    F = np.array([f0, f1, f2])
    
    return U, F


def compute_pressure(gamma, rho, m, e):
    """
    Parameters
    ----------
    gamma : TYPE
        DESCRIPTION.
    rho : 1D array
        Densities ρ
    m : 1D array
        Momenta m
    e : 1D array
        Energies per unit volume e

    Returns
    -------
    p : 1D array
        Pressures p
    """
    
    # Velocities
    u = m / rho

    # Pressure
    p = (e - 0.5 * rho * u**2) * (gamma - 1)
    
    return p


def myplot(rho, u, p, x_i, x_f, xs, t, gamma, method_name="", savemyplot=False, n=0):
    """
    Parameters
    ----------
    rho : 1D array
        Densities ρ
    u : 1D array
        Velocities u
    p : 1D array
        Pressures p
    x_i : float
        Initial x-coordinate
    x_f : float
        Final x-coordinate
    xs : 1D array
        All x-coordinates
    method_name : string
        Name of the computational method
    savemyplot : boolean
        T/F to save the plot or not
    n : int
        iteration number 

    Returns
    -------
    None.
    """
    
    fig, axes = plt.subplots(nrows=4, ncols=1, dpi=200, figsize=(5,6))
    fig.tight_layout()
    the_time = format(t, '.4f')
    
    # Plot the density
    plt.xlim(x_i, x_f)
    plt.subplot(4, 1, 1)
    plt.title(method_name + ": " + r"$t$" + " = " + the_time + " s")
    plt.plot(xs, rho, 'g-')
    plt.ylabel(r'$\rho$',fontsize=16)
    plt.tick_params(axis='x',bottom=False,labelbottom=False)
    plt.ylim(0, 1.5)
    plt.xlim(x_i, x_f)
    plt.grid(True)
    
    # Plot the velocity
    plt.subplot(4, 1, 2)
    plt.plot(xs, u, 'r-')
    plt.ylabel(r'$u$',fontsize=16)
    plt.tick_params(axis='x',bottom=False,labelbottom=False)
    plt.ylim(0, 1.5)
    plt.xlim(x_i, x_f)
    plt.grid(True)
    
    # Plot the pressure
    plt.subplot(4, 1, 3)
    plt.plot(xs, p, 'b-')
    plt.ylabel(r'$p$',fontsize=16)
    plt.tick_params(axis='x',bottom=False,labelbottom=False)
    plt.ylim(0, 1.5)
    plt.xlim(x_i, x_f)
    plt.grid(True)
    
    # Plot the internal energy per unit mass
    plt.subplot(4, 1, 4)
    plt.plot(xs, p/((gamma - 1) * rho), 'k-')
    plt.ylabel(r'$\epsilon$',fontsize=16)
    plt.grid(True)
    plt.xlabel(r'$x$',fontsize=16)
    plt.xlim(x_i, x_f)
    plt.ylim(1, 3)
    plt.subplots_adjust(left=0.2)
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(top=0.95)
    
    if savemyplot == True:
        
        figname = str(method_name) + " " + str(n-1).zfill(5) + ".png"
        plt.savefig(figname)
        plt.close()
    else:
        plt.show()
    
def godunov_solution(F, U, dx, dt, U0, F0, gamma, U_new):
    """
    Parameters
    ----------
    F : 2D array
        Component F(U) of the vector form for nonlinear hyperbolic
        conservation laws.
    U : 2D array
        Component U of the vector form for nonlinear hyperbolic
        conservation laws.
    dx : float
        Spatial step.
    dt : float
        Time step.
    U0 : 2D array
        Initial state of component U.
    F0 : 2D array
        Initial state of component F(U).
    gamma : float
        Ratio of specific heats
    U_new : 2D array
        Array to store the updated state of component U.

    Returns
    -------
    U_new : 2D array
        The updated state of component U.
    """
    
    # I. Predictor step
    # =================
    F_pos = np.roll(F, -1)
    F_neg = np.roll(F, 1)
    U_pos = np.roll(U, -1)
    U_neg = np.roll(U, 1)
    
    U_half_pos = 0.5 * (U_pos + U) - (dt / (dx)) * (F_pos - F)
    U_half_pos[:,0] = U0[:,0] ; U_half_pos[:,-1] = U0[:,-1] # Dirichlet bounds
    
    U_half_neg = 0.5 * (U + U_neg) - (dt / (dx)) * (F - F_neg)
    U_half_neg[:,0] = U0[:,0] ; U_half_neg[:,-1] = U0[:,-1] # Dirichlet bounds
    
    
    # II. Corrector step
    # ==================
    F_half_pos = construct_F(U_half_pos, gamma)
    F_half_neg = construct_F(U_half_neg, gamma)

    U_new = U - (dt/dx) * (F_half_pos - F_half_neg)
    U_new[:,0] = U0[:,0] ; U_new[:,-1] = U0[:,-1]   # Dirichlet bounds
    
    return U_new


def add_artificial_viscosity(U_new, U0, viscosity, dt, dx):
    """
    Parameters
    ----------
    U_new : 2D array
        The updated state of component U.
    U0 : 2D array
        Initial state of component U.
    viscosity : float
        Constant number for artificial viscosity.
    dt : float
        Time step.
    dx : float
        Spatial step.

    Returns
    -------
    U_update : 2D array
        Updated state of component U with addition of artificial viscosity.
    """
    
    diff1 = np.roll(U_new, -1) - U_new
    diff12 = np.abs(diff1) * diff1
    diff2 = diff12 - np.roll(diff12, 1)
    
    U_update = U_new + viscosity *(dt / dx) * diff2
    
    U_update[:,0] = U0[:,0] ; U_update[:,-1] = U0[:,-1]   # Dirichlet bounds
    
    return U_update


def hyman_predictor_corrector_solution(U, U_new, U0, gamma, dt, dx, nx, delta = 0.8):
    """
    Parameters
    ----------
    U : 2D array
        Component U of the vector form for nonlinear hyperbolic
        conservation laws.
    U_new : 2D array
        Array to store the updated state of component U.
    U0 : 2D array
        Initial state of component U.
    gamma : float
        Ratio of specific heats
    dt : float
        Time step.
    dx : float
        Spatial step.
    nx : int
        Number of elements in mesh.
    delta : float
        Number that affects the stability of the scheme

    Returns
    -------
    U_new : 2D array
        The updated state of component U.
    """
    
    r = U[0, :]                             # Densisites ρ
    m = U[1, :]                             # Momenta 
    e = U[2, :]                             # Energy per unit volume
    p = compute_pressure(gamma, r, m, e)    # Pressures p
    
    # I. Predictor step
    # =================
    c = np.sqrt(gamma * p / r) # Local speed of sound where p = pressure, r = density
    
    phi_half_pos = (1/(4*dx)) * (np.roll(U+c, -1) + U+c) * (np.roll(U, -1) - U)
    Betas = (1/3)*(np.roll(U+c,-1) > (U+c+dx/3)) + (1.0)*(np.roll(U+c,-1) < (U+c+dx/3))
    phi_half_pos *= Betas
    phi_half_neg = (1/(4*dx)) * (U+c + np.roll(U+c, 1)) * (U - np.roll(U, 1))
    

    F = construct_F(U, gamma)
    DFin = (1/(12*dx)) * (-np.roll(F, -2) + 8*np.roll(F, -1) -8*np.roll(F, 1) + np.roll(F, 2))
    Pin = DFin - delta * (phi_half_pos - phi_half_neg)
    U_half = U - dt * Pin
    U_half[:,0] = U0[:,0] ; U_half[:,-1] = U0[:,-1]   # Dirichlet bounds
    U_half[:,1] = U0[:,1] ; U_half[:,-2] = U0[:,-2]   # Dirichlet bounds
    
    # II. Corrector step
    # ==================
    F_half = construct_F(U_half, gamma)
    DFin_half = (1/(12*dx)) * (-np.roll(F_half, -2) + 8*np.roll(F_half, -1) -8*np.roll(F_half, 1) + np.roll(F_half, 2))
    U_new = U - (dt/2) * (DFin_half + Pin)
    U_new[:,0] = U0[:,0] ; U_new[:,-1] = U0[:,-1]   # Dirichlet bounds
    U_new[:,1] = U0[:,1] ; U_new[:,-2] = U0[:,-2]   # Dirichlet bounds
    
    return U_new
    
    
def lax_wendroff_solution(F, U, dx, dt, U0, F0, gamma, U_new):
    """
    Parameters
    ----------
    F : 2D array
        Component F(U) of the vector form for nonlinear hyperbolic
        conservation laws.
    U : 2D array
        Component U of the vector form for nonlinear hyperbolic
        conservation laws.
    dx : float
        Spatial step.
    dt : float
        Time step.
    U0 : 2D array
        Initial state of component U.
    F0 : 2D array
        Initial state of component F(U).
    gamma : float
        Ratio of specific heats
    U_new : 2D array
        Array to store the updated state of component U.

    Returns
    -------
    U_new : 2D array
        The updated state of component U.
    """
    
    # I. Predictor step
    # =================
    F_pos = np.roll(F, -1)
    F_neg = np.roll(F, 1)
    U_pos = np.roll(U, -1)
    U_neg = np.roll(U, 1)
    
    U_half_pos = 0.5 * (U_pos + U) - (dt / (2*dx)) * (F_pos - F)
    U_half_pos[:,0] = U0[:,0] ; U_half_pos[:,-1] = U0[:,-1] # Dirichlet bounds
    
    U_half_neg = 0.5 * (U + U_neg) - (dt / (2*dx)) * (F - F_neg)
    U_half_neg[:,0] = U0[:,0] ; U_half_neg[:,-1] = U0[:,-1] # Dirichlet bounds
    
    
    # II. Corrector step
    # ==================
    F_half_pos = construct_F(U_half_pos, gamma)
    F_half_neg = construct_F(U_half_neg, gamma)

    U_new = U - (dt/dx) * (F_half_pos - F_half_neg)
    U_new[:,0] = U0[:,0] ; U_new[:,-1] = U0[:,-1]   # Dirichlet bounds
    
    return U_new



def roe_solution(F, U, dx, dt, U0, gamma, U_new):
    """
    Parameters
    ----------
    F : 2D array
        Component F(U) of the vector form for nonlinear hyperbolic
        conservation laws.
    U : 2D array
        Component U of the vector form for nonlinear hyperbolic
        conservation laws.
    dx : float
        Spatial step.
    dt : float
        Time step.
    U0 : 2D array
        Initial state of component U.
    gamma : float
        Ratio of specific heats
    U_new : 2D array
        Array to store the updated state of component U.

    Returns
    -------
    U_new : 2D array
        The updated state of component U.
    """

    # Compute basic variables
    rho = U[0, :]                           # Densisites ρ
    m = U[1, :]                             # Momenta 
    e = U[2, :]                             # Energy per unit volume
    p = compute_pressure(gamma, rho, m, e)   # Pressures p
    u = m / rho                             # velocities
    
    # Compute H (eq12a)
    H = (gamma * p)/((gamma-1) * rho) + 0.5 * u**2
    
    # Take averages (we will need to fix U endpoints at very end of computation)
    r1 = np.sqrt(rho)
    r2 = np.sqrt(np.roll(rho, -1))
    u_half = (r1 * u + r2 * np.roll(u, -1)) / (r1 + r2)
    H_half = (r1 * H + r2 * np.roll(H, -1)) / (r1 + r2)
    a_half = np.sqrt( (gamma - 1) * (H_half - 0.5 * u_half**2) )

    # Compute auxillary variables
    alpha1 = (gamma - 1) * u_half**2 / (2 * a_half**2)
    alpha2 = (gamma - 1) / a_half**2

    Au_vecs = np.zeros(np.shape(U))
    
    for j in range(len(u_half)):
        
        """
        Lambda Matrix in A = SΛS–1
        """
        Lam = np.zeros((3,3))
        Lam[0,0] = np.abs(u_half[j] - a_half[j])
        Lam[1,1] = np.abs(u_half[j])
        Lam[2,2] = np.abs(u_half[j] + a_half[j])
        
        """
        S Matrix in A = SΛS–1
        """
        S = np.array([[1, 1, 1], 
                      [(u_half[j] - a_half[j]), u_half[j], (u_half[j] + a_half[j])], 
                      [(H_half[j] - u_half[j] * a_half[j]), 0.5*u_half[j]**2, (H_half[j] + u_half[j] * a_half[j])]])
        
        """
        S Inverse Matrix in A = SΛS–1
        """
        S_inv_1 = np.array([0.5*(alpha1[j] + u_half[j]/a_half[j]),
                            -0.5*(alpha2[j]*u_half[j] + 1/a_half[j]),
                            alpha2[j]/2]) # Row 1
        
        S_inv_2 = np.array([(1-alpha1[j]), alpha2[j]*u_half[j], -alpha2[j]]) # Row 2
        
        S_inv_3 = np.array([0.5*(alpha1[j] - u_half[j]/a_half[j]),
                            -0.5*(alpha2[j]*u_half[j] - 1/a_half[j]),
                            alpha2[j]/2]) # Row 3
        
        S_inv = np.array([S_inv_1, S_inv_2, S_inv_3])
        
        """
        A Matrix in A = SΛS–1
        """
        A = np.dot(np.dot(S, Lam), S_inv)

        """
        U_dif = |A| · (U[i+1] – U[i]) (eq16)
        """
        U_dif = np.roll(U, -1)[:,j] - U[:, j]
        Au_vecs[:, j] = np.dot(A, U_dif)
    
    """
    Update solution
    """
    F_halves = 0.5 * ( F + np.roll(F, -1) ) - 0.5 * Au_vecs
    F_half_pos = F_halves
    F_half_neg = np.roll(F_halves, 1)
    U_new = U - (dt/dx) * (F_half_pos - F_half_neg)
    U_new[:,0] = U0[:,0] ; U_new[:,-1] = U0[:,-1]   # Dirichlet bounds
    
    return U_new



def compute_error(u, u_hat):
    """
    Parameters
    ----------
    u : 1d array
        Numerical values.
    u_hat : 1d array
        Analytical values.

    Returns
    -------
    error : float
        Average L2 Norm.
    """
    sum_squares = np.sum((u - u_hat)**2 )
    error = np.sqrt(sum_squares) / len(u)
    
    return error






















