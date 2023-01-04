"""
Created on Wed Mar 23 13:36:21 2022
@author: Sasha Bakker
"""
import numpy as np
import sod_tube_functions as stf
import copy
import sys
import gitlab_sod_analytical as gsa
import time


#dx = 0.00125 # Case 1
#dx = 0.0025 # Case 2
#dx = 0.005 # Case 3
#dx = 0.01 # Case 4
#dx = 0.02 # Case 5

gamma = 1.4                         # Ratio of specific heats
dx = 1/400                          # Step size
x_i = 0.0; x_f = 1.0                # Limits of computational domain
xs = np.arange(x_i, x_f + dx, dx)   # Mesh over the computational domain
nx = len(xs)                        # Number of elements in mesh
sigma = 0.9                         #
nt = 402                            # Number of time steps (ends at 0.2 s)
dt = 0.0005                         # Time step
ti = 0.0; tf = (nt-1)*dt            # Limits of time domain
ts = np.linspace(ti, tf, nt)        # Mesh over time domain
skip = 20 #400                          # Number of plots to skip
savemyplot = False                  # Boolean T/F to save plots in directory


# Build Initial Conditions
p0 = np.ones(nx)                    # Pressures p 
r0 = np.ones(nx)                    # Densisites ρ
u0 = np.ones(nx)                    # Velocities u
nx_half = int(nx/2)                 # 1/2 of number of elements

# Shocktube Problem
p0[:nx_half] *= 1.0  ; p0[nx_half:] *= 0.1
r0[:nx_half] *= 1.0  ; r0[nx_half:] *= 0.125
u0[:nx_half] *= 0.0  ; u0[nx_half:] *= 0.0

# Construct U, F(U)
U0, F0 = stf.construct_system(p0, r0, u0, gamma)

# Construct U_new to hold updated solution
U_new = np.zeros(len(U0))

method_names = ["Lax-Wendroff", "Lax-Wendroff with AV", "Godunov",
                "Godunov with AV", "Hyman Predictor Corrector", "Roe's Scheme", "Analytical (GitLab)"]

method_ids = np.arange(1, len(method_names)+1, 1)

for ids in method_ids:
    
    method_name = method_names[ids-1]
    print(f"Enter {ids} for {method_name}")
    
selected_method = int(input("\nEnter here: "))



if (selected_method in method_ids) == False:
    print(f"Error: Input '{selected_method}' does not correspond to a method.")
    sys.exit()

method_name = method_names[selected_method - 1]

start_time = time.time()

if selected_method != 7:
    """
    NUMERICAL SOLUTION
    """
    for n in range(1, nt):
        
        # `U` = solution (n-1); U_new = solution (n)
        if n == 1:
            F = copy.deepcopy(F0)
            U = copy.deepcopy(U0)
        else:
            F = stf.construct_F(U, gamma)
        
        """
        Execute chosen method
        """
        if selected_method == 1:
            U_new = stf.lax_wendroff_solution(F, U, dx, dt, U0, F0, gamma, U_new) # Lax-Wendroff
            
        elif selected_method == 2:
            U_new = stf.lax_wendroff_solution(F, U, dx, dt, U0, F0, gamma, U_new)
            U_new = stf.add_artificial_viscosity(U_new, U0, 1, dt, dx) # Lax-Wendroff AV
        
        elif selected_method == 3:
            U_new = stf.godunov_solution(F, U, dx, dt, U0, F0, gamma, U_new) # Godunov
        
        elif selected_method == 4:
            U_new = stf.godunov_solution(F, U, dx, dt, U0, F0, gamma, U_new)
            U_new = stf.add_artificial_viscosity(U_new, U0, 2, dt, dx) # Godunov AV
        
        elif selected_method == 5:
            U_new = stf.hyman_predictor_corrector_solution(U, U_new, U0, gamma, dt, dx, nx) # Hyman
        
        elif selected_method == 6:
            U_new = stf.roe_solution(F, U, dx, dt, U0, gamma, U_new) # Roe's Scheme
            
        
        # Plot solution
        # ==================
        U_new, U = U, U_new                         # `U` = solution (n); U_new = solution (n-1)
        rho = U_new[0, :]                           # Densisites ρ
        m = U_new[1, :]                             # Momenta 
        e = U_new[2, :]                             # Energy per unit volume
        p = stf.compute_pressure(gamma, rho, m, e)   # Pressures p
        t = ts[n-1]                                 # Time at solution (n-1)
        u = m / rho
        
        if (n-1)%skip == 0:
            # Plot solution (n-1)
            stf.myplot(rho, u, p, x_i, x_f, xs, t, gamma, method_name, savemyplot, n)


elif selected_method == 7:
    """
    ANALYTICAL SOLUTION
    """
    for n in range(1, nt):
        
        if (n-1)%skip == 0:

            positions, regions, values = gsa.solve(left_state=(p0[0], r0[0], u0[0]), right_state=(p0[-1], r0[-1], u0[-1]), 
                                                       geometry=(x_i, x_f, 0.5), t=ts[n-1], gamma=gamma, npts=nx)
            p = values['p']
            rho = values['rho']
            u = values['u']
            
            # Plot solution
            stf.myplot(rho, u, p, x_i, x_f, xs, ts[n-1], gamma, method_name, savemyplot, n)

print("--- %s seconds ---" % (time.time() - start_time))       
        
    
    































    