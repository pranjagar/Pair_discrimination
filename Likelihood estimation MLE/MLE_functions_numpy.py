# MLE basic functions

import numpy as np
import pandas as pd
import math as m
import sympy as sym
import scipy as sci
import matplotlib.ticker as ticker
import random as rand
import matplotlib.pyplot as plt
import cmath as cm
from IPython.display import display, Latex
from sympy import pprint
from scipy.optimize import minimize

# Constants and fixed equations
w = m.e**((2/3)*m.pi*(1j))     # third root of unity
POVM_vec = (1/(2**.5))*(np.array([[0,1,-1],[-1,0,1],[1,-1,0],[0,w,-w**2],[-1,0,w**2],[1,-w,0],[0,w**2,-w],[-1,0,w],[1,-w**2,0]]))  # an array of POVM direction vectors
POVM_elts = [(1/3)*np.outer(np.conjugate(POVM_vec[i]),POVM_vec[i]) for i in range(len(POVM_vec))]   # a list of POVM matrices
M = [[np.trace(np.dot(POVM_elts[i],POVM_elts[j])) for i in range(len(POVM_elts))] for j in range(len(POVM_elts))]     # creating M matrix using POVM definition
u_0 = [1/3 for i in range(9)]           # cerating u_0 vector, to create the inverse matrix
M_inv = 3*np.outer(u_0,u_0) + 12*(np.eye(9) - np.outer(u_0,u_0))        # creating the inverse matrix

# Function Creaing States 
theta1, theta2, phi1, phi2, alpha = sym.symbols('theta1 theta2 phi1 phi2 alpha')
c1, c2, s1, s2, w1, w2 = sym.cos(theta1), sym.cos(theta2), sym.sin(theta1), sym.sin(theta2), sym.cos(alpha)**2, sym.sin(alpha)**2

params00 = [0,m.pi/2,0,0,m.pi/4]
params00complex = [0,m.pi/2,0,m.pi/2,m.pi/4]
# # analytic formuale for the probabilities, as given in the notes
pr1 = (1/6)*(w1*s1**2*(1+c1**2 -2*m.sqrt(2)*sym.cos(phi1)*s1*c1)+w2*s2**2*(1+c2**2-2*m.sqrt(2)*sym.cos(phi2)*s2*c2))
# pr2 = (1/6)*(w1*c1**2*(1+s1**2 -2*m.sqrt(2)*sym.cos(phi1)*c1*s1) + w2*c2**2*(1+s2**2*-2*m.sqrt(2)*sym.cos(phi2)*c2*s2))       # original
pr2 = (1/6)*(w1*abs(-c1**2 + m.sqrt(2)*sym.exp(sym.I*phi1)*s1*c1)**2 + w2*abs(-c2**2 + m.sqrt(2)*sym.exp(sym.I*phi2)*s2*c2)**2 )# trying this new one to see if equivalent
pr3 = (1/6)*(w1*(c1**4+s1**4 -2*sym.cos(2*phi1)*s1**2*c1**2)+ w2*(c2**4+ s2**4 - 2*sym.cos(2*phi2)*s2**2*c2**2))
pr4 = (1/6)*(w1*s1**2*(1+c1**2 -2*m.sqrt(2)*sym.cos(phi1+2*m.pi/3)*s1*c1)+ w2*s2**2*(1+c2**2-2*m.sqrt(2)*sym.cos(phi2+2*m.pi/3)*s2*c2))
pr5= (1/6)*(w1*c1**2*(1+s1**2 -2*m.sqrt(2)*sym.cos(phi1+2*m.pi/3)*c1*s1)+ w2*c2**2*(1+s2**2-2*m.sqrt(2)*sym.cos(phi2+2*m.pi/3)*c2*s2))
pr6 = (1/6)*(w1*(c1**4+s1**4-2*sym.cos(2*phi1 - 2*m.pi/3)*c1**2*s1**2)+ w2*(c2**4+s2**4-2*sym.cos(2*phi2 - 2*m.pi/3)*c2**2*s2**2))
pr7 = (1/6)*(w1*s1**2*(1+c1**2-2*m.sqrt(2)*sym.cos(phi1-2*m.pi/3)*s1*c1)+ w2*s2**2*(1+c2**2-2*m.sqrt(2)*sym.cos(phi2-2*m.pi/3)*s2*c2))
pr8 = (1/6)*(w1*c1**2*(1+s1**2-2*m.sqrt(2)*sym.cos(phi1-2*m.pi/3)*c1*s1)+ w2*c2**2*(1+s2**2-2*m.sqrt(2)*sym.cos(phi2-2*m.pi/3)*c2*s2))
pr9 = (1/6)*(w1*(c1**4+s1**4-2*sym.cos(2*phi1+2*m.pi/3)*c1**2*s1**2)+ w2*(c2**4+s2**4-2*sym.cos(2*phi2+2*m.pi/3)*c2**2*s2**2))
pr = [pr1, pr2,pr3, pr4,pr5, pr6,pr7, pr8,pr9]      # list of analytic prob.
pr0 = [(i.subs({phi1:0, phi2:0})).evalf() for i in pr]        # For the simplified case when There are no phi angles

pr_lambdified = [sym.lambdify((theta1, theta2, phi1, phi2, alpha), i) for i in pr]
def pr_num(vars):                           # simply input the list, not indicidiual angles
    return [i(*vars) for i in pr_lambdified]

def L(vars = False, n = False, p = False):        # vars is list of 5 angles
    n = [i/sum(n) for i in n]               # normalizing in case nju is given
    if vars is not False:
        prob = pr_num(vars)
        p = prob
    if p == False and vars == False:        #  theoretical minimum value, setting p = n
        p = n 
    lnl_p = [i * np.log(j) if j > abs(1e-17) else 0 for i, j in zip(n, p)]      # manually putting zero to avoid nan error
    loss = sum([abs(i) for i in lnl_p])
    return loss
#   L([618, 594, 1176, 995, 2316, 562, 1241, 1175, 1323])  # eg. theoretical min
#   L([618, 594, 1176, 995, 2316, 562, 1241, 1175, 1323], vars = [0, m.pi/2, 0, 0, m.pi/4])  # eg. Loss at the given angles with given collapses

def Creating_states(params):
    # Unpack parameters
    theta1, theta2, phi1, phi2, alpha = params

    # Calculate coefficients
    a0, a1 = np.cos(theta1), np.exp(1j*phi1)*np.sin(theta1)
    b0, b1 = np.cos(theta2), np.exp(1j*phi2)*np.sin(theta2)

    # Create states
    psi0 = np.array([a0, a1])
    psi1 = np.array([b0, b1])

    # Create square states
    psi0psi0 = np.array([psi0[0]*psi0[0], psi0[1]*psi0[1], np.sqrt(2)*psi0[0]*psi0[1]])
    psi1psi1 = np.array([psi1[0]*psi1[0], psi1[1]*psi1[1], np.sqrt(2)*psi1[0]*psi1[1]])
    # note the basis order is now how prof Hillery is using ie. [|00>, |11>, |+>] (instead of the previous order [|00>, |+>, |11> ])
    rho = np.cos(alpha)**2 * np.outer(psi0psi0, psi0psi0.conj()) + np.sin(alpha)**2 * np.outer(psi1psi1, psi1psi1.conj())
    return [[psi0, psi1], [psi0psi0, psi1psi1], rho]


def num_experiment(N = 10000, params = [0,m.pi/2,0,0,m.pi/4], seed = None, an_pr = True, nju = False):
    if seed is not None:
        rand.seed(seed)
    
    creation = Creating_states(params = params) # theoretical rho
    states = creation[0]
    sq_states = creation[1]
    rho = creation[2]
    prob_vec_raw =  [np.trace(np.dot(POVM_elts[i],rho)) for i in range(9)]    # created list of Th probabilities
    prob_vec = [round(i.real, 16) for i in prob_vec_raw if abs(i.imag) < 1e-14]          # cleaned up theoretical prob vector
    
    if an_pr == True:
        prob_vec = [i.subs({theta1: params[0], theta2: params[1], phi1:params[2], phi2: params[3], alpha: params[4]}) for i in pr]  # using prof hillery's analytic probabilities
        prob_vec = [round(i.evalf(), 16) for i in prob_vec]

    POVM_dir_symbols = ['d1','d2','d3','d4','d5','d6','d7','d8','d9']      # symbols to indicate collapsed direction
    #prob distribution is simply the corresponding elements of the prob_vec
    collapse_dir_vec = rand.choices(POVM_dir_symbols, weights=prob_vec, k = N)   # choosing collapse directions with weights for N trials
    
    nj_vec = [collapse_dir_vec.count(f'd{i+1}') for i in range(9)]
    if nju is not False:                                                    
        nj_vec = nju
    pj_num_vec = [i/sum(nj_vec) for i in nj_vec]                                  # numerical prob vector     

    r_vec = np.dot(M_inv,pj_num_vec)
    rho_num_list = [r_vec[i]*POVM_elts[i] for i in range(len(POVM_elts))]   # list of matrices, see equation 7 in notes pair_disc.pdf
    rho_num = np.zeros_like(rho_num_list[0])
    # Loop for reconstructing the numerical matrix
    for i in rho_num_list:
        rho_num = np.add(rho_num, i)
    return [states, rho, rho_num, prob_vec, nj_vec]

#fidelity function using parameters
def fid(params1 = [0,m.pi/2,0,0,m.pi/4], params2 = [0,m.pi/2,0,0,m.pi/4], coeff_mode = False ):
    #if simply the coeffs 4-lists are given
    if coeff_mode == True:
        fid0 = (abs(sum([np.conj(params1[i])*params2[i] for i in range(0,2)])))**2
        fid1 = (abs(sum([np.conj(params1[i])*params2[i] for i in range(2,4)])))**2
    else:               # generating the qm states from the parameters to find the fidelities
        states1 = Creating_states(params1)[0]
        states2 = Creating_states(params2)[0]

        # Calculate the fidelities between the corresponding states
        fid0 = np.abs(np.dot(states1[0].conj(), states2[0]))**2
        fid1 = np.abs(np.dot(states1[1].conj(), states2[1]))**2
    return [fid0, fid1]

def Inversion_new( N = 10000, params = [0,m.pi/2, m.pi/4], threshold = 'variable', seed = False, an_pr = True, nju = False):    # coeffs are as list [theta_1, theta_2, alpha] in radians
    # seed = 42
    experiment = num_experiment(N= N, params = params, an_pr= an_pr, seed = seed)
    if nju is not False:
        N = int(sum(nju))                                                    # changing N coz it later affects thresholds etc.
        experiment = num_experiment(N= N, params = params, an_pr= an_pr, seed = seed, nju = nju)        
    rho = experiment[1]
    rho_num = experiment[2]                                                # !!! can enhance using postprocessing here (2*rho_num = rho_num+ conj(rho_num)). can also do directly in num_expt funciton
    prob_vec = experiment[3]                                                # theoretical prob vector
    nj_vec = experiment[4]                                                  # unnormalized counts
    eigenvalues, eigenvectors = np.linalg.eig(rho_num)
    # evals = np.array([i.real for i in eigenvalues if abs(i.imag)< .0001])  # !!! does this rounding affect things later..?
    # evals = np.array([i.real for i in eigenvalues if abs(i.imag)< 1e-15])  # !!! does this rounding affect things later..?
    evals = eigenvalues
    evecs = np.around(eigenvectors, decimals=15)
    min_eval = np.min([abs(i) for i in evals])
    def solve_quadratic(a, b, c):
        d = (b**2) - (4*a*c)
        sol1 = (-b-np.sqrt(d))/(2*a)
        sol2 = (-b+np.sqrt(d))/(2*a)
        return [sol1, sol2]
    if len(params) ==3:                                 # For studying the simpler case with phis = 0
        a0, a1 = np.cos(params[0]), np.sin(params[0])
        b0, b1 = np.cos(params[1]), np.sin(params[1])
    elif len(params) == 5:
        a0, a1 = np.cos(params[0]), np.exp(params[2]*1j)*np.sin(params[0])
        b0, b1 = np.cos(params[1]), np.exp(params[3]*1j)*np.sin(params[1])
    c = [a0, a1, b0, b1]                                # four list of state coefficients

    if threshold != 'variable':
        if type(threshold) is str:
            print("Error! Threshold should be a number ")
    else:
        threshold = (4.2)/(N**.5)           # variable threshold

    xi = evecs[:, np.argmin([abs(i) for i in evals])]     # xi vector is same as the perp vector 
    c00,c01,c11 = xi[0],xi[2],xi[1]                         # defining In this weird way coz new basis arrangement, last elt is the |+> basis therefore c01 = xi[2]
    # the reconstructing the states' coefficients if-else loops
    c_raw = 0
    if abs(c00) > threshold and abs(c11) > threshold: 
        # print("condition 1")              
        if abs(c11)/abs(c00) < 2:
            soln  =  solve_quadratic(c00, np.sqrt(2)*c01, c11)
            # soln  =  solve_quadratic(c00, 2*c01, c11)
            z_a,z_b = soln[0], soln[1]
            a1 = (1/(np.sqrt(1+abs(z_a)**2)))           
            a0 = np.conj(z_a)*a1
            b1 = (1/(np.sqrt(1+abs(z_b)**2)))
            b0 = np.conj(z_b)*b1
            c_raw = [a0,a1,b0,b1]
        else:
            soln  =  solve_quadratic(c11, np.sqrt(2)*c01, c00)
            # soln  =  solve_quadratic(c11, 2*c01, c00)
            z_a,z_b = soln[0], soln[1]
            a0 = (1/(np.sqrt(1+abs(z_a)**2)))                          
            a1 = np.conj(z_a)*a0
            b0 = (1/(np.sqrt(1+abs(z_b)**2)))
            b1 = np.conj(z_b)*b0
            c_raw = [a0,a1,b0,b1]
            # print(z_a,z_b)
            # print('coeffs',a0,a1,b0,b1)
    elif abs(c00) <= threshold and abs(c11) > threshold:      # some cases are excluded if we don't use eqality
        a0, a1 = 1,0        # random choice, matching later
        # z = np.conj((-ca settrecer11)/(2*c01))     # z defined as (a0*/a1*)
        z = np.conj((-c11)/(np.sqrt(2)*c01))     # z defined as (a0*/a1*)
        b1= (1/(np.sqrt(1+abs(z)**2))) 
        b0= np.conj(z)*b1
        # print('condition #2')
        c_raw = [a0,a1,b0,b1]
    elif abs(c00) > threshold and abs(c11) <= threshold:
        a0, a1 = 0,1        
        # z = np.conj((-2*c01)/(c00))      # z defined as (b0*/b1*)
        z = np.conj((-(np.sqrt(2))*c01)/(c00))      # z defined as (b0*/b1*)
        b1= (1/(np.sqrt(1+abs(z)**2))) 
        b0= np.conj(z)*b1
        # print('condition #3, z = ',z)
        c_raw = [a0,a1,b0,b1]
    elif abs(c00) <= threshold and abs(c11) <= threshold:
        a0, a1 = 0,1        
        b0, b1 = 1,0     
        c_raw = [a0,a1,b0,b1]

    f00,f01 = (abs((c_raw[0]*c[0]+c_raw[1]*c[1])))**2, (abs((c_raw[0]*c[2]+c_raw[1]*c[3])))**2
    f10,f11 = (abs((c_raw[2]*c[0]+c_raw[3]*c[1])))**2, (abs((c_raw[2]*c[2]+c_raw[3]*c[3])))**2

    if  (f00 + f11) > (f01+ f10) :       
        c_num = c_raw
    else:
        c_num = [c_raw[2],c_raw[3],c_raw[0],c_raw[1]]

    F_a = (abs(sum([np.conj(c[i])*c_num[i] for i in range(0,2)])))**2
    F_b = (abs(sum([np.conj(c[i])*c_num[i] for i in range(2,4)])))**2
    # finding final parameters from the final coeffs
    tht1 = np.arccos(abs(c_num[0]))      # magnutude coz by construction any phase is pushed into the global phase (which also changes the relative phase)
    tht2 = np.arccos(abs(c_num[2]))
    #finding the phi angles
    phi_angles = np.angle(c_num)
    # angles = (180/m.pi)*angles  #in degrees
    phi_corrections = [180*(m.pi/180) if 90*(m.pi/180) < i < 180*(m.pi/180) or 270*(m.pi/180) < i < 360*(m.pi/180) else 0 for i in [tht1, tht2]]
    del_phis = [(phi_angles[0]-phi_angles[1])-phi_corrections[0], (phi_angles[2]-phi_angles[3])-phi_corrections[1]]
    phi1, phi2 = del_phis[0], del_phis[1]
    #finding the priors, see notebook
    psi0_num, psi1_num = [c_num[0], c_num[1]], [c_num[2], c_num[3]]
    psi0psi0_num = np.array([psi0_num[0]*psi0_num[0], psi0_num[1]*psi0_num[1], (np.sqrt(2)*psi0_num[0]*psi0_num[1])], dtype = complex) 
    psi1psi1_num = np.array([psi1_num[0]*psi1_num[0], psi1_num[1]*psi1_num[1], (np.sqrt(2)*psi1_num[0]*psi1_num[1])], dtype = complex)              # @@@ the square rooting is simplistic. see notebook for better. (cross term last due to prof's basis arrangement)
    overlap_num = np.dot(np.conjugate(psi0psi0_num),psi1psi1_num)
    psi1_cross_psi1_num = np.outer(psi1psi1_num, np.conjugate(psi1psi1_num))
    prior0 = (np.dot(np.dot(np.conjugate(psi0psi0_num), (rho_num - psi1_cross_psi1_num)), psi0psi0_num))/(1-abs(overlap_num)**2)        # see derivation in the notebook
    alpha = np.arccos(np.sqrt(prior0))
  
    psi0_cross_psi0_num = np.outer(psi0psi0_num, np.conjugate(psi0psi0_num))        # trying using the arcsin() of root(p1), earlier it was using the arc cos of root p0
    prior1 = (np.dot(np.dot(np.conjugate(psi1psi1_num), (rho_num - psi0_cross_psi0_num)), psi1psi1_num))/(1-abs(overlap_num)**2)        # see derivation in the notebook
    alpha_1 = np.arcsin(np.sqrt(prior1))
    
    final_params = [tht1, tht2, phi1, phi2, alpha]  
    
    Inversion_Fid_manual = [F_a, F_b]
    Inversion_fid_coeff = fid(c, c_num,  coeff_mode = True)
    Inversion_Fid_params = fid(params, final_params, coeff_mode = False)

    Fidelities = [Inversion_Fid_manual, Inversion_fid_coeff, Inversion_Fid_params]

    return [Fidelities , params ,final_params, rho, rho_num, prob_vec, nj_vec]

# Additional Fucntions 

def generate_collapses(params, N=1000, N_expt=2500):
    def measure_once(params, N):
        prob_vec = pr_num(params)
        POVM_dir_symbols = ['d1','d2','d3','d4','d5','d6','d7','d8','d9']
        collapse_dir_vec = rand.choices(POVM_dir_symbols, weights=prob_vec, k=N)
        nj_vec = [collapse_dir_vec.count(f'd{I+1}') for I in range(9)]
        return nj_vec
    nju_big_list = [measure_once(params, N) for _ in range(N_expt)]
    return nju_big_list

def ppm_errors(tru, nju, method = 'CG'):
    inv = Inversion_new(params = tru, N =int(sum(nju)), threshold = 'variable', nju = nju)
    inv_params = inv[2]
    sol = minimize(L, inv_params , method = method , args=(nju))
    sol_params = sol.x

    inv_params_deg = [i*180/m.pi for i in inv_params]
    sol_params_deg = [i*180/m.pi for i in sol_params]

    fid_inv_0 = fid(tru, inv_params)
    fid_inv_1 = fid([tru[1], tru[0], tru[3], tru[2], tru[4]], inv_params)
    fid_opt_0 = fid(tru, sol.x)
    fid_opt_1 = fid([tru[1], tru[0], tru[3], tru[2], tru[4]], sol.x)
    # choosing the biiger sum of fids, coz sometimes states come out "flipped"
    if sum(fid_inv_0) > sum(fid_inv_1):
        fid_inv = fid_inv_0
    else:
        fid_inv = fid_inv_1
    
    if sum(fid_opt_0) > sum(fid_opt_1):
        fid_opt = fid_opt_0
    else:
        fid_opt = fid_opt_1
    
    ppm_errors_inv = [(1-i)*1e6 for i in fid_inv]
    ppm_errors_opt = [(1-i)*1e6 for i in fid_opt]
    
    x= 7#dummy for breakpoint
    return [ppm_errors_inv, ppm_errors_opt]
