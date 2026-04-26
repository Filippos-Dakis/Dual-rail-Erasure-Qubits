import numpy as np
from scipy.special import ellipk, ellipkm1, ellipe, ellipeinc, ellipkinc, ellipj

# COMPLETE elliptic integral of FIRST kind 
def K_(p):
    return ellipk(p)*(p<=0.8) + ellipkm1(1- p) * (p>0.8)

# COMPLETE elliptic integral of SECOND kind
def E_(p):
    return ellipe(p)

# elliptic sine
def sn_(x,p=1):
    SN, _, _, _ = ellipj(x, p)
    return SN

# elliptic cosine
def cn_(x,p=1):
    _, CN, _, _ = ellipj(x, p)
    return CN


def dn_(x,p=1):
    _, _, DN, _ = ellipj(x, p)
    return DN


def am_(x,p=1):
    _, _, _, AMM= ellipj(x, p)
    return AMM


def lambda_func(p,sigma): 
    """Lambda function defined  "Backlund transformations and knots of constant torsion"  
       see Eq.(9)"""
    p_2 = p**2                                                                                  # geometric modulus squared
    E_p = E_(p_2)                                                                               # Complete eliptic integral of Second kind
    mu  = (1/4) * np.sqrt( (p**-2 - sigma**2)**2 + 4*sigma**2)

    lambda_1 = (sigma**2 - (p**-2) + 2) / 4                                                     # see page 9 
    ksi      = np.pi/2 - 2*np.arctan(2*sigma / (p**-2 - sigma**2 + 4*mu))                       # see Eq.(9)

    E_ksi_p_prime = ellipeinc(ksi,1-p_2)                                                        # Incomplete of Second kind                   
    F_ksi_p_prime = ellipkinc(ksi,1-p_2)                                                        # Incomplete of First kind

    Lambda = E_ksi_p_prime + (E_p/K_(p_2) - 1)* F_ksi_p_prime + lambda_1*p*sigma/mu             # see Eq. (9)

    return Lambda


def F_fun(p, sigma):
    """F(ksi,p) defined  "Backlund transformations and knots of constant torsion"  
       see Eq.(9)"""
    p_2 = p**2                                                                                  # geometric modulus squared
    mu  = (1/4) * np.sqrt( (p**-2 - sigma**2)**2 + 4*sigma**2)

    ksi = np.pi/2 - 2*np.arctan(2*sigma / (p**-2 - sigma**2 + 4*mu))

    F_ksi_p_prime = ellipkinc(ksi,1-p_2)
    
    return F_ksi_p_prime

# === Central difference for derivatives ===
def central_diff(f, dt):
    df       = np.zeros_like(f)
    df[1:-1] = (f[2:] - f[:-2]) / (2 * dt)
    df[0]    = (f[1] - f[0]) / dt
    df[-1]   = (f[-1] - f[-2]) / dt
    return df

def Theta_(p, x, N = 200):
    # Theta_0 function 
    # N : number of orders in the summation

    p_2       = p**2
    p_2_prime = 1 - p_2
    Kp        = K_(p_2)
    Kp_prime  = K_(p_2_prime)
    q         = np.exp(-np.pi * Kp_prime/Kp)
    u         = x*np.pi/(2* Kp)

    m     = np.arange(1, N+1)[:, np.newaxis]         # shape (N, 1)
    u     = np.atleast_1d(u)[np.newaxis, :]          # ensures u has shape (1, M)

    terms = (-1)**m * q**(m**2) * np.cos(2 * m * u)  # shape (N, M)

    return 1 + 2 *np.sum(terms, axis =0)                   


def Theta_1(p, x, N = 200):
    # Theta_1 function 
    # N : number of orders in the summation
    p_2       = p**2
    p_2_prime = 1 - p_2
    Kp        = K_(p_2)
    Kp_prime  = K_(p_2_prime)
    q         = np.exp(-np.pi * Kp_prime/Kp)
    u         = x*np.pi/(2* Kp)

    m     = np.arange(1, N+1)[:, np.newaxis]  # shape (N, 1)
    u     = np.atleast_1d(u)[np.newaxis, :]   # ensures u has shape (1, M)

    terms = q**(m**2) * np.cos(2 * m * u)     # shape (N, M)

    return 1 + 2 *np.sum(terms, axis =0)   


def Eta_1(p, x, N = 200):
    # Eta_1 function 
    # N : number of orders in the summation
    p_2       = p**2
    p_2_prime = 1 - p_2
    Kp        = K_(p_2)
    Kp_prime  = K_(p_2_prime)
    q         = np.exp(-np.pi * Kp_prime/Kp)
    u         = x*np.pi/(2* Kp)

    m     = np.arange(1, N+1)[:, np.newaxis]           # shape (N, 1)
    u     = np.atleast_1d(u)[np.newaxis, :]            # ensures u has shape (1, M)


    terms = q**((m - 1/2)**2) * np.cos( (2*m-1) * u)  # shape (N, M)

    return 2 *np.sum(terms, axis =0)


def Rcostheta(p, sigma, mu, x):
    # X component 
    # N : number of orders in the summation
    p_2     = p**2
    p_prime = np.sqrt(1-p_2)

    c1 = np.sqrt(2*K_(p_2)*p_prime / (p*np.pi))

    Lambda = lambda_func(p, sigma)
    F_hat  = F_fun(p,sigma)

    THETA_0 = Theta_(p, x ,200)

    THETA_1a = Theta_1(p, x - 1j*F_hat ,200)
    THETA_1b = Theta_1(p, x + 1j*F_hat ,200)

    ETA_1   = Eta_1(p, 1j*F_hat ,200)

    numerator = np.exp(-1j * Lambda *x) * THETA_1a + np.exp(1j * Lambda * x) * THETA_1b
    denom     = 2 * mu * THETA_0 * ETA_1

    return c1 * numerator/denom

def Rsintheta(p, sigma, mu, x):
    # Y component
    # N : number of orders in the summation
    p_2     = p**2
    p_prime = np.sqrt(1-p_2)

    c1 = np.sqrt(2*K_(p_2)*p_prime / (p*np.pi))

    Lambda = lambda_func(p, sigma)
    F_hat  = F_fun(p,sigma)

    THETA_0 = Theta_(p, x ,200)

    THETA_1a = Theta_1(p, x - 1j*F_hat ,200)
    THETA_1b = Theta_1(p, x + 1j*F_hat ,200)

    ETA_1   = Eta_1(p, 1j*F_hat ,200)

    numerator = np.exp(-1j * Lambda *x) * THETA_1a - np.exp(1j * Lambda * x) * THETA_1b
    denom     = 1j * 2 * mu * THETA_0 * ETA_1

    return c1 * numerator/denom

def jacobiZ(p,x, N=200):
    # Jaobi Z function
    # N : number of orders in the summation
    p_2       = p**2
    p_2_prime = 1 - p_2
    Kp        = K_(p_2)
    K_p_prime = K_(p_2_prime)

    m     = np.arange(1, N+1)[:, np.newaxis]                                  # shape (N, 1)
    u     = np.atleast_1d(x)[np.newaxis, :]                                   # ensures u has shape (1, M)

    terms = np.sin(m * np.pi * u / Kp) / np.sinh(m * np.pi * K_p_prime / Kp)  # shape (N, M)

    return (np.pi/Kp) *np.sum(terms, axis =0)

def zeta_x(p,mu, x, N=200):
    # Z component using Jacobi function
    # N : number of orders in the summation
    return (1/(mu * p)) * jacobiZ(p, x, N)

def zeta_x2(p,mu, x, N=200):
    # Z component using Theta function and its derivative
    # N : number of orders in the summation
    THETA_0       = Theta_(p,x, N)
    dx            = x[1] - x[0]
    THETA_0_prime = central_diff(THETA_0, dx)

    return (1/(mu * p)) * THETA_0_prime / THETA_0

def fourier_coefs(t, y, N_order):
    """ This function computes the Fourier series coefficients
     Inputs:
        t       : time vector
        y       : function we want to expand in Fourier series
        N_order : highest order we keep in the expansion
    Outputs:
        c_n     : complex fourier coeefficients     
        
    """
    period   = t[-1] - t[0]                                     # time windows in which we compute the Fourier expansion
    N_order += 1                                                # total orders, we start from 0-th order DC component 
    c_n       = np.zeros(N_order)*(1+1j)                        # initialize the array of coefficients
    c_n[0]    = y.sum()/y.size                                  # 0th order coefficient, DC component
    
    for n in range(1,N_order):
       c      = y*( np.cos(2*n*np.pi*t/period) + 1j* np.sin(2*n*np.pi*t/period) )     # integrant
       c_n[n] = 2 * np.sum(c)/c.size                                                  # numerical integration

    return c_n

def f_in_FS(t, cn):
    """ This function computes the Fourier expansion
    Inputs:
        t   : time vector
        cn  : complex Fourier coefficients 
    Outputs:
        f   : Fourier expansion  
    """
    period        = t[-1] - t[0]                                # time windows in which we compute the Fourier expansion
    c_n_real      = np.real(cn)                                 # cosine coefs
    c_n_imaginary = np.imag(cn)                                 # sine coefs
    f             = c_n_real[0] * np.ones(len(t)) + 1j*0        # DC componenet

    for n in range(1,len(cn)):
        f += c_n_real[n] * np.cos(n * 2*np.pi * t/period) + c_n_imaginary[n] * np.sin(n * 2*np.pi *t/period)  # add order by order
   
    return f

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def frenet_rhs(t, Y, kappa_fun, tau_fun, speed_fun):
    """ This function computes the LHS of the FS equations  
    Inputs:
        t         : time vector
        Y         : TNB vectors at t=0
        kappa_fun : curvature
        tau_fun   : torsion
        speed_fun : speed function
    Outputs:
        dT, dN, dB : left hand side of FS equations
    """
    T = Y[0:3]                                  # initial Tangent
    N = Y[3:6]                                  # initial Normal
    B = Y[6:9]                                  # initial Binormal

    k   = kappa_fun(t)                          # curvature non-constant
    tau = tau_fun                               # torsion here is constant !!
    v   = speed_fun                             # speed here is constant !!! 

    dT = v * k * N                              # FS equations
    dN = v * (-k * T + tau * B)                 #
    dB = v *(-tau * N)                          #

    return np.concatenate([dT, dN, dB])


def pulse_optimizer(c_n, Y0, time, speed, R_target, T_g):
    
    T0 = Y0[0:3]                                                                    # inital T, N, B vectors
    N0 = Y0[3:6]                                                                    #
    B0 = Y0[6:9]                                                                    #

    l         = int((len(c_n)-2)/2)                                                 # first "2*l" elements are the Fourier coeffs
    cn        = c_n[0:l] + 1j*c_n[l :-2]                                            # fourier coefs         (to be optimized)
    ampl      = c_n[-2]                                                             # curvature amplitude   (to be optimized)
    c_t       = c_n[-1]                                                             # constant torsion      (to be optimized)

  
    time = np.linspace(0, T_g, 2**10)
    
    curvature = ampl * f_in_FS(time, cn)                                            # compute the curvature
    # torsion   = c_t                                                               # constant torsion
    

    kappa_fun = interp1d(time, curvature, kind='cubic', fill_value="extrapolate")   # interp needed for sole_ivp

    """
    # tau_fun   = interp1d(time, torsion,   kind='cubic', fill_value="extrapolate")      #
    # taun_fun = c_t
    # speed_fun = interp1d(time, speed,     kind='cubic', fill_value="extrapolate")      #
    # speed_fun = speed 
    """

    sol = solve_ivp(                                                                # solve FS-equations for a set of (kappa, tau)
        fun    = lambda t, y: frenet_rhs(t, y, kappa_fun, c_t, speed, ),
        t_span = (time[0], time[-1]),
        y0     = Y0,
        t_eval = time,
        method = 'RK45',
        rtol   = 1e-9,
        atol   = 1e-9
    )

    T = sol.y[0:3, :].T                                                             # tangent vector T(t)  shape (N,3)
    N = sol.y[3:6, :].T                                                             # normal vector N(t)         (N,3)
    B = sol.y[6:9, :].T                                                             # binormal vector B(t)       (N,3)
    
    # dt  = time[1] - time[0]                                                       # dt needed for Tangent integration
    # r   = np.cumsum(T * speed[:,None], axis=0) * dt                               # curve 
    # r   = np.cumsum(T, axis=0) * speed * dt
    # DR  = np.linalg.norm( r[-1,:] - r[0,:] )                                      # closure condition   (part of the cost function)
    r   = np.trapezoid(T, time*speed, axis=0)
    DR  = np.linalg.norm(r)
    eqs = []                                                                        # conte
    eqs.append(DR)                                                                  # closed curve condition

    #B_times_kappa = B * curvature[:, np.newaxis]                                    # B(t) * |curvature(t)|
    #integral      = np.trapezoid(B_times_kappa,time*speed, axis=0)                  # integral_0^Tg  B(t) * |curvature(t)| dt
    #tangent_area  = np.linalg.norm( integral)                                       # norm of total tangent area
    #eqs.append(tangent_area/10) 
    

    R_F_0  = np.array([-B0, N0, T0])                                                # initial frame matrix
    R_F_Tg = np.array([-B[-1,:], N[-1,:], T[-1,:]])                                 # frame matrix at t=T_g 
    R_u0     = R_F_Tg @ (R_F_0.T)                                                   # SO(3) gate matrix

    eqs.append(1- fidelity(R_u0, R_target))                                         # gate fidelity 

    eqs.append( R_u0[0,0] - R_target[0,0] )                                         # conditions for Ryz(pi/2) in SO(3)
    eqs.append( R_u0[1,1] - R_target[1,1] )                                         # these elements must be zero
    eqs.append( R_u0[2,2] - R_target[2,2] )
    eqs.append( R_u0[0,1] - R_target[0,1] )                                         # 
    eqs.append( R_u0[0,2] - R_target[0,2] )
    eqs.append( R_u0[1,2] - R_target[1,2] )

    
                                    
    eqs.append(0.0 if abs(np.real(curvature[0]))  < 1e-2 else np.real(curvature[0]) )   # pulse starts from 0
    eqs.append(0.0 if abs(np.real(curvature[-1])) < 1e-2 else np.real(curvature[-1]))   # pulse ends at 0

    #eqs.append(0.0 if abs(c_t)< 0.5 else c_t)                                          # torsion condition 
    #eqs.append(0.0 if abs(c_t)> 0.005 else 1/(c_t +1))                                 # torsion condition 
  
    # eqs.append(0.0 if  np.max(np.abs(curvature))*time[-1] < 20 else (20 - np.max(np.abs(curvature))*time[-1]) )

    #eqs.append(0.0 if abs(1/c_t) < 1000 else np.abs(1/c_t))

    # m1 = abs(c_t) * T_g
    # m2 = np.max( abs(curvature/(4*c_t)) )

    #eqs.append( 0.0 if m1 <= 1. else (m1 -1) )
    #eqs.append( 0.0 if m2 <= 1. else (m2 -1) )

    print(eqs[0:3])                                    # prints the values of the parameters you want to minimize

    return eqs


def R_rodrigues(v, theta):
    """ This function uses Rodrigues rotation formula and returns a rotation matrix of angle \theta around axis v
    Inputs:
        v       : unit vector in 3D space defining the axis of rotation
        theta   : angle of rotation
    Outputs:
        R       : rotation matrix in SO(3) """
    
    v = v / np.linalg.norm(v)                                       # make sure the vector is normalized
    vx = v[0]                                                       # x component
    vy = v[1]                                                       # y component
    vz = v[2]                                                       # z component
    K = np.array([[0., -vz, vy],                                    # K matrix (see Wikipedia Rodrigues rotation formula)
                  [vz, 0., -vx],
                  [-vy, vx, 0.]])
    K2 = K @ K                                                      # K^2 matrix 
    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta)) * K2        # final rotation matrix
    return R

def fidelity(R_g, R_u0, d = 2):
    """ This function computes the fidelity between two 3x3 matrices. See Eq.(23) in "An automated geometric space curve approach for designing
        dynamically corrected gates"  """
    
    return (d + 1 + np.trace((R_g.T)  @ R_u0)) / (d * (d+1))

def adjoint_Representation(U):
    s_x = np.array([[0, 1],
                    [1, 0]])

    s_y = np.array([[0, -1j],
                    [1j, 0]])

    s_z = np.array([[1,  0],
                    [0, -1]])

    S = np.stack([s_x, s_y, s_z])
    
    R_U0 = np.zeros((3,3))
    U_dagger = U.conj().T

    for p in range(0,3):
        for q in range(0,3):
            R_U0[p, q] = np.real( (1/2) * np.trace(U_dagger @ S[p,:,:] @ U @ S[q,:,:]) )

    return R_U0

def expm(v, phi):
    s_x = np.array([[0, 1],
                    [1, 0]])

    s_y = np.array([[0, -1j],
                    [1j, 0]])

    s_z = np.array([[1,  0],
                    [0, -1]])
    v = v/np.linalg.norm(v)
    return np.cos(phi)*np.eye(2) + 1j*np.sin(phi) *(v[0] * s_x + v[1]*s_y + v[2]*s_z)

def print_matrix_sci(A, precision=3, width=10):
    """
    Pretty-print a 2D NumPy array in scientific notation,
    wrapped in an ASCII box:
      /          \
      | … matrix |
      \          /
    
    Parameters:
    - A: np.ndarray, 2D array (matrix)
    - precision: int, number of decimal places (default: 3)
    - width: int, total width of each printed element (default: 10)
    """
    if A.ndim != 2:
        raise ValueError("Input must be a 2D array (matrix).")
    
    rows, cols = A.shape
    # Format string for each element
    fmt = f"{{: {width}.{precision}e}}"
    # Compute the inner width: total characters in one row
    inner_width = cols * width + (cols - 1) * 1  # spaces between elements
    
    # Top border
    print("/" + " " * inner_width + "\\")
    # Matrix rows
    for row in A:
        line = " ".join(fmt.format(val) for val in row)
        print("|" + line + "|")
    # Bottom border
    print("\\" + " " * inner_width + "/")

def adj_2_SU2(R):
    theta = np.arccos((np.trace(R)-1)/2)
    nx = R[2,1] - R[1,2]
    ny = R[0,2] - R[2,0]
    nz = R[1,0] - R[0,1]
    n  = [nx, ny, nz] #/ (2*np.sin(theta))
    return expm(n, theta/2)

