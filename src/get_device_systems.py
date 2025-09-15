import sympy as sp
import control as ct

from src.get_controllers import sympy_to_tf

def get_hydro_tf():
    """
    returns:
        hydro system
        pi controler parameters
        hydro time constant
    """
    # get hydro system from Verena paper
    Rg, Rt = 0.03, 0.38
    taug, taur, tauw = 0.2, 5, 1
    hydro_t_constant = Rt/Rg*taur

    def get_hydro_tf():
        s = sp.symbols('s')
        T_hydro = -1/Rg / (taug*s + 1) * (taur*s+1) / (Rt/Rg*taur*s+1) * (1-tauw*s) / (1+1/2*tauw*s)
        return sympy_to_tf(sp.simplify(T_hydro))

    T_hydro = get_hydro_tf()
    T_hydro = ct.tf(T_hydro.num, T_hydro.den, inputs=['u'], outputs=['y'])

    # set pi params for hydro
    # calculated and tuned in matlab
    ps = {"kp": -0.0796, "ki": -0.09788}
    
    return T_hydro, ps, hydro_t_constant

def get_bess_io_sys(tau_BESS=0.1, t_drop=7, drop_exp=1.2):
    """
    params:
        tau_BESS: time constant of the battery
        t_drop: time after which the battery output starts to drop (in seconds)
    """
    p_BESS = {'tau': tau_BESS, 't_drop': t_drop, 'drop_exp': drop_exp} 

    def update(t, x, u, params={}):
        tau = params.get('tau')
        x0 = - 1/tau * x[0] + u[0]         # np.clip(u[0], -np.inf, E_max - x[1])
        x1 = x[0]                         # energy state
        return [x0, x1]
    
    def output(t, x, u, params={}):
        if t <= params.get('t_drop'):
            return 1/params.get('tau') * x[0]  # no limitation
        else:
            return 1/params.get('tau') * x[0] / ((t-params.get('t_drop'))**params.get('drop_exp'))

    return ct.NonlinearIOSystem(
        update, output, inputs=['u'], outputs=['y'], states=2, params=p_BESS
    )

def get_sc_io_sys(tau_SC=0.1, t_drop=1, drop_exp=2):
    """
    params:
        tau_BESS: time constant of the battery
        t_drop: time after which the battery output starts to drop (in seconds)
    """
    p_BESS = {'tau': tau_SC, 't_drop': t_drop, 'drop_exp': drop_exp} 

    def update(t, x, u, params={}):
        tau = params.get('tau')
        x0 = - 1/tau * x[0] + u[0]         # np.clip(u[0], -np.inf, E_max - x[1])
        x1 = x[0]                         # energy state
        return [x0, x1]
    
    def output(t, x, u, params={}):
        if t <= params.get('t_drop'):
            return 1/params.get('tau') * x[0]  # no limitation
        else:
            return 1/params.get('tau') * x[0] / ((t-params.get('t_drop'))**params.get('drop_exp'))

    return ct.NonlinearIOSystem(
        update, output, inputs=['u'], outputs=['y'], states=2, params=p_BESS
    )