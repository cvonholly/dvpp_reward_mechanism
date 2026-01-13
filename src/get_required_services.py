from src.check_qualification import create_curve

def get_ffr(T_MAX_FFR=10, n_points=1000,
            make_step=True):
    """
    Create FFR service requirement curves
    """
    # FFR
    t_a_max = .7  # activation time. is .7s, 1s or 1.3s depending on frequency deviation
    t_a = 1  # activation time for the minimum curve
    Dp = 1   # set to 1 per unit
    MP_max = 1 *  Dp  # set to 1 per unit * Dp
    START = MP_max if make_step else 0
    require_ffr = {(0, t_a): (0, 0), (t_a, 5): (1/Dp, 1/Dp), (5, 10): (1/Dp, 0)}
    require_ffr_max = {(0, t_a_max): (START, MP_max), (t_a_max, 5): (MP_max, MP_max), (5, 10): (MP_max, 0)}
    ts_ffr_max, input_ffr_max = create_curve(require_ffr_max, t_max=T_MAX_FFR, n_points=n_points)
    _, curve_ffr_min = create_curve(require_ffr, t_max=T_MAX_FFR, n_points=n_points)
    return ts_ffr_max, input_ffr_max, curve_ffr_min

# FCR
def get_fcr(T_MAX_FCR=60, n_points=1000,
            make_step=True):
    """
    Create FCR service requirement curves
    """
    ti, ta, = 2, 30
    Dp = 1   # set to 1 per unit
    RP_MAX = 1  # se to 1 per unit
    START = RP_MAX if ti==0 else 0
    require_fcr = {(ti, ta): (0, 1/Dp), (ta, 60): (1/Dp, 1/Dp)}
    require_fcr_max = {(0, ta): (START, RP_MAX), (ta, 60): (RP_MAX, RP_MAX)}
    ts_fcr_max, input_fcr_max = create_curve(require_fcr_max, t_max=T_MAX_FCR, n_points=n_points)
    _, curve_fcr_min = create_curve(require_fcr, t_max=T_MAX_FCR, n_points=n_points)  # minimum curve
    return ts_fcr_max, input_fcr_max, curve_fcr_min

# FCR-D
def get_fcr_d(T_MAX=60, n_points=1000,
              make_step=True):
    D1 = .86  # at 7.5s
    DFINAL = 1
    DMAX = 1 
    DMAX_FINAL = 1.05
    START = DMAX if make_step else .25
    require_fcr_d = {(0.05, 7.5): (0, D1), (7.5, T_MAX): (D1, DFINAL)}
    require_fcr_d_max = {(0, 7.5): (START, DMAX), (7.5, T_MAX): (DMAX, DMAX_FINAL)}
    ts_fcr_d_max, input_fcr_d_max = create_curve(require_fcr_d_max, t_max=T_MAX, n_points=n_points)
    _, curve_fcr_d_min = create_curve(require_fcr_d, t_max=T_MAX, n_points=n_points)  # minimum curve
    return ts_fcr_d_max, input_fcr_d_max, curve_fcr_d_min


def get_ffr_fcr(T_MAX=60, n_points=1000):
    """
    Create combined FFR + FCR-N service requirement curves

    split 50% FFR and 50% FCR-N
    """
    ts_fcr_max, input_fcr_max, curve_fcr_min = get_fcr(T_MAX_FCR=T_MAX, n_points=n_points,
                                                       make_step=False)
    _, input_ffr_max, curve_ffr_min = get_ffr(T_MAX_FFR=T_MAX, n_points=n_points)
    curve_ffrfcr_min = 1/2 * curve_fcr_min + 1/2 * curve_ffr_min
    input_ffrfcr_max = 1/2 * input_fcr_max + 1/2 * input_ffr_max
    return ts_fcr_max, input_ffrfcr_max, curve_ffrfcr_min

def get_ffr_fcr_d(T_MAX=60, n_points=1000):
    """
    Create combined FFR + FCR-D service requirement curves

    split 50%/50% between FFR and FCR-D
    """
    ts_fcr_max, input_fcr_max, curve_fcr_min = get_fcr_d(T_MAX=T_MAX, n_points=n_points,
                                                       make_step=False)
    _, input_ffr_max, curve_ffr_min = get_ffr(T_MAX_FFR=T_MAX, n_points=n_points)
    curve_ffrfcr_min = 1/2 * curve_fcr_min + 1/2 * curve_ffr_min
    input_ffrfcr_max = 1/2 * input_fcr_max + 1/2 * input_ffr_max
    return ts_fcr_max, input_ffrfcr_max, curve_ffrfcr_min

