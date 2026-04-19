from functools import partial

import numpy as np
import scipy
import torch
import sys
import time
from numba import njit
# import fssa

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, minimize
from tqdm import tqdm

"""
Script for analytically calculating the participation ratio and related quantities.
Uses Numba for translation to machine code and faster execution since we need to loop over 
2**(d**2-1) possible syndromes.
Includes some other useful methods such as bootstrap resampling.
"""


# np.int = int


def progressbar(it, prefix="", size=60, out=sys.stdout):
    """
    Custom progressbar.
    """
    count = len(it)
    start = time.time()  # time estimate start

    def show(j):
        x = int(size * j / count)
        # time estimate calculation and string
        remaining = ((time.time() - start) / j) * (count - j)
        mins, sec = divmod(remaining, 60)  # limited to minutes
        time_str = f"{int(mins):02}:{sec:03.1f}"
        print(f"{prefix}[{u'█' * x}{('.' * (size - x))}] {j}/{count} Est wait {time_str}", end='\r', file=out,
              flush=True)

    show(0.1)  # avoid div/0
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True, file=out)


def simple_bootstrap(x, f=np.mean, c=0.68, r=10):
    """ Use bootstrap resampling to estimate a statistic and
    its uncertainty.

    x (1d array): the data
    f (function): the statistic of the data we want to compute
    c (float): confidence interval in [0, 1]
    r (int): number of bootstrap resamplings

    Returns estimate of stat, upper error bar, lower error bar.
    """
    assert 0 <= c <= 1, 'Confidence interval must be in [0, 1].'
    # number of samples
    n = len(x)

    # stats of resampled datasets
    fs = np.asarray(
        [f(x[np.random.randint(0, n, size=n)]) for _ in range(r)]
    )
    # estimate and upper and lower limits
    med = 50  # median of the data
    val = np.percentile(fs, med)
    high = np.percentile(fs, med * (1 + c))
    low = np.percentile(fs, med * (1 - c))
    # estimate and uncertainties
    return val, high - val, val - low



def w_func(xs, ys, ys_eb):
    y_bar = ((xs[2] - xs[1]) * ys[0] - (xs[0] - xs[1]) * ys[2]) / (xs[2] - xs[0])

    delta_square = ys_eb[1] ** 2 + (ys_eb[0] * (xs[2] - xs[1]) / (xs[2] - xs[0])) ** 2 + (
                ys_eb[2] * (xs[0] - xs[1]) / (xs[2] - xs[0])) ** 2
    return np.power(ys[1] - y_bar, 2) / delta_square


def convert_data(list_ps, list_Is, list_Is_eb, Ls):
    main_list = []

    n_sizes = len(Ls)
    main_list.append(n_sizes)

    # compute offsets

    for idx, L in enumerate(Ls):
        offset = 0
        # offset += 1 # L
        offset += len(list_ps[idx])  # xs
        # offset += len(list_Is[idx]) #ys
        # offset += len(list_Is_eb[idx]) #ys_eb

        main_list.append(offset)

    # add data

    for idx, L in enumerate(Ls):
        main_list.append(L)
        main_list.extend([x for x in list_ps[idx]])
        main_list.extend([x for x in list_Is[idx]])
        main_list.extend([x for x in list_Is_eb[idx]])

    return tuple(main_list)


def objective_function(X, *args, transition="first", derivative=0):
    "Objective function for collapse"

    pc, inverse_nu = X

    # args = args_all[0]

    n_sizes = args[0]
    total_points = np.array(args[1:n_sizes + 1])

    xs = []
    ys = []
    ys_eb = []

    general_offset = 1 + len(total_points)
    size_offset = 0

    for idx, l in enumerate(range(n_sizes)):

        start = general_offset + size_offset

        L = args[start]

        xs_old = np.array(args[start:start + total_points[idx]])

        ys_old = np.array(args[start + total_points[idx]:start + 2 * total_points[idx]])

        ys_eb_old = np.array(args[start + 2 * total_points[idx]:start + 3 * total_points[idx]])

        if transition == "first" and derivative == 0:
            xs.extend((xs_old - pc) * np.power(L, inverse_nu))  # This is the scaling function
            ys.extend(ys_old)
            ys_eb.extend(ys_eb_old)

        elif transition == "first" and derivative == 1:

            xs.extend((xs_old - pc) * np.power(L, inverse_nu))  # This is the scaling function
            ys.extend(ys_old / np.power(L, inverse_nu))
            ys_eb.extend(ys_eb_old)

        size_offset += 3 * total_points[idx] + 1  # update the offset

    xs = np.array(xs)
    ys = np.array(ys)
    ys_eb = np.array(ys_eb)

    indices = np.argsort(xs)

    xs = xs[indices]
    ys = ys[indices]
    ys_eb = ys_eb[indices]

    O = 0.0

    for i in range(len(xs) - 2):
        Xs = xs[i:i + 3]
        Ys = ys[i:i + 3]
        Eb = ys_eb[i:i + 3]

        O += w_func(Xs, Ys, Eb)

    # print(O)
    return O / (len(xs) - 2)


def compute_critical_exponents(list_xs, list_ys, list_ys_eb, Ls, guess_xc, guess_nu, transition, derivative):
    parameters = convert_data(list_xs, list_ys, list_ys_eb, Ls)

    fixed_func = partial(objective_function, transition=transition, derivative=derivative)

    x0 = guess_xc
    y0 = 1 / guess_nu

    # res = minimize(objetcive_function, (x0,y0), args=parameters, method='Nelder-Mead', tol=1e-5, options={'disp': False})

    function_values = []

    def wrapped_func(x, *args):
        val = fixed_func(x, *args)
        function_values.append(val)
        return val

    value_at_start = fixed_func((x0, y0), *parameters)

    # print("Function value at initial point:", value_at_start)
    # Run the optimization
    res = minimize(
        wrapped_func,
        (x0, y0),
        args=parameters,
        method='Nelder-Mead',
        tol=1e-5,
        options={'disp': False}
    )

    # Now `function_values` contains the function value at each evaluation
    # print(function_values)

    # res = minimize(fixed_func, (x0, y0), args=parameters, method='Nelder-Mead', tol=1e-5, options={'disp': False})

    return res


def compute_error_bar(list_xs, list_ys, list_ys_eb, Ls, guess_xc, guess_nu, transition, derivative):
    "Compute error bar in critical parameters"

    n_samples = 250

    pcs = np.zeros(n_samples)

    inv_nus = np.zeros(n_samples)

    for i in range(n_samples):

        list_ys_sample = []
        for x, d in enumerate(Ls):
            # ys_l = np.zeros(list_xs[x].shape[0])
            # for idx,e in enumerate(list_xs[x]):

            ys_l = np.random.normal(list_ys[x], list_ys_eb[x], np.array(list_xs[x]).shape)

            list_ys_sample.append(ys_l)

        res = compute_critical_exponents(list_xs, list_ys_sample, list_ys_eb, Ls, guess_xc, guess_nu, transition,
                                         derivative)

        pc, inv_nu = res.x

        pcs[i] = pc
        inv_nus[i] = inv_nu

    c = 0.95
    med = 50
    val = np.percentile(pcs, med)
    high = np.percentile(pcs, med * (1 + c))
    low = np.percentile(pcs, med * (1 - c))

    errorbar_pc = high - low

    val = np.percentile(inv_nus, med)
    high = np.percentile(inv_nus, med * (1 + c))
    low = np.percentile(inv_nus, med * (1 - c))

    errorbar_inv_nu = high - low

    return errorbar_pc, errorbar_inv_nu


def read_code(d, k, seed=0, c_type='sur'):
    try:
        return torch.load('src/qec_transformer/code/' + c_type + '_d{}_k{}_seed{}'.format(d, k, seed))
    except FileNotFoundError:
        return torch.load('code/' + c_type + '_d{}_k{}_seed{}'.format(d, k, seed))


class Loading_code():
    """
    Class to load stabilizer and logical operators of any qec code.
    """

    def __init__(self, info):
        self.g_stabilizer, self.logical_opt, self.pure_es = info[0], info[1], info[2]
        self.n, self.m = self.g_stabilizer.size(1), self.g_stabilizer.size(0)
        self.physical_qubits = self.get_physical_qubits()
        self.PCM = PCM(self.g_stabilizer)

    def get_physical_qubits(self):
        gs = self.g_stabilizer
        n, m = self.n, self.m
        phys = {}.fromkeys(range(n))
        for i in range(n):
            phys[i] = []
            for j in range(m):
                if gs[j, i] != 0:
                    phys[i].append(j)
        return phys


def PCM(g_stabilizer):
    """get parity check matrix from stabilizers"""
    n = g_stabilizer.size(1)
    M = rep(input=g_stabilizer)
    PCM = torch.zeros_like(M)
    PCM[:, :n], PCM[:, n:] = M[:, n:], M[:, :n]
    return PCM


def rep(input, dtype=torch.float32, device='cpu'):
    """turn [0, 1, 2, 3] to binary [[0, 0], [0, 1], [1, 0], [1, 1]] """
    if len(input.size()) <= 1:
        input = torch.unsqueeze(input, dim=0)
    n, m = input.size(1), input.size(0)
    bin = torch.zeros(m, n * 2, device=device, dtype=dtype)
    # print(bin.size(), input.size())
    bin[:, :n] = input % 2
    bin[:, n:] = torch.floor(input / 2)
    return bin.to(dtype).to(device)


def xyz(b_opt):
    """inverse of rep() function, turn binary operator to normal representation"""
    if len(b_opt.size()) == 2:
        n = int(b_opt.size(1) / 2)
        opt = b_opt[:, :n] + 2 * b_opt[:, n:]
    elif len(b_opt.size()) == 1:
        n = int(b_opt.size(0) / 2)
        opt = b_opt[:n] + 2 * b_opt[n:]
    return opt.squeeze()


def opt_prod(opt1, opt2, dtype=torch.float32, device='cpu'):
    """operator product"""
    opt1, opt2 = opt1.to(dtype).to(device), opt2.to(dtype).to(device)
    '''proudct rule of V4 group'''
    b_opt1, b_opt2 = rep(opt1), rep(opt2)
    b_opt = (b_opt1 + b_opt2) % 2
    '''turn [[0, 0], [0, 1], [1, 0], [1, 1]] back to binary [0, 1, 2, 3]'''
    opt = xyz(b_opt)
    return opt.squeeze()


def opts_prod(list_of_opts, dtype=torch.float32, device='cpu'):
    """product of list of operators"""
    s = list_of_opts.to(dtype).to(device)
    if len(s.size()) == 2:
        opt1 = s[0]
        a = 0
    else:
        a = 1
        opt1 = s[:, 0, :]
    b_opt1 = rep(opt1)
    for i in range(s.size(a) - 1):
        if len(s.size()) == 2:
            opt2 = s[i + 1]
        else:
            opt2 = s[:, i + 1, :]
        b_opt2 = rep(opt2)
        b_opt1 = (b_opt1 + b_opt2) % 2
    '''turn [[0, 0], [0, 1], [1, 0], [1, 1]] back to binary [0, 1, 2, 3]'''
    opt = xyz(b_opt1)
    return opt.squeeze()


def confs_to_opt(confs, gs, dtype=torch.float32, device='cpu'):
    """
        get operators from the configurations and generators
        confs: (batch, m), configurations
        gs : (m, n), generators
    """
    confs, gs = confs.to(dtype).to(device), gs.to(dtype).to(device)
    if len(torch.tensor(confs.size())) <= 1:
        confs = torch.unsqueeze(confs, dim=0)
    batch = confs.size(0)
    s = torch.tensor([gs.tolist()] * batch, device=device, dtype=dtype).permute((2, 0, 1))
    s = (s * confs).permute((1, 2, 0))
    opt = opts_prod(s)
    return opt


def commute(a, b, intype=('nor', 'nor'), dtype=torch.float32, device='cpu'):
    """
        calculate commutation relation of operators
    """
    a, b = a.to(dtype).to(device), b.to(dtype).to(device)
    if len(a.size()) < 2:
        a = torch.unsqueeze(a, dim=0)
    if len(b.size()) < 2:
        b = torch.unsqueeze(b, dim=0)
    if intype == ('nor', 'nor'):
        I = torch.eye(a.size(1), device=device, dtype=dtype)
        bin_a = rep(a).squeeze()
        bin_b = rep(b).squeeze()
    elif intype == ('bin', 'bin'):
        I = torch.eye(int(a.size(1) / 2), device=device, dtype=dtype)
        bin_a = a
        bin_b = b
    elif intype == ('nor', 'bin'):
        I = torch.eye(a.size(1))
        bin_a = rep(a).squeeze()
        bin_b = b
    elif intype == ('bin', 'nor'):
        I = torch.eye(int(a.size(1) / 2), device=device, dtype=dtype)
        bin_a = a
        bin_b = rep(b).squeeze()

    Zero = torch.zeros_like(I, device=device, dtype=dtype)
    A = torch.cat([Zero, I], dim=0)
    B = torch.cat([I, Zero], dim=0)
    gamma = torch.cat([A, B], dim=1)
    return ((bin_a @ gamma @ bin_b.T) % 2).squeeze()


@njit
def compute_syndrome(ex, ez, ps, stabilizer, logical, gamma):
    ys = ex * ez
    xs = ex - ys
    zs = ez - ys
    total = ys + xs + zs

    pe = ((ps / 3) ** np.sum(total > 0) * (1 - ps) ** (len(ex) - np.sum(total > 0)))

    e = np.concatenate((ex, ez)).reshape(1, -1).astype(np.float32)

    s = ((stabilizer @ gamma @ e.T) % 2).reshape(-1, 1)
    l = ((logical @ gamma @ e.T) % 2).reshape(-1, 1)

    return s, l, pe


@njit
def inner_loop(error_z, ex, ps, stabilizer, logical, gamma, syndrome_p, logical_p):
    for ez in error_z:
        s, l, pe = compute_syndrome(ex, error_z, ps, stabilizer, logical, gamma)

        s_decimal = binary_to_decimal_array(s)
        l_decimal = binary_to_decimal_array(l)

        syndrome_p[s_decimal] += pe
        logical_p[s_decimal, l_decimal] += pe


@njit
def decimal_to_binary_array(decimals, bit_width):
    # Prepare the output array
    n = len(decimals)
    binary_array = np.zeros((n, bit_width), dtype=np.uint8)

    # Loop through each bit position
    for j in range(bit_width):
        # Create a bit mask for the j-th bit
        mask = 1 << (bit_width - j - 1)

        # Extract the j-th bit across all decimal values and store it
        binary_array[:, j] = (decimals & mask) >> (bit_width - j - 1)

    return binary_array


@njit
def binary_to_decimal_array(binary):
    # Get the number of binary numbers and their bit width
    binary = binary.ravel().astype(np.uint8)
    decimal_value = 0  # Initialize as an integer
    for j in np.arange(binary.size):
        decimal_value |= binary[j] << (binary.size - j - 1)
    return decimal_value


@njit
def compute_syndrome_batch(ex, error_z_batch, ps, stabilizer, logical, gamma, mode):
    batch_size, ex_length = error_z_batch.shape

    # Broadcasting `ex` to match the batch size
    ex_expanded = ex[np.newaxis, :]
    ex_expanded = np.broadcast_to(ex_expanded, (batch_size, ex_length))

    # Forming e_batch
    e_batch = np.hstack((ex_expanded, error_z_batch)).astype(np.float32)

    # Compute ys, xs, zs, total in one go
    ys = ex_expanded * error_z_batch
    xs = ex_expanded - ys
    zs = error_z_batch - ys
    total = ys + xs + zs

    # Active counts
    active_counts = np.sum(total > 0, axis=1)

    # Prepare ps and pe
    len_ps = len(ps)
    pe = np.empty((len_ps, batch_size), dtype=np.float32)

    if mode == 'depolarizing':
        for i in range(len_ps):
            pe[i, :] = ((ps[i] / 3) ** active_counts) * ((1 - ps[i]) ** (len(ex) - active_counts))
    else:
        for i in range(len_ps):
            pe[i, :] = ((ps[i]) ** active_counts) * ((1 - ps[i]) ** (len(ex) - active_counts))

    # Compute s and l
    s = (stabilizer @ (gamma @ e_batch.T)) % 2
    l = (logical @ (gamma @ e_batch.T)) % 2

    return s.T, l.T, pe


@njit
def inner_loop_vectorized(error_z, ex, ps, stabilizer, logical, gamma, syndrome_p, logical_p, mode):
    # Compute `s`, `l`, and `pe` for the batch of `error_z`
    s_batch, l_batch, pe_batch = compute_syndrome_batch(ex, error_z, ps, stabilizer, logical, gamma, mode)

    # Convert `s_batch` and `l_batch` to decimal indices
    for i in np.arange(s_batch.shape[0]):
        s_decimal = binary_to_decimal_array(s_batch[i])
        l_decimal = binary_to_decimal_array(l_batch[i])

        # Accumulate probabilities in `syndrome_p` and `logical_p`
        syndrome_p[s_decimal] += pe_batch[:, i]
        logical_p[s_decimal, l_decimal] += pe_batch[:, i]


@njit
def normalize_p(logical_p, syndrome_p, ps, s):
    for l in np.arange(np.shape(logical_p)[1]):
        for n in np.arange(len(ps)):
            if np.abs(syndrome_p[s, n]) < 1e-10:
                continue
            logical_p[s, l, n] /= syndrome_p[s, n]


# @njit(nogil=True)
def get_pr(d: int, ps, stabilizer, logical, qubits: int, mode='depolarizing', approx=None):
    if approx is None:
        approx = qubits

    # All possible X errors, written as [0, 0, 1] for an X error on qubit 3.
    # Stored as decimal numbers
    errs = np.arange(2 ** (qubits))

    # Array to save syndrome probabilities
    syndrome_p = np.zeros((2 ** (qubits - 1), len(ps)))
    # Array to save logical operator probabilities.
    logical_p = np.zeros((2 ** (qubits - 1), 4, len(ps)))

    error_x = decimal_to_binary_array(errs, qubits)
    print(np.shape(error_x))
    # print(error_x)

    # Approximation: Considers only error chains that act on up to int(approx) qubits.
    if approx < qubits:
        na = 0
        for i in range(approx + 1):
            na += int(scipy.special.binom(qubits, i))
        error_x_2 = np.zeros((na, qubits))
        idx = 0
        for i, e in enumerate(error_x):
            if np.sum(e) <= approx:
                error_x_2[idx] = error_x[i]
                idx += 1
        error_x = error_x_2
        # print(error_x)
    print(np.shape(error_x))
    if mode == 'depolarizing':
        # Same z errors as x errors
        error_z = error_x.copy()
    else:
        error_z = np.zeros((1, qubits)).astype(np.float32)

    identity = np.eye(qubits)
    zero = np.zeros_like(identity)

    gamma = np.vstack((np.hstack((zero, identity)), np.hstack((identity, zero)))).astype(np.float32)
    stabilizer = stabilizer.astype(np.float32)
    logical = logical.astype(np.float32)

    # Use tqdm here for tracking progress
    for ex in tqdm(error_x):
        inner_loop_vectorized(error_z, ex, ps, stabilizer, logical, gamma, syndrome_p, logical_p, mode)
        # progress.update(1)

    for s in np.arange(np.shape(logical_p)[0]):
        normalize_p(logical_p, syndrome_p, ps, s)

    # assert np.sum(np.abs(np.sum(logical_p, axis=1) - np.ones((2 ** (qubits - 1), len(ps))))) < 1e-3
    # assert np.sum(np.abs(np.sum(syndrome_p, axis=0) - np.ones(len(ps)))) < 1e-3
    epsilon = 1e-12

    # Define different quantities that show the phase transition. They are all based on some non-linear function
    # that is continuous in [0,1].

    # f(x) = x * log(x) --> equal to coherent information. See notes.
    ci = np.log(2) + np.sum(syndrome_p * np.sum((logical_p) * np.log(logical_p + epsilon), axis=1), axis=0)
    entropy = - np.sum(syndrome_p * np.sum((logical_p) * np.log(logical_p + epsilon), axis=1), axis=0)

    # f(x) = x**2 --> participation ratio averaged over different syndromes.
    pr = np.sum(syndrome_p * np.sum((logical_p) ** 2, axis=1), axis=0)

    var = np.sum(syndrome_p * np.var(logical_p[:, [0, 2], :], axis=1, ddof=0), axis=0)
    # dif = 4/3 * pr - 1/3 - (ci/(2 * np.log(2)) + 0.5)
    dif = np.sum(
        syndrome_p * np.sum(4 / 3 * logical_p ** 2 - 1 / (2 * np.log(2)) * logical_p * np.log(logical_p + epsilon),
                            axis=1), axis=0) - 4 / 3
    # result = np.sum(np.sum((logical_p) ** 2, axis=1), axis=0)

    return ci / np.log(2), syndrome_p, logical_p


def get_abort_probability(thresholds, distance):
    noises = np.arange(0., 0.4, 0.01)
    assert distance <= 5
    result = np.zeros((len(thresholds), len(noises)))
    for j, c in enumerate(thresholds):
        print(c)
        for i, d in enumerate([distance]):
            p_abort = np.zeros_like(noises)
            syndrome_p = np.loadtxt(f'src/analytical/s_p_d{d}')
            logical_p = np.loadtxt(f'src/analytical/log_p_d{d}')
            logical_p = logical_p.reshape(logical_p.shape[0], logical_p.shape[1] // len(noises), len(noises))

            mask = (logical_p > 1 - c).any(axis=1)
            mask = ~mask
            for n in np.arange(len(noises)):
                accepted_s = syndrome_p[mask[:, n]]
                p_abort[n] += accepted_s[:, n].sum()
        result[j] = p_abort
    return noises, result


def find_crossing(noises, distances, dict1, dict2, noise_model, quant):
    '''

    :param noises:
    :param distances:
    :param predictions:
    :param uncertainties:
    :param n_samples:
    :param p0:
    :return:

    Insert this at top of optimize.py:

    numpy.int = int

    def wrap_function(function, args):
        ncalls = [0]
        if function is None:
            return ncalls, None

        def function_wrapper(*wrapper_args):
            ncalls[0] += 1
            return function(*(wrapper_args + args))

        return ncalls, function_wrapper
    '''

    '''
    if noise_model == 'depolarizing':
        p0 = 0.18
    elif noise_model == 'phenomenological':
        p0 = 0.04
    elif noise_model == 'circuit-level':
        p0 = 0.003
    else:
        raise ValueError(f'noise_model {noise_model} not supported')

    predictions = np.zeros((len(n), 2))
    uncertainties = np.zeros((len(n), 2))

    pr = list(dict1.values())  # list of tuples containing mean, uplimit, lowlimit
    # get here median, upper error bar, lower error bar
    pr_m = list(map(lambda x: x[0], pr))
    predictions[:, 0] = pr_m

    pr = list(dict2.values())  # list of tuples containing mean, uplimit, lowlimit
    # get here median, upper error bar, lower error bar
    pr_m = list(map(lambda x: x[0], pr))
    predictions[:, 1] = pr_m

    if quant == 'ci':
        unc = list(map(lambda x: 0.5 * (x[1] + x[2]), pr))
        uncertainties[:, 0] = unc
        unc = list(map(lambda x: 0.5 * (x[1] + x[2]), pr))
        uncertainties[:, 1] = unc

    result = fssa.autoscale(l=distances, rho=noises, a=np.transpose(predictions), da=np.transpose(uncertainties),
                            rho_c0=p0, nu0=1, zeta0=1)
    return result '''
    raise NotImplementedError('Need to uncommented code and insert given function in scipy.optimize.')


def find_crossing2(noises, dict1, dict2, noise_model, quant, n_samples=10000, p0=None):
    """
    Find the crossing by doing spline interpolation of two given arrays.
    :param n_samples:
    :param uncertainties: Uncertainties for predictions
    :param noises: Noise strengths.
    :param predictions: shape (len(noises)x2): predictions for class 0 and 1
    :return:
    """
    if p0 is None:
        if noise_model == 'depolarizing':
            p0 = [0.12, 0.19]
        elif noise_model == 'phenomenological':
            p0 = [0.12, 0.19]
        elif noise_model == 'circuit-level':
            p0 = [0.12, 0.19]
        else:
            raise ValueError(f'noise_model {noise_model} not supported')

    predictions = np.zeros((len(noises), 2))
    uncertainties = np.zeros((len(noises), 2))
    maximum = len(noises)

    if quant == 'abort':
        predictions[:, 1] = np.ones(len(noises))

    print('Reminder: right factor for uncertainty? Assume: num=100,000.')
    if quant == 'ci' or quant == 'pl':
        pr = list(dict1.values())  # list of tuples containing mean, uplimit, lowlimit
        # get here median, upper error bar, lower error bar
        pr_m = list(map(lambda x: x[0], pr))
        predictions[:len(pr_m), 0] = pr_m
        maximum = min(maximum, len(pr_m))

        pr = list(dict2.values())  # list of tuples containing mean, uplimit, lowlimit
        # get here median, upper error bar, lower error bar
        pr_m = list(map(lambda x: x[0], pr))
        predictions[:len(pr_m), 1] = pr_m
        maximum = min(maximum, len(pr_m))

        unc = list(map(lambda x: 0.5 * (x[1] + x[2]), pr))
        uncertainties[:len(pr_m), 0] = unc
        unc = list(map(lambda x: 0.5 * (x[1] + x[2]), pr))
        uncertainties[:len(pr_m), 1] = unc
    elif quant == 'abort':
        pr = list(dict1.values())  # list of tuples containing mean, uplimit, lowlimit
        # get here median, upper error bar, lower error bar
        pr_m = list(map(lambda x: x[3], pr))
        predictions[:len(pr_m), 0] = pr_m
        maximum = min(maximum, len(pr_m))

        pr = list(dict2.values())  # list of tuples containing mean, uplimit, lowlimit
        # get here median, upper error bar, lower error bar
        pr_m = list(map(lambda x: x[3], pr))
        predictions[:len(pr_m), 1] = pr_m
        maximum = min(maximum, len(pr_m))

        uncertainties[:len(pr_m), 0] = np.sqrt(np.array(pr_m) / 100000)
        uncertainties[:len(pr_m), 1] = np.sqrt(np.array(pr_m) / 100000)
    else:
        raise ValueError(f'quant {quant} not supported')

    def linear_crossing(x1, x2, y10, y20, y11, y21):
        m0 = (y20 - y10) / (x2 - x1)
        m1 = (y21 - y11) / (x2 - x1)
        n0 = -m0 * x1 + y10
        n1 = -m1 * x1 + y11
        denom = m0 - m1
        if denom == 0:
            return -1  # Parallel lines
        crossing = (n1 - n0) / denom
        return crossing if x1 < crossing < x2 else -1

    all_crossings = []

    for _ in range(n_samples):
        # Sample from normal distributions
        sampled_preds = np.random.normal(loc=predictions, scale=uncertainties)

        for i in range(len(noises) - 1):
            x1, x2 = noises[i], noises[i + 1]
            y10, y20 = sampled_preds[i, 0], sampled_preds[i + 1, 0]
            y11, y21 = sampled_preds[i, 1], sampled_preds[i + 1, 1]
            cross = linear_crossing(x1, x2, y10, y20, y11, y21)
            if cross != -1 and p0[0] < cross < p0[1]:
                all_crossings.append(cross)
    # print(all_crossings)
    return np.mean(all_crossings), np.std(all_crossings, ddof=1)


def function(p, alpha, t):
    return alpha * p ** (t + 1)


def get_scaling(p, y, y_err):
    """
        Fit y = alpha * p^(t + 1) to the given data with uncertainties.

        Parameters:
            p (array-like): Input x-values.
            y (array-like): Observed y-values.
            y_err (array-like): Uncertainties in y-values.

        Returns:
            popt (ndarray): Optimal values for the parameters (alpha, p).
            perr (ndarray): One standard deviation errors on the parameters.
        """
    p = np.asarray(p)
    y = np.asarray(y)
    y_err = np.asarray(y_err)

    # Initial guess: alpha ~ y[0], p ~ (y[-1]/y[0])**(1/(t[-1] - t[0]))
    alpha0 = 10
    t0 = 2
    initial_guess = [alpha0, t0]

    popt, pcov = curve_fit(function, p, y, sigma=y_err, absolute_sigma=True, p0=initial_guess)
    perr = np.sqrt(np.diag(pcov))  # standard deviation errors from covariance matrix

    return popt, perr


if __name__ == '__main__':
    # Analytically calculating the participation ratio

    epsilon = 1e-12
    noises = np.arange(0., 0.4, 0.01)

    for d in [3]:
        g_stabilizer = np.loadtxt('code/stabilizer_' + 'rsur' + '_d{}_k{}'.format(d, 1))
        print(g_stabilizer)
        logical_opt = np.loadtxt('code/logical_' + 'rsur' + '_d{}_k{}'.format(d, 1))
        pr, s_p, l_p = get_pr(d=d, ps=noises, stabilizer=g_stabilizer, logical=logical_opt, qubits=d ** 2,
                              mode='depolarizing')
        # np.savetxt(f's_p_d{d}_surface', s_p)
        # np.savetxt(f'l_p_d{d}_surface', l_p.reshape(l_p.shape[0], -1))
        plt.plot(noises, pr, label='d={}_surface'.format(d))

    plt.vlines(0.189, -1, 1, linestyles='dashed', color='red', label='threshold')
    # plt.hlines(0.25, 0, 0.4, linestyles='dashed', color='black', linewidth=1)
    plt.legend()
    plt.xlabel('noise probability p')
    plt.ylabel('participation ratio')
    # plt.ylim(0.45, 0.65)
    plt.show()

