import itertools
import os

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import Loading_code, read_code


def get_pr(d: int, noises):
    """

    :param d: distance of the code
    :param noises: list of noise values for which we calculate the participation ratio
    :return: participation ratio
    """
    # generate all possible error configurations
    errs = torch.as_tensor(list(map(list, itertools.product([0, 1], repeat=d))))

    # think about second part of parity check matrix
    print(os.getcwd())
    # info = read_code(d=d, k=1, seed=0, c_type='rsur')
    if d == 3:
        stabilizer = torch.as_tensor([[2, 2, 0], [0, 2, 2]])
        logical = torch.as_tensor([[2, 2, 2]])
        pure_es = torch.as_tensor([[0, 1, 1], [1, 1, 0]])
    elif d == 5:
        stabilizer = torch.as_tensor([[2, 2, 0, 0, 0], [0, 2, 2, 0, 0], [0, 0, 2, 2, 0], [0, 0, 0, 2, 2]])
        logical = torch.as_tensor([[2, 2, 2, 2, 2]])
        pure_es = torch.as_tensor([[0, 1, 1, 1, 1], [1, 1, 0, 0, 0], [0, 0, 0, 1, 1], [1, 1, 1, 1, 0]])
    elif d == 7:
        stabilizer = torch.as_tensor(
            [[2, 2, 0, 0, 0, 0, 0], [0, 2, 2, 0, 0, 0, 0], [0, 0, 2, 2, 0, 0, 0], [0, 0, 0, 2, 2, 0, 0],
             [0, 0, 0, 0, 2, 2, 0], [0, 0, 0, 0, 0, 2, 2]])
        logical = torch.as_tensor([[2, 2, 2, 2, 2, 2, 2]])
        pure_es = torch.as_tensor(
            [[0, 1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1, 0]])
    else:
        raise Exception('Distance ', d, ' not yet supported.')
    info = (stabilizer, logical, pure_es)
    Code = Loading_code(info)

    # e1 = Code.pure_es
    # for i in range(Code.m):
    #     conf = commute(e1[i], e1)
    #     idx = conf.nonzero().squeeze().to('cpu')
    #     sta = Code.g_stabilizer[idx]
    #     e1[i] = opts_prod(torch.vstack([e1[i], sta]))

    # g = torch.vstack([e1, Code.logical_opt, Code.g_stabilizer]
    # get the syndrome for each error
    syndromes = list(commute(errs, Code.g_stabilizer))
    # get the logical operator for each error
    log_confs = list(commute(errs, Code.logical_opt))
    errs = list(errs)
    # Here, the pr for each different noise value is stored
    result = torch.zeros(len(noises))
    test = torch.zeros(len(noises))

    # get a list of all occurring syndromes
    possible_syndromes = list({tuple(tensor.tolist()): tensor for tensor in syndromes}.values())
    for s in tqdm(possible_syndromes):
        # print('neu')
        # list of all errors and logical operators whose syndrome is equal to s
        errors_for_syndrome = [(error, logical) for syndrome, error, logical in zip(syndromes, errs, log_confs) if
                               torch.equal(s, syndrome)]
        # calculate probability of the syndrome
        p_s = torch.sum(torch.stack([p_error(error, noises) for error, _ in errors_for_syndrome]), dim=0)
        # calculate probability of the logical operator given the syndrome
        p_lx = torch.sum(torch.stack([p_error(error, noises) for error, logical in errors_for_syndrome if
                                      torch.equal(logical, torch.tensor(1, device='cpu'))]), dim=0) / p_s
        p_l1 = torch.sum(torch.stack([p_error(error, noises) for error, logical in errors_for_syndrome if
                                      torch.equal(logical, torch.tensor(0, device='cpu'))]), dim=0) / p_s
        # p_lz = torch.sum(torch.stack([p_error(error, noises) for error, logical in errors_for_syndrome if
        #                               torch.equal(logical, torch.tensor([1, 0], device='cpu'))]), dim=0) / p_s
        # p_ly = torch.sum(torch.stack([p_error(error, noises) for error, logical in errors_for_syndrome if
        #                               torch.equal(logical, torch.tensor([1, 1], device='cpu'))]), dim=0) / p_s

        # calculate participation ratio
        result += p_s * (p_lx ** 2 + p_l1 ** 2)  # + p_ly ** 2 + p_lz ** 2)
        test += p_s
    print('total', test)
    # return participation ratio
    return result


def p_error(error, p):
    res = ((p) ** torch.sum(error > 0) * (1 - p) ** (len(error) - torch.sum(error > 0)))
    return res


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


if __name__ == '__main__':
    noises = torch.arange(0, 0.7, 0.01)
    for d in [3, 5, 7]:
        pr = get_pr(d=d, noises=noises)
        plt.plot(noises, pr, label='d={}'.format(d))
    plt.legend()
    plt.show()
