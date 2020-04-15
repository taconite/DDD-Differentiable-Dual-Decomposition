import torch
import time
import pdb
import random
import os

import copy

import numpy as np

assert torch.cuda.is_available()
cuda_device = torch.device('cuda')

seed = 132
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

from fixed_point_iteration import FixedPointIteration as FPI
from fixed_point_iteration import FixedPointIterationFast as FPIFast
# from lltm import LLTM

def generate_all_states(n_nodes, n_states):
    all_states = [[]]
    for n_idx in range(n_nodes):
        new_states = []
        for state in all_states:
            for s_idx in range(n_states):
                new_state = copy.copy(state)
                new_state.append(s_idx)
                new_states.append(new_state)

        all_states = new_states

    return all_states

batch_size = 12
state_size = 21
grid_h = 11
grid_w = 11


fpi = FPI(state_size).to(cuda_device)
fpi_fast = FPIFast(state_size).to(cuda_device)

targets = torch.randint(low=0, high=state_size, size=(batch_size, grid_h, grid_w), device=cuda_device, requires_grad=False)

xv, yv, zv = torch.meshgrid([torch.arange(0, batch_size), torch.arange(0, grid_h), torch.arange(0, grid_w)])

brute_force = False

if brute_force:
    all_states = generate_all_states(grid_h * grid_w, state_size)
    max_potential = -1e6
    for state in all_states:
        state = np.reshape(np.array(state, dtype=np.int32), (batch_size, grid_h, grid_w))
        potential = unary[xv, state, yv, zv].sum() + \
             horizontal_pairwise[xv[:, :, :grid_w-1], state[:, :, :grid_w-1], state[:, :, 1:], yv[:, :, :grid_w-1], zv[:, :, :grid_w-1]].sum() + \
             vertical_pairwise[xv[:, :grid_h-1, :], state[:, :grid_h-1, :], state[:, 1:, :], yv[:, :grid_h-1, :], zv[:, :grid_h-1, :]].sum()
        if potential > max_potential:
            max_potential = potential

    print('Ground-truth potential: {:.3f}'.format(max_potential))

criterion = torch.nn.CrossEntropyLoss().cuda()

for i in range(20):
    unary = ( np.random.rand(batch_size, state_size, grid_h, grid_w).astype(np.float32) - 0.5 ) * 20
    horizontal_pairwise = ( np.random.rand(batch_size, state_size, state_size, grid_h, grid_w-1).astype(np.float32) - 0.5 ) * 20
    vertical_pairwise = ( np.random.rand(batch_size, state_size, state_size, grid_h-1, grid_w).astype(np.float32) - 0.5 ) * 20
    for _ in range(10):
        unary_torch = torch.tensor(unary, device=cuda_device, requires_grad=True)
        pairwise_h_torch = torch.tensor(horizontal_pairwise, device=cuda_device, requires_grad=True)
        pairwise_v_torch = torch.tensor(vertical_pairwise, device=cuda_device, requires_grad=True)
        horizontal_unary = unary_torch / 2.0
        vertical_unary = unary_torch / 2.0

        # start = time.time()
        horizontal_unary, vertical_unary, horizontal_marginals, vertical_marginals = fpi_fast(
            horizontal_unary,
            vertical_unary,
            pairwise_h_torch,
            pairwise_v_torch
        )
        torch.cuda.synchronize()
        # if i >= 5:
        #     forward += time.time() - start

    marginals_fast = (horizontal_marginals + vertical_marginals).cpu().detach().clone().numpy()

    loss = criterion(horizontal_marginals + vertical_marginals, targets)

    # start = time.time()
    loss.backward()
    torch.cuda.synchronize()
    # backward += time.time() - start

    grad_u_fast = unary_torch.grad.cpu().detach().clone().numpy()
    grad_p_h_fast = pairwise_h_torch.grad.cpu().detach().clone().numpy()
    grad_p_v_fast = pairwise_v_torch.grad.cpu().detach().clone().numpy()
    grad_gamma_fast = fpi_fast.gamma.grad.item()
    print (loss)

    for _ in range(10):
        unary_torch = torch.tensor(unary, device=cuda_device, requires_grad=True)
        pairwise_h_torch = torch.tensor(horizontal_pairwise, device=cuda_device, requires_grad=True)
        pairwise_v_torch = torch.tensor(vertical_pairwise, device=cuda_device, requires_grad=True)
        horizontal_unary = unary_torch / 2.0
        vertical_unary = unary_torch / 2.0

        # start = time.time()
        horizontal_unary, vertical_unary, horizontal_marginals, vertical_marginals = fpi(
            horizontal_unary,
            vertical_unary,
            pairwise_h_torch,
            pairwise_v_torch
        )
        torch.cuda.synchronize()
        # if i >= 5:
        #     forward += time.time() - start

    marginals = (horizontal_marginals + vertical_marginals).cpu().detach().clone().numpy()

    loss = criterion(horizontal_marginals + vertical_marginals, targets)

    # start = time.time()
    loss.backward()
    torch.cuda.synchronize()
    # backward += time.time() - start

    grad_u = unary_torch.grad.cpu().detach().clone().numpy()
    grad_p_h = pairwise_h_torch.grad.cpu().detach().clone().numpy()
    grad_p_v = pairwise_v_torch.grad.cpu().detach().clone().numpy()
    grad_gamma = fpi.gamma.grad.item()
    print (loss)

    assert (np.all(np.abs(marginals - marginals_fast) < 1e-4))
    # assert (np.all(marginals - marginals_fast))

    assert (np.all(np.abs(grad_u_fast - grad_u) < 1e-4))
    assert (np.all(np.abs(grad_p_h_fast - grad_p_h) < 1e-4))
    assert (np.all(np.abs(grad_p_v_fast - grad_p_v) < 1e-4))

    assert (np.all(np.abs(grad_gamma_fast - grad_gamma) < 1e-4))
    # print (unary.grad)
    # print (horizontal_pairwise.grad.sum())
    # print (vertical_pairwise.grad.sum())

    # primal_sol = torch.argmax(horizontal_marginals + vertical_marginals, dim=1)
    # horizontal_sol = torch.argmax(horizontal_marginals, dim=1)
    # vertical_sol = torch.argmax(vertical_marginals, dim=1)
    # disagreement = (horizontal_sol != vertical_sol).sum()

    # lb = unary[xv, primal_sol, yv, zv].sum() + \
    #      horizontal_pairwise[xv[:, :, :grid_w-1], primal_sol[:, :, :grid_w-1], primal_sol[:, :, 1:], yv[:, :, :grid_w-1], zv[:, :, :grid_w-1]].sum() + \
    #      vertical_pairwise[xv[:, :grid_h-1, :], primal_sol[:, :grid_h-1, :], primal_sol[:, 1:, :], yv[:, :grid_h-1, :], zv[:, :grid_h-1, :]].sum()

    # ub = torch.max(horizontal_marginals[:, :, :, 0], dim=1)[0].sum() + \
    #      torch.max(vertical_marginals[:, :, 0, :], dim=1)[0].sum()

    # print('Upper bound: {:.3f}, lower bound: {:.3f}'.format(ub, lb))
    # print('# of disagreeing nodes: {}'.format(disagreement))

    # if (abs(lb - ub) < 1e-4):
    #     break

# print('Forward: {:.3f} s | backward: {:.3f} s'.format(forward / 15, backward / 15))

# forward = 0
# backward = 0
# for _ in range(100000):
#     start = time.time()
#     new_h, new_C = rnn(X, (h, C))
#     torch.cuda.synchronize()
#     forward += time.time() - start
#
#     start = time.time()
#     (new_h.sum() + new_C.sum()).backward()
#     torch.cuda.synchronize()
#     backward += time.time() - start
#
# print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))
