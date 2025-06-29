import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, \
    SequentialLR
from torch.utils.data import DataLoader
from typing import Any, Callable
from torch.optim import Optimizer

from .data.dataset import DepolarizingSurfaceData, BitflipSurfaceData
from tqdm import tqdm
from utils import simple_bootstrap

"""
This script contains the training and inference loop.
"""


def gaussian_kernel(x, y, sigma):
    """Compute the Gaussian kernel between two tensors."""
    if x.dtype is not torch.float32:
        x = x.to(dtype=torch.float32)
    if y.dtype is not torch.float32:
        y = y.to(dtype=torch.float32)
    x_norm = x.pow(2).sum(1).reshape(-1, 1)
    y_norm = y.pow(2).sum(1).reshape(1, -1)
    K = torch.exp(-((x_norm + y_norm - 2.0 * torch.mm(x, y.t())) / (2.0 * sigma ** 2)))
    return K


def mmd_loss(x, y, sigma=1.0):
    """Compute the MMD loss between samples x and y."""
    K_xx = gaussian_kernel(x, x, sigma)
    K_yy = gaussian_kernel(y, y, sigma)
    K_xy = gaussian_kernel(x, y, sigma)
    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()


def training_loop(model: nn.Module, dataset, val_set, init_optimizer: Callable[[Any], Optimizer], device, epochs=100,
                  batch_size=100, l=5., mode='depolarizing', activate_scheduler: bool = True,
                  include_mmd: bool = False, load_pretrained=False, act_wandb=True):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    model.to(device)

    optimizer = init_optimizer((model.parameters()))

    warmup_epochs = 5
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs)
    # decay_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
    # decay_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)

    counter = 0
    best_loss = float('inf')

    '''
    if load_pretrained:
        initial_requires_grad = {}  # Dictionary to store the original state
        for name, param in model.named_parameters():
            initial_requires_grad[name] = param.requires_grad  # Save the state
            param.requires_grad = False  # Freeze all parameters

        for name, param in model.positional_encoding.named_parameters():
            param.requires_grad = True
    '''
    for epoch in range(epochs):
        avg_loss = 0
        avg_mmd = torch.tensor(0.0, device=device)
        num_batches = 0
        '''
        if load_pretrained:
            if epoch == 2 * warmup_epochs:
                for name, param in model.named_parameters():
                    param.requires_grad = initial_requires_grad[name]  # Restore the original state
        '''
        # Training
        model.train()
        with tqdm(train_loader, unit="batch") as epoch_pbar:
            for batch in epoch_pbar:
                epoch_pbar.set_description(f"Epoch {epoch}")
                # add start token?
                # torch.mps.synchronize()
                # start = time.time()
                optimizer.zero_grad()

                # if batch_idx == 1:
                #    print('Train data: ', batch.data)

                # Possibility to include MMD loss for training.
                # Slows down training. Sample size needs to be very high to achieve similar mean embedding.
                # Mainly useful if autoregressive transformer is used for both next stabilizer and logical prediction.
                if include_mmd:
                    q_samples = model.sample_density()
                    mmd = mmd_loss(q_samples, batch)
                else:
                    mmd = torch.tensor(0)

                log_prob, mi_loss = model.log_prob(batch, include_mi=True)
                loss = torch.mean((-log_prob), dim=0) + l * mmd
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                avg_loss += loss
                # avg_mmd += mmd.item()
                num_batches += 1

                od = OrderedDict()
                od["loss"] = loss.item()
                od['mi'] = mi_loss.item()
                epoch_pbar.set_postfix(od)

                # torch.mps.synchronize()
                # total = time.time() - start
                # print(f'Total loop time: {total}.6f s')
                # allocated = torch.mps.current_allocated_memory()
                # print(f"Allocated memory: {allocated / (1024 ** 2):.2f} MB")
                # torch.mps.empty_cache()
                if device == torch.device('cuda'):
                    torch.cuda.empty_cache()

            avg_loss /= num_batches
            avg_mmd /= num_batches
            print(f"Epoch {epoch + 1}, Loss: {avg_loss}, MMD: {avg_mmd}")

            # Validation
            # print(model.forward(torch.tensor([[0, 0, 0, 0, 0, 1, 0, 0]], device=device),
            #                     torch.tensor([[2, 1]], device=device)))
            model.eval()
            # print(model.forward(torch.tensor([[0, 0, 0, 0, 0, 1, 0, 0]], device=device),
            #                     torch.tensor([[2, 1]], device=device)))

            with torch.no_grad():
                val_loss = 0
                num_batches = 0
                for (batch_idx, batch) in enumerate(val_loader):
                    # print(batch.data)
                    # print(model.forward(torch.tensor([[0, 0, 0, 0, 0, 1, 0, 0]], device=device), torch.tensor([[2, 1]], device=device)))
                    # print(model.predict_logical(torch.tensor([[0, 0, 0, 0, 0, 1, 0, 0]], device=device)))
                    log_prob = model.log_prob(batch)
                    # output = model(batch)
                    loss = torch.mean((-log_prob), dim=0)
                    # loss = criterion(output, batch.float())
                    val_loss += loss.item()
                    num_batches += 1
                val_loss /= num_batches
                # if val_loss < best_loss:
                #     best_loss = val_loss
                print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")
                # exit(-1)

                if val_loss < best_loss:
                    model.save()
                    torch.save(log_prob, f'data/result_{model.name}.pt')
                    best_loss = val_loss

                # previous_loss = val_loss
                if activate_scheduler:
                    if epoch < warmup_epochs - 1:
                        warmup_scheduler.step()
                        print(warmup_scheduler._last_lr)
                    else:
                        decay_scheduler.step()
                        # decay_scheduler.step(val_loss)
                        print(decay_scheduler._last_lr)
                    # scheduler.step()
                    '''
                    if scheduler.get_last_lr()[0] < 2e-8:
                        counter += 1
                    if counter > 10:
                        break
                    '''
    return model


def online_training(model: nn.Module, dataset, init_optimizer: Callable[[Any], Optimizer], device, epochs=100,
                    batch_size=100, num_data=1000000, include_mi=False, from_checkpoint=False, checkpoint=None):
    scaler = None
    use_amp = True
    model.to(device)
    optimizer = init_optimizer((model.parameters()))
    if device == torch.device('cuda'):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    warmup_epochs = 5
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs)
    # decay_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
    # decay_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-9)
    start = 0

    if from_checkpoint:
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            # if device == torch.device('cuda'):
            #     scaler.load_state_dict(checkpoint["scaler"])
            # start = checkpoint["last_epoch"] + 1
            # decay_scheduler.load_state_dict(checkpoint["scheduler"])

    best_loss = float('inf')

    '''
    if load_pretrained:
        initial_requires_grad = {}  # Dictionary to store the original state
        for name, param in model.named_parameters():
            initial_requires_grad[name] = param.requires_grad  # Save the state
            param.requires_grad = False  # Freeze all parameters

        for name, param in model.positional_encoding.named_parameters():
            param.requires_grad = True
    '''

    # model.mem_token.register_hook(print_memory_grad)

    # register_hooks(model)

    # torch.autograd.set_detect_anomaly(True)
    for epoch in np.arange(start, epochs):
        avg_loss = 0
        avg_mi = torch.tensor(0.0, device=device)
        num_batches = 0

        # Generate data for current epoch:
        dataset = dataset.initialize(num_data)
        train_set, val_set = dataset.get_train_val_data()
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

        # Training
        model.train()
        # model.eval()
        with tqdm(train_loader, unit="batch") as epoch_pbar:
            for batch in epoch_pbar:
                epoch_pbar.set_description(f"Epoch {epoch}")
                # add start token?
                # torch.mps.synchronize()
                # start = time.time()
                optimizer.zero_grad()

                # if batch_idx == 1:
                #    print('Train data: ', batch.data)

                # Possibility to include MMD loss for training.
                # Slows down training. Sample size needs to be very high to achieve similar mean embedding.
                # Mainly useful if autoregressive transformer is used for both next stabilizer and logical prediction.

                if include_mi:
                    log_prob, mi_loss = model.log_prob(batch, include_mi)
                    log_prob_mean = torch.mean((-log_prob), dim=0)
                    if -mi_loss < 0.1:
                        loss = log_prob_mean  # + 10000 * mi_loss
                    else:
                        loss = log_prob_mean
                else:
                    log_prob = model.log_prob(batch, include_mi)
                    loss = torch.mean((-log_prob), dim=0)
                    log_prob_mean = loss
                    mi_loss = torch.tensor(0)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    loss.backward()
                    optimizer.step()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Log the gradient norms
                '''
                if len(gradients["encoder"]) > 0 and len(gradients["decoder"]) > 0:
                    avg_encoder_grad = sum(gradients["encoder"]) / len(gradients["encoder"])
                    avg_decoder_grad = sum(gradients["decoder"]) / len(gradients["decoder"])

                    print(
                        f"Epoch {epoch}, Encoder Grad Norm: {avg_encoder_grad:.4f}, Decoder Grad Norm: {avg_decoder_grad:.4f}")
                '''
                # gradients["encoder"].clear()  # Reset after each step
                # gradients["decoder"].clear()

                avg_loss += loss
                avg_mi += mi_loss.item()
                num_batches += 1

                od = OrderedDict()
                od["loss"] = log_prob_mean.item()
                od["mi"] = mi_loss.item()
                epoch_pbar.set_postfix(od)

                # torch.mps.synchronize()
                # total = time.time() - start
                # print(f'Total loop time: {total}.6f s')
                # allocated = torch.mps.current_allocated_memory()
                # print(f"Allocated memory: {allocated / (1024 ** 2):.2f} MB")
                # torch.mps.empty_cache()
                if device == torch.device('cuda'):
                    torch.cuda.empty_cache()

            avg_loss /= num_batches
            avg_mi /= num_batches
            print(f"Epoch {epoch + 1}, Loss: {avg_loss}, MI: {avg_mi}, BCE: {avg_loss - avg_mi}")

            # Validation
            # print(model.forward(torch.tensor([[0, 0, 0, 0, 0, 1, 0, 0]], device=device),
            #                     torch.tensor([[2, 1]], device=device)))
            model.eval()
            # print(model.forward(torch.tensor([[0, 0, 0, 0, 0, 1, 0, 0]], device=device),
            #                     torch.tensor([[2, 1]], device=device)))

            with torch.no_grad():
                val_loss = 0
                num_batches = 0
                for (batch_idx, batch) in enumerate(val_loader):
                    # print(batch.data)
                    # print(model.forward(torch.tensor([[0, 0, 0, 0, 0, 1, 0, 0]], device=device), torch.tensor([[2, 1]], device=device)))
                    # print(model.predict_logical(torch.tensor([[0, 0, 0, 0, 0, 1, 0, 0]], device=device)))

                    # with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    log_prob = model.log_prob(batch, include_mi=False)
                    # output = model(batch)
                    loss = torch.mean((-log_prob), dim=0)
                    # loss = criterion(output, batch.float())
                    val_loss += loss.item()
                    num_batches += 1
                val_loss /= num_batches
                # if val_loss < best_loss:
                #     best_loss = val_loss
                print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")
                # exit(-1)

                if val_loss < best_loss:
                    if from_checkpoint:
                        checkpoint = {"model": model.state_dict(),
                                      "optimizer": optimizer.state_dict(),
                                      "scheduler": decay_scheduler.state_dict(),
                                      "last_epoch": epoch + 1}
                        if device == torch.device('cuda'):
                            checkpoint["scaler"] = scaler.state_dict()
                        torch.save(checkpoint, f'data/checkpoint_{model.name}')
                    else:
                        model.save()
                    best_loss = val_loss
                    torch.save(val_loss, f'data/val_loss_{model.name}')

                # previous_loss = val_loss

                if epoch < warmup_epochs - 1:
                    warmup_scheduler.step()
                    print(warmup_scheduler._last_lr)
                else:
                    # decay_scheduler.step(val_loss)
                    decay_scheduler.step()
                    print(decay_scheduler._last_lr)
                # scheduler.step()
            if torch.isnan(avg_loss):
                raise ValueError("loss is NaN")
    return model

def eval_log_op(model, distance, noise, device, num=10000, batch_size=100, mode='depolarizing', measurement_input=True):
    """
    Evaluates the model by predicting logical operators.
    """
    model.eval()
    model.to(device)

    logical = 'maximal' if model.readout == 'transformer-decoder' else 'string'

    px, py, pz, p1 = None, None, None, None
    if mode == 'depolarizing':
        data = (DepolarizingSurfaceData(distance=distance,
                                        noise=noise,
                                        name='eval',
                                        load=False,
                                        device=device,
                                        only_syndromes=True,
                                        logical=logical)
                .initialize(num))
        # print(data.get_syndromes().size())
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

        if model.readout == 'transformer-decoder':
            data_1 = data.get_syndromes()

            if not measurement_input:
                data_1 = data_1[:, :, 0]
                data_2 = torch.cat((data_1, torch.full((data_1.size(0), 1), 0, dtype=torch.long, device=device)), dim=1)
                data_3 = torch.cat((data_1, torch.full((data_1.size(0), 1), 1, dtype=torch.long, device=device)), dim=1)
            else:
                data_2 = torch.cat(
                    (data_1, torch.full((data_1.size(0), 1, data_1.size(2)), 0, dtype=torch.long, device=device)), dim=1)
                data_3 = torch.cat((data_1,
                                    torch.cat((torch.full((data_1.size(0), 1, 2), 1, dtype=torch.long, device=device),
                                              torch.full((data_1.size(0), 1, 1), 0, dtype=torch.long,
                                                         device=device)), dim=2))
                                   , dim=1)

            data_loader_1 = DataLoader(data_1, batch_size=batch_size, shuffle=False)
            data_loader_2 = DataLoader(data_2, batch_size=batch_size, shuffle=False)
            data_loader_3 = DataLoader(data_3, batch_size=batch_size, shuffle=False)

            pb1 = torch.empty(num, device=device)
            pb2_0 = torch.empty(num, device=device)
            pb2_1 = torch.empty(num, device=device)

            for batch_idx, batch in enumerate(data_loader_1):
                pb1[batch_idx * batch_size: (batch_idx + 1) * batch_size] = model.predict_logical(batch)
            for batch_idx, batch in enumerate(data_loader_2):
                pb2_0[batch_idx * batch_size: (batch_idx + 1) * batch_size] = model.predict_logical(batch)
            for batch_idx, batch in enumerate(data_loader_3):
                pb2_1[batch_idx * batch_size: (batch_idx + 1) * batch_size] = model.predict_logical(batch)

            px = pb1 * (1 - pb2_1)
            pz = (1 - pb1) * pb2_0
            py = pb2_1 * pb1
            p1 = (1 - pb1) * (1 - pb2_0)
            """
            data_loader_1 = DataLoader(data_1, batch_size=batch_size, shuffle=False)
            pb = torch.empty(num, 4, device=device)
            for batch_idx, batch in enumerate(data_loader_1):
                pb[batch_idx * batch_size: (batch_idx + 1) * batch_size] = model.predict_logical(batch)

            px = pb[:, 2]
            py = pb[:, 3]
            pz = pb[:, 1]
            p1 = pb[:, 0]
            """
        if model.readout == 'conv':
            py = torch.empty(num, device=device)
            px = torch.empty(num, device=device)
            pz = torch.empty(num, device=device)
            p1 = torch.empty(num, device=device)
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                plog = model.predict_logical(batch)
                px_list = plog[:, 0]
                pz_list = plog[:, 1]

                # x_log_out = 0.5 * (
                #             torch.ones(px_list.size(0), device=px_list.device) - torch.prod(1 - 2 * px_list, dim=1))
                # z_log_out = 0.5 * (
                #             torch.ones(pz_list.size(0), device=pz_list.device) - torch.prod(1 - 2 * pz_list, dim=1))

                x_log_out = px_list
                z_log_out = pz_list
                py[batch_idx * batch_size: (batch_idx + 1) * batch_size] = x_log_out * z_log_out
                px[batch_idx * batch_size: (batch_idx + 1) * batch_size] = x_log_out * (1 - z_log_out)
                pz[batch_idx * batch_size: (batch_idx + 1) * batch_size] = (1 - x_log_out) * z_log_out
                p1[batch_idx * batch_size: (batch_idx + 1) * batch_size] = (1 - x_log_out) * (1 - z_log_out)

                # For conv readout with cross-entropy loss
                # py[batch_idx * batch_size: (batch_idx + 1) * batch_size] = plog[:, 3]
                # px[batch_idx * batch_size: (batch_idx + 1) * batch_size] = plog[:, 2]
                # pz[batch_idx * batch_size: (batch_idx + 1) * batch_size] = plog[:, 1]
                # p1[batch_idx * batch_size: (batch_idx + 1) * batch_size] = plog[:, 0]
            # assert (torch.abs(torch.sum(torch.cat((p1, px, pz, py), dim=1), dim=1) - torch.full((px.size(1), 1), 1, device=device)) < 1e-3).all()

        assert (px is not None) and (py is not None) and (pz is not None) and (p1 is not None)

        result = np.zeros(num)
        for i in range(num):
            result[i] += px[i] * torch.log(px[i])
            result[i] += py[i] * torch.log(py[i])
            result[i] += pz[i] * torch.log(pz[i])
            result[i] += p1[i] * torch.log(p1[i])
            # result[i] += px[i] ** 2
            # result[i] += py[i] ** 2
            # result[i] += pz[i] ** 2
            # result[i] += p1[i] ** 2
    else:
        data = (BitflipSurfaceData(distance=distance,
                                   noise=noise,
                                   name='eval',
                                   load=False,
                                   device=device,
                                   only_syndromes=True)
                .initialize(num))
        # print(data.get_syndromes().size())
        data_1 = data.get_syndromes()

        raise NotImplementedError('Bit-flip readout not yet implemented.')

        pb1 = model.predict_logical(data_1)

        px = pb1
        p1 = (1 - pb1)

        result = np.zeros(num)
        for i in range(num):
            result[i] += px[i] ** 2
            result[i] += p1[i] ** 2

    return (np.log(2) + result) / np.log(2)


def eval_log_error_rate(model, distance, noise, device, num=100000, batch_size=100, mode='depolarizing', threshold=0.5, increased_num=False):
    model.eval()
    model.to(device)
    if increased_num:
        num = 10 * num
    model.to(torch.float32)
    if mode == 'depolarizing':
        data = (DepolarizingSurfaceData(distance=distance,
                                        noise=noise,
                                        name='eval',
                                        load=False,
                                        device=device,
                                        only_syndromes=False)
                .initialize(num))
    else:
        raise ValueError(f"Mode {mode} not supported.")
    data = data[:, :, 0]
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    log = data[:, -2:]
    pred = torch.zeros_like(log)
    c = threshold
    for i in np.arange(log.size(1)):
        for batch_idx, batch in enumerate(data_loader):
            current_pred = model.predict_logical(batch[:, :-(2 - i)])
            current_pred[(c <= current_pred) & (current_pred <= 1 - c)] = -1
            pred[batch_idx * batch_size: (batch_idx + 1) * batch_size, i] = torch.where(current_pred == -1, current_pred, (current_pred >= 0.5).float())
            # pred[batch_idx * batch_size: (batch_idx + 1) * batch_size, i] = torch.bernoulli(current_pred)
    num_errors = 0
    discarded = 0
    print('Forward pass done.')
    for shot in range(num):
        actual_for_shot = log[shot]
        predicted_for_shot = pred[shot]
        if torch.any(predicted_for_shot == -1):
            discarded += 1
            continue
        if not torch.equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    rate = num_errors / (num - discarded)
    print('Discarded: ', discarded/num)
    print(rate)
    return rate, (rate * (1 - rate) / num) ** 0.5, (rate * (1 - rate) / num) ** 0.5, discarded/num  # from Luis, ask why?


def estimate_uncertainty_val_loss(iteration, dist, noise, model, data, device, batch_size=1000, num=500000):
    print('Distance ', dist, ', Noise ', noise)
    model.to(device)
    model.eval()
    checkpoint = torch.load(f'data/checkpoint_{iteration}_{dist}_{noise}', map_location=device)
    model.load_state_dict(checkpoint["model"])

    data.initialize(num)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    loss = torch.zeros(len(data))

    with torch.no_grad():
        num_batches = 0
        for (batch_idx, batch) in enumerate(tqdm(data_loader)):
            log_prob = model.log_prob(batch, include_mi=False)
            # output = model(batch)
            loss[batch_idx * batch_size:(batch_idx + 1) * batch_size] = -log_prob
            num_batches += 1

        mean, std_up, std_down = simple_bootstrap(loss.cpu().detach().numpy())
        print(mean, std_up, std_down)
    return std_up, std_down
