#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import random
import math
import torch
from torch import nn
import numpy as np
from tqdm import tqdm

def _param_dot_product(param1, param2):
    product = 0
    for k in param1.keys():
        product += torch.sum(param1[k] * param2[k])
    return product

def _param_distance(param1, param2):
    dist = 0
    for k in param1.keys():
        dist += (param1[k] - param2[k]).double().norm(2).item()
    return math.sqrt(dist)


def FedAvg(w, idx_users, args, client_losses, curr_sel_clients, weights, similarity):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w)).to(args.device)

    if args.sel_scheme == 'ideal':
        sel_clients = []
        for client in range(args.sel_clients):
            change = -100 * torch.ones(args.num_users).to(args.device)
            delta_curr = {k: torch.zeros(v.shape, dtype=v.dtype).to(args.device) for k, v in w[0].items()}
            diff = {k: torch.zeros(v.shape, dtype=v.dtype).to(args.device) for k, v in w[0].items()}
            # current aggregated gradient
            for idx in range(len(idx_users)):
                for k in delta_curr.keys():
                    delta_curr[k] = delta_curr[k] + (w[idx][k] * weights[idx_users[idx]]).type(delta_curr[k].dtype)
            # current difference from global gradient
            for k in diff.keys():
                diff[k] = (client + 1) * w_avg[k] - client * delta_curr[k]
            # update change
            for idx in range(len(idx_users)):
                client_id = idx_users[idx]
                change[client_id] = 0 if client_id in sel_clients else _param_dot_product(w[idx], diff) + args.lambda_loss * client_losses[idx]
            sel_client = torch.argmax(change).item()
            sel_clients.append(sel_client)
            weights = [1/(client + 1) if i in sel_clients else 0 for i in range(args.num_users)]
            for iter in range(args.iteration):
                delta_curr = {k: torch.zeros(v.shape, dtype=v.dtype).to(args.device) for k, v in w[0].items()}
                diff = {k: torch.zeros(v.shape, dtype=v.dtype).to(args.device) for k, v in w[0].items()}
                # current aggregated gradient
                for idx in range(len(idx_users)):
                    for k in delta_curr.keys():
                        delta_curr[k] = delta_curr[k] + (w[idx][k] * weights[idx_users[idx]]).type(delta_curr[k].dtype)
                # current difference from global gradient
                for k in diff.keys():
                    diff[k] = w_avg[k] - delta_curr[k]
                # update weight
                for client_id in sel_clients:
                    idx = np.where(idx_users == client_id)[0][0]
                    weights[client_id] += _param_dot_product(w[idx], diff) + args.lambda_loss * client_losses[idx]

        w_avg = {k: torch.zeros(v.shape, dtype=v.dtype).to(args.device) for k, v in w[0].items()}
        for idx in range(len(idx_users)):
                for k in w_avg.keys():
                    w_avg[k] = w_avg[k] + (w[idx][k] * weights[idx_users[idx]]).type(w_avg[k].dtype)

    elif args.sel_scheme == 'practical':
        sel_clients = []
        param_product = torch.zeros(args.num_users, args.num_users).to(args.device)
        norm_num = 0
        norm_avg = 0
        # update parameter products
        for i in curr_sel_clients:
            idx = np.where(idx_users == i)[0][0]
            for j in curr_sel_clients:
                jdx = np.where(idx_users == j)[0][0]
                param_product[i][j] = _param_dot_product(w[idx], w[jdx])
                if i == j:
                    norm_num += 1
                    norm_avg += param_product[i][j]
        norm_avg /= norm_num
        # update similarity matrix
        for i in curr_sel_clients:
            for j in curr_sel_clients:
                if i != j:
                    similarity[i][j] = param_product[i][j] / (math.sqrt(param_product[i][i]) * math.sqrt(param_product[j][j]))

        weights = [1/(args.sel_clients + 1) if i in curr_sel_clients else 0 for i in range(args.num_users)]
        for iter in range(args.iteration):
            # update weight
            for client_id in curr_sel_clients:
                for i in idx_users:
                    if client_id in curr_sel_clients and i in curr_sel_clients:
                        weights[client_id] += (1 / len(idx_users) - weights[i]) * param_product[client_id][i]
                    else:
                        similar = random.uniform(-0.5, 0.8) if similarity[client_id][i] == 0 else similarity[client_id][i]
                        if client_id in curr_sel_clients:
                            weights[client_id] += (1 / len(idx_users) - weights[i]) * 0.5 * (param_product[client_id][client_id] + norm_avg.to(args.device)) * similar
                        elif i in curr_sel_clients:
                            weights[client_id] += (1 / len(idx_users) - weights[i]) * 0.5 * (param_product[i][i] + norm_avg.to(args.device)) * similar
                        else:
                            weights[client_id] += (1 / len(idx_users) - weights[i]) * norm_avg.to(args.device) * similar
                idx = np.where(idx_users == client_id)[0][0]
                weights[client_id] += args.lambda_loss * client_losses[idx]

        # update w_avg by updated weights
        w_avg = {k: torch.zeros(v.shape, dtype=v.dtype).to(args.device) for k, v in w[0].items()}
        for client_id in curr_sel_clients:
                idx = np.where(idx_users == client_id)[0][0]
                for k in w_avg.keys():
                    w_avg[k] = w_avg[k] + (w[idx][k] * weights[client_id]).type(w_avg[k].dtype)

        for client in range(args.sel_clients):
            change = torch.zeros(args.num_users).to(args.device)
            # update change
            for client_id in idx_users:
                change[client_id] = 0
                if client_id not in sel_clients:
                    for i in idx_users:
                        if client_id in curr_sel_clients and i in curr_sel_clients:
                            change[client_id] += ((client + 1) / len(idx_users) - client * weights[i]) * param_product[client_id][i]
                        else:
                            similar = random.uniform(-0.5, 0.8) if similarity[client_id][i] == 0 else similarity[client_id][i]
                            if client_id in curr_sel_clients:
                                change[client_id] += ((client + 1) / len(idx_users) - client * weights[i]) * 0.5 * (param_product[client_id][client_id] + norm_avg.to(args.device)) * similar
                            elif i in curr_sel_clients:
                                change[client_id] += ((client + 1) / len(idx_users) - client * weights[i]) * 0.5 * (param_product[i][i] + norm_avg.to(args.device)) * similar
                            else:
                                change[client_id] += ((client + 1) / len(idx_users) - client * weights[i]) * norm_avg.to(args.device) * similar
                    idx = np.where(idx_users == client_id)[0][0]
                    change[client_id] += args.lambda_loss * client_losses[idx]
            sel_clients.append(torch.argmax(change).item())
            weights = [1/(client + 1) if i in sel_clients else 0 for i in range(args.num_users)]

        curr_sel_clients.clear()
        for i in range(len(sel_clients)):
            curr_sel_clients.append(sel_clients[i])
                    
    elif args.sel_scheme == 'divfl':
        sel_clients = []
        list_clients = list(range(args.num_users))
        dist_matrix = torch.zeros(args.num_users, args.num_users).to(args.device)
        for i in range(args.num_users):
            for j in range(args.num_users):
                dist_matrix[i][j] = _param_distance(w[i], w[j])
        for iter in range(args.sel_clients):
            sel_client = 0
            min_score = 1000
            for kk in list_clients:
                G_ks = 0
                for k in range(args.num_users):
                    min_dist = 100
                    for i in sel_clients:
                        dist = dist_matrix[k][i]
                        min_dist = dist if dist < min_dist else min_dist
                    dist = dist_matrix[k][kk]
                    min_dist = dist if dist < min_dist else min_dist
                    G_ks += min_dist
                if G_ks < min_score:
                    min_score = G_ks
                    sel_client = kk
            sel_clients.append(sel_client)
            list_clients.remove(sel_client)
        # update w_avg by updated weights
        old = copy.deepcopy(w_avg)
        w_avg = {k: torch.zeros(v.shape, dtype=v.dtype).to(args.device) for k, v in w[0].items()}
        weights = (1. + torch.zeros(args.num_users).to(args.device)) / args.sel_clients
        for client_id in sel_clients:
                for k in w_avg.keys():
                    w_avg[k] = w_avg[k] + (w[client_id][k] * weights[client_id]).type(w_avg[k].dtype)

    elif args.sel_scheme == 'power-of-choice':
        client_losses = np.array(client_losses)
        topk = np.argpartition(client_losses, -args.sel_clients)[-args.sel_clients:]
        weights = (1. + torch.zeros(args.num_users).to(args.device)) / args.sel_clients
        w_avg = {k: torch.zeros(v.shape, dtype=v.dtype).to(args.device) for k, v in w[0].items()}
        for client_id in topk:
            for k in w_avg.keys():
                w_avg[k] = w_avg[k] + (w[client_id][k] * weights[client_id]).type(w_avg[k].dtype)

    elif args.sel_scheme == 'random':
        topk = np.random.permutation(args.num_users)[-args.sel_clients:]
        weights = (1. + torch.zeros(args.num_users).to(args.device)) / args.sel_clients
        old = copy.deepcopy(w_avg)
        w_avg = {k: torch.zeros(v.shape, dtype=v.dtype).to(args.device) for k, v in w[0].items()}
        for client_id in topk:
            for k in w_avg.keys():
                w_avg[k] = w_avg[k] + (w[client_id][k] * weights[client_id]).type(w_avg[k].dtype)
        print(_param_distance(old, w_avg))
    return w_avg
