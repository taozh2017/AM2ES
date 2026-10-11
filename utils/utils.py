import math
import collections
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement."""
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]


def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]


def collate_MIL_survival(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    omic = torch.cat([item[1] for item in batch], dim=0).type(torch.FloatTensor)
    label = torch.LongTensor([item[2] for item in batch])
    event_time = np.array([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])
    return [img, omic, label, event_time, c]


def collate_MIL_survival_cluster(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    cluster_ids = torch.cat([item[1] for item in batch], dim=0).type(torch.LongTensor)
    omic = torch.cat([item[2] for item in batch], dim=0).type(torch.FloatTensor)
    label = torch.LongTensor([item[3] for item in batch])
    event_time = np.array([item[4] for item in batch])
    c = torch.FloatTensor([item[5] for item in batch])
    return [img, cluster_ids, omic, label, event_time, c]


def collate_MIL_survival_sig(batch):
    path_features_5x = torch.cat([item[0] for item in batch], dim=0)
    path_features_10x = torch.cat([item[1] for item in batch], dim=0)
    path_features_20x = torch.cat([item[2] for item in batch], dim=0)

    omic1 = torch.cat([item[3] for item in batch], dim=0).type(torch.FloatTensor)
    omic2 = torch.cat([item[4] for item in batch], dim=0).type(torch.FloatTensor)
    omic3 = torch.cat([item[5] for item in batch], dim=0).type(torch.FloatTensor)
    omic4 = torch.cat([item[6] for item in batch], dim=0).type(torch.FloatTensor)
    omic5 = torch.cat([item[7] for item in batch], dim=0).type(torch.FloatTensor)
    omic6 = torch.cat([item[8] for item in batch], dim=0).type(torch.FloatTensor)

    label = torch.LongTensor([item[9] for item in batch])
    event_time = np.array([item[10] for item in batch])
    c = torch.FloatTensor([item[11] for item in batch])
    
    return [path_features_5x, path_features_10x, path_features_20x, 
            omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c]


def get_simple_loader(dataset, batch_size=1):
    kwargs = {'num_workers': 0} if device.type == "cuda" else {}
    loader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(dataset), collate_fn=collate_MIL, **kwargs)
    return loader 


def get_split_loader(split_dataset, training=False, testing=False, weighted=False, mode='coattn', batch_size=1):
    """
    Returns either the validation loader or training loader 
    """
    if mode == 'coattn':
        collate = collate_MIL_survival_sig
    elif mode == 'cluster':
        collate = collate_MIL_survival_cluster
    else:
        collate = collate_MIL_survival
    
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}

    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler=WeightedRandomSampler(weights, len(weights)), collate_fn=collate, **kwargs)    
            else:
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler=RandomSampler(split_dataset), collate_fn=collate, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler=SequentialSampler(split_dataset), collate_fn=collate, **kwargs)
    else:
        ids = np.random.choice(np.arange(int(len(split_dataset)*0.1), len(split_dataset)), replace=False)
        loader = DataLoader(split_dataset, batch_size=1, sampler=SubsetSequentialSampler(ids), collate_fn=collate, **kwargs)

    return loader


def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print(f'Total number of parameters: {num_params}')
    print(f'Total number of trainable parameters: {num_params_train}')


def generate_split(cls_ids, val_num, test_num, samples, n_splits=5, seed=7, label_frac=1.0, custom_test_ids=None):
    indices = np.arange(samples).astype(int)
    
    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []
        
        if custom_test_ids is not None:
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices)
            remaining_ids = possible_indices

            if val_num[c] > 0:
                val_ids = np.random.choice(possible_indices, val_num[c], replace=False)
                remaining_ids = np.setdiff1d(possible_indices, val_ids)
                all_val_ids.extend(val_ids)

            if custom_test_ids is None and test_num[c] > 0:
                test_ids = np.random.choice(remaining_ids, test_num[c], replace=False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)
            else:
                sample_num = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sorted(sampled_train_ids), sorted(all_val_ids), sorted(all_test_ids)


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)


def calculate_error(Y_hat, Y):
    return 1. - Y_hat.float().eq(Y.float()).float().mean().item()


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))                                           
    weight_per_class = [N / len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  

    return torch.DoubleTensor(weight)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def dfs_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) 
    c = c.view(batch_size, 1).float() 
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) 
        
    S_padded = torch.cat([torch.ones_like(c), S], 1) 
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    return loss.mean()


def nll_loss_with_margin_and_ranking(hazards, S, Y, c,
                                     alpha=0.4, eps=1e-7,
                                     m_hazard=0.2, w_margin=1.0,
                                     m_rank=0.1, w_rank=1.0,
                                     min_mean_hazard=0.01, w_anti=0.1):
    """
    改进后的 NLL loss，带 aggressive margin 和 death-only pairwise ranking：
      hazards: [B, T] 每个 bin 的 hazard 概率
      S: [B, T] 或 None 
      Y: [B, 1] ground truth bin index (0..T-1)
      c: [B, 1] censorship (1 for censored, 0 for event)
    """
    batch_size = Y.size(0)
    Y = Y.view(batch_size, 1).long()
    c = c.view(batch_size, 1).float()

    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)

    S_padded = torch.cat([torch.ones_like(c), S], dim=1) 

    # base NLL terms
    hazard_at_y = torch.gather(hazards, 1, Y).clamp(min=eps)          
    S_at_y = torch.gather(S_padded, 1, Y).clamp(min=eps)              
    S_at_yp1 = torch.gather(S_padded, 1, Y+1).clamp(min=eps)         

    uncensored_loss = -(1 - c) * (torch.log(S_at_y) + torch.log(hazard_at_y))
    censored_loss = - c * torch.log(S_at_yp1)
    neg_l = censored_loss + uncensored_loss
    base_nll = ((1.0 - alpha) * neg_l + alpha * uncensored_loss).mean()

    # aggressive margin on hazards
    uncensored_mask = (c.squeeze(1) == 0)
    margin_loss = torch.tensor(0.0, device=hazards.device)
    if uncensored_mask.sum() > 0:
        haz_true = hazard_at_y.squeeze(1)[uncensored_mask]
        margin_loss = F.relu(m_hazard - haz_true).mean()

    # death-only pairwise ranking
    with torch.no_grad():
        Y_vec = Y.squeeze(1) 
    score = -S.sum(dim=1)
    rank_loss = torch.tensor(0.0, device=hazards.device)

    unc_idx = torch.nonzero(uncensored_mask, as_tuple=False).squeeze(1)
    if unc_idx.numel() > 1:
        Y_u = Y_vec[unc_idx]              
        score_u = score[unc_idx]          
        
        Yi = Y_u.unsqueeze(1)             
        Yj = Y_u.unsqueeze(0)             
        earlier_mask = (Yi < Yj)          
        
        if earlier_mask.sum() > 0:
            si = score_u.unsqueeze(1)    
            sj = score_u.unsqueeze(0)    
            margin_mat = (m_rank - (si - sj)) * earlier_mask.float()
            rank_loss = F.relu(margin_mat).sum() / (earlier_mask.float().sum() + 1e-12)

    # anti-collapse
    anti_collapse = F.relu(min_mean_hazard - hazards.mean()) 

    return base_nll + w_margin * margin_loss + w_rank * rank_loss + w_anti * anti_collapse


def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    c = c.view(batch_size, 1).float()
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
        
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y)+eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg
    return loss.mean()


class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None): 
        alpha_val = alpha if alpha is not None else self.alpha
        return ce_loss(hazards, S, Y, c, alpha=alpha_val)


class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        alpha_val = alpha if alpha is not None else self.alpha
        return nll_loss(hazards, S, Y, c, alpha=alpha_val)
        

class CoxSurvLoss(object):
    def __call__(self, hazards, S, c, **kwargs):
        current_batch_len = len(S)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i, j] = S[j] >= S[i]

        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * (1-c))
        return loss_cox


def l1_reg_all(model, reg_type=None):
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()
    return l1_reg


def l1_reg_modules(model, reg_type=None):
    l1_reg = 0
    l1_reg += l1_reg_all(model.fc_omic)
    l1_reg += l1_reg_all(model.mm)
    return l1_reg


def get_custom_exp_code(args):
    # Dataset Base Name
    exp_code = '_'.join(args.split_dir.split('_')[:2])
    dataset_path = 'dataset_csv'
    param_code = ''

    # Model Encoding
    model_mapping = {
        'max_net': 'SNN',
        'amil': 'AMIL',
        'deepset': 'DS',
        'mi_fcn': 'MIFCN',
        'mcat': 'MCAT',
        'motcat': 'MOTCAT',
        'anchor': 'ANCHOR'
    }
    if args.model_type in model_mapping:
        param_code += model_mapping[args.model_type]
    else:
        raise NotImplementedError

    # Loss Encoding
    param_code += f'_{args.bag_loss}'
    param_code += f'_a{args.alpha_surv}'

    # Hyperparameter Encoding
    if args.lr != 2e-4:
        param_code += f'_lr{format(args.lr, ".0e")}'

    if args.reg_type != 'None':
        param_code += f'_reg{format(args.lambda_reg, ".0e")}'

    param_code += f'_{args.which_splits.split("_")[0]}'

    if args.batch_size != 1:
        param_code += f'_b{args.batch_size}'

    if args.gc != 1:
        param_code += f'_gc{args.gc}'

    if args.apply_sigfeats:
        param_code += '_sig'
        dataset_path += '_sig'

    if args.fusion != "None":
        param_code += f'_{args.fusion}'

    args.exp_code = f"{exp_code}_{param_code}"
    args.param_code = param_code
    args.dataset_path = dataset_path

    return args