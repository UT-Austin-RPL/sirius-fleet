import torch
import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

activation_map = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
}

loss_name_mapping = {
    "total_loss": "loss/Total Loss",
    "action_loss": "loss/Policy Loss",
    "dyn_loss": "loss/Dyn Loss",
    "recons_loss": "loss/Dyn_Recons",
    "reconstruction_loss": "loss/Dyn_Recons",
    "recon_embedding_loss": "loss/Dyn_Recons_Embedding",
    "kl_loss": "loss/Dyn_KL",
    "reward_loss": "loss/Reward Loss"
}

def log_stoch_inputs(log, info):
    mean, std = info["predictions"]["obs_embedding"].chunk(2, -1)

    min_std = 0.1
    max_std = 2.0
    std = max_std * torch.sigmoid(std) + min_std

    log_data_attributes(log, "embedding_mean", mean)
    log_data_attributes(log, "embedding_std", std)
    
    ####
    mean, std = info["predictions"]["pred_obs_embedding"].chunk(2, -1)

    min_std = 0.1
    max_std = 2.0
    std = max_std * torch.sigmoid(std) + min_std

    log_data_attributes(log, "pred_embedding_mean", mean)
    log_data_attributes(log, "pred_embedding_std", std)

def log_data_attributes(log, key, entry):
    log[key + "/max"] = entry.max().item()
    log[key + "/min"] = entry.min().item()
    log[key + "/mean"] = entry.mean().item()
    log[key + "/std"] = entry.std().item()
    
    
def kl_loss(pred, actual, zdistr, kl_balance):
    d = zdistr
    d_pred, d_actual = d(pred), d(actual)
    loss_kl_exact = D.kl.kl_divergence(d_actual, d_pred)  # (T,B,I)

    # Analytic KL loss, standard for VAE
    if kl_balance < 0:
        loss_kl = loss_kl_exact
    else:
        actual_to_pred = D.kl.kl_divergence(d_actual, d(TensorUtils.detach(pred)))
        pred_to_actual = D.kl.kl_divergence(d(TensorUtils.detach(actual)), d_pred)

        loss_kl = (1 - kl_balance) * actual_to_pred + kl_balance * pred_to_actual

    # average across batch
    return loss_kl.mean()

def zdistr(z):
    return diag_normal(z)

def diag_normal(z, min_std=0.1, max_std=2.0):
    mean, std = z.chunk(2, -1)
    std = max_std * torch.sigmoid(std) + min_std
    return D.independent.Independent(D.normal.Normal(mean, std), 1)

def half_batch(batch, idx):

    if idx == 0:
        start, end = 0, batch["actions"].shape[1] // 2
    else:
        start, end = batch["actions"].shape[1] // 2, batch["actions"].shape[1]

    batch_half = {}
    for key, value in batch.items():
        if value is None:
            batch_half[key] = value
            continue
        if isinstance(value, dict):
            batch_half[key] = {}
            for key2, value2 in value.items():
                batch_half[key][key2] = value2[:, start:end]
        else:
            if key == "classifier_weights":
                batch_half[key] = value[:]
            else:
                batch_half[key] = value[:, start:end]
            
    return batch_half

def select_batch(batch, start, end):

    selected = {}
    for key, value in batch.items():
        if value is None:
            selected[key] = value
            continue
        if isinstance(value, dict):
            selected[key] = {}
            for key2, value2 in value.items():
                selected[key][key2] = value2[:, start:end]
        else:
            selected[key] = value[:, start:end]    
            
    return selected

def create_reward_loss(rew):
    
    reward_loss = nn.MSELoss(reduction="none") # default
    
    if rew.rew_class == "binary" and rew.binary_loss == "bce":
        reward_loss = nn.BCEWithLogitsLoss(reduction="none")
    elif rew.rew_class == "three_class":
        reward_loss = nn.CrossEntropyLoss(reduction="none")

    return reward_loss

def confusion_matrix(y_true, y_pred, num_classes):
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        conf_matrix[t.long(), p.long()] += 1
    conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix_no_nan = torch.where(torch.isnan(conf_matrix_norm), torch.zeros_like(conf_matrix_norm), conf_matrix_norm)
    return torch.diag(conf_matrix_no_nan), conf_matrix_no_nan
