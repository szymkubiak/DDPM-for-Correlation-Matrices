import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F


def select_empirical_dataset(start_date, end_date, emp_corr_matrices_list, emp_dates_list, emp_context_data_list=None):
    sel_corr_matrices_list = []
    sel_dates_list = []
    sel_context_list = []
    for d_num, d in enumerate(emp_dates_list):
        if pd.to_datetime(d) >= pd.to_datetime(start_date) and pd.to_datetime(d) <= pd.to_datetime(end_date):
            sel_corr_matrices_list.append(emp_corr_matrices_list[d_num])
            sel_dates_list.append(d)
            if emp_context_data_list is not None:
                sel_context_list.append(emp_context_data_list[d_num])
    return sel_corr_matrices_list, sel_dates_list, sel_context_list

def load_dataset(batch_size, emp_corr_matrices_list, context_list):
    emp_corr_matrices_array = np.array(emp_corr_matrices_list)
    corr_matrix_size = emp_corr_matrices_array.shape[2]
    num_of_emp_corr_matrices = emp_corr_matrices_array.shape[0]
    emp_corr_matrices_array = emp_corr_matrices_array.reshape(num_of_emp_corr_matrices, 1, corr_matrix_size, corr_matrix_size)
    emp_corr_matrices_tensor = torch.from_numpy(emp_corr_matrices_array).float()

    if len(context_list) == 0:
        pytorch_dataset = torch.utils.data.TensorDataset(emp_corr_matrices_tensor)
        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        context_array = np.array(context_list)
        context_array = context_array.reshape(num_of_emp_corr_matrices, 2)
        context_tensor = torch.from_numpy(context_array).float()
        pytorch_dataset = torch.utils.data.TensorDataset(emp_corr_matrices_tensor, context_tensor)
        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def diffusion_process(timesteps):
    # define beta schedule
    betas = linear_beta_schedule(timesteps=timesteps)

    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    return betas, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance