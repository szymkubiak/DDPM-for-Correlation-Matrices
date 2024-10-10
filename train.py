import os
import torch
import numpy as np
import pickle
import datetime
from torch.optim import Adam
from utility import select_empirical_dataset, load_dataset, extract, diffusion_process
from unet import Unet
from configs import configs
import torch.nn.functional as F

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def loss_func(denoise_model, x_start, t, noise=None, context=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, context=context)
    loss = F.mse_loss(noise, predicted_noise)
    return loss

model_names = ['futures', 'futures_cond', 'fixed income', 'fixed income_cond', 'stocks', 'stocks_cond']
for model_name in model_names:
    config = configs[model_name]
    dataset_name = config['dataset_name']
    batch_size = config['batch_size']
    dim_mults = config['dim_mults']
    init_dim = config['init_dim']
    train_end_dt = config['train_end_dt']
    init_conv = config['init_conv']
    cut_output_dim_h = config['cut_output_dim_h']
    cut_output_dim_v = config['cut_output_dim_v']
    conditional = config['conditional']
    timesteps = 1000

    data_file = open('Data/' + dataset_name + ' data.pickle', 'rb')
    if dataset_name in ['futures', 'stocks']:
        emp_corr_matrices_list, emp_dates_list, emp_context_data_list = pickle.load(data_file)
    elif dataset_name in ['fixed income', 'fixed income_cond']:
        emp_corr_matrices_list, _, _, _, emp_dates_list, emp_context_data_list = pickle.load(data_file)
    if not conditional:
        emp_context_data_list = None

    start_date = datetime.date(1990, 1, 1)
    end_date = train_end_dt
    train_corr_matrices_list, train_dates_list, train_context_list = select_empirical_dataset(start_date, end_date, emp_corr_matrices_list, emp_dates_list, emp_context_data_list)
    dataloader = load_dataset(batch_size, train_corr_matrices_list, train_context_list)
    matrix_size = train_corr_matrices_list[0].shape[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Unet(
        dim=matrix_size,
        init_dim=init_dim,
        dim_mults=dim_mults,
        init_conv=init_conv,
        cut_output_dim_h=cut_output_dim_h,
        cut_output_dim_v=cut_output_dim_v,
        allow_context=conditional,
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)
    epochs = 2501
    model_path = 'Models//'+ model_name
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    betas, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance = diffusion_process(timesteps)

    for epoch in range(epochs):
        optimizer.param_groups[0]['lr'] = 0.001 * 0.77 ** (epoch // 100)
        for step, data in enumerate(dataloader):
            optimizer.zero_grad()
            batch = data[0].to(device)
            if conditional:
                context = data[1].to(device) * 100
            else:
                context = None
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            loss = loss_func(model, batch, t, context=context)
            loss.backward()
            optimizer.step()
        print(['Epoch:', epoch, "Loss:", loss.item()])
        if epoch % 500 == 0:
            states = [
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                step,
            ]
            torch.save(states, os.path.join(model_path, "ckpt.pth"))

