import os
import torch
import numpy as np
import pickle
import datetime
from utility import select_empirical_dataset, load_dataset, extract, diffusion_process
from unet import Unet
from configs import configs
from tqdm import tqdm

@torch.no_grad()
def sample(model, matrix_size, batch_size=64, context=None):
    return p_sample_loop(model, shape=(batch_size, 1, matrix_size, matrix_size), context=context)

@torch.no_grad()
def p_sample(model, x, t, t_index, context):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, context) / sqrt_one_minus_alphas_cumprod_t
    )
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape, context):
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape, device=device)
    imgs = []
    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, context)
        imgs.append(img.cpu().numpy())
    return imgs

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
        allow_context=conditional
    )
    model.to(device)

    model_path = 'Models//'+ model_name
    betas, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance = diffusion_process(timesteps)

    states = torch.load(os.path.join(model_path, "ckpt.pth"))
    model.load_state_dict(states[0])
    model.eval()

    if not conditional:
        all_samples = []
        for b_num in range(10):
            sampled_images = sample(model, matrix_size=matrix_size, batch_size=batch_size)
            for gen_sample in sampled_images[-1]:
                all_samples.append(gen_sample[0])
            open_file = open('samples_DM_' + model_name + '.pickle', "wb")
            pickle.dump(all_samples, open_file)
            open_file.close()

    else:

        #Generate CDM_Train and CDM_Test data
        for data_period in ['Train', 'Test']:
            if data_period == 'Train':
                sel_context_list = train_context_list
            if data_period == 'Test':
                start_date = train_end_dt
                end_date = max(emp_dates_list)
                test_corr_matrices_list, test_dates_list, test_context_list = select_empirical_dataset(start_date, end_date,
                                                                                                        emp_corr_matrices_list,
                                                                                                        emp_dates_list,
                                                                                                        emp_context_data_list)
                sel_context_list = test_context_list
            batches_num = int(np.ceil(len(sel_context_list) / batch_size))
            all_sampled_data = []
            for i in range(0, batches_num):
                context_data = np.zeros((batch_size,2))
                sel_data = sel_context_list[batch_size * i: batch_size * (i + 1)]
                context_data[:len(sel_data), :] = np.array(sel_data).reshape(-1, 2)
                context_data = context_data.reshape(64, 2)
                context_tensor = torch.tensor(context_data, dtype=torch.float32)
                context_tensor = context_tensor.to(device)
                sampled_data = sample(model, matrix_size=matrix_size, batch_size=batch_size, context=context_tensor)
                for gen_sample in sampled_data[-1]:
                    all_sampled_data.append(gen_sample[0])
            all_sampled_data = all_sampled_data[:len(sel_context_list)]
            open_file = open('samples_CDM_' + data_period + '_' + model_name + '.pickle', "wb")
            pickle.dump(all_sampled_data, open_file)
            open_file.close()
