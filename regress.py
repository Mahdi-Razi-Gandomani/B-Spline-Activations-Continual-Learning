import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datasets import set_seed
from models import create_model
import os

def generate_data(n_peak=5, n_num_per_peak=100):
    n_sample = n_peak * n_num_per_peak
    x_grid = torch.linspace(-1, 1, steps=n_sample)
    x_centers = 2/n_peak * (np.arange(n_peak) - n_peak/2 + 0.5)
    x_sample = torch.stack([
        torch.linspace(-1/n_peak, 1/n_peak, steps=n_num_per_peak) + center 
        for center in x_centers
    ]).reshape(-1,)
    
    y = 0.
    for center in x_centers:
        y += torch.exp(-(x_grid - center)**2 * 300)
    
    y_sample = 0.
    for center in x_centers:
        y_sample += torch.exp(-(x_sample - center)**2 * 300)
    
    return x_grid, y, x_sample, y_sample, x_centers


def regression_run(model, optimizer, lr, x_grid, x_sample, y_sample, n_peak, n_num_per_peak, device):
    ys = []
    for group_id in range(n_peak):
        dataset = {}
        dataset['train_input'] = x_sample[group_id * n_num_per_peak:(group_id+1)*n_num_per_peak][ : , None]
        dataset['train_label'] = y_sample[group_id * n_num_per_peak:(group_id+1)*n_num_per_peak][ : , None]
        inputs = dataset['train_input']
        targets = dataset['train_label']
        inputs, targets = inputs.to(device), targets.to(device)
        criterion = nn.MSELoss()
        optimizer_instance = optimizer(model.parameters(), lr=lr)
        epochs = 1000

        for _ in range(epochs):
            model.train()
            optimizer_instance.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_instance.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(x_grid[:, None].to(device))
        ys.append(y_pred.cpu().numpy()[:, 0])
    
    return ys


def regression_multi_run(activation, optimizer, lr, num_seeds, n_peak, n_num_per_peak):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate data once
    x_grid, y, x_sample, y_sample, x_centers = generate_data(n_peak, n_num_per_peak)
    
    # Store results for all seeds
    all_results = []
    
    for seed in range(num_seeds):
        set_seed(seed)
        ####
        model = create_model('mlp', 1, 1, act=activation, act_cfg={'num_control_points': 15, 'degree': 1, 'start_point': -1.0, 'end_point': 1.0}).to(device)
        #####
        ys = regression_run(model, optimizer, lr, x_grid, x_sample, y_sample, n_peak, n_num_per_peak, device)
        all_results.append(ys)

    all_results = np.array(all_results) 
    
    # Compute mean and std across seeds
    mean_results = np.mean(all_results, axis=0)
    std_results = np.std(all_results, axis=0)
    
    return mean_results, std_results, x_grid, y


def plot_results(mean_results, std_results, x_grid, y, n_peak, activation_name, save_dir):
    fig, axes = plt.subplots(1, n_peak, figsize=(15, 2))
    plt.subplots_adjust(wspace=0.25, hspace=0)
    
    for i in range(n_peak):
        ax = axes[i] if n_peak > 1 else axes
        
        # Plot ground truth
        ax.plot(x_grid.numpy(), y.numpy(), color='black', alpha=0.1, label='Ground Truth')
        
        # Plot mean prediction
        ax.plot(x_grid.numpy(), mean_results[i], color='blue', label='Mean Prediction')
        
        # Plot uncertainty band
        ax.fill_between(x_grid.numpy(), mean_results[i] - std_results[i], mean_results[i] + std_results[i], color='blue', alpha=0.2, label='±1 Std')
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 2)
        ax.set_title(f'Task {i+1}')
        
        if i == 0:
            ax.legend(fontsize=8)
    
    plt.suptitle(f'Continual Learning with {activation_name.upper()} Activation', y=1.05)
    plt.tight_layout()
    

    if save_dir:
        viz_dir = os.path.join(save_dir, 'viz_reg')
        os.makedirs(viz_dir, exist_ok=True)
        save_path = os.path.join(viz_dir, f'{activation_name}_continual_regression_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig) 


def continual_reg(num_seeds=3, n_peak=7, save_dir='./results'):
    acts = ['relu', 'bspline']
    for act in acts:
        print(f"Running experiment for {act}")
        if act == 'relu':
            optimizer = torch.optim.Adam # performs better with adam in this task
            lr = 0.001
        elif act == 'bspline':
            optimizer = torch.optim.SGD
            lr = 0.003 
        else:
            return ValueError()
        
        n_num_per_peak=100         
        mean_results, std_results, x_grid, y = regression_multi_run(act, optimizer, lr, num_seeds, n_peak, n_num_per_peak)

        plot_results(mean_results, std_results, x_grid, y, n_peak, act, save_dir)
    
