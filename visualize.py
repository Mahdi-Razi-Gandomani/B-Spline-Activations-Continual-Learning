import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


plt.style.use('seaborn-v0_8-paper')


def load_results(save_dir, exp_name):
    path = Path(save_dir) / exp_name / 'all_results.json'
    with open(path, 'r') as f:
        return json.load(f)


def heat_plot(acc_mat_mean, title, save_path=None):
    T = len(acc_mat_mean)
    fig, ax = plt.subplots(figsize=(5, 4))
    

    im = ax.imshow(acc_mat_mean, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(T))
    ax.set_yticks(np.arange(T))
    ax.set_xticklabels([f'{i+1}' for i in range(T)])
    ax.set_yticklabels([f'{i+1}' for i in range(T)])
    ax.set_xlabel('Task $j$')
    ax.set_ylabel('Trained on tasks $1...i$')
    ax.set_title(title)
    
    # Annotate cells with values
    for i in range(T):
        for j in range(T):
            if acc_mat_mean[i][j] > 0:
                text = ax.text(j, i, f'{acc_mat_mean[i][j]:.2f}', ha='center', va='center', color='white' if acc_mat_mean[i][j] > 0.5 else 'black', fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def per_task_acc(acc_mat_mean, title, save_path=None):
    acc = np.array(acc_mat_mean)
    T = acc.shape[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    
    for task_j in range(T):
        ys = acc[ :, task_j].copy()
        ys[ : task_j] = np.nan  
        
        ax.plot(range(1, T+1), ys, marker='o', label=f'Task {task_j+1}', linewidth=1.5, markersize=6)

    ax.set_xlabel('Training phase')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(range(1, T+1))
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def avg_over_time(results_dict, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for idx, (exp_name, exp_result) in enumerate(results_dict.items()):
        if exp_result is None:
            continue
        
        acc_mean = np.array(exp_result['accuracy_matrix']['mean'])
        acc_std = np.array(exp_result['accuracy_matrix']['std']) if 'std' in exp_result['accuracy_matrix'] else None
        
        T = acc_mean.shape[0]
        

        avg_accs = []
        std_accs = []
        
        for t in range(T):
            valid_accs = []
            
            for task_j in range(t + 1):
                if acc_mean[t][task_j] > 0:
                    valid_accs.append(acc_mean[t][task_j])
            
            avg_accs.append(np.mean(valid_accs) if valid_accs else 0)
            
            
            if acc_std is not None and valid_accs:
                std_accs.append(np.std(valid_accs) / (np.sqrt(len(valid_accs)) * 2))
            else:
                std_accs.append(0)
        
        x = np.arange(1, T + 1)
        

        if any(s > 0 for s in std_accs):
            ax.fill_between(x, np.array(avg_accs) - np.array(std_accs), np.array(avg_accs) + np.array(std_accs), alpha=0.15)
        

        ax.plot(x, avg_accs, linewidth=3, label=exp_name, marker='o', markersize=5)
    
    ax.set_xlabel('Training phase', fontsize=14, fontweight='normal')
    ax.set_ylabel('Average accuracy', fontsize=14, fontweight='normal')
    

    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def metrics_compa(results_dict, metric_names, save_path=None):
    exp_names = [name for name in results_dict.keys() if results_dict[name] is not None]
    n_metrics = len(metric_names)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(3*n_metrics, 3.5))
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        
        means = []
        valid_names = []
        
        for name in exp_names:
            if metric in results_dict[name]['cl_metrics']:
                means.append(results_dict[name]['cl_metrics'][metric]['mean'])
                valid_names.append(name)
        
        if not valid_names:
            continue
        
        x = np.arange(len(valid_names))
        

        ax.bar(x, means, capsize=4, alpha=0.7, edgecolor='black', linewidth=0.8)
        for i, v in enumerate(means):
            if v >= 0:
                ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
            else:
                ax.text(i, v, f'{v:.2f}', ha='center', va='top', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_names, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').capitalize())
        ax.set_title(metric.replace('_', ' ').capitalize())
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()




def vizz(save_dir, exp_list, names, output_dir='./visualizations'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_dict = {}

    for exp_key, full_name in zip(exp_list, names):
        results_dict[exp_key] = load_results(save_dir, full_name)

    for act_name, exp_result in results_dict.items():
        acc_mean = exp_result['accuracy_matrix']['mean']
        
        heat_plot(acc_mean, f'Accuracy matrix: {act_name}', save_path=output_path / f'acc_matrix_{act_name}.png')
        
        per_task_acc(acc_mean, title=f'Per task accuracy: {act_name}', save_path=output_path / f'per_task_accuracy_{act_name}.png')


    avg_over_time(results_dict, save_path=output_path / 'average_accuracy_comparison.png')
    
    metric_names = ['accuracy', 'forgetting', 'bwt', 'plasticity']
    metrics_compa(results_dict, metric_names, save_path=output_path / 'metrics_comparison.png')
    

    print(f"\nAll visualizations saved to: {output_path}")
