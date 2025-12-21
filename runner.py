from trainer import run_multi_seed
from visualize import vizz
import argparse
from pathlib import Path
from regress import continual_reg


CONFIGS = {
    'relu': {'lr': 0.01, 'optimizer': 'sgd'},
    'tanh': {'lr': 0.005, 'optimizer': 'sgd'},
    'gelu': {'lr': 0.01, 'optimizer': 'sgd'},
    'prelu': {'lr': 0.01, 'optimizer': 'sgd'},
    'swish': {'lr': 0.01, 'optimizer': 'sgd'},
    'bspline': {'num_control_points': 20, 'degree': 3, 'start_point': -2.0, 'end_point': 2.0, 'lr': 0.0003, 'optimizer': 'sgd'}
}


def run_act_comp(dataset, num_seeds, save_dir='./results', use_compile=True):
    print(f"\nDataset: {dataset}\n")

    if 'split_mnist' in dataset or 'rotated_mnist' in dataset:
        num_tasks = 5
        batch_size = 64
    else:  # permuted_mnist
        num_tasks = 10
        batch_size = 64
    epochs = 5

    activations = ['relu', 'sigmoid', 'tanh', 'gelu', 'prelu', 'swish', 'bspline']
    results = {}
    folder_names = []
    for act in activations:
        print('\nAct: ', act)

        act_cfg = {}
        if act == 'bspline':
            act_cfg = {
                'num_control_points': CONFIGS[act]['num_control_points'],
                'degree': CONFIGS[act]['degree'],
                'start_point': CONFIGS[act]['start_point'],
                'end_point': CONFIGS[act]['end_point']
            }

        cfg = {
            'name': f'{dataset}_{act}',
            'dataset': dataset,
            'num_tasks': num_tasks,
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': CONFIGS[act]['lr'],
            'optimizer': CONFIGS[act]['optimizer'],
            'activation': act,
            'act_cfg': act_cfg,
            'save_dir': save_dir,
            'use_compile': use_compile
        }
        folder_names.append(cfg['name'])
        result = run_multi_seed(cfg, num_seeds =num_seeds, vb = True)
        results[act] = result

    vizz(save_dir, activations, folder_names, output_dir=str(Path(save_dir) / f'{dataset}_visualizations'))

    return results



def run_ablations(dataset, num_seeds, save_dir='./results', use_compile=True):

    print(f"\nDataset: {dataset}\n")

    all_results = {}

    # Determine num_tasks and batch_size based on dataset
    if 'split_mnist' in dataset or 'rotated_mnist' in dataset:
        num_tasks = 5
        batch_size = 64
    else:  # permuted_mnist
        num_tasks = 10
        batch_size = 64
    epochs = 5
    lr = 0.0003
    optimizer = 'sgd'


    print("\nControl Points")
    cp_vals = [3, 5, 7, 10, 15, 20, 30, 50]
    cp_results = {}
    cp_names = []
    folder_names = []
    for cp in cp_vals:
        print("\n # control points: ", cp)
        name = f'cp_{cp}'
        cp_names.append(name)
        cfg = {
            'name': f'{dataset}_bspline_{name}',
            'dataset': dataset,
            'num_tasks': num_tasks,
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': lr,
            'optimizer': optimizer,
            'activation': 'bspline',
            'act_cfg': {'num_control_points': cp, 'degree': 3, 'start_point': -2.0, 'end_point': 2.0},
            'save_dir': save_dir,
            'use_compile': use_compile
        }
        folder_names.append(cfg['name'])
        result = run_multi_seed(cfg, num_seeds=num_seeds, vb=True)
        cp_results[name] = result
    all_results['control_points'] = cp_results
    vizz(save_dir, cp_names, folder_names, output_dir=str(Path(save_dir) / f'{dataset}_viz_cp'))


    print("\nDegrees")
    deg_vals = [1, 2, 3, 4, 5]
    deg_results = {}
    deg_names = []
    folder_names = []
    for deg in deg_vals:
        print("\n Deg: ", deg)
        name = f'deg_{deg}'
        deg_names.append(name)
        cfg = {
            'name': f'{dataset}_bspline_{name}',
            'dataset': dataset,
            'num_tasks': num_tasks,
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': lr,
            'optimizer': optimizer,
            'activation': 'bspline',
            'act_cfg': {'num_control_points': 20, 'degree': deg, 'start_point': -2.0, 'end_point': 2.0},
            'save_dir': save_dir,
            'use_compile': use_compile
        }
        folder_names.append(cfg['name'])
        result = run_multi_seed(cfg, num_seeds=num_seeds, vb=True)
        deg_results[name] = result
    all_results['degree'] = deg_results
    vizz(save_dir, deg_names, folder_names, output_dir=str(Path(save_dir) / f'{dataset}_viz_degree'))


    print("\nInit")
    init_vals = ['random', 'identity', 'relu', 'leaky_relu']
    init_results = {}
    init_names = []
    folder_names = []
    for init in init_vals:
        print('\n Init: ', init) 
        name = f'init_{init}'
        init_names.append(name)
        cfg = {
            'name': f'{dataset}_bspline_{name}',
            'dataset': dataset,
            'num_tasks': num_tasks,
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': lr,
            'optimizer': optimizer,
            'activation': 'bspline',
            'act_cfg': {'num_control_points': 20, 'degree': 3, 'start_point': -2.0, 'end_point': 2.0, 'init': init},
            'save_dir': save_dir,
            'use_compile': use_compile
        }
        folder_names.append(cfg['name'])
        result = run_multi_seed(cfg, num_seeds=num_seeds, vb=True)
        init_results[name] = result
    all_results['initialization'] = init_results
    vizz(save_dir, init_names, folder_names, output_dir=str(Path(save_dir) / f'{dataset}_viz_init'))


    
    print("\nShared Activation")
    shared_act_vals = [True, False]
    shared_act_results = {}
    shared_act_names = []
    folder_names = []
    for shared_act in shared_act_vals:
        print("\n shared?: ", shared_act)
        name = f'shared_act_{shared_act}'
        shared_act_names.append(name)
        cfg = {
            'name': f'{dataset}_bspline_{name}',
            'dataset': dataset,
            'num_tasks': num_tasks,
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': lr,
            'optimizer': optimizer,
            'activation': 'bspline',
            'act_cfg': {'num_control_points': 20, 'degree': 3, 'start_point': -2.0, 'end_point': 2.0, 'shared_act': shared_act},
            'shared_act': shared_act,
            'save_dir': save_dir,
            'use_compile': use_compile
        }
        folder_names.append(cfg['name'])
        result = run_multi_seed(cfg, num_seeds=num_seeds, vb=True)
        shared_act_results[name] = result
    all_results['shared_activation'] = shared_act_results
    vizz(save_dir, shared_act_names, folder_names, output_dir=str(Path(save_dir) / f'{dataset}_viz_shared_act'),
         num_tasks=num_tasks, num_seeds=num_seeds)

    return all_results


def run_combine(dataset, method, num_seeds, save_dir='./results', use_compile=True):
    print(f"\nDataset: {dataset}, method: {method}\n")

    if 'split_mnist' in dataset or 'rotated_mnist' in dataset:
        num_tasks = 5
        batch_size = 64
    else:  # permuted_mnist
        num_tasks = 10
        batch_size = 64
    epochs = 5

    activations = ['relu', 'bspline']
    results = {}
    folder_names = []
    for act in activations:
        print('\nAct: ', act)

        act_cfg = {}
        if act == 'bspline':
            act_cfg = {
                'num_control_points': CONFIGS[act]['num_control_points'],
                'degree': CONFIGS[act]['degree'],
                'start_point': CONFIGS[act]['start_point'],
                'end_point': CONFIGS[act]['end_point']
            }

        cfg = {
            'name': f'{dataset}_{method}_{act}',
            'dataset': dataset,
            'num_tasks': num_tasks,
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': CONFIGS[act]['lr'],
            'optimizer': CONFIGS[act]['optimizer'],
            'activation': act,
            'act_cfg': act_cfg,
            'method': method,
            'save_dir': save_dir,
            'use_compile': use_compile
        }
        folder_names.append(cfg['name'])
        result = run_multi_seed(cfg, num_seeds=num_seeds, vb=True)
        results[act] = result

    vizz(save_dir, activations, folder_names, output_dir=str(Path(save_dir) / f'{dataset}_visualizations_{method}'),
         num_tasks=num_tasks, num_seeds=num_seeds)

    return results




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run continual learning experiments with B-spline activations')

    parser.add_argument('--exp', type=str, default='compare', choices=['compare', 'ablations', 'regression', 'combine_methods'])
    parser.add_argument('--dataset', type=str, default='permuted_mnist')
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument('--num_peaks', type=int, default=7) # for regression exp
    parser.add_argument('--method', type=str, default='ewc')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--no_compile', action='store_true')

    args = parser.parse_args()

    use_compile = not args.no_compile


    if args.exp == 'compare':
        run_act_comp(args.dataset, args.num_seeds, args.save_dir, use_compile)
    elif args.exp == 'ablations':
        run_ablations(args.dataset, args.num_seeds, args.save_dir, use_compile)
    elif args.exp == 'regression':
        continual_reg(args.num_seeds, args.num_peaks, args.save_dir)
    elif args.exp == 'combine_methods':
        run_combine(args.dataset, args.method, args.num_seeds, args.save_dir, use_compile)
