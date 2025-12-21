import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import get_dataset, set_seed
from models import create_model
import json
from pathlib import Path
from metrics import CLMetrics, CompMetrics
from tqdm import tqdm
from ewc import get_ewc


class Trainer:
    def __init__(self, model, tasks, cfg, device):
        self.model = model.to(device)
        self.tasks = tasks
        self.cfg = cfg
        self.device = device
        self.acc_matrix = np.zeros((len(tasks), len(tasks)))
        self.history = []
        self.epoch_accs = []
        self.comp_metrics = CompMetrics()
        self.optimizer = self._make_optimizer()
        self.crite = nn.CrossEntropyLoss()
        
        self.use_ewc = cfg.get('use_ewc', False)
        self.use_er = cfg.get('use_er', False)

        if self.use_ewc:
            self.regularizer = get_ewc(cfg, self.model, self.device)
        if self.use_er:
            self.er_buffer_x = []
            self.er_buffer_y = []
            self.er_buffer_max = self.cfg.get('er_buffer_size', 500)
            self.er_lambda = self.cfg.get('er_lambda', 1.0)
            self.reservoir_count = 0

    def _make_optimizer(self):
        lr = self.cfg.get('lr')
        opt_name = self.cfg.get('optimizer')
        if opt_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=self.cfg.get('momentum', 0.9))
        else:
            return optim.Adam(self.model.parameters(), lr=lr)


    def train_task(self, task_id, train_loader, vb=True):
        self.model.train()
        self.comp_metrics.start_timer()

        task_losses = []
        task_accs = []
        epochs = self.cfg.get('epochs')

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"task {task_id+1}, epoch {epoch+1}/{epochs}") if vb else train_loader

            # Store samples from this task for future replay
            if self.use_er:
                task_samples_x = []
                task_samples_y = []

            
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)

                if self.use_er and epoch == 0:
                    task_samples_x.append(data.detach().cpu())
                    task_samples_y.append(target.detach().cpu())

                if self.use_ewc:
                    prev_params = {name: p.detach().clone() for name, p in self.model.named_parameters()}

                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.crite(output, target)

 
                if self.use_er and len(self.er_buffer_x) > 0:
                    replay_x, replay_y = self._sample_er_batch()
                    replay_x = replay_x.to(self.device)
                    replay_y = replay_y.to(self.device)
                    replay_out = self.model(replay_x)
                    replay_loss = self.crite(replay_out, replay_y)
                    loss = loss + self.er_lambda * replay_loss

                if self.use_ewc:
                    loss = loss + self.regularizer.penalty(self.model)
                
                loss.backward()
                self.optimizer.step()

                if self.use_ewc:
                    self.regularizer.after_step(self.model, prev_params)
                    
                epoch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)

                if vb:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100. * correct / total:.2f}%'})
                        
                    
            if self.use_er and epoch == 0 and task_samples_x:
                self._update_er_buffer(task_samples_x, task_samples_y)
                
            avg_loss = epoch_loss / len(train_loader)
            accuracy = correct / total
            task_losses.append(avg_loss)
            task_accs.append(accuracy)

            # eval all tasks after epoch
            epoch_eval = self.eval_all_tasks(task_id, vb=False)
            self.epoch_accs.append(epoch_eval)

            if vb:
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")

        task_time = self.comp_metrics.end_timer()

        return {
            'avg_loss': np.mean(task_losses),
            'final_loss': task_losses[-1],
            'avg_accuracy': np.mean(task_accs),
            'final_accuracy': task_accs[-1],
            'time_seconds': task_time,
        }



    def eval_task(self, task_id, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)

        return correct / total if total > 0 else 0.0


    def eval_all_tasks(self, current_task, vb=True):
        accs = []
        for task_id in range(current_task + 1):
            acc = self.eval_task(task_id, self.tasks[task_id]['test'])
            accs.append(acc)

            if vb:
                print(f"Task {task_id+1}: {acc:.2f}")

        return accs


    def train_continual(self, vb=True):
        for task_id, task in enumerate(self.tasks):
            print(f"\nTraining Task {task_id + 1}/{len(self.tasks)}")
            train_stats = self.train_task(task_id, task['train'], vb=vb)
            self.history.append(train_stats)

            if self.use_ewc:
                self.regularizer.on_task_end(self.model, task['train'], self.crite, self.device)
            
            print(f"\nafter Task {task_id + 1}:")
            accs = self.eval_all_tasks(task_id, vb=True)
            self.acc_matrix[task_id, : len(accs)] = accs


        results = self.final_metrics()

        return results


    def final_metrics(self):
        cl_metrics_obj = CLMetrics(self.acc_matrix)
        cl_metrics = cl_metrics_obj.all_metrics()
        comp_metrics = self.comp_metrics.summary()

        return {
            'accuracy_matrix': self.acc_matrix.tolist(),
            'cl_metrics': cl_metrics,
            'comput_metrics': comp_metrics,
            'history': self.history,
            'epoch_accs': self.epoch_accs
        }


    
    def _update_er_buffer(self, data_batches, target_batches):
        all_data = torch.cat(data_batches, dim=0)
        all_targets = torch.cat(target_batches, dim=0)
        
        n_samples = len(all_data)
        for i in range(n_samples):
            x = all_data[i]
            y = all_targets[i]
            
            if len(self.er_buffer_x) < self.er_buffer_max:
                self.er_buffer_x.append(x)
                self.er_buffer_y.append(y)
            else:
                j = np.random.randint(0, self.reservoir_count + 1)
                if j < self.er_buffer_max:
                    self.er_buffer_x[j] = x
                    self.er_buffer_y[j] = y
            
            self.reservoir_count += 1

    def _sample_er_batch(self):
        size = min(len(self.er_buffer_x), 64)
        
        inds = torch.randint(0, len(self.er_buffer_x), (size, ))
        xs = torch.stack([self.er_buffer_x[i] for i in inds], dim=0)
        ys = torch.stack([self.er_buffer_y[i] for i in inds], dim=0)
        
        return xs, ys


    
    def save_results(self, save_dir):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        results = self.final_metrics()
        with open(save_path / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)


def run_experiment(cfg, seed, vb=True):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('\ndevice: ', device)

    tasks = get_dataset(cfg['dataset'], cfg['num_tasks'], cfg['batch_size'])

    dname = cfg['dataset'].lower()
    if 'permuted_mnist' in dname or 'rotated_mnist' in dname:
        in_size = 784
        out_size = 10
        hidden_sizes = [256, 256]
        model_type = 'mlp'
    elif 'split_mnist' in dname:
        in_size = 784
        out_size = 2
        hidden_sizes = [256, 256]
        model_type = 'mlp'
    # elif 'cifar10' in dname:
    #     in_size = 3
    #     out_size = 10 // cfg['num_tasks']
    #     model_type = 'cnn'
    else:
        raise ValueError()

    model = create_model(model_type, in_size, out_size, cfg['activation'], hidden_sizes=hidden_sizes,act_cfg=cfg.get('act_cfg', {}), shared_act=cfg.get('shared_act', False))

    use_compile = cfg.get('use_compile', False)
    if use_compile and hasattr(torch, 'compile'):
        model = torch.compile(model, mode='reduce-overhead')

    trainer = Trainer(model, tasks, cfg, device)
    results = trainer.train_continual(vb=vb)

    save_dir = Path(cfg.get('save_dir', './results')) / cfg['name'] / f'seed_{seed}'
    trainer.save_results(str(save_dir))

    return results


def run_multi_seed(cfg, num_seeds, vb=True):
    all_results = []

    for seed in range(num_seeds):
        print(f"seed: {seed + 1}/{num_seeds}")

        results = run_experiment(cfg, seed, vb = vb)
        all_results.append(results)

    # Combine results
    cl_keys = all_results[0]['cl_metrics'].keys()
    comb_cl = {}
    for key in cl_keys:
        vals = [r['cl_metrics'][key] for r in all_results]
        comb_cl[key] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals))
        }

    acc_mats = [np.array(r['accuracy_matrix']) for r in all_results]
    
    comb_acc = {'mean': np.mean(acc_mats, axis=0).tolist(), 'std': np.std(acc_mats, axis=0).tolist()}
    comp_keys = all_results[0]['comput_metrics'].keys() 
    comb_comp = {}
    for key in comp_keys:
        vals = [r['comput_metrics'][key] for r in all_results]
        comb_comp[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    comb = {
        'cl_metrics': comb_cl,
        'accuracy_matrix': comb_acc,
        'comput_metrics': comb_comp,
        'num_seeds': len(all_results)
    }



    save_dir = Path(cfg.get('save_dir', './results')) / cfg['name']
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'all_results.json', 'w') as f:
        json.dump(comb, f, indent=2)


    return comb
