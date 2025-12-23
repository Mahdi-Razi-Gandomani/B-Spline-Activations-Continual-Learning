import torch


class EWC():
    def __init__(self, model, device, lam=1000, samples=200):
        self.device = device
        self.lam = lam
        self.samples = samples
        self.prev_params = {}
        self.fisher = {}
        for name, param in model.named_parameters():
            self.prev_params[name] = param.detach().clone()
    def penalty(self, model):
        if not self.fisher:
            return 0.0
        
        reg = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher:
                reg += (self.fisher[name] * (param - self.prev_params[name]) ** 2).sum()
        return 0.5 * self.lam * reg
    def on_task_end(self, model, dataloader, crite, device):
        fisher = {}
        counts = {}
        model.eval()
        total_seen = 0
        
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            loss = crite(output, target)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                grad = param.grad.detach() ** 2
                if name not in fisher:
                    fisher[name] = grad.clone()
                    counts[name] = 1
                else:
                    fisher[name] += grad
                    counts[name] += 1
            total_seen += data.size(0)
            if total_seen >= self.samples:
                break
        with torch.no_grad():
            for name in fisher:
                fisher[name] /= max(counts[name], 1)
            # Online update
            if self.fisher:
                for name in fisher:
                    self.fisher[name] = fisher[name] + self.fisher.get(name, 0.0)
            else:
                self.fisher = fisher
            
            for name, param in model.named_parameters():
                self.prev_params[name] = param.detach().clone()
def get_ewc(cfg, model, device):
        return EWC(model,device,lam=cfg.get('lam', 1000.0),samples=cfg.get('samples', 200))
