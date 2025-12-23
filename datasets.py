import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# train_data = datasets.MNIST(root='./data', train=True, download=True)
# all_pixels = torch.tensor([np.array(img).astype(np.float32).reshape(-1) / 255.0 for img, _ in train_data])
# mean = all_pixels.mean().item()
# std = all_pixels.std().item()
# print(mean)
# print(std)



class PermutedMNIST(Dataset):
    def __init__(self, task_id, train=True, root='./data'):
        self.data = datasets.MNIST(root=root, train=train, download=True)
        np.random.seed(task_id)
        self.perm = np.random.permutation(784)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = np.array(img).astype(np.float32).reshape(-1) / 255.0
        img = img[self.perm]
        return torch.tensor(img), label


def get_permuted_mnist(num_tasks=5, batch_size=64):
    tasks = []
    for t in range(num_tasks):
        train_set = PermutedMNIST(t)
        test_set = PermutedMNIST(t, train=False)
        tasks.append({'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),
            'test': DataLoader(test_set, batch_size=batch_size)
        })
    return tasks


class SplitMNIST(Dataset):
    def __init__(self, task_id, train=True, root='./data'):
        self.full = datasets.MNIST(root=root, train=train, download=True)
        classes = [task_id * 2, task_id * 2 + 1]
        self.ind = [i for i, (_, l) in enumerate(self.full) if l in classes]
        self.label_map = {classes[0]: 0, classes[1]: 1}

    def __len__(self):
        return len(self.ind)

    def __getitem__(self, idx):
        img, label = self.full[self.ind[idx]]
        img = np.array(img).astype(np.float32).reshape(-1) / 255.0
        return torch.tensor(img), self.label_map[label]


def get_split_mnist(num_tasks=5, batch_size=64):
    tasks = []
    for t in range(num_tasks):
        train_set = SplitMNIST(t, train=True)
        test_set = SplitMNIST(t, train=False)
        tasks.append({'train': DataLoader(train_set, batch_size=batch_size, shuffle=True), 'test': DataLoader(test_set, batch_size=batch_size)})
    return tasks





def get_dataset(name, num_tasks, batch_size):
    if name == 'permuted_mnist':
        return get_permuted_mnist(num_tasks, batch_size)
    elif name == 'split_mnist':
        return get_split_mnist(num_tasks, batch_size)
    else:
        return ValueError('Unknown dataset')
