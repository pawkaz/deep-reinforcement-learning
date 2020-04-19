import torch
import math
from torch import nn

class Multinomial(nn.Module):
    """
    Multinomail distribution based on the sum tree data structure.
    Args:
    probs(torch.Tensor): Given probabilities
    dtype(torch.dtype): Type of underlaying datastructure
    normalize: Apply normalization to the given probabiblities
    """
    def __init__(self, priorities: torch.Tensor, dtype=torch.float32, normalize: bool = False, device='cpu'):
        super(Multinomial, self).__init__()
        self.dtype = dtype
        self.normalize = normalize
        self.capacity = priorities.shape[0]
        self.test = torch.ones(5)
        self.device = device        
        self.tree = nn.Parameter(self._create_tree(priorities, None), requires_grad=False)
        
    @classmethod
    def create(cls, priorities: torch.Tensor, dtype=torch.float32, normalize: bool = False, device='cpu'):
        dist = cls(priorities, dtype, normalize, device)
        return torch.jit.script(dist)
        
    @torch.jit.export
    def _find(self, idx: torch.Tensor, sums: torch.Tensor) -> torch.Tensor:
        left = 2 * idx + 1
        right = left + 1
        not_done = left.lt(self.tree.size(0))
        left_value = self.tree[left]

        while not_done.sum() > 0:
            go_left = sums.le(left_value)
            idx_new = torch.where(go_left, left, right)
            sums_new = torch.where(go_left, sums, sums - left_value)
            idx = torch.where(not_done, idx_new, idx)
            sums = torch.where(not_done, sums_new, sums)

            left = 2 * idx + 1
            right = left + 1
            not_done = left.lt(self.tree.size(0))
            left_value[not_done] = self.tree[left[not_done]]

        return idx
    
    
    @property
    def total_priority(self) -> int:
        """
        Returns total sum of given probabilities.
        """
        return self.tree[0].item()
    
    @torch.jit.export
    def sample(self, size: int, equal_mode:bool=True):
        """
        Draws samples from a distribution.
        """

        samples = torch.rand(size, dtype=self.dtype, device=self.tree.device) * self.tree[0]
        if equal_mode:
            offsets = torch.linspace(0, self.tree[0], size+1, device=self.tree.device)[:-1]
            samples /= size
            samples += offsets

        idxs = torch.zeros(size, dtype=torch.long, device=self.tree.device)

        return self._find(idxs, samples) - self.capacity + 1

    def __call__(self, size: int):
        return self.sample(size)
    

    @torch.jit.export
    def update(self, probs: torch.Tensor):
        """
        Updates distribution by a given probabilities.
        """
        self.tree = self._create_tree(probs, self.tree)
        
    @torch.jit.export
    def _create_tree(self, probs: torch.Tensor, tree:torch.Tensor):
        
        capacity = probs.shape[0]    
        self.capacity = capacity             
        if tree is None:
            tree = torch.zeros(2 * capacity - 1, dtype=self.dtype, device=self.device)
        
        full_levels = int(torch.log2(torch.tensor(capacity).float()))

        leafs_in_last_row = int(2 * capacity - 2**(full_levels+1))

        tree[-capacity:].copy_(probs)

        if self.normalize:
            tree[-capacity:].div_(probs.sum())

        row_size = int(2**full_levels)
        start_row_idx = int(2 * capacity - 1 - row_size - leafs_in_last_row)

        if leafs_in_last_row > 0:
            tree[start_row_idx: int(start_row_idx + (leafs_in_last_row / 2))].copy_(tree[-leafs_in_last_row:].view(-1, 2).sum(1))

        upper_row = tree[start_row_idx: start_row_idx + row_size]

        for l in range(full_levels):
            row_size = int(row_size / 2)
            start_row_idx -= row_size
            upper_row = upper_row.view(-1, 2).sum(1)
            tree[start_row_idx:start_row_idx+row_size].copy_(upper_row)
        
        return tree