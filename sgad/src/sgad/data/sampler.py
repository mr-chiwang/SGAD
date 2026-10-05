# Copyright SenseTime. All Rights Reserved.
# LoFTR-derived portions are licensed under Apache-2.0; see THIRD_PARTY_LICENSES.
# Adapted LoFTR dataset sampling for SGAD region matching.

import torch
from torch.utils.data import Sampler, ConcatDataset


class RandomConcatSampler(Sampler):
    """Sample a fixed number of indices from every ``ConcatDataset`` subset.

    Sampling may use replacement; combined indices may be shuffled and
    repeated. Generator state advances across ``__iter__`` calls, while
    recreating the sampler with the same seed repeats the sequence. The
    dataset is expected to be partitioned across ranks before construction.

    Args:
        shuffle: Shuffle the combined indices from all subsets.
        repeat: Repeat the sampled indices for training.
    """
    def __init__(self,
                 data_source: ConcatDataset,
                 n_samples_per_subset: int,
                 subset_replacement: bool=True,
                 shuffle: bool=True,
                 repeat: int=1,
                 seed: int=None):
        if not isinstance(data_source, ConcatDataset):
            raise TypeError("data_source should be torch.utils.data.ConcatDataset")
        
        self.data_source = data_source
        self.n_subset = len(self.data_source.datasets)
        self.n_samples_per_subset = n_samples_per_subset
        self.n_samples = self.n_subset * self.n_samples_per_subset * repeat
        self.subset_replacement = subset_replacement
        self.repeat = repeat
        self.shuffle = shuffle
        self.generator = torch.manual_seed(seed)
        assert self.repeat >= 1
        
    def __len__(self):
        return self.n_samples
    
    def __iter__(self):
        indices = []
        # Sample from each subset.
        for d_idx in range(self.n_subset):
            low = 0 if d_idx==0 else self.data_source.cumulative_sizes[d_idx-1]
            high = self.data_source.cumulative_sizes[d_idx]
            if self.subset_replacement:
                rand_tensor = torch.randint(low, high, (self.n_samples_per_subset, ),
                                            generator=self.generator, dtype=torch.int64)
            else:  # Sample without replacement, then pad if needed.
                len_subset = len(self.data_source.datasets[d_idx])
                rand_tensor = torch.randperm(len_subset, generator=self.generator) + low
                if len_subset >= self.n_samples_per_subset:
                    rand_tensor = rand_tensor[:self.n_samples_per_subset]
                else:
                    rand_tensor_replacement = torch.randint(low, high, (self.n_samples_per_subset - len_subset, ),
                                                            generator=self.generator, dtype=torch.int64)
                    rand_tensor = torch.cat([rand_tensor, rand_tensor_replacement])
            indices.append(rand_tensor)
        indices = torch.cat(indices)
        if self.shuffle:
            rand_tensor = torch.randperm(len(indices), generator=self.generator)
            indices = indices[rand_tensor]

        # Repeat the sampled indices, optionally reshuffling each copy.
        if self.repeat > 1:
            repeat_indices = [indices.clone() for _ in range(self.repeat - 1)]
            if self.shuffle:
                _choice = lambda x: x[torch.randperm(len(x), generator=self.generator)]
                repeat_indices = map(_choice, repeat_indices)
            indices = torch.cat([indices, *repeat_indices], 0)
        
        assert indices.shape[0] == self.n_samples
        return iter(indices.tolist())
