# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Any
import os
from webdataset import TarWriter
import filelock


class RepeatedDatasetWrapper(Dataset):
    def __init__(self, original_dataset, num_repeats):
        """
        Dataset wrapper that repeats the original dataset n times.

        Args:
            original_dataset (torch.utils.data.Dataset): The original dataset to be repeated.
            num_repeats (int): The number of times the dataset should be repeated.
        """
        self.original_dataset = original_dataset
        self.num_repeats = num_repeats

    def __getitem__(self, index):
        """
        Retrieve the item at the given index.
        
        Args:
            index (int): The index of the item to be retrieved.
        """
        original_index = index % len(self.original_dataset)
        return self.original_dataset[original_index]

    def __len__(self):
        """
        Get the length of the dataset after repeating it n times.
        
        Returns:
            int: The length of the dataset.
        """
        return len(self.original_dataset) * self.num_repeats


class SubsampleDatasetWrapper(Dataset):
    def __init__(self, original_dataset, dataset_size, seed=0, return_orig_idx=False):
        """
        Dataset wrapper that randomly subsamples the original dataset.

        Args:
            original_dataset (torch.utils.data.Dataset): The original dataset to be subsampled.
            dataset_size (int): The size of the subsampled dataset.
            seed (int): The seed to use for selecting the subset of indices of the original dataset.
            return_orig_idx (bool): Whether to return the original index of the item in the original dataset.
        """
        self.original_dataset = original_dataset
        self.dataset_size = dataset_size or len(original_dataset)
        self.return_orig_idx = return_orig_idx
        np.random.seed(seed)
        self.indices = np.random.permutation(len(self.original_dataset))[:self.dataset_size]

    def __getitem__(self, index):
        """
        Retrieve the item at the given index.
        
        Args:
            index (int): The index of the item to be retrieved.
        """
        original_index = self.indices[index]
        sample = self.original_dataset[original_index]
        return sample, original_index if self.return_orig_idx else sample

    def __len__(self):
        """
        Get the length of the dataset after subsampling it.
        
        Returns:
            int: The length of the dataset.
        """
        return len(self.indices)

# from MMOMA / AION
class GroupedShardWriter:
    """Similar with wds.ShardWriter but:
    - manages several aligned TarWriter, one for each member of the group
    - distributes the shards across multiple processes.
    The group may refer to a set of modalities.
    This class allows to write tar files in parallel,
    while keeping shard numbers algined.
    Shards with the same number should contain exactly the same keys.

    """

    def __init__(
        self,
        output_dir: str,
        groups: List[str],
        pattern: str,
        maxcount: int = 100000,
        maxsize: float = 3000000000,
        start_shard: int = 0,
        verbose: bool = True,
        distributed: bool = False,
        **kw,
    ):
        self.verbose = verbose
        self.kw = kw
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.shard = start_shard
        self.output_dir = os.path.abspath(output_dir)
        self.groups = groups
        self.common_pattern = pattern
        self.patterns = self.compose_patterns()
        self.total = 0
        self.count = 0
        self.size = 0
        self.tarstreams = {group: None for group in self.groups}
        self.distributed = distributed
        self.next_stream()

    def compose_patterns(self) -> List[str]:
        """Create the patterns corresponding to each group.
        It will also create directories if not existing.

        """
        if os.path.isdir(self.output_dir):
            print(f"Warning: {self.output_dir} already exists")
        else:
            # Directory may be created in between with parallelism
            # Hence exist_ok
            os.makedirs(self.output_dir, exist_ok=True)

        patterns = []
        for group in self.groups:
            group_dir = os.path.join(self.output_dir, group)
            os.makedirs(group_dir, exist_ok=True)
            pattern = os.path.join(group_dir, self.common_pattern)
            patterns.append(pattern)
        return patterns

    def next_stream(self):
        """Close the current streams and move to the next."""
        self.finish()
        self.shard = self.get_shard_number()
        if self.verbose:
            print(f"# writing shard {self.shard} {self.count} {self.size / 1e9:.1f}")
        for group, pattern in zip(self.groups, self.patterns):
            filename = pattern % self.shard
            self.tarstreams[group] = TarWriter(filename, **self.kw)
        self.count = 0
        self.size = 0
        if not self.distributed:
            self.shard += 1

    def write(self, obj: Dict[str, Any]):
        if (
            list(self.tarstreams.values())[0] is None
            or self.count >= self.maxcount
            or self.size >= self.maxsize
        ):
            self.next_stream()
        if set(obj.keys()) != set(self.groups):
            raise ValueError(
                f"Object keys {list(obj.keys())} don't match groups {self.groups}"
            )
        max_size = 0
        for key, val in obj.items():
            size = self.tarstreams[key].write(val)
            max_size = size if size > max_size else max_size
        self.count += 1
        self.total += 1
        self.size += max_size

    def finish(self):
        for group in self.tarstreams:
            tarstream = self.tarstreams[group]
            if tarstream is not None:
                tarstream.close()
            self.tarstreams[group] = None

    def close(self):
        self.finish()
        del self.tarstreams
        del self.shard
        del self.count
        del self.size

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def get_shard_number(self) -> int:
        """Retrive the shard number either directly or from locked file."""
        if not self.distributed:
            shard_number = self.shard
        else:
            shard_number_file = os.path.join(os.path.join(self.output_dir), "shard")
            with filelock.FileLock(f"{shard_number_file}.lock"):
                if os.path.isfile(shard_number_file):
                    with open(shard_number_file, "r") as f:
                        shard_number = int(f.read().strip())
                else:
                    shard_number = self.shard
                with open(shard_number_file, "w") as f:
                    f.write(str(shard_number + 1))
        return shard_number