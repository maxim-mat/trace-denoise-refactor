import torch
import lightning as L
from torch.utils.data import DataLoader, random_split
from functools import partial
from .dataset import TracesDataset, collate_traces_batch, collate_traces_batch_probabilistic


class TracesDataModule(L.LightningDataModule):
    """Lightning DataModule for trace data."""
    
    def __init__(
        self,
        labels,
        data,
        final_channels,
        batch_size=32,
        num_workers=0,
        val_split=0.1,
        test_split=0.1,
        padding_value=0,
        one_hot_labels=False,
        target_length=None,
        probabilistic=False,
        pin_memory=True,
        persistent_workers=False,
        seed=42,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['labels', 'data'])
        
        self.labels = labels
        self.data = data
        self.final_channels = final_channels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.padding_value = padding_value
        self.one_hot_labels = one_hot_labels
        self.target_length = target_length
        self.probabilistic = probabilistic
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seed = seed
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.full_dataset = None
        
    def setup(self, stage=None):
        if self.full_dataset is None:
            self.full_dataset = TracesDataset(self.labels, self.data)
        
        dataset_size = len(self.full_dataset)
        test_size = int(self.test_split * dataset_size)
        val_size = int(self.val_split * dataset_size)
        train_size = dataset_size - val_size - test_size
        
        if self.val_split > 0 and val_size == 0:
            val_size = 1
            train_size -= 1
        if self.test_split > 0 and test_size == 0:
            test_size = 1
            train_size -= 1
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed),
        )
        
    def _get_collate_fn(self):
        collate_fn = (
            collate_traces_batch_probabilistic 
            if self.probabilistic 
            else collate_traces_batch
        )
        return partial(
            collate_fn,
            final_channels=self.final_channels,
            padding_value=self.padding_value,
            one_hot_labels=self.one_hot_labels,
            target_length=self.target_length,
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )
