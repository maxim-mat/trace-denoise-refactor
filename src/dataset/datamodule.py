import torch
import lightning as L
from torch.utils.data import DataLoader, random_split
from functools import partial
from .dataset import TracesDataset, collate_traces_batch, collate_traces_batch_probabilistic, collate_traces_mask
import logging
from src.utils.pm_utils import discover_process, get_petri_net_flow_matrix
from src.utils.graph_utils import prepare_process_model_for_gnn, prepare_process_model_for_heterognn

logger = logging.getLogger(__name__)


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
        use_padding_mask=False,
        pin_memory=True,
        persistent_workers=False,
        seed=42,
        get_flow_matrix=False,
        get_graph_data=False,
        process_discovery_method=None,
        remove_duplicates=True,
        activity_names=None,
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
        self.use_padding_mask = use_padding_mask
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.get_flow_matrix = get_flow_matrix
        self.get_graph_data = get_graph_data
        self.process_discovery_method = process_discovery_method
        self.remove_duplicates = remove_duplicates
        self.activity_names = activity_names
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.full_dataset = None

        self.flow_matrix = None
        self.graph_data = None
        self.process = None  # tuple of process_model, init_marking, final_marking
        
    def setup(self, stage=None):
        if self.full_dataset is None:
            self.full_dataset = TracesDataset(self.labels, self.data)
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset,
            [1 - (self.val_split + self.test_split), self.val_split, self.test_split],
            generator=torch.Generator().manual_seed(self.seed),
        )

        if self.get_flow_matrix or self.get_graph_data:
            logger.info("Discovering process")
            self.process = self._run_discovery()
            if self.get_flow_matrix:
                logger.info("Getting flow matrix")
                self.flow_matrix = self._get_flow_matrix()
                logger.debug(f"Flow matrix shape: {self.flow_matrix.shape}")
            if self.get_graph_data:
                logger.info("Getting graph data")
                self.graph_data = self._get_graph_data()

    def _run_discovery(self):
        process_model, init_marking, final_marking = discover_process(self.train_dataset, 
        self.process_discovery_method, self.remove_duplicates, self.activity_names)
        return process_model, init_marking, final_marking

    def _get_flow_matrix(self):
        flow_matrix = get_petri_net_flow_matrix(*self.process)
        return flow_matrix

    def _get_graph_data(self):
        graph_data = prepare_process_model_for_gnn(*self.process)
        return graph_data
        
    def _get_collate_fn(self):
        if self.use_padding_mask:
            return partial(
                collate_traces_mask,
                one_hot_labels=self.one_hot_labels,
            )

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
