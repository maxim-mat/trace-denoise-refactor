from dataclasses import dataclass


@dataclass
class Config:
    data_path: str = None
    summary_path: str = None
    device: str = None
    parallelize: bool = None
    num_epochs: int = None
    learning_rate: float = None
    num_timesteps: int = None
    train_percent: float = None
    num_workers: int = None
    test_every: int = None
    num_classes: int = None
    batch_size: int = None
    conditional_dropout: float = None
    matrix_dropout: float = None  # also used for graph dropout
    eval_train: bool = None
    mode: str = None
    predict_on: str = None
    seed: int = None
    enable_matrix: bool = False
    enable_gnn: bool = False
    hetero: bool = False
    gnn_pooling: str = None
    gamma: float = None
    matrix_type: str = None
    activity_names: dict[int, str] = None
    process_discovery_method: str = None
    round_precision: int = 2

    def __post_init__(self):
        if self.mode == "uncond":
            self.conditional_dropout = 1.0
        if self.activity_names is not None:
            self.activity_names = {int(k): v for k, v in self.activity_names.items()}
