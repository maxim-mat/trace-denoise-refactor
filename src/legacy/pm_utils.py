import multiprocessing
from typing import Callable

import numpy as np
import pandas as pd
import pm4py
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.conformance.alignments.petri_net.variants import state_equation_a_star
from pm4py.objects.conversion.log import converter
import torch
import torch.nn.functional as F
from itertools import groupby
from src.dataset.dataset import TracesDataset
from sktr.sktr import convert_dataset_to_df, from_discovered_model_to_PetriNet, recover_single_trace, \
    sfmx_mat_to_sk_trace
import sktr.sktr
from utils.config import Config


def prepare_df_cols_for_discovery(df):
    df_copy = df.copy()
    df_copy.loc[:, 'order'] = df_copy.groupby('case:concept:name').cumcount()
    df_copy.loc[:, 'time:timestamp'] = pd.to_datetime(df_copy['order'])

    return df_copy


def convert_dataset_to_train_process_df(deterministic, stochastic, cfg: Config):
    dk_process_df, _ = convert_dataset_to_df(deterministic, stochastic, cfg.activity_names)
    return prepare_df_cols_for_discovery(dk_process_df)


def convert_dataset_to_stochastic_traces(dataset: TracesDataset, cfg: Config):
    _, stochastic_list = convert_dataset_to_df(dataset, cfg.activity_names)
    return stochastic_list


def resolve_process_discovery_method(method_name: str) -> Callable:
    match method_name:
        case "inductive":
            return pm4py.discover_petri_net_inductive
        case _:
            raise AttributeError(f"Unsupported discovery method: {method_name}")


def subsample_time_series(trace_data: dict, num_indexes: int, axis: int = 0):
    """
    reduces trace lengths to num_indexes
    :param trace_data: original dk and sk traces
    :param num_indexes: length of new, under-sampled, sequences
    :param axis: axis along which to sample
    :return: reduced length df and sk traces and respective sampled indexes
    """
    dk_sample, sk_sample, sample_indexes = [], [], []
    for dk_trace, sk_trace in zip(trace_data['target'], trace_data['stochastic']):
        if dk_trace.shape[0] != sk_trace.shape[0]:
            raise ValueError(f"trace lengths: {dk_trace.shape[0]} (dk), {sk_trace.shape[0]} (sk) missmatch")
        random_indexes = sorted(np.random.choice(dk_trace.shape[axis], min(num_indexes, len(dk_trace)), replace=False))
        dk_sample.append(dk_trace[random_indexes])
        sk_sample.append(sk_trace[random_indexes])
        sample_indexes.append(random_indexes)
    return {'target': dk_sample, 'stochastic': sk_sample}, sample_indexes


def train_sktr(dataset: TracesDataset, cfg: Config) -> sktr.sktr.PetriNet:
    """
    run process discovery on the train dataset
    :param dataset: train dataset
    :param cfg:
    :return: process model petri net
    """
    df_train = convert_dataset_to_train_process_df(dataset, cfg)
    net, init_marking, final_marking = pm4py.discover_petri_net_inductive(df_train)
    model = from_discovered_model_to_PetriNet(net, non_sync_move_penalty=1)
    return model


def remove_duplicates_trace(trace):
    return torch.tensor([x.item() for x, _ in groupby(trace)])


def remove_duplicates_dataset(dataset: TracesDataset):
    stochastics = [x[1] for x in dataset]
    deterministics = [torch.argmax(x[0], dim=-1) for x in dataset]
    deterministics = [remove_duplicates_trace(x) for x in deterministics]
    return deterministics, stochastics


def dataset_to_list(dataset: TracesDataset):
    deterministics = [torch.argmax(x[0], dim=-1) for x in dataset]
    stochastics = [x[1] for x in dataset]
    return deterministics, stochastics


def discover_dk_process(dataset: TracesDataset, cfg: Config, preprocess=dataset_to_list):
    deterministic, stochastic = preprocess(dataset)
    df_train = convert_dataset_to_train_process_df(deterministic, stochastic, cfg)
    process_discovery_method = resolve_process_discovery_method(cfg.process_discovery_method)
    return process_discovery_method(df_train)


def process_sk_trace(sk_trace: pd.DataFrame, activity_names, round_precision, model, non_sync_penalty):
    print("started processing trace")
    recovered_trace = recover_single_trace(
        sfmx_mat_to_sk_trace(sk_trace, 0, activity_names=activity_names, round_precision=round_precision),
        model, non_sync_penalty, activity_names
    )
    print("ended processing trace")
    return recovered_trace


def evaluate_sktr_on_dataset(dataset: TracesDataset, model: sktr.sktr.PetriNet, cfg: Config):
    stochastic_traces_matrices = convert_dataset_to_stochastic_traces(dataset, cfg)
    args_list = [
        (sk_trace, cfg.activity_names, cfg.round_precision, model, 1)
        for sk_trace in stochastic_traces_matrices
    ]
    with multiprocessing.Pool(processes=cfg.num_workers) as pool:
        recovered_traces = list(
            pool.starmap(process_sk_trace, args_list)
        )

    return recovered_traces


def pad_to_multiple_of_n(tensor, n=32):
    # Get the original spatial dimension (D)
    _, height, width = tensor.shape  # Expecting shape (1, C, D, D)

    # Compute the padding needed to make the dimensions divisible by 32
    pad_height = (n - height % n) % n
    pad_width = (n - width % n) % n

    # Since it's a square, pad equally along height and width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Apply padding (PyTorch padding order: left, right, top, bottom)
    padded_tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))

    return padded_tensor


def list_to_traces(traces_list, activity_names):
    df_deterministic = pd.DataFrame(
        {
            'concept:name': [activity_names[i] for trace in traces_list for i in trace],
            'case:concept:name': [str(i) for i, trace in enumerate(traces_list) for _ in range(len(trace))]
        }
    )

    return prepare_df_cols_for_discovery(df_deterministic)


def traces_tensor_to_list(traces_tensor, limit=None, remove_duplicates=True):
    if limit is None:
        if remove_duplicates:
            return [remove_duplicates_trace(xi).tolist() for xi in traces_tensor]
        else:
            return [xi.tolist() for xi in traces_tensor]
    else:
        if remove_duplicates:
            return [remove_duplicates_trace(xi).tolist()[:limit] for xi in traces_tensor]
        else:
            return [xi.tolist()[:limit] for xi in traces_tensor]


def conformance_measure(traces_tensor, process_model, initial_marking, final_marking, activity_names,
                        limit=None, remove_duplicates=True, approximate=False):
    traces = traces_tensor_to_list(traces_tensor, limit, remove_duplicates)
    df_traces = list_to_traces(traces, activity_names)
    log = converter.apply(df_traces)
    if approximate:
        alignment_measures = alignments.apply_log(log, process_model, initial_marking, final_marking,
                                                  variant=state_equation_a_star)
    else:
        alignment_measures = alignments.apply_log(log, process_model, initial_marking, final_marking)
    return [d['fitness'] for d in alignment_measures]


def simulate_process_model(process_model: pm4py.PetriNet, init_marking: pm4py.Marking,
                           final_marking: pm4py.Marking, n_traces, max_length):
    pass
