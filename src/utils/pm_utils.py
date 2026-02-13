from src.dataset.dataset import TracesDataset
import pandas as pd
from torch import tensor
from itertools import groupby
import pm4py
import networkx as nx


DISCOVERY_METHODS = {
    "inductive": pm4py.discover_petri_net_inductive,
}


def remove_duplicates_trace(trace):
    return tensor([x.item() for x, _ in groupby(trace)])


def prepare_dataset_for_discovery(dataset: TracesDataset, remove_duplicates=True, activity_names: dict = None):
    deterministic = [x[0] for x in dataset]
    if remove_duplicates:
        deterministic = [remove_duplicates_trace(x) for x in deterministic]

    if activity_names is None:
        activity_names = {i: f'activity_{i}' for i in range(dataset.n_classes)}
    discovery_df = pd.DataFrame(
        {
            'concept:name': [activity_names[i.item()] for trace in deterministic for i in trace],
            'case:concept:name': [str(i) for i, trace in enumerate(deterministic) for _ in range(len(trace))]
        }
    )
    discovery_df.loc[:, 'order'] = discovery_df.groupby('case:concept:name').cumcount()
    discovery_df.loc[:, 'time:timestamp'] = pd.to_datetime(discovery_df['order'])

    return discovery_df


def prepare_df_cols_for_discovery(df):
    df_copy = df.copy()
    df_copy.loc[:, 'order'] = df_copy.groupby('case:concept:name').cumcount()
    df_copy.loc[:, 'time:timestamp'] = pd.to_datetime(df_copy['order'])

    return df_copy


def discover_process(
    dataset: TracesDataset, 
    process_discovery_method: str, 
    remove_duplicates=True, 
    activity_names: dict = None
):
    discovery_df = prepare_dataset_for_discovery(dataset, remove_duplicates, activity_names)
    try:
        process_discovery_method = DISCOVERY_METHODS[process_discovery_method]
    except KeyError:
        raise AttributeError(f"Unsupported discovery method: {process_discovery_method}")
    return process_discovery_method(discovery_df)


def get_petri_net_flow_matrix(process_model: pm4py.PetriNet, init_marking: pm4py.Marking,
                                                  final_marking: pm4py.Marking):
    pn_nx = pm4py.convert_petri_net_to_networkx(process_model, init_marking, final_marking)
    flow_matrix = nx.adjacency_matrix(pn_nx).todense()

    return flow_matrix
