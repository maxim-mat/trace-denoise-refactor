from typing import Iterable, Tuple, Dict
from pm4py.objects.petri_net.utils import reachability_graph
import networkx as nx
import torch
# from node2vec import Node2Vec
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.utils import from_networkx
import torch_geometric
import pm4py
from src.utils.config import Config
from pm4py.objects.petri_net.obj import PetriNet

def get_onehot_features(graph: nx.Graph) -> Iterable:
    node_names = np.array([n[0] for n in graph.nodes(data=True)]).reshape(-1, 1)
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    return one_hot_encoder.fit_transform(node_names)


def get_index_features(graph: nx.Graph) -> Iterable:
    return [[x] for x in reversed(np.arange(graph.number_of_nodes()))]


def get_feature_collections(graph: nx.Graph) -> Iterable[Iterable]:
    # n2v_features = get_node2vec_features(graph, cfg)
    index_features = get_index_features(graph)
    return [index_features]


def add_features_to_graph(graph: nx.Graph, feature_collections: Iterable[Iterable]) -> None:
    """
    modifies nodes in place with features under the attribute 'x'
    :param feature_collections:
    :param graph: graph to modify
    :return:
    """
    for node, *feature_collection in zip(graph.nodes(data=True), *feature_collections):
        # flatten each feature vector to single array
        node[1]['x'] = np.array([x for feature in feature_collection for x in feature])


def prepare_process_model_for_gnn(process_model: pm4py.PetriNet, init_marking: pm4py.Marking,
                                  final_marking: pm4py.Marking) -> torch_geometric.data.Data:
    model_nx = pm4py.convert_petri_net_to_networkx(process_model, init_marking, final_marking)
    feature_collections = get_feature_collections(model_nx)
    add_features_to_graph(model_nx, feature_collections)
    data = from_networkx(model_nx)
    return data


def _node_type_from_pm4py_obj(node_obj, node_attrs: dict) -> str:
    """
    Determine hetero node type: 'place' or 'transition'.
    Works when nx node keys are pm4py Place/Transition objects (typical in pm4py conversion),
    and also supports a fallback to nx node attributes if needed.
    """
    # Most reliable: the pm4py object class
    if isinstance(node_obj, PetriNet.Place):
        return "place"
    if isinstance(node_obj, PetriNet.Transition):
        return "transition"

    # Fallbacks if your nx graph stores something else as node keys
    t = node_attrs['attr'].get("type", None)
    if t in {"place", "transition"}:
        return t

    raise TypeError(
        f"Could not infer node type for node={node_obj!r}. "
        f"Expected PetriNet.Place/PetriNet.Transition or node_attrs['type'] in {{'place','transition'}}."
    )


def heterodata_from_petri_nx(model_nx: nx.DiGraph) -> HeteroData:
    """
    Expects:
      - model_nx nodes have attribute 'x' (np array or list) already attached
      - model_nx represents a Petri net (bipartite) with Place/Transition nodes

    Returns:
      HeteroData with:
        data['place'].x: [num_places, feat_dim]
        data['transition'].x: [num_transitions, feat_dim]
        data['place','to','transition'].edge_index
        data['transition','to','place'].edge_index
    """
    data = HeteroData()

    # 1) Build per-type index maps in a deterministic order
    place_nodes = []
    trans_nodes = []

    for n, attrs in model_nx.nodes(data=True):
        ntype = _node_type_from_pm4py_obj(n, attrs)
        if "x" not in attrs:
            raise KeyError(f"Node {n!r} is missing required feature attribute 'x'.")
        if ntype == "place":
            place_nodes.append(n)
        else:
            trans_nodes.append(n)

    place_id: Dict[object, int] = {n: i for i, n in enumerate(place_nodes)}
    trans_id: Dict[object, int] = {n: i for i, n in enumerate(trans_nodes)}

    # 2) Stack node features per type
    def _stack_features(nodes):
        xs = []
        for i, n in enumerate(nodes):
            x = model_nx.nodes[n]["x"]
            x = np.asarray(x).reshape(-1)  # ensure 1D
            xs.append([i])
        if len(xs) == 0:
            return torch.empty((0, 0))
        return torch.from_numpy(np.stack(xs, axis=0))

    data["place"].x = _stack_features(place_nodes)
    data["transition"].x = _stack_features(trans_nodes)

    # 3) Build hetero edge_index tensors
    pt_src, pt_dst = [], []
    tp_src, tp_dst = [], []

    for u, v in model_nx.edges():
        u_type = _node_type_from_pm4py_obj(u, model_nx.nodes[u])
        v_type = _node_type_from_pm4py_obj(v, model_nx.nodes[v])

        if u_type == "place" and v_type == "transition":
            pt_src.append(place_id[u])
            pt_dst.append(trans_id[v])
        elif u_type == "transition" and v_type == "place":
            tp_src.append(trans_id[u])
            tp_dst.append(place_id[v])
        else:
            # Petri nets should be bipartite; if you see this, something upstream is off
            raise ValueError(f"Non-bipartite edge detected: {u_type} -> {v_type} for edge ({u!r}, {v!r})")

    def _edge_index(src, dst) -> torch.Tensor:
        if len(src) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor([src, dst], dtype=torch.long)

    data[("place", "to", "transition")].edge_index = _edge_index(pt_src, pt_dst)
    data[("transition", "to", "place")].edge_index = _edge_index(tp_src, tp_dst)

    # Optional: keep maps for debugging / alignment
    data.place_nodes = place_nodes
    data.transition_nodes = trans_nodes

    return data


def prepare_process_model_for_heterognn(
    process_model: pm4py.PetriNet,
    init_marking: pm4py.Marking,
    final_marking: pm4py.Marking,
    cfg: Config
) -> HeteroData:
    model_nx = pm4py.convert_petri_net_to_networkx(process_model, init_marking, final_marking)

    feature_collections = get_feature_collections(model_nx, cfg)
    add_features_to_graph(model_nx, feature_collections)

    hetero = heterodata_from_petri_nx(model_nx)
    return hetero
