from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union


def parent_path(path: str) -> Optional[str]:
    return path.rsplit('/', 1)[0] if '/' in path else None


def path_depth(path: str) -> int:
    return path.count('/')


@dataclass
class HierarchyLayout:
    node_ids: List[int]
    client_ids: List[int]
    edge_ids: List[int]
    cloud_id: int
    path_by_id: Dict[int, str]
    role_by_id: Dict[int, str]
    parent_by_id: Dict[int, Optional[int]]
    children_by_id: Dict[int, List[int]]
    edge_to_clients: Dict[int, List[int]]
    edge_to_edges: Dict[int, List[int]]
    terminal_edge_ids: List[int]
    root_edge_ids: List[int]



def infer_hierarchy_layout(node_ids: Sequence[int], cid2series: Dict[int, str]) -> HierarchyLayout:
    """Infer a recursive cloud/edge/client layout from hierarchy paths.

    A deduplicated hierarchy can be unbalanced.  For example, ``Total/CT``
    can be both an edge-level series and a leaf while ``Total/MASS`` still
    has third-level children.  Such leaf edges are recorded in
    ``terminal_edge_ids`` and are trained directly by their edge server.

    Every non-root internal node is an edge server.  An edge can therefore
    aggregate both direct leaf clients and lower-level edge servers.  A leaf
    directly below the cloud is represented by a terminal edge so that the
    cloud always receives one upload per root subtree.

    The historical function name is retained for API compatibility; the
    hierarchy may contain any number of levels greater than one.
    """
    node_ids = [int(x) for x in node_ids]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError('node_ids contains duplicate ids.')
    missing_ids = [cid for cid in node_ids if cid not in cid2series]
    if missing_ids:
        raise KeyError(f'cid2series is missing node ids {missing_ids}')

    path_by_id = {int(cid): str(cid2series[int(cid)]) for cid in node_ids}
    path_to_ids: Dict[str, List[int]] = {}
    for cid, path in path_by_id.items():
        path_to_ids.setdefault(path, []).append(cid)
    duplicate_paths = {path: ids for path, ids in path_to_ids.items() if len(ids) > 1}
    if duplicate_paths:
        raise ValueError(f'Hierarchy paths must be unique; duplicates={duplicate_paths}')

    depths = {cid: path_depth(path_by_id[cid]) for cid in node_ids}
    uniq_depths = sorted(set(depths.values()))

    if len(uniq_depths) < 2:
        raise ValueError(
            'This HFL implementation requires at least 2 hierarchy levels '
            f'(cloud plus descendants). Got depths={uniq_depths}.'
        )
    if uniq_depths != list(range(uniq_depths[0], uniq_depths[-1] + 1)):
        raise ValueError(f'Hierarchy levels must be consecutive. Got depths={uniq_depths}.')

    cloud_depth = uniq_depths[0]
    cloud_ids = [cid for cid in node_ids if depths[cid] == cloud_depth]

    if len(cloud_ids) != 1:
        raise ValueError(f'Expected exactly one cloud node, got {cloud_ids}')

    path_to_id = {path_by_id[cid]: cid for cid in node_ids}
    parent_by_id: Dict[int, Optional[int]] = {}
    children_by_id: Dict[int, List[int]] = {cid: [] for cid in node_ids}

    for cid in node_ids:
        ppath = parent_path(path_by_id[cid])
        pid = path_to_id.get(ppath) if ppath is not None else None
        parent_by_id[cid] = pid
        if pid is not None:
            children_by_id[pid].append(cid)

    for cid in node_ids:
        if cid not in cloud_ids and parent_by_id[cid] is None:
            raise ValueError(
                f'Node {cid} ({path_by_id[cid]}) has no hierarchy parent. '
                'All intermediate hierarchy paths must be present.'
            )

    cloud_id = cloud_ids[0]
    root_edge_ids = list(children_by_id[cloud_id])
    if not root_edge_ids:
        raise ValueError(f'Cloud node {cloud_id} has no descendants.')

    # Internal descendants are reusable edge servers.  A direct child of the
    # cloud is also an edge even when it is a leaf, allowing a uniform cloud
    # aggregation interface for unbalanced hierarchies.
    edge_ids = [
        cid for cid in node_ids
        if cid != cloud_id and (bool(children_by_id[cid]) or cid in root_edge_ids)
    ]
    client_ids = [
        cid for cid in node_ids
        if cid != cloud_id and cid not in edge_ids
    ]
    role_by_id: Dict[int, str] = {
        cid: ('cloud' if cid == cloud_id else ('edge' if cid in edge_ids else 'client'))
        for cid in node_ids
    }

    edge_to_clients = {
        eid: [cid for cid in children_by_id[eid] if role_by_id[cid] == 'client']
        for eid in edge_ids
    }
    edge_to_edges = {
        eid: [cid for cid in children_by_id[eid] if role_by_id[cid] == 'edge']
        for eid in edge_ids
    }
    terminal_edge_ids = [
        eid for eid in edge_ids
        if not edge_to_clients[eid] and not edge_to_edges[eid]
    ]

    for eid in edge_ids:
        pid = parent_by_id[eid]
        if pid != cloud_id and pid not in edge_ids:
            raise ValueError(
                f'Edge node {eid} ({path_by_id[eid]}) must have the cloud or another edge as parent; '
                f'got parent={pid}.')
    for cid in client_ids:
        pid = parent_by_id[cid]
        if pid not in edge_ids:
            raise ValueError(f'Client node {cid} ({path_by_id[cid]}) must have an edge parent; got {pid}.')

    return HierarchyLayout(
        node_ids=node_ids,
        client_ids=client_ids,
        edge_ids=edge_ids,
        cloud_id=cloud_id,
        path_by_id=path_by_id,
        role_by_id=role_by_id,
        parent_by_id=parent_by_id,
        children_by_id=children_by_id,
        edge_to_clients=edge_to_clients,
        edge_to_edges=edge_to_edges,
        terminal_edge_ids=terminal_edge_ids,
        root_edge_ids=root_edge_ids,
    )


def infer_three_level_layout(
    node_ids: Sequence[int], cid2series: Dict[int, str]
) -> HierarchyLayout:
    """Backward-compatible alias for :func:`infer_hierarchy_layout`."""
    return infer_hierarchy_layout(node_ids=node_ids, cid2series=cid2series)



def resolve_edge_trainable_flags(edge_ids: Sequence[int], spec: Union[bool, Sequence[int], Dict[int, bool]]) -> Dict[int, bool]:
    edge_ids = [int(x) for x in edge_ids]
    if isinstance(spec, bool):
        return {eid: bool(spec) for eid in edge_ids}
    if isinstance(spec, dict):
        return {eid: bool(spec.get(eid, False)) for eid in edge_ids}
    selected = set(int(x) for x in spec)
    return {eid: (eid in selected) for eid in edge_ids}



def resolve_edge_aggregation_rounds(edge_ids: Sequence[int], default_rounds: int, per_edge_spec: Optional[Dict[int, int]] = None) -> Dict[int, int]:
    edge_ids = [int(x) for x in edge_ids]
    rounds = {eid: max(1, int(default_rounds)) for eid in edge_ids}
    if per_edge_spec is None:
        return rounds
    for raw_eid, raw_rounds in per_edge_spec.items():
        eid = int(raw_eid)
        if eid not in rounds:
            raise ValueError(f'edge_agg_rounds_by_edge contains unknown edge id {eid}')
        rounds[eid] = max(1, int(raw_rounds))
    return rounds
