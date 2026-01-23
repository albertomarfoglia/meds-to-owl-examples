"""
metrics_utils.py

Unified utilities for computing tabular and RDF graph metrics
under a standard output folder structure.

Author: Alberto Marfoglia
"""

from pathlib import Path
from typing import Dict, Any, List, TypedDict
from collections import Counter
import statistics
import json

import pandas as pd
from rdflib import Graph, Namespace, RDF, URIRef, BNode, Literal

from .synthetic_generator import EVENT_COLUMNS

# =============================================================================
# Tabular metrics (CSV + Parquet)
# =============================================================================

def _count_neurovasc_events_from_csv(df: pd.DataFrame) -> Dict[str, int]:
    """
    Compute event metrics from syn_data.csv.

    Static events are defined as:
        (#columns in CSV) - (#event columns)
    """
    n_rows = len(df)
    n_features = len(df.columns)
    n_timed_events = (df[EVENT_COLUMNS] > -1).to_numpy().sum()
    n_static_events = n_rows * (n_features - len(EVENT_COLUMNS))

    return {
        "n_rows": n_rows,
        "n_features": n_features,
        "total_events":  n_timed_events + n_static_events,
    }


def _count_rows_in_parquet_dir(path: Path) -> int:
    """
    Count total rows across all .parquet files in a directory.
    """
    if not path.exists():
        return 0

    total = 0
    for parquet_file in sorted(path.glob("*.parquet")):
        df = pd.read_parquet(parquet_file)
        total += len(df)
    return total



def _count_neurovasc_intermediate_events(base: Path) -> Dict[str, int]:
    """
    Compute event metrics from standard intermediate parquet layout.
    """
    df_patients = pd.read_parquet(base / "patients.parquet")
    df_administrations = pd.read_parquet(base / "administrations.parquet")
    df_procedures = pd.read_parquet(base / "procedures.parquet")

    n_patients =  len(df_patients["subject_id"].unique())
    n_administrations = len(df_administrations)
    n_procedures = len(df_procedures)
    n_static_events = n_patients * len(df_patients.drop(columns=["subject_id"]).columns)

    return {
        "n_patients": n_patients,
        "n_demographics": n_static_events,
        "n_administrations": n_administrations,
        "n_procedures": n_procedures,
        "total_events": n_static_events + n_administrations + n_procedures
    }


def count_MEDS_splits(
    output_path: str,
) -> Dict[str, int]:
    """
    Compute dataset split sizes from standard output/data layout.
    """
    base = Path(output_path) / "data"

    train = _count_rows_in_parquet_dir(base / "train")
    held_out = _count_rows_in_parquet_dir(base / "held_out")
    tuning = _count_rows_in_parquet_dir(base / "tuning")

    return {
        "train": train,
        "held_out": held_out,
        "tuning": tuning,
        "total_events": train + held_out + tuning,
    }

# =============================================================================
# RDF graph metrics
# =============================================================================
from collections import Counter, defaultdict

MEDS = Namespace("https://teamheka.github.io/meds-ontology#")


def _build_graph_index(g: Graph):
    """
    Single pass over all triples; return structures used later:
      - subject_triple_counts: Counter(subject -> number of outgoing triples)
      - adjacency: dict(subject -> list of object URIRefs)   (only URIRefs)
      - predicate_counter: Counter of predicates
      - sets: subjects_set, predicates_set, objects_set
      - node-type sets: iri_nodes, bnode_nodes, literal_nodes
      - class_instances_map: dict(class_uri -> list of subjects)
    """
    subject_triple_counts = Counter()
    adjacency = defaultdict(list)
    predicate_counter = Counter()

    subjects_set = set()
    predicates_set = set()
    objects_set = set()

    iri_nodes = set()
    bnode_nodes = set()
    literal_nodes = set()

    class_instances_map = defaultdict(list)

    for s, p, o in g.triples((None, None, None)):
        # global sets
        subjects_set.add(s)
        predicates_set.add(p)
        objects_set.add(o)

        # predicate frequency and per-subject triple count
        predicate_counter[p] += 1
        subject_triple_counts[s] += 1

        # node kinds
        if isinstance(s, URIRef):
            iri_nodes.add(s)
        elif isinstance(s, BNode):
            bnode_nodes.add(s)
        elif isinstance(s, Literal):
            literal_nodes.add(s)

        if isinstance(p, URIRef):
            iri_nodes.add(p)
        elif isinstance(p, BNode):
            bnode_nodes.add(p)
        elif isinstance(p, Literal):
            literal_nodes.add(p)

        if isinstance(o, URIRef):
            iri_nodes.add(o)
            # adjacency only store URIRef neighbors (matches original behavior)
            adjacency[s].append(o)
        elif isinstance(o, BNode):
            bnode_nodes.add(o)
        elif isinstance(o, Literal):
            literal_nodes.add(o)

        # collect rdf:type instances
        if p == RDF.type:
            class_instances_map[o].append(s)

    return {
        "subject_triple_counts": subject_triple_counts,
        "adjacency": dict(adjacency),
        "predicate_counter": predicate_counter,
        "subjects_set": subjects_set,
        "predicates_set": predicates_set,
        "objects_set": objects_set,
        "iri_nodes": iri_nodes,
        "bnode_nodes": bnode_nodes,
        "literal_nodes": literal_nodes,
        "class_instances_map": dict(class_instances_map),
    }


def _count_recursive_using_index(subject, adjacency, subject_triple_counts):
    """
    Iterative DFS using the pre-built adjacency and subject_triple_counts.
    Counts each subject's outgoing triples once. Avoids repeated graph queries.
    """
    stack = [subject]
    visited = set()
    total = 0

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)

        # add the number of outgoing triples for this node (0 if none)
        total += subject_triple_counts.get(node, 0)

        # push neighbors (only URIRef neighbors were stored)
        for neigh in adjacency.get(node, ()):
            if neigh not in visited:
                stack.append(neigh)

    return total


def _compute_MEDS_graph_stats(g: Graph, event_triple_mode: str = "direct") -> Dict[str, Any]:
    idx = _build_graph_index(g)

    predicate_counter: Counter = idx["predicate_counter"]
    subject_triple_counts: Counter = idx["subject_triple_counts"]
    adjacency: Dict = idx["adjacency"]

    stats: Dict[str, Any] = {}

    # basic counts (len(g) is efficient)
    stats["total_triples"] = len(g)
    stats["distinct_subjects"] = len(idx["subjects_set"])
    stats["distinct_predicates"] = len(idx["predicates_set"])
    stats["distinct_objects"] = len(idx["objects_set"])

    stats.update({
        "distinct_iris": len(idx["iri_nodes"]),
        "distinct_bnodes": len(idx["bnode_nodes"]),
        "distinct_literals": len(idx["literal_nodes"]),
    })

    # distinct resources: subjects union objects that are URIRef/BNode
    resources = set(idx["subjects_set"]) | {
        o for o in idx["objects_set"] if isinstance(o, (URIRef, BNode))
    }
    stats["distinct_resources"] = len(resources)

    # classes of interest
    meds_classes = {
        "Event": MEDS.Event,
        "Subject": MEDS.Subject,
        "Code": MEDS.Code,
        "LabelSample": MEDS.LabelSample,
        "SubjectSplit": MEDS.SubjectSplit,
        "ValueModality": MEDS.ValueModality,
        "DatasetMetadata": MEDS.DatasetMetadata,
    }

    # convert class_instances_map (built during indexing) into the named dict
    class_instances_map = idx["class_instances_map"]
    class_instances = {
        name: class_instances_map.get(uri, [])
        for name, uri in meds_classes.items()
    }

    stats["class_counts"] = {
        name: len(instances)
        for name, instances in class_instances.items()
    }

    # compute triples per event
    event_nodes = class_instances.get("Event", [])
    triples_per_event = []
    for ev in event_nodes:
        if event_triple_mode == "direct":
            triples_per_event.append(subject_triple_counts.get(ev, 0))
        elif event_triple_mode == "recursive":
            triples_per_event.append(_count_recursive_using_index(ev, adjacency, subject_triple_counts))
        else:
            raise ValueError(f"Unknown mode: {event_triple_mode}")

    stats["n_events"] = len(triples_per_event)

    if triples_per_event:
        stats.update({
            "triples_per_event_mean": round(statistics.mean(triples_per_event), ndigits=2),
            "triples_per_event_median": statistics.median(triples_per_event),
            "triples_per_event_min": min(triples_per_event),
            "triples_per_event_max": max(triples_per_event),
            "triples_per_event_pstdev": round(statistics.pstdev(triples_per_event), ndigits=2),
            "triples_per_event_stdev": round(statistics.stdev(triples_per_event) if len(triples_per_event) > 1 else 0.0, ndigits=2),
        })
    else:
        stats.update({
            "triples_per_event_mean": 0.0,
            "triples_per_event_median": 0.0,
            "triples_per_event_min": 0,
            "triples_per_event_max": 0,
            "triples_per_event_pstdev": 0.0,
            "triples_per_event_stdev": 0.0,
        })

    stats["distinct_predicate_frequencies"] = {str(k): v for k, v in predicate_counter.items()}
    #stats["top_10_predicates"] = [(str(p), c) for p, c in predicate_counter.most_common(10)]

    return stats


# =============================================================================
# Orchestration
# =============================================================================

class TabularInput(TypedDict):
    data: pd.DataFrame
    timed_columns: List[str]


def compute_MEDS_graph_metrics_for_neurovasc(
    MEDS_ETL_output_path: str,
    graph: Graph,
    tabular_data: pd.DataFrame,
    MEDS_intermediate: Path,
    event_triple_mode: str = "direct",
) -> Dict[str, Any]:
    """
    Collect all metrics assuming standard folder structure.
    """

    metrics = compute_MEDS_graph_metrics(MEDS_ETL_output_path, graph, event_triple_mode)

    tabular_metrics = _count_neurovasc_events_from_csv(tabular_data)

    intermediate_metrics = _count_neurovasc_intermediate_events(base = MEDS_intermediate)

    metrics["consistency_checks"]["tabular_vs_intermediate_match"] = bool(
        tabular_metrics["total_events"] == intermediate_metrics["total_events"]
    )
    metrics["consistency_checks"]["intermediate_vs_splits_match"] = bool(
        intermediate_metrics["total_events"] == metrics["MEDS_METRICS"]["total_events"]
    )

    metrics["derived_metrics"]["events_per_patient"] = round(
        intermediate_metrics["total_events"] / intermediate_metrics["n_patients"], 
        ndigits=2
    ),

    metrics["TABULAR_METRICS"] = tabular_metrics
    metrics["INTERMEDIATE_METRICS"] = intermediate_metrics
    return metrics


def compute_MEDS_graph_metrics(
    MEDS_ETL_output_path: str,
    graph: Graph,
    event_triple_mode: str = "direct",
) -> Dict[str, Any]:
    """
    Collect all metrics assuming standard folder structure.
    """
    meds_metrics = count_MEDS_splits(MEDS_ETL_output_path)

    graph_metrics = _compute_MEDS_graph_stats(graph, event_triple_mode)

    consistency_checks: Dict[str, bool] = {
        "graph_event_count_match":
            bool(meds_metrics["total_events"] == graph_metrics["class_counts"]["Event"]),
    }

    derived_metrics: Dict[str, float] = {
        "avg_triples_per_event":
            round(graph_metrics["triples_per_event_mean"], ndigits=2),
        "graph_density":
            round(graph_metrics["total_triples"] / graph_metrics["distinct_resources"], ndigits=2),
    }

    return {
        "MEDS_METRICS": meds_metrics,
        "GRAPH_METRICS": graph_metrics,
        "consistency_checks": consistency_checks,
        "derived_metrics": derived_metrics,
    }


# =============================================================================
# Serialization
# =============================================================================

import numpy as np

def save_stats_json(stats: Dict[str, Any], path: str) -> None:
    """
    Save metrics dict as formatted JSON.
    Creates parent directories if they do not exist.
    """
    ppath = Path(path)

    # Ensure parent directory exists
    if ppath.parent:
        ppath.parent.mkdir(parents=True, exist_ok=True)

    with ppath.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x))

