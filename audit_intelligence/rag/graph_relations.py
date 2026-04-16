from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import pandas as pd


def build_transaction_graph(dataframe: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    if dataframe.empty:
        return graph

    for _, row in dataframe.iterrows():
        merchant = str(row.get("DESCRIPTION", "UNKNOWN")).strip() or "UNKNOWN"
        category = str(row.get("CATEGORY", "UNCATEGORIZED")).strip() or "UNCATEGORIZED"
        graph.add_node(merchant, node_type="merchant")
        graph.add_node(category, node_type="category")
        graph.add_edge(merchant, category, weight=graph.get_edge_data(merchant, category, {}).get("weight", 0) + 1)
    return graph

def graph_summary(graph: nx.Graph) -> Dict[str, object]:
    if graph.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "top_categories": []}

    category_counts = Counter()
    for node, data in graph.nodes(data=True):
        if data.get("node_type") == "category":
            category_counts[node] = graph.degree(node)
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "top_categories": category_counts.most_common(5),
    }
