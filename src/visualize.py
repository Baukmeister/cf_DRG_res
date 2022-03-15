"""Plots event sequences as graphs"""

import networkx as nx
import matplotlib.pyplot as plt
import re


def plot_sequence(sequence):
    G = nx.DiGraph()

    nodes_with_labels = {i: l for i, l in enumerate(sequence)}
    nodes = nodes_with_labels.keys()

    for node,label in nodes_with_labels.items():
        nodes_with_labels[node] = (
            re.sub("(.{10})", "\\1-\n", label, 0, re.DOTALL)
        )

    G.add_nodes_from(nodes)

    for idx in range(len(sequence)):
        if idx != len(sequence)-1:
            G.add_edge(idx, idx+1)

    pos = dict(zip(nodes_with_labels.keys(), [[v, 1] for v in range(len(sequence))]))

    plt.figure(figsize=(len(sequence)*3, 5))
    nx.draw(
        G,
        pos=pos,
        labels=nodes_with_labels,
        node_size=3000,
        arrows=True,
        node_color="skyblue",
        node_shape="s",
        linewidths=100)
    plt.show()