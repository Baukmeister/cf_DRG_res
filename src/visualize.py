"""Plots event sequences as graphs"""

import networkx as nx
import matplotlib.pyplot as plt
import re


def shorten_sequence(sequence):
    return [elem for i, elem in enumerate(sequence) if i == 0 or sequence[i - 1] != elem]

def plot_sequence(sequence):
    G = nx.DiGraph()

    short_sequence = shorten_sequence(sequence)
    nodes_with_labels = {i: l for i, l in enumerate(short_sequence)}
    nodes = nodes_with_labels.keys()

    for node,label in nodes_with_labels.items():
        nodes_with_labels[node] = (
            re.sub("(.{10})", "\\1-\n", label, 0, re.DOTALL).strip("1-\n")
        )

    G.add_nodes_from(nodes)

    for idx in range(len(short_sequence)):
        if idx != len(short_sequence)-1:
            G.add_edge(idx, idx+1)

    pos = dict(zip(nodes_with_labels.keys(), [[v, 1] for v in range(len(short_sequence))]))

    plt.figure(figsize=(len(short_sequence)*3, 5),dpi=300)
    nx.draw(
        G,
        pos=pos,
        labels=nodes_with_labels,
        node_size=8000,
        arrows=True,
        node_color="skyblue",
        node_shape="s",
        linewidths=5
    )
    plt.show()