"""Plots event sequences as graphs"""

import networkx as nx
import matplotlib.pyplot as plt
import re


def shorten_sequence(sequence):
    return [elem for i, elem in enumerate(sequence) if i == 0 or sequence[i - 1] != elem]


def plot_sequence(sequence, title="Event plot", output_folder=None):
    G = nx.DiGraph()
    items_per_row = 6

    short_sequence = shorten_sequence(sequence)
    nodes_with_labels = {i: l for i, l in enumerate(short_sequence)}
    nodes = nodes_with_labels.keys()

    for node, label in nodes_with_labels.items():
        wrapped_text = (
            re.sub("(.{10})", "\\1-\n", label, 0, re.DOTALL).strip("1-\n")
        )

        nodes_with_labels[node] = wrapped_text if len(wrapped_text) < 60 else wrapped_text[:60] + "..."

    G.add_nodes_from(nodes)

    for idx in range(len(short_sequence)):
        if idx != len(short_sequence) - 1:
            G.add_edge(idx, idx + 1)

    pos = dict(zip(nodes_with_labels.keys(),
                   [[(v % items_per_row), 100 - (int(v / items_per_row))] for v in range(len(short_sequence))]))

    plot_height = (int(((len(short_sequence) - 1)) / items_per_row) + 1) * 5
    plt.figure(figsize=(items_per_row * 3, plot_height), dpi=300)
    plt.title(title, fontsize=40)
    plt.tight_layout()
    nx.draw(
        G,
        pos=pos,
        labels=nodes_with_labels,
        font_size=17,
        font_weight="bold",
        node_size=15000,
        arrows=True,
        node_color="skyblue",
        node_shape="s",
        linewidths=5
    )

    if output_folder:
        plt.savefig(f"{output_folder}/{title}.png")
    else:
        plt.show()

    plt.close()
