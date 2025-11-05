from copyreg import pickle
import random
from utils import collect_queries
import json
from typing import List
import torch
import networkx as nx
from sentence_transformers import SentenceTransformer
from typing import List, Union
import matplotlib.pyplot as plt


class GraphSampler:
    def __init__(
        self,
        model_name:    str,
        info:           json,
        queries:       List[str],
        target_query:  str = "",
        low:           float = 0.3,
        high:          float = 0.8
    ):
        # Include target_query in queries
        self.queries = queries
        self.target_query = target_query
        self.model_name = model_name
        self.graph_info = info
        self.low = low
        self.high = high

        # lazy‐init
        self._embedder = None
        self._embeds = None
        self._graph = None
        self._ensure_graph()

    def _ensure_embeddings(self):
        if self._embeds is None:
            self._embedder = SentenceTransformer(self.model_name, device="cpu")
            self._embeds = self._embedder.encode(
                self.queries,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            self._target_embeds = self._embedder.encode(
                self.target_query,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            self._sims = self._embedder.similarity(self._embeds, self._embeds)
            self._target_sims = self._embedder.similarity(
                self._embeds, self._target_embeds)

    def _ensure_graph(self):
        if self._graph is None:
            self._ensure_embeddings()
            n = len(self.queries)
            G = nx.Graph()
            G.add_nodes_from(range(n))
            for i in range(n):
                G.nodes[i]["sim_with_target"] = self._target_sims[i].item()

            if self.graph_info:
                G.graph["info"] = self.graph_info

            for i in range(n):
                for j in range(i + 1, n):
                    sim = self._sims[i][j].item()
                    if self.low < sim < self.high:
                        G.add_edge(i, j, weight=sim)

            self.target_neighbors = [i for i in range(
                n) if G.nodes[i]["sim_with_target"] > self.low]
            self._graph = G

            import numpy as np
            degrees = [deg for _, deg in G.degree()]
            for node, neighbors in G.adjacency():
                print(f"{node} - {self.queries[node]}: {list(neighbors)}")
            print(f"- Min degree: {min(degrees)}")
            print(f"- Max degree: {max(degrees)}")
            print(f"- Mean degree: {np.mean(degrees):.2f}")
            print(f"- Median degree: {np.median(degrees):.2f}")

    def sample(
        self,
        sample_strategy:       str,
        sample_distribution:    str,
        num_queries: int = 5,
        step_length:  int = 2
    ) -> List[str]:
        """
        strategy:
          - "random_walk": walk from start_query for walk_length steps
          - "random_node": pick num_samples nodes at random
        """
        if sample_strategy == "graph_path_constraint" or sample_strategy == "graph_path_vanilla":
            if sample_strategy == "graph_path_constraint":
                idx = random.choice(self.target_neighbors)
            else:
                idx = random.choice(list(self._graph.nodes))
            path = [idx]
            for _ in range(num_queries-1):
                idx_in_length_l = [idx]
                for l in range(step_length):
                    if isinstance(self._graph, nx.DiGraph):
                        avail_neighbors = [n for idx_l in idx_in_length_l for n in self._graph.predecessors(
                            idx_l) if n not in path]
                    else:
                        avail_neighbors = [n for idx_l in idx_in_length_l for n in self._graph.neighbors(
                            idx_l) if n not in path]
                    idx_in_length_l = avail_neighbors
                if not avail_neighbors:
                    break
                idx = random.choice(avail_neighbors)
                path.append(idx)
            if sample_strategy == "graph_path_vanilla":
                return path, [self.queries[i] for i in path]
            else:
                return reversed(path), [self.queries[i] for i in reversed(path)]

        elif sample_strategy == "random_node":
            path = random.sample(self._graph.nodes, num_queries)
            return path, [self.queries[i] for i in path]

        else:
            raise ValueError(
                f"Unknown sample strategy {sample_strategy!r}")

    def sample_multi(
        self,
        sample_strategy: str = "random_node",
        sample_distribution: str = "uniform",
        num_samples: int = 5,
        num_queries: int = 5,
        step_length: int = 1
    ) -> List[List[str]]:
        """
        Sample multiple sets of queries.
        """
        samples_idxs = []
        samples = []
        for _ in range(num_samples):
            for _ in range(5):
                path_idxs, path = self.sample(
                    sample_strategy=sample_strategy,
                    sample_distribution=sample_distribution,
                    num_queries=num_queries,
                    step_length=step_length
                )
                if len(path) == num_queries:
                    samples_idxs.append(path_idxs)
                    samples.append(path)
                    break
        return samples
