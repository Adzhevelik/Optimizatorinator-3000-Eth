import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from typing import List, Optional


class BlockGraphBuilder:

    def __init__(self, temporal_hops: Optional[List[int]] = None, window_size: int = 48):
        self.temporal_hops = temporal_hops or [1, 5, 15, 30]
        self.window_size = window_size

    def build_graph(self, features: np.ndarray) -> Data:
        n = len(features)
        src, dst = [], []
        for hop in self.temporal_hops:
            for i in range(hop, n):
                src.append(i - hop)
                dst.append(i)
                src.append(i)
                dst.append(i - hop)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        x = torch.tensor(features, dtype=torch.float32)
        return Data(x=x, edge_index=edge_index)

    def build_sequence_graphs(self, feature_matrix: np.ndarray) -> List[Data]:
        graphs = []
        for end in range(self.window_size, len(feature_matrix) + 1):
            window = feature_matrix[end - self.window_size : end]
            graphs.append(self.build_graph(window))
        return graphs


class GasGNN(nn.Module):

    def __init__(self, in_channels: int, hidden: int = 64, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, dropout=dropout, concat=False)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.norm2 = nn.LayerNorm(hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        return h

    def encode_last(self, data: Data) -> torch.Tensor:
        emb = self.forward(data.x, data.edge_index)
        return emb[-1]


class GraphFeatureExtractor:

    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        window_size: int = 48,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.hidden = hidden
        self.builder = BlockGraphBuilder(window_size=window_size)
        self.gnn = GasGNN(in_channels=in_channels, hidden=hidden).to(self.device)
        self.is_fitted = False

    def _pretrain(self, feature_matrix: np.ndarray, log_prices: np.ndarray, epochs: int = 15):
        graphs = self.builder.build_sequence_graphs(feature_matrix)
        targets = torch.tensor(
            log_prices[self.builder.window_size - 1 :], dtype=torch.float32
        ).to(self.device)

        proj = nn.Linear(self.hidden, 1).to(self.device)
        optimizer = torch.optim.AdamW(
            list(self.gnn.parameters()) + list(proj.parameters()),
            lr=1e-3,
            weight_decay=1e-4,
        )

        self.gnn.train()
        for _ in range(epochs):
            for i, graph in enumerate(graphs):
                graph = graph.to(self.device)
                emb = self.gnn.encode_last(graph)
                pred = proj(emb).squeeze()
                loss = nn.functional.huber_loss(pred, targets[i])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.is_fitted = True

    def extract(self, feature_matrix: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("GraphFeatureExtractor must be fitted before extract()")

        self.gnn.eval()
        graphs = self.builder.build_sequence_graphs(feature_matrix)
        embeddings = []

        with torch.no_grad():
            for graph in graphs:
                graph = graph.to(self.device)
                emb = self.gnn.encode_last(graph).cpu().numpy()
                embeddings.append(emb)

        padding = np.zeros((self.builder.window_size - 1, self.hidden))
        return np.vstack([padding, np.array(embeddings)])
    