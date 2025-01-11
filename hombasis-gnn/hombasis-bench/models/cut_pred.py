import torch
import torch.nn as nn
from torch_geometric.nn.models import MLP
from torch_geometric.nn.encoding import PositionalEncoding
from torch_geometric.utils import scatter
import torch.nn.functional as F
from models.layers import GCNLayer, GATLayer, GINLayer, MLPReadout, GINLayerSig


class GINCutPred(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        count_dim: int,
        # pe_dim: int,
        num_layers: int,
        batch_norm: bool,
        residual: bool,
        readout: str,
        max_nodes: int = 121,  # max number of nodes in dataset graphs
        mode: str = "node",
    ):

        super(GINCutPred, self).__init__()
        self.hidden_dim = hidden_dim
        self.count_dim = count_dim
        # self.pe_dim = pe_dim
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.readout = readout
        self.max_nodes = max_nodes
        self.mode = mode

        # 1-hot encode + linear node features
        self.encoder = nn.Embedding(
            num_embeddings=121, embedding_dim=hidden_dim  # max num of nodes
        )

        # encode homcounts in 2-layer MLP
        if count_dim > 0:
            self.count_encoder = MLP(
                in_channels=count_dim,
                hidden_channels=count_dim,
                out_channels=count_dim,
                num_layers=2,
            )
            concat_feature_dim = hidden_dim + count_dim
        else:
            concat_feature_dim = hidden_dim

        # GIN message passing layers
        self.convs = nn.ModuleList(
            [
                GINLayer(
                    hidden_dim, hidden_dim, batch_norm=batch_norm, residual=residual
                )
                for _ in range(self.num_layers - 1)
            ]
        )
        self.convs.insert(
            0,
            GINLayer(
                concat_feature_dim, hidden_dim, batch_norm=batch_norm, residual=residual
            ),
        )

        # decoder
        self.decoder = MLP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=1,
            num_layers=2,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, counts, use_counts, batch):
        # encode features
        embeds_h = self.encoder(x)
        embeds_h = torch.squeeze(embeds_h)

        if use_counts:
            count_h = self.count_encoder(counts)
            h = torch.cat((embeds_h, count_h), dim=1)

        else:
            h = embeds_h

        for layer in self.convs:
            h = layer(x=h, edge_index=edge_index)

        # decoding step
        if self.mode == "node":

            logits = self.decoder(h)  # shape [N, 1]
            preds = self.sigmoid(logits)  # shape [N, 1]

            # want to regroup these node predictions into shape [B, 121]
            num_graphs = batch.max().item() + 1  # B
            padded_outs = []

            for g in range(num_graphs):
                # Indices of nodes that belong to graph g
                node_indices = (batch == g).nonzero(as_tuple=True)[
                    0
                ]  # shape [num_nodes_in_graph_g]
                g_preds = preds[node_indices].squeeze(-1)

                # Pad if needed
                num_nodes_g = g_preds.size(0)
                if num_nodes_g < self.max_nodes:
                    pad_size = self.max_nodes - num_nodes_g
                    g_preds = F.pad(g_preds, (0, pad_size), value=0.0)
                elif num_nodes_g > self.max_nodes:  # safety
                    g_preds = g_preds[: self.max_nodes]
                padded_outs.append(g_preds)

            out_batched = torch.stack(padded_outs, dim=0)
            return out_batched  # shape [B, 121]

        elif self.mode == "edge":

            edge_src = edge_index[0]
            edge_dst = edge_index[1]
            # pool node embeddings at endpoints u, v
            edge_embs = torch.maximum(h[edge_src], h[edge_dst])  # shape [E, hidden_dim]

            logits = self.decoder(edge_embs)  # shape [E, 1]
            preds = self.sigmoid(logits).squeeze(-1)  # shape [E]

            edge_graph = batch[edge_src]  # shape [E]

            num_graphs = batch.max().item() + 1  # graphs in batch
            max_edges = 304
            padded_outs = []

            for g in range(num_graphs):
                edge_indices = (edge_graph == g).nonzero(as_tuple=True)[0]
                g_preds = preds[edge_indices]

                # Pad
                num_edges_g = g_preds.size(0)
                if num_edges_g < max_edges:
                    pad_size = max_edges - num_edges_g
                    g_preds = F.pad(g_preds, (0, pad_size), value=0.0)
                elif num_edges_g > max_edges:
                    g_preds = g_preds[:max_edges]
                padded_outs.append(g_preds)

            out_batched = torch.stack(padded_outs, dim=0)  # [B, 304]
            return out_batched

        else:
            raise ValueError(
                f"Mode {self.mode} is not a recognized mode! Only 'edge' or 'node' supported."
            )


class GCNCutPred(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        count_dim: int,
        num_layers: int,
        batch_norm: bool,
        residual: bool,
        readout: str,
        max_nodes: int = 121,  # max number of nodes in dataset graphs
        mode: str = "node",
    ):

        super(GCNCutPred, self).__init__()
        self.hidden_dim = hidden_dim
        self.count_dim = count_dim
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.readout = readout
        self.max_nodes = max_nodes
        self.mode = mode

        # 1-hot encode + linear node features
        self.encoder = nn.Embedding(num_embeddings=121, embedding_dim=hidden_dim)

        # encode homcounts in 2-layer MLP
        if count_dim > 0:
            self.count_encoder = MLP(
                in_channels=count_dim,
                hidden_channels=count_dim,
                out_channels=count_dim,
                num_layers=2,
            )

        concat_feature_dim = hidden_dim + count_dim

        # GCN message passing layers
        self.convs = nn.ModuleList(
            [
                GCNLayer(
                    hidden_dim, hidden_dim, batch_norm=batch_norm, residual=residual
                )
                for _ in range(self.num_layers - 1)
            ]
        )
        self.convs.insert(
            0,
            GCNLayer(
                concat_feature_dim, hidden_dim, batch_norm=batch_norm, residual=residual
            ),
        )

        # decoder
        self.decoder = MLP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=1,
            num_layers=2,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, counts, use_counts, batch):
        # encode features
        embeds_h = self.encoder(x)
        embeds_h = torch.squeeze(embeds_h)

        if use_counts:
            count_h = self.count_encoder(counts)
            h = torch.cat((embeds_h, count_h), dim=1)
        else:
            h = embeds_h

        # model step
        for conv in self.convs:
            h = conv(x=h, edge_index=edge_index)

        # decoding step
        if self.mode == "node":

            logits = self.decoder(h)  # shape [N, 1]
            preds = self.sigmoid(logits)  # shape [N, 1]

            # want to regroup these node predictions into shape [B, 121]
            num_graphs = batch.max().item() + 1  # B
            padded_outs = []

            for g in range(num_graphs):
                # Indices of nodes that belong to graph g
                node_indices = (batch == g).nonzero(as_tuple=True)[
                    0
                ]  # shape [num_nodes_in_graph_g]
                g_preds = preds[node_indices].squeeze(-1)

                # Pad if needed
                num_nodes_g = g_preds.size(0)
                if num_nodes_g < self.max_nodes:
                    pad_size = self.max_nodes - num_nodes_g
                    g_preds = F.pad(g_preds, (0, pad_size), value=0.0)
                elif num_nodes_g > self.max_nodes:  # safety
                    g_preds = g_preds[: self.max_nodes]
                padded_outs.append(g_preds)

            out_batched = torch.stack(padded_outs, dim=0)
            return out_batched  # shape [B, 121]

        elif self.mode == "edge":

            edge_src = edge_index[0]
            edge_dst = edge_index[1]
            # pool node embeddings at endpoints u, v
            edge_embs = torch.maximum(h[edge_src], h[edge_dst])  # shape [E, hidden_dim]

            logits = self.decoder(edge_embs)  # shape [E, 1]
            preds = self.sigmoid(logits).squeeze(-1)  # shape [E]

            edge_graph = batch[edge_src]  # shape [E]

            num_graphs = batch.max().item() + 1  # graphs in batch
            max_edges = 304
            padded_outs = []

            for g in range(num_graphs):
                edge_indices = (edge_graph == g).nonzero(as_tuple=True)[0]
                g_preds = preds[edge_indices]

                # Pad
                num_edges_g = g_preds.size(0)
                if num_edges_g < max_edges:
                    pad_size = max_edges - num_edges_g
                    g_preds = F.pad(g_preds, (0, pad_size), value=0.0)
                elif num_edges_g > max_edges:
                    g_preds = g_preds[:max_edges]
                padded_outs.append(g_preds)

            out_batched = torch.stack(padded_outs, dim=0)  # [B, 304]
            return out_batched

        else:
            raise ValueError(
                f"Mode {self.mode} is not a recognized mode! Only 'edge' or 'node' supported."
            )


class GATCutPred(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        hidden_out_dim: int,
        count_dim: int,
        num_layers: int,
        num_heads: int,
        batch_norm: bool,
        residual: bool,
        readout: str,
        max_nodes: int = 121,  # max number of nodes in dataset graphs
        mode: str = "node",
    ):

        super(GATCutPred, self).__init__()
        self.hidden_dim = hidden_dim
        self.count_dim = count_dim
        self.num_layers = num_layers
        self.readout = readout
        self.out_dim = hidden_out_dim
        self.max_nodes = max_nodes
        self.mode = mode

        head_hidden_dim = hidden_dim * num_heads
        # 1-hot encode + linear node features
        self.encoder = nn.Embedding(
            num_embeddings=121,
            embedding_dim=head_hidden_dim,
        )

        # encode homcounts in 2-layer MLP
        if count_dim > 0:
            self.count_encoder = MLP(
                in_channels=count_dim,
                hidden_channels=count_dim,
                out_channels=count_dim,
                num_layers=2,
            )

        concat_feature_dim = head_hidden_dim + count_dim

        self.prepare_gat = nn.Linear(concat_feature_dim, head_hidden_dim)

        # GAT message passing layers
        self.convs = nn.ModuleList(
            [
                GATLayer(
                    head_hidden_dim,
                    hidden_dim,
                    num_heads,
                    batch_norm=batch_norm,
                    residual=residual,
                )
                for _ in range(self.num_layers - 1)
            ]
        )
        self.convs.append(
            GATLayer(
                head_hidden_dim,
                hidden_out_dim,
                num_heads=1,
                batch_norm=batch_norm,
                residual=residual,
            )
        )

        # decoder
        self.decoder = MLP(
            in_channels=hidden_out_dim,
            hidden_channels=hidden_out_dim,
            out_channels=1,
            num_layers=2,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, counts, use_counts, batch):
        # encode features
        embeds_h = self.encoder(x)
        embeds_h = torch.squeeze(embeds_h)

        if use_counts:
            count_h = self.count_encoder(counts)
            h = torch.cat((embeds_h, count_h), dim=1)
            h = self.prepare_gat(h)
        else:
            h = embeds_h

        # model step
        for conv in self.convs:
            h = conv(x=h, edge_index=edge_index)

        # decoding step
        if self.mode == "node":

            logits = self.decoder(h)  # shape [N, 1]
            preds = self.sigmoid(logits)  # shape [N, 1]

            # want to regroup these node predictions into shape [B, 121]
            num_graphs = batch.max().item() + 1  # B
            padded_outs = []

            for g in range(num_graphs):
                # Indices of nodes that belong to graph g
                node_indices = (batch == g).nonzero(as_tuple=True)[
                    0
                ]  # shape [num_nodes_in_graph_g]
                g_preds = preds[node_indices].squeeze(-1)

                # Pad if needed
                num_nodes_g = g_preds.size(0)
                if num_nodes_g < self.max_nodes:
                    pad_size = self.max_nodes - num_nodes_g
                    g_preds = F.pad(g_preds, (0, pad_size), value=0.0)
                elif num_nodes_g > self.max_nodes:  # safety
                    g_preds = g_preds[: self.max_nodes]
                padded_outs.append(g_preds)

            out_batched = torch.stack(padded_outs, dim=0)
            return out_batched  # shape [B, 121]

        elif self.mode == "edge":

            edge_src = edge_index[0]
            edge_dst = edge_index[1]
            # pool node embeddings at endpoints u, v
            edge_embs = torch.maximum(h[edge_src], h[edge_dst])  # shape [E, hidden_dim]

            logits = self.decoder(edge_embs)  # shape [E, 1]
            preds = self.sigmoid(logits).squeeze(-1)  # shape [E]

            edge_graph = batch[edge_src]  # shape [E]

            num_graphs = batch.max().item() + 1  # graphs in batch
            max_edges = 304
            padded_outs = []

            for g in range(num_graphs):
                edge_indices = (edge_graph == g).nonzero(as_tuple=True)[0]
                g_preds = preds[edge_indices]

                # Pad
                num_edges_g = g_preds.size(0)
                if num_edges_g < max_edges:
                    pad_size = max_edges - num_edges_g
                    g_preds = F.pad(g_preds, (0, pad_size), value=0.0)
                elif num_edges_g > max_edges:
                    g_preds = g_preds[:max_edges]
                padded_outs.append(g_preds)

            out_batched = torch.stack(padded_outs, dim=0)  # [B, 304]
            return out_batched

        else:
            raise ValueError(
                f"Mode {self.mode} is not a recognized mode! Only 'edge' or 'node' supported."
            )
