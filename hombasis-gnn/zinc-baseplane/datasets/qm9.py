import gzip
import json
import numpy as np
from os import path as osp
import torch
from torch_geometric import data as tgdata
from tqdm import tqdm

CHEMICAL_ACC_NORMALISING_FACTORS = [
    0.066513725,
    0.012235489,
    0.071939046,
    0.033730778,
    0.033486113,
    0.004278493,
    0.001330901,
    0.004165489,
    0.004128926,
    0.00409976,
    0.004527465,
    0.012292586,
    0.037467458,
]


def map_qm9_to_pyg(json_file, make_undirected=True, remove_dup=False):
    # We're making the graph undirected just like the original repo.
    # Note: make_undirected makes duplicate edges, so we need to preserve edge types.
    # Note: The original repo also add self-loops. We don't need that given how we see hops.
    edge_index = np.array(
        [[g[0], g[2]] for g in json_file["graph"]]
    ).T  # Edge Index
    edge_attributes = np.array(
        [g[1] - 1 for g in json_file["graph"]]
    )  # Edge type (-1 to put in [0, 3] range)
    if (
        make_undirected
    ):  # This will invariably cost us edge types because we reduce duplicates
        edge_index_reverse = edge_index[[1, 0], :]
        # Concat and remove duplicates
        if remove_dup:
            edge_index = torch.LongTensor(
                np.unique(
                    np.concatenate([edge_index, edge_index_reverse], axis=1),
                    axis=1,
                )
            )
        else:
            edge_index = torch.LongTensor(
                np.concatenate([edge_index, edge_index_reverse], axis=1)
            )
            edge_attributes = torch.LongTensor(
                np.concatenate(
                    [edge_attributes, np.copy(edge_attributes)], axis=0
                )
            )
    x = torch.FloatTensor(np.array(json_file["node_features"]))
    y = torch.FloatTensor(np.array(json_file["targets"]).T)
    return tgdata.Data(
        x=x, edge_index=edge_index, edge_attr=edge_attributes, y=y
    )


class QM9Dataset(tgdata.InMemoryDataset):
    def __init__(
        self, root, split, transform=None, pre_transform=None, pre_filter=None
    ):
        self.json_gzip_path = f".dataset_src/QM9_{split}.jsonl.gz"
        self.homcount_path = f".dataset_src/QM9_{split}_homcounts.json"
        
        new_root = osp.join(root, split)
        super(QM9Dataset, self).__init__(
            new_root, transform, pre_transform, pre_filter
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        data_list_new = []
        
        with open(self.homcount_path, 'r') as hom_f:
            hom_data = json.load(hom_f)

        with gzip.open(self.json_gzip_path, "rb") as f:
            data = f.read().decode("utf-8")
            
            count = 0
            for line in tqdm(data.splitlines()):
                graph_dict = json.loads(line)
                graph_torch = map_qm9_to_pyg(graph_dict)                
                
                np_homcounts = np.array(hom_data[str(count)])
                ghom = np.sum(np_homcounts, axis=0)
                
                graph_torch.homcounts = torch.FloatTensor(ghom)
                
                graph_tran = self.pre_transform(graph_torch)
                if self.pre_filter is not None:
                    if not self.pre_filter(graph_tran):
                        continue
                if graph_tran is not None:
                    data_list_new.append(graph_tran)
                    
                count += 1

        data, slices = self.collate(data_list_new)
        torch.save((data, slices), self.processed_paths[0])
