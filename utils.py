import torch
import random
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    Adjs = []
    Adj = torch.tensor(Adjs, dtype=torch.float)
    if 'Adj' in batch[0]:
        for f in batch:
            Adj = []
            for i in range(0, max_len):
                Adj.append([0] * max_len)
            A = f['Adj']
            for i in range(0, len(A)):
                for j in range(0, len(A)):
                    Adj[i][j] = A[i][j]
            Adjs.append(Adj)
        Adj = torch.tensor(Adjs, dtype=torch.float)

    adj_syntactic_dependency_trees = []
    adj_syntactic_dependency_tree = torch.tensor(adj_syntactic_dependency_trees, dtype=torch.float)
    if 'adj_syntactic_dependency_tree' in batch[0]:
        for f in batch:
            adj_syntactic_dependency_tree = []
            for i in range(0, max_len):
                adj_syntactic_dependency_tree.append([0] * max_len)
            A = f['adj_syntactic_dependency_tree']
            for i in range(0, len(A)):
                for j in range(0, len(A)):
                    adj_syntactic_dependency_tree[i][j] = A[i][j]
            adj_syntactic_dependency_trees.append(adj_syntactic_dependency_tree)
        adj_syntactic_dependency_tree = torch.tensor(adj_syntactic_dependency_trees, dtype=torch.float)
    output = (input_ids, input_mask, labels, entity_pos, hts, Adj, adj_syntactic_dependency_tree)
    return output
