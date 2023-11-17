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
    adj_mentions = []
    adj_trees = []
    if 'Adj' in batch[0] and 'adj_syntactic_dependency_tree' in batch[0]:
        for f in batch:
            adj_mention = []
            adj_tree = []
            adj_ = f['Adj']
            adj_tree_ = f['adj_syntactic_dependency_tree']
            for i in range(max_len):
                adj_mention.append(adj_[i][:max_len])
                adj_tree.append(adj_tree_[i][:max_len])
            adj_mentions.append(adj_mention)
            adj_trees.append(adj_tree)
    adj_mention = torch.tensor(adj_mentions, dtype=torch.float)
    adj_tree = torch.tensor(adj_trees, dtype=torch.float)

    output = (input_ids, input_mask, labels, entity_pos, hts, adj_mention, adj_tree)
    return output
