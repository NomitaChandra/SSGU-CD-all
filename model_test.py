from cProfile import label
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from long_seq import process_long_input
import numpy as np
import math
from losses import *
from model_utils.attn_unet import AttentionUNet
from model_utils.agcn import GraphConvolution, TypeGraphConvolution
from allennlp.modules.matrix_attention import DotProductMatrixAttention, CosineMatrixAttention, BilinearMatrixAttention
from model_utils.ACC_UNet import ACC_UNet


class DocREModel(nn.Module):
    def __init__(self, args, config, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        # todo ugdre
        self.sizeA = 256
        self.gc1 = GraphConvolution(config.hidden_size, self.sizeA)
        self.dropout = nn.Dropout(0.5)

        self.args = args
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()

        self.margin = args.m
        if args.isrank:
            self.rels = args.num_class - 1
        else:
            self.rels = args.num_class
        # 768 * 2, 768
        # self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        # self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        # self.head_extractor = nn.Linear(config.hidden_size + self.sizeA + args.unet_out_dim, emb_size)
        # self.tail_extractor = nn.Linear(config.hidden_size + self.sizeA + args.unet_out_dim, emb_size)
        self.head_extractor = nn.Linear(config.hidden_size + 2 * args.unet_out_dim, emb_size)
        self.tail_extractor = nn.Linear(config.hidden_size + 2 * args.unet_out_dim, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

        self.bertdrop = nn.Dropout(0.6)
        self.unet_in_dim = args.unet_in_dim
        self.unet_out_dim = args.unet_in_dim
        self.liner = nn.Linear(config.hidden_size + self.sizeA, args.unet_in_dim)
        self.min_height = args.max_height
        self.channel_type = args.channel_type
        self.segmentation_net = AttentionUNet(input_channels=args.unet_in_dim,
                                              class_number=args.unet_out_dim,
                                              down_channel=args.down_dim)
        self.segmentation_net_acc_unet = ACC_UNet(n_channels=args.unet_in_dim, n_classes=args.unet_out_dim)

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        entity_es = []
        entity_as = []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            # for e in entity_pos[i]:
            for entity_num, e in enumerate(entity_pos[i]):
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(c).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(c).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)
            # 这句话是新增的，这句话很离谱
            for _ in range(self.min_height - entity_num - 1):
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            # 这句话是新增的
            entity_es.append(entity_embs)
            entity_as.append(entity_atts)

            # if len(hts[i]) == 0:
            #     hss.append(torch.FloatTensor([]).to(sequence_output.device))
            #     tss.append(torch.FloatTensor([]).to(sequence_output.device))
            #     rss.append(torch.FloatTensor([]).to(sequence_output.device))
            #     continue
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            # 将 ht 降维
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            # rrs = contract("rl,ld->rd", ht_att, sequence_output[i])
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)

        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss, entity_es, entity_as

    def get_channel_map(self, sequence_output, entity_as):
        # sequence_output = sequence_output.to('cpu')
        # attention = attention.to('cpu')
        bs, _, d = sequence_output.size()
        # ne = max([len(x) for x in entity_as])  # 本次bs中的最大实体数
        ne = self.min_height
        index_pair = []
        for i in range(ne):
            tmp = torch.cat((torch.ones((ne, 1), dtype=int) * i, torch.arange(0, ne).unsqueeze(1)), dim=-1)
            index_pair.append(tmp)
        index_pair = torch.stack(index_pair, dim=0).reshape(-1, 2).to(sequence_output.device)
        map_rss = []
        for b in range(bs):
            entity_atts = entity_as[b]
            h_att = torch.index_select(entity_atts, 0, index_pair[:, 0])
            t_att = torch.index_select(entity_atts, 0, index_pair[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[b], ht_att)
            map_rss.append(rs)
        map_rss = torch.cat(map_rss, dim=0).reshape(bs, ne, ne, d)
        return map_rss

    def square_loss(self, yPred, yTrue, margin=1.):
        if len(yPred) == 0:
            return torch.FloatTensor([0]).cuda()
        loss = 0.25 * (yPred * yTrue - margin) ** 2
        return torch.mean(loss.sum() / yPred.shape[0])

    def get_ht(self, rel_enco, hts):
        htss = []
        for i in range(len(hts)):
            ht_index = hts[i]
            for (h_index, t_index) in ht_index:
                htss.append(rel_enco[i, h_index, t_index])
        htss = torch.stack(htss, dim=0)
        return htss

    def forward(self, input_ids=None, attention_mask=None, labels=None, entity_pos=None, hts=None, list_feature_id=None,
                Adj=None, adj_syntactic_dependency_tree=None):
        sequence_output, attention = self.encode(input_ids, attention_mask)

        # todo ugdre
        sequence_output = self.dropout(sequence_output)
        # GCN
        Adj = F.normalize(Adj)
        sequence_output_A = torch.relu(self.gc1(sequence_output, Adj))
        sequence_output = torch.cat([sequence_output, sequence_output_A], dim=2)

        hs, rs, ts, entity_embs, entity_as = self.get_hrt(sequence_output, attention, entity_pos, hts)
        bs, sequen_len, d = sequence_output.shape
        if self.channel_type == 'context-based':
            feature_map = self.get_channel_map(sequence_output, entity_as)
            # print('feature_map:', feature_map.shape)
            attn_input = self.liner(feature_map).permute(0, 3, 1, 2).contiguous()
        elif self.channel_type == 'similarity-based':
            ent_encode = sequence_output.new_zeros(bs, self.min_height, d)
            for _b in range(bs):
                entity_emb = entity_embs[_b]
                entity_num = entity_emb.size(0)
                ent_encode[_b, :entity_num, :] = entity_emb
            # similar0 = ElementWiseMatrixAttention()(ent_encode, ent_encode).unsqueeze(-1)
            similar1 = DotProductMatrixAttention()(ent_encode, ent_encode).unsqueeze(-1)
            similar2 = CosineMatrixAttention()(ent_encode, ent_encode).unsqueeze(-1)
            similar3 = BilinearMatrixAttention(self.emb_size, self.self.emb_size).to(
                ent_encode.device)(ent_encode, ent_encode).unsqueeze(-1)
            attn_input = torch.cat([similar1, similar2, similar3], dim=-1).permute(0, 3, 1, 2).contiguous()
        else:
            raise Exception("channel_type must be specify correctly")

        attn_map = self.segmentation_net(attn_input)
        # attn_map = self.segmentation_net_acc_unet(attn_input)
        # attn_map = attn_map.permute(0, 2, 3, 1).contiguous()
        h_t = self.get_ht(attn_map, hts)
        rs = h_t

        # Binary Classifier
        # a = torch.cat([hs, rs], dim=1)
        # b = self.head_extractor(a)
        # hr进行了拼接变成了(1376, 1536)，然后再变回之前的维度(1376, 768)
        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))  # zs
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))  # zo
        # (1376, 768) -> (1376, 12, 64)
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels))
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())

            output = [loss.to(sequence_output), output]
        return output
