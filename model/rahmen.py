#!/usr/bin/env python3

import logging
import os
from time import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score

import torch
from torch import nn, optim
import torch.nn.functional as F
import dgl
import dgl.function as fn

import utils
from model.sample import HrtDataLoader


class RAHMeN(nn.Module):
    def __init__(
            self,
            num_layers,
            relations,
            feat_dim,
            embed_dim,
            dim_a,
            graph_dtype,
            bias=True,
            dropout=0.,
            activation=None,
            edge_wt=False
    ):
        super(RAHMeN, self).__init__()
        self.num_layers = num_layers
        self.relations = relations
        self.num_relations = len(self.relations)
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.dim_a = dim_a
        self.graph_dtype = graph_dtype
        self.bias = bias
        self.dropout = dropout
        self.activation = activation.casefold()
        self.edge_wt = edge_wt

        self.layers = nn.ModuleList([
            RAHMeNLayer(
                relations=self.relations,
                in_dim=self.feat_dim,
                out_dim=self.embed_dim,
                dim_a=self.dim_a,
                bias=self.bias,
                dropout=self.dropout,
                activation=self.activation
            )
        ])
        for _ in range(1, self.num_layers):
            self.layers.append(
                RAHMeNLayer(
                    relations=self.relations,
                    in_dim=self.embed_dim,
                    out_dim=self.embed_dim,
                    dim_a=self.dim_a,
                    bias=self.bias,
                    dropout=self.dropout,
                    activation=self.activation
                )
            )

    @staticmethod
    def get_score(vec1, vec2, eps=1e-10):
        norm = torch.linalg.norm(vec1) * torch.linalg.norm(vec2)
        eps = torch.tensor(eps)
        return ( torch.dot(vec1, vec2) / torch.max(norm, eps) ).item()

    def forward(self, blocks, expand_feat=True, return_attn=False):
        if expand_feat:
            # Expand initial features for each relation
            h = blocks[0].srcdata['feat'].unsqueeze(1).expand(-1, self.num_relations, -1)
        else:
            h = blocks[0].srcdata['feat']

        attn = None
        for layer, block in zip(self.layers, blocks):
            h, attn = layer(block, h, return_attn=return_attn)

        if return_attn:
            return h, attn
        else:
            return h


class RAHMeNLayer(nn.Module):
    def __init__(
            self,
            relations,
            in_dim,
            out_dim,
            dim_a,
            bias=True,
            dropout=0.,
            activation=None,
            edge_wt=False
    ):
        super(RAHMeNLayer, self).__init__()
        self.relations = relations
        self.num_relations = len(self.relations)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim_a = dim_a
        self.act_str = activation
        self.edge_wt = edge_wt

        self.dropout = nn.Dropout(dropout)
        self.activation = self._get_activation_fn(self.act_str)

        self.msg_fn = fn.u_mul_e if self.edge_wt else fn.copy_u
        self.reduce_fn = fn.mean

        self.self_weights = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.in_dim, self.out_dim)
        )
        self.neigh_weights = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.in_dim, self.out_dim)
        )
        self.attention = SemanticAttention(self.num_relations, self.out_dim, self.dim_a)

        self.bias = nn.Parameter(torch.FloatTensor(self.out_dim)) if bias else None
        self.norm = nn.LayerNorm(self.out_dim, elementwise_affine=True)

        self.reset_parameters()

    def reset_parameters(self):
        nonlinearity = 'relu' if self.act_str == 'relu' else 'leaky_relu'
        nn.init.kaiming_uniform_(self.self_weights, nonlinearity=nonlinearity)
        nn.init.kaiming_uniform_(self.neigh_weights, nonlinearity=nonlinearity)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @staticmethod
    def _get_activation_fn(activation):
        if activation is None:
            act_fn = None
        elif activation == 'relu':
            act_fn = F.relu
        elif activation == 'elu':
            act_fn = F.elu
        elif activation == 'gelu':
            act_fn = F.gelu
        else:
            raise ValueError('Invalid activation function.')

        return act_fn

    def forward(self, block, node_feat, return_attn=False):
        node_feat = self.dropout(node_feat)
        num_dst_nodes = block.number_of_dst_nodes()

        h_self = torch.empty(self.num_relations, num_dst_nodes, self.in_dim, device=block.device)
        h_neigh = torch.empty(self.num_relations, num_dst_nodes, self.in_dim, device=block.device)
        with block.local_scope():
            for i, relation in enumerate(self.relations):
                block.srcdata[relation] = node_feat[:, i]
                feat_dst = node_feat[:num_dst_nodes, i]

                # Handle graphs with no edges of given edge type
                # Set neighbor msg to zero vector
                if block.number_of_edges(etype=relation) == 0:
                    block.dstdata['neigh'] = torch.zeros(
                        feat_dst.shape[0], self.in_dim).to(feat_dst)

                msg_fn = self.msg_fn(relation, 'w', 'm') if self.edge_wt else self.msg_fn(relation, 'm')

                block.update_all(
                    msg_fn,
                    self.reduce_fn('m', 'neigh'),
                    etype=relation
                )

                h_self[i] = feat_dst
                h_neigh[i] = block.dstdata['neigh']

        h_self = torch.matmul(h_self, self.self_weights)
        h_neigh = torch.matmul(h_neigh, self.neigh_weights)

        h = h_self + h_neigh

        if self.bias is not None:
            h += self.bias
        if self.norm:
            h = self.norm(h)
        if self.activation:
            h = self.activation(h)

        return self.attention(h, return_attn=return_attn)


class SemanticAttention(nn.Module):
    def __init__(self, num_relations, in_dim, dim_a, dropout=0.):
        super(SemanticAttention, self).__init__()
        self.num_relations = num_relations
        self.in_dim = in_dim
        self.dim_a = dim_a
        self.dropout = nn.Dropout(dropout)

        self.weights_s1 = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.in_dim, self.dim_a)
        )
        self.weights_s2 = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.dim_a, self.num_relations)
        )

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.weights_s1.data, gain=gain)
        nn.init.xavier_uniform_(self.weights_s2.data)

    def forward(self, h, return_attn=False):
        # Shape of h: (num_relations, batch_size, dim)

        attention = F.softmax(
            torch.matmul(
                torch.tanh(
                    torch.matmul(h, self.weights_s1)
                ),
                self.weights_s2
            ),
            dim=0
        ).permute(1, 0, 2)

        attention = self.dropout(attention)

        # Output shape: (batch_size, num_relations, dim)
        h = torch.matmul(attention, h.permute(1, 0, 2))

        return h, attention if return_attn else None


def train_model(
        model,
        train_G,
        val_edges,
        neigh_sampler,
        loss_module,
        num_walks=20,
        walk_length=10,
        window_size=5,
        batch_size=64,
        EPOCHS=50,
        lr=1e-3,
        patience_limit=3,
        num_workers=2,
        device='cpu',
        model_dir='saved_models/model'
):
    model.to(device)
    loss_module.to(device)

    os.makedirs(model_dir, exist_ok=True)

    # TODO: Weight decay?
    optimizer = optim.Adam(
        [{'params': model.parameters()},
         {'params': loss_module.parameters()}],
        lr=lr
    )

    start = time()
    train_hrt = utils.generate_pairs(
        utils.generate_walks(train_G, num_walks=num_walks, walk_length=walk_length),
        window_size=window_size,
        num_workers=num_workers
    )
    end = time()

    train_loader = HrtDataLoader(
        g=train_G,
        hrt_tuples=train_hrt,
        block_sampler=neigh_sampler,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    logging.info(f'Generating random walk tuples... {end - start:.2f}s')

    expand_feat = len(train_G.ndata['feat'].shape) < 3

    start_train = time()
    patience = 0
    val_aucs, val_f1s, val_prs = [], [], []
    val_score = 0
    best_score, best_epoch = np.NINF, None
    for epoch in range(EPOCHS):
        model.train()

        data_iter = tqdm(
            train_loader,
            desc=f'Epoch: {epoch:02}',
            total=len(train_loader),
            position=0
        )

        loss, avg_loss = None, 0.
        for i, (blocks, head_invmap, etype_idx, tails) in enumerate(data_iter):
            heads = blocks[-1].dstdata[dgl.NID][head_invmap]

            optimizer.zero_grad()

            embeds = model(blocks, expand_feat=expand_feat)
            loss = loss_module(heads, embeds[head_invmap, etype_idx], tails)

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            data_iter.set_postfix({
                'val_score': val_score,
                'avg_loss': avg_loss / (i+1)
            })

        aucs, f1s, prs = eval_model(model, train_G, val_edges, neigh_sampler, batch_size, num_workers, device)
        auc_mean, f1_mean, pr_mean = np.mean(aucs), np.mean(f1s), np.mean(prs)
        val_score = auc_mean

        logging.info(f'{epoch:02}: ROC-AUC: {auc_mean:.4f} | F1: {f1_mean:.4f} | PR-AUC: {pr_mean:.4f}')

        val_aucs.append(auc_mean)
        val_f1s.append(f1_mean)
        val_prs.append(pr_mean)

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            patience = 0

            # Save checkpoint
            torch.save({
                'epoch': best_epoch,
                'score': best_score,
                'loss': loss,
                'loss_state_dict': loss_module.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f'{model_dir}/checkpoint.pt')
        else:
            patience += 1
            if patience >= patience_limit:
                logging.info(f'Early stopping. Loading best model from epoch {best_epoch}...')
                checkpoint = torch.load(f'{model_dir}/checkpoint.pt')
                model.load_state_dict(checkpoint['model_state_dict'])
                break
    end_train = time()
    logging.info(f'Total training time... {end_train - start_train:.2f}s')

    # Save final model separately
    model_name = os.path.normpath(model_dir).split(os.sep)[-1]
    torch.save(model.state_dict(), f'{model_dir}/{model_name}.pt')

    return val_aucs, val_f1s, val_prs


def eval_model(
        model,
        g,
        eval_edges,
        neigh_sampler,
        batch_size=64,
        num_workers=2,
        device='cpu'
):
    model.eval()
    with torch.no_grad():
        embeds = get_embeddings(model, g, neigh_sampler, batch_size=batch_size, num_workers=num_workers, device=device)

        aucs, f1s, prs = [], [], []
        for i, rel in enumerate(model.relations):
            src, dst, labels = eval_edges[rel]

            # Skip edge type if no eval edges of that type
            if not src:
                continue

            y_true, y_scores = [], []
            for s, d, lbl in tqdm(zip(src, dst, labels), position=0):
                src_emb = embeds[s, i]
                dst_emb = embeds[d, i]

                y_scores.append(model.get_score(src_emb, dst_emb))
                y_true.append(lbl)

            num_true = sum(y_true)
            sorted_pred = sorted(y_scores)
            threshold = sorted_pred[-num_true]

            y_pred = [
                1 if pred > threshold else 0
                for pred in y_scores
            ]

            ps, rs, _ = precision_recall_curve(y_true, y_scores)
            aucs.append(roc_auc_score(y_true, y_scores))
            f1s.append(f1_score(y_true, y_pred))
            prs.append(auc(rs, ps))

    return aucs, f1s, prs


def get_embeddings(
        model,
        g,
        neigh_sampler,
        return_attn=False,
        batch_size=64,
        num_workers=2,
        device='cpu'
):
    expand_feat = len(g.ndata['feat'].shape) < 3
    model.to(device)

    dummy_hrt = [
        [n.item(), -1, -1]
        for n in g.nodes()
    ]

    node_loader = HrtDataLoader(
        g=g,
        hrt_tuples=dummy_hrt,
        block_sampler=neigh_sampler,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model.eval()
    with torch.no_grad():
        embeds = torch.empty(g.num_nodes(), model.num_relations, model.embed_dim, device=device)
        if return_attn:
            attn = torch.empty(g.num_nodes(), model.num_relations, model.num_relations, device=device)

        for blocks, node_invmap, _, _ in tqdm(node_loader, desc='Getting Embeddings', position=0):
            nodes = blocks[-1].dstdata[dgl.NID][node_invmap]
            out = model(blocks, expand_feat=expand_feat, return_attn=return_attn)

            if return_attn:
                embeds[nodes], attn[nodes] = out
            else:
                embeds[nodes] = out

    # Push to cpu for sklearn eval functions
    if return_attn:
        return embeds.cpu(), attn.cpu()
    else:
        return embeds.cpu()
