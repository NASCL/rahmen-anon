#!/usr/bin/env python3

import torch
import dgl
import networkx as nx
from collections import defaultdict
from tqdm import tqdm


class MotifCounterV2:
    def __init__(self, G):
        self.dgl_G = G
        self.nx_G = nx.Graph(dgl.to_networkx(G))

        self.motifs = ['triangle', 'star', 'clique', 'tailed',
                       'cycle', 'four_path', 'four_star', 'chord_cycle']

    def generate_motif_graph(self, include_edge=True, print_totals=False):
        motif_G = nx.MultiGraph()
        totals = {m: 0 for m in self.motifs}

        tmp_motifs = ['edge'] + self.motifs if include_edge else self.motifs
        node_motifs = {
            n: {m: 0 for m in tmp_motifs}
            for n in self.nx_G.nodes()
        }

        for u, v in tqdm(self.nx_G.edges()):
            if include_edge:
                node_motifs[u]['edge'] += 1
                node_motifs[v]['edge'] += 1
                motif_G.add_edge(u, v, key=0, etype='edge', w=1)

            edge_motifs = {m: 0 for m in self.motifs}

            u_neigh, v_neigh = set(self.nx_G.neighbors(u)), set(self.nx_G.neighbors(v))
            tri = u_neigh & v_neigh
            star_u = u_neigh - tri - {v}
            star_v = v_neigh - tri - {u}

            tri_cnt = len(tri)
            star_u_cnt = len(star_u)
            star_v_cnt = len(star_v)
            clique_cnt = 0
            tailed_cnt = 0
            chord_cycle_cnt = 0
            cycle_cnt = 0
            fourstar_cnt = 0

            for w in tri:
                w_neighbors = set(self.nx_G.neighbors(w)) - {u, v}

                cliques = w_neighbors & tri  # where nebs from w also form triangle with u & v
                clique_cnt += len(cliques)

                center_chord_cycles = tri - {w} - cliques
                chord_cycle_cnt += len(center_chord_cycles) / 2  # will be double counted when chord

                # where nebs of w intersect with a 3-star from u or v
                border_chord_cycles = ((w_neighbors & star_u) | (w_neighbors & star_v)) - cliques
                chord_cycle_cnt += len(border_chord_cycles)

                chord_cycles = center_chord_cycles | border_chord_cycles

                tailed = (w_neighbors | star_u | star_v) - chord_cycles - cliques
                tailed_cnt += len(tailed)

            clique_cnt //= 2

            for w in star_u:
                w_neighbors = set(self.nx_G.neighbors(w))

                cycles = w_neighbors & star_v
                cycle_cnt += len(cycles)

                fourstar_cnt -= len(w_neighbors & star_u) / 2
                tailed_cnt += len(w_neighbors & star_u) / 2

            for w in star_v:
                w_neighbors = set(self.nx_G.neighbors(w))
                fourstar_cnt -= len(w_neighbors & star_v) / 2
                tailed_cnt += len(w_neighbors & star_v) / 2

            fourpath_cnt = star_u_cnt * star_v_cnt - cycle_cnt
            fourstar_cnt += (star_u_cnt - 1) / 2 * star_u_cnt + (star_v_cnt - 1) / 2 * star_v_cnt

            edge_motifs['triangle'] = tri_cnt
            edge_motifs['star'] = (star_u_cnt + star_v_cnt)
            edge_motifs['clique'] = clique_cnt
            edge_motifs['tailed'] = tailed_cnt
            edge_motifs['cycle'] = cycle_cnt
            edge_motifs['four_path'] = fourpath_cnt
            edge_motifs['four_star'] = fourstar_cnt
            edge_motifs['chord_cycle'] = chord_cycle_cnt

            for motif_type, motif_count in edge_motifs.items():
                node_motifs[u][motif_type] += motif_count
                node_motifs[v][motif_type] += motif_count

                if motif_count > 0:
                    key = self.motifs.index(motif_type)
                    if include_edge:
                        key += 1
                    motif_G.add_edge(u, v, key=key, etype=motif_type, w=motif_count)

            totals['triangle'] += tri_cnt
            totals['star'] += (star_u_cnt + star_v_cnt)
            totals['clique'] += clique_cnt
            totals['tailed'] += tailed_cnt
            totals['cycle'] += cycle_cnt
            totals['four_path'] += fourpath_cnt
            totals['four_star'] += fourstar_cnt
            totals['chord_cycle'] += chord_cycle_cnt

        if print_totals:
            # NEEDS ADJUSTING ON SOME, BUT TRIS AND CLIQUES SHOULD BE CORRECT
            print()
            print('TOTALS')
            for m, cnt in totals.items():
                if m == 'four_path':
                    val = 1  # only gets counted when edge in question is in the middle of the path
                elif m == 'star':
                    val = 2
                elif m in ['triangle', 'four_star']:
                    val = 3
                elif m in ['cycle', 'tailed']:
                    val = 4
                elif m == 'chord_cycle':
                    val = 5
                else:
                    val = 6

                print(f'{m}: {cnt / val}')

        # return self.convert_motif_graph_dgl(motif_G), node_motifs
        return node_motifs

    def convert_motif_graph_dgl(self, motif_G):
        graph_data = defaultdict(lambda: ([], []))
        edge_data = defaultdict(list)
        for src, dst, data in motif_G.edges(data=True):
            edge_schema = ('node', data['etype'], 'node')

            graph_data[edge_schema][0].append(src)
            graph_data[edge_schema][1].append(dst)

            edge_data[edge_schema].append(data['w'])

        G = dgl.heterograph(graph_data)

        G.ndata['train_mask'] = self.dgl_G.ndata['train_mask']
        G.ndata['val_mask'] = self.dgl_G.ndata['val_mask']
        G.ndata['test_mask'] = self.dgl_G.ndata['test_mask']
        G.ndata['feat'] = self.dgl_G.ndata['feat']
        G.ndata['label'] = self.dgl_G.ndata['label']

        for edge_schema, val in edge_data.items():
            tmp = torch.FloatTensor(val)
            if edge_schema[1] != 'edge':
                tmp /= tmp.max()
            edge_data[edge_schema] = tmp

        G.edata['w'] = edge_data

        G = dgl.add_reverse_edges(G, copy_ndata=True, copy_edata=True)

        return G
