import torch.nn as nn
import torch
import math


class RAA(nn.Module):
    """Attention for relational aware adj"""

    def __init__(self, in_dim, out_dim, lamda):
        super(RAA, self).__init__()
        self.lamda = lamda
        self.Wq = nn.Linear(in_dim, out_dim, bias=False)
        self.Wk = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, n_emb, h_adj_emb, r_adj_emb, t_adj_emb, mask_r_adj, mask_u_n_r_len):
        d0, d1, d2, d3 = r_adj_emb.size()[0], r_adj_emb.size()[1], r_adj_emb.size()[2], r_adj_emb.size()[3]

        # get plausibility weights
        pla_scores = torch.norm(h_adj_emb + r_adj_emb - t_adj_emb, p=2, dim=-1)
        exp = (1 / torch.exp(pla_scores)) * mask_r_adj
        exp_sum = exp.sum(dim=-1, keepdim=True)
        exp_sum = exp_sum.expand([d0, d1, d2, d3])
        pla_weights = exp / (exp_sum + 0.000001)

        # get original attention scores
        Q, K = self.Wq(n_emb), self.Wk(n_emb)
        d_k = Q.size()[-1]
        ori_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.pow(d_k, 2)
        ori_scores = ori_scores.unsqueeze(1).expand([d0, d1, d2, d3])

        # get raa weights
        raa_scores = ori_scores * (1 + self.lamda * pla_weights)
        exp = torch.exp(raa_scores) * mask_u_n_r_len
        exp_sum = exp.sum(dim=-1, keepdim=True)
        raa_weights = exp / (exp_sum + 0.000001)

        return raa_weights


class Attention1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Attention1, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim, bias=True)
        self.lin2 = nn.Linear(out_dim, 1, bias=False)

    def forward(self, inputs, mask):
        mlp_out = torch.tanh(self.lin1(inputs))
        exp = torch.exp(self.lin2(mlp_out).squeeze(-1)) * mask
        exp_sum = exp.sum(-1, keepdim=True)
        alpha = (exp / exp_sum).unsqueeze(1)
        out = torch.matmul(alpha, inputs).squeeze()
        return out


class Attention2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Attention2, self).__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)
        self.U = nn.Linear(in_dim, out_dim, bias=True)
        self.V = nn.Linear(out_dim, 1, bias=False)

    def forward(self, inputs, query, mask):
        mlp_out = torch.tanh(self.W(inputs) + self.U(query).unsqueeze(1))
        scores = self.V(mlp_out).squeeze(-1)
        exp = torch.exp(scores) * mask
        exp_sum = exp.sum(-1, keepdim=True)
        alpha = (exp / exp_sum).unsqueeze(1)
        out = torch.matmul(alpha, inputs).squeeze()
        return out


class GCNLayers(nn.Module):
    """GCN layers for encoding subgraph"""

    def __init__(self, in_dim, out_dim, num_layers):
        super(GCNLayers, self).__init__()
        self.num_layers = num_layers
        self.weight_list = nn.ModuleList()
        for i in range(self.num_layers):
            self.weight_list.append(nn.Linear(in_dim, out_dim, bias=False))

        self.out = nn.Linear(in_dim, out_dim)

    def forward(self, adj, inputs):
        d0, d1, d2, d3 = adj.size()[0], adj.size()[1], adj.size()[2], inputs.size()[-1]
        inputs = inputs.unsqueeze(1).expand([d0, d1, d2, d3])
        for i in range(self.num_layers):
            Ax = torch.matmul(adj, inputs)
            AxW = self.weight_list[i](Ax)
            aAxW = torch.relu(AxW)
            inputs = aAxW

        out = self.out(inputs)
        return out


class BiRNN(nn.Module):
    """BiRNN for encoding item sequence and user sequence"""

    def __init__(self, input_size, hid_size, drop_rate, rnn_cell):
        super(BiRNN, self).__init__()
        self.rnn = rnn_cell(input_size=input_size,
                            hidden_size=hid_size,
                            num_layers=1,
                            bias=True,
                            batch_first=True,
                            dropout=drop_rate,
                            bidirectional=True)

    def forward(self, packed_batch):
        encoder_outputs_packed, _ = self.rnn(packed_batch)
        encoder_outputs, len_list = torch.nn.utils.rnn.pad_packed_sequence(encoder_outputs_packed, batch_first=True)
        return encoder_outputs, len_list


class UPIACM(nn.Module):
    def __init__(self, args, n_user, n_item, n_entity, n_relation):
        super(UPIACM, self).__init__()

        self.n_user = n_user
        self.n_item = n_item
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.embed_dim = args.embed_dim
        self.att1_dim = args.att1_dim
        self.att2_dim = args.att2_dim
        self.raa_dim = args.raa_dim
        self.lamda = args.lamda

        self.num_layers = args.num_layers

        self._init_embedding()
        self._init_rnn()
        self._init_gcn()
        self._init_attention()
        self._init_mutual_module()

    def _init_embedding(self):
        self.user_embed = nn.Embedding(self.n_user, self.embed_dim, padding_idx=0)
        self.entity_embed = nn.Embedding(self.n_entity, self.embed_dim, padding_idx=0)
        self.relation_embed = nn.Embedding(self.n_relation, self.embed_dim, padding_idx=0)
        self.user_embed.weight.data[1:, :].uniform_(-1.0, 1.0)
        self.entity_embed.weight.data[1:, :].uniform_(-1.0, 1.0)
        self.relation_embed.weight.data[1:, :].uniform_(-1.0, 1.0)

    def _init_rnn(self):
        self.u_rnn = BiRNN(input_size=self.embed_dim, hid_size=int(self.embed_dim / 2), drop_rate=0.0, rnn_cell=nn.GRU)
        self.i_rnn = BiRNN(input_size=self.embed_dim, hid_size=int(self.embed_dim / 2), drop_rate=0.0, rnn_cell=nn.GRU)

    def _init_gcn(self):
        self.u_gcn = GCNLayers(in_dim=self.embed_dim, out_dim=self.embed_dim, num_layers=self.num_layers)
        self.i_gcn = GCNLayers(in_dim=self.embed_dim, out_dim=self.embed_dim, num_layers=self.num_layers)

    def _init_attention(self):
        self.u_raa = RAA(in_dim=self.embed_dim, out_dim=self.raa_dim, lamda=self.lamda)
        self.u_raagcn_atten = Attention1(in_dim=self.embed_dim, out_dim=self.att1_dim)
        self.u_i_rnn_atten = Attention2(in_dim=self.embed_dim, out_dim=self.att2_dim)

        self.i_raa = RAA(in_dim=self.embed_dim, out_dim=self.raa_dim, lamda=self.lamda)
        self.i_raagcn_atten = Attention1(in_dim=self.embed_dim, out_dim=self.att1_dim)
        self.i_u_rnn_atten = Attention2(in_dim=self.embed_dim, out_dim=self.att2_dim)

    def _init_mutual_module(self):
        self.gate_lin = nn.Linear(self.embed_dim * 4, self.embed_dim * 2, bias=True)

    def forward(self, packaged_batch):
        u_i_len_tensor, u_ori_idx_tensor, user_tensor, u_i_tensor, u_n_tensor, \
        u_h_adj_tensor, u_r_adj_tensor, u_t_adj_tensor, u_n_len_tensor, u_r_len_tensor, \
        i_u_len_tensor, i_ori_idx_tensor, item_tensor, i_u_tensor, i_n_tensor, \
        i_h_adj_tensor, i_r_adj_tensor, i_t_adj_tensor, i_n_len_tensor, i_r_len_tensor, label_tensor = packaged_batch

        # --------------------- look up embedding ------------------------
        u_i_emb = self.entity_embed(u_i_tensor)
        u_n_emb = self.entity_embed(u_n_tensor)
        u_h_adj_emb = self.entity_embed(u_h_adj_tensor)
        u_r_adj_emb = self.relation_embed(u_r_adj_tensor)
        u_t_adj_emb = self.entity_embed(u_t_adj_tensor)

        i_u_emb = self.user_embed(i_u_tensor)
        i_n_emb = self.entity_embed(i_n_tensor)
        i_h_adj_emb = self.entity_embed(i_h_adj_tensor)
        i_r_adj_emb = self.relation_embed(i_r_adj_tensor)
        i_t_adj_emb = self.entity_embed(i_t_adj_tensor)

        # --------------------- get mask dimension ------------------------
        u_d0, u_d1, u_d2, u_d3 = u_r_adj_tensor.size()[0], u_r_adj_tensor.size()[1], u_r_adj_tensor.size()[2], \
                                 u_r_adj_tensor.size()[3]
        i_d0, i_d1, i_d2, i_d3 = i_r_adj_tensor.size()[0], i_r_adj_tensor.size()[1], i_r_adj_tensor.size()[2], \
                                 i_r_adj_tensor.size()[3]

        # --------------------- mask ---------------------------
        # mask for item size
        temp1 = torch.arange(0, int(u_i_len_tensor.max())).unsqueeze(1).cuda()
        temp2 = torch.arange(0, int(i_u_len_tensor.max())).unsqueeze(1).cuda()
        mask_u_i_len = ((temp1 < u_i_len_tensor.unsqueeze(0)).float()).transpose(-2, -1)
        mask_i_u_len = ((temp2 < i_u_len_tensor.unsqueeze(0)).float()).transpose(-2, -1)

        # mask for relation adj
        mask_u_r_adj = torch.ne(u_r_adj_tensor, 0.0).float()
        mask_i_r_adj = torch.ne(i_r_adj_tensor, 0.0).float()

        # mask for node size, adj
        mask_u_n_len_adj = torch.matmul(torch.ne(u_n_tensor, 0.0).float().unsqueeze(-1),
                                        torch.ne(u_n_tensor, 0.0).float().unsqueeze(-2))
        mask_i_n_len_adj = torch.matmul(torch.ne(i_n_tensor, 0.0).float().unsqueeze(-1),
                                        torch.ne(i_n_tensor, 0.0).float().unsqueeze(-2))

        # mask for relation size
        temp1 = torch.arange(0, int(u_r_len_tensor.max())).unsqueeze(1).cuda()
        temp2 = torch.arange(0, int(i_r_len_tensor.max())).unsqueeze(1).cuda()
        mask_u_r_len = ((temp1 < u_r_len_tensor.unsqueeze(0)).float()).transpose(-2, -1)
        mask_i_r_len = ((temp2 < i_r_len_tensor.unsqueeze(0)).float()).transpose(-2, -1)

        # jointly mask for node size and relation size
        mask_u_n_r_len = mask_u_n_len_adj.unsqueeze(1).expand([u_d0, u_d1, u_d2, u_d3]) * \
                         mask_u_r_len.unsqueeze(-1).unsqueeze(-1).expand([u_d0, u_d1, u_d2, u_d3])
        mask_i_n_r_len = mask_i_n_len_adj.unsqueeze(1).expand([i_d0, i_d1, i_d2, i_d3]) * \
                         mask_i_r_len.unsqueeze(-1).unsqueeze(-1).expand([i_d0, i_d1, i_d2, i_d3])

        # --------------------- user preference -------------------------
        # bi-rnn for item sequence
        u_packed_batch = torch.nn.utils.rnn.pack_padded_sequence(u_i_emb, u_i_len_tensor, batch_first=True)
        u_rnn_out, _ = self.u_rnn(u_packed_batch)
        # raa-gcn for graph
        u_raa_adj_weights = self.u_raa(u_n_emb, u_h_adj_emb, u_r_adj_emb, u_t_adj_emb, mask_u_r_adj, mask_u_n_r_len)
        u_gcn_output = self.u_gcn(u_raa_adj_weights, u_n_emb)
        u_pool_output = self._pooling(u_gcn_output, mask_u_n_r_len, u_n_len_tensor)
        # attention for raa-gcn output
        u_graph_rep = self.u_raagcn_atten(u_pool_output, mask_u_r_len)
        u_rnn_atten_out = self.u_i_rnn_atten(u_rnn_out, u_graph_rep, mask_u_i_len)

        # --------------------- item attractiveness ---------------------
        # bi-rnn
        i_packed_batch = torch.nn.utils.rnn.pack_padded_sequence(i_u_emb, i_u_len_tensor, batch_first=True)
        i_rnn_out, _ = self.i_rnn(i_packed_batch)
        # raa-gcn
        i_raa_adj_weights = self.i_raa(i_n_emb, i_h_adj_emb, i_r_adj_emb, i_t_adj_emb, mask_i_r_adj, mask_i_n_r_len)
        i_gcn_output = self.i_gcn(i_raa_adj_weights, i_n_emb)
        i_pool_output = self._pooling(i_gcn_output, mask_i_n_r_len, i_n_len_tensor)
        # attention for raa-gcn output
        i_graph_rep = self.i_raagcn_atten(i_pool_output, mask_i_r_len)
        i_rnn_atten_out = self.i_u_rnn_atten(i_rnn_out, i_graph_rep, mask_i_u_len)

        # --------------------- aggregating ---------------------
        u_rep = torch.cat([u_graph_rep, u_rnn_atten_out], dim=-1)
        i_rep = torch.cat([i_graph_rep, i_rnn_atten_out], dim=-1)

        # --------------------- index aligning ---------------------
        u_rep = u_rep[u_ori_idx_tensor]
        i_rep = i_rep[i_ori_idx_tensor]

        # --------------------- gated interaction ---------------------
        sum_rep = torch.cat([u_rep, i_rep], dim=-1)
        o_rep = 1 - torch.sigmoid(self.gate_lin(sum_rep))

        u_final_rep = o_rep * torch.tanh(u_rep)
        i_final_rep = o_rep * torch.tanh(i_rep)

        # --------------------- CTR probability ---------------------
        logits = torch.sigmoid((u_final_rep * i_final_rep).sum(dim=-1)).squeeze()

        return logits

    def _pooling(self, gcn_output, mask, n_len_tensor):
        d0, d1, d2, d3 = gcn_output.size()[0], gcn_output.size()[1], gcn_output.size()[2], gcn_output.size()[3]
        mask = mask.sum(-1, keepdim=True).ne(0.0).float().expand([d0, d1, d2, d3])
        gcn_output = gcn_output * mask
        pooling_out = gcn_output.sum(dim=-2) / n_len_tensor.unsqueeze(-1).unsqueeze(-1).expand([d0, d1, 1])

        return pooling_out
