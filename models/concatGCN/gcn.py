import torch
import numpy as np
import copy #provides common shallow and deep copy operations.
import math #a built-in library of math classes provided by Python
import torch.nn as nn #contains some common functions used in neural networks
import torch.nn.functional as F #Such as ReLU, pool, DropOut, etc
from torch.autograd import Variable #PyTorch's automatic differential engine that supports neural network training.
from tree import Tree, head_to_tree, tree_to_adj

class GCNClassifier(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        in_dim = args.hidden_dim
        self.args = args
        self.gcn_model = GCNAbsaModel(args, emb_matrix=emb_matrix)
        self.classifier = nn.Linear(in_dim, args.num_class)

    def forward(self, inputs):
        outputs, h_sy, h_se, h_csy, h_cse= self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, outputs, h_sy, h_se, h_csy, h_cse


class GCNAbsaModel(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        self.args = args
        emb_matrix = torch.Tensor(emb_matrix)
        self.emb_matrix = emb_matrix

        self.in_drop = nn.Dropout(args.input_dropout) #Dropout：Prevents overfitting

        # create embedding layers
        self.emb = nn.Embedding(args.token_vocab_size, args.emb_dim, padding_idx=0)
        if emb_matrix is not None:
            self.emb.weight = nn.Parameter(emb_matrix.cuda(), requires_grad=False)

        self.pos_emb = nn.Embedding(args.pos_vocab_size, args.pos_dim, padding_idx=0) if args.pos_dim > 0 else None  # POS emb
        self.post_emb = nn.Embedding(args.post_vocab_size, args.post_dim, padding_idx=0) if args.post_dim > 0 else None  # position emb

        # rnn layer
        self.in_dim = args.emb_dim + args.post_dim + args.pos_dim
        self.rnn = nn.LSTM(self.in_dim, args.rnn_hidden, 1, batch_first=True, bidirectional=True)

        # attention adj layer
        self.attn = MultiHeadAttention(args.head_num, args.rnn_hidden * 2)

        self.h_weight = nn.Parameter(torch.FloatTensor(2).normal_(0.5, 0.5))
        #nn.Parameter（）：Converts an untrainable type Tensor into a trainable type parameter and registers that parameter as part of the host model.
        #The purpose of torch.FloatTensor() is to convert a given list or numpy into a tensor of the floating point type.

        # gcn layer
        self.gcn1 = GCN(args, args.hidden_dim, args.num_layers)
        self.gcn2 = GCN(args, args.hidden_dim, args.num_layers)
        self.gcn_common = GCN(args, args.hidden_dim, args.num_layers)

        # ***** Wang modified the code*****(begin)
        # MLP Layer
        self.linear = nn.Linear(3 * self.args.num_layers * args.hidden_dim, args.hidden_dim)
        # This step builds a fully connected layer, with the input dimension being the sum of dimensions of the multi-layer MAFGCN and the output dimension being the dimension of the hidden state vector.
        # Among them, 3 represents syntax, semantics, and common GCN; self.args.num_layers indicates the number of layers of MAFGCN; args.hidden_dim indicates the dimension of hidden state vector.

        self.affine1 = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        self.affine2 = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        self.affine3 = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        # This step builds three weight matrices that are used in the multi-emission transform module.

        self.args.initializer(self.affine1)
        self.args.initializer(self.affine2)
        self.args.initializer(self.affine3)
        # In this step, the three weight matrices constructed in the previous step are assigned initial values to make the experimental results reproducible.
        # ***** Wang modified the code*****(end)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(batch_size, self.args.rnn_hidden, 1, True)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens.cpu(), batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def create_embs(self, tok, pos, post):
        # embedding
        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.args.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.args.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=2) #torch.cat splices two tensors together.
        embs = self.in_drop(embs)
        return embs

    def inputs_to_att_adj(self, input, score_mask):
        attn_tensor = self.attn(input, input, score_mask)  # [batch_size, head_num, seq_len, seq_len]
        attn_tensor = torch.sum(attn_tensor, dim=1)
        attn_tensor = select(attn_tensor, self.args.top_k) * attn_tensor
        return attn_tensor

    def forward(self, inputs):
        tok, asp, pos, head, post, dep, mask, l, adj = inputs           # unpack inputs
        # embedding
        embs = self.create_embs(tok, pos, post)

        # bi-lstm encoding
        rnn_hidden = self.encode_with_rnn(embs, l, tok.size(0))  # [batch_size, seq_len, hidden]
        # rnn_hidden[:,0,:] = torch.zeros_like(rnn_hidden[:,0,:])
        score_mask = torch.matmul(rnn_hidden, rnn_hidden.transpose(-2, -1))
        score_mask = (score_mask == 0)
        score_mask = score_mask.unsqueeze(1).repeat(1, self.args.head_num, 1, 1).cuda()

        # init adj
        att_adj = self.inputs_to_att_adj(rnn_hidden, score_mask)  # [batch_size, head_num, seq_len, hidden]  #inputs)

        # ***** Wang modified the code*****(begin)
        for i in range(self.args.num_layers):
        # Do the following for each layer in MAFGCN
            if i==0:#Floor 1
                h_sy = self.gcn1.first_layer(adj, rnn_hidden)
                h_se = self.gcn2.first_layer(att_adj, rnn_hidden)
                h_csy = self.gcn_common.first_layer(adj, rnn_hidden)
                h_cse = self.gcn_common.first_layer(att_adj, rnn_hidden)
                h_sy_inputs = torch.cat((rnn_hidden, h_sy), dim=-1)
                h_se_inputs = torch.cat((rnn_hidden, h_se), dim=-1)
                h_csy_inputs = torch.cat((rnn_hidden, h_csy), dim=-1)
                h_cse_inputs = torch.cat((rnn_hidden, h_cse), dim=-1)
                h_com = (self.h_weight[0] * h_csy + self.h_weight[1] * h_cse) / 2
            # For the first layer GCN, the input vector is rnn_hidden.
            # This part of the code obtains the output of the first layer of syntax, semantics, and common GCN, and obtains the input of the next layer of gcn

            else:
                # concat the last layer's out with input_feature as the current input
                h_sy = self.gcn1.next_layer(adj, h_sy_inputs,i, score_mask, 'syntax')
                h_se = self.gcn2.next_layer(att_adj, h_se_inputs,i, score_mask, 'semantic')
                h_csy = self.gcn_common.next_layer(adj, h_csy_inputs,i, score_mask, 'syntax')
                h_cse = self.gcn_common.next_layer(att_adj, h_cse_inputs,i, score_mask, 'semantic')
                h_sy_inputs = torch.cat((h_sy_inputs, h_sy), dim=-1)
                h_se_inputs = torch.cat((h_se_inputs, h_se), dim=-1)
                h_csy_inputs = torch.cat((h_csy_inputs, h_csy), dim=-1)
                h_cse_inputs = torch.cat((h_cse_inputs, h_cse), dim=-1)
                h_com = (self.h_weight[0] * h_csy + self.h_weight[1] * h_cse) / 2
            # The output of the syntax, semantics and common GCN of the n (n>=1) layer is obtained respectively, and the input of the gcn of the next layer is obtained

            # * affine module
            h_sy_1 = torch.bmm(F.softmax(torch.bmm(torch.matmul(h_sy, self.affine1), torch.transpose(h_se, 1, 2)), dim=-1),
                               h_se)
            h_se_1 = torch.bmm(F.softmax(torch.bmm(torch.matmul(h_se, self.affine3), torch.transpose(h_sy, 1, 2)), dim=-1),
                               h_sy)
            h_com_1 = torch.bmm(
                F.softmax(torch.bmm(torch.matmul(h_com, self.affine3), torch.transpose(h_sy, 1, 2)), dim=-1),
                h_sy)
            h_sy_2 = torch.bmm(
                F.softmax(torch.bmm(torch.matmul(h_sy_1, self.affine2), torch.transpose(h_com, 1, 2)), dim=-1),
                h_com)
            h_se_2 = torch.bmm(
                F.softmax(torch.bmm(torch.matmul(h_se_1, self.affine2), torch.transpose(h_com, 1, 2)), dim=-1),
                h_com)
            h_com_2 = torch.bmm(
                F.softmax(torch.bmm(torch.matmul(h_com_1, self.affine1), torch.transpose(h_se, 1, 2)), dim=-1),
                h_se)  # torch.bmm(): computes matrix multiplication for two tensors
            h_sy=h_sy_2
            h_se=h_se_2
            h_com=h_com_2
            # The above part of the code implements the multi-affine transformation module

            mask2=mask
            asp_wn = mask.sum(dim=1).unsqueeze(-1)  # aspect words num
            mask = mask.unsqueeze(-1).repeat(1, 1, self.args.hidden_dim)  # mask for h
            h_sy_mean = (h_sy * mask).sum(dim=1) / asp_wn
            h_se_mean = (h_se * mask).sum(dim=1) / asp_wn
            h_com_mean = (h_com * mask).sum(dim=1) / asp_wn
            # The above part of the code masks off all parts except aspect words and averages out the vectors of aspect words

            if i==0:
                outputs = torch.cat((h_sy_mean, h_se_mean, h_com_mean), dim=-1)  # mask h
            else:
                outputs = torch.cat((outputs,h_sy_mean, h_se_mean, h_com_mean), dim=-1)  # mask h
            # This section of code implements the multi-layer fusion module, which connects the output of the multi-layer MAFGCN

            if (i+1)==self.args.num_layers:
                outputs = F.relu(self.linear(outputs))
            # Use the fully connected layer built earlier

            mask=mask2
        # ***** Wang modified the code*****(end)

        return outputs, h_sy, h_se, h_csy, h_cse


class GCN(nn.Module):
    def __init__(self, args, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.args = args
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = args.rnn_hidden * 2

        # drop out
        self.in_drop = nn.Dropout(args.input_dropout)
        self.gcn_drop = nn.Dropout(args.gcn_dropout)

        # gcn layer
        self.W = nn.ModuleList()
        self.attn = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim + layer * self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

            # attention adj layer
            self.attn.append(MultiHeadAttention(args.head_num, input_dim)) if layer != 0 else None

    def GCN_layer(self, adj, gcn_inputs, denom, l): #denom：分母项
        Ax = adj.bmm(gcn_inputs)
        AxW = self.W[l](Ax)
        AxW = AxW / denom
        gAxW = F.relu(AxW) + self.W[l](gcn_inputs)
        # if dataset is not laptops else gcn_inputs = self.gcn_drop(gAxW)（如果数据集不是laptops，gcn_inputs = self.gcn_drop(gAxW)）
        gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
        return gcn_inputs

    # ***** Wang modified the code*****(begin)
    def first_layer(self, adj, inputs):
        # gcn
        denom = adj.sum(2).unsqueeze(2) + 1  # norm adj
        out = self.GCN_layer(adj, inputs, denom, 0)
        return out
    # The first layer of MAFGCN performs simple GCN operations

    def next_layer(self, adj, inputs,i, score_mask, type):
        # The adjacency matrix of the NTH layer (n>=1) semantic gcn is obtained from the output of the previous layer.
        denom = adj.sum(2).unsqueeze(2) + 1  # norm adj
        if type == 'semantic':
            # Find the adjacency matrix of semantic module
            # att_adj
            adj = self.attn[i - 1](inputs, inputs, score_mask)  # [batch_size, head_num, seq_len, dim]
            adj = torch.sum(adj, dim=1)

            adj = select(adj, self.args.top_k) * adj
            denom = adj.sum(2).unsqueeze(2) + 1  # norm adj

        out = self.GCN_layer(adj, inputs, denom, i)
        # GCN operation
        return out
    # ***** Wang modified the code*****(end)

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape))  # Change to param
    return h0.cuda(), c0.cuda()


def select(matrix, top_num):
    batch = matrix.size(0)
    len = matrix.size(1)
    matrix = matrix.reshape(batch, -1)
    maxk, _ = torch.topk(matrix, top_num, dim=1)

    for i in range(batch):
        matrix[i] = (matrix[i] >= maxk[i][-1]) #Assign 1 or 0 to matrix[i]
    matrix = matrix.reshape(batch, len, len)
    matrix = matrix + matrix.transpose(-2, -1)

    # selfloop
    for i in range(batch):
        matrix[i].fill_diagonal_(1)

    return matrix


def clones(module, N): #clones：复制品
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    # d_model:hidden_dim，h:head_num
    def __init__(self, head_num, hidden_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % head_num == 0

        self.d_k = int(hidden_dim // head_num)
        self.head_num = head_num
        self.linears = clones(nn.Linear(hidden_dim, hidden_dim), 2)
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, score_mask, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if score_mask is not None:
            scores = scores.masked_fill(score_mask, -1e9)

        b = ~score_mask[:, :, :, 0:1]
        p_attn = F.softmax(scores, dim=-1) * b.float()
        if dropout is not None:
            p_attn = dropout(p_attn)
        return p_attn

    def forward(self, query, key, score_mask):
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.head_num, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        # The zip function takes multiple iterables and combines the I-th element of each iterable to form a new iterator of type tuple.
        attn = self.attention(query, key, score_mask, dropout=self.dropout)

        return attn