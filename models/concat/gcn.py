import torch
import numpy as np
import copy #提供通用浅和深拷贝操作。
import math #Python提供的内置数学类函数库
import torch.nn as nn #包含了神经网络中使用的一些常用函数
import torch.nn.functional as F #如ReLU，pool，DropOut等
from torch.autograd import Variable #PyTorch的自动差分引擎，可为神经网络训练提供支持。
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

        self.in_drop = nn.Dropout(args.input_dropout) #Dropout：防止过拟合

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

        self.h_weight = nn.Parameter(torch.FloatTensor(2).normal_(0.5, 0.5)) #nn.Parameter（）：将一个不可训练的类型Tensor转换成可以训练的类型parameter，并且会向宿主模型注册该参数，成为一部分。
        #torch.FloatTensor()的作用就是把给定的list或者numpy转换成浮点数类型的tensor。

        # gcn layer
        self.gcn1 = GCN(args, args.hidden_dim, args.num_layers)
        self.gcn2 = GCN(args, args.hidden_dim, args.num_layers)
        self.gcn_common = GCN(args, args.hidden_dim, args.num_layers)

        # MLP Layer
        self.linear = nn.Linear(3 * self.args.num_layers * args.hidden_dim, args.hidden_dim)

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
        embs = torch.cat(embs, dim=2) #torch.cat是将两个张量（tensor）拼接在一起，cat是concatnate的意思，即拼接，联系在一起。
        embs = self.in_drop(embs)
        return embs

    def inputs_to_att_adj(self, input, score_mask):
        attn_tensor = self.attn(input, input, score_mask)  # [batch_size, head_num, seq_len, seq_len]
        attn_tensor = torch.sum(attn_tensor, dim=1)
        attn_tensor = select(attn_tensor, self.args.top_k) * attn_tensor
        return attn_tensor

    def forward(self, inputs):
        tok, asp, pos, head, post, dep, mask, l, adj = inputs           # unpack（解压缩） inputs
        # embedding
        embs = self.create_embs(tok, pos, post)

        # bi-lstm encoding
        rnn_hidden = self.encode_with_rnn(embs, l, tok.size(0))  # [batch_size, seq_len, hidden]
        score_mask = torch.matmul(rnn_hidden, rnn_hidden.transpose(-2, -1))
        score_mask = (score_mask == 0)
        score_mask = score_mask.unsqueeze(1).repeat(1, self.args.head_num, 1, 1).cuda()

        # init adj
        att_adj = self.inputs_to_att_adj(rnn_hidden, score_mask)  # [batch_size, head_num, seq_len, hidden]  #inputs)

        for i in range(self.args.num_layers):
            if i==0:#第1层
                h_sy = self.gcn1.first_layer(adj, rnn_hidden)
                h_se = self.gcn2.first_layer(att_adj, rnn_hidden)
                h_csy = self.gcn_common.first_layer(adj, rnn_hidden)
                h_cse = self.gcn_common.first_layer(att_adj, rnn_hidden)
                h_sy_inputs = torch.cat((rnn_hidden, h_sy), dim=-1)
                h_se_inputs = torch.cat((rnn_hidden, h_se), dim=-1)
                h_csy_inputs = torch.cat((rnn_hidden, h_csy), dim=-1)
                h_cse_inputs = torch.cat((rnn_hidden, h_cse), dim=-1)
                h_com = (self.h_weight[0] * h_csy + self.h_weight[1] * h_cse) / 2
            else:
                # concat the last layer's out with input_feature as the current input（用input_feature作为当前输入连接最后一层）
                h_sy = self.gcn1.next_layer(adj, h_sy_inputs,i, score_mask, 'syntax')
                h_se = self.gcn2.next_layer(att_adj, h_se_inputs,i, score_mask, 'semantic')
                h_csy = self.gcn_common.next_layer(adj, h_csy_inputs,i, score_mask, 'syntax')
                h_cse = self.gcn_common.next_layer(att_adj, h_cse_inputs,i, score_mask, 'semantic')
                h_sy_inputs = torch.cat((h_sy_inputs, h_sy), dim=-1)
                h_se_inputs = torch.cat((h_se_inputs, h_se), dim=-1)
                h_csy_inputs = torch.cat((h_csy_inputs, h_csy), dim=-1)
                h_cse_inputs = torch.cat((h_cse_inputs, h_cse), dim=-1)
                h_com = (self.h_weight[0] * h_csy + self.h_weight[1] * h_cse) / 2
            mask2=mask
            asp_wn = mask.sum(dim=1).unsqueeze(-1)  # aspect words num
            mask = mask.unsqueeze(-1).repeat(1, 1, self.args.hidden_dim)  # mask for h
            h_sy_mean = (h_sy * mask).sum(dim=1) / asp_wn
            h_se_mean = (h_se * mask).sum(dim=1) / asp_wn
            h_com_mean = (h_com * mask).sum(dim=1) / asp_wn
            if i==0:
                outputs = torch.cat((h_sy_mean, h_se_mean, h_com_mean), dim=-1)  # mask h
            else:
                outputs = torch.cat((outputs,h_sy_mean, h_se_mean, h_com_mean), dim=-1)  # mask h
            if (i+1)==self.args.num_layers:
                outputs = F.relu(self.linear(outputs))
            mask=mask2

        '''h_sy = self.gcn1(adj, rnn_hidden, score_mask, 'syntax')
        h_se = self.gcn2(att_adj, rnn_hidden, score_mask, 'semantic')
        h_csy = self.gcn_common(adj, rnn_hidden, score_mask, 'syntax')
        h_cse = self.gcn_common(att_adj, rnn_hidden, score_mask, 'semantic')
        h_com = (self.h_weight[0] * h_csy + self.h_weight[1] * h_cse) / 2'''


        # avg pooling asp feature
        # asp_wn = mask.sum(dim=1).unsqueeze(-1)                    # aspect words num
        # mask = mask.unsqueeze(-1).repeat(1, 1, self.args.hidden_dim)    # mask for h
        # h_sy_mean = (h_sy * mask).sum(dim=1) / asp_wn
        # h_se_mean = (h_se * mask).sum(dim=1) / asp_wn
        # h_com_mean = (h_com * mask).sum(dim=1) / asp_wn
        # outputs = torch.cat((h_sy_mean, h_se_mean, h_com_mean), dim=-1)     # mask h
        # outputs = F.relu(self.linear(outputs))
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

    def first_layer(self, adj, inputs):
        # gcn
        denom = adj.sum(2).unsqueeze(2) + 1  # norm adj
        out = self.GCN_layer(adj, inputs, denom, 0)
        return out

    def next_layer(self, adj, inputs,i, score_mask, type):
        # 第二层之后gcn输入的adj是根据前一层隐藏层输出求得的
        denom = adj.sum(2).unsqueeze(2) + 1  # norm adj
        if type == 'semantic':
            # att_adj
            adj = self.attn[i - 1](inputs, inputs, score_mask)  # [batch_size, head_num, seq_len, dim]

            if self.args.second_layer == 'max':
                probability = F.softmax(adj.sum(dim=(-2, -1)), dim=0)
                max_idx = torch.argmax(probability, dim=1)
                adj = torch.stack([adj[i][max_idx[i]] for i in range(len(max_idx))], dim=0)
            else:
                adj = torch.sum(adj, dim=1)

            adj = select(adj, self.args.top_k) * adj
            denom = adj.sum(2).unsqueeze(2) + 1  # norm adj

        out = self.GCN_layer(adj, inputs, denom, i)
        return out

    '''def forward(self, adj, inputs, score_mask, type):
        # gcn
        denom = adj.sum(2).unsqueeze(2) + 1  # norm adj
        out = self.GCN_layer(adj, inputs, denom, 0)
        # 第二层之后gcn输入的adj是根据前一层隐藏层输出求得的

        for i in range(1, self.layers):
            # concat the last layer's out with input_feature as the current input（用input_feature作为当前输入连接最后一层）
            inputs = torch.cat((inputs, out), dim=-1)

            if type == 'semantic':
                # att_adj
                adj = self.attn[i - 1](inputs, inputs, score_mask)  # [batch_size, head_num, seq_len, dim]

                if self.args.second_layer == 'max':
                    probability = F.softmax(adj.sum(dim=(-2, -1)), dim=0)
                    max_idx = torch.argmax(probability, dim=1)
                    adj = torch.stack([adj[i][max_idx[i]] for i in range(len(max_idx))], dim=0)
                else:
                    adj = torch.sum(adj, dim=1)

                adj = select(adj, self.args.top_k) * adj
                denom = adj.sum(2).unsqueeze(2) + 1  # norm adj

            out = self.GCN_layer(adj, inputs, denom, i)
        return out'''



def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape))  # 改为param
    return h0.cuda(), c0.cuda()


def select(matrix, top_num):
    batch = matrix.size(0)
    len = matrix.size(1)
    matrix = matrix.reshape(batch, -1)
    maxk, _ = torch.topk(matrix, top_num, dim=1)

    for i in range(batch):
        matrix[i] = (matrix[i] >= maxk[i][-1]) #给matrix[i]赋1或0
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
                             for l, x in zip(self.linears, (query, key))] #zip 函数是可以接收多个可迭代对象，然后把每个可迭代对象中的第i个元素组合在一起，形成一个新的迭代器，类型为元组。
        attn = self.attention(query, key, score_mask, dropout=self.dropout)

        return attn