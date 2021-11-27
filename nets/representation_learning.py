import os
import time
from tqdm import tqdm
import torch
import math
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.nn import DataParallel

# from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to
import torch.nn as nn
from torch.autograd import Variable

# from nets.attention_model import AttentionModel
from nets.graph_encoder import GraphAttentionEncoder


# represatation learning!!!
class RepLearn(nn.Module):
    def __init__(self, z_dim, batch_size, output_type="continuous"):
        super(RepLearn, self).__init__()
        self.batch_size = batch_size
        self.encoder = GraphAttentionEncoder(n_heads=8, embed_dim=128, n_layers=2)
        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.cpc_optimizer = torch.optim.Adam(
            RepLearn.parameters(self), lr=1e-3
        )

    def encode(self, x, detach=False, ema=False):
        '''
        Encoder: z_t = e(x_t)
        :para x = x_t, x y coordinates
        :return z_t, value in r2
        '''
        if ema:
            with torch.no_grad():
                z_out = self.encoder(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()

        return z_out

    def compute_logits(self, z_a, z_pos):
        '''
        Uses logits trick for RepLearn：
        '''
        # print(len(z_pos))
        # print(z_pos[0].size())
        Wz = torch.matmul(self.W, z_pos.T)
        # print(Wz.size())
        logits = torch.matmul(z_a, Wz)
        # print(logits.size())
        logits = logits - torch.max(logits, 1)[0][:, None]
        # print(logits.size())
        # exit()
        return logits

class SACRepLearn(object):
    '''Represatation learning with soft actor critic'''

    def __init__(self):

        self.encoder = GraphAttentionEncoder(n_heads=8, embed_dim=128, n_layers=2)

        self.RepLearn = RepLearn(z_dim=128,
                                 batch_size=512,
                                 output_type='continuous'
                                 ).cuda()

        # optimizer for critic encoder for reconstruction loss
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=1e-3
        )

        self.cpc_optimizer = torch.optim.Adam(
            self.RepLearn.parameters(), lr=1e-3
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def update_cpc(self, obs_anchor, obs_pos, selected):
        # z_a = self.RepLearn.encode(obs_anchor)
        # z_pos = self.RepLearn.encode(obs_pos, ema=True)
        # print(z_a[1].size(), z_pos[1].size())
        # # torch.Size([256, 128]) torch.Size([256, 128])

        z_a = obs_anchor
        z_pos = obs_pos
        # print(z_a.size())     # torch.Size([256, 21, 128])
        # print(z_pos.size())   # torch.Size([256, 1, 128])

        # print(z_a[0].size()) # torch.Size([512, 20, 128])
        # print(z_a[1].size()) # torch.Size([512, 128])

        # logits = self.RepLearn.compute_logits(z_a[1], z_pos[1])
        logits = self.RepLearn.compute_logits(z_a, z_pos.squeeze(1))
        # labels = torch.arange(logits.shape[0]).long().cuda()
        labels = selected
        loss = self.cross_entropy_loss(logits, labels)

        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        loss = Variable(loss, requires_grad=True)
        loss.backward(retain_graph=True)

        self.encoder_optimizer.step()
        self.cpc_optimizer.step()

# class LSTM(nn.Module):
#     def __init__(self):
#         super(LSTM, self).__init__()
#         # 此处的参数还需要调整一下
#         self.lstm = nn.LSTM(
#             input_size=1,
#             hidden_size=32,
#             num_layers=1,
#             batch_first=True,
#         )
#         self.hidden = (torch.autograd.Variable(torch.zeros(1, 1, 32)),torch.autograd.Variable(torch.zeros(1, 1, 32)))
#         self.out = nn.Linear(32, 1)
#
#     def forward(self, x):
#         # x (batch, time_step, input_size)
#         # h_state (n_layers, batch, hidden_size)
#         # r_out (batch, time_step, output_size)
#         r_out, _ = self.lstm(x)
#         # hidden_state 也要作为 RNN 的一个输入
#         self.hidden = (Variable(self.hidden[0]),
#                        Variable(self.hidden[1]))
#         # 可以把这一步去掉，在loss.backward（）中加retain_graph=True，主要是Varible有记忆功能，而张量没有
#         outs = []  # 保存所有时间点的预测值
#         for time_step in range(r_out.size(1)):  # 对每一个时间点计算 output
#             outs.append(self.out(r_out[:, time_step, :]))
#         return torch.stack(outs, dim=1)

class LSTM(nn.Module):
    def __init__(self, input_size=128, hidden_size=32, output_size=128, seq_len=10, is_bidir=False, dropout_p=0):
        super(LSTM, self).__init__();
        # batch_size, seq_len, input_size(embedding_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.is_bidir = is_bidir
        self.dropout_p = dropout_p
        self.rnn = nn.LSTM(self.input_size, self.hidden_size,
                            num_layers=2,
                            batch_first=True,
                            dropout=self.dropout_p,
                            bidirectional=self.is_bidir).cuda()
        self.fc_input_size = 2*self.hidden_size if self.is_bidir else self.hidden_size
        self.dropout = nn.Dropout(p=self.dropout_p).cuda()
        self.linear = nn.Linear(self.fc_input_size , self.output_size).cuda()

    def forward(self, x, seq_len):
        # x: [batch_size, seq_len, input_size]
        output, (hidden, cell) = self.rnn(x)
        batch_size, seq_len, hidden_size = output.shape
        output = output.view(batch_size, seq_len, hidden_size)
        output = self.linear(output)
        output = self.dropout(output)
        return output[:,-1,:].view(-1,1) # [batch_size, output_size]

class CPCLayer(nn.Module):

    ''' Computes dot product between true and predicted embedding vectors '''

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):

        # Compute dot product among vectors
        preds, y_encoded = inputs
        dot_product = torch.mean(y_encoded * preds, dim=-1)
        dot_product = torch.mean(dot_product, dim=-1, keepdims=True)  # average along the temporal dimension

        # Keras loss functions take probabilities
        dot_product_probs = torch.sigmoid(dot_product)

        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

class network_cpc(nn.Module):
    def __init__(self, batch_size):
        super(network_cpc, self).__init__()

        # Define encoder model
        self.batch_size = batch_size
        self.encoder = GraphAttentionEncoder(n_heads=8, embed_dim=128, n_layers=2)
        # self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        # self.output_type = output_type
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=1e-3
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.cpc_optimizer = torch.optim.Adam(
            network_cpc.parameters(self), lr=1e-3
        )
        self.LSTM = LSTM()

    def network_autoregressive(self, x, step_num):
        ''' Define the network that integrates information along the sequence '''

        # x = keras.layers.GRU(units=256, return_sequences=True)(x)
        # x = keras.layers.BatchNormalization()(x)
        x = self.LSTM(x, seq_len=step_num).cuda()
        # x = model_ar(x)

        return x

    def network_prediction(self, context, code_size, predict_terms):

        ''' Define the network mapping context to multiple embeddings '''

        output = []
        for i in range(predict_terms):
            # print(context.size())
            # exit()
            output.append(nn.Linear(1, 1).cuda()(context))

        # for i in range(predict_terms):
        #     outputs.append(
        #         keras.layers.Dense(units=code_size, activation="linear", name='z_t_{i}'.format(i=i))(context))
        #
        # if len(outputs) == 1:
        #     output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
        # else:
        #     output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(outputs)

        return output

    def RepLearning(self, input_context, input_y, terms, predict_terms, code_size, learning_rate, selected, step_num):

        # 输入term（x_input）个点，network_autoregressive整出来context
        # 之后network_prediction用来预测preds
        # y_input是实际得到的点（AM）， 输入predict_terms个点，实际这里就一个
        # 然后计算LOSS，dot_product_probs = CPCLayer()([preds, y_encoded])（没想明白怎么搞）
        # 原文此处应该有一个binary_crossentropy在cpc model里，没整明白怎么加进去

        # 此处input包含node_embedding，需要自己拆分出来context和选定的点

        # 选择已经走过的点
        x_encoded = input_context
        context = self.network_autoregressive(x_encoded, step_num=step_num)
        preds = self.network_prediction(context, code_size, predict_terms)
        preds = torch.stack(preds).view(-1, 1, 128)

        # def network_prediction(self, context, code_size, predict_terms):

        # 下一步要选择的点，selected的最后一维度
        y_encoded = input_y.detach()
        # y_encoded = Variable(y_encoded, requires_grad=True)

        # loss = self.cross_entropy_loss(logits, labels)
        # loss = F.binary_cross_entropy(torch.sigmoid(x_encoded), y_encoded)
        # loss = F.binary_cross_entropy(torch.sigmoid(context.view(-1, 1, 128)), y_encoded)
        loss = F.binary_cross_entropy(torch.sigmoid(preds), y_encoded)


        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        loss = Variable(loss, requires_grad=True)
        loss.backward(retain_graph=True)

        self.encoder_optimizer.step()
        self.cpc_optimizer.step()

        # dot_product = torch.mean(y_encoded * preds, dim=-1)
        # dot_product = torch.mean(dot_product, dim=-1, keepdims=True)  # average along the temporal dimension
        #
        # # Keras loss functions take probabilities
        # dot_product_probs = torch.sigmoid(dot_product)

        # 应该看这个rep和哪一个更接近，然后就选择哪一个，然后和实际的y作比较，一个标0，一个标1，做binary entropy
        # logits = dot_product_probs
        # label 需要重新整一下
        # labels = selected
        
        # return dot_product_probs
       

