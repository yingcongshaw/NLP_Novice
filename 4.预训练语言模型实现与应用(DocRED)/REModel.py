import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
import math

class REModel(nn.Module):
    def __init__(self, config, bert_model):
        super(REModel, self).__init__()
        self.config = config
        self.use_distance = True

        # 隐藏层维度
        hidden_size = 256
        self.bert = bert_model

        # BERT模型的隐藏层维度
        bert_hidden_size = self.bert.config.hidden_size
        # 线性变换层，将BERT输出的特征维度降到设定的hidden_size
        self.linear_re = nn.Linear(bert_hidden_size, hidden_size)

        # 使用特征提取时计算每对实体对在句子中的距离
        if self.use_distance:
            self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)
            # 对于输入的数据进行双线性的变换
            self.bili = torch.nn.Bilinear(hidden_size+config.dis_size, hidden_size+config.dis_size, config.relation_num)
        else:
            self.bili = torch.nn.Bilinear(hidden_size, hidden_size, config.relation_num)

    def forward(self, context_idxs, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h, context_masks):
        # 输入BERT模型中得到的上下文特征. context_idxs 为token的序列.
        context_output = self.bert(context_idxs, attention_mask=context_masks)[0]
        # 经过线性变换后，特征维度变为hidden_size
        context_output = self.linear_re(context_output)
        # 经过映射，得到头实体和尾实体的特征
        # 二维时，就是普通的矩阵乘法; 多维时，把多出的一维作为batch提出来，其他部分做矩阵乘法。
        start_re_output = torch.matmul(h_mapping, context_output)
        end_re_output = torch.matmul(t_mapping, context_output)

        if self.use_distance:
            # 分别将头实体和尾实体特征与实体间距离特征进行拼接
            # dis_h_2_t 和dis_t_2_h表示两个实体之间的距离，context_masks为padding mask
            s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
            t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)
            # 通过双线性变换得到预测结果
            predict_re = self.bili(s_rep, t_rep)
        else:
            # 通过双线性变换得到预测结果
            predict_re = self.bili(start_re_output, end_re_output)
            
        # 返回预测结果
        return predict_re
