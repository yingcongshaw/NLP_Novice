import torch
from torch import nn
from torch.autograd import Variable

import numpy as np
from torch_geometric.nn import GCNConv


class SimplePredictionLayer(nn.Module):
    def __init__(self, config):
        super(SimplePredictionLayer, self).__init__()
        self.input_dim = config.input_dim

        # 计算supporting fact 的logits
        self.sp_linear = nn.Linear(self.input_dim, 1)
        # 计算start位置的logits
        self.start_linear = nn.Linear(self.input_dim, 1)
        # 计算end位置的logits
        self.end_linear = nn.Linear(self.input_dim, 1)

        # 计算分类任务的输出类别的logits(# yes/no/ans)
        self.type_linear = nn.Linear(self.input_dim, config.label_type_num)   

        # 用于保存计算的S矩阵和掩码的缓存，以避免多次计算
        self.cache_S = 0
        self.cache_mask = None

    def get_output_mask(self, outer):
        '''
        获取输出掩码
        '''
        # (batch, 512, 512)
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        # triu 生成上三角矩阵，tril生成下三角矩阵，这个相当于生成了(512, 512)的矩阵表示开始-结束的位置，答案长度最长为15
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch, input_state):
        # 从批次中获取查询、上下文掩码和所有映射
        query_mapping = batch['query_mapping']  # (batch, 512) 不一定是512，可能略小
        context_mask = batch['context_mask']  # bert里实际有输入的位置
        all_mapping = batch['all_mapping']  # (batch_size, 512, max_sent) 每个句子的token对应为1

        #模型预测出的答案的开始和结束位置的logits(未经过softmax激活函数处理的处理)
        start_logits = self.start_linear(input_state).squeeze(2) - 1e30 * (1 - context_mask)
        end_logits = self.end_linear(input_state).squeeze(2) - 1e30 * (1 - context_mask)

        # 支持性分类的logits(N x sent x 512 x 300)
        sp_state = all_mapping.unsqueeze(3) * input_state.unsqueeze(2)  
        sp_state = sp_state.max(1)[0]
        # (batch_size, max_seq_len, 1)
        sp_logits = self.sp_linear(sp_state)

        # 模型预测出的答案类型的logits
        type_state = torch.max(input_state, dim=1)[0]
        type_logits = self.type_linear(type_state)

        # 找结束位置用的开始和结束位置概率之和
        # (batch, 512, 1) + (batch, 1, 512) -> (512, 512)
        outer = start_logits[:, :, None] + end_logits[:, None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        # 这个是query_mapping (batch, 512),不允许预测query的内容
        if query_mapping is not None:  
            outer = outer - 1e30 * query_mapping[:, :, None]   

        # 预测出的答案在文本中的开始和结束位置的索引,相当于找到了outer中最大值的i和j坐标
        start_position = outer.max(dim=2)[0].max(dim=1)[1]
        end_position = outer.max(dim=1)[0].max(dim=1)[1]

        # 返回sp_logits(batch_size, max_seq_len)，每个元素代表了对应输入序列的判断标准
        return start_logits, end_logits, type_logits, sp_logits.squeeze(2), start_position, end_position
    
class BertSupportNet(nn.Module):
    """
    joint train bert and graph fusion net
    """

    def __init__(self, config, encoder):
        super(BertSupportNet, self).__init__()
        # self.bert_model = BertModel.from_pretrained(config.bert_model)
        self.encoder = encoder
        self.graph_fusion_net = SupportNet(config)

    def forward(self, batch, debug=False):
        doc_ids, doc_mask, segment_ids = batch['context_idxs'], batch['context_mask'], batch['segment_idxs']
        # 获取BERT模型的第一个元素，即最后一层输出.roberta不可以输入token_type_ids
        all_doc_encoder_layers = self.encoder(input_ids=doc_ids,
                                              token_type_ids=segment_ids,#可以注释
                                              attention_mask=doc_mask)[0]
        
        # 将编码输出存储到Batch字典中
        batch['context_encoding'] = all_doc_encoder_layers

        # 将Batch字典传入graph_fusion_net模型中，得到预测结果
        return self.graph_fusion_net(batch)


class SupportNet(nn.Module):
    """
    Packing Query Version
    """

    def __init__(self, config):
        super(SupportNet, self).__init__()
        self.config = config  # 就是args
        # self.n_layers = config.n_layers  # 2
        self.max_query_length = 50
        self.prediction_layer = SimplePredictionLayer(config)

        # GNN layer
        self.gnn_layer = GCNConv(config.input_dim, 512)  

    def forward(self, batch,debug=False):
        # 获取上下文编码输出
        context_encoding = batch['context_encoding']

        # GNN的edge_index实现还有问题
        context_encoding = self.gnn_layer(context_encoding, [2,10240])

        # 将Batch和上下文编码输出传入预测层模型中，得到预测结果
        predictions = self.prediction_layer(batch, context_encoding)

        # 解析预测结果
        start_logits, end_logits, type_logits, sp_logits, start_position, end_position = predictions

        return start_logits, end_logits, type_logits, sp_logits, start_position, end_position
