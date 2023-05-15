import torch
import numpy as np
from numpy.random import shuffle

IGNORE_INDEX = -100


class DataIteratorPack(object):
    def __init__(self, features, example_dict,bsz, device, sent_limit, entity_limit,
                 entity_type_dict=None, sequential=False,):
        self.bsz = bsz   # batch_size
        self.device = device
        self.features = features
        self.example_dict = example_dict
        # self.entity_type_dict = entity_type_dict
        self.sequential = sequential
        self.sent_limit = sent_limit
        # self.para_limit = 4 
        # self.entity_limit = entity_limit
        # 示例指针的初妈位置
        self.example_ptr = 0
        if not sequential:
            shuffle(self.features)  

    def refresh(self):
        '''
        重置示例指针
        '''
        self.example_ptr = 0
        # 按乱序迭代对特征数据进行重置
        if not self.sequential:
            shuffle(self.features)

    def empty(self):
        '''
        判断示例指针是否超出特征数据范围
        '''
        return self.example_ptr >= len(self.features)

    def __len__(self):
        '''
        返回批次数量
        '''
        return int(np.ceil(len(self.features)/self.bsz))

    def __iter__(self):
        '''
        生成迭代器（根据数据的属性，构建多个张量并将特征数据的值拷贝到相应的张量中，
        最后返回一个包含多个张量的数据字典）
        '''
        # BERT input
        # 存储上下文的词索引。(批大小，上下文中的最大词数)
        context_idxs = torch.LongTensor(self.bsz, 512)
        # 存储上下文的掩码，表示哪些位置是有效的词
        context_mask = torch.LongTensor(self.bsz, 512)
        # 存储上下文的片段标识符，用于区分不同的句子片段
        segment_idxs = torch.LongTensor(self.bsz, 512)
       
        # 存储查询的位置映射
        query_mapping = torch.Tensor(self.bsz, 512).cuda(self.device)
        # 存储句子开始位置的映射
        start_mapping = torch.Tensor(self.bsz, self.sent_limit, 512).cuda(self.device)
        # 存储所有句子位置的映射
        all_mapping = torch.Tensor(self.bsz, 512, self.sent_limit).cuda(self.device)


        # Label tensor
        # 存储答案的起始位置
        y1 = torch.LongTensor(self.bsz).cuda(self.device)   
        # 存储答案的结束位置
        y2 = torch.LongTensor(self.bsz).cuda(self.device)
        # 存储问题类型的标签
        q_type = torch.LongTensor(self.bsz).cuda(self.device)   
        # 存储句子是否为支持句的标签
        is_support = torch.FloatTensor(self.bsz, self.sent_limit).cuda(self.device)


        

        while True:
            # 如果当前示例指针超过了数据集的长度，结束循环
            if self.example_ptr >= len(self.features):
                break
            # 获取当前批次的起始索引和批次大小
            start_id = self.example_ptr  
            cur_bsz = min(self.bsz, len(self.features) - start_id)   
            # 获取当前批次的特征
            cur_batch = self.features[start_id: start_id + cur_bsz]  
            # 根据输入掩码的总和对当前批次进行排序，以便进行填充
            cur_batch.sort(key=lambda x: sum(x.doc_input_mask), reverse=True)  

            ids = []
            max_sent_cnt = 0
            max_entity_cnt = 0
            # 初始化映射相关的张量
            for mapping in [start_mapping, all_mapping,  query_mapping]:
                mapping.zero_()   
           
            is_support.fill_(0)
            
            # 遍历当前批次下的每个样例
            for i in range(len(cur_batch)):    
                case = cur_batch[i]            
                # print(f'all_doc_tokens is {case.doc_tokens}')
                # 将输入张量复制到相应的变量中
                context_idxs[i].copy_(torch.Tensor(case.doc_input_ids))
                context_mask[i].copy_(torch.Tensor(case.doc_input_mask))
                segment_idxs[i].copy_(torch.Tensor(case.doc_segment_ids))
                
                # 为输入中的每个单词设置查询映射
                for j in range(case.sent_spans[0][0] - 1):
                    query_mapping[i, j] = 1

                
                # 根据答案类型设置标签张量
                if case.ans_type == 0:
                    if len(case.end_position) == 0:
                        y1[i] = y2[i] = 0   
                    elif case.end_position[0] < 512:
                        y1[i] = case.start_position[0]   
                        y2[i] = case.end_position[0]
                    else:
                        y1[i] = y2[i] = 0
                    q_type[i] = 0
                elif case.ans_type == 1:
                    y1[i] = IGNORE_INDEX  
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 1  
                elif case.ans_type == 2:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 2
                elif case.ans_type == 3:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 3

                # 根据句子跨度设置是映射和是否为支持句的标签
                for j, sent_span in enumerate(case.sent_spans[:self.sent_limit]):   
                    is_sp_flag = j in case.sup_fact_ids   
                    start, end = sent_span
                    if start < end:  
                        is_support[i, j] = int(is_sp_flag)   
                        all_mapping[i, start:end+1, j] = 1   
                        start_mapping[i, j, start] = 1       

                # 将样例的ID添加到ids列表中
                ids.append(case.qas_id)
                max_sent_cnt = max(max_sent_cnt, len(case.sent_spans))

            # 计算输入长度和最大上下文长度
            input_lengths = (context_mask[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            # 更新示例指针
            self.example_ptr += cur_bsz

            yield {
                'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                'context_mask': context_mask[:cur_bsz, :max_c_len].contiguous(),
                'segment_idxs': segment_idxs[:cur_bsz, :max_c_len].contiguous(),
                'query_mapping': query_mapping[:cur_bsz, :max_c_len].contiguous(),
                'y1': y1[:cur_bsz],
                'y2': y2[:cur_bsz],
                'ids': ids,
                'q_type': q_type[:cur_bsz],
                'start_mapping': start_mapping[:cur_bsz, :max_sent_cnt, :max_c_len],
                'all_mapping': all_mapping[:cur_bsz, :max_c_len, :max_sent_cnt],
                
                'is_support':is_support[:cur_bsz, :max_sent_cnt].contiguous(),
            }
