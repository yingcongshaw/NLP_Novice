import numpy as np
import os
import json
import argparse
from transformers import BertTokenizer, RobertaTokenizer

MODEL_CLASSES = {
    'bert': BertTokenizer,
    'roberta': RobertaTokenizer
}

# 将训练数据中出现过的所有关系类型映射为其对应的唯一标识符。方便模型的训练和评估过程中对关系类型进行统一处理。
rel2id = {'P1376': 79, 'P607': 27, 'P136': 73, 'P137': 63, 'P131': 2, 'P527': 11, 'P1412': 38, 'P206': 33, 'P205': 77, 'P449': 52, 'P127': 34, 'P123': 49, 'P86': 66, 'P840': 85, 'P355': 72, 'P737': 93, 'P740': 84, 'P190': 94, 'P576': 71, 'P749': 68, 'P112': 65, 'P118': 40, 'P17': 1, 'P19': 14, 'P3373': 19, 'P6': 42, 'P276': 44, 'P1001': 24, 'P580': 62, 'P582': 83, 'P585': 64, 'P463': 18, 'P676': 87, 'P674': 46, 'P264': 10, 'P108': 43, 'P102': 17, 'P25': 81, 'P27': 3, 'P26': 26, 'P20': 37, 'P22': 30, 'Na': 0, 'P807': 95, 'P800': 51, 'P279': 78, 'P1336': 88, 'P577': 5, 'P570': 8, 'P571': 15, 'P178': 36, 'P179': 55, 'P272': 75, 'P170': 35, 'P171': 80, 'P172': 76, 'P175': 6, 'P176': 67, 'P39': 91, 'P30': 21, 'P31': 60, 'P36': 70, 'P37': 58, 'P35': 54, 'P400': 31, 'P403': 61, 'P361': 12, 'P364': 74, 'P569': 7, 'P710': 41, 'P1344': 32, 'P488': 82, 'P241': 59, 'P162': 57, 'P161': 9, 'P166': 47, 'P40': 20, 'P1441': 23, 'P156': 45, 'P155': 39, 'P150': 4, 'P551': 90, 'P706': 56, 'P159': 29, 'P495': 13, 'P58': 53, 'P194': 48, 'P54': 16, 'P57': 28, 'P50': 22, 'P1366': 86, 'P1365': 92, 'P937': 69, 'P140': 50, 'P69': 25, 'P1198': 96, 'P1056': 89}


def main():

    parser = argparse.ArgumentParser()
    # 指定预训练模型的类型。常见的模型类型包括BERT、GPT、RoBERTa等。
    parser.add_argument("--model_type", type=str, default="bert")
    # 如果指定了预训练模型的名称（例如"bert-base-uncased"），则会从Hugging Face的模型仓库中下载该模型。
    # 如果指定了本地目录的路径，则会加载本地保存的模型。
    # 如果指定的是已经 fine-tuned 过的模型，那么可以直接使用该模型进行推理。
    # 如果指定的是一个预训练模型，则可以继续进行 fine-tuning，以便针对特定的任务进行微调。
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-cased")
    # 在模型训练或推理时是否将文本转换为小写字母。
    # 小写字母比大写字母更加通用，同时也可以降低词表大小，缩短训练和推理时间。
    # 但是，处理的文本包含有意义的大写字母，比如人名或专有名词，那么可能丢失这些信息。
    parser.add_argument("--do_lower_case", action="store_true")
    # 源数据集的存放文件夹地址
    parser.add_argument('--data_dir', type = str, default =  "docred_data")
    # 模型训练完后（如转化为BERT格式的数据集）输出的目录
    parser.add_argument('--output_dir', type = str, default = "prepro_data")
    # 输入文本最大长度，超过最大长度的部分将被截断，不足最大长度的部分将被填充。取决于输入数据集的文本长度分布。通常，值在128到512之间。
    parser.add_argument('--max_seq_length', type = int, default = 512)

    args = parser.parse_args()
    model_type = args.model_type
    data_dir = args.data_dir
    output_dir = args.output_dir
    max_seq_length = args.max_seq_length

    assert(model_type in ['bert', 'roberta'])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_annotated_file_name = os.path.join(data_dir, 'train_annotated.json')
    dev_file_name = os.path.join(data_dir, 'dev.json')
    test_file_name = os.path.join(data_dir, 'test.json')

    fact_in_annotated_train = set([])

    #加载模型
    tokenizer = MODEL_CLASSES[model_type].from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    def save_data_format(ori_data, is_training):
        data = []
        for i in range(len(ori_data)):
            # sents的长度
            Ls = [0]
            L = 0
            for x in ori_data[i]['sents']:
                L += len(x)
                Ls.append(L)

            vertexSet =  ori_data[i]['vertexSet']
            # point position added with sent start position
            for j in range(len(vertexSet)):
                for k in range(len(vertexSet[j])):
                    vertexSet[j][k]['sent_id'] = int(vertexSet[j][k]['sent_id'])

                    sent_id = vertexSet[j][k]['sent_id']
                    dl = Ls[sent_id]
                    pos1 = vertexSet[j][k]['pos'][0]
                    pos2 = vertexSet[j][k]['pos'][1]
                    vertexSet[j][k]['pos'] = (pos1+dl, pos2+dl)

            ori_data[i]['vertexSet'] = vertexSet

            item = {}
            item['vertexSet'] = vertexSet
            labels = ori_data[i].get('labels', [])

            train_triple = set([])
            new_labels = []
            for label in labels:
                # 关系类型 如'P1376'
                rel = label['r']
                assert(rel in rel2id)
                # 把关系类型关映射为其对应的唯一标识符，如'P1376'-> 79
                label['r'] = rel2id[label['r']]

                # 有问题相关的三元组索引
                train_triple.add((label['h'], label['t']))


                label['in_annotated_train'] = False
                # question: 这里的fact_in_annotated_train代表什么？
                # answer: 知识图谱的符号三元组（RDF）。用于存储训练数据中看到的唯一（头实体、尾实体）对。
                #         如果同一对实体之间的两个关系被不同地注释，那么在训练集中只保留其中一个关系，以避免在训练期间混淆模型。
                if is_training:
                    for n1 in vertexSet[label['h']]:
                        for n2 in vertexSet[label['t']]:
                            fact_in_annotated_train.add((n1['name'], n2['name'], rel))
                else:
                    for n1 in vertexSet[label['h']]:
                        for n2 in vertexSet[label['t']]:
                            if (n1['name'], n2['name'], rel) in fact_in_annotated_train:
                                label['in_annotated_train'] = True


                new_labels.append(label)

            item['labels'] = new_labels
            item['title'] = ori_data[i]['title']

            # question: 这里的na_triple代表什么？
            # answer:数据集中没有与问题相关的三元组集合（更确切地说，是该头实体和尾实体的索引集合）。
            #       以方便后面将三元组中的每一个元素替换为“不可用”([unused])三元组。
            na_triple = []
            for j in range(len(vertexSet)):
                for k in range(len(vertexSet)):
                    if (j != k):
                        if (j, k) not in train_triple:
                            na_triple.append((j, k))

            item['na_triple'] = na_triple
            item['Ls'] = Ls
            item['sents'] = ori_data[i]['sents']
            data.append(item)
        return data

    def init(data_file_name, rel2id, max_seq_length = 512, is_training = True, suffix=''):
        ori_data = json.load(open(data_file_name))
        data = save_data_format(ori_data, is_training)
        print ('data_len:', len(data))

        # 将dict类型的数据转成str类型，并写入到json文件
        print("Saving files")
        json.dump(data , open(os.path.join(output_dir, suffix + '.json'), "w"))
        json.dump(rel2id , open(os.path.join(output_dir, 'rel2id.json'), "w"))


        sen_tot = len(ori_data)
        # 保存被Bert模型处理后的token化的输入文本
        bert_token = np.zeros((sen_tot, max_seq_length), dtype = np.int64)
        # 保存被Bert模型对应的mask信息，用于指示输入文本中的哪些部分是padding的，哪些部分是有意义的
        bert_mask = np.zeros((sen_tot, max_seq_length), dtype = np.int64)
        # question: 这里的bert_starts_ends代表什么？
        # answer: 保存了每个Token在原始文本中的起始和终位置信息
        bert_starts_ends = np.ones((sen_tot, max_seq_length, 2), dtype = np.int64) * (max_seq_length - 1)

        if model_type=='bert':
            # 循环遍历所有的原始数据
            for i in range(len(ori_data)):
                item = ori_data[i]
                tokens = []
                # 所有句子的词
                for sent in item['sents']:
                    tokens += sent
                #使用tokenizer把词分割为subwords，再将所有subwords拼接成一个长的list：flatten_subwords（map函数和列表推导式进行操作）
                subwords = list(map(tokenizer.tokenize, tokens))
                subword_lengths = list(map(len, subwords))
                flatten_subwords = [x for x_list in subwords for x in x_list ]

                # 将[CLS]和[SEP] token添加到tokens列表的开头和结尾。
                tokens = [ tokenizer.cls_token ] + flatten_subwords[: max_seq_length - 2] + [ tokenizer.sep_token ]
                # question: token_start_idxs的公式中为什么要加1
                # answer:加1是因为在BERT中，第一个token是特殊标记[CLS]，变像的等于真正句子中的起始位置为1，而不是0。
                token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
                # question: 为什么要把token_start_idxs >= max_seq_length-1的都置为max_seq_length - 1
                # answer:在数据预处理时，需要将长度超过max_seq_length的句子截断； 
                #       截断后的句子，需要对应地调整其对应的实体的起始和结束位置
                token_start_idxs[token_start_idxs >= max_seq_length-1] = max_seq_length - 1
                # 将前面所有元素的和计算出来，并加上当前元素的值，即可得到当前元素在tokens中的位置。
                token_end_idxs = 1 + np.cumsum(subword_lengths)
                token_end_idxs[token_end_idxs >= max_seq_length-1] = max_seq_length - 1

                # 将tokens转换为BERT的输入id，然后用0进行填充，以保证所有的输入都有相同的长度。
                tokens = tokenizer.convert_tokens_to_ids(tokens)
                pad_len = max_seq_length - len(tokens)
                # 使用1和0构造一个mask列表，以指示哪些输入是实际的token，哪些是填充的。
                mask = [1] * len(tokens) + [0] * pad_len
                tokens = tokens + [0] * pad_len

                bert_token[i] = tokens
                bert_mask[i] = mask 

                bert_starts_ends[i, :len(subword_lengths), 0] = token_start_idxs
                bert_starts_ends[i, :len(subword_lengths), 1] = token_end_idxs
        else:
            # 循环遍历所有的原始数据
            for i in range(len(ori_data)):
                item = ori_data[i]
                # 将其所有句子的单词拼接在一起，获得完整的文本
                words = []
                for sent in item['sents']:
                    words += sent

                # 将原始文本按照空格分为单词，将单词合并为一个字符串
                idxs = []
                text = ""
                for word in words:
                    if len(text)>0:
                        text = text  + " "
                    idxs.append(len(text))
                    text += word
                
                # 分词器将这个字符串分为子词，得到一个子词序列。
                subwords = tokenizer.tokenize(text)

                # 通过char2subwords列表来跟踪文本中每个字符对应的子词索引
                char2subwords = []
                # 将文本中第一个字符所对应的子词的索引设置为0，
                # 然后迭代处理每个字符，将其所对应的子词的索引附加到char2subwords列表中
                L = 0
                sub_idx = 0
                L_subwords = len(subwords)
                while sub_idx < L_subwords:
                    subword_list = []
                    prev_sub_idx = sub_idx
                    while sub_idx < L_subwords:
                        subword_list.append(subwords[sub_idx])
                        sub_idx += 1

                        subword = tokenizer.convert_tokens_to_string(subword_list)
                        sub_l = len(subword)
                        if text[L:L+sub_l]==subword:
                            break
                    
                    # 确保char2subwords的长度与text相同
                    # 将所对应的子词的索引附加到char2subwords列表中
                    assert(text[L:L+sub_l]==subword)
                    char2subwords.extend([prev_sub_idx] * sub_l)

                    L += len(subword)

                if len(text) > len(char2subwords):
                    text = text[:len(char2subwords)]

                assert(len(text)==len(char2subwords))
                # 将[CLS]和[SEP] token添加到tokens列表的开头和结尾。
                tokens = [ tokenizer.cls_token ] + subwords[: max_seq_length - 2] + [ tokenizer.sep_token ]

                L_ori = len(tokens)
                # 子词列表转换为整数ID，以便模型可以理解
                tokens = tokenizer.convert_tokens_to_ids(tokens)

                # 在列表的末尾填充0，以使其长度为max_seq_length
                pad_len = max_seq_length - len(tokens)
                mask = [1] * len(tokens) + [0] * pad_len
                tokens = tokens + [0] * pad_len

                bert_token[i] = tokens
                bert_mask[i] = mask 

                # 对于每个单词，我们首先计算其对应的第一个子词的索引。
                # 然后，我们确定单词的长度，并在char2subwords列表中找到其最后一个字符所对应的子词的索引。
                # 最后，我们在bert_starts_ends数组中为该单词存储起始和结束索引。
                for j in range(len(words)):
                    idx = char2subwords[idxs[j]] + 1
                    idx = min(idx, max_seq_length-1)

                    x = idxs[j] + len(words[j])
                    if x == len(char2subwords):
                        idx2 = L_ori
                    else:
                        idx2 = char2subwords[x] + 1
                        idx2 = min(idx2, max_seq_length-1)

                    bert_starts_ends[i][j][0] = idx
                    bert_starts_ends[i][j][1] = idx2

        print("Finishing processing")
        np.save(os.path.join(output_dir, suffix + '_bert_token.npy'), bert_token)
        np.save(os.path.join(output_dir, suffix + '_bert_mask.npy'), bert_mask)
        np.save(os.path.join(output_dir, suffix + '_bert_starts_ends.npy'), bert_starts_ends)
        print("Finish saving")


    init(train_annotated_file_name, rel2id, max_seq_length = max_seq_length, is_training = True, suffix='train')
    init(dev_file_name, rel2id, max_seq_length = max_seq_length, is_training = False, suffix='dev')
    init(test_file_name, rel2id, max_seq_length = max_seq_length, is_training = False, suffix='test')


if __name__ == '__main__':
    main()
