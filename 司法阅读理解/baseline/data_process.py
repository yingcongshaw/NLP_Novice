from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import gzip
import pickle
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizer




class Example(object):

    def __init__(self,
                 qas_id,
                 qas_type,
                 doc_tokens,
                 question_text,
                 sent_num,
                 sent_names,
                 sup_fact_id,
                 para_start_end_position,
                 sent_start_end_position,
                 entity_start_end_position,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.qas_type = qas_type
        self.doc_tokens = doc_tokens
        self.question_text = question_text
        self.sent_num = sent_num
        self.sent_names = sent_names
        self.sup_fact_id = sup_fact_id
        self.para_start_end_position = para_start_end_position
        self.sent_start_end_position = sent_start_end_position
        self.entity_start_end_position = entity_start_end_position
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

    def gatherAttrs(self):
        return ",".join("{}={}"
                        .format(k, getattr(self, k))
                        for k in self.__dict__.keys())

    def __str__(self):
        return "[{}:{}]".format(self.__class__.__name__, self.gatherAttrs())

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 qas_id,
                 doc_tokens,
                 doc_input_ids,
                 doc_input_mask,
                 doc_segment_ids,
                 query_tokens,
                 query_input_ids,
                 query_input_mask,
                 query_segment_ids,
                 sent_spans,
                 sup_fact_ids,
                 ans_type,
                 token_to_orig_map,
                 start_position=None,
                 end_position=None):
        # 样本的问题ID
        self.qas_id = qas_id
        # 文档标记
        self.doc_tokens = doc_tokens
        # 文档输入ID
        self.doc_input_ids = doc_input_ids
        # 文档输入掩码
        self.doc_input_mask = doc_input_mask
        # 文档片段标识符
        self.doc_segment_ids = doc_segment_ids

        # 查询标记
        self.query_tokens = query_tokens
        # 查询输入ID
        self.query_input_ids = query_input_ids
        # 查询输入掩码
        self.query_input_mask = query_input_mask
        # 查询片段标识符
        self.query_segment_ids = query_segment_ids

        # 句子位置
        self.sent_spans = sent_spans
        # 支持事实ID列表
        self.sup_fact_ids = sup_fact_ids
        # 答案类型（0，1，2，3）
        self.ans_type = ans_type
        # 标记到原始文档位置的映射关系
        self.token_to_orig_map=token_to_orig_map

        # 答案的起始位置列表
        self.start_position = start_position
        # 答案的结束位置列表
        self.end_position = end_position




def check_in_full_paras(answer, paras):
    '''
    将给定的答案answer是否存在于一段段落paras中进行检查
    '''
    full_doc = ""
    for p in paras:
        full_doc += " ".join(p[1])
    return answer in full_doc


def read_examples( full_file, full_data_2019):
    '''
    从文件中读取案例
    '''
    with open(full_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)    

    if full_data_2019:
           with open(full_file, 'r', encoding='utf-8') as reader:
                full_data_2019 = json.load(reader)  
                full_data.extend(full_data_2019)  

    def is_whitespace(c):
        '''
        检查是否为空白字符
        '''
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    # 计数器
    cnt = 0
    examples = []
    for case in tqdm(full_data):   
        key = case['_id']
        qas_type = "" # case['type']
        sup_facts = set([(sp[0], sp[1]) for sp in case['supporting_facts']])   
        sup_titles = set([sp[0] for sp in case['supporting_facts']]) 
        orig_answer_text = case['answer']

        sent_id = 0
        doc_tokens = []
        sent_names = []
        sup_facts_sent_id = []
        sent_start_end_position = []
        para_start_end_position = []
        entity_start_end_position = []
        ans_start_position, ans_end_position = [], []

        JUDGE_FLAG = orig_answer_text == 'yes' or orig_answer_text == 'no' or orig_answer_text=='unknown'  or orig_answer_text=="" # judge_flag??
        FIND_FLAG = False

        # 沿所有句子累积
        char_to_word_offset = [] 
        prev_is_whitespace = True

        # 所有案例的标题
        titles = set()
        para_data=case['context']
        # 迭代案例中的案例内容，获取段落标题和句子。
        for paragraph in para_data:  
            title = paragraph[0]
            sents = paragraph[1]   

            titles.add(title)  
            is_gold_para = 1 if title in sup_titles else 0  

            para_start_position = len(doc_tokens)  

            for local_sent_id, sent in enumerate(sents):  
                if local_sent_id >= 100:  
                    break

                # Determine the global sent id for supporting facts
                local_sent_name = (title, local_sent_id)   
                sent_names.append(local_sent_name) 
                # 如果句子名称在支持事实集合中，则将对应的全局句子ID添加到列表中 
                if local_sent_name in sup_facts:
                    sup_facts_sent_id.append(sent_id)   
                sent_id += 1   
                # 将句子拼接成字符串
                sent=" ".join(sent)
                sent += " "

                sent_start_word_id = len(doc_tokens)           
                sent_start_char_id = len(char_to_word_offset)  

                # 遍历句子中和每个字符
                for c in sent:  
                    # 判断字符是否为空白字符
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        # 如果前一个字符为空白字符，则新添加
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        # 如果前一个字符不空白字符，则追加到最后一个元素上
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                # 将字符对应的词级偏移添加到列表中
                sent_end_word_id = len(doc_tokens) - 1  
                sent_start_end_position.append((sent_start_word_id, sent_end_word_id))  

                # Answer char position
                answer_offsets = []
                offset = -1

                tmp_answer=" ".join(orig_answer_text)
                while True:
                    # 在句子中查找答案的偏移量
                    offset = sent.find(tmp_answer, offset + 1)
                    if offset != -1:
                        answer_offsets.append(offset)   
                    else:
                        break

                # answer_offsets = [m.start() for m in re.finditer(orig_answer_text, sent)]
                if not JUDGE_FLAG and not FIND_FLAG and len(answer_offsets) > 0:
                    # 找到答案，将答案词级偏移添加到位置列表中
                    FIND_FLAG = True   
                    for answer_offset in answer_offsets:
                        start_char_position = sent_start_char_id + answer_offset   
                        end_char_position = start_char_position + len(tmp_answer) - 1  
                       
                        ans_start_position.append(char_to_word_offset[start_char_position])
                        ans_end_position.append(char_to_word_offset[end_char_position])



               
                if len(doc_tokens) > 382:                   
                    break

            # 记录当前段落的结束位置
            para_end_position = len(doc_tokens) - 1
            # 将段落的超始位置、结束位置、标题和是否为支持事实的标志添加到列表中
            para_start_end_position.append((para_start_position, para_end_position, title, is_gold_para))  

        if len(ans_end_position) > 1:
            cnt += 1    
        if key <10:
            print("qid {}".format(key))
            print("qas type {}".format(qas_type))
            print("doc tokens {}".format(doc_tokens))
            print("question {}".format(case['question']))
            print("sent num {}".format(sent_id+1))
            print("sup face id {}".format(sup_facts_sent_id))
            print("para_start_end_position {}".format(para_start_end_position))
            print("sent_start_end_position {}".format(sent_start_end_position))
            print("entity_start_end_position {}".format(entity_start_end_position))
            print("orig_answer_text {}".format(orig_answer_text))
            print("ans_start_position {}".format(ans_start_position))
            print("ans_end_position {}".format(ans_end_position))
       
        example = Example(
            qas_id=key,
            qas_type=qas_type,
            doc_tokens=doc_tokens,
            question_text=case['question'],
            sent_num=sent_id + 1,
            sent_names=sent_names,
            sup_fact_id=sup_facts_sent_id,
            para_start_end_position=para_start_end_position, 
            sent_start_end_position=sent_start_end_position,
            entity_start_end_position=entity_start_end_position,
            orig_answer_text=orig_answer_text,
            start_position=ans_start_position,   
            end_position=ans_end_position)
        examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length):
    '''
    把样本转化为特征
    '''
    # max_query_length = 50

    # 存储特征的列表
    features = []
    # 记录处理失败的数量
    failed = 0
    # 遍历输入的样本数据
    for (example_index, example) in enumerate(tqdm(examples)):  
        if example.orig_answer_text == 'yes':
            ans_type = 1
        elif example.orig_answer_text == 'no':
            ans_type = 2
        elif example.orig_answer_text == 'unknown':
            ans_type = 3
        else:
            ans_type = 0   # 统计answer type
        
        # 初始化查询文本的标记列表，以"[CLS]"开始
        query_tokens = ["[CLS]"]
        for token in example.question_text.split(' '):
            # 使用tokenizer对查询文本进行分词，并加入到query_tokens中
            query_tokens.extend(tokenizer.tokenize(token))
        if len(query_tokens) > max_query_length - 1:
            # 限制查询文本的最大长度
            query_tokens = query_tokens[:max_query_length - 1]
        # 在查询文本末尾加入"[SEP]"标记，表示查询文本的结束
        query_tokens.append("[SEP]")

        # para_spans = []
        # entity_spans = []
        # 句子位置的列表
        sentence_spans = []
        # 整个文档的标记列表
        all_doc_tokens = []
        # 原始文档位置到标记列表的映射
        orig_to_tok_index = []
        # 标记到原始文档位置的映射
        orig_to_tok_back_index = []
        # 存储标记到原始文档位置的映射，初始值为0
        tok_to_orig_index = [0] * len(query_tokens)

        # 初始化整个文档的标记列表，以"[CLS]"开始。同上面初始化查询文本的标记列表。
        all_doc_tokens = ["[CLS]"]   
        for token in example.question_text.split(' '):
            all_doc_tokens.extend(tokenizer.tokenize(token))
        if len(all_doc_tokens) > max_query_length - 1:
            all_doc_tokens = all_doc_tokens[:max_query_length - 1]
        all_doc_tokens.append("[SEP]")

        # 遍历原始文档中的每个标记
        for (i, token) in enumerate(example.doc_tokens):  
            # 记录当前标记在all_doc_tokens中的位置  
            orig_to_tok_index.append(len(all_doc_tokens))  
            # 使用tokenizer对当前标记进行分词
            sub_tokens = tokenizer.tokenize(token)
            #将当前标记与原始文档中的位置建立映射关系，tok_to_orig_index的长度与all_doc_tokens相同，对应位置的值为原始文档中的位置。
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i) 
                # 将分词后的子标记加入到all_doc_tokens中   
                all_doc_tokens.append(sub_token)
            # 记录当前标记分词后的最后一个子标记在all_doc_tokens中的位置，用于后续重定位标记位置。
            orig_to_tok_back_index.append(len(all_doc_tokens) - 1)  



        def relocate_tok_span(orig_start_position, orig_end_position, orig_text):
            '''
            重新定位答案在标记中的起始和结束位置
            '''
            if orig_start_position is None:  
                return 0, 0
            # 通过orig_to_tok_index找到对应的标记位置
            tok_start_position = orig_to_tok_index[orig_start_position]
            if orig_end_position < len(example.doc_tokens) - 1:  
                tok_end_position = orig_to_tok_index[orig_end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1  

            # 返回重新定位后的标记起始和结束位置
            return _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, orig_text)

        ans_start_position, ans_end_position = [], []
        # 遍历样本中的答案起始和结束位置
        for ans_start_pos, ans_end_pos in zip(example.start_position, example.end_position): 
            # 通过调用relocate_tok_span函数将其重新定位 
            s_pos, e_pos = relocate_tok_span(ans_start_pos, ans_end_pos, example.orig_answer_text)
            # 重新定位后的起始和结束位置存储在ans_start_position和ans_end_position列表中
            ans_start_position.append(s_pos)  
            ans_end_position.append(e_pos)

        
        # 遍历示例中的句子起始和结束位置
        for sent_span in example.sent_start_end_position:   
            if sent_span[0] >= len(orig_to_tok_index) or sent_span[0] >= sent_span[1]:
                continue  
            # 通过orig_to_tok_index和orig_to_tok_back_index将其转换为标记的起始和结束位置
            sent_start_position = orig_to_tok_index[sent_span[0]] 
            sent_end_position = orig_to_tok_back_index[sent_span[1]] 
            # 将转换后的句子位置存储在sentence_spans列表中
            sentence_spans.append((sent_start_position, sent_end_position)) 

        

        # 将整个文档的标记列表截断到最大序列长度-1，并在末尾添加"[SEP]"标记，形成模型输入的文档标记列表。
        all_doc_tokens = all_doc_tokens[:max_seq_length - 1] + ["[SEP]"]
        # 使用tokenizer将 文档标记列表 转换为对应的输入id列表
        doc_input_ids = tokenizer.convert_tokens_to_ids(all_doc_tokens)
        # 使用tokenizer将 查询标记列表 转换为对应的输入id列表
        query_input_ids = tokenizer.convert_tokens_to_ids(query_tokens)

        # 文档的输入掩码标记文档的有效部分为1，填充部分为0。
        doc_input_mask = [1] * len(doc_input_ids)
        # 片段标识符用于区分查询部分和文档部分：查询部分的标识符为0，文档部分的标识符为1。
        doc_segment_ids = [0] * len(query_input_ids) + [1] * (len(doc_input_ids) - len(query_input_ids))

        # 如果文档的输入id列表长度小于最大序列长度，则在末尾填充0，使其长度达到最大序列长度。
        while len(doc_input_ids) < max_seq_length:
            doc_input_ids.append(0)
            doc_input_mask.append(0)
            doc_segment_ids.append(0)

        # 查询的输入掩码标记查询的有效部分为1，填充部分为0。
        query_input_mask = [1] * len(query_input_ids)
        # 查询部分的片段标识符都为0。
        query_segment_ids = [0] * len(query_input_ids)

        # 如果查询的输入id列表长度小于最大查询长度，则在末尾填充0，使其长度达到最大查询长度。
        while len(query_input_ids) < max_query_length:
            query_input_ids.append(0)
            query_input_mask.append(0)
            query_segment_ids.append(0)

        # 确保文档和查询的输入id列表、输入掩码和片段标识符的长度符合预期的最大长度。
        assert len(doc_input_ids) == max_seq_length
        assert len(doc_input_mask) == max_seq_length
        assert len(doc_segment_ids) == max_seq_length
        assert len(query_input_ids) == max_query_length
        assert len(query_input_mask) == max_query_length
        assert len(query_segment_ids) == max_query_length

        # 调用get_valid_spans，对句子位置进行处理，确保句子位置在最大序列长度内。
        sentence_spans = get_valid_spans(sentence_spans, max_seq_length)
       
        # 获取支持事实的id列表（sup_fact_ids），并将其限制在句子位置数量内。
        sup_fact_ids = example.sup_fact_id
        sent_num = len(sentence_spans)
        sup_fact_ids = [sent_id for sent_id in sup_fact_ids if sent_id < sent_num]
        # 如果限制后的支持事实id列表长度与原始长度不一致，则增加失败计数。
        if len(sup_fact_ids) != len(example.sup_fact_id):
            failed += 1
        if example.qas_id <10:
            print("qid {}".format(example.qas_id))
            print("all_doc_tokens {}".format(all_doc_tokens))
            print("doc_input_ids {}".format(doc_input_ids))
            print("doc_input_mask {}".format(doc_input_mask))
            print("doc_segment_ids {}".format(doc_segment_ids))
            print("query_tokens {}".format(query_tokens))
            print("query_input_ids {}".format(query_input_ids))
            print("query_input_mask {}".format(query_input_mask))
            print("query_segment_ids {}".format(query_segment_ids))
            print("sentence_spans {}".format(sentence_spans))
            print("sup_fact_ids {}".format(sup_fact_ids))
            print("ans_type {}".format(ans_type))
            print("tok_to_orig_index {}".format(tok_to_orig_index))
            print("ans_start_position {}".format(ans_start_position))
            print("ans_end_position {}".format(ans_end_position))

        features.append(
            InputFeatures(qas_id=example.qas_id,
                          doc_tokens=all_doc_tokens,
                          doc_input_ids=doc_input_ids,
                          doc_input_mask=doc_input_mask,
                          doc_segment_ids=doc_segment_ids,
                          query_tokens=query_tokens,
                          query_input_ids=query_input_ids,
                          query_input_mask=query_input_mask,
                          query_segment_ids=query_segment_ids,
                          sent_spans=sentence_spans,
                          sup_fact_ids=sup_fact_ids,
                          ans_type=ans_type,
                          token_to_orig_map=tok_to_orig_index,
                          start_position=ans_start_position,
                          end_position=ans_end_position)
        )
    return features


def _largest_valid_index(spans, limit):
    '''
    返回第一个结束位置大于等于限制值limit的spans索引
    '''
    for idx in range(len(spans)):
        if spans[idx][1] >= limit:
            return idx


def get_valid_spans(spans, limit):
    '''
    检查每个spans的结束位置是否小于限制值。
    如果有大于限定值，则起始位置相同，结束位置设置为限定值减一
    '''
    new_spans = []
    for span in spans:
        if span[1] < limit:
            new_spans.append(span)
        else:
            new_span = list(span)
            new_span[1] = limit - 1
            new_spans.append(tuple(new_span))
            break
    return new_spans


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # 使用tokenizer对原始答案文本进行分词
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    # 在给定的输入范围内遍历所有可能的起始和结束位置组合。
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            # 将文档tokens中对应的标记连接起来形成一个文本片段
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            # 检查是否与分词后的答案文本匹配(匹配则返回新的起始和结束位置)
            if text_span == tok_answer_text:
                return new_start, new_end

    # 不匹配则返回输入的起始和结束位置。
    return input_start, input_end


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--example_output", required=True, type=str)
    parser.add_argument("--feature_output", required=True, type=str)

    # Other parameters
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=15, type=int, help="Batch size for predictions.")
    parser.add_argument("--full_data", type=str, required=True)   
    parser.add_argument("--full_data_2019", type=str, required=False)   
    parser.add_argument('--tokenizer_path',type=str,required=True)


    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    # tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_path)
    
    # 从文件中读取样本
    examples = read_examples( full_file=args.full_data, full_data_2019 = args.full_data_2019 )
    with gzip.open(args.example_output, 'wb') as fout:
        pickle.dump(examples, fout)

    # 把样本转化为特征
    features = convert_examples_to_features(examples, tokenizer, max_seq_length=512, max_query_length=50)
    with gzip.open(args.feature_output, 'wb') as fout:
        pickle.dump(features, fout)











