
import collections
from transformers import  BasicTokenizer, BertTokenizer
import logging


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """
    Project the tokenized prediction back to the original text.
    通过字符对齐的方式将预测的分词文本投影回原始文本    
    """

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        
        return (ns_text, ns_to_s_map)

    
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    # 对原始文本进行分词
    tok_text = " ".join(tokenizer.tokenize(orig_text))

    # 在分词后的文本中查询预测文本的起始位置 
    start_position = tok_text.find(pred_text)
    if start_position == -1:   
        if verbose_logging:
            print("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    # 将去除空格后的原始文本和分词后的文本的字符对齐
    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):   
        if verbose_logging:
            logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # 使用字符对齐将分词后的文本字条索引映射回原始的字符索引
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    # 映射起始位置
    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]
    if orig_start_position is None:
        if verbose_logging:
            print("Couldn't map start position")
        return orig_text

    # 映射结束位置
    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]
    if orig_end_position is None:
        if verbose_logging:
            print("Couldn't map end position")
        return orig_text

    # 从原始文本中提取预测文本
    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def convert_to_tokens(example, features, ids, y1, y2, q_type):
    '''
    将预测结果转换为回答文本。根据问题类型和答案位置，将预测结果转换为相应的回答文本，
    并返回一个字典，其中键是问题的ID，值是对应的回答文本。
    '''
    answer_dict = dict()
    
    # 根据问题类型和答案位置将结果转换为回答文本
    for i, qid in enumerate(ids):   # article id
        answer_text = ''
        if q_type[i] == 0:
            # 获取文章的分词列表
            doc_tokens = features[qid].doc_tokens
            # 获取答案的分词列表
            tok_tokens = doc_tokens[y1[i]: y2[i] + 1]
            # 获取分词对应原始文本的映射关系
            tok_to_orig_map = features[qid].token_to_orig_map
            if y2[i] < len(tok_to_orig_map):   
                # 原始文本中答案的起始和结束位置
                orig_doc_start = tok_to_orig_map[y1[i]]
                orig_doc_end = tok_to_orig_map[y2[i]]
                # 根据映射关系获取原始文本中的答案文本
                orig_tokens = example[qid].doc_tokens[orig_doc_start:(orig_doc_end + 1)]  
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())  
                orig_text = " ".join(orig_tokens).strip('[,.;]')    

                # 将预测的分词文本投影回原始文本
                final_text = get_final_text(tok_text, orig_text, do_lower_case=False, verbose_logging=False)
                answer_text = final_text
        elif q_type[i] == 1:
            answer_text = 'yes'
        elif q_type[i] == 2:
            answer_text = 'no'
        elif q_type[i] == 3:
            answer_text = 'unknown'
        answer_dict[qid] = answer_text
    return answer_dict

