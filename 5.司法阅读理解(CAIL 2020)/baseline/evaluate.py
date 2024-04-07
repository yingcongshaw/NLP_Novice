import sys
import ujson as json
import re
import string
from collections import Counter
import pickle

def normalize_answer(s):
    '''
    答案标准化
    '''

    def remove_articles(text):
        '''
        移除冠词（a,an,the）
        '''
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        '''
        去除多余的空格
        '''
        return ' '.join(text.split())

    def remove_punc(text):
        '''
        移除标点符号
        '''
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    '''
    计算F1分数。
    - prediction: 预测的答案字符串
    - ground_truth: 实际的答案字符串
    返回：
    - F1分数、精确率和召回率
    '''
    # 对预测答案和实际答案进行标准化处理
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    # 定义ZERO_METRIC为(0, 0, 0)，表示当预测和实际答案为特殊值（'yes', 'no', 'unknown'）且不相等时的度量指标
    ZERO_METRIC = (0, 0, 0)
    # 如果标准化后的预测答案是特殊值且与标准化后的实际答案不相等，返回ZERO_METRIC
    if normalized_prediction in ['yes', 'no', 'unknown'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    # 如果标准化后的实际答案是特殊值且与标准化后的预测答案不相等，返回ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'unknown'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    # 将预测答案和实际答案转换为字符列表
    prediction_tokens = list(normalized_prediction)
    ground_truth_tokens = list(normalized_ground_truth)
    # 计算预测答案和实际答案的公共字符频数
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    # 计算相同字符的数量
    num_same = sum(common.values())
    # 如果相同字符数量为0，返回ZERO_METRIC
    if num_same == 0:
        return ZERO_METRIC
    # 计算精确率、召回率和F1
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    # 返回F1分数、精确率和召回率
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    '''
    判断预测答案与实际答案标准化后是否完全匹配。
    参数：
    - prediction: 预测的答案字符串
    - ground_truth: 实际的答案字符串
    返回值：
    - 是否完全匹配（True/False）
    '''
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    '''
    更新评估指标。
    参数：
    - metrics: 存储评估指标的字典
    - prediction: 预测的答案字符串
    - gold: 实际的答案字符串
    返回值：
    - 完全匹配指标（em）、精确率（prec）和召回率（recall）
    '''
    # 预测答案与实际答案的完全匹配指标（em）
    em = exact_match_score(prediction, gold)
    # 预测答案与实际答案的F1分数（f1）、精确率（prec）和召回率（recall）
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def update_sp(metrics, prediction, gold):
    '''
    更新支持文本预测的评估指标。
    参数：
    - metrics: 存储评估指标的字典
    - prediction: 预测的支持文本列表
    - gold: 实际的支持文本列表
    返回值：
    - 完全匹配指标（em）、精确率（prec）和召回率（recall）
    '''
    # 将预测的支持文本列表和实际的支持文本列表转换为集合
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    # 初始化真阳性（tp）、假阳性（fp）和假阴性（fn）的计数为0
    tp, fp, fn = 0, 0, 0
    # 遍历预测的支持文本列表：
    for e in cur_sp_pred:
        # 如果当前支持文本在实际的支持文本列表中，增加tp计数
        if e in gold_sp_pred:
            tp += 1        
        else:
            fp += 1
    for e in gold_sp_pred:
        # 如果当前支持文本不在预测的支持文本列表中，增加fn计数
        if e not in cur_sp_pred:
            fn += 1
    # 计算精确率（prec）、召回率（recall）和F1分数（f1）
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    # 如果fp和fn都为0，将完全匹配指标（em）、精确率、召回率和F1分数都设为1
    if fp + fn == 0:
        em, prec, recall, f1 = 1.0, 1.0, 1.0, 1.0
    else:
        em = 0.0
    # 将完全匹配指标、精确率、召回率加到metrics中
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    # 只返回完全匹配指标、精确率和召回率
    return em, prec, recall

def eval(prediction_file, gold_file):
    '''
    评估预测结果与真实答案的性能。
    参数：
    - prediction_file: 预测结果文件路径
    - gold_file: 真实答案文件路径
    '''
    # 从预测结果文件和真实答案文件中加载数据
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)

    # 初始化评估指标的字典metrics
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}
    for dp in gold:
        # 当前样本的ID
        cur_id = str(dp['_id'])

        # can_eval_joint用于表示是否可以评估联合结果的指标。
        # 在评估过程中，有两个条件需要满足才能计算联合结果的指标（联合精确率、联合召回率和联合F1分数）：
        # 1.答案预测结果中存在当前样本的答案 (cur_id)。
        # 2.支持文本事实预测结果中存在当前样本的支持文本事实。
        # 如果其中任何一个条件不满足，即预测结果中缺少答案或支持文本事实的预测结果，就无法计算联合结果的指标。在这种情况下，将can_eval_joint设置为False，表示无法评估联合结果。
        can_eval_joint = True
        # 如果当前ID在预测结果中找不到答案，打印"missing answer"并将can_eval_joint设为False
        if cur_id not in prediction['answer']:
            print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        # 否则，调用update_answer函数更新评估指标，并将结果保存到em、prec和recall中
        else:
            em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], dp['answer'])
        if cur_id not in prediction['sp']:
            print('missing sp fact {}'.format(cur_id))
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, prediction['sp'][cur_id], dp['supporting_facts'])
            
        #如果可以评估联合结果
        if can_eval_joint:
            # 计算联合精确率（joint_prec）、联合召回率（joint_recall）和F1分数（joint_f1）
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            # 更新联合完全匹配指标（joint_em）
            joint_em = em * sp_em

            # 将联合指标加到metrics中
            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    N = len(gold)
    for k in metrics.keys():
        # 计算评估指标的平均值
        metrics[k] /= N

    # 打印评估指标
    print(metrics)

if __name__ == '__main__':
    eval(sys.argv[1], sys.argv[2])