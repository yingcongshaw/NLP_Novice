# coding=utf-8
# This is a sample Python script.

import sys
# 用于打印JSON编码的评分结果
import json
import numpy as np
from scipy.stats import spearmanr


def eval(submit_file):
    sims = np.load(submit_file[1])
    score = np.load('ground_truth.npy')
    # 计算Spearman相关系数
    spcor = spearmanr(score, sims)[0]
    # 以下返回值主要用于aistudio的比赛，但是本次实验不设置比赛，大家只用看score的值
    return {
        "score": spcor,              #替换value为最终评测分数
        "errorMsg": "success",      #错误提示信息，仅在code值为非0时打印
        "code": 0,                  #code值为0打印score，非0打印errorMsg
        "data": [
            {
                "score": spcor 
            }
        ]
    }

if __name__ == '__main__':
    # 打印格式必须为JSON编码的字符串
    print(json.dumps(eval(sys.argv)))
