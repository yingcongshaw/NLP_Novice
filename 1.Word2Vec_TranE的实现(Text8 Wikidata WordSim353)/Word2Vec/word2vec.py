import gensim
import logging
import multiprocessing
import os
import re
import sys
from gensim.models import word2vec
import numpy as np
from time import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

if __name__ == '__main__':

    data_path = 'text8'
    t = time()

    # loading dataset
    sents = word2vec.Text8Corpus('text8')

    # training word2vec
    model = gensim.models.Word2Vec(sents,
                                   size=200,
                                   window=10,
                                   min_count=10,
                                   workers=multiprocessing.cpu_count())

    # saving to file
    model.save("word2vec_gensim")
    model.wv.save_word2vec_format("word2vec_org",
                                  "vocabulary",
                                  binary=False)

    print ("Total time: %d s" % (time() - t))
    
    # testing on wordsim353
    sims = []
    with open('wordsim353/combined.csv') as f:
        for line in f.readlines()[1:]:
            l = line.strip().split(',')
            if l[0] in model.wv.vocab and l[1] in model.wv.vocab: # 过滤掉不在词表内的词
                sims.append(model.wv.similarity(l[0], l[1])) # 模型打分
    
    np.save('score.npy', np.array(sims))