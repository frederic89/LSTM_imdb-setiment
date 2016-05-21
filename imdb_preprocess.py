#  -*- coding:utf-8 -*-
"""
This script is what created the dataset pickled.

1) You need to download this file and put it in the same directory as this file.
https://github.com/moses-smt/mosesdecoder/raw/master/scripts/tokenizer/tokenizer.perl . Give it execution permission.

2) Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/ and extract it in the current directory.

3) Then run this script.
"""

dataset_path = '/home/gyq-mac/PycharmProjects/aclImdb/'

import numpy
import cPickle as pkl

from collections import OrderedDict

import glob
import os

from subprocess import Popen, PIPE

# tokenizer.perl is from Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
tokenizer_cmd = ['./tokenizer.perl', '-l', 'en', '-q', '-']


def tokenize(sentences):
    print 'Tokenizing..',
    text = "\n".join(sentences)  # 把所有的句子变成一个长文本
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
    tok_text, _ = tokenizer.communicate(text)  # 向子进程传入对象参数 text
    toks = tok_text.split('\n')[:-1] # 恢复为列表
    print 'Done'

    return toks


def build_dict(path):
    sentences = []
    currdir = os.getcwd()
    os.chdir('%s/pos/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir('%s/neg/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)

    sentences = tokenize(sentences)

    print 'Building dictionary..',
    wordcount = dict()
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = wordcount.values() # 返回一个单词的计数组成的数组
    keys = wordcount.keys()     # 返回一个单词组成的数组

    # 返回一个数组：给出从大到小降序的给出排序的位置索引,[::-1]表示反转数组
    sorted_idx = numpy.argsort(counts)[::-1] #没有[::-1]代表从小到大

    worddict = dict()
    # 新建一个dict，键是单词，值是单词的索引，索引中的“0”和“1”留给UNK
    for idx, ss in enumerate(sorted_idx):
        #每到一个idx时，就刷新一次word，依照词频从最大到最小
        #print(idx,ss)
        word = keys[ss]   # keys[ss]是一个具体的单词，按从大到小的顺序读取具体单词，并临时地(当时的idx)赋值在word变量里
        worddict[word] = idx + 2  # 生成词编号的重要步骤， leave 0 and 1 (UNK)

    print numpy.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict


def grab_data(path, dictionary):
    sentences = []
    currdir = os.getcwd()
    os.chdir(path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip()) # 把所有文件的句子变成列表
    os.chdir(currdir)
    sentences = tokenize(sentences) # sentences 返回值是所有句子中切分为单词的列表

    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]
        # ss 中的每一个words（w）转换为dictionary中的索引号(dictionary这个dict的values)

    return seqs  # 句中的每个单词转化为dictionary对应的单词索引号，所有的句子组成数组seqs返回，没有的单词索引号赋为1


def main():
    # Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/
    path = dataset_path
    dictionary = build_dict(os.path.join(path, 'train'))

    train_x_pos = grab_data(path + 'train/pos', dictionary)
    train_x_neg = grab_data(path + 'train/neg', dictionary)
    train_x = train_x_pos + train_x_neg
    train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg) # 生成正负例的y值

    test_x_pos = grab_data(path + 'test/pos', dictionary)
    test_x_neg = grab_data(path + 'test/neg', dictionary)
    test_x = test_x_pos + test_x_neg
    test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)

    f = open('imdb.pkl', 'wb')
    pkl.dump((train_x, train_y), f, -1)
    pkl.dump((test_x, test_y), f, -1)
    f.close()

    f = open('imdb.dict.pkl', 'wb')
    pkl.dump(dictionary, f, -1)
    f.close()


if __name__ == '__main__':
    main()
