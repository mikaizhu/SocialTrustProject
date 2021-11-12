# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:50:39 2019

@author: zxy
"""


#使用5个小组成员的语料库进行训练
#挑出连接5个小组的成员，那他们是普通用户呢还是意见领袖呢？
#用这些用户作为意见领袖，去影响其余小组成员之间的影响
#在5个小组中找到一些普通用户


import string
import numpy as np
import pandas as pd

from nltk import word_tokenize #切词
from nltk import pos_tag#标注词性
from nltk import WordNetLemmatizer#词形还原

import pickle

def getStopWords(fileName):
	with open(fileName) as f_stop:
		sw = [line.strip() for line in f_stop.readlines() if line.strip()]
	return set(list(string.punctuation) + [i for i in range(10)] + sw)

def parallelizePreprocess(df,func,partitionsNum):
	df_split = np.array_split(df,partitionsNum)
	with mp.Pool(partitionsNum) as p:
		df_return = pd.concat(p.map(func,df_split))
	return df_return

def corpusPreprocess(df):
	#1、替换特殊字符
	df["text"] = df.text.apply(replaceWords)
	#2、切词
	df["text"] = df.text.apply(word_tokenize)
	#3、标注词性
	df["text"] = df.text.apply(pos_tag)
	#4、单词还原
	df["text"] = df.text.apply(restore)
	#5、去除标点符号和停用词
	df["text"] = df.text.apply(deleteStopWords)
	#6、除去包含有数字的词语
	df["text"] = df.text.apply(detail_number_word)
	#7、计算文档单词长度
	df["allLength"] = df.text.apply(calcLength)
	return df

def transform2Bagwords(dict_,series):
	return [dict_.doc2bow(text) for text in series.values]

wnl = WordNetLemmatizer()#创建词性还原器
stopWords = getStopWords("stopwords.txt")

replaceWords = lambda x: x.replace("{COMMA}",",").replace("{RET}","\n").replace("{APOST}","'").lower()

wnl = WordNetLemmatizer()
def lemmatize_all(params):
	word = params[0]
	tag = params[1]
	if tag.startswith('NN'):
		return wnl.lemmatize(word, pos='n')
	elif tag.startswith('VB'):
		return wnl.lemmatize(word, pos='v')
	elif tag.startswith('JJ'):
		return wnl.lemmatize(word, pos='a')
	elif tag.startswith('R'):
		return wnl.lemmatize(word, pos='r')
	else:
		return word

restore = lambda word_pos: [lemmatize_all(element) for element in word_pos]

deleteStopWords = lambda x: [i for i in x if i not in stopWords]

detail_number_word = lambda x: [i for i in x if str.isalpha(i)]

calcLength = lambda x: len(x)


if __name__ == "__main__":

	group_list = [117291968282998,1239798932720607,1443890352589739,25160801076,335787510131917]
	listname = []#9187
	datalist = []
	for groupID in group_list:
		a_corpus = pd.read_csv("corpus/{}.csv".format(groupID),index_col="id")
		a_corpus["groupID"] = groupID
		listname.extend(a_corpus.userID.tolist())
		datalist.append(a_corpus)
	setname = list(set(listname))#6166
	#有3000人跨小组讨论
	result = pd.concat(datalist)
	#result = result.astype({"text":str})
	result2 = result.groupby("userID").apply(sum)
	result2["userID"] = result2.index#index和userID是完全一致的

	if 0:#将5个小组成员的预料汇总
		#data中times等于0，说明帖子发了之后没有被评论转发之类的
		data = pd.concat([result2, pd.DataFrame(columns=["allLength"])])
		#2、并行预处理语料库,去除说话长度小的
		num_cores = mp.cpu_count()-1#多核训练
		data = parallelizePreprocess(data,corpusPreprocess,num_cores)

		#删除文本长度小于wordFrequentThreshold的记录
		wordFrequentThreshold = 200
		final_data = data[data.allLength>=wordFrequentThreshold]

		if 0:#测试
			sum(data.index==data.userID)==len(data)
			sum(final_data.index==final_data.userID) ==len(final_data)

		with open("corpus/fiveGroups_corpus_dataframe",'wb') as f:
			pickle.dump(final_data,f)

