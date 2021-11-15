#TODO

#使用5个小组成员的语料库进行训练
#挑出连接5个小组的成员，那他们是普通用户呢还是意见领袖呢？
#用这些用户作为意见领袖，去影响其余小组成员之间的影响
#在5个小组中找到一些普通用户


# part1 自然语言数据处理部分

"""
使用模块：nltk
操作步骤：
    - 文本数据清洗
        1、替换特殊字符
	    2、切词
	    3、标注词性(方便后面还原)
	    4、单词还原(去掉单词的词缀，提取单词的主干部分)
	    5、去除标点符号和停用词
	    6、除去包含有数字的词语
	    7、计算文档单词长度
comment post like member 中，一共五个社交小组，每个小组有对应的gid(group id)
corpus数据中，有每个(5个)小组的互动信息, 小组成员可能重叠
"""
import string
import numpy as np
import pandas as pd

from nltk import word_tokenize #切词
from nltk import pos_tag#标注词性
from nltk import WordNetLemmatizer#词形还原

import multiprocessing as mp

import pickle

group_list = [117291968282998,1239798932720607,1443890352589739,25160801076,335787510131917]
listname = []#5个小组一共9187人
datalist = []
for groupID in group_list:
	a_corpus = pd.read_csv("corpus/{}.csv".format(groupID),index_col="id")
	a_corpus["groupID"] = groupID
	listname.extend(a_corpus.userID.tolist())
	datalist.append(a_corpus)
setname = list(set(listname))#5 group 去重后，一共6166
#有3000人跨小组讨论
result = pd.concat(datalist)
#result = result.astype({"text":str})
# result2 是统计每个id用户，互动总数, 包括postTimes， commentTimes
# 下面需要手动下载nltk中的
# 1. punk 2. wordnet
# 在谷歌中搜索nltk punk/english pickle 和 wordnet下载，每个文件大小差不多10MB
# 然后放在你的conda环境中，像我的为
"""
~/miniconda3/envs/asr/bin/nltk_data目录结构如下
.
├── corpora
│   ├── wordnet
│   └── wordnet.zip
└── tokenizers
    └── punkt
"""
result2 = result.groupby("userID").apply(sum)
result2["userID"] = result2.index#index和userID是完全一致的
data = pd.concat([result2, pd.DataFrame(columns=["allLength"])])

def getStopWords(fileName):
	with open(fileName) as f_stop:
		sw = [line.strip() for line in f_stop.readlines() if line.strip()]
	return set(list(string.punctuation) + [i for i in range(10)] + sw)

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

stopWords = getStopWords("stopwords.txt")
wnl = WordNetLemmatizer()#创建词性还原器
replaceWords = lambda x: x.replace("{COMMA}",",").replace("{RET}","\n").replace("{APOST}","'").lower()
restore = lambda word_pos: [lemmatize_all(element) for element in word_pos]
deleteStopWords = lambda x: [i for i in x if i not in stopWords]
detail_number_word = lambda x: [i for i in x if str.isalpha(i)]
calcLength = lambda x: len(x)


num_cores = mp.cpu_count()-1 # 多核训练
data = parallelizePreprocess(data,corpusPreprocess,num_cores)

# 删除文本长度小于wordFrequentThreshold的记录
wordFrequentThreshold = 200
final_data = data[data.allLength>=wordFrequentThreshold]

with open("data/fiveGroups_corpus_dataframe",'wb') as f:
	pickle.dump(final_data,f)

# part2 lda 主题模型训练部分，寻找每个文本的主题
# Lda 主题模型，必须使用词频训练，然后再调用sklearn中的lda模型

with open("corpus/fiveGroups_corpus_dataframe",'rb') as f:
    data = pickle.load(f)

docLst = data.text.tolist()
word2str = [" ".join(list_) for list_ in docLst]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib
import os

tf_vectorizer = CountVectorizer(max_df=0.7, min_df=40)
tf = tf_vectorizer.fit_transform(word2str)
# tf.shape # 3506个文本 ， 7287个单词, 表示这个单词在这篇文章中出现的次数
os.makedirs("models/fiveGroups",exist_ok=True) # 递归生成文件，如果为True，文件如果存在则不会报错
with open("models/fiveGroups/term_frequence_transformer",'wb') as f:
	pickle.dump(tf_vectorizer,f)

"""
开始选取主题：一共有3506个文本，为每个文本挑出20个主题到101个主题之间，这是个
超参数，需要测试不同效果如何

词频矩阵讲解 countvectorizer：https://zhuanlan.zhihu.com/p/56531952
"""

num_cores = mp.cpu_count()-1#多核训练
for n_topics in range(20,101):
    print("######################topic:{}######################".format(n_topics))
    os.makedirs("models/fiveGroups/{}_topic".format(n_topics),exist_ok=True)

    #verbose是用来定义日志输出的等级的，0就是无输出，1简化输出，2细致输出
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=1000, max_doc_update_iter=100, evaluate_every=1,mean_change_tol=1e-3,
    learning_method='batch', n_jobs=num_cores, random_state=9966,verbose=2)
    lda.fit(tf)
    #保存模型
    with open("models/fiveGroups/{}_topic/ldamodel".format(n_topics),'wb') as f:
        pickle.dump(lda,f)
    joblib.dump(lda, "models/fiveGroups/{}_topic/ldamodel.pkl".format(n_topics))

# part3 选出5个小组的stubborn 互信度
"""
LDA: https://zhuanlan.zhihu.com/p/36394491
"""
import joblib
import scipy.io as sio
import os
import matplotlib.pyplot as plt

def paintTopicDistri(matrix,version,n_topics):
	plt.style.use("classic")#设置图形风格
	fig2 = plt.figure()
	ax2 = plt.axes()
	im = ax2.imshow(matrix.T,interpolation='none', vmin=0, vmax=1)
	plt.colorbar(im,orientation='horizontal',shrink=1)
	ax2.set_xlabel("doc index")
	ax2.set_ylabel("topic index")

	fig2.set_facecolor("white")
	ax2.set_title("opinion distribution")
	#ax2.axis("off")
	ax2.tick_params(direction="out",which='major',width=0.02,length=0.5,top='off',right='off',labelsize=6)
	# Hide the right and top spines
	ax2.spines['right'].set_linewidth(0.01)
	ax2.spines['top'].set_linewidth(0.01)
	ax2.spines['left'].set_linewidth(0.01)
	ax2.spines['bottom'].set_linewidth(0.01)

	fig2.savefig("models/fiveGroups/{}_topic/solution{}_X.pdf".format(n_topics,version))


def paintWordDistri(matrix,n_topics):
	plt.style.use("classic")#设置图形风格
	fig2 = plt.figure()
	ax2 = plt.axes()
	im = ax2.imshow(matrix,interpolation='none', vmin=0, vmax=1)
	plt.colorbar(im,orientation='horizontal',shrink=1)
	ax2.set_xlabel("word index")
	ax2.set_ylabel("topic index")

	fig2.set_facecolor("white")
	ax2.set_title("opinion distribution")
	#ax2.axis("off")
	ax2.tick_params(direction="out",which='major',width=0.02,length=0.5,top='off',right='off',labelsize=6)
	# Hide the right and top spines
	ax2.spines['right'].set_linewidth(0.01)
	ax2.spines['top'].set_linewidth(0.01)
	ax2.spines['left'].set_linewidth(0.01)
	ax2.spines['bottom'].set_linewidth(0.01)
	fig2.savefig("models/fiveGroups/{}_topic/word_distr.pdf".format(n_topics))



def paint_tsne(distribution):
	#plt.style.use("classic")#设置图形风格
	from sklearn.manifold import TSNE
	x = TSNE(n_components=2).fit_transform(distribution)
	rng = np.random.RandomState(500)
	rng.shuffle(colors)
	for num, (x1,x2) in enumerate(x):
		c = colors[labelList[num]]
		plt.scatter(x1,x2,s=10,c=c)
	plt.show()
	#paint_tsne(solution2_topic_distri_all)

with open("corpus/fiveGroups_corpus_dataframe",'rb') as f:
	data = pickle.load(f)

with open("models/fiveGroups/term_frequence_transformer",'rb') as f:
	tf_vectorizer = pickle.load(f)

data = data.astype({"groupID":np.int64})
showTable = data.drop(["text"],axis=1)

data = data.astype({"groupID":np.int64})
group_list = [117291968282998,1239798932720607,1443890352589739,25160801076,335787510131917]
#					1373						20 				130 					272 			18
data = data[data.groupID==117291968282998]
showTable = showTable[showTable.groupID==117291968282998]
print(len(showTable))

#第一套方案：选出活跃的当做stubborn，不活跃的当做normal
#post阈值a
a = 1
#interact阈值b
b = 10
#beInteracted阈值c
c = 10
#经常发帖，经常互动，经常被互动 则为stubborn
#其余为normal
solution1_showTable = showTable

solution1_stubbornIDs = showTable[(showTable.postTimes>=a) & (showTable.interactTimes>=b) & (showTable.beInteractedTimes>=c)].index
solution1_normalIDs = list(set(showTable.index) - set(solution1_stubbornIDs))

solution1_stubbornTable = showTable.loc[solution1_stubbornIDs]
solution1_normalTable = showTable.loc[solution1_normalIDs]

#Passing list-likes to .loc or [] with any missing label will raise KeyError in the future, you can use .reindex() as an alternative.
solution1_stubbrn_data = data.loc[solution1_stubbornTable.userID.index].dropna()
solution1_normals_data = data.loc[solution1_normalTable.userID.index].dropna()
solution1_corpusDataFrame = pd.concat([solution1_stubbrn_data,solution1_normals_data])
print("经过语料库初步处理后的stubbrn有{}个".format(solution1_stubbrn_data.shape[0]))
print("经过语料库初步处理后的normal有{}个".format(solution1_normals_data.shape[0]))
rng = np.random.RandomState(0)
solution1_sIDs = rng.choice(range(solution1_stubbrn_data.shape[0]),58,replace=False).tolist()
solution1_nIDS = rng.choice(range(solution1_normals_data.shape[0]),58*2,replace=False).tolist()
solution1_stubbrn = solution1_corpusDataFrame.iloc[solution1_sIDs]
solution1_normal = solution1_corpusDataFrame.iloc[solution1_nIDS]

solution1_finalCorpusDataframde = pd.concat([solution1_stubbrn,solution1_normal])
solution1_docLst = solution1_finalCorpusDataframde.text.tolist()
solution1_word2str = [" ".join(list_) for list_ in solution1_docLst]
#处理单词, 超过max_df的文章中出现过
solution1_tf = tf_vectorizer.transform(solution1_word2str)

solution1_n_topics = 42
lda = joblib.load('models/fiveGroups/{}_topic/ldamodel.pkl'.format(solution1_n_topics))
lda.n_jobs = mp.cpu_count()
# 获得文档和主题的分布概率矩阵，一共174个文档，42个主题
solution1_doc_topic_dist = lda.transform(solution1_tf)


os.makedirs('matlab_result', exist_ok=True)
sio.savemat("matlab_result/solution1_X.mat",{"solution1_X":solution1_doc_topic_dist})
paintTopicDistri(solution1_doc_topic_dist,1,solution1_n_topics)

pseudocount = lda.components_#训练出来的伪计数
topic_word_dist = pseudocount / pseudocount.sum(axis=1)[:, np.newaxis]
paintWordDistri(pseudocount,solution1_n_topics)
