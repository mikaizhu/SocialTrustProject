import multiprocessing as mp
#from sklearn.externals import joblib
import pickle
import numpy as np
from gensim import models , corpora
import joblib

if __name__ == "__main__":
    #从语料库中选出一些人来预测主题分布
    with open("corpus/fiveGroups_corpus_dataframe",'rb') as f:
    	data = pickle.load(f)
    with open("models/fiveGroups/term_frequence_transformer",'rb') as f:
    	tf_vectorizer = pickle.load(f) # p8文件中，利用所有预料训练的一个词频统计实例

    docLst = data.text.tolist()
    word2str = [" ".join(list_) for list_ in docLst]
    tf = tf_vectorizer.transform(word2str)

    if 1:
    	from sklearn.feature_extraction.text import CountVectorizer
    	tf_vectorizer = CountVectorizer(max_df=0.7, min_df=40)
    	tf = tf_vectorizer.fit_transform(word2str)

    #tf_vectorizer的匹配规则导致docList和tf中的单词不一致，有可能tf中的单词docList中没有
    #我们使用的时tf进行训练，因此在重构语料库的时候应该从tf中重构词袋模型的语料库newText

    sklearn_dictionary = tf_vectorizer.vocabulary_
    sklearn_id2token = dict(zip(sklearn_dictionary.values(), sklearn_dictionary.keys()))
    true_dictionary = corpora.Dictionary()
    true_dictionary.token2id =sklearn_dictionary
    true_dictionary.id2token = sklearn_id2token

    print("根据词频重构训练数据集文章...")
    newTexts = []
    termFreq = tf.toarray()
    for doc in termFreq:
    	adoc = []
    	for widx,freq in enumerate(doc):
    		adoc.extend([sklearn_id2token[widx]]*freq)
    	newTexts.append(adoc)

    if 1:
    #gensim 计算coherence
    	coherences = []
    	num_cores = mp.cpu_count()
    	CM = models.CoherenceModel(topics=[],texts=newTexts,dictionary=true_dictionary,coherence="c_v",topn=20,processes=num_cores-1)
    	for n_topics in range(20,101):
    		print("topic:{}".format(n_topics))
    		lda = joblib.load('models/fiveGroups/{}_topic/ldamodel.pkl'.format(n_topics))
    		lda.n_jobs = num_cores
    		#找到topic-word 分布中的靠前的10个词，先拿出模型中的主题词分布
    		pseudocount = lda.components_#训练出来的伪计数
    		topic_word_dist = pseudocount / pseudocount.sum(axis=1)[:, np.newaxis]

    		def get_pertopic_sorted(topic):
    			"""
    			返回某个主题中的所有词的序号，这些词按照在该主题中的出现频率从高到低进行排列
    			"""
    			idx=topic_word_dist[topic].argsort()[::-1]#一个主题中的词频率 从高到低 排列的所有此在词典中的序号
    			#print(sum(topic_word_dist[topic][idx]))
    			id2str = [sklearn_id2token[item] for item in idx]
    			return id2str
    		pertopic_words = [get_pertopic_sorted(topic_id) for topic_id in range(n_topics)]

    		print("开始计算主题数{}为时的主题相干性...".format(n_topics))
    		CM.topics = pertopic_words
    		coherence = CM.get_coherence()
    		print(coherence)
    		with open("gensim_coherence.txt","a+") as f:
    			f.write("{}:{}\n".format(n_topics,coherence))
    		coherences.append(coherence)

    #检测主题词分布的单词和语料库的单词是否一样
    if 1:
    	a = []
    	for i in pertopic_words:
    		a.extend(i)
    	a1 = list(set(a)) #7328

    	b = []
    	for i in newTexts:
    		b.extend(i)
    	b1 = set(b)# len(b1) 8104

    	result = []
    	for i in a1:
    		if i not in b1:
    			#print(i)
    			result.append(i)
    	print(len(result))
    	#topics:每个主题中的词按照概率从大到小排列
    	#topn:每个主题中会被用来当作代表的词的个数

    	#gensimde dictionary和sklearn的dictionary的区别
    	#sklearn的dictionary就是普通的字典


    if 1:
    	#这是一个使用gensim结合sklearn进行主题相干性计算的test
    	from gensim.test.utils import common_corpus, common_dictionary
    	from gensim import models , corpora
    	#在topics中出现了 字典 中没有的单词就会出现KeyError
    	#topics，common_texts，common_dictionary这三个参数的词语最好 是完全一致的，不然会出现计算中的除数为0错误
    	topics = [
    				['human'],
    				["computer"],
    				["system","trees"]
    				]
    	common_texts =[['human', 'interface', 'computer'],
    						['survey', 'user', 'computer', 'system', 'response', 'time'],
    						['eps', 'user', 'interface', 'system'],
    						['system', 'human', 'system', 'eps'],
    						['user', 'response', 'time'],
    						['trees',"hello"],
    						['graph', 'trees'],
    						['graph', 'minors', 'trees'],
    						['graph', 'minors', 'survey']]+[['minors', 'trees']]*20
    	common_dictionary.add_documents(common_texts)
    	cm = models.CoherenceModel(topics=topics,topn=6, texts=common_texts, dictionary=common_dictionary, coherence='c_v')
    	coherence = cm.get_coherence()  # get coherence value
    	print(coherence)


    #网页请求获取主题相干性
    if 1:
    	import os
    	os.makedirs("topicWords/fiveGroups",exist_ok=True)
    	import requests as rq
    	#palmetto_uri="http://palmetto.aksw.org/palmetto-webapp/service/cv?words="
    	palmetto_uri = "http://localhost:18080/palmetto-webapp/service/cv?words="
    	num_cores = mp.cpu_count()
    	for n_topics in range(20,101):
    		print("####################topic:{}####################".format(n_topics))
    		lda = joblib.load('models/fiveGroups/{}_topic/ldamodel.pkl'.format(n_topics))
    		lda.n_jobs = num_cores
    		#找到topic-word 分布中的靠前的10个词，先拿出模型中的主题词分布
    		pseudocount = lda.components_#训练出来的伪计数
    		topic_word_dist = pseudocount / pseudocount.sum(axis=1)[:, np.newaxis]

    		def get_pertopic_sorted(topic):
    			"""
    			返回某个主题中的所有词的序号，这些词按照在该主题中的出现频率从高到低进行排列
    			"""
    			idx=topic_word_dist[topic].argsort()[::-1]#一个主题中的词频率 从高到低 排列的所有此在词典中的序号
    			#print(sum(topic_word_dist[topic][idx]))
    			id2str = [sklearn_id2token[item] for item in idx]
    			return id2str
    		pertopic_words = [get_pertopic_sorted(topic_id)[:10] for topic_id in range(n_topics)]
    		with open("topicWords/fiveGroups/{}_topics.txt".format(n_topics),"w") as f:
    			for item in pertopic_words:
    				f.write(" ".join(item)+"\n")
    		if 1:
    			send_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36",
    								"Connection": "keep-alive",
    								"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    								"Accept-Language": "zh-CN,zh;q=0.8"}
    			with open("webCoherence.txt",'a+') as f:
    				f.write("{}:\n".format(n_topics))
    			webresult = []
    			for index,topic in enumerate(pertopic_words):
    				print(index)
    				flag = False
    				while not flag:
    					r = rq.post(url=palmetto_uri+" ".join(topic),headers=send_headers,timeout=None)
    					flag = r.ok
    					if flag:
    						print(float(r.text))
    						webresult.append(float(r.text))
    						with open("webCoherence.txt",'a+') as f:
    							f.write("{}\n".format(float(r.text)))
    					else:
    						print("error")
    						continue
    			with open("webCoherence.txt",'a+') as f:
    				f.write("mean:{}\n".format(np.mean(webresult)))
