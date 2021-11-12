# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:05:36 2019

@author: zxy
"""

import re
import numpy as np
import multiprocessing as mp
import pickle
import matplotlib.pyplot as plt
import os


def paint_likelihood(n_topic):
	p = re.compile("(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")
	with open("models/fiveGroups/{}_topic/gensim.log".format(n_topic),"r") as f:
		matches = [p.findall(l) for l in f]
	matches = [m for m in matches if len(m) > 0]
	tuples = [t[0] for t in matches]
	#perplexity = [float(t[1]) for t in tuples]#越小越好
	liklihood = np.array([float(t[0]) for t in tuples])*100
	iter_ = list(range(0,len(tuples)))
	fig = plt.figure()
	ax = plt.axes()
	ax.plot(iter_,liklihood,c="black")
	ax.set_ylabel("log liklihood")#越大越好
	ax.set_xlabel("passes")
	ax.set_title("Topic Model Convergence")
	fig.savefig("models/fiveGroups/{}_topic/likelihood.pdf".format(n_topic))

def paintTopicDistri(matrix):
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
	
	fig2.savefig("models/fiveGroups/{}_topic/X.pdf".format(n_topics))

if __name__ == "__main__":


	with open("corpus/fiveGroups_corpus_dataframe",'rb') as f:
		data = pickle.load(f)

	docLst = data.text.tolist()
	word2str = [" ".join(list_) for list_ in docLst]
	from sklearn.feature_extraction.text import CountVectorizer
	#处理单词, 超过max_df的文章中出现过
	tf_vectorizer = CountVectorizer(max_df=0.7, min_df=40)
	tf = tf_vectorizer.fit_transform(word2str)
	os.makedirs("models/fiveGroups",exist_ok=True)
	with open("models/fiveGroups/term_frequence_transformer",'wb') as f:
		pickle.dump(tf_vectorizer,f)
	#wordfqMatrix = tf.toarray()
	if 0:#训练
		from sklearn.decomposition import LatentDirichletAllocation
		from sklearn.externals import joblib

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


