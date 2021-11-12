# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 00:23:58 2019

@author: zxy
"""


import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import scipy.io as sio
import multiprocessing as mp



colors = ['aliceblue','antiquewhite','aqua','aquamarine','azure','beige','bisque','black','blanchedalmond','blue','blueviolet',
	'brown','burlywood','cadetblue','chartreuse','chocolate','coral','cornflowerblue','cornsilk',
	'crimson','cyan','darkblue','darkcyan','darkgoldenrod','darkgray','darkgreen','darkgrey','darkkhaki',
	'darkmagenta','darkolivegreen','darkorange','darkorchid','darkred','darksalmon','darkseagreen',
	'darkslateblue','darkslategray','darkslategrey','darkturquoise','darkviolet','deeppink',
	'deepskyblue','dimgray','dimgrey','dodgerblue','firebrick','floralwhite','forestgreen',
	'fuchsia','gainsboro','ghostwhite','gold','goldenrod','gray','green','greenyellow','grey',
	'honeydew','hotpink','indianred','indigo','ivory','khaki','lavender','lavenderblush',
	'lawngreen','lemonchiffon','lightblue','lightcoral','lightcyan','lightgoldenrodyellow','lightgray',
	'lightgreen','lightgrey','lightpink','lightsalmon','lightseagreen','lightskyblue','lightslategray',
	'lightslategrey','lightsteelblue','lightyellow','lime','limegreen','linen','magenta','maroon',
	'mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen','mediumslateblue',
	'mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','mintcream','mistyrose',
	'moccasin','navajowhite','navy','oldlace','olive','olivedrab','orange','orangered','orchid',
	'palegoldenrod','palegreen','paleturquoise','palevioletred','papayawhip','peachpuff',
	'peru','pink','plum','powderblue','purple','rebeccapurple','red','rosybrown','royalblue',
	'saddlebrown','salmon','sandybrown','seagreen','seashell','sienna','silver','skyblue',
	'slateblue','slategray','slategrey','snow','springgreen','steelblue','tan','teal','thistle',
	'tomato','turquoise','violet','wheat','white','whitesmoke','yellow','yellowgreen']
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


if __name__ == "__main__":

	#从语料库中选出一些人来预测主题分布
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



###############################################################################
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
	
	solution1_stubbornIDs = showTable[(showTable.postTimes>=a) & (showTable.interactTimes>=b) & (showTable.beInteractedTimes>=c)].userID.tolist()
	solution1_normalIDs = list(set(showTable.userID) - set(solution1_stubbornIDs))

	solution1_stubbornTable = showTable.loc[solution1_stubbornIDs]
	solution1_normalTable = showTable.loc[solution1_normalIDs]
	#Passing list-likes to .loc or [] with any missing label will raise KeyError in the future, you can use .reindex() as an alternative.
	solution1_stubbrn_data = data.loc[solution1_stubbornTable.userID.values].dropna()
	solution1_normals_data = data.loc[solution1_normalTable.userID.values].dropna()
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


	#由p_10中的主题相干性决定了这里的topic数量为42
	from sklearn.externals import joblib
	solution1_n_topics = 42
	lda = joblib.load('models/fiveGroups/{}_topic/ldamodel.pkl'.format(solution1_n_topics))
	lda.n_jobs = mp.cpu_count()
	solution1_doc_topic_dist = lda.transform(solution1_tf)

	sio.savemat("matlab_result/solution1_X.mat",{"solution1_X":solution1_doc_topic_dist})
	paintTopicDistri(solution1_doc_topic_dist,1,solution1_n_topics)
	
	pseudocount = lda.components_#训练出来的伪计数
	topic_word_dist = pseudocount / pseudocount.sum(axis=1)[:, np.newaxis]
	paintWordDistri(pseudocount,solution1_n_topics)
	
	
	
	
	
###############################################################################


	#第二套方案，选出类中心的当做stubborn，不活跃的当做normal
	#首先得到该小组所有成员的主题分布solution2_topic_distri_all
	solution2_docLst = data.text.tolist()
	solution2_word2str = [" ".join(list_) for list_ in solution2_docLst]
	solution2_tf = tf_vectorizer.transform(solution2_word2str)
	solution2_topic_distri_all = lda.transform(solution2_tf)

	#接着对该小组成员进行聚类
	from sklearn.cluster import KMeans
	import multiprocessing as mp
	from sklearn.metrics.pairwise import paired_distances
	from sklearn import metrics
	num_cores = mp.cpu_count()-1

	
	#轮廓系数决定聚类个数
	if 0:
		score_results = []
		steps = 10
		for step in range(steps):
			score_result = []#轮廓系数越大越好
			print("第{}次计算...".format(step))
			for n_clusters in range(2,100):
				kmeans = KMeans(n_clusters=n_clusters,max_iter=1000,algorithm="full",
										n_jobs=num_cores,random_state=step*n_clusters).fit(solution2_topic_distri_all)
				centroid, labelList, inertia = kmeans.cluster_centers_ , kmeans.labels_ , kmeans.inertia_
				silhouette_score = metrics.silhouette_score(solution2_topic_distri_all, labelList, metric='euclidean')

				score_result.append(silhouette_score)
			score_results.append(score_result)
		score_results = np.array(score_results)
		
		sum_score_results = score_results.sum(axis=0)/steps
		max_index = int(np.argwhere(sum_score_results==max(sum_score_results)))
		print("最佳聚类个数为：",max_index+2)
		plt.plot([i+2 for i in range(len(sum_score_results))], sum_score_results)
		solution2_n_clusters = max_index+2#加2是因为从2开始计算的聚类个数

	solution2_n_clusters = 40
	#重开一个showtable
	solution2_showTable = showTable
	

	solution2_kmeans = KMeans(n_clusters=solution2_n_clusters,max_iter=1000,algorithm="full",
							n_jobs=num_cores,random_state=23456).fit(solution2_topic_distri_all)
	solution2_centroid, solution2_labelList = solution2_kmeans.cluster_centers_ , solution2_kmeans.labels_
	solution2_centers = np.array([solution2_centroid[index] for index in solution2_labelList])
	#欧氏距离越小越好
	solution2_distances = paired_distances(solution2_topic_distri_all,solution2_centers,metric="euclidean")
	
	solution2_showTable["distance"] = solution2_distances
	solution2_showTable["classLabel"] = solution2_labelList
	
	solution2_df_groupList = [solution2_showTable.groupby("classLabel").get_group(i).sort_values(["distance"],ascending=True) for i in range(solution2_n_clusters)]
	solution2_sorted_df = pd.concat(solution2_df_groupList)#还未分离stub和normal
	
	
	################################
	#确定stubb和normal数目在这里确定#
	################################
	#一个类别10个人，3个stubb，7个norm
	amount_perclass = 3
	base = 1#一个类中的stubborn数
	stubb_counts = solution2_n_clusters*base
	norm_counts = solution2_n_clusters*(amount_perclass-base)
	s2_a = [solution2_sorted_df[solution2_sorted_df.classLabel == i] for i in range(solution2_n_clusters)]
	s2_b = [item.iloc[:base] for item in s2_a]#stubb个数为3*solution2_n_clusters
	solution2_stubb = pd.concat(s2_b)
	s2_c = [item.iloc[base:amount_perclass] for item in s2_a]#norm个数为7*solution2_n_clusters
	solution2_norm = pd.concat(s2_c)
	solution2_IDs = pd.concat([solution2_stubb,solution2_norm]).index
	solution2_finalCorpusDataframde = data.loc[solution2_IDs]

	solution2_docLst2 = solution2_finalCorpusDataframde.text.tolist()
	solution2_word2str2 = [" ".join(list_) for list_ in solution2_docLst2]

	solution2_tf = tf_vectorizer.transform(solution2_word2str2)

	solution2_n_topics = solution1_n_topics
	lda = joblib.load('models/fiveGroups/{}_topic/ldamodel.pkl'.format(solution2_n_topics))
	lda.n_jobs = mp.cpu_count()
	solution2_doc_topic_dist = lda.transform(solution2_tf)

	sio.savemat("matlab_result/solution2_X.mat",{"solution2_X":solution2_doc_topic_dist})
	paintTopicDistri(solution2_doc_topic_dist,2,solution2_n_topics)







	###############################################################################
	#第三套方案：将类中心的讲话比较多的人当做stubborn，要用到的表是经过了方案二的showtable，里面包含了classLabel
	
	#重开一个showtable，由于聚类数目和solution2相同，可以直接用
	solution3_n_clusters = solution2_n_clusters
	solution3_showTable = solution2_showTable

	#post阈值d,其实这个小组每个人发贴次数并不多
	d = 1
	#interact阈值e
	e = 10
	#beInteracted阈值f
	f = 10
	#经常发帖，经常互动，经常被互动 则为stubborn
	#其余为normal
	solution3_stubbornIDs = showTable[(showTable.postTimes>=d) & (showTable.interactTimes>=e) & (showTable.beInteractedTimes>=f)].userID.tolist()
	solution3_normalIDs = list(set(showTable.userID) - set(solution3_stubbornIDs))
	solution3_showTable["talkative"] = 0
	print("stubb有：{}. normal有：{}.".format(len(solution3_stubbornIDs),len(solution3_normalIDs)))

	func = lambda x: 1 if x.userID in solution3_stubbornIDs else 0
	solution3_showTable["talkative"] = solution3_showTable.apply(func,axis=1)

	#第一步，按类别分类
	solution3_df_groupList = [solution3_showTable.groupby("classLabel").get_group(i).sort_values(["distance"],ascending=True) for i in range(solution3_n_clusters)]
	#第二步，在每个类别中选择talkative的人当做stubborn
	solution3_sorted_df = pd.concat(solution3_df_groupList)
	#此时的solution3_sorted_df是classLable和distance排好序的
	#只需要在每个类别中从头到尾进行遍历，依次取出足够的stubb和normal即可
	solution3_final_stubbIDs = []
	solution3_final_normIDs = []
	for classLabel in range(solution3_n_clusters):
		#在一个类别中找stubb和norm
		temp_s = []
		temp_n = []
		for userID,record in solution3_sorted_df[solution3_sorted_df.classLabel==classLabel].iterrows():
			#print(userID)
			#print(int(record.classLabel))
			if int(record.talkative) and len(temp_s)<base:#若是话痨,且stubb未满
				temp_s.append(userID)
			if not int(record.talkative) and len(temp_n)<(amount_perclass-base):
				temp_n.append(userID)

		if len(temp_s) < base:#如果未在类中找到话痨选手，则补充离类中心较近的普通成员为意见领袖
			print("编号为{}的类中不够意见领袖，已补充{}个".format(classLabel,base-len(temp_s)))
			for userID,record in solution3_sorted_df[solution3_sorted_df.classLabel==classLabel].iterrows():
				if len(temp_s)<base:#若是话痨,且stubb未满
					temp_s.append(userID)

		if len(temp_n) < amount_perclass-base:#如果未在类中找到沉默选手，则补充离类中心最远的意见领袖为普通用户
			print("编号为{}的类中不够普通用户，已补充{}个".format(classLabel,amount_perclass-base-len(temp_n)))
			for userID,record in list(solution3_sorted_df[solution3_sorted_df.classLabel==classLabel].iterrows())[::-1]:
				if len(temp_n)<amount_perclass-base:
					temp_n.append(userID)
		solution3_final_stubbIDs.extend(temp_s)
		solution3_final_normIDs.extend(temp_n)
	solution3_IDs = solution3_final_stubbIDs+solution3_final_normIDs
	solution3_finalCorpusDataframde = data.loc[solution3_IDs]

	solution3_docLst = solution3_finalCorpusDataframde.text.tolist()
	solution3_word2str = [" ".join(list_) for list_ in solution3_docLst]

	solution3_tf = tf_vectorizer.transform(solution3_word2str)

	solution3_n_topics = solution2_n_topics
	lda = joblib.load('models/fiveGroups/{}_topic/ldamodel.pkl'.format(solution3_n_topics))
	lda.n_jobs = mp.cpu_count()
	solution3_doc_topic_dist = lda.transform(solution3_tf)

	sio.savemat("matlab_result/solution3_X.mat",{"solution3_X":solution3_doc_topic_dist})
	paintTopicDistri(solution3_doc_topic_dist,3,solution3_n_topics)
	
	
	
	
	
	
	
