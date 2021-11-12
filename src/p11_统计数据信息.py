# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 11:05:24 2020

@author: zxy
"""

import multiprocessing as mp
from sklearn.externals import joblib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pylab import mpl
#https://blog.csdn.net/qq_40170358/article/details/79980225
myfont = matplotlib.font_manager.FontProperties(fname='C:\\Windows\\Fonts\\simsun.ttc')
mpl.rcParams['axes.unicode_minus'] = False


"""
绘图形状：https://matplotlib.org/api/markers_api.html

"""





if __name__ == "__main__":
	
	
	with open("finalData",'rb') as f:
		final_data = pickle.load(f)
		
#############################################################################################
	#对处理完毕后的1327个用户的数据进行统计
	#每个类中用户的数目有多少
	if 1:
		def plotBar(X_sequence,Y_sequence):
			plt.style.use("classic")#设置图形风格
			fig = plt.figure(figsize=(10,5))
			ax = plt.axes()
			ax.bar(X_sequence, Y_sequence)
			fig.set_facecolor("white")
			ax.set_xlabel("类别标号",fontproperties=myfont)
			ax.set_ylabel("类中用户数",fontproperties=myfont)
			fig.savefig("models/fiveGroups/{}_topic/[中文]1327人的类中用户数.png".format(42),dpi=500)
			
	
		
		X_cluster_indexs = []
		Y_member_nums = []
		for classLabel in range(40):#这1327个用户聚成40个类
			X_cluster_indexs.append(classLabel)
			Y_member_nums.append(len(final_data[final_data.classLabel==classLabel]))
		plotBar(X_cluster_indexs,Y_member_nums)


#############################################################################################
	if 1:
		#绘制1327人的TSNE图
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
		
		def paint_tsne(distribution,labelList,name,size):
			plt.style.use("classic")#设置图形风格
			fig = plt.figure(figsize=(10,5))
			ax = plt.axes()
			#plt.style.use("classic")#设置图形风格
			from sklearn.manifold import TSNE
			x = TSNE(n_components=2).fit_transform(distribution)
			rng = np.random.RandomState(500)
			rng.shuffle(colors)
			for num, (x1,x2) in enumerate(x):
				c = colors[labelList[num]]
				ax.scatter(x1,x2,s=size,c=c)
			
			fig.set_facecolor("white")
			ax.set_xlabel("t-sne X",fontproperties=myfont)
			ax.set_ylabel("t-sne Y",fontproperties=myfont)
			fig.savefig("models/fiveGroups/{}_topic/{}.png".format(42,name),dpi=500)

		#提取1327人的意见分布
		with open("models/fiveGroups/term_frequence_transformer",'rb') as f:
			tf_vectorizer = pickle.load(f)
		doc_list = final_data.text.tolist()
		word2str = [" ".join(list_) for list_ in doc_list]
		tf = tf_vectorizer.transform(word2str)
		lda = joblib.load('models/fiveGroups/{}_topic/ldamodel.pkl'.format(42))
		lda.n_jobs = mp.cpu_count()-1
		doc_topic_dist = lda.transform(tf)
		labelList = final_data.classLabel.tolist()
		paint_tsne(doc_topic_dist,labelList,"[中文]1327人的tsne分布",size=10)


		if 1:#120人的
			#绘制被选的120名用户的t-sne图案
			#120人的意见分布
			with open("models/fiveGroups/{}_topic/solution3_finalCorpusDataframe".format(42),"rb") as f:
				CorpusDataframe_120 = pickle.load(f)
		
			doc_list_120 = CorpusDataframe_120.text.tolist()
			word2str_120 = [" ".join(list_) for list_ in doc_list_120]
			tf_120 = tf_vectorizer.transform(word2str_120)
			lda = joblib.load('models/fiveGroups/{}_topic/ldamodel.pkl'.format(42))
			lda.n_jobs = mp.cpu_count()-1
			doc_topic_dist_120 = lda.transform(tf_120)
			labelList_120 = CorpusDataframe_120.classLabel.tolist()
			paint_tsne(doc_topic_dist_120,labelList_120,"[中文]120人的tsne分布",size=100)
			print(len(CorpusDataframe_120))
		
		if 0:#240人的
			#绘制被选的120名用户的t-sne图案
			#120人的意见分布
			with open("models/fiveGroups/{}_topic/solution3_finalCorpusDataframe".format(42),"rb") as f:
				CorpusDataframe_120 = pickle.load(f)
		
			doc_list_120 = CorpusDataframe_120.text.tolist()
			word2str_120 = [" ".join(list_) for list_ in doc_list_120]
			tf_120 = tf_vectorizer.transform(word2str_120)
			lda = joblib.load('models/fiveGroups/{}_topic/ldamodel.pkl'.format(42))
			lda.n_jobs = mp.cpu_count()-1
			doc_topic_dist_120 = lda.transform(tf_120)
			labelList_120 = CorpusDataframe_120.classLabel.tolist()
			paint_tsne(doc_topic_dist_120,labelList_120,"[中文]120人的tsne分布",size=100)
			print(len(CorpusDataframe_120))
		

#############################################################################################
	if 1:
		def plotlines(X_sequence,Y_sequence,max_index):
			plt.style.use("classic")#设置图形风格
			fig = plt.figure(figsize=(10,5))
			ax = plt.axes()
			ax.plot(X_sequence, Y_sequence)
			ax.plot(X_sequence[max_index], Y_sequence[max_index], 'ro',markersize=5)
			fig.set_facecolor("white")
			ax.set_xlabel("聚类个数",fontproperties=myfont)
			ax.set_ylabel("轮廓系数",fontproperties=myfont)
			fig.savefig("models/fiveGroups/{}_topic/[中文]轮廓系数图.png".format(42),dpi=500)
		
		
		X_cluster_indexs = []
		Y_xishu = []
		with open("models/fiveGroups/{}_topic/聚类的轮廓系数.txt".format(42),"r") as f:
			best_cluster_num = int(f.readline().strip().split(":")[-1])
			for line in f.readlines():
				cluster_num = int(line.strip().split(",")[0])
				xishu = float(line.strip().split(",")[1])
				X_cluster_indexs.append(cluster_num)
				Y_xishu.append(xishu)
		plotlines(X_cluster_indexs,Y_xishu,best_cluster_num-2)
		
#############################################################################################
	#gensim_Coherence的主题相干性的图 ok
	if 1:
		def plotScatter(X_sequence,Y_sequence,max_index):
			"""
			绘制散点图
			"""
			plt.style.use("classic")#设置图形风格
			fig = plt.figure(figsize=(10,5))
			fig.set_facecolor("white")
			ax = plt.axes()
			ax.plot(X_sequence, Y_sequence, 'go',markersize=6)
			ax.plot(X_sequence[max_index], Y_sequence[max_index], 'ro',markersize=5)
			
			ax.set_xlabel("模型中的主题数目",fontproperties=myfont)
			ax.set_ylabel("平均主题相干性",fontproperties=myfont)
			fig.savefig("models/fiveGroups/{}_topic/[中文]gensim_coherence.png".format(42),dpi=500)
	
		X_index = []
		Y_gensim_coherences = []
		with open("gensim_coherence.txt","r") as f:
			lines = f.readlines()
			topic_start = 20
			for line in lines:
				coherence = float(line.strip().split(":")[1])
				X_index.append(topic_start)
				topic_start += 1
				Y_gensim_coherences.append(coherence)
		maxindex = Y_gensim_coherences.index(max(Y_gensim_coherences))
		plotScatter(X_index,Y_gensim_coherences,maxindex)
#############################################################################################
	
	
	#手动取出42个主题中某些主题的词语，用表格装在论文中 ok
	
	
#############################################################################################
	#先在matlab中运行，再到这里绘制影响关系矩阵
	def paintW(matrix,name):
		plt.style.use("classic")#设置图形风格
		fig = plt.figure()
		ax = plt.axes()
		im = ax.imshow(matrix,interpolation='none', vmin=0, vmax=1)
		plt.colorbar(im,orientation='vertical',shrink=1)
		ax.set_xlabel("成员编号",fontproperties=myfont)
		ax.set_ylabel("成员编号",fontproperties=myfont)
		
		fig.set_facecolor("white")
		ax.set_title("小组用户影响关系",fontproperties=myfont)
		#ax2.axis("off")
		ax.tick_params(direction="out",which='major',width=0.02,length=0.5,top='off',right='off',labelsize=6)
		# Hide the right and top spines
		ax.spines['right'].set_linewidth(0.01)
		ax.spines['top'].set_linewidth(0.01)
		ax.spines['left'].set_linewidth(0.01)
		ax.spines['bottom'].set_linewidth(0.01)
		fig.savefig("matlab_result/[中文]W{}.png".format(name),dpi=500)

	if 1:
		import scipy.io as sio
		solution1_W = sio.loadmat("matlab_result/W1.mat")["W_hat"]
		paintW(solution1_W,1)
		
		solution2_W = sio.loadmat("matlab_result/W2.mat")["W_hat"]
		paintW(solution2_W,2)
	
		solution3_W = sio.loadmat("matlab_result/W3.mat")["W_hat"]
		paintW(solution3_W,3)
#############################################################################################
	#绘制网络结构图
	#1、输出边文件
	if 0:
		rows = len(solution3_W)
		cols = len(solution3_W[0])
		with open("gephi_result/edges.csv",'w') as f:
			f.write("Source,Target,Weight\n")
			for i in range(rows):
				for j in range(cols):
					f.write("{},{},{}\n".format(i,j,solution3_W[i][j]))
	#2、节点文件在[中文]p9最后部分已经完成
#############################################################################################
	#绘制模型收敛图
	if 1:
		#1、随机种子固定，生成一个100为用户的随机向量
		user_num = 10
		rng1 = np.random.RandomState(500)
		#int_nums = rng1.choice(100,user_num,replace=False)#有放回的均匀抽样
		original_opinion_vector = [i for i in range(0,user_num*10,10)]
		#print(max(original_opinion_vector))
		#2、转移矩阵
		transition_matrix = []
		rng2 = np.random.RandomState(1000)
		for i in range(user_num):
			row = rng2.choice(120,user_num)
			scale_row = row/sum(row)
			transition_matrix.append(scale_row)
		transition_matrix = np.array(transition_matrix)

		all_period_opinion_vectors = []
		all_period_opinion_vectors.append(original_opinion_vector)
		steps = 5
		for step in range(steps):
			res = np.dot(transition_matrix,all_period_opinion_vectors[step])
			all_period_opinion_vectors.append(list(res))
		#print(all_period_opinion_vectors)
		all_period_opinion_vectors = np.transpose(all_period_opinion_vectors)
		
		def plotDeGroot(X_sequence,steps_opinions,name):
			plt.style.use("classic")#设置图形风格
			fig = plt.figure(figsize=(10,5))
			ax = plt.axes()
			for Y_sequence in steps_opinions:
				ax.plot(X_sequence, Y_sequence)
			fig.set_facecolor("white")
			ax.set_xlabel("讨论阶段",fontproperties=myfont,size=15)
			ax.set_ylabel(r"用户对参数$\theta$的意见值",fontproperties=myfont,size=15)
			fig.savefig("models/fiveGroups/{}_topic/{}.png".format(42,name),dpi=500)
		X_indexs = [i for i in range(steps+1)]
		plotDeGroot(X_indexs,all_period_opinion_vectors,name="[中文]DeGroot")
		
		
		
		#带有意见领袖的DeGroot
		stubb_num = 2
		rng3 = np.random.RandomState(427)
		stubb_index = rng3.choice(range(user_num),stubb_num,replace=False).tolist()
		transition_matrix_with_stubb = transition_matrix
		stubb_transition_matrix = np.zeros((stubb_num,user_num))
		for i in stubb_index:
			transition_matrix_with_stubb[i]=0
			transition_matrix_with_stubb[i][i]=1
		#不能直接对拼接的矩阵进行 行调换位置，那样会变成某个人对另一个人的置信度为1，而不是自己对自己的置信度为1
		#而应该在对角线上进行调换位置
		withStubb_all_period_opinion_vectors = []
		withStubb_all_period_opinion_vectors.append(original_opinion_vector)

		steps2 = 30
		for step in range(steps2):
			res = np.dot(transition_matrix_with_stubb,withStubb_all_period_opinion_vectors[step])
			withStubb_all_period_opinion_vectors.append(list(res))
		withStubb_all_period_opinion_vectors = np.transpose(withStubb_all_period_opinion_vectors)
		withStubb_X_indexs = [i for i in range(steps2+1)]
		
		plotDeGroot(withStubb_X_indexs,withStubb_all_period_opinion_vectors,name="[中文]带有意见领袖的DeGroot")
		
		
		
		#分析意见领袖数目对收敛快慢的影响
		matrixs = []
		stepsssssssssss = 30
		for stubb_num in range(1,7):#共6个图
			rng4 = np.random.RandomState(427)
			stubb_index = rng4.choice(range(user_num),stubb_num,replace=False).tolist()
			transition_matrix_with_stubb = transition_matrix
			stubb_transition_matrix = np.zeros((stubb_num,user_num))
			for i in stubb_index:
				transition_matrix_with_stubb[i]=0
				transition_matrix_with_stubb[i][i]=1

			withStubb_all_period_opinion_vectors = []
			withStubb_all_period_opinion_vectors.append(original_opinion_vector)
			for step in range(stepsssssssssss):
				res = np.dot(transition_matrix_with_stubb,withStubb_all_period_opinion_vectors[step])
				withStubb_all_period_opinion_vectors.append(list(res))
			withStubb_all_period_opinion_vectors = np.transpose(withStubb_all_period_opinion_vectors)
			matrixs.append(withStubb_all_period_opinion_vectors)
		
		
		XXXXXX = [i for i in range(stepsssssssssss+1)]
		plt.style.use("classic")#设置图形风格
		fig,axs = plt.subplots(2, 3, figsize=(8, 6))
		fig.set_facecolor("white")

		axs[0,0].set_ylabel(r"用户对参数$\theta$的意见值",fontproperties=myfont,size=15)
		axs[0,0].text(25,82,"意见领袖数目=10", ha='right', wrap=True,fontproperties=myfont)

		axs[0,1].text(25,82,"意见领袖数目=15", ha='right', wrap=True,fontproperties=myfont)

		axs[0,2].text(25,82,"意见领袖数目=20", ha='right', wrap=True,fontproperties=myfont)
		
		axs[1,0].set_xlabel("讨论阶段",fontproperties=myfont,size=15)
		axs[1,0].set_ylabel(r"用户对参数$\theta$的意见值",fontproperties=myfont,size=15)
		axs[1,0].text(25,82,"意见领袖数目=25", ha='right', wrap=True,fontproperties=myfont)
		
		axs[1,1].set_xlabel("讨论阶段",fontproperties=myfont,size=15)
		axs[1,1].text(25,82,"意见领袖数目=30", ha='right', wrap=True,fontproperties=myfont)
		
		axs[1,2].set_xlabel("讨论阶段",fontproperties=myfont,size=15)
		axs[1,2].text(25,82,"意见领袖数目=35", ha='right', wrap=True,fontproperties=myfont)

		for Y_sequence in matrixs[0]:
			axs[0, 0].plot(XXXXXX,Y_sequence)
		for Y_sequence in matrixs[1]:
			axs[0, 1].plot(XXXXXX,Y_sequence)
		for Y_sequence in matrixs[2]:
			axs[0, 2].plot(XXXXXX,Y_sequence)
		for Y_sequence in matrixs[3]:
			axs[1, 0].plot(XXXXXX,Y_sequence)
		for Y_sequence in matrixs[4]:
			axs[1, 1].plot(XXXXXX,Y_sequence)
		for Y_sequence in matrixs[5]:
			axs[1, 2].plot(XXXXXX,Y_sequence)
		fig.savefig("models/fiveGroups/{}_topic/{}.png".format(42,"[中文]意见领袖数目影响收敛速度"),dpi=500)
		
#############################################################################################
	if 1:
		def plotDirichlet(vectors,name,title):
			plt.style.use("classic")#设置图形风格
			fig = plt.figure()
			fig.set_facecolor("white")
			from mpl_toolkits.mplot3d import Axes3D
			ax = plt.subplot(projection='3d')#画3D图的时候必须要导入Axes3D
			ax.scatter(vectors[0], vectors[1], vectors[2], c=vectors.T, marker='o',s=50)
			ax.set_title(title)
			x1=np.array([1, 0, 0, 1])
			y1=np.array([0, 1, 0, 0])
			z1=np.array([0, 0, 1, 0])
			ax.plot(x1,y1,z1,'y--',lw=2)#将所有点从头连到尾

			ax.set_xlabel('X')
			ax.invert_xaxis()
			ax.set_ylabel('Y')
			ax.set_zlabel('Z')
			ax.invert_zaxis()
			ax.legend([r"$x+y+z=1$"])

			#ax.set_xlim3d(0,1)
			#ax.set_ylim3d(0,1)
			#ax.set_zlim3d(0,1)
			fig.savefig("models/fiveGroups/{}_topic/{}.png".format(42,name),dpi=500,bbox_inches='tight')

		rng = np.random.RandomState(seed=100)
		probList = rng.dirichlet([0.1,0.1,0.1],1000).T #值越大的维度，对应的轴 点越多
		plotDirichlet(probList,name="[中文]狄利克雷分布_alpha=0.1",title=r"$\alpha=0.1$")

		probList = rng.dirichlet([1,1,1],1000).T #值越大的维度，对应的轴 点越多
		plotDirichlet(probList,name="[中文]狄利克雷分布_alpha=1",title=r"$\alpha=1$")

		probList = rng.dirichlet([10,10,10],1000).T #值越大的维度，对应的轴 点越多
		plotDirichlet(probList,name="[中文]狄利克雷分布_alpha=10",title=r"$\alpha=10$")
#############################################################################################
	if 0:
		size = 1000
		Y_beta = np.random.beta(5,5,size)
		Y_beta = np.sort(Y_beta,axis=0)
		X_indexs = [i for i in range(size)]
		plt.plot(X_indexs,Y_beta)
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
#############################################################################################