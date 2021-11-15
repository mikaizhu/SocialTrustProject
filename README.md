# 流程说明

项目目录结构参考：https://github.com/SpikeKing/DL-Project-Template/blob/master/README.md

**目录说明**: 

```
├── experiments # 复现代码的结果
├── run.sh # 提交到github上
└── src # 原版代码的结果
```

要初步了解本项目，代码部分请按顺序阅读：

- p1-p11.py文件
- main_facebook.m文件

论文部分阅读顺序为：

1. 答辩ppt
2. 毕业论文

主要数据：

1. Facebook原始数据(里面有post like comment member.csv文件, 一共五个小组)
2. corpus(五个小组中用户的交互数据，比如互动时间等)

# 数据说明

首先是4个csv文件: comment post member like, 分别对应用户的评论，发帖， 个人，
点赞四个信息

每个csv中，主要可以利用的特征有：

- gid: group ID 因为一共有5个小组，每个小组都有唯一的group ID
- id: 每个用户的唯一标识符号


comment.csv 文件中：用户的评论别人的帖子信息

- 一个id应该对应一个名字，如果一个id对应多个名字，或者一个名字对应多个id，则将
  这些数据删除
- msg为用户的互动消息，为文本内容，后阶段应该对文本内容清洗
- gid: group id

post.csv: 用户的发帖子信息

- url表示评论的帖子地址
- msg表示该评论的信息
- likes表示该评论有多少人点赞

like.csv: 

- gid: group id 一共5个小组
- response: 类似于评论下面的表情标签，有like ，love sad angry等8中类型数据，这
  个数据暂时不知道怎么使用

member.csv: 小组成员信息

- ID：一个id对应一个名字
- name：小组成员的名字
- gid：group ID 一共5个小组
- url: 暂时不知道什么作用

# 数据处理流程

1. 成员数据清洗

处理错误信息, 如一个ID对应多个名字，一个名字对应多个ID, 或者发帖内容缺失

清洗方法: 删除对应的数据

对应文件：以member为基本，所以先对member数据处理

> p1: 处理member.csv文件, 如果一个ID对应多个名字，则从member.csv中删除这些数据

> p2: 以p1为基础，以member.csv为基准，检查post.csv文件，因为post文件有5个小组
> ，检查每个小组中成员能不能从member.csv中找到，不能则删除对应数据

> p3: 以p1为基础，检查comment, 和p2做同样的处理

> p4: 对like文件做同样的处理

2. 文本数据清洗

有文本信息的文件有：post.csv comment.csv

文本中待清洗的数据：post或者comment中，有可能存在超链接，即引用某个成员的评论
或者帖子，这些内容是以超链接的形式存在的，而不是文本内容存在的，所以要替换成文
本内容

清洗方法：根据post的信息，对post本身和comment数据进行替换，因为所有url可以从post中找到URL，处理流程为，使用正则表达式，从msg中提取出URL，然后再在post中查找该URL对应的内容，并对原URL进行替换

对应文件：以post为基本，所以先对post处理

> p5: 使用正则表达式先从comment和post中提取对应的URL，然后从post文件中找到对应
> URL的msg内容，如果找到则将URL替换成文本内容，否则替换成空格, 替换成空格可以
> 方便后续用re进行下一阶段数据处理. p5数据中还对http开头的URL进行提取，将post
> comment中http or https开头的URL替换成空格, 最后生成处理完msg的commet 和post
> 两个文件, msg中还有些特殊字符还没处理

> p6: 根据pid特征，如果like中pid不在post中，则删除like这一行数据

> p7: 前面代码只是替换了其中的URL，要将文本数据放入到模型中，好要对文本进一步
> 处理, 包括  
>
> ```
> 操作步骤：
>     - 数据清洗
>       1、替换特殊字符({COMMA}--逗号, {RET}--换行等)
> 	    2、切词
> 	    3、标注词性(方便后面还原)
> 	    4、单词还原(去掉单词的词缀，提取单词的主干部分)
> 	    5、去除标点符号和停用词(因为停用词都是小写的，所以这个步骤放后面)
> 	    6、除去包含有数字的词语
> 	    7、计算文档单词长度
> ```
> 
> 经过该步骤后，然后可以直接将文本放入到模型中提取特征, p7文件用到了新的数据,
> 文件为corpus/{groupID}.csv, 所以一共有5个文件(5个小组，每个小组一个文件),每
> 个文件包括该讨论组所有成员的互动信息(互动时间，内容)
> 1. 首先将5个小组的语料汇总，即将5个csv表格合并 
> 2. 进行数据清洗7个步骤, 最后数据为['i', 'love', 'you'] 类型的列表
> 3. 删除文本小于200(自己设定)的长度的数据
> 4. 将清洗完的数据保存 fiveGroups_corpus_dataframe

> p8: 使用LDA模型对文本进行训练, 训练多个模型，假设选取文本中存在20个主题，那
> 么就要训练得到20个主题的LDA模型，使用for循环训练20-100个主题模型，从中找出效
> 果最好的模型个数, 注意这里只调用了LDA的fit方法, 要转化成分布矩阵还要用
> transformer方法, 
>
> ```
> lda参考资料：
> 
> - https://zhuanlan.zhihu.com/p/52142752： 这篇文章介绍了sklearn中的lda模型用法
>   流程，说明lda要先转化为词频，使用的是CounterVectorizer()函数，统计每片文章中
>   ，单词出现的频率，比如10个文章一共有100个单词组成，那么最后函数输出为一个
>   10*100的矩阵，这个矩阵每一行表示单词出现的频率
> - 经过lda获得的是什么？[参考](https://towardsdatascience.com/latent-dirichlet-allocation-for-topic-modelling-explained-algorithm-and-python-scikit-learn-c65a82e7304d) 主题为我们自己选定的,假设为20，模型会从所有的文本中选取20个主题，每行都是一个主题的概率分布，概率值最大，说明当前文本属于该主题的可能性比较高. 所以最后得到的是每个文本的主题概率分布矩阵, fit命令就是只从数据中学习有多少主题
> ```

> p9: 因为数据中有5个小组，按之前的文件数据处理后，只有小组号为117291968282998的小组还剩下1k多人，我们最后从该小组中选出Stubborn，其他人为normal, 一共有三种方式
> 方式1: 自动选取阈值，即互动数，被点赞数，达到一定阈值，设置为stubborn，该小
> 组其他人设置为normal, 选出的stubborn有500多人，normal有700多人，但我们最后只
> 从中挑出58个stubborn，58*2个normal, 从corpus数据中找出这些用户，然后用之前统计好的词频(用到了所有用户)，对这些用户的文本进行词频统计。
> 将这些词频，放入到lda模型中，得到主题和词的分布矩阵，此时调用之前fit之后的
> 42topics的模型，用的transformer函数，转换成58*3，42二维的矩阵。

> ```
> fit 和 transformer函数的区别：
> 
> fit: Learn model for the data X with variational Bayes method.
> transformer: Transform data X according to the fitted model.
> 
> 总结：
> 
> fit就是先设定你要学习多少主题，如30，然后该函数会利用lda，从你提供的所有
> 数据中，提取出30个主题。
> 
> transformer就是利用之前训练好的模型，将当前样本，转换成主题-词概率分布矩阵，该
> 概率之和为1.
> ```

> p10: 和p9文件可以说是并行的，该文件确定了LDA主题模型的最佳主题为42个, tf_vectorizer为p8文件中，利用sklearn中的词频统计，学习到的模型.
> 训练流程如下：
> 1. 首先重构训练文本，前面通过sklearn中的CountVectorizer，得到文本中的单词字
>    典，即首先构造CountVectorizer实例，然后fit+transformer转换成词频矩阵，
>    vocabulary得到单词-序号字典。利用该字典，因为将每一行的词频矩阵，转换成新
>    的文本，因为词频矩阵长度为7k左右，和字典长度一行，序号为0的词频，和字典中
>    序号为0的单词，是一一对应的，如果词频为3，需要为0，单词对应为aa，那么新的
>    文本就有3个aa。第一个文本如：['aa', 'aa', 'aa' ....]
> 2. 使用gensim计算连贯性, 通常我们不知道一篇文章统计多少个主题效果比较好，使
>    用该方法可以计算，方法为先训练20-60个主题模型，然后对每个模型结果打分，分
>    数高的为最佳效果

下面着重介绍下p10文件中确定stubborn和normal的三种方法：

```
注意此时的文本数据都已经利用词频重构过

第一种方法：自己设定阈值，超过阈值的认为是stubborn，低于阈值的为normal

第二种方法：使用kmeans来确定stubborn和normal，判断kmeans聚类结果的好坏使用的是
轮廓系数，轮廓系数越大，则聚类效果越好，[轮廓系数：[参考文章](https://blog.csdn.net/weixin_44344462/article/details/89337770)]
kmeans的输入数据为数值型，所以先通过lda，选择为42个主题(由文件p10决定), 将3k多
个用户的文本训练为用户-主题分布矩阵，然后使用kmeans对所有用户-主题分布矩阵进行
聚类。k-means有三个属性centroid表示聚类的中心点，如3*42表示一共三个聚类,
label表示每个用户的分类，如果是2分类，则标签用0-1两类别表示, inertia表示所有用
户到聚类中心点的距离平方和.得到每个用户的分类标签后，然后可以将所有用户-主题矩
阵，和标签，输入到sklearn中的metric.silhouette_score中计算轮廓系数. 

第二种方法的搜索流程思路很简单，通过聚类，以及轮廓系数，可以将所有用户-主题分
布矩阵，挑出最佳的分类结果，然后找出每个kmeans类别的中心点，计算每个类别用户向
量到该类中心点的距离，并根据类别，将每个用户的距离从小到大排序好，距离中心越近
的作为stubborn，后面的作为normal, 然后将这些用户数据挑出来, 再利用
countvector.transformer 和lda trans，得到初始意见矩阵

第三种方法, 结合了方法1和方法2, 先人为设定阈值，选择stubborn和normal，然后从方
法2中，如果距离中心越近，又在自己选择的stubborn中，两个条件都满足则选为
stubborn，如果没有找到，则选择距离中心近的
```
