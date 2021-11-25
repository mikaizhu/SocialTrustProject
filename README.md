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

# 数据说明(Part 1)

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

下面着重介绍下p10文件中确定stubborn和normal的三种方法：

# KDD文章说明

依赖cvx安装：
- mac `sudo spctl --master-disable
` 解除开发者模式
- 下载cvx：http://cvxr.com/cvx/beta/
- cd cvx, cvx_setup

文件说明:
- synthetic_data_sim_part3.m
- synthetic_data_sim_part2.m
- synthetic_data_sim.m
- main_after_X.m
- main.m
- FB_Step2_det_stubborn_greedy.m
- FB_Step1_est_op_modelP.m
- FB_Step1_est_op_model3.m
- FB_Step1_est_op_model2.m
- FB_Step1_est_op_foramine.m
- FB_Step1_est_op.m
- det_stubborn_greedy.m
- do_sht_magic_csv.m


代码阅读顺序:

1. main.m

> 第一阶段
> 代码的第一部分是数据处理，数据为json文件(导入也为json格式的数据)，对数据处理完后，发现一共只有370个
> 帖子，为了保证最后模型效果比较好，这里要筛选出比较受欢迎的帖子，筛选方法为：
> 一个帖子的评论数和喜欢数之和，超过设定的阈值(15)，则保留下来, 最后只剩下55个帖子。(get_best_posts.m)
> get_agents函数从best_posts中所有帖子中的like， post comment三个指标中，统计哪些用户点
> 赞，发帖，或者评论，如果有以上行为，则记录在表格中，并统计这个用户的活跃度
> activity，每有一次以上行为就加1, 然后根据activity的数量，从高到低对用户进行
> 排序, 然后将用户的名字，用A1-A(lenght)进行替换

> 第二阶段
> 构建行为字典, 字典中的部分内容如下
> ```
> dico = struct('keyword','*LIKE*','value',1,'group',2);
> dico = [dico, struct('keyword','*TEXT*WITHOUT*KEYWORDS*','value',1,'group',1)]; % 为结构体添加一行内容
> dico = [dico, struct('keyword',';)','value',2,'group',1)];
> ```
> 然后根据自己构建的这个struct字典，输入到term_doc函数中term_doc(dico, posts,
> agents) 生成词文档矩阵, 经过数据处理后，一共有55个帖子，然后其中这55个帖子中
> 一共有95个用户互动，作者定义了19种状态，即生成19*55*95的矩阵，55表示55个帖子
> ，95表示所有帖子中有95个用户互动。term_doc先对每个post进行遍历，如果这个帖子
> ，有人like过，则找到这些like的用户，并将矩阵C(2,post_i, user_i)设置为1。然后
> 对这个post的message，即帖子的内容进行判断，如果有内容，则找到这个帖子是谁发
> 的，并将C(1, post_id, user_id)设置为1.最后遍历该post中message中的所有状态,
> 如commets中有没有笑脸，有没有感叹词等，如果有，统计出现的词语次数, 在对应的
> 用户位置加频率，最后得到C矩阵(C{1}, C{2}, C{19}分别对应发帖人，点赞帖子的人
> ，do nothing}, 可以在论文中找到)

> P矩阵是action-altitude分布矩阵，作者定义的有19个action，3个态度, 因此P矩阵为
> 19*3的概率分布矩阵

> dictionary字典：该字典中只有两个group，其中一个group表示喜欢，支持，另一个
> group为不支持, 每个group中有对应的举动，如表情等，还有对应的value,表示感情的
> 强烈

> get_agents: 统计这些帖子中，哪些用户的有点赞，评论，这些操作，记录在Y中，并
> 统计用户的活跃度，即每有一次操作，就加1

> nb_group: 是论文作者分类的结果，即作者认为有两种态度，一种是支持，另一种是不
> 支持，支持为group 1，在group 1中，普通用户会对帖子做一些列表示，如大笑，
> haha等关键字在post的comments中.

> parse函数：函数返回X_old，95x110维度的矩阵，95为95个用户，110为2个group，与
> 55个posts，该函数的功能为，从之前定义的dictionary中，得到一个评分矩阵，55*2
> 表示，每个帖子，都有两种状态，支持和不支持，比如用户1，对帖子1支持和不支持，
> 统计在第一列和第二列中，再统计该用户对第二个帖子的感情，统计在第三列第四列之
> 中。如此得到每个用户对所有帖子的感情分布

> normalize: 对X_old矩阵做normalize，每一列除以该列的最大值

> get_graph: 返回G & H， G为95x95的矩阵，H为95*55的矩阵, G矩阵的含义为用户之间
> 互动矩阵，即先对每个帖子遍历，利用get_agents函数，得到该帖子中所有活跃的用户
> ，第二层循环对每个用户遍历，比如用户1下有其他用户同时讨论了这个帖子，那么G矩
> 阵中第一行，对应的用户列就会设置为1. H 矩阵为该用户如果讨论过这个帖子则设置
> 为1, G表示哪些用户没有联系过，没有联系就为0

> B_MASK就是之前的互动矩阵G，为频率，作者设置了30个stubborn，如果互动次数超过0
> ，则设置为1， 否则为0，所以B，D mask都是一个0-1矩阵, 这里的mask为凸优化中的
> 条件

> isnan表示如果矩阵中，是否有nan值，如果有设置为1

>  得到C矩阵后，可以根据C计算出每个用户的lambda值，因为只有95个agents，最后
>  lambda是95*1的列向量, 每个用户的计算公式为`sum(vec(C(:,:,nn))) / length(best_posts);` 这里vec的作用是，将一个矩阵，拉直为一个向量
> lambda 其实是每个用户是否有在帖子中互动过的参数，如果为0，表示该用户在所有帖
> 子中每任何互动，过滤掉这些用户，更新C和lambda向量以及agents矩阵, n_groups=3
> 是作者认为人们对一个帖子的主要态度有，喜欢，不喜欢，中立，在这三种态度下，人
> 们会对帖子做一系列的举动


> 根据论文中公式11，12，可以得出，C矩阵和X矩阵P矩阵以及系数lambda, M表示用户对
> 一个事情的态度，可能有中立，同意，不同意三种，再根据论文公式16，17，18，可以
> 将利用C矩阵求P和X问题化为一个最小二乘问题，这个问题可以用BCD算法解决，算法流
> 程, BCD算法是将一个非凸优化问题，分成很多个子问题，每个子问题都是凸优化问题
> ，凸优化解决用cvx模块， https://web.stanford.edu/class/ee364a/lectures/cvx_tutorial.pdf

> 通过BCD算法求出意见矩阵X后，然后要从意见矩阵中，找到stubborn 和normal，然后
> 通过cvx求解公式（7)的凸优化方程，解出BD分块矩阵。

整体流程：

1. 首先是数据清理，挑选出活跃的post
2. 定义dictionary，从post中获取C矩阵
3. 从C矩阵中，利用BCD算法，得到用户意见矩阵X
4. 利用Greedy算法，从X矩阵中找到stubborn
5. 通过X，求解凸优化问题，解出B D 矩阵

