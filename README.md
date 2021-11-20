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
> 果最好的模型个数.

lda参考资料：

- https://zhuanlan.zhihu.com/p/52142752： 这篇文章介绍了sklearn中的lda模型用法
  流程，说明lda要先转化为词频，使用的是CounterVectorizer()函数，统计每片文章中
  ，单词出现的频率，比如10个文章一共有100个单词组成，那么最后函数输出为一个
  10*100的矩阵，这个矩阵每一行表示单词出现的频率
- 经过lda获得的是什么？[参考](https://towardsdatascience.com/latent-dirichlet-allocation-for-topic-modelling-explained-algorithm-and-python-scikit-learn-c65a82e7304d) 主题为我们自己选定的,假设为20，模型会从所有的文本中选取20个主题，每行都是一个主题的概率分布，概率值最大，说明当前文本属于该主题的可能性比较高. 所以最后得到的是每个文本的主题概率分布矩阵, fit命令就是只从数据中学习有多少主题

# KDD文章说明

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


>  得到C矩阵后，可以根据C计算出每个用户的lambda值，因为只有95个agents，最后
>  lambda是95*1的列向量, 每个用户的计算公式为`sum(vec(C(:,:,nn))) / length(best_posts);` 这里vec的作用是，将一个矩阵，拉直为一个向量
> lambda 其实是每个用户是否有在帖子中互动过的参数，如果为0，表示该用户在所有帖
> 子中每任何互动，过滤掉这些用户，更新C和lambda向量以及agents矩阵

> 这里nb_groups不知道什么意思

