import pandas as pd
from pathlib import Path
import re
import numpy as np

# part1
# 清洗member中的重复的id，删除数据, member 数据中有些id对应多个名字，将这些数据删除

member = pd.read_csv('../facebook原始数据/member.csv')

df1 = member.groupby('id')['name'].nunique() # 统计每个id中，名字有多少唯一的
col = df1[df1 > 1].reset_index()['id']

member = member[~member['id'].isin(list(col))]

member.to_csv('data/member.csv', index=False)

# part 2
# 对post数据进行同样的操作

post = pd.read_csv('../facebook原始数据/post.csv')
post1 = post.groupby('id')['name'].nunique()
col = post1[post1 == 1].index
post = post[post.gid.isin(member.gid.unique()) & post.id.isin(member.id.unique()) & post.id.isin(col)]

post.to_csv('data/post.csv', index=False)

# part 3
# same as part2

comment = pd.read_csv('../facebook原始数据/comment.csv')
comment1 = comment.groupby('id')['name'].nunique()
col = comment1[comment1 == 1].index
comment = comment[comment.gid.isin(member.gid.unique()) & comment.id.isin(member.id.unique()) & comment.id.isin(col)]

comment.to_csv('data/comment.csv', index=False)

# part 4
# same as part2 & part3

like = pd.read_csv('../facebook原始数据/like.csv')
like1 = like.groupby('id')['name'].nunique()
col = like1[like1 == 1].index
like = like[like.gid.isin(member.gid.unique()) & like.id.isin(member.id.unique()) & like.id.isin(col)]
like.to_csv('data/like.csv', index=False)

# part 5
# 替换msg中的url信息

"""
comment 和 post 中有msg信息，msg中有些内容为url，url才是真实的comment内容，因此要将url进行替换

总的来说，数据中有三种链接:
    eg: FYI{COMMA} regarding the sewer and lateral issues{RET}https://www.facebook.com/groups/1239798932720607/permalink/1325132780853888/
https://www.facebook.com/groups/1451554835093070/permalink/1783012365280647/{RET}{RET}WHO{APOST}S BABY IS THIS???{RET}Found running on Church Rd off off Township Line!! {RET}He wants to come home and he{APOST}s so sweet!!{RET}Please share all over Elkins Park and surrounding areas! Ty!!
https://www.facebook.com/groups/117291968282998/permalink/1709495192395993/

从上面可以看到，评论中很多其他词汇，有http开始的，有www开始的链接, 都要删除, 如果是facebook中的链接，则是评论，先进行替换
"""
def replacePostLink(regular,table,pTable):
	"""
	替换帖子链接
	"""
	newtable = table.copy()
	for index,row in enumerate(newtable.itertuples()):
		result = regular.findall(str(row.msg))
		if result !=[]:
			for i in result:#结果有多个
				link = i[0]
				#到post中去查找该link所对应的内容,TrueMsg才是真正所对应的内容
				msgRows = pTable["msg"].loc[pTable.url==link].values
				if msgRows.size > 0:#其类型是np.ndarray;不为空说明找到，为空则将其替换成空格
					TrueMsg = msgRows[0]
					#在comment表中修改数值,使用at定位会比较快
					newtable.at[index,"msg"] = newtable.at[index,"msg"].replace(link,TrueMsg)
				else:
					newtable.at[index,"msg"] = newtable.at[index,"msg"].replace(link," ")
	return newtable

def delMediaLink(regular,table):
	"""
	删除媒体链接
	"""
	newtable = table.copy()
	for index,row in enumerate(newtable.itertuples()):
		#此处只在查找过程中替换，并未在真正的文件中替换
		result = regular.findall(str(row.msg).replace("{COMMA}"," ").replace("{RET}"," ").replace("{APOST}"," ")+" ") # 后面加空格是为了正则表达式提取
		if result !=[]:#与replacePostLink函数不同的原因：两个pattern的匹配组只有一个括号
			for link in result:#结果有多个
				newtable.at[index,"msg"] = newtable.at[index,"msg"].replace(link," ")
	return newtable


#commentTable = pd.read_csv("CSVdata/old/comment.csv")
#postTable = pd.read_csv("CSVdata/old/post.csv")
post = pd.read_csv('./data/post.csv')
comment = pd.read_csv('./data/comment.csv')

postLinkPattern = re.compile(r'((https?://)?www.facebook.com/groups/.*?/permalink/\d+/)',re.S)
#每次都是在帖子中查找link所对应的内容，因此每次均需要传入postTable
newCommentTable1 = replacePostLink(postLinkPattern,comment,post)
newpostTable1 = replacePostLink(postLinkPattern,post,post)

#提取http或https开头的链接，并删除
mediaLinkPattern1 = re.compile(r"(https?.*?)\s",re.S)
newCommentTable2 = delMediaLink(mediaLinkPattern1,newCommentTable1)
newpostTable2 = delMediaLink(mediaLinkPattern1,newpostTable1)

#在上面的基础之上，进一步提取www开头的链接并删除
mediaLinkPattern2 = re.compile(r"(www\..*?)\s",re.S)
Final_CommentTable = delMediaLink(mediaLinkPattern2,newCommentTable2)
Final_postTable = delMediaLink(mediaLinkPattern2,newpostTable2)

#写入文件，此时文件中仍含有{COMMA}\{RET}\{APOST}标识符
Final_CommentTable.to_csv("data/new_comment.csv",index=False)
Final_postTable.to_csv("data/new_post.csv",index=False)

# part 6
# 删除comment和like表中 pid不在post中的数据, 同时丢弃缺失值所在的行

def getSubtableInPostPids(groundTrue,table):
	"""
	获取table中的pid符合post表中的pid的子表，并返回
	"""
	return table[table.pid.isin(TruePidCollection)]

postTable = pd.read_csv("./data/new_post.csv",na_values=" ")#将整个表为空的转化成nan,且将内容是“ ”空格的地方转化成nan
postTable["likes"].fillna(0,inplace=True)#将likes列中为nan的转化成0.
postTable.dropna(inplace=True)#将msg是nan的行丢弃掉,inplace表示直接在原来的表中更改

commentTable = pd.read_csv("./data/new_comment.csv",na_values=" ")#将整个表为空的转化成nan,且将内容是“ ”空格的地方转化成nan
commentTable.dropna(inplace=True)#将msg是nan的行丢弃掉

likeTable = pd.read_csv("./data/like.csv")#like表中没有缺失值


TruePidCollection = postTable.pid.unique()
newCommentTable = getSubtableInPostPids(TruePidCollection,commentTable)
newLikeTable = getSubtableInPostPids(TruePidCollection,likeTable)

postTable.to_csv("data/final_post.csv",index=False)
newCommentTable.to_csv("data/final_comment.csv",index=False)
newLikeTable.to_csv("data/final_like.csv",index=False)

