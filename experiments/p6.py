import pandas as pd
"""
***以下两步执行顺序不分先后***
1、以new文件夹中的post为基准，检查comment和like表中的pid,若不在post表pid所构成的范围内，则将相应的记录丢弃
2、清理由于前面p5步骤中，替换链接link之后msg为空的记录

***注意***：
	由于在p7中要计算互动次数，我们统一将msg为空格的那一次互动记作无效互动，因此在本文件中进行pandas表的nan值排除
"""

def getSubtableInPostPids(groundTrue,table):
	"""
	获取table中的pid符合post表中的pid的子表，并返回
	"""
	return table[table.pid.isin(TruePidCollection)]



if __name__ == "__main__":

	postTable = pd.read_csv("./CSVdata/new/post.csv",na_values=" ")#将整个表为空的转化成nan,且将内容是“ ”空格的地方转化成nan
	postTable["likes"].fillna(0,inplace=True)#将likes列中为nan的转化成0.
	postTable.dropna(inplace=True)#将msg是nan的行丢弃掉,inplace表示直接在原来的表中更改

	commentTable = pd.read_csv("./CSVdata/new/comment.csv",na_values=" ")#将整个表为空的转化成nan,且将内容是“ ”空格的地方转化成nan
	commentTable.dropna(inplace=True)#将msg是nan的行丢弃掉

	likeTable = pd.read_csv("./CSVdata/new/like.csv")#like表中没有缺失值


	TruePidCollection = postTable.pid.unique()
	newCommentTable = getSubtableInPostPids(TruePidCollection,commentTable)
	newLikeTable = getSubtableInPostPids(TruePidCollection,likeTable)

	postTable.to_csv("./CSVdata/final/post.csv",index=False)
	newCommentTable.to_csv("./CSVdata/final/comment.csv",index=False)
	newLikeTable.to_csv("./CSVdata/final/like.csv",index=False)
