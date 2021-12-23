import numpy as np
import pandas as pd
import re

def replacePostLink(regular,table,pTable):
	"""
	替换帖子链接
	"""
	newtable = table.copy()
	for index,row in enumerate(newtable.itertuples()):
		result = regular.findall(str(row.msg))
		if result != []:
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
		result = regular.findall(str(row.msg).replace("{COMMA}"," ").replace("{RET}"," ").replace("{APOST}"," ")+" ")
		if result != []:#与replacePostLink函数不同的原因：两个pattern的匹配组只有一个括号
			for link in result:#结果有多个
				newtable.at[index,"msg"] = newtable.at[index,"msg"].replace(link," ")
	return newtable

def testOK(regular,table):
	print(table.msg.str.extractall(regular))

if __name__ == "__main__":
	commentTable = pd.read_csv("CSVdata/old/comment.csv")
	postTable = pd.read_csv("CSVdata/old/post.csv")

	if 1:
		postLinkPattern = re.compile(r'((https?://)?www.facebook.com/groups/.*?/permalink/\d+/)',re.S)
		#每次都是在帖子中查找link所对应的内容，因此每次均需要传入postTable
		newCommentTable1 = replacePostLink(postLinkPattern,commentTable,postTable)
		newpostTable1 = replacePostLink(postLinkPattern,postTable,postTable)

	if 1:
		#提取http或https开头的链接，并删除
		mediaLinkPattern1 = re.compile(r"(https?.*?)\s",re.S)
		newCommentTable2 = delMediaLink(mediaLinkPattern1,newCommentTable1)
		newpostTable2 = delMediaLink(mediaLinkPattern1,newpostTable1)

	if 1:
		#在上面的基础之上，进一步提取www开头的链接并删除
		mediaLinkPattern2 = re.compile(r"(www\..*?)\s",re.S)
		Final_CommentTable = delMediaLink(mediaLinkPattern2,newCommentTable2)
		Final_postTable = delMediaLink(mediaLinkPattern2,newpostTable2)

	if 1:
		#检查是否匹配完毕
		testOK(mediaLinkPattern1,Final_CommentTable)
		testOK(mediaLinkPattern1,Final_postTable)
		testOK(mediaLinkPattern2,Final_CommentTable)
		testOK(mediaLinkPattern2,Final_postTable)

	if 1:
		#写入文件，此时文件中仍含有{COMMA}\{RET}\{APOST}标识符
		Final_CommentTable.to_csv("CSVdata/new/comment.csv",index=False)
		Final_postTable.to_csv("CSVdata/new/post.csv",index=False)

