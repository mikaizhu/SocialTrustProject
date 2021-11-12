# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:10:43 2019

@author: zxy
"""

import sqlite3 as db
import pandas as pd
import json

def readTheWholeTable(tableName):
	cmd = "select * from {}".format(tableName)
	return pd.read_sql(cmd,conn)


if __name__ == "__main__":
	conn = db.connect("database/database.sqlite")
	oldcommentTable = readTheWholeTable("comment")
	groupIDs = oldcommentTable.gid.unique()
	frames = []
	for groupID in groupIDs:
		print(groupID,type(groupID))
		
		with open("json/"+groupID+".json") as f:
			id_name_dict = json.load(f)
		
		IDList = list(id_name_dict.keys())#从文件获取该小组成员id的list
		newSubCommentTable = oldcommentTable.loc[(oldcommentTable.gid.isin([groupID])) & (oldcommentTable.id.isin(IDList))]#获取正常异常记录
		frames.append(newSubCommentTable)
	result = pd.concat(frames)
	result.to_csv("CSVdata/old/comment.csv",index=False)
	#test = pd.read_csv("CSVdata/comment.csv")