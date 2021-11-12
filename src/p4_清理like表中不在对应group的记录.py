# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:16:43 2019

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
	oldLikeTable = readTheWholeTable("like")
	groupIDs = oldLikeTable.gid.unique()
	frames = []
	for groupID in groupIDs:
		print(groupID,type(groupID))
		
		with open("json/"+groupID+".json") as f:
			id_name_dict = json.load(f)
		
		IDList = list(id_name_dict.keys())#从文件获取该小组成员id的list
		nameList = list(id_name_dict.values())
		#由于原始表中name有可能为空，因此多加一个限制条件
		newSubLikeTable = oldLikeTable.loc[(oldLikeTable.gid.isin([groupID])) & (oldLikeTable.id.isin(IDList)) & (oldLikeTable.name.isin(nameList))]#获取正常记录
		frames.append(newSubLikeTable)
	result = pd.concat(frames)
	result.to_csv("CSVdata/old/like.csv",index=False)
	result.to_csv("CSVdata/new/like.csv",index=False)