# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:37:06 2019

@author: zxy
"""
"""
	以p1中清理过的member为基准，检查每条post，其对应的发帖人id以及其字面上的gid是否在json中能找到，不能找到的话直接将该post记录删除
"""
import sqlite3 as db
import pandas as pd
import json

def readTheWholeTable(tableName):
	cmd = "select * from {}".format(tableName)
	return pd.read_sql(cmd,conn)


if __name__ == "__main__":
	conn = db.connect("./facebook原始数据/database.sqlite")
	oldpostTable = readTheWholeTable("post")
	groupIDs = oldpostTable.gid.unique()
	frames = []
	for groupID in groupIDs:
		print(groupID,type(groupID))

		with open("json/"+groupID+".json") as f:
			id_name_dict = json.load(f)

		IDList = list(id_name_dict.keys())#从文件获取该小组成员id的list
		newSubPostTable = oldpostTable.loc[(oldpostTable.gid.isin([groupID])) & (oldpostTable.id.isin(IDList))]#获取正常异常记录
		frames.append(newSubPostTable)

	result = pd.concat(frames)
	result.to_csv("CSVdata/old/post.csv",index=False)


