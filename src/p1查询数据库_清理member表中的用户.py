# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:32:52 2019

@author: zxy
"""


"""
	清理member表中一个id对应多个name的情况，直接将每个小组中的异常id记录删除，存入新的xls文件中的多个表中
"""
import sqlite3 as db
import pandas as pd
import json

def readTheWholeTable(tableName):
	cmd = "select * from {}".format(tableName)
	return pd.read_sql(cmd,conn)

if __name__ == "__main__":
	conn = db.connect("./facebook原始数据/database.sqlite")
	oldmemberTable = readTheWholeTable("member")
	groupIDs = oldmemberTable.gid.unique()

	frames = []#子表格

	for groupID in groupIDs:
		print(groupID,type(groupID))
		someRowsTable = oldmemberTable[["id","name"]].loc[oldmemberTable.gid==groupID]
		#查看哪些人共用了同一个id，建立字典  key=id,value=name,将相同id的name添加进字典
		rows = someRowsTable.values
		keys = set([item[0] for item in rows])
		#初始化大字典
		id_name_dict = {}
		for key in keys:
			id_name_dict[key] = []
		#添加信息
		for row in rows:
			if row[1] not in id_name_dict[row[0]]:
				id_name_dict[row[0]].append(row[1])

		wrongIDCollection = []#异常ID集合
		for id_,name in id_name_dict.items():
				if len(name) > 1:
					print(id_)
					wrongIDCollection.append(id_)
					id_name_dict.pop(id_)
		new_id_name_dict = {k:v[0] for k,v in id_name_dict.items()}#字典推导式
		newSubMemberTable = oldmemberTable.loc[(oldmemberTable.gid.isin([groupID])) & (~oldmemberTable.id.isin(wrongIDCollection))]#删除异常记录

		with open("json/{}.json".format(groupID),"w") as f:
			json.dump(new_id_name_dict,f)

		frames.append(newSubMemberTable)

	result = pd.concat(frames)
	#member表格已经被完全确认，记录进CSVdata的final文件夹中
	result.to_csv("CSVdata/final/member.csv",index=False)












