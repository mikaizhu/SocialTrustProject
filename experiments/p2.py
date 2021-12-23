import sqlite3 as db
import pandas as pd
import json

def readTheWholeTable(tableName):
	cmd = "select * from {}".format(tableName)
	return pd.read_sql(cmd,conn)


if __name__ == "__main__":
	conn = db.connect("database/database.sqlite")
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
