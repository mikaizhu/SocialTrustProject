import sqlite3 as db
import pandas as pd
import json
import os

def readTheWholeTable(tableName):
    cmd = "select * from {}".format(tableName)
    return pd.read_sql(cmd,conn)

conn = db.connect("database/database.sqlite")
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
    temp_id_name_dict = {}

    if not os.path.exists('./json'):
        os.mkdir('./json')
    if not os.path.exists('./CSVdata/final'):
        os.makedirs('./CSVdata/final')

    for id_,name in id_name_dict.items():
        if len(name) > 1:
            #print(id_)
            wrongIDCollection.append(id_)
        else:
            temp_id_name_dict[id_] = name
        new_id_name_dict = {k:v[0] for k,v in temp_id_name_dict.items()}#字典推导式
        newSubMemberTable = oldmemberTable.loc[(oldmemberTable.gid.isin([groupID])) & (~oldmemberTable.id.isin(wrongIDCollection))]#删除异常记录

        with open("json/{}.json".format(groupID),"w") as f:
            json.dump(new_id_name_dict,f)
            frames.append(newSubMemberTable)

result = pd.concat(frames)
#member表格已经被完全确认，记录进CSVdata的final文件夹中
result.to_csv("CSVdata/final/member.csv",index=False)
