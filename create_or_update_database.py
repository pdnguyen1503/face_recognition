import pymysql

def insertOrUpdate(Id,Name):
    connection = pymysql.connect(host="localhost", user="nguyenpc", passwd="nguyen1503", database="FaceBase")
    cursor = connection.cursor()
    cmd="SELECT * FROM People WHERE ID="+str(Id)
    cursor.execute(cmd)
    data = cursor.fetchall()
    isRecordExist=0
    for row in data:
        isRecordExist=1
    if isRecordExist == 1:
        cmd="UPDATE People SET Name="+"'"+str(Name)+"'"+" WHERE ID="+str(Id) + ";"
        print("Update user success"")
    else:
        cmd="INSERT INTO People(ID,Name) Values("+str(Id)+","+"'"+str(Name)+"'"+");"
        print("Add user success"")
    cursor.execute(cmd)
    connection.commit()
    connection.close()
id=raw_input('\n enter userId end press <return> ==>  ')
name=raw_input('\n enter userName end press <return> ==>  ')
insertOrUpdate(id,name)