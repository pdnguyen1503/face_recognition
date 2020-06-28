import cv2
import os
import numpy as np
from PIL import Image
import pymysql



#Some global variable 
recognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
fileData = 'dataSet'

#Funtion to get image user
def getImage(Id, Name):
    cam = cv2.VideoCapture(0)
    #set something for camera 
    cam.set(3,640)
    cam.set(4,640) 
    minW = 0.2*cam.get(3)
    minH = 0.2*cam.get(4)
    #input information of user
    idFace = int(Id)
    nameUser = Name
    #Remove all Image in dataSet
    imagePaths=[os.path.join(fileData,f) for f in os.listdir(fileData)]
    for imagePath in imagePaths:
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        if(idFace == ID):
            os.remove(imagePath)
    #Somthing for image
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    sampleNum=0
    while(True):
        _, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(
            gray,
            scaleFactor = 1.3,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
            )
        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,225,0), 2)
            sampleNum +=1
            cv2.imwrite(fileData+'/'+(nameUser)+'.'+str(idFace)+'.'+str(sampleNum)+'.jpg', gray[y: y+ h, x: x+ w])
        cv2.imshow('Image for user', frame)
        k = cv2.waitKey(10) & 0xff 
        if sampleNum == 50:
            break
        elif cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
#insert or update id and name of user
def insertOrUpdate():
    connection = pymysql.connect(host="localhost", user="nguyenpc", passwd="nguyen1503", database="FaceBase")
    cursor = connection.cursor()
    cmdAllPerson = "SELECT * FROM People"
    cursor.execute(cmdAllPerson)
    allPerSon = cursor.fetchall()
    print('All person in DataBase \n')
    print(allPerSon)
    Id=raw_input('\n enter userId end press <return> ==>  ')
    Name=raw_input('\n enter userName end press <return> ==>  ')
    cmd="SELECT * FROM People WHERE ID="+str(Id)
    cursor.execute(cmd)
    data = cursor.fetchall()
    isRecordExist=0
    for row in data:
        isRecordExist=1
    if isRecordExist == 1:
        cmd="UPDATE People SET Name="+"'"+str(Name)+"'"+" WHERE ID="+str(Id) + ";"
        print("Update user success")
        
    else:
        cmd="INSERT INTO People(ID,Name) Values("+str(Id)+","+"'"+str(Name)+"'"+");"
        print("Add user success")
    cursor.execute(cmd)
    connection.commit()
    connection.close()
    #do something with image faces
    getImage(Id, Name)

while(True):
    insertOrUpdate()
    answer = raw_input('Do you want to add user: [y/n]')
    if not answer or answer[0].lower() != 'y':
        print('\n you choose add User:')
        break
    print('---------Traing------------')
#-----Traing -----
def getImagesAndLabels(fileData):
    imagePaths=[os.path.join(fileData,f) for f in os.listdir(fileData)] 
    faceSamples=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L');
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces = faceDetector.detectMultiScale(faceNp)
        for (x, y, w, h) in faces:
            faceSamples.append(faceNp[y:y+h,x:x+w])
            IDs.append(ID)
        cv2.imshow("Training Face",faceNp)
        cv2.waitKey(10)
    return IDs, faceSamples
IDs, faces=getImagesAndLabels(fileData)
#Trainning then save in file traingingData
recognizer.train(faces, np.array(IDs))
recognizer.write('recognizer/trainningData.yml')
cv2.destroyAllWindows()
print('\n Training success')