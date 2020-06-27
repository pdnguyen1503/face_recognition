import cv2
import pymysql
import numpy as np

recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainningData.yml")
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
fontface = cv2.FONT_HERSHEY_SIMPLEX

#get all user from phpmyadmin by ID
def getProfile():
    connection = pymysql.connect(host="localhost", user="nguyenpc", passwd="nguyen1503", database="FaceBase")
    cursor = connection.cursor()
    cmd="SELECT * FROM `People`"
    cursor.execute(cmd)
    people = cursor.fetchall()
    peopleArray  = np.array(people)
    i = 0
    personArray = ['unkown']
    while i<len(peopleArray):
    	personArray.append(peopleArray[i][1])
    	i+=1
    return personArray
personAll = getProfile()
print("All user in database: \n"+ personAll)
cam = cv2.VideoCapture(0)
cam.set(3, 720)
cam.set(4, 720)
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, image =cam.read()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale( 
        gray,
        scaleFactor = 1.3,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
        )

    for(x, y, w, h) in faces:
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,255), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence < 70):
            nameUser = personAll[id]
            
        else :
            nameUser = "Unknow"
        cv2.putText(image, str(nameUser), (x,y), fontface, 1, (255,255,0), 2)
    cv2.imshow('Face Recognition',image) 
    if cv2.waitKey(1)==ord('q'):
        break;

print("\n Exit face recognition")
cam.release()
cv2.destroyAllWindows()