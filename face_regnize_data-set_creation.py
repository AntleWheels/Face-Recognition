import cv2,os # we use the os library whenever we use the directory path
haar_file = 'haarcascade_frontalface_default.xml' # we use the haarcascade algorithm for detecting the face 
datasets = 'datasets'  # it initializes the main directry 
sub_data ='guru' # it initializes the subdirectry
path = os.path.join(datasets,sub_data)#datasets/rahul
if not os.path.isdir(path):#Check weather the path is presented in the directry
    os.mkdir(path) #If there is no path then it will make a new directry
(width, height) = (200, 200) #Initializing the Height and Width for the Picture 
face_cascade = cv2.CascadeClassifier(haar_file)#syntax for loading the algorithm
webcam = cv2.VideoCapture(0) # it initializes for the algorithm to access the webcam the default is 0 , If we use the default one then use the 

count = 1 # we are initializing the count value
while count <51: #It takes the 51 input images 
    print(count)
    (_,img) = webcam.read() # This line is used to read the webcam fro capturing the images 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converting the colorful images to the grayscale images 
    faces = face_cascade.detectMultiScale(gray, 1.3, 4) #used for detecting the multiple face from the image 
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y + h, x:x + w] #only cropping the face
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('%s/%s.png' % (path,count), face_resize)#count.png
    count += 1
    cv2.imshow('OpenCV', img)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()