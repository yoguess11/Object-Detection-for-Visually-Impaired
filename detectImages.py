import cv2

#Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#classifier is a classifiers saying something is a cat, dog, face, etc...

#choose an image to detect faces in 
vid = cv2.VideoCapture(0)

while True:

    is_success, img = vid.read()
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
    for (x1, y1, w, h) in face_coordinates:
        cv2.rectangle(img, (x1,y1), (x1+w, y1+h), (255,0, 0))
    
    #Show image--both lines below need to be run
    cv2.imshow("Face Detector", img)
    #waitkey waits for a keyPress before closing the pop-up; otherwise, it closes instantly--you can press any key to resume
    key = cv2.waitKey(1)
    
    if key == 88 or key == 120:
        break 

print("complete!")
