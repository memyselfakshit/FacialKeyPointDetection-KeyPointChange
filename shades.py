from my_CNN_model import *
import cv2
import numpy as np
import keras

import speech_recognition as sr
import threading
import os
import nltk



def task2():
    r = sr.Recognizer()
    print("thread2")
    while 1:
        with sr.Microphone() as source:
            print("say something")
            audio=r.listen(source)
            tokens = nltk.word_tokenize(r.recognize_google(audio))
            print(tokens)
            tagged = nltk.pos_tag(tokens)
            print(tagged)
            
            length = len(tagged) - 1
            a = list()
            for tuple1 in tagged:
                print(tuple1[1])
                log = (tuple1[1][0] == 'N')
                if log == True:
                    a.append(tuple1[0])
            print(a)
            print("Time over")
        try:
            print("text : " + r.recognize_google(audio))
        except:
            pass



if __name__== "__main__" :
    # Load the model built in the previous step

    #face_cascade = cv2.CascadeClassifier('C:\\Users\\akshi\\abc.xml')
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    #eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    my_model = load_my_CNN_model('my_model')

    # Face cascade to detect faces
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

    # Define the upper and lower boundaries for a color to be considered "Blue"
    blueLower = np.array([100, 60, 60])
    blueUpper = np.array([140, 255, 255])

    # Define a 5x5 kernel for erosion and dilation
    kernel = np.ones((5, 5), np.uint8)

    print('checking 1')

    # Define filters
    filters = ['images/sunglasses.png', 'images/sunglasses_2.png', 'images/sunglasses_3.jpg', 'images/sunglasses_4.png', 'images/sunglasses_5.jpg', 'images/sunglasses_6.png']
    filterIndex = 0

    filters_moustache = ['Moustache/1.png', 'Moustache/2.png', 'Moustache/3.png']
    filterIndex_moustache  = 0
    camera = cv2.VideoCapture(0)
    t2=threading.Thread(target=task2, name='t2')
   # t1.start()
    t2.start()
    print("mehul be bi=ola hai")
        # Keep looping
    while True:
        # Grab the current paintWindow
        (grabbed, frame) = camera.read()
        frame = cv2.flip(frame, 1)
        frame2 = np.copy(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Add the 'Next Filter' button to the frame
        frame = cv2.rectangle(frame, (500,10), (620,65), (235,50,50), -1)
        cv2.putText(frame, "NEXT FILTER", (512, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
  #      print('checking 2')
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.25, 6)

        # Determine which pixels fall within the blue boundaries and then blur the binary image
        blueMask = cv2.inRange(hsv, blueLower, blueUpper)
        blueMask = cv2.erode(blueMask, kernel, iterations=2)
        blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
        blueMask = cv2.dilate(blueMask, kernel, iterations=1)

        # Find contours (bottle cap in my case) in the image
        (_, cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)
        center = None

 #       print('checking 3')
        # Check to see if any contours were found
        if len(cnts) > 0:
        	# Sort the contours and find the largest one -- we
        	# will assume this contour correspondes to the area of the bottle cap
            cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
            # Get the radius of the enclosing circle around the found contour
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            # Draw the circle around the contour
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            # Get the moments to calculate the center of the contour (in this case Circle)
            M = cv2.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

            if center[1] <= 65:
                if 500 <= center[0] <= 620: # Next Filter
                    filterIndex += 1
                    filterIndex %= 6
                    filterIndex_moustache += 1
                    filterIndex_moustache %= 6
                    continue
#        print('checking 4')
        for (x, y, w, h) in faces:

            # Grab the face
            gray_face = gray[y:y+h, x:x+w]
            color_face = frame[y:y+h, x:x+w]

            # Normalize to match the input format of the model - Range of pixel to [0, 1]
            gray_normalized = gray_face / 255

            # Resize it to 96x96 to match the input format of the model
            original_shape = gray_face.shape # A Copy for future reference
            face_resized = cv2.resize(gray_normalized, (96, 96), interpolation = cv2.INTER_AREA)
            face_resized_copy = face_resized.copy()
            face_resized = face_resized.reshape(1, 96, 96, 1)

            # Predicting the keypoints using the model
            keypoints = my_model.predict(face_resized)

            # De-Normalize the keypoints values
            keypoints = keypoints * 48 + 48

            # Map the Keypoints back to the original image
            face_resized_color = cv2.resize(color_face, (96, 96), interpolation = cv2.INTER_AREA)
            face_resized_color2 = np.copy(face_resized_color)

            # Pair them together
            points = []
            for i, co in enumerate(keypoints[0][0::2]):
                points.append((co, keypoints[0][1::2][i]))



            # Add Sunglasses FILTER to the frame
            
            sunglasses = cv2.imread(filters[filterIndex], cv2.IMREAD_UNCHANGED)
            sunglass_width = int((points[7][0]-points[9][0])*1.1)
            sunglass_height = int((points[10][1]-points[8][1])/1.1)
            sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height), interpolation = cv2.INTER_CUBIC)
            transparent_region = sunglass_resized[:,:,:3] != 0
            face_resized_color[int(points[9][1]):int(points[9][1])+sunglass_height, int(points[9][0]):int(points[9][0])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region]
            
            #print('checking 5')
            # Add Moustache FILTER to the frame
            moustaches = cv2.imread(filters_moustache[filterIndex_moustache], cv2.IMREAD_UNCHANGED)
            moustache_width = int((points[11][0]-points[12][0])*1.1)
            moustache_height = int((points[10][1]-points[8][1])/2.1)
            moustache_resized = cv2.resize(moustaches, (moustache_width, moustache_height), interpolation = cv2.INTER_CUBIC)
            transparent_region = moustache_resized[:,:,:3] != 0

            face_resized_color[int(points[10][1]):int(points[10][1])+moustache_height, int(points[12][0]):int(points[12][0])+moustache_width,:][transparent_region] = moustache_resized[:,:,:3][transparent_region]
            

            # Resize the face_resized_color image back to its original shape
            frame[y:y+h, x:x+w] = cv2.resize(face_resized_color, original_shape, interpolation = cv2.INTER_CUBIC)

            # Add KEYPOINTS to the frame2
            for keypoint in points:
                cv2.circle(face_resized_color2, keypoint, 1, (0,255,0), 1)

            frame2[y:y+h, x:x+w] = cv2.resize(face_resized_color2, original_shape, interpolation = cv2.INTER_CUBIC)

            # Show the frame and the frame2
            cv2.imshow("Selfie Filters", frame)
    #        cv2.imshow("Facial Keypoints", frame2)

        # If the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

   #     print('checking 6')
    # Cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
   # t1=threading.Thread(target=task1, name='t1')

   # t1.join()
    t2.join()


