import cv2
import time
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(dir_path, 'exercise.avi')

def record():
    # Open the camera
    cap = cv2.VideoCapture(0)
    if (cap.isOpened() == False): 
        print("Error reading video stream")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('exercise.avi', 
            cv2.VideoWriter_fourcc(*'MJPG'),
            10, size
        )
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        # Read and display each frame
        ret, img = cap.read()
        cv2.imshow('PoseTrainer Video Feed',img)
        k = cv2.waitKey(125)
        # Specify the countdown
        j = 30 # in frames
        # set the key for the countdown to begin
        if k == ord('q'):
            while j>=10:
                ret, img = cap.read()
                # Display the countdown after 10 frames
                if j%10 == 0:
                    # draw the countdown using puttext
                    cv2.putText(
                        img,
                        str(j//10),
                        (250,250), 
                        font, 
                        7,
                        (255,255,255),
                        10,
                        cv2.LINE_AA)
                cv2.imshow('PoseTrainer Video Feed',img)
                cv2.waitKey(125)
                j = j-1
            else:
                j = 250 # frames to capture
                while(j >= 10):
                    ret, img = cap.read()
                    if ret == True: 
                        # Write the frame into the
                        # file 'exercise.avi'
                        result.write(img)
                        # Display the frame
                        # saved in the file
                        if j%10 == 0:
                            # draw the countdown using puttext
                            cv2.putText(
                                img,
                                'Rec..' + str(j//10),
                                (250,250), 
                                font, 
                                2,
                                (0,0,255),
                                10,
                                cv2.LINE_AA)
                        cv2.imshow('PoseTrainer Video Feed', img)
                        # Press S on keyboard 
                        # to stop the process
                        if cv2.waitKey(1) & 0xFF == ord('s'):
                            break
                    j = j-1
                else:
                    j = 30
                    while(j >= 10):
                        j = j-1
                        ret, img = cap.read()
                        if ret == True:
                            # Display the frame
                            # saved in the file
                            cv2.putText(
                                img,
                                'Done!',
                                (250,250), 
                                font, 
                                4,
                                (0,255,0),
                                5,
                                cv2.LINE_AA)
                            cv2.imshow('PoseTrainer Video Feed', img)
                    print("The video was successfully saved")
                    break
        # Press Esc to exit
        elif k == 27:
            break
    cap.release()
    result.release()
    cv2.destroyAllWindows()
