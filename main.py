from utils import *
import cv2
import numpy as np
import time
import os
from keras.models import load_model
from pickle import load as load_pickle
import requests
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# ======================================================

DETECTION_COLOR_MAP = {'haar': (255, 0, 0),
                       'hog': (0, 255, 0),
                       'cnn': (0, 0, 255)}

RECOGNITION_COLOR_MAP = {'lbph': (255, 0, 255),
                         'facenet': (255, 255, 0)}

# How much to downsize the image for the detection step
DETECTION_SCALE_FACTOR = .25
DETECTION_METHOD = None
RECOGNITION_METHOD = None

# Load FaceNet model/weight. Download if necessary
print('>> Loading face recognition models.')
facenet_dir = os.path.join(os.getcwd(), 'recognition/facenet')
facenet_model_dir = os.path.join(facenet_dir, 'model.h5')
model_url = "https://drive.google.com/uc?export=download&id=1PZ_6Zsy1Vb0s0JmjEmVd8FS99zoMCiN1"
if not os.path.exists(facenet_model_dir):
    print('Could not find facenet model.')
    confirm_download = input("The model is 88 Mb. Download now? (y/n): ")
    if confirm_download.lower() == 'y':
        r = requests.get(model_url, stream=True)
        with open(facenet_model_dir, "wb") as f:
            for data in tqdm(r.iter_content()):
                f.write(data)
        print('>> Downloaded facenet model to: {}'.format(facenet_model_dir))
facenet_weights_dir = os.path.join(facenet_dir, 'weights.h5')
weights_url = "https://drive.google.com/uc?export=download&id=1e6PHRlIeayAsvRGpYUwvstklvJy-3H5B"
if not os.path.exists(facenet_weights_dir):
    print('Could not find weights for facenet model.')
    confirm_download = input("Weights file is 88 Mb. Download now? (y/n): ")
    if confirm_download.lower() == 'y':
        r = requests.get(weights_url, stream=True)
        with open(facenet_weights_dir, "wb") as f:
            for data in tqdm(r.iter_content()):
                f.write(data)
        print('>> Downloaded facenet weights to: {}'.format(facenet_weights_dir))
facenet_model = load_model('recognition/facenet/model.h5')
facenet_model.load_weights('recognition/facenet/weights.h5')

# For some reason the first prediction takes way longer
facenet_model.predict_on_batch(np.zeros(shape=(1, 160, 160, 3)))
with open('recognition/facenet/trainer.pickle', 'rb') as f:
    facenet_embeddings = load_pickle(f)

# Load LBPH model
lbph_model = cv2.face.LBPHFaceRecognizer_create()
lbph_model.read('detection/lbph_trainer.yml')
print('>> Done loading face recognition models.')

# Create folder(s) to store training data
name = input("Enter your name: ").lower()
if not os.path.exists('data/'):
    os.makedir('data/')
data_dir = os.path.join(os.getcwd(), 'data/')
names = [name for name in os.listdir(data_dir)
         if os.path.isdir(os.path.join(data_dir, name))]
face_codex = {name: i+1 for i, name in enumerate(sorted(names))}
save_directory = os.path.join('data', str(name))
if not os.path.exists(save_directory):
    os.makedir(save_directory)

# Load up webcam. Default res is 1280x720
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception('Could not open camera!')

# main loop
deleted_timestamp = 0
screenshot_timestamp = 0
saved_face = None
fps_list = [0]*6  # Computes fps as a moving average
while True:

    start_time = time.time()
    ret, frame = cap.read()
    height, width = frame.shape[0:2]

    if ret:

        # Perform face detection on a scaled down image, for better performance
        frame_scaled = cv2.resize(frame, 
                                  (int(DETECTION_SCALE_FACTOR*width), int(DETECTION_SCALE_FACTOR*height)))
        
        # Write info onto screen
        cv2.putText(frame, 'fps: {:.2f}'.format(np.mean(fps_list)), 
                    org=(10, int(.45*frame.shape[0])),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, 
                    color=(0, 0, 0), 
                    thickness=2)
        frame = label_frame(frame)

        # If a screenshot was recently taken, display it in top left
        if time.time() - deleted_timestamp <= 2 and saved_face is None:
            cv2.putText(frame, 
                        'Deleted last face.', 
                        org=(10, 30), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, 
                        color=(255, 255, 255), 
                        thickness=2)
            screenshot_h, screenshot_w, _ = deleted_face.shape
            frame[40:40+screenshot_h, 10:10+screenshot_w, :] = deleted_face
        if time.time() - screenshot_timestamp <= 2 and saved_face is not None:
            cv2.putText(frame, 
                        'Face saved!', 
                        org=(10, 30), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, 
                        color=(255, 255, 255), 
                        thickness=2)
            screenshot_h, screenshot_w, _ = saved_face.shape
            frame[40:40+screenshot_h, 10:10+screenshot_w, :] = saved_face

        if DETECTION_METHOD:

            # Get bounding boxes (x, y, w, h) of all faces in frame
            faces_scaled = detect_face(frame_scaled, method=DETECTION_METHOD)

            # Scale face bboxes back up
            faces = []
            for face in faces_scaled:
                faces.append(tuple(int(x/DETECTION_SCALE_FACTOR) for x in face))

            # Draw box around faces
            color = DETECTION_COLOR_MAP[DETECTION_METHOD]
            frame = draw_rectangles(frame, faces, color=color)

        if RECOGNITION_METHOD:

            color = RECOGNITION_COLOR_MAP[RECOGNITION_METHOD]

            for (x, y, w, h) in faces:

                cropped_face = frame_raw[y:y+h, x:x+w]

                if RECOGNITION_METHOD == 'lbph':
                    face_resized = cv2.resize(cropped_face, dsize=(200, 200))
                    name, confidence = recognize_face(cropped_face, method='lbph', model=lbph_model,
                                                      face_codex=face_codex)

                elif RECOGNITION_METHOD == 'facenet':

                    name, confidence = recognize_face(cropped_face, method='facenet', model=facenet_model,
                                                      face_codex=face_codex, embeddings=facenet_embeddings)

                conf_text = str(round(100*confidence, 2))+'%' if confidence != 0 else ''
                cv2.putText(frame, 
                            name, 
                            org=(x, y-int(.01*frame.shape[0])),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=1.5, 
                            color=color, 
                            thickness=2)
                cv2.putText(frame, 
                            conf_text, 
                            org=(x+150, y-int(.01*frame.shape[0])),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=1.5, 
                            color=color, 
                            thickness=2)

        cv2.imshow('frame', frame)
        fps_list.pop(0)
        fps_list.append(1/(time.time() - start_time))

    # User input commands
    k = cv2.waitKey(1) & 0xFF
    if k == ord('0'):
        print('>> Turning off face detection and recognition.')
        DETECTION_METHOD = None
        RECOGNITION_METHOD = None
    elif k == ord('1'):
        print('>> Switching to haar detection.')
        DETECTION_METHOD = 'haar'
    elif k == ord('2'):
        print('>> Switching to histogram of oriented gradients (HOG) detection.')
        DETECTION_METHOD = 'hog'
    elif k == ord('3'):
        print('>> Switching to MMOD CNN detection.')
        DETECTION_METHOD = 'cnn'
    elif k == ord('s'):
        if DETECTION_METHOD:
            if len(faces) == 1:
                x, y, w, h = faces[0]
                screenshot_timestamp = time.time()
                saved_face = frame_raw[y:y+h, x:x+w]
                saved_face = cv2.resize(src=saved_face, dsize=(200, 200))

                # Save to directory
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)
                    print('>> Creating save directory: '+save_directory)
                saved_filename = save_directory+'/'+str(len(os.listdir(save_directory))+1)+'.jpg'
                cv2.imwrite(saved_filename, saved_face)
                print('>> Saving face grab to: {}'.format(saved_filename))

            elif len(faces) > 1:
                print('>> Too many faces on screen.')
            else:
                print('>> No face was detected, so no screenshot was saved.')
        else:
            print('>> Could not save face, as no detection method is active.')
    elif k == ord('z'):
        if saved_face is not None:
            deleted_face = draw_x(saved_face)
            deleted_timestamp = time.time()
            saved_face = None
            os.remove(saved_filename)
            print('>> Deleted last screen grab: '+saved_filename)
        else:
            print('>> No recent face grab to delete.')
    elif k == ord('q'):
        if DETECTION_METHOD:
            print('>> Switching to LBPH facial recognition.')
            RECOGNITION_METHOD = 'lbph'
        else:
            print('>> Choose a detection method before picking a recognition method')
    elif k == ord('w'):
        if DETECTION_METHOD:
            print('>> Switching to FaceNet facial recognition.')
            RECOGNITION_METHOD = 'facenet'
        else:
            print('>> Choose a detection method before picking a recognition method')
    elif k == ord('b'):
        print('>> Saved screenshot!')
        cv2.imwrite('frame.jpg', frame)
    elif k == 27:
        print(">> Quitting program.")
        break

cap.release()
cv2.destroyAllWindows()
