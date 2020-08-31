from keras.models import load_model
import os
from utils import face_trainer
import cv2

data_dir = os.path.join(os.getcwd(), 'data/')
names = [name for name in os.listdir(data_dir)
         if os.path.isdir(os.path.join(data_dir, name))]
face_codex = {name: i+1 for i, name in enumerate(sorted(names))}

# Train LBPH
print('>> Training LBPH model.')
model = cv2.face.LBPHFaceRecognizer_create()
save_dir = os.path.join(os.getcwd(), 'recognition/lbph_trainer.yml')
face_trainer(face_codex, method='lbph', model=model, save_dir=save_dir)

# Train facenet
print('>> Training facenet.')
model = load_model('recognition/facenet_keras.h5')
save_dir = os.path.join(os.getcwd(), 'recognition/trainer.pickle')
face_trainer(face_codex, method='facenet', model=model, save_dir=save_dir)

print('>> Done training.')