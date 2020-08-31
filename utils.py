import cv2
import numpy as np
import dlib
import os
import time
from pickle import dump, HIGHEST_PROTOCOL


def draw_rectangles(image_array, rectangles, make_copy=True, color=(0, 0, 0), fill=False):
    """
    Draws rectangles on the image array.

    :param image_array: 3d numpy array of image
    :param rectangles: list of tuples, of size (x, y, w, h), corresponding to bounding boxes of rectangles
    :param make_copy: bool. Set to true to return a new image_array to draw on, false to draw on original one
    :param color: Tuple of (r,g,b) values.
    :param fill: Bool. set to True to fill the rectangle
    :return: Returns a copy of image_array, with boxes drawn on
    """

    if make_copy:
        image_array = image_array.copy()

    for (x, y, w, h) in rectangles:
        thickness = -1 if fill else int(.0025*np.sqrt(image_array.size))
        cv2.rectangle(img=image_array, pt1=(x, y), pt2=(x+w, y+h), color=color, thickness=thickness)

    return image_array


def draw_x(image_array, make_copy=True):
    """
    Draws a red X over the inputted image.

    :param image_array: ndarray, of dim 2 (greyscale) or 3 (color)
    :param make_copy: Bool. Set to True to return a copy of inputted image.
    :return: image of same size and shape as image_array
    """

    if make_copy:
        image_array = image_array.copy()

    if image_array.ndim == 2:
        height, width = image_array.shape
    elif image_array.ndim == 3:
        height, width, _ = image_array.shape
    else:
        raise Exception('image_array argument for draw_x() must be of dim 2 (grey scale) or 3 (color).')

    thickness = int(.01*np.sqrt(height*width))
    cv2.line(image_array, pt1=(0, 0), pt2=(width, height), color=(0, 0, 255), thickness=thickness)
    cv2.line(image_array, pt1=(width, 0), pt2=(0, height), color=(0, 0, 255), thickness=thickness)

    return image_array


def detect_face(image_array, method='cnn'):
    """
    Perform face detection on an image, using the requested method. Returns a list of tuples, corresponding
    to the bounding boxes of all detected faces.

    :param image_array: ndarray of dim 2 or 3, for grey scale or color images respectively.
    :param method: string, indicating the desired method of face detection. Current options are:
        'cnn' : Uses a pre-trained cnn which uses an MMOD loss function.
        'hog' : Computes the histogram of oriented gradients of the image.
        'haar': Performs Haar cascades using a pre-trained xml file.

    :returns faces: List of tuples, corresponding to the bounding boxes of all detected faces. Each tuple
                    is of shape (x, y, w, h)
    """

    if method == 'cnn':
        cnn_filepath = 'detection/mmod_human_face_detector.dat'
        face_detector = dlib.cnn_face_detection_model_v1(cnn_filepath)  # Outputs list of dlib rectangles
        faces = face_detector(image_array, 0)
        return [(face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()) for face in faces]
    elif method == 'hog':
        face_detector = dlib.get_frontal_face_detector()
        faces = face_detector(image_array, 0)
        return [(face.left(), face.top(), face.width(), face.height()) for face in faces]
    elif method == 'haar':
        face_cascade = cv2.CascadeClassifier('detection/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image_array)  # Returns a tuple of 1D numpy arrays, of length 4
        return [tuple(face) for face in faces]

    return False


def face_trainer(face_codex, method, model, image_dir='data/'):
    """
    Trains a model using the provided directory of face pictures.

    :param face_codex: Dict containing integer ids of every unique person to classify. Has the following form:
        face_codex = {'jeff': 1, 'katy': 2, ...}
    :param method: String indicating the recognition model you want to train. Current options:
            'lbph'
            'facenet'
    :param model: Model object for whatever method you are training.
            'lbph' : model = cv2.face.LBPHFaceRecognizer_create()
            'facenet' : Keras model
    :param image_dir: String of the directory containing all images to train with. Directory MUST
    have the following layout:

        data
          - jeff
              1.jpg
              2.jpg
              ....
          - katy
              1.jpg
              2.jpg
              ...

        In the above example, you would then put image_directory='data/' as the argument

    :param save_dir: String of filename that the trained model will be saved to. The extension is automatically
            determined based on the desired training method (yml for lbph, pickle for facenet).
    """

    if method == 'lbph':

        # Compile list of all images and their corresponding ids. id is the name of directory the images are in
        list_of_images = []
        list_of_ids = []
        for name in os.listdir(image_dir):
            directory = image_dir+name
            if not os.path.isdir(directory):
                continue
            id = face_codex[name]

            for image_file in os.listdir(directory):
                if image_file.endswith('.jpg'):
                    img = cv2.imread(directory+'/'+image_file, 0)
                    list_of_images.append(np.array(img))
                    list_of_ids.append(id)

        ts = time.time()
        num_people = sum(os.path.isdir(image_dir+i) for i in os.listdir(image_dir))
        print('>> Training LBPH model off of {} images and {} people...'.format(len(list_of_ids), num_people))
        model.train(list_of_images, np.array(list_of_ids))
        print('>> Done training LBPH model. Total time: {:.2f} s'.format(time.time()-ts))
        model.write(save_dir)
        print('>> Saving LBPH model to: {}'.format(save_name))

    elif method == 'facenet':

        # Package training data into arrays of shape (m, 160, 160, 3).
        train_x, train_y = _load_training_samples_for_facenet(face_codex, image_dir=image_dir)

        # Calculate average embeddings of each face in the data directory. Shape is (m, 128)
        print('>> Running FaceNet model on training set...')
        embeddings = model.predict_on_batch(train_x)
        print('>> Finished running FaceNet model.')

        # Build dict of average embeddings for all people in training set
        database = {}
        for name, id in face_codex.items():

            array = embeddings[train_y == id, :]
            database[name] = np.mean(array, axis=0)

        # Save avg embeddings
        with open(save_dir, 'wb') as f:
            dump(database, f, protocol=HIGHEST_PROTOCOL)
        print('>> Saved FaceNet embeddings to '+save_name)
    else:
        print(">> Did not recognize '{}' for face training. Valid options are 'lbph' and 'facenet'.".format(method))
        return False


def recognize_face(image, method, model, face_codex, embeddings=None):
    """
    Given a cropped image of a face, returns the identity of that person and its confidence in its prediction.

    :param image: ndarray of size (w, w, 3). Channels should be in BGR order.
    :param method: String indicating the method to perform face recognition. Current options are:
            'lbph' : Local binary pattern histogram.
            'facenet' : runs image as a mini-batch through the FaceNet CNN
    :param model: Model object to perform the face recognition, depending on the desired method.
            'lbph' : trained recognizer object, from cv2.face.LBPHFaceRecognizer_create()
            'facenet' : pretrained keras model of FaceNet
    :param face_codex: Dict containing integer ids for all unique people to classify.
    :param embeddings: Dict of embeddings for all classified users. Only used for facenet recognition
    :return: closest_match: String, name of the person who it identified
             confidence: Float from 0 to 1, indicating its confidence in is prediction.
    """

    if image.ndim != 3:
        raise Exception('Image array for recognize_faces() must be ndarray of dim 3.')

    if method == 'lbph':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        id, dist = model.predict(image)

        inverted_codex = {v: k for k, v in face_codex.items()}
        closest_match = inverted_codex[id]
        confidence = max(1 - dist/100, 0)
        if confidence < .5:
            closest_match = 'Unknown'
            confidence = 0

        return closest_match, confidence

    elif method == 'facenet':
        img = _load_image_for_facenet(image)
        img_embedding = model.predict_on_batch(img)

        dist_dict = {name: np.linalg.norm(img_embedding - emb) for name, emb in embeddings.items()}

        closest_match = min(dist_dict, key=dist_dict.get)
        confidence = max(1 - dist_dict[closest_match]/100, 0)
        if confidence < .5:
            closest_match = 'Unknown'
            confidence = 0

        return closest_match, confidence


def standardize(image_array):
    """
    Standardize image data by subtracting out mean and normalizing to standard deviation.

    :param image_array: ndarray of a single image
    :return: ndarray of same shape as image_array
    """
    return (image_array - image_array.mean())/image_array.std()


def _load_image_for_facenet(image):
    """
    Loads in an image and prepares it for the FaceNet CNN (resizing, standardizing, etc).

    :param image: ndarray of image, of shape (W, W, 3)
    :return: ndarray of shape (1, 160, 160, 3)
    """

    image = cv2.resize(image, dsize=(160, 160))
    image = np.expand_dims(image, axis=0)
    image = standardize(image)
    return image


def _load_training_samples_for_facenet(face_codex, image_dir='data/'):
    """
    Loads all sample images from data directory, packages them into an array so that they can passed through FaceNet.
    Note that the FaceNet CNN we're using requires color images of size 160x160, hence the reshaping.

    :param face_codex: Dict of integer ids for each unique person in training data. Should look like following:
            face_codex = { 'jeff' : 1,
                           'katy' : 2,
                           ... }
    :param image_directory: String. Name of folder that contains all subfolders of image samples. Subfolders
            should only contain images and should have the name of the person in those images.
    :return: images: ndarray of shape (m, 160, 160, 3), where m = number of training samples
             ids: ndarray of shape (m,), representing the ground truth ids of the images
    """

    list_of_images = []
    list_of_ids = []
    for folder in os.listdir(image_dir):
        path = image_dir+folder
        if not os.path.isdir(path):
            continue
        id = face_codex[folder]

        for image_file in os.listdir(path):
            if image_file.endswith('.jpg'):
                img = _load_image_for_facenet(cv2.imread(path+'/'+image_file))

                list_of_images.append(img)
                list_of_ids.append(id)

    images = np.concatenate(list_of_images, axis=0)
    ids = np.stack(list_of_ids, axis=0)

    return images, ids


def label_frame(frame):
    """
    Adds a bunch of text commands onto image. Putting this stuff here because it's clunky and looks obnoxious
    in the main loop.

    :param frame: ndarray, image of shape (w, w, 3)
    :return: ndarray of same size as input
    """

    cv2.putText(frame, 'Detection methods:', org=(10, int(.5*frame.shape[0])),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    cv2.putText(frame, '1 - Haar Cascades', org=(10, int(.55*frame.shape[0])),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(230, 0, 0), thickness=1)
    cv2.putText(frame, '2 - H.O.G.', org=(10, int(.6*frame.shape[0])),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 230, 0), thickness=1)
    cv2.putText(frame, '3 - CNN', org=(10, int(.65*frame.shape[0])),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 230), thickness=1)
    cv2.putText(frame, '0 - None', org=(10, int(.7*frame.shape[0])),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=1)
    cv2.putText(frame, 'Recognition methods:', org=(10, int(.75*frame.shape[0])),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    cv2.putText(frame, 'q - L.B.P.H', org=(10, int(.8*frame.shape[0])),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 255), thickness=1)
    cv2.putText(frame, 'w - FaceNet CNN', org=(10, int(.85*frame.shape[0])),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 0), thickness=1)
    cv2.putText(frame, 's - save face', org=(int(.01*frame.shape[1]), int(.98*frame.shape[0])),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    cv2.putText(frame, 'z - delete last face', org=(int(.3*frame.shape[1]), int(.98*frame.shape[0])),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    cv2.putText(frame, 'esc - Close program', org=(int(.72*frame.shape[1]), int(.98*frame.shape[0])),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)

    return frame
