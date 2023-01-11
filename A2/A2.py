import numpy as np
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import os
global image_paths, target_size
basedir = '../Datasets/celeba'
images_dir = os.path.join(basedir,'img')
labels_filename = 'labels.csv'

import numpy as np
# from keras.preprocessing import image
from tensorflow.keras.preprocessing import image
import cv2
import dlib

# PATH TO ALL IMAGES

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')


# how to find frontal human faces in an image using 68 landmarks.  These are points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth.

# The face detector we use is made using the classic Histogram of Oriented
# Gradients (HOG) feature combined with a linear classifier, an image pyramid,
# and sliding window detection scheme.  The pose estimator was created by
# using dlib's implementation of the paper:
# One Millisecond Face Alignment with an Ensemble of Regression Trees by
# Vahid Kazemi and Josephine Sullivan, CVPR 2014
# and was trained on the iBUG 300-W face landmark dataset (see https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
#     C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
#     300 faces In-the-wild challenge: Database and results.
#     Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image

def extract_features_labels():
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        labels:      an array containing the label for each image in which a face was detected
    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    labels = {line.split('\t')[0] : int(line.split('\t')[3]) for line in lines[1:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for img_path in image_paths:
            #windows: file_name= img_path.split('.')[2].split('\\')[-1]
            #mac: file_name= img_path.split('.')[2].split('/')[-1]
            file_name= img_path.split('.')[2].split('\\')[-1]

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_labels.append(labels[file_name])

    landmark_features = np.array(all_features)
    labels = (np.array(all_labels) + 1)/2 # simply converts the -1 into 0, so male=0 and female=1
    return landmark_features, labels

def printOutput(m, te_Y):
    print("accuracy", accuracy_score(list(zip(*te_Y))[0], m))
    print("precision", precision_score(list(zip(*te_Y))[0], m, average = None))
    print("recall", recall_score(list(zip(*te_Y))[0], m, average = None))

"""
svm
accuracy 0.890608875128999
precision [0.88607595 0.89494949]
recall [0.88983051 0.89134809]

knn
k = 17
accuracy 0.8875128998968008
precision [0.89200864 0.88339921]
recall [0.875 0.89939638]

adaboost 38
accuracy 0.8875128998968008
precision [0.88865096 0.88645418]
recall [0.87923729 0.89537223]
"""

def a2():
    #training:4795 testing:969
    tr_X, y = extract_features_labels()
    tr_Y = np.array([y, -(y - 1)]).T
    basedir = '../Datasets/celeba_test'
    images_dir = os.path.join(basedir,'img')
    te_X, y = extract_features_labels()
    te_Y = np.array([y, -(y - 1)]).T

    print("Task A2 Results: ")

    svmc = svm.SVC(kernel = "linear").fit(tr_X.reshape((len(tr_X), 68*2)), list(zip(*tr_Y))[0])
    svmc_pred = svmc.predict(te_X.reshape((len(te_X), 68*2)))
    print("SVM: ")
    printOutput(svmc_pred, te_Y)

    """
    aknn = []
    def img_knn(training_images, training_labels, test_images, test_labels, k):
        
        classifier = KNeighborsClassifier(n_neighbors=k, algorithm = 'brute')
        
        scores = cross_val_score(classifier, tr_X.reshape((len(tr_X), 68*2)), list(zip(*tr_Y))[0], cv = 5)

        aknn.append([scores.mean(), scores.std()])

    for i in range(1, 70):
        pred=img_knn(tr_X.reshape((len(tr_X), 68*2)), list(zip(*tr_Y))[0], te_X.reshape((len(te_X), 68*2)),list(zip(*te_Y))[0], i)

    aada = []
    def img_ada(training_images, training_labels, test_images, test_labels, i):
        
        classifier = AdaBoostClassifier(n_estimators=i, random_state=0)
        
        scores = cross_val_score(classifier, tr_X.reshape((len(tr_X), 68*2)), list(zip(*tr_Y))[0], cv = 5)

        aada.append([scores.mean(), scores.std()])

    for i in range(1, 200):
        pred=img_ada(tr_X.reshape((len(tr_X), 68*2)), list(zip(*tr_Y))[0], te_X.reshape((len(te_X), 68*2)),list(zip(*te_Y))[0], i)

    """

    print("Adaboost: ")
    ada = AdaBoostClassifier(n_estimators=38, random_state=0)
    ada.fit(tr_X.reshape((len(tr_X), 68*2)), list(zip(*tr_Y))[0])
    ada_pred = ada.predict(te_X.reshape((len(te_X), 68*2)))
    printOutput(ada_pred, te_Y)

    print("KNN: ")
    knn = KNeighborsClassifier(n_neighbors=41, algorithm = 'brute')
    knn.fit(tr_X.reshape((len(tr_X), 68*2)), list(zip(*tr_Y))[0])
    knn_pred = knn.predict(te_X.reshape((len(te_X), 68*2)))
    printOutput(knn_pred, te_Y)
