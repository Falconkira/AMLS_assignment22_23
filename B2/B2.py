import numpy as np
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import os
global image_paths, target_size
basedir = '../Datasets/cartoon_set'
grayscale = True
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
    
    if grayscale:
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype('uint8')
    else:
        gray = resized_image
        
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
    labels = {line.split('\t')[0] : int(line.split('\t')[1]) for line in lines[1:]}
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
    labels = np.array(all_labels)
    return landmark_features, labels

def printOutput(m, te_Y):
    print("accuracy", accuracy_score(te_Y, m))
    print("precision", precision_score(te_Y, m, average = None))
    print("recall", recall_score(te_Y, m, average = None))

#changing dataset with specific features needed
def selectingFeatures(x, y, tr_X, te_X):
    training = []
    for ele in tr_X.reshape((len(tr_X), 68*2)):
        training.append(ele[x:y])
    testing = []
    for ele in te_X.reshape((len(te_X), 68*2)):
        testing.append(ele[x:y])
    return training, testing


"""
training: 8194 testing: 2041
num of sample for each class: [1608, 1573, 1744, 1652, 1617]

svm ovr:
accuracy 0.34149926506614403
precision [0.32653061 0.24137931 0.22413793 0.37875752 0.43953935]
recall [0.34449761 0.17241379 0.15294118 0.45873786 0.60263158]

svm ovo:
accuracy 0.3410093091621754
precision [0.33254157 0.23741007 0.21967213 0.37524558 0.43939394]
recall [0.33492823 0.16256158 0.15764706 0.46359223 0.61052632]

KNN: k = 91
accuracy 0.329250367466928
precision [0.30406852 0.26558266 0.21960784 0.37665198 0.41330645]
recall [0.33971292 0.24137931 0.13176471 0.41504854 0.53947368]

adaboost: 35
accuracy 0.32778049975502205
precision [0.28351648 0.232      0.21524664 0.36860068 0.41366224]
recall [0.30861244 0.14285714 0.11294118 0.52427184 0.57368421]

svm ovr without gray:
accuracy 0.33620689655172414
precision [0.27314815 0.21086262 0.23050847 0.39795918 0.45698925]
recall [0.27699531 0.1598063  0.15560641 0.47215496 0.63909774]

svm ovo without gray:
accuracy 0.3381226053639847
precision [0.27654321 0.21543408 0.23225806 0.39919355 0.4540636 ]
recall [0.2629108  0.1622276  0.16475973 0.47941889 0.64411028]

svm ovr wihout gray with only eye:
accuracy 0.3089080459770115
precision [0.24210526 0.24468085 0.25603865 0.31724138 0.38714734]
recall [0.26995305 0.11138015 0.12128146 0.44552058 0.61904762]

svm ovo wihout gray with only eye:
accuracy 0.3089080459770115
precision [0.24145299 0.24864865 0.25345622 0.31724138 0.38714734]
recall [0.26525822 0.11138015 0.12585812 0.44552058 0.61904762]


svm ovr with only eye:
accuracy 0.308672219500245
precision [0.27272727 0.22058824 0.19254658 0.32673267 0.38652482]
recall [0.33014354 0.11083744 0.07294118 0.48058252 0.57368421]

svm ovo with only eye:
accuracy 0.3081822635962763
precision [0.27272727 0.21 0.20114943 0.32673267 0.3869258 ]
recall [0.32296651 0.10344828 0.08235294 0.48058252 0.57631579]

"""

def b2():
    tr_X, y = extract_features_labels()
    tr_Y = np.array([y, -(y - 1)]).T
    basedir = '../Datasets/cartoon_set_test'
    images_dir = os.path.join(basedir,'img')
    te_X, y = extract_features_labels()
    te_Y = np.array([y, -(y - 1)]).T
    tr_Y = list(zip(*tr_Y))[0]
    te_Y = list(zip(*te_Y))[0]

    print("Task B2 Results: ")

    svmovr = svm.SVC(kernel = "linear").fit(tr_X.reshape((len(tr_X), 68*2)), tr_Y)
    svmovr_pred = svmovr.predict(te_X.reshape((len(te_X), 68*2)))
    print("SVM OVR")
    printOutput(svmovr_pred, te_Y)

    svmovo = OneVsOneClassifier(svm.SVC(kernel = "linear")).fit(tr_X.reshape((len(tr_X), 68*2)), tr_Y)
    svmovo_pred = svmovo.predict(te_X.reshape((len(te_X), 68*2)))
    print("SVM OVO")
    printOutput(svmovo_pred, te_Y)

    """
    aknn = []
    #sqr(8194) = 91
    for k in range(1, 92):
        classifier = KNeighborsClassifier(n_neighbors=k, algorithm = 'brute')
        scores = cross_val_score(classifier, tr_X.reshape((len(tr_X), 68*2)), tr_Y, cv = 5)
        aknn.append([scores.mean(), scores.std()])

    aada = []

    for i in range(1, 200):    
        classifier = AdaBoostClassifier(n_estimators=i, random_state=0)
        scores = cross_val_score(classifier, tr_X.reshape((len(tr_X), 68*2)), tr_Y, cv = 5)
        aada.append([scores.mean(), scores.std()])
    """

    knn = KNeighborsClassifier(n_neighbors=91, algorithm = 'brute')
    knn.fit(tr_X.reshape((len(tr_X), 68*2)), tr_Y)
    knn_pred = knn.predict(te_X.reshape((len(te_X), 68*2)))
    print("KNN")
    printOutput(knn_pred, te_Y)

    ada = AdaBoostClassifier(n_estimators=35, random_state = 0)
    ada.fit(tr_X.reshape((len(tr_X), 68*2)), tr_Y)
    ada_pred = ada.predict(te_X.reshape((len(te_X), 68*2)))
    print("Adaboost")
    printOutput(ada_pred, te_Y)

    tx, tex = selectingFeatures(72, 96, tr_X, te_X)

    svmovrt = svm.SVC(kernel = "linear").fit(tx, tr_Y)
    svmovrt_pred = svmovrt.predict(tex)
    print("SVM OVR E")
    printOutput(svmovrt_pred, te_Y)

    svmovot = OneVsOneClassifier(svm.SVC(kernel = "linear")).fit(tx, tr_Y)
    svmovot_pred = svmovot.predict(tex)
    print("SVM OVO E")
    printOutput(svmovot_pred, te_Y)

    #preprocssing images without grayscale
    grayscale = False
    tr_X, y = extract_features_labels()
    tr_Y = np.array([y, -(y - 1)]).T
    basedir = '../Datasets/cartoon_set_test'
    images_dir = os.path.join(basedir,'img')
    te_X, y = extract_features_labels()
    te_Y = np.array([y, -(y - 1)]).T
    tr_Y = list(zip(*tr_Y))[0]
    te_Y = list(zip(*te_Y))[0]

    svmovr = svm.SVC(kernel = "linear").fit(tr_X.reshape((len(tr_X), 68*2)), tr_Y)
    svmovr_pred = svmovr.predict(te_X.reshape((len(te_X), 68*2)))
    print("SVM OVR C")
    printOutput(svmovr_pred, te_Y)

    svmovo = OneVsOneClassifier(svm.SVC(kernel = "linear")).fit(tr_X.reshape((len(tr_X), 68*2)), tr_Y)
    svmovo_pred = svmovo.predict(te_X.reshape((len(te_X), 68*2)))
    print("SVM OVO C")
    printOutput(svmovo_pred, te_Y)

    tx, tex = selectingFeatures(72, 96, tr_X, te_X)
    
    svmovrt = svm.SVC(kernel = "linear").fit(tx, tr_Y)
    svmovrt_pred = svmovrt.predict(tex)
    print("SVM OVR C E")
    printOutput(svmovrt_pred, te_Y)

    svmovot = OneVsOneClassifier(svm.SVC(kernel = "linear")).fit(tx, tr_Y)
    svmovot_pred = svmovot.predict(tex)
    print("SVM OVO C E")
    printOutput(svmovot_pred, te_Y)