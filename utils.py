import cv2, pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

VEC_SIZE = 256 * 256 * 3
svm = pickle.load(open('MODELS/svm_model_opt.sav', 'rb'))
pca = pickle.load(open('MODELS/pca_model_opt.sav', 'rb'))


def contrast_resize(img_loc, new_size):
    """
    contrast new test image with CLAHE cv2 function
    :param img_loc: path to 2048x2048 image
    :param new_size: 2D nd-array containing new x, y dimensions of image
    :return: img, resized and contrasted cv2 image
    """
    
    img = cv2.imread(img_loc, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8, 8))
    img_l = clahe.apply(l)
    img_l = cv2.merge((img_l, a, b))
    final = cv2.cvtColor(img_l, cv2.COLOR_LAB2BGR)
    final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    img = cv2.resize(final, new_size, interpolation = cv2.INTER_NEAREST)
    return img


