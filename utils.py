import cv2, pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report

VEC_SIZE = 256 * 256 * 3

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


def train(data, tags):
    """
    train svm using pca and junocam images
    :param data: nd-array containing unrolled dataset images
    :param tags: 1D array containing binary labels for each dataset image
    :return: svm, trained support vector machine
             pca, trained principal component analysis dimsionality reduction transform
    """
    
    # split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, tags, test_size = 0.2)

    # perform pca transform
    pca = PCA(n_components = 0.9, svd_solver = 'full').fit(X_train)
    print('Decomposed test set to', pca.components_.shape[0], 'components')
    X_train_PCA = pca.transform(X_train)

    # apply pca transform to x_test
    X_test_PCA = pca.transform(X_test)

    # fit classifier to data
    C_RANGE = np.logspace(-3, 3, 20)
    G_RANGE = np.logspace(-9, 1, 20)
    param_grid = { 'C': C_RANGE,
                   'gamma': ['scale', 'auto'],
                   'gamma': G_RANGE,
                   'degree': [2, 3, 4],
                   'kernel': ['rbf', 'poly'], 
                   'coef0': [0, 1],
                   'class_weight': ['balanced'] }

    svm = GridSearchCV(SVC(), param_grid, n_jobs = -1, verbose = 0)
    svm.fit(X_train_PCA, y_train)

    Copt = svm.best_params_['C'] # svm cost parameter
    Kopt = svm.best_params_['kernel'] # kernel function
    Gopt = svm.best_params_['gamma'] # gamma of RBF kernel
    Dopt = svm.best_params_['degree'] # degree of polynomial kernel
    Zopt = svm.best_params_['coef0'] # independent term in poly kernel

    print('\nOptimal SVM parameter values:')
    print('C:', Copt)
    print('kernel:', Kopt)
    print('gamma:', Gopt)
    print('degree:', Dopt)
    print('coef0:', Zopt, '\n')

    # generate report
    print('Calculating metrics...')
    y_pred = svm.predict(X_test_PCA)
    print(classification_report(y_test, y_pred))
    scores = cross_val_score(svm, X_test_PCA, y_test, cv = 6)
    print('\nAverage cross-validate score: ', scores.mean())

    return svm, pca


def load():
    svm = pickle.load(open('MODELS/svm_model_opt.sav', 'rb'))
    pca = pickle.load(open('MODELS/pca_model_opt.sav', 'rb'))
    return svm, pca

    
def save(svm, pca):
    pickle.dump(svm, open('MODELS/svm_model_opt.sav', 'wb'))
    pickle.dump(pca, open('MODELS/pca_model_opt.sav', 'wb'))


svm, pca = load()

def break_blocks(img):
    """
    recursively shift, divide, and classify an image
    :param img: 2048x2048 cv2 image
    :return: detections, nd-array shape 2048x2048 - contains accumulated binary labels at each pixel
    """
    
    full_size = img.shape[0]
    block_size = int(full_size / 2)

    if block_size > 64:
        detections = np.zeros((full_size, full_size))
        
        ## shift at larger block sizes (through testing, determined to be not very useful at smaller ones)
        if full_size > 1024:
            _size = 128
            _img = np.copy(img[_size:(full_size - _size), _size:(full_size - _size), :])
            detections[_size:(full_size - _size), _size:(full_size - _size)] += break_blocks(_img)
                
        blocks = np.zeros((4, block_size, block_size, 3))
        blocks[0, :, :, :] = img[0 : block_size, 0 : block_size, :]
        blocks[1, :, :, :] = img[0 : block_size, block_size : full_size, :]
        blocks[2, :, :, :] = img[block_size : full_size, 0 : block_size, :]
        blocks[3, :, :, :] = img[block_size : full_size, block_size : full_size, :]

        valid_indices = []
        testing_blocks = np.empty((4, VEC_SIZE))
        for i in range(4):
            block = cv2.resize(blocks[i, :, :], (256, 256), interpolation = cv2.INTER_NEAREST)
            green_locs = np.where((block[:, :, 0] == 0) & (block[:, :, 1] != 0) & (block[:, :, 2] != 0))[0]
            block = np.reshape(block, VEC_SIZE)
            zero_locs = np.where(block < 10)[0]
            if (zero_locs.size < block.size / 3) and (green_locs.size < 100):
                valid_indices = np.append(valid_indices, i)
                testing_blocks[i, :] = block
        
        current_dets = np.zeros(4)

        if len(valid_indices) != 0:
            testing_blocks = testing_blocks[np.array(valid_indices).astype(int), :]
            transform_test = pca.transform(testing_blocks / 255)
            split_pred = svm.predict(transform_test)
            index = 0
            for i in range(4):
                if i in valid_indices:
                    current_dets[i] = split_pred[index]
                    index += 1

        block_detections = np.zeros((full_size, full_size))
        block_detections[0 : block_size, 0 : block_size] = np.full((block_size, block_size), current_dets[0]) + break_blocks(blocks[0, :, :, :])
        block_detections[0 : block_size, block_size : full_size] = np.full((block_size, block_size), current_dets[1]) + break_blocks(blocks[1, :, :, :])
        block_detections[block_size : full_size, 0 : block_size] = np.full((block_size, block_size), current_dets[2]) + break_blocks(blocks[2, :, :, :])
        block_detections[block_size : full_size, block_size : full_size] = np.full((block_size, block_size), current_dets[3]) + break_blocks(blocks[3, :, :, :])

        detections += block_detections
        
        return detections
    else:
        return np.zeros((full_size, full_size))

    
def test(img_loc):
    """
    contrast and classify new test image with divide/shift/conquer pipeline
    :param img_loc: path to 2048x2048 image
    :return: detections, nd-array shape 2048x2048 - contains accumulated binary labels at each pixel
    """
    
    img = contrast_resize(img_loc, (2048, 2048))
    img = img[:, :, :3]

    detections = break_blocks(img)
    detections = np.rint(detections)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (18, 9));
    
    ax1.imshow(img.astype(int).clip(0, 255));
    ax1.title.set_text('Original Test Image');
    ax1.axis('off');
    
    normalized = detections / np.linalg.norm(detections)
    
    ax2.imshow(normalized.astype(float).clip(0, 1))
    ax2.title.set_text('Normalized Decisions');
    ax2.axis('off');
    
    superimposed = np.empty(img.shape)
    for i in range(3):
        superimposed[:, :, i] = normalized * img[:, :, i]
    superimposed *= 255.0 / superimposed.max()
        
    ax3.imshow(superimposed.astype(int).clip(0, 255));
    ax3.title.set_text('Superimposed Decisions');
    ax3.axis('off');
    
    fig.savefig('FIGURES/detection_maps_' + str(img_loc.split('/')[-1].split('.')[0].split('-')[0]) + '.png')
    
    return detections