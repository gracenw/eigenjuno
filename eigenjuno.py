import cv2, pickle, os, json, math, fnmatch
import spiceypy as spice
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from datetime import datetime

np.seterr(divide = 'ignore', invalid = 'ignore') ## divide by zero errors pop up sometimes - they aren't critical, so we ignore them :)

VEC_SIZE = 256 * 256 * 3
VEC_DIM = (256, 256, 3)

def contrast_resize(img_loc, new_size):
    """
    contrast new test image with CLAHE cv2 function
    ---
    parameters: 
    img_loc - path to 2048x2048 image
    new_size - 2D nd-array containing new x, y dimensions of image
    ---
    returns:
    img - resized and contrasted cv2 image
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
    train svm using pca and junocam images, saves models for safekeeping
    ---
    parameters: 
    data - nd-array containing unrolled dataset images
    tags - 1D array containing binary labels for each dataset image
    ---
    returns:
    svm - trained support vector machine
    pca - trained principal component analysis dimsionality reduction transform
    """
#     transform_data = np.empty((data.shape[0] * 4, VEC_SIZE))
#     transform_tags = np.empty(data.shape[0] * 4)
    
#     for i in range(data.shape[0]):
#         transform_data[(i * 4) + 0, :] = np.copy(data[i, :])
#         transform_data[(i * 4) + 1, :] = np.rot90(data[i, :].reshape(VEC_DIM), k = 1, axes = (0, 1)).reshape(VEC_SIZE)
#         transform_data[(i * 4) + 2, :] = np.rot90(data[i, :].reshape(VEC_DIM), k = 2, axes = (0, 1)).reshape(VEC_SIZE)
#         transform_data[(i * 4) + 3, :] = np.rot90(data[i, :].reshape(VEC_DIM), k = 3, axes = (0, 1)).reshape(VEC_SIZE)
#         transform_tags[(i * 4) + 0] = tags[i]
#         transform_tags[(i * 4) + 1] = tags[i]
#         transform_tags[(i * 4) + 2] = tags[i]
#         transform_tags[(i * 4) + 3] = tags[i]
        
#     data = np.copy(transform_data)
#     tags = np.copy(transform_tags)
    X_train, X_test, y_train, y_test = train_test_split(data, tags, test_size = 0.2) ## split data into training and test sets

    pca = PCA(n_components = 0.9, svd_solver = 'full').fit(X_train) ## generate pca transform
    print('Decomposed test set to', pca.components_.shape[0], 'components')
    X_train_PCA = pca.transform(X_train)
    X_test_PCA = pca.transform(X_test) ## apply pca transform to x_test

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
    svm.fit(X_train_PCA, y_train) ## fit classifier to data

    Copt = svm.best_params_['C'] ## svm cost parameter
    Kopt = svm.best_params_['kernel'] ## kernel function
    Gopt = svm.best_params_['gamma'] ## gamma of RBF kernel
    Dopt = svm.best_params_['degree'] ## degree of polynomial kernel
    Zopt = svm.best_params_['coef0'] ## independent term in poly kernel

    print('\nOptimal SVM parameter values:')
    print('C:', Copt)
    print('kernel:', Kopt)
    print('gamma:', Gopt)
    print('degree:', Dopt)
    print('coef0:', Zopt, '\n')

    print('Calculating metrics...') ## generate report
    y_pred = svm.predict(X_test_PCA)
    print(classification_report(y_test, y_pred))
    scores = cross_val_score(svm, X_test_PCA, y_test, cv = 6)
    print('\nAverage cross-validate score: ', scores.mean())
    
    pickle.dump(svm, open('MODELS/svm_model_' + datetime.now().strftime("%d:%m:%Y_%H:%M:%S") + '.sav', 'wb'))

    return svm, pca


def load_models():
    """
    load optimal svm and pca models
    ---
    parameters: 
    none
    ---
    returns:
    svm - optimal trained support vector machine
    pva - optimal trained principal component analysis dimsionality reduction transform
    """
    svm = pickle.load(open('MODELS/svm_model_opt.sav', 'rb'))
    pca = pickle.load(open('MODELS/pca_model_opt.sav', 'rb'))
    return svm, pca

    
def save_models(svm, pca):
    """
    save optimal svm and pca models
    ---
    parameters: 
    svm - optimal trained support vector machine
    pva - optimal trained principal component analysis dimsionality reduction transform
    ---
    returns:
    none
    """
    pickle.dump(svm, open('MODELS/svm_model_opt.sav', 'wb'))
    pickle.dump(pca, open('MODELS/pca_model_opt.sav', 'wb'))


svm, pca = load_models()

def break_blocks(img):
    """
    recursively shift, divide, and classify an image
    ---
    parameters: 
    img - 2048x2048x3 cv2 image
    ---
    returns:
    detections - nd-array shape 2048x2048 containing accumulated bianry labels at each pixel
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

    
def test(TEST_DIR, TEST_NAME):
    """
    contrast and classify new test image with divide/shift/conquer pipeline
    ---
    parameters: 
    img_loc - path t0 2048x2048 test image
    ---
    returns:
    detections - nd-array of shape 2048x2048 containing accumulated bianry labels at each pixel
    """
    img_loc = os.path.join(TEST_DIR, str(TEST_NAME) + '-Stitched.png')
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
    
    fig.savefig('FIGURES/detection_maps_' + str(TEST_NAME) + '.png')
    
    return detections


def read_data(TRAIN_DIR, TRAIN_SIZE):
    """
    read training/validation image and label data from dataset
    ---
    parameters: 
    TRAIN_DIR - os.path object containing location of training images
    TRAIN_SIZE - number of distinct images in dataset
    ---
    returns:
    data - nd-array of shape TRAIN_SIZExVEC_DIM containing all image data of dataset
    tags - 1D-array containing associated binary labels of dataset images
    """
    tags = np.empty(TRAIN_SIZE)
    data = np.empty((TRAIN_SIZE, VEC_SIZE))

    for r, d, f in os.walk(TRAIN_DIR + 'ONE/'):
        for file in f:
            num = int(file.split('.')[0])
            tags[num - 1] = 1
            img = mpimg.imread(os.path.join(TRAIN_DIR, 'ONE/' + str(num) + '.png'))
            data[num - 1, :] = img[:, :, :3].reshape(VEC_SIZE)

    for r, d, f in os.walk(TRAIN_DIR + 'ZERO/'):
        for file in f:
            num = int(file.split('.')[0])
            tags[num - 1] = 0
            img = mpimg.imread(os.path.join(TRAIN_DIR, 'ZERO/' + str(num) + '.png'))
            data[num - 1, :] = img[:, :, :3].reshape(VEC_SIZE)

    tags = tags.astype('uint8')
    
    return data, tags


def load_kernels(KERNEL_DIR):
    """
    load appropriate SPICE kernels for computing geometries
    ---
    parameters: 
    KERNEL_DIR - os.path object containing location of all needed kernels
    ---
    returns:
    none
    """
    KERNELS = []
    for r, d, f in os.walk(KERNEL_DIR):
        for file in f:
            if not fnmatch.fnmatch(file, '*Zone.Identifier'):
                KERNELS.append(file)

    spice.kclear()
    for k in KERNELS:
        spice.furnsh(os.path.join(KERNEL_DIR, k))
        
        
def get_etime(TEST_DIR, TEST_NAME):
    """
    convert image time in UTC to ephemeris time
    ---
    parameters: 
    TEST_DIR - os.path object containing loation of TEST_NAME test image
    TEST_NAME - test image number
    ---
    returns:
    et - SPICE ephemeris time object
    """
    with open(os.path.join(TEST_DIR, TEST_NAME + '-Metadata.json'), 'r') as f:
        img_json = json.load(f)
        image = img_json['FILE_NAME']
        image_time = img_json['START_TIME']
    et = spice.str2et(image_time)
    return et
    

def get_position(et):
    """
    get position and orientation of junocam relative to jupiter
    ---
    parameters: 
    et - SPICE ephemeris time of the spacecraft at unknown
    ---
    returns:
    long - planetocentric longitude of junocam projected onto jupiter
    lat - planetocentric latitude of junocam prohect onto jupiter
    orient - 3x3 array describing the orientation of the camera relative to jupiter
    """
    pos, lt = spice.spkpos('JUNO', et, 'IAU_JUPITER', 'NONE', 'JUPITER')

    JUNO_TO_CUBE = np.matrix([[-0.0059163, -0.0142817, -0.9998805], 
                              [ 0.0023828, -0.9998954,  0.0142678], 
                              [-0.9999797, -0.0022981,  0.0059497]])
    CUBE_TO_CAM = (R.from_euler('zyx', [0.69, -0.469,  0.583])).as_matrix()
    rot_matrix = CUBE_TO_CAM * JUNO_TO_CUBE
    pos = rot_matrix * pos.reshape(-1, 1)

    _, long, lat = spice.reclat(np.ravel(pos))
    long, lat = np.array([long, lat]) * 180 / math.pi

    orient = spice.pxform('IAU_JUPITER', 'JUNO_JUNOCAM', et)
    
    return long, lat, orient


def raster_to_planeto(TEST_DIR, TEST_NAME):
    """
    convert surface raster of rectangular coordinates to planetocentric long/lat
    ---
    parameters: 
    TEST_DIR - os.path object containing loation of TEST_NAME test image
    TEST_NAME - test image number
    ---
    returns:
    coords - nd-array of shape 2048x2048x2 containing the planetocentric latitude and longitude of each point in image
    """
    raster = np.load(os.path.join(TEST_DIR, str(TEST_NAME) + '-Raster.npy'))
    coords = np.empty((2048, 2048, 2))
    
    for i in range(2048):
        for j in range(2048):
            if abs(np.mean(raster[i, j, :])) != 0:
                _, long, lat = spice.reclat(raster[i, j, :])
                coords[i, j, :] = np.array([long, lat]) * 180 / math.pi
            else:
                coords[i, j, :] = np.array([np.nan, np.nan]) # outside range
                
    np.save(os.path.join(TEST_DIR, str(TEST_NAME) + '-Coords'), coords)
    
    return coords


def log_detections(TEST_DIR, TEST_NAME, detections):
    """
    record long/lat of areas where storms are detected in log file
    ---
    parameters: 
    detections - nd-array of shape 2048x2048 containing accumulated bianry labels at each pixel
    TEST_DIR - os.path object containing loation of TEST_NAME test image
    TEST_NAME - test image number
    ---
    returns:
    none
    """
    iterations = 16 ## predetermined
    
    det_indices = np.argwhere(detections / iterations > 0.75)
    det_coords = np.empty((det_indices.shape[0], 2))
    
    coords = raster_to_planeto(TEST_DIR, TEST_NAME)
    
    for i in range(det_indices.shape[0]):
        det_coords[i, :] = coords[det_indices[i, 0], det_indices[i, 1]]
        
    np.save('LOGS/' + str(TEST_NAME), det_coords)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (18, 9));
    
    img_loc = os.path.join(TEST_DIR, str(TEST_NAME) + '-Stitched.png')
    img = contrast_resize(img_loc, (2048, 2048))
    img = img[:, :, :3]
    
    ax1.imshow(img.astype(int).clip(0, 255));
    ax1.title.set_text('Original Test Image');
    ax1.axis('off');
    
    threshold = np.where(detections / iterations > 0.5, 1, 0)
    
    ax2.imshow(threshold.astype(float).clip(0, 1))
    ax2.title.set_text('Threshold Decisions');
    ax2.axis('off');
    
    superimposed = np.empty(img.shape)
    for i in range(3):
        superimposed[:, :, i] = threshold * img[:, :, i]
    superimposed *= 255.0 / superimposed.max()
        
    ax3.imshow(superimposed.astype(int).clip(0, 255));
    ax3.title.set_text('Superimposed Decisions');
    ax3.axis('off');
    
    fig.savefig('FIGURES/threshold_maps_' + str(TEST_NAME) + '.png')


def retrain(data, tags, new_imgs):
    """
    retrain the svm and pca models using new data
    ---
    parameters: 
    new_imgs - nd-array containing new num_imgsx2048x2048x3 cv2 images to add to training set
    ---
    returns:
    pca - updated trained principal component analysis dimsionality reduction transform
    svm - updated trained support vector machine
    """
    num_imgs = new_images.shape[0]
    retrain_blocks = np.empty((num_imgs * 64, VEC_SIZE))
    
    for img in range(num_imgs):
        
    
    curr = TRAIN_SIZE + 1
    for i in range(retrain_blocks.shape[0]):
        plt.imsave(str(curr) + '.png', retrain_blocks[i, :].reshape((VEC_DIM)))
        curr += 1