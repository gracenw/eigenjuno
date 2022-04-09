import cv2, pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA


VEC_SIZE = 256 * 256 * 3
svm = pickle.load(open('MODELS/svm_model_opt.sav', 'rb'))
pca = pickle.load(open('MODELS/pca_model_opt.sav', 'rb'))


def contrast_resize(img_loc, new_size):
    """Returns a CLAHE contrasted image of original found at input location, resized to new_size parameter."""
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


def break_blocks(img):
    """Function to recursively shift, divide, and classify a 2048x2048 image. 
    Returns a 2048x2048 array containing the sum of the class labels predicted for each pixel at each shift and resolution."""
    full_size = img.shape[0]
    block_size = int(full_size / 2)
    if block_size > 128:
        shift_size = int(block_size / 2)
        detections = np.empty((full_size, full_size))
    
        # rotate slices of image horizontally
        for i in range(4):
            shift_img = np.empty(img.shape)
            shift_img[0:(full_size - (shift_size * i)), :, :] = np.copy(img[(i * shift_size):full_size, :, :])
            shift_img[(full_size - (shift_size * i)):full_size, :, :] = np.copy(img[0:(i * shift_size), :, :])
            
            blocks = np.empty((4, block_size, block_size, 3))
            blocks[0, :, :, :] = shift_img[0 : block_size, 0 : block_size, :]
            blocks[1, :, :, :] = shift_img[0 : block_size, block_size : full_size, :]
            blocks[2, :, :, :] = shift_img[block_size : full_size, 0 : block_size, :]
            blocks[3, :, :, :] = shift_img[block_size : full_size, block_size : full_size, :]

            valid_indices = []
            testing_blocks = np.empty((4, VEC_SIZE))
            for i in range(4):
                block = cv2.resize(blocks[i, :, :], (256, 256), interpolation = cv2.INTER_NEAREST)
                green_locs = np.where((block[:, :, 0] == 0) & (block[:, :, 1] != 0) & (block[:, :, 2] != 0))[0]
                block = np.reshape(block, VEC_SIZE)
                zero_locs = np.where(block < 0.035)[0]
                if (zero_locs.size < block.size / 3) and (green_locs.size < block.size / 10):
                    valid_indices = np.append(valid_indices, i)
                    testing_blocks[i, :] = block

            current_dets = np.zeros(4)

            if len(valid_indices) != 0:
                testing_blocks = testing_blocks[np.array(valid_indices).astype(int), :]
                transform_test = pca.transform(testing_blocks)
                split_pred = svm.predict(transform_test)
                index = 0
                for i in range(4):
                    if i in valid_indices:
                        current_dets[i] = split_pred[index]
                        index += 1

            block_detections = np.zeros((full_size, full_size))

            block_detections[0 : block_size, 0 : block_size] = np.full((block_size, block_size), current_dets[0]) \
                        + break_blocks(blocks[0, :, :, :])
            block_detections[0 : block_size, block_size : full_size] = np.full((block_size, block_size), current_dets[1]) \
                        + break_blocks(blocks[1, :, :, :])
            block_detections[block_size : full_size, 0 : block_size] = np.full((block_size, block_size), current_dets[2]) \
                        + break_blocks(blocks[2, :, :, :])
            block_detections[block_size : full_size, block_size : full_size] = np.full((block_size, block_size), current_dets[3]) \
                        + break_blocks(blocks[3, :, :, :])
            
            unshift_dets = np.empty((full_size, full_size))
            unshift_dets[0:(i * shift_size), :] = np.copy(block_detections[(full_size - (shift_size * i)):full_size, :])
            unshift_dets[(i * shift_size):full_size, :] = np.copy(block_detections[0:(full_size - (shift_size * i)), :])
            
            detections += unshift_dets

        # rotate slices of image vertically
        for i in range(4):        
            shift_img = np.empty(img.shape)
            shift_img[:, 0:(full_size - (shift_size * i)), :] = np.copy(img[:, (i * shift_size):full_size, :])
            shift_img[:, (full_size - (shift_size * i)):full_size, :] = np.copy(img[:, 0:(i * shift_size), :])
            
            blocks = np.empty((4, block_size, block_size, 3))
            blocks[0, :, :, :] = shift_img[0 : block_size, 0 : block_size, :]
            blocks[1, :, :, :] = shift_img[0 : block_size, block_size : full_size, :]
            blocks[2, :, :, :] = shift_img[block_size : full_size, 0 : block_size, :]
            blocks[3, :, :, :] = shift_img[block_size : full_size, block_size : full_size, :]

            valid_indices = []
            testing_blocks = np.empty((4, VEC_SIZE))
            for i in range(4):
                block = cv2.resize(blocks[i, :, :], (256, 256), interpolation = cv2.INTER_NEAREST)
                green_locs = np.where((block[:, :, 0] == 0) & (block[:, :, 1] != 0) & (block[:, :, 2] != 0))[0]
                block = np.reshape(block, VEC_SIZE)
                zero_locs = np.where(block < 0.035)[0]
                if (zero_locs.size < block.size / 3) and (green_locs.size < block.size / 10):
                    valid_indices = np.append(valid_indices, i)
                    testing_blocks[i, :] = block

            current_dets = np.zeros(4)

            if len(valid_indices) != 0:
                testing_blocks = testing_blocks[np.array(valid_indices).astype(int), :]
                transform_test = pca.transform(testing_blocks)
                split_pred = svm.predict(transform_test)
                index = 0
                for i in range(4):
                    if i in valid_indices:
                        current_dets[i] = split_pred[index]
                        index += 1

            block_detections = np.zeros((full_size, full_size))

            block_detections[0 : block_size, 0 : block_size] = np.full((block_size, block_size), current_dets[0]) \
                        + break_blocks(blocks[0, :, :, :])
            block_detections[0 : block_size, block_size : full_size] = np.full((block_size, block_size), current_dets[1]) \
                        + break_blocks(blocks[1, :, :, :])
            block_detections[block_size : full_size, 0 : block_size] = np.full((block_size, block_size), current_dets[2]) \
                        + break_blocks(blocks[2, :, :, :])
            block_detections[block_size : full_size, block_size : full_size] = np.full((block_size, block_size), current_dets[3]) \
                        + break_blocks(blocks[3, :, :, :])
            
            unshift_dets = np.empty((full_size, full_size))
            unshift_dets[:, 0:(i * shift_size)] = np.copy(block_detections[:, (full_size - (shift_size * i)):full_size])
            unshift_dets[:, (i * shift_size):full_size] = np.copy(block_detections[:, 0:(full_size - (shift_size * i))])
            
            detections += unshift_dets
        
        return detections
    else:
        return np.zeros((full_size, full_size))


def classify(img_loc):
    """Performs contrasting and classification of new test image found at input location.
    Returns the detections and results figure."""
    img = contrast_resize(img_loc, (2048, 2048))
    img = img[:, :, :3]
    
    detections = break_blocks(img)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (18, 9));

    ax1.imshow(img);
    ax1.title.set_text('Original Test Image');
    ax1.axis('off');

    superimposed = np.copy(img)
    for i in range(3):
        superimposed[:, :, i] = detections * img[:, :, i]
    ax2.imshow(superimposed);
    ax2.title.set_text('Classifier Decisions\nSuperimposed on Image');
    ax2.axis('off');

#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (18, 18));

#     ax1.imshow(img);
#     ax1.title.set_text('Original Test Image');
#     ax1.axis('off');
    
#     normalize = detections / (np.square(detections)).sum()
#     ax2.imshow(normalize);
#     ax2.title.set_text('Normalized Detections Across\nResolutions and Shifts');
#     ax2.axis('off');
    
#     threshold = np.where(normalize == 0.5, 1, 0)
#     ax3.imshow(threshold);
#     ax3.title.set_text('Decision Threshold == 0.5\nAcross Resolutions and Shifts');
#     ax3.axis('off');

#     superimposed = np.copy(img)
#     for i in range(3):
#         superimposed[:, :, i] = threshold * img[:, :, i]
#     ax4.imshow(superimposed);
#     ax4.title.set_text('Decision Threshold\nSuperimposed on Image');
#     ax4.axis('off');
    
    return detections, fig