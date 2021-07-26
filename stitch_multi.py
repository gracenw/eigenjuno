import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from PIL import Image
import json, cv2, warnings, os, threading, time
from Util import *
from Framelets import *
from JitterCorrection import *

#ubuntu kept giving me warnings and they were annoying
warnings.filterwarnings("ignore")

#image directory
os.chdir('/mnt/c/users/grace/documents/juno_stitching/images')

#stitch function for thread, just copy pasted the original code from example 1
def stitch(im_num):
    image = 'raw' + str(im_num) + '.png'
    im_info = 'meta' + str(im_num) + '.json'
    with open(im_info, 'rb') as json_file:
        im_info_dir = json.load(json_file)

    img = Image.open(image)
    im_ar = np.array(img)
    s1, s2 = im_ar.shape

    #determines the resolution of the image, will be (2*ratio) by (2*ratio)
    ratio = 1024

    start_time = im_info_dir["START_TIME"]
    frame_delay = float(im_info_dir["INTERFRAME_DELAY"].split()[0])+0.001

    start_correction, frame_delay = correct_image_start_time_and_frame_delay(im_ar, start_time, frame_delay)

    framelets = generate_framelets(im_ar, start_time, start_correction, frame_delay)
    #removed color correction - im_ar.astype(np.int) ** 2

    cam_pos, cam_orient = get_junocam_jupiter_rel_pos_orient(start_time, start_correction + 17 * frame_delay)

    y, x = np.mgrid[-ratio:ratio,-ratio:ratio]
    x += 300
    y += 50
    rays = np.concatenate([x[...,None], y[...,None], np.ones(((ratio*2),(ratio*2),1))*fl[0]], axis=-1)
    rays = rays.dot(cam_orient)

    surface_raster, _ = project_onto_jupiter_surf(cam_pos, rays)

    colors = np.zeros(((ratio*2),(ratio*2),3))
    color_counts = np.zeros(((ratio*2),(ratio*2),3))

    for k,framelet in enumerate(framelets):
        print(str(im_num) + ': processing framelet {} of {}..'.format(k+1, len(framelets)))
        col = framelet.color
        brightnesses, valid_map = framelet.get_pixel_val_at_surf_point(surface_raster)
        colors[...,2-col] += brightnesses
        color_counts[...,2-col] += valid_map

    colors /= np.maximum(color_counts, 1)
    colors *= 255 / np.max(colors)

    colors = colors.astype(np.uint8)

    new_img = Image.fromarray(colors)
    new_img.save("out" + str(im_num) + ".png")


#begin threads
t1 = threading.Thread(target=stitch, args=(2,))
#t2 = threading.Thread(target=stitch, args=(2,))
#t3 = threading.Thread(target=stitch, args=(3,))
t1.start()
#t2.start()
#t3.start()
t1.join()
#t2.join()
#t3.join()
print("done! :D")
