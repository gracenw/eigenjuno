import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import json, cv2, warnings, os, threading, time, multiprocessing
from os.path import isfile
import sys
import spiceypy as spice
from utils import *
from framelets import *
from jitter import *

# stitch function for thread, taken largely from the reddit post linked in repo desc
def stitch(meta_num):
    im_info = loc + '/' + meta_num + '-Metadata.json'
    with open(im_info, 'r') as json_file:
        im_info_dir = json.load(json_file)
        image = loc + '/' + im_info_dir['FILE_NAME']
    
    img = Image.open(image)
    im_ar = np.array(img)
    s1, s2 = im_ar.shape

    start_time = im_info_dir["START_TIME"]
    frame_delay = float(im_info_dir["INTERFRAME_DELAY"].split()[0])+0.001
    
    start_correction, frame_delay = correct_image_start_time_and_frame_delay(im_ar, start_time, frame_delay)
    
    framelets = generate_framelets(im_ar, start_time, start_correction, frame_delay)

    cam_pos, cam_orient = get_junocam_jupiter_rel_pos_orient(start_time, start_correction + 17 * frame_delay)

    y, x = np.mgrid[-1024:1024,-1024:1024]
    x += 300
    y += 50
    rays = np.concatenate([x[...,None], y[...,None], np.ones(((2048),(2048),1))*fl[0]], axis=-1)
    rays = rays.dot(cam_orient)

    surface_raster, _ = project_onto_jupiter_surf(cam_pos, rays)

    colors = np.zeros(((2048),(2048),3))
    color_counts = np.zeros(((2048),(2048),3))

    for k,framelet in enumerate(framelets):
        print(str(meta_num) + ': processing framelet {} of {}..'.format(k+1, len(framelets)))
        col = framelet.color
        brightnesses, valid_map = framelet.get_pixel_val_at_surf_point(surface_raster) # this is where it slows down
        colors[...,2-col] += brightnesses
        color_counts[...,2-col] += valid_map

    colors /= np.maximum(color_counts, 1)
    colors *= 255 / np.max(colors)

    colors = colors.astype(np.uint8)

    new_img = Image.fromarray(colors)
    new_img.save("out_" + str(meta_num) + ".png")

#begin multithreaded execution - this will process all images in a directory, assuming they are from the same perijove
def main():
    print('begin execution ...|')
    
    folder = sys.argv[1]
    loc = '/home/uzumochi/eigenjuno/data/raw/' + folder
    path = '/home/uzumochi/eigenjuno/stitching/kernels/'
    KERNEL_LIST = [ path+"naif0012.tls",
                    path+"de430.bsp",
                    path+"juno_v12.tf",
                    path+"jup310.bsp",
                    path+"jno_sclkscet_00094.tsc",
                    path+folder+".bsp",
                    path+"pck00010.tpc",
                    path+folder+".bc" ]
    spice.furnsh(KERNEL_LIST)
    
    img_nums = []
    for files in os.listdir(loc):
        if files.endswith('.json') and not isfile('out_' + files[0 : 4] + '.png'):
            img_nums.append(files[0 : 4])
    print_str = 'Processing the following images: '
    for im in img_nums:
        print_str += (im + ', ')
    print_str = print_str[:-2]
    print(print_str)

    num_cpus = multiprocessing.cpu_count()
    curr = 0

    while curr < len(img_nums):
        threads = []
        num_threads = num_cpus if num_cpus < len(img_nums) else len(img_nums)
        for t in range(num_threads):
            threads.append(threading.Thread(target = stitch, args = (str(img_nums[curr]), )))
            threads[t].start()
            curr += 1
        for t in threads:
            t.join()

    print("done! :D")