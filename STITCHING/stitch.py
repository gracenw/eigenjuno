## code taken from this repo - https://github.com/cosmas-heiss/JunoCamRawImageProcessing - modifiable under gnu license
## all credit to the original code author, Cosmas Heiss

import numpy as np
import spiceypy as spice
from scipy.interpolate import RectBivariateSpline
import csv, json, cv2, os, fnmatch
from PIL import Image
from scipy.spatial.transform import Rotation as R

np.seterr(divide='ignore', invalid='ignore')

print(spice.tkvrsn("TOOLKIT"))

KERNEL_DIR = os.path.expanduser("~/eigenjuno/STITCHING/KERNELS/")

KERNELS = []
for r, d, f in os.walk(KERNEL_DIR):
    for file in f:
        if not fnmatch.fnmatch(file, '*Zone.Identifier'):
            KERNELS.append(file)

spice.kclear()
for k in KERNELS:
    spice.furnsh(os.path.join(KERNEL_DIR, k))

JUPITER_EQUATORIAL_RADIUS = 71492  # km
JUPITER_POLAR_RADIUS = 66854  # km
STRIPE_LENGTH = 1648  # px
STRIPE_HEIGHT = 128  # px
STANDARD_IMAGE_START_TIME_OFFSET = 0.06188  # secs

PHOTOACTIVE_PIXELS = np.ones((STRIPE_LENGTH, STRIPE_HEIGHT), dtype=np.int)
PHOTOACTIVE_PIXELS[:23+1,:] = 0
PHOTOACTIVE_PIXELS[1631-1:,:] = 0

MIN_SURFACE_ANGLE = 10*np.pi/180
MIN_SUN_ANGLE = 5*np.pi/180

BLUE_JUPITER_THRESHOLD = 35
GREEN_JUPITER_THRESHOLD = 75
RED_JUPITER_THRESHOLD = 75

# color channel codes
JUNO_JUNOCAM = -61500
JUNO_JUNOCAM_BLUE = -61501
JUNO_JUNOCAM_GREEN = -61502
JUNO_JUNOCAM_RED = -61503

LAPLACIAN_EDGE_DETECTION_THRESHOLD = 0.1

# camera distortion parameters
# INS-6150_DISTORTION_X
cx = [814.21, 814.21, 814.21]
# INS-6150_DISTORTION_Y
cy = [158.48, 3.48, -151.52]
# INS-6150_DISTORTION_K1
k1 = [-5.9624209455667325e-08, -5.9624209455667325e-08, -5.9624209455667325e-08]
# INS-6150_DISTORTION_K2
k2 = [2.7381910042256151e-14, 2.7381910042256151e-14, 2.7381910042256151e-14]
# INS-6150_FOCAL_LENGTH/INS-6150#_PIXEL_SIZE
fl = [10.95637/0.0074, 10.95637/0.0074, 10.95637/0.0074]

def get_junocam_jupiter_rel_pos_orient(time_str, add_seconds=0):
    """
    :param time_str: str (time stamp from image specific json data file)
    :param add_seconds: int or float (seconds to add to time specified by time str for conveniency)
    :return: ndarray(3), position
        ndarray(3,3), orientation matrix
    """
    et = spice.str2et(time_str)+add_seconds
    pos, light_time = spice.spkpos("Juno", [et], 'IAU_JUPITER', 'NONE', 'JUPITER BARYCENTER')
    pos = np.array(pos[0])
    
    orient = spice.pxform("IAU_JUPITER", "JUNO_JUNOCAM", et)
    orient = np.array(orient)
    
    return pos, orient

def get_sun_jupiter_rel_pos(time_str, add_seconds=0):
    et = spice.str2et(time_str) + add_seconds
    pos, light_time = spice.spkpos("SUN", [et], 'IAU_JUPITER', 'NONE', 'JUPITER BARYCENTER')
    return np.array(pos[0])

def undistort(cam_x, cam_y, color):
    # undistort function from juno_junocam_v03.ti changed for using numpy arrays
    cam_x_old = cam_x.copy()
    cam_y_old = cam_y.copy()
    for i in range(5):  # fixed number of iterations for simplicity
        r2 = (cam_x**2+cam_y**2)
        dr = 1+k1[color]*r2+k2[color]*r2*r2
        cam_x = cam_x_old/dr
        cam_y = cam_y_old/dr
    return cam_x, cam_y

def get_lightray_vectors(pixel_x, pixel_y, color):
    """
    :param pixel_x: nd-ndarray of x pixel coords
    :param pixel_y: nd-ndarray of y pixel coords
    :param color: int (0-2) specifying color 0:blue, 1:green, 2:red
    :return: (n+1)d-ndarray of float 3d-lightray vectors in camera reference frame (n+1)-dimension are x,y,z components
    """
    cam_x = pixel_x-cx[color]
    cam_y = pixel_y-cy[color]
    cam_x, cam_y = undistort(cam_x, cam_y, color)
    return np.stack([cam_x, cam_y, np.full_like(cam_x, fl[color])], axis=-1)

def distort(cam, color):
    """
    :param cam: ndarray(..,2)
    :param color: int, color 0:blue 1:green 2:red
    :return: ndarray(..,2), undistortet values
    """
    r2 = np.sum(cam**2, axis=-1)
    dr = 1 + k1[color] * r2 + k2[color] * r2**2
    return cam * dr[...,None]

def get_pixel_coord_from_lightray(ray_dirs, color):
    behind_sensor = (ray_dirs[...,2] <= 0)
    ray_dirs[behind_sensor, 2] = 1
    alpha = ray_dirs[...,2] / fl[color]
    cam = ray_dirs[...,:2] / alpha[...,None]
    cam = distort(cam, color)
    cam[...,0] += cx[color]
    cam[...,1] += cy[color]
    return np.where(behind_sensor[...,None], -1, cam)

x, y = np.indices((STRIPE_LENGTH, STRIPE_HEIGHT))
# list of precomputed 3d-lighray vectors in camera reference frame for 3 colors 0:blue, 1:green, 2:red
CAMERA_STRIPE_VECTORS = [get_lightray_vectors(x, y, color).transpose(1, 0, 2) for color in range(3)]

def project_onto_jupiter_surf(pos, direct):
    """
    :param pos: ndarray(..,3), origin of the rays
    :param direct: ndarray(..,3), direction of rays in whatever order
    :return: ndarray(..,3), position of projected points closest to pos
        ndarray(..), mask of if the ray hit the sphere
    """
    b, a = JUPITER_POLAR_RADIUS, JUPITER_EQUATORIAL_RADIUS
    # equations where the line intersects the jupiter surface
    q1 = b**2*direct[..., 0]**2+b**2*direct[..., 1]**2+a**2*direct[..., 2]**2
    q2 = 2*pos[..., 0]*direct[..., 0]*b**2+2*pos[..., 1]*direct[..., 1]*b**2+2*pos[..., 2]*direct[..., 2]*a**2
    q3 = pos[..., 0]**2*b**2+pos[..., 1]**2*b**2+pos[..., 2]**2*a**2-float(a**2*b**2)

    p, q = q2/q1, q3/q1    

    tmp = (0.5*p)**2-q
    mask = tmp >= 0

    s = -p*0.5-np.sqrt(tmp*mask)

    return (pos+s[..., None]*direct)*mask[..., None], mask

def rotate_around_axis(n, v, alpha):
    n /= np.linalg.norm(n)
    return n*n.dot(v) + np.cos(alpha)*np.cross(np.cross(n, v), n) + np.sin(alpha)*np.cross(n, v)

def project_tangential_plane(center, x_extent, y_extent, x_res, y_res, orientation):
    """
    :param center: ndarray(3) of float, center of map on jupiter surface
    :param x_extent: float, extent of the x-axis of the map in km
    :param y_extent: float, extent of the y-axis of the map in km
    :param x_res: int, number of pixels on x-axis
    :param y_res: int, number of pixels on y-axis
    :param orientation: float, rotating orientation of map around center in radiants
    :return: ndarray(x_res, y_res, 3) 3d positions for map pixels
    """
    assert np.sqrt(x_extent**2 + y_extent**2) < JUPITER_POLAR_RADIUS
    a = JUPITER_POLAR_RADIUS / JUPITER_EQUATORIAL_RADIUS
    if center[0] == 0 and center[1] == 0:
        north_tang_vector = np.array([1, 0, 0])
    else:
        xy_distance = np.sqrt(center[0]**2+center[1]**2)
        tmp = -center[2] / (a * xy_distance)
        north_tang_vector = np.array([tmp * center[0], a * center[1], a])
        north_tang_vector /= np.linalg.norm(north_tang_vector)

    east_tang_vector = np.cross(north_tang_vector, center)
    east_tang_vector /= np.linalg.norm(east_tang_vector)

    normal_vector = np.cross(east_tang_vector, north_tang_vector)
    north_tang_vector = rotate_around_axis(normal_vector, north_tang_vector, orientation)
    east_tang_vector = rotate_around_axis(normal_vector, east_tang_vector, orientation)

    x_raster, y_raster = np.meshgrid(np.linspace(-x_extent/2, x_extent/2, x_res),
                                     np.linspace(-y_extent/2, y_extent/2, y_res))

    raster = center[None, None, :] + x_raster[..., None] * north_tang_vector[None, None, :]\
             + y_raster[..., None] * east_tang_vector[None, None, :]

    points_on_jup, _ = project_onto_jupiter_surf(raster, -normal_vector[None, None, :])
    
    return points_on_jup

## FRAMELETS
class Framelet:
    def __init__(self, cam_pos, cam_orient, sun_pos, pixel_val, color):
        # pixel_val is expected to be squared compared to the raw_image
        self.color = color
        self.img = pixel_val.T
        self.cam_pos = cam_pos
        self.cam_orient = cam_orient
        self.inv_cam_orient = np.linalg.inv(cam_orient)
        self.sun_pos = sun_pos

        self.interp_function = RectBivariateSpline(np.arange(STRIPE_LENGTH), np.arange(STRIPE_HEIGHT), self.img)

    @staticmethod
    def get_photoactive_pixels(pixel_coords):
        int_px_coords = np.rint(pixel_coords).astype(np.int)
        outside_stripe_mask = (int_px_coords[...,0] < STRIPE_LENGTH) & (int_px_coords[...,0] >= 0)
        outside_stripe_mask &= (int_px_coords[...,1] < STRIPE_HEIGHT) & (int_px_coords[...,1] >= 0)
        int_px_coords *= outside_stripe_mask[...,None]
        return PHOTOACTIVE_PIXELS[int_px_coords[...,0], int_px_coords[...,1]] * outside_stripe_mask

    def get_valid_angle_pixels(self, surf_pos):
        a = JUPITER_POLAR_RADIUS / JUPITER_EQUATORIAL_RADIUS
        surf_normal = surf_pos.copy()
        surf_normal[...,0:2] *= a
        surf_normal[...,2] /= a

        rays = surf_pos - self.cam_pos

        scalar_prod = np.sum(rays*surf_normal, axis=-1)
        cos_angle = -scalar_prod/(np.linalg.norm(rays, axis=-1)*np.linalg.norm(surf_normal, axis=-1))

        return cos_angle > np.cos(np.pi/2-MIN_SURFACE_ANGLE)

    def get_sun_correction(self, surf_pos):
        a = JUPITER_POLAR_RADIUS / JUPITER_EQUATORIAL_RADIUS
        surf_normal = surf_pos.copy()
        surf_normal[..., 0:2] *= a
        surf_normal[..., 2] /= a

        sun_rays = surf_pos - self.sun_pos

        scalar_prod = np.sum(sun_rays * surf_normal, axis=-1)
        cos_angle = -scalar_prod / (np.linalg.norm(sun_rays, axis=-1) * np.linalg.norm(surf_normal, axis=-1))
        mask = cos_angle > np.cos(np.pi/2-MIN_SUN_ANGLE)

        return 1 / np.where(mask, cos_angle, 1), mask

    def get_pixel_val_at_surf_point(self, pos, sun_brightness_correction=True):
        """
        :param pos: ndarray(..,3), positions of jupiter surface points in 3d space
        :return: ndarray(..) of float, brightness values for these points, val is set to 0 if not part of usable sensor
                 ndarray(..) of bool, mask of which surface positions are actually recorded by the sensor
        """
        rays = pos-self.cam_pos
        rays = rays.dot(self.inv_cam_orient)
        pixel_coords = get_pixel_coord_from_lightray(rays, self.color)
        valid = self.get_photoactive_pixels(pixel_coords)
        valid *= self.get_valid_angle_pixels(pos)

        if sun_brightness_correction:
            sun_correction, sun_angle_mask = self.get_sun_correction(pos)
            valid *= sun_angle_mask
        else:
            sun_correction = 1

        in_shape = pos.shape[:-1]
        pixel_coords = pixel_coords.reshape(-1,2)
        
        out_val = self.interp_function(pixel_coords[...,0], pixel_coords[...,1], grid=False)
        out_val = np.maximum(out_val, 0) # need to do that because the spline interpol sometimes goes negative

        return out_val.reshape(in_shape) * sun_correction * valid, valid.astype(np.bool)

    def get_max_pixel_distances_of_surf_points(self, pos):
        """
        :param pos:  ndarray(n,m,3), positions of jupiter surface points in 3d space
        :return: float, maximum point distance in pixel space
        """
        rays = pos - self.cam_pos
        rays = rays.dot(self.inv_cam_orient)
        pixel_coords = get_pixel_coord_from_lightray(rays, self.color)
        valid = self.get_photoactive_pixels(pixel_coords)
        valid *= self.get_valid_angle_pixels(pos)
        _, sun_angle_mask = self.get_sun_correction(pos)
        valid *= sun_angle_mask

        dist_x = np.linalg.norm(pixel_coords[1:,:,:]-pixel_coords[:-1,:,:], axis=-1)
        dist_x = np.maximum(np.pad(dist_x, ((1, 0), (0, 0)), mode='constant'),
                            np.pad(dist_x, ((0, 1), (0, 0)), mode='constant'))
        dist_y = np.linalg.norm(pixel_coords[:,1:,:]-pixel_coords[:,:-1,:], axis=-1)
        dist_y = np.maximum(np.pad(dist_y, ((0, 0), (1, 0)), mode='constant'),
                            np.pad(dist_y, ((0, 0), (0, 1)), mode='constant'))

        return np.maximum(dist_x, dist_y), valid

def generate_framelets(image_array, start_time, start_correction, frame_delay):
    """
    :param image_array: ndarray(k*384,1648), raw image array
    :param start_time: float, start of imaging in et-J2000 time
    :param start_correction: float, correction of start of image time
    :param frame_delay: float, seconds between framelets
    :return:
    """
    framelets = []
    s1, s2 = image_array.shape
    for k in range(s1 // 384):
        stripe_delay = start_correction + k * frame_delay
        cam_pos, cam_orient = get_junocam_jupiter_rel_pos_orient(start_time, add_seconds=stripe_delay)
        sun_pos = get_sun_jupiter_rel_pos(start_time, add_seconds=stripe_delay)

        for color in range(3):
            stripe = image_array[k*384+color*128:k*384+(color+1)*128]

            new_framelet = Framelet(cam_pos,
                                    cam_orient,
                                    sun_pos,
                                    stripe,
                                    color)

            framelets.append(new_framelet)

    return framelets

## JITTER
def get_raw_image_mask(raw_image_array):
    """
    :param raw_image_array: ndarray(k*384, 1648), array of image data from the raw image
    :return: ndarray of uint8, mask of where in the picture jupiter actually is for alignment
    """
    s1, s2 = raw_image_array.shape
    mask = np.zeros((s1, s2), dtype=np.uint8)
    for k in range(0, s1, 384):
        mask[k:k + 128] = (raw_image_array[k:k + 128] > BLUE_JUPITER_THRESHOLD)
        mask[k + 128:k + 256] = (raw_image_array[k + 128:k + 256] > GREEN_JUPITER_THRESHOLD)
        mask[k + 256:k + 384] = (raw_image_array[k + 256:k + 384] > RED_JUPITER_THRESHOLD)
    return mask

def detect_valid_edge_pixels(stripe):
    """
    :param stripe: ndarray(128, 1648), stripe mask
    :return: ndarray(1648) of int, edge indices
             ndarray(1648) of int, mask of which pixels can be used for edge alignment (no noise clear edge)
             ndarray(1648) of int, direction of edge, 1 for edge in up-direction, -1 for edge in down-direction
    """
    stripe = stripe.astype(np.int8)
    y_coords = np.arange(128)
    # detecting the edge
    edge = np.argmax(np.abs(stripe[1:, :]-stripe[:-1, :]), axis=0)
    edge_dir = (stripe[1:, :]-stripe[:-1, :])[edge, np.arange(STRIPE_LENGTH)]

    # finding the indices where values above edge and below edge are all the same by first filling up the rest
    # and the using np.all to detect if the not-filled up part is uniform
    valid_up = np.where(y_coords[:, None] <= edge[None, :], stripe, stripe[None, 0, :])
    valid_down = np.where(y_coords[:, None] > edge[None, :], stripe, stripe[None, -1, :])

    # valid are indices where there are different pixels at the top and bottom
    valid = stripe[None, 0, :] != stripe[None, -1, :]
    valid_up = np.all(valid_up == valid_up[None, 0, :], axis=0)
    valid_down = np.all(valid_down == valid_down[None, 0, :], axis=0)

    return edge, valid_up * valid_down * valid, edge_dir

def detect_sharp_edges(edge, image_stripe, color):
    """
    :param edge: ndarray(1648) of int, edge indices as retuned from detect_valid_edge_pixels
    :param image_stripe: ndarray(128, 1648), stripe of the raw image
    :param color: int (0-2), color of stripe introduced for different edge thresholding for the different colors
                            but not needed yet
    :return: ndarray(1648) of bool, mask on x-dimension similar to second return from detect_valid_edge_pixels
    """
    edge_sharpness = cv2.Laplacian(image_stripe, cv2.CV_32F, ksize=7) / (255 * 49)
    edge_sharpness = cv2.blur(np.abs(edge_sharpness), ksize=(7, 7))

    return edge_sharpness[edge, np.arange(STRIPE_LENGTH)] > LAPLACIAN_EDGE_DETECTION_THRESHOLD

def detect_mask_offset(raw_image_mask, jupiter_mask, raw_image):
    """
    :param raw_image_mask: ndarray of uint8, mask retuned by get_jupiter_mask on the image
    :param jupiter_mask: ndarray of uint8, jupiter mask constructed by position and orientation of spaceprobe
    :return: float, start time offset in seconds for correcting jitter
    """
    s1, s2 = raw_image_mask.shape
    assert jupiter_mask.shape == (s1, s2)

    offset_sum = 0
    offset_count = 0
    for color, k in enumerate(range(0, s1, 128)):
        raw_image_mask_stripe = raw_image_mask[k:k + 128]
        jupiter_mask_stripe = jupiter_mask[k:k + 128]

        jupiter_mask_edge, jupiter_mask_valid, jupiter_mask_egde_dir = detect_valid_edge_pixels(jupiter_mask_stripe)
        raw_image_mask_edge, raw_image_mask_valid, raw_image_egde_dir = detect_valid_edge_pixels(raw_image_mask_stripe)
        sharp_valid = detect_sharp_edges(raw_image_mask_edge, raw_image[k:k + 128], color % 3)

        valid = jupiter_mask_valid * raw_image_mask_valid * (jupiter_mask_egde_dir == raw_image_egde_dir) * sharp_valid
        offset_sum += np.sum((jupiter_mask_edge-raw_image_mask_edge)*valid)
        offset_count += np.sum(valid)

    return offset_sum, offset_count

def correct_image_start_time_and_frame_delay(raw_image_array, raw_image_time_stamp, frame_delay):
    time_offset = STANDARD_IMAGE_START_TIME_OFFSET

    s1, s2 = raw_image_array.shape
    raw_image_mask = get_raw_image_mask(raw_image_array)
    direction_array = np.concatenate(CAMERA_STRIPE_VECTORS, axis=0)
    
    for i in range(2):
        pixel_offset = 0
        pixel_offset_count = 0
        frame_wise_offsets = []

        for k in range(s1//384):
            cam_pos, cam_orient = get_junocam_jupiter_rel_pos_orient(raw_image_time_stamp,
                                                                     add_seconds=time_offset + k * frame_delay)
            direction_array_new = direction_array.dot(cam_orient)
            _, jupiter_mask = project_onto_jupiter_surf(cam_pos, direction_array_new)
            offset_sum, offset_count = detect_mask_offset(raw_image_mask[k * 384:(k+1) * 384],
                                                          jupiter_mask,
                                                          raw_image_array[k * 384:(k+1) * 384])
            pixel_offset += offset_sum / max(offset_count, 1)
            pixel_offset_count += (offset_count > 0)
            if offset_count != 0:
                frame_wise_offsets.append((k, offset_sum/offset_count))
        # 2rpm = 12deg/sec, approx 5deg per stripe and 128px per stripe
        time_offset += 5/(12*128)*pixel_offset/pixel_offset_count

        # frame delay is changed by mean change in offset per stripe
        tmp = [5/(12*128)*(y[1]-x[1])/(y[0]-x[0]) for x, y in zip(frame_wise_offsets[:-1], frame_wise_offsets[1:])]
        frame_delay += sum(tmp) / len(tmp)
    return time_offset, frame_delay

def correct_image_start_time(raw_image_array, raw_image_time_stamp, frame_delay):
    time_offset = STANDARD_IMAGE_START_TIME_OFFSET

    s1, s2 = raw_image_array.shape
    raw_image_mask = get_raw_image_mask(raw_image_array)
    direction_array = np.concatenate(CAMERA_STRIPE_VECTORS, axis=0)

    for i in range(2):
        pixel_offset = 0
        pixel_offset_count = 0

        for k in range(s1//384):
            cam_pos, cam_orient = get_junocam_jupiter_rel_pos_orient(raw_image_time_stamp,
                                                                     add_seconds=time_offset + k * frame_delay)
            direction_array_new = direction_array.dot(cam_orient)
            _, jupiter_mask = project_onto_jupiter_surf(cam_pos, direction_array_new)
            offset_sum, offset_count = detect_mask_offset(raw_image_mask[k * 384:(k+1) * 384],
                                                          jupiter_mask,
                                                          raw_image_array[k * 384:(k+1) * 384])
            pixel_offset += offset_sum / max(offset_count, 1)
            pixel_offset_count += (offset_count > 0)
        # 2rpm = 12deg/sec, approx 5deg per stripe and 128px per stripe
        time_offset += 5/(12*128)*pixel_offset/pixel_offset_count
    return time_offset