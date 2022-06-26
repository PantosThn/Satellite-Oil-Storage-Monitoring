#from skimage import data
from skimage import filters
#from skimage import exposure
from skimage import measure
from skimage import segmentation
from skimage import morphology
from skimage import color
import cv2
from fastai.vision import *

import PIL
import numpy as np
import pathlib
import os

from fastai.vision import *
import torchvision.transforms as T
import base64

def crop(box, image, factor_x=0.5, factor_y=0.6):
    y_min, x_min, y_max, x_max = (box[0], box[1], box[2], box[3])
    # print(self.gt_coords)

    # scale for tank cropping
    margin_x = int((x_max-x_min)*factor_x)
    margin_y = int((y_max-y_min)*factor_y)
    # print(margin_x, margin_y)

    # y_min, y_max, x_min, x_max values for cropping
    c_y_min = max(y_min - margin_y, 0)
    c_y_max = max(y_max + int(margin_y//2), 0)
    c_x_min = max(x_min - margin_x, 0)
    c_x_max = max(x_max + margin_x, 0)

    # actual margins, given that the calculated margin might extend beyond the image
    margin_y_true = y_min - c_y_min
    margin_x_true = x_min - c_x_min

    # coordinates of the actual bounding box relative to the crop box
    c_bbox_relative = [margin_y_true, margin_x_true, (y_max-y_min)+margin_y_true, (x_max-x_min)+margin_x_true]
    # print(self.bbox_relative)

    # crop section of the image
    c_tank_crop = image.data[:, c_y_min:c_y_max, c_x_min:c_x_max].permute(1,2,0).numpy()
    return c_tank_crop, c_bbox_relative

class Tank():
    def __init__(self, box, image, factor_x=0.5, factor_y=0.6):

        self.image = image
        self.gt_coords = (box[0], box[1], box[2], box[3]) # bounding box coordinates
        y_min, x_min, y_max, x_max = self.gt_coords

        # scale for tank cropping
        margin_x = int((x_max-x_min)*factor_x)
        margin_y = int((y_max-y_min)*factor_y)

        # y_min, y_max, x_min, x_max values for cropping
        self.y_min = max(y_min - margin_y, 0)
        self.y_max = max(y_max + int(margin_y//2), 0)
        self.x_min = max(x_min - margin_x, 0)
        self.x_max = max(x_max + margin_x, 0)

        # actual margins, given that the calculated margin might extend beyond the image
        margin_y_true = y_min - self.y_min
        margin_x_true = x_min - self.x_min

        # coordinates of the actual bounding box relative to the crop box
        self.bbox_relative = [margin_y_true, margin_x_true, (y_max-y_min)+margin_y_true, (x_max-x_min)+margin_x_true]

        # crop section of the image
        self.tank_crop = self.image.data[:, self.y_min:self.y_max, self.x_min:self.x_max].permute(1,2,0).numpy()

        self.proc_tank()
        self.get_regions()

    def proc_tank(self):
        # HSV conversion
        hsv = color.rgb2hsv(self.tank_crop)
        H = hsv[:,:,0]
        S = hsv[:,:,1]
        V = hsv[:,:,2]

        # LAB conversion
        lab = color.rgb2lab(self.tank_crop)
        l1 = lab[:,:,0]
        l2 = lab[:,:,1]
        l3 = lab[:,:,2]

        # Enhanced image
        self.tank_hsv = -(l1+l3)/(V+1)

        # Threshold values
        t1 = filters.threshold_minimum(self.tank_hsv)
        t2 = filters.threshold_mean(self.tank_hsv)

        # Thresholding
        self.tank_thresh = self.tank_hsv > (0.5*t1 + 0.4*t2)

        # Processed, labeled image
        self.label_image = measure.label(morphology.area_closing(morphology.closing(
            segmentation.clear_border(filters.hessian(self.tank_thresh)))))

    def get_regions(self):
        # Regions within image
        self.regions_all = measure.regionprops(self.label_image)

        self.regions = []

        # Some regions are noise. This ensures that regions have a decent area ( > 25 px),
        # that the region intersects the boudning box around the tank (removes lots of noisy features)
        # and that the processed region is also present in the thresholded image (the hessian filter can sometimes
        # add artifacts that need to be removed this day)
        for region in self.regions_all:
            if intersection(self.bbox_relative, region.bbox) > 300:
                if region.area > 25:
                    b = region.bbox
                    if abs(self.tank_thresh[b[0]:b[2], b[1]:b[3]].mean() - region.image.mean()) < 0.06:
                        self.regions.append(region)

        # areas of all regions
        areas = np.array([i.area for i in self.regions])

        # if there are more than two areas found, take the two largest
        # 1 - ratio of the two largest areas calculates the volume estimation
        if len(areas) > 1:
            idx2, idx1 = areas.argsort()[-2:]
            self.volume = 1 - self.regions[idx2].area / self.regions[idx1].area
        # if only 1 area is found, tank is assumed to be full
        else:
            idx2 = 0
            idx1 = 0
            self.volume = 1

        # Blank image onto which to paste only the two shadow regions
        self.blank = np.zeros(self.tank_crop.shape[:2])

        for region in [self.regions[idx1], self.regions[idx2]]:
            y_min, x_min, y_max, x_max = region.bbox
            self.blank[y_min:y_max, x_min:x_max] += region.image.astype('uint8')

        # get contours of shadows
        self.contours = measure.find_contours(self.blank, 0.5)
        if len(self.contours) > 1:
            # If there are multiple contours, take the two longest
            contour_idxs = np.array([len(i) for i in self.contours]).argsort()[-2:]
        else:
            contour_idxs = [0]
        self.contours_select = [self.contours[i] for i in contour_idxs]

def intersection(bb1, bb2):
    """
    intersection calculates the pixel area intersection between two bounding boxes
    """
    y_min1, x_min1, y_max1, x_max1 = bb1
    y_min2, x_min2, y_max2, x_max2 = bb2

    x_left = max(x_min1, x_min2)
    x_right = min(x_max1, x_max2)
    y_top = max(y_min1, y_min2)
    y_bottom = min(y_max1, y_max2)

    intersection = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top+1)
    return intersection

def check_bb(bbox, shape):
    """
    The algorithm is designed to work with tanks that are fully in frame.
    Bounding boxes that reach the edge of an image (indicating the tank
    extends beyond the image) are excluded from processing.
    """
    c, h, w = shape
    ymin, xmin, ymax, xmax = bbox
    if(xmin<=2 or xmin>=w-2):
        return False
    if(xmax<=2 or xmax>=w-2):
        return False
    if(ymin<=2 or ymin>=h-2):
        return False
    if(ymax<=2 or ymax>=h-2):
        return False
    return True

class MultiTank():
    def __init__(self, bbs, image):
        self.image = image
        # check bounding boxes aren't at the edge of the image
        self.bbs = [i for i in bbs if check_bb(i, image.shape)]
        self.tanks = []
        for i in self.bbs:
            try:
                self.tanks.append(Tank(i, image))
            except:
                print('Error')
                pass

        self.create_masks()
    def get_volumes(self, figsize=(12,12), ax=None):
        coords = [i.gt_coords for i in self.tanks]
        classes = list(range(len(self.tanks)))
        labels = ['{:.3f}'.format(i.volume) for i in self.tanks]
        bbox_vol = ImageBBox.create(*self.image.size, coords, classes, classes=labels)
        return labels

    def plot_contours(self, figsize=(12,12)):
        fig, ax = plt.subplots(figsize=figsize)
        show_image(self.image, ax=ax)

        colors = np.linspace(0, 1, len(self.tanks))

        for i, tank in enumerate(self.tanks):
            for contour in tank.contours_select:
                ax.plot(contour[:,1]+tank.x_min, contour[:,0]+tank.y_min, color=plt.cm.rainbow(colors[i]))
    def create_masks(self):
        mask = np.zeros(self.image.shape[1:])
        colors = np.linspace(0, 1, len(self.tanks))

        for i, tank in enumerate(self.tanks):
            tank_blank = (tank.blank > 0) * (i + 1)
            mask[tank.y_min:tank.y_max, tank.x_min:tank.x_max] += tank_blank

        self.mask = mask
        self.mask_binary = mask > 0

def draw_outputs(img, outputs, class_names):
    convert_name = {
        'Tank': 'T',
        'Tank Cluster': 'TC',
        'Floating Head Tank': 'FHT'
    }

    boxes, scores, volumes, classes, nums = outputs
    volumes = [float(i) for i in volumes]
    scores = np.array(scores)
    boxes, scores, classes, nums = boxes[0], scores[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        color = [0,0,0]
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        color[int(classes[i])]=255
        img = cv2.rectangle(img, x1y1, x2y2, tuple(color), 2)
        img = cv2.putText(img, '{} {:.2f} V {:.2f}'.format(
            convert_name[class_names[int(classes[i])]],
            scores[i],
            volumes[i]),
            x1y1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2)
    return img
