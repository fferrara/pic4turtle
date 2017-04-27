"""
Use: python extract_bbox.py dir_from dir_to bboxs_file

The script extract the bounding box of each image in dir_from and copy the
resulting image in dir_to, preserving the name.
bboxs_file must be a CSV file separated by whitespaces. It should contain a line for each file present in dir_from.

bboxs_file syntax:
/path/to/filename x0 y0 x1 y1
"""
import cv2
import sys
import os

def get_rectangles_by_expanding(rectangle, img_width, img_height):
    # rectangle = (x0, y0, x1, y2)
    # expanding left = (0, y0, x0, y2)
    ret1 = (0, rectangle[1], rectangle[0], rectangle[3])
    # expanding right = (x1, y0, img_width, y2)
    ret2 = (rectangle[2], rectangle[1], img_width, rectangle[3])
    # expanding top = (x0, 0, x1, y0)
    ret3 = (rectangle[0], 0, rectangle[2], rectangle[1])
    # expanding bottom = (x0, y1, x1, img_height
    ret4 = (rectangle[0], rectangle[3], rectangle[2], img_height)

    return [ret1, ret2, ret3, ret4]

def generate_negative_examples(dir_from, dir_to, bboxs_file):
    with open(bboxs_file) as f:
        lines = [l.strip().split(' ') for l in f.readlines()]
        bboxs = {os.path.basename(l[0]) : l[1:] for l in lines}


    for f in os.listdir(dir_from):
        if any(f.upper().endswith(ext) for ext in ['.JPG', '.JPEG', '.PNG']):
            img = cv2.imread(os.path.join(dir_from, f))
            bbox = [int(c) for c in bboxs[f]]

            negatives = get_rectangles_by_expanding(bbox, img.shape[1], img.shape[0])
            cuts = 0
            for n in negatives:
                # n is rectangle (x0, y0, x1, y1)
                if n[2] - n[0] > 100 and n[3] - n[1] > 100:
                    cut = img[n[1]:n[3], n[0]:n[2]]
                    new_img = os.path.join(os.path.abspath(dir_to), '%s_cut_%d.jpg'%(os.path.basename(f), cuts))
                    cuts += 1
                    cv2.imwrite(new_img, cut)



def extract(dir_from, dir_to, bboxs_file, square=False):
    with open(bboxs_file) as f:
        lines = [l.strip().split(' ') for l in f.readlines()]
        bboxs = {os.path.basename(l[0]) : l[1:] for l in lines}

    for f in os.listdir(dir_from):
        if any(f.upper().endswith(ext) for ext in ['.JPG', '.JPEG', '.PNG']):
            img = cv2.imread(os.path.join(dir_from, f))
            x0, y0, x1, y1 = [int(c) for c in bboxs[f]]

            if square: # square the bbox
                image_height, image_width = img.shape[:2]
                width = x1 - x0
                height = y1 - y0

                if width > height:
                    # adjust y
                    pad = (width - height) / 2
                    y0 -= pad
                    y1 += pad

                    y0 = 0 if y0 < 0 else y0
                    y1 = image_height if y1 > image_height else y1
                elif height > width:
                    # adjust x
                    pad = (height - width) / 2
                    x0 -= pad
                    x1 += pad

                    x0 = 0 if x0 < 0 else x0
                    x1 = image_width if x1 > image_width else x1

            cut = img[y0:y1, x0:x1]

            new_img = os.path.join(os.path.abspath(dir_to), f)
            cv2.imwrite(new_img, cut)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'NO'

    dir_from = sys.argv[1]
    dir_to = sys.argv[2]
    bboxs_file = sys.argv[3]

    extract(dir_from, dir_to, bboxs_file, False)



