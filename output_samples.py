# -*- coding: utf-8 -*-

import classes.knutil as ku
import classes.boxtools as bt
import cv2
import numpy as np
import matplotlib.pyplot as plt


def write_contours_and_hierarchy_data_to_textfile(knpage, outdir=None, ext=None):
    if not hasattr(knpage, 'contours'):
        knpage.getContours()
    outfilename = ku.mkFilename(knpage, 'contour_and_hierarchy_data', outdir, ext)
    with open(outfilename, 'w') as f:
        f.write("contours\n")
        for cnt in knpage.contours:
            f.writelines(str(cnt))
            f.write("\n")

        f.write("\n\nhierarchy\n")
        for hic in knpage.hierarchy:
            f.writelines(str(hic))
            f.write("\n")
    return outfilename

def write_boxes_coordinates_data_to_textfile(knpage, outdir=None, ext=None):
    """
    このコマンドが発行された時点でのknpage.boxesの座標データをテキストファイルとして出力する。
    knpage.boxesはknpageが生成されたとき空のリストとして作成され、その後多くの関数により内容が増減する
    :param knpage:
    :param outdir:
    :return:
    """
    outfilename = ku.mkFilename(knpage, '_boxes_coodinates_data', outdir, ext)
    with open(outfilename, 'w') as f:
        f.write("knpage.boxes\n")
        f.write('len(boxes) : %s \n' % str(len(knpage.boxes)))
        for box in knpage.boxes:
            f.write(str(box) + "\n")
        f.write("\n")
    return outfilename


def write_binarized_to_file(knpage, outdir):
    if not hasattr(knpage, 'binarized'):
        knpage.getBinarized()
    outfilename = ku.mkFilename(knpage, '_binarized', outdir)
    knpage.write(outfilename, knpage.binarized)


def write_gradients(knpage, outdir=None):
    knpage.getGradients()
    for n in ['sobel', 'scharr', 'laplacian']:
        if n in knpage.parameters:
            outfilename = ku.mkFilename(knpage, '_' + n, outdir)
            img = getattr(knpage, 'gradients_' + n)
            print(outfilename)
            cv2.imwrite(outfilename, img)


def writeContour(knpage):
    img_of_contours = np.zeros(knpage.img.shape, np.uint8)
    for point in knpage.contours:
        x, y = point[0][0]
        cv2.circle(img_of_contours, (x, y), 1, [0, 0, 255])

def write_contours_bounding_rect_to_file(knpage, outdir=None):
    if not hasattr(knpage, 'contours'):
        knpage.getContours()
    om = np.zeros(knpage.img.shape, np.uint8)
    for cnt in knpage.contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(om, (x, y), (x + w, y + h), [0, 255, 0])
        if (int(w) in range(60, 120)) or (int(h) in range(60, 120)):
            knpage.centroids.append((x + w / 2, y + h / 2))
            cv2.circle(om, (int(x + w / 2),
                            int(y + h / 2)), 5, [0, 255, 0])
    outfilename = ku.mkFilename(knpage, '_cont_rect', outdir)
    knpage.write(outfilename, om)
    return outfilename

def write_original_with_contour_to_file(knpage, outdir=None):
    if not hasattr(knpage, 'contours'):
        knpage.getContours()
    knpage.orig_w_cont = knpage.img.copy()
    for point in knpage.contours:
        x, y = point[0][0]
        cv2.circle(knpage.orig_w_cont, (x, y), 2, [0, 0, 255])
    outfilename = ku.mkFilename(knpage, '_orig_w_cont', outdir)
    knpage.write(outfilename, knpage.orig_w_cont)
    return outfilename

def write_original_with_contour_and_rect_to_file(knpage, outdir=None):
    if not hasattr(knpage, 'contours'):
        knpage.getContours()
    knpage.orig_w_cont_and_rect = knpage.img.copy()
    om = knpage.orig_w_cont_and_rect
    for cnt in knpage.contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(om, (x, y), (x + w, y + h), [0, 255, 0])
        if (int(w) in range(60, 120)) or (int(h) in range(60, 120)):
            knpage.centroids.append((x + w / 2, y + h / 2))
            cv2.circle(om, (int(x + w / 2),
                            int(y + h / 2)), 5, [0, 255, 0])
        cx, cy = cnt[0][0]
        cv2.circle(om, (cx, cy), 2, [0, 0, 255])
    outfilename = ku.mkFilename(knpage, '_orig_w_cont_and_rect', outdir)
    knpage.write(outfilename, knpage.orig_w_cont_and_rect)
    return outfilename

def write_boxes_to_file(knpage, outdir=None, target=None, fix=None):
    """
    output 3 image file of boxes. differnece of these files is "order of boxes"
        1. boxes as generated order
        2. boxes as sorted by x-axis
        3. boxes as sorted by y-axis
    :param knpage:
    :param outdir:
    :param target: a tuple of 2 integers.  (start, end) means the range of boxes you want to output
    :param fix:
    :return: a list of 3 output file names
    """
    if outdir is None:
        outdir = knpage.pagedir
    if target is None:
        s, e = None, None
    else:
        s, e = target
    knpage.getCentroids()
    knpage.sort_boxes()
    boxes = knpage.boxes[s:e]
    x_sorted_boxes = knpage.x_sorted_boxes[s:e]
    y_sorted_boxes = knpage.y_sorted_boxes[s:e]
    outfilenames = []
    for t in [(boxes, '_boxes%s' % fix),
              (x_sorted_boxes, '_x_sorted_boxes%s' % fix),
              (y_sorted_boxes, '_y_sorted_boxes%s' % fix)]:
        om = np.zeros(knpage.img.shape, np.uint8)
        for box in t[0]:
            x, y, w, h = box
            cv2.rectangle(om, (x, y), (x + w, y + h), [0, 255, 0])
        outfilename = ku.mkFilename(knpage, t[1], outdir)
        outfilenames.append(outfilename)
        cv2.imwrite(outfilename, om)
    return outfilenames

def write_collected_boxes_to_file(knpage, outdir=None):
    if not hasattr(knpage, 'collected_boxes'):
        knpage.collect_boxes()

    om = np.zeros(knpage.img.shape, np.uint8)
    for box in knpage.collected_boxes:
        x, y, w, h = box
        cv2.rectangle(om, (x, y), (x + w, y + h), [0, 0, 255])
    outfilename = ku.mkFilename(knpage, '_collected_box', outdir)
    knpage.write(outfilename, om)
    return outfilename

def write_original_with_collected_boxes_to_file(knpage, outdir=None):
    """
    collect_boxes()を３回繰り返して抜けを補おうという、ちと情けない方法によっているため
    テストに時間がかかることに注意
    :param knpage:
    :param outdir:
    :return:
    """
    if not hasattr(knpage, 'collected_boxes'):
        knpage.collect_boxes()
        knpage.boxes = knpage.collected_boxes
        knpage.collect_boxes()
        knpage.boxes = knpage.collected_boxes
        knpage.collect_boxes()

    knpage.orig_w_collected = knpage.img.copy()
    om = knpage.orig_w_collected
    for box in knpage.collected_boxes:
        x, y, w, h = box
        cv2.rectangle(om, (x, y), (x + w, y + h), [0, 0, 255])
    outfilename = ku.mkFilename(knpage, '_orig_w_collected_box', outdir)
    knpage.write(outfilename, om)
    return outfilename

def write_all(knpage, outdir=None):
    write_contours_and_hierarchy_data_to_textfile(knpage, outdir)
    write_binarized_file(knpage, outdir)
    write_contours_bounding_rect_to_file(knpage, outdir)
    write_original_with_contour_file(knpage, outdir)
    write_original_with_contour_and_rect_file(knpage, outdir)
    write_collected_boxes_to_file(knpage, outdir)
    write_original_with_collected_boxes_to_file(knpage, outdir)

def save_collected_boxes_histogram_to_file(knpage, outdir=None):
    """
    横軸 : collected_box の面積
    縦軸 : collected_box の個数
    :return: fullpath of output
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    areas = list(map(bt.get_box_area, knpage.collected_boxes))
    ax.hist(areas, bins=50)
    outfilename = ku.mkFilename(knpage, '_collected_box_area_histogram', outdir)
    fig.savefig(outfilename)
    return outfilename
