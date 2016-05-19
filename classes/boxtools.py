# -*- coding: utf-8 -*-
import json
import os
import os.path
import time
import numpy as np
import itertools
import sys
import cv2


def amend_adjs(adjs, char_len):
    """
    recheck_adjs の結果を受け取り再度チェックする
    recheck_adjs　の方法には文字数の推定に誤差があり、本当は４文字なのに３文字と判定するといった誤りがよくおこる。
    その場合、結果のリストの各要素の領域に重複が発生する。
    重複があったときにこのamend_adjsにadjsを送って再評価する

    その方法:

    　　
    :param adjs: list of boxes :
    :param char_len: : 期待する文字数（recheck_adjsで失敗した文字数に1を足した数)
    :return: list of list of boxes :
    """
    ret_list = []
    for i in range(0, int(char_len)):
        ret_list.append([])

    for box in adjs:
        centroid = (box[0] + box[2] / 2, box[1] + box[3] / 2)
        i = 0
        while i < char_len:
            if is_in_area(centroid, char_areas[i]):
                ret_list[i].append(box)
                break
            i = i + 1

    return ret_list

def clustering(adjs, char_len):
    """

    :return:
    """
    x1, y1, total_width, total_height = get_boundingBox(adjs)

    if total_width < char_size or total_height < char_size:
        return [adjs]

    x2 = x1 + total_width
    y2 = y1 + total_height
    char_len = round(float(max(total_height, total_width) / min(total_height, total_width)))

    if char_len == 1:
        return [adjs]



    char_areas = []
    if total_width > total_height:
        char_width = round(float(total_width) / char_len)
        for i in range(0, int(char_len)):
            char_areas.append((x1 + i * char_width, y1, x2 + (i + 1) * char_width, y2))
    else:
        char_height = round(float(total_height) / char_len)
        for i in range(0, int(char_len)):
            char_areas.append((x1, y1 + i * char_height, x2, y1 + (i + 1) * char_height))

    ret_list = []
    for i in range(0, int(char_len)):
        ret_list.append([])

    for box in adjs:
        centroid = (box[0] + box[2] / 2, box[1] + box[3] / 2)
        i = 0
        while i < char_len:
            if is_in_area(centroid, char_areas[i]):
                ret_list[i].append(box)
                break
            i = i + 1


def recheck_adjs(adjs, char_size=20):
    """
    :param adjs: list of boxes
    :param char_size: integer: 文字と見なす大きさの下限値
    :return: list of list of boxes
    adjs として渡されたboxのリストが、２文字以上をカバーしてしまっていないかチェックし
    ２文字以上ならば、１文字ずつに分解して、それらをリストにして返す
    方法:
      渡されたリストを包む最も外側の長方形の縦横比から文字数を決めてしまう
       height > width  ならば  round(float(height) / width) を、
       height < width  ならば  round(float(width) / height)　を文字数と見なす
      最も外側の長方形を文字数で均等に分割し、adjsの各boxをそれぞれに配分してリストにする
      各boxの所属は、は各boxの重心の所属により決める
      文字数が１と判定されたときは、単にadjsをリストにいれたもの[adjs]を返す
    char_size が必要な理由:
      あまり小さいboxのリストを処理しても時間の無駄なので、まず渡されたリストの外接方形のサイズをチェックし
      char_sizeに満たなければ、渡されたリストをリストにくるんでそのまま返して終了する
    """

    try:
        ret_list = clustering(adjs, char_len)
        # if intersect_exists(convert_to_bounding(ret_list)):
        #    char_len = char_len + 1
        #    ret_list = clustering(adjs, char_len)
    except:
        ret_list = [adjs]

    return ret_list

def recheck_adjs_new(adjs, char_size=20):
    """
    :param adjs: list of boxes
    :param char_size: integer: 文字と見なす大きさの下限値
    :return: list of list of boxes
    adjs として渡されたboxのリストが、２文字以上をカバーしてしまっていないかチェックし
    ２文字以上ならば、１文字ずつに分解して、それらをリストにして返す
    方法:
      渡されたリストを包む最も外側の長方形の縦横比から文字数を決めてしまう
       height > width  ならば  round(float(height) / width) を、
       height < width  ならば  round(float(width) / height)　を文字数と見なす
      最も外側の長方形を文字数で均等に分割し、adjsの各boxをそれぞれに配分してリストにする
      各boxの所属は、は各boxの重心の所属により決める
      文字数が１と判定されたときは、単にadjsをリストにいれたもの[adjs]を返す
    char_size が必要な理由:
      あまり小さいboxのリストを処理しても時間の無駄なので、まず渡されたリストの外接方形のサイズをチェックし
      char_sizeに満たなければ、渡されたリストをリストにくるんでそのまま返して終了する
    """
    x1, y1, total_width, total_height = get_boundingBox(adjs)

    if total_width < char_size or total_height < char_size:
        return [adjs]

    x2 = x1 + total_width
    y2 = y1 + total_height
    char_len = round(float(max(total_height, total_width) / min(total_height, total_width)))

    if char_len == 1:
        return [adjs]

    direction = 0

    try:
        ret_list = clustering2(adjs, char_len, direction)
        if intersect_exists(convert_to_bounding(ret_list)):
            char_len = char_len + 1
            ret_list = clustering2(adjs, char_len, direction)
    except:
        ret_list = [adjs]

    return ret_list

def clustering2(boxes, k, direction):
    """
    box のリストをうけとり、それらをk個に分類して返す
    分類にはk-means法を使う
    クラスタの重心からの距離は1次元ではかる
      縦書きなら、y 軸方向の距離
      横書きなら、x 軸方向の距離
    :param boxes: list of boxes
    :param k: demanded number for output clusters
    :param direction: 0 or 1 : 0 = 縦書き, 1 = 横書き
    :return: list of list of boxes
    　　この外側のlistの要素数はk となる
    """
    if direction == 0:
        data = [box[1]+box[3]/2 for box in boxes]
    else:
        data = [box[0]+box[2]/2 for box in boxes]
    data = np.float32(np.array(data))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 10, flags)

    ret_list = []
    for i in range(0, k):
        ret_list.append([])
    for i in range(0, len(boxes)):
        ret_list[labels[i]].append(boxes[i])
    return ret_list

def convert_to_bounding(list_of_list_of_boxes):
    """
    list of list of boxes -> list of (bounding boxes)
    :param list_of_list_of_boxes:
    :return:
    """
    ret = []
    for list in list_of_list_of_boxes:
        ret.append(get_boundingBox(list))
    return ret

def intersect_exists(boxes_list):
    """
    box のリストを受け取り、要素間に重なるところがあるか調べる
    :param boxes_list:
    :return: boolean :
       True : 要素間に重なるところが一箇所でも存在するとき
       False: 要素間に重なるところが一箇所も存在しない
    """
    num = len(boxes_list)
    for i in range(0,num - 1):
        for j in range(i+1, num):
            if intersect(boxes_list[i], boxes_list[j], 0, 0):
                return True
    return False

def intersect(box1, box2, x_margin=None, y_margin=None):
    """
    box1 と box2 が交わるか接するならtrueを返す。
    marginを指定することですこし離れていても接すると判定.
    指定しないとdefault値として(20, 8)をmerginとする
    """
    xm, ym = (20, 8)  # default
    if x_margin:
        xm = x_margin
    if y_margin:
        ym = y_margin
    ax1, ay1, w1, h1 = box1
    ax2 = ax1 + w1
    ay2 = ay1 + h1
    bx1, by1, w2, h2 = box2
    bx2 = bx1 + w2
    by2 = by1 + h2

    if h_apart(ax1, ax2, bx1, bx2, xm):
        return False
    elif v_apart(ay1, ay2, by1, by2, ym):
        return False
    else:
        return True


def h_apart(ax1, ax2, bx1, bx2, xm):
    return ax2 < (bx1 - xm) or (bx2 + xm) < ax1


def v_apart(ay1, ay2, by1, by2, ym):
    return ay2 < (by1 - ym) or (by2 + ym) < ay1

def get_box_area(box):
    """

    :param box: tuple of integer : (x, y , w, h)
    :return: integer : area of the box
    """
    return box[2] * box[3]

def is_in_area(point, area):
    """
    return true if point is in area
    :param point: (x, y)
    :param area: (x1, y1, x2, y2)
    :return:
    """
    x, y = point
    x1, y1, x2, y2 = area
    return (x1 < x < x2) and (y1 < y < y2)

def get_boundingBox(boxes):
    """
    複数のboxを包むboxを返す
    :param boxes: list of boxes 各要素（box）の形式:(x,y,w,h)
    :return: a tuple of 4 integers which means a box 出力のboxの形式:(x,y,w,h)
      (x,y,w,h) を座標で表現すると(x,y,x+w,y+h)　となる
    """
    target = [(x, y, x + w, y + h) for (x, y, w, h) in boxes]
    x1, y1, d1, d2 = list(map(min, list(zip(*target))))
    d1, d2, x2, y2 = list(map(max, list(zip(*target))))
    # (x,y,x+w,y+h) -> (x,y,x,y)
    return (x1, y1, x2 - x1, y2 - y1)


def in_margin(box, le, ri, up, lo):
    x1, y1 = box[0:2]
    x2, y2 = map(sum, zip(box[0:2], box[2:4]))
    return (y2 < up) or (y1 > lo) or (x2 < le) or (x1 > ri)


def sweep_boxes_in_page_margin(obj, boxes, mgn=None):
    """
    pageの余白に存在すると思われるboxは排除
    box: [x, y, w, h]
    """

    if mgn:
        pgmgn_x, pgmgn_y = mgn
    else:
        pgmgn_x, pgmgn_y = obj.parameters['page']['pgmgn']

    left_mgn = obj.width * pgmgn_x
    right_mgn = obj.width * (1 - pgmgn_x)
    upper_mgn = obj.height * pgmgn_y
    lower_mgn = obj.height * (1 - pgmgn_y)

    return [x for x in boxes
                if not in_margin(x, left_mgn, right_mgn, upper_mgn, lower_mgn)]
