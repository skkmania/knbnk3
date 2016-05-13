# -*- coding: utf-8 -*-
import json
import os
import os.path
import time
import numpy as np
import itertools
import sys
import cv2

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
        char_width = round(float(total_width)/char_len)
        for i in range(0, int(char_len)):
            char_areas.append((x1 + i*char_width, y1, x2 + (i+1)*char_width, y2))
    else:
        char_height = round(float(total_height) / char_len)
        for i in range(0, int(char_len)):
            char_areas.append((x1, y1 + i * char_height, x2, y1 + (i + 1) * char_height))

    ret_list = []
    for i in range(0, int(char_len)):
        ret_list.append([])

    for box in adjs:
        centroid = (box[0] + box[2]/2, box[1] + box[3]/2)
        i = 0
        while i < char_len:
            if is_in_area(centroid, char_areas[i]):
                ret_list[i].append(box)
                break
            i = i + 1

    return ret_list


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


def sweep_boxes_in_page_margin(self, mgn=None):
    """
    pageの余白に存在すると思われるboxは排除
    box: [x, y, w, h]
    """

    if mgn:
        self.pgmgn_x, self.pgmgn_y = mgn
    else:
        self.pgmgn_x, self.pgmgn_y = self.parameters['page']['pgmgn']

    left_mgn = self.width * self.pgmgn_x
    right_mgn = self.width * (1 - self.pgmgn_x)
    upper_mgn = self.height * self.pgmgn_y
    lower_mgn = self.height * (1 - self.pgmgn_y)

    self.boxes = [x for x in self.boxes
                  if not self.in_margin(x, left_mgn,
                                        right_mgn, upper_mgn, lower_mgn)]
