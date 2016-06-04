# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
import json
import os.path
from operator import itemgetter
from functools import reduce
import classes.knutil as ku
import classes.boxtools as bt
#from operator import itemgetter, attrgetter


class KnCharException(Exception):
    def __init__(self, value):
        if value is None:
            self.initException()
        else:
            self.value = value

    def __str__(self):
        return repr(self.value)

    def printException(self):
        print("KnChar Exception.")

    @classmethod
    def paramsFileNotFound(self, value):
        print(('%s not found.' % value))

    @classmethod
    def initException(self):
        print("parameter file name must be specified.")


class KnCharParamsException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class KnChar:
    """
    属性
    img
    i,j  pageの何行目、何文字目
    stat
    alt
    binarized
    imgfname
    imgfullpath
    paramfname
    boxes
    candidates
    gray
    centroids   contoursの重心のリスト
    gradients_sobel
    gradients_laplacian
    gradients_scharr
    contours
    """
    def __init__(self, img, i=None, j=None):
        if img is None:
            raise KnCharException('img is None')
        # self.parameters = params
        self.img = img
        if i:
            self.i = i
        if j:
            self.j = j

