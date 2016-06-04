# -*- coding: utf-8 -*-
import pytest
import cv2
from unittest import TestCase as tc
from hamcrest import *
from classes.knchar import KnChar
from classes.knutil import DATA_DIR


class TestKnChar():

    #@pytest.fixture
    def test_new(self):
        img = cv2.imread(DATA_DIR+'/kaisetu/gyou_0_moji_0.jpeg')
        kn = KnChar(img)
        assert kn.img != None
        assert kn.img.shape != (100,100,3)
        return kn
