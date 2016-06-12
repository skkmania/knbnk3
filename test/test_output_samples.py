# -*- coding: utf-8 -*-

import pytest
from unittest import TestCase as tc
from hamcrest import *
from classes.knpage import KnPage
from classes.knutil import DATA_DIR
from output_samples import *
import os.path

class TestOutputSamples():

    def test_boxes_coodinates_data_to_textfile(self, kn005):
        kn = KnPage(params=kn005)
        write_boxes_coordinates_data_to_textfile(kn, DATA_DIR, ext=".when_zero.txt")
        kn.getContours()
        write_boxes_coordinates_data_to_textfile(kn, DATA_DIR, ext=".after_getContour.txt")
        kn.getCentroids()
        write_boxes_coordinates_data_to_textfile(kn, DATA_DIR, ext=".after_getCentroids.txt")
        assert kn != None

    def test_write_contours_and_hierarchy_data_to_textfile(self, kn005):
        kn = KnPage(params=kn005)
        fpath = write_contours_and_hierarchy_data_to_textfile(kn, DATA_DIR, ext='.txt')
        assert kn != None
        assert os.path.exists(fpath)


    def test_write_contours_bounding_rect_to_file(self, kn005):
        kn = KnPage(params=kn005)
        write_contours_bounding_rect_to_file(kn, DATA_DIR)
        assert kn != None

    def test_write_binarized_to_file(self, kn005):
        kn = KnPage(params=kn005)
        kn.getBinarized()
        write_binarized_to_file(kn, DATA_DIR)
        assert kn != None

    def test_writeContour(self, kn005):
        kn = KnPage(params=kn005)
        kn.getContours()
        writeContour(kn)
        assert kn != None

    def test_write_gradients(self, kn005):
        kn = KnPage(params=kn005)
        kn.getGradients()
        write_gradients(kn, DATA_DIR)
        assert kn != None

    def test_write_collected_boxes_to_file(self, kn005):
        kn = KnPage(params=kn005)
        write_collected_boxes_to_file(kn, DATA_DIR)
        assert kn != None

    def test_write_original_with_collected_boxes_to_file(self, kn005):
        kn = KnPage(params=kn005)
        write_original_with_collected_boxes_to_file(kn, DATA_DIR)
        assert kn != None

    def test_write_original_with_contour_to_file(self, kn005):
        kn = KnPage(params=kn005)
        write_original_with_contour_to_file(kn, DATA_DIR)
        assert kn != None

    def test_write_original_with_contour_and_rect_to_file(self, kn005):
        kn = KnPage(params=kn005)
        write_original_with_contour_and_rect_to_file(kn, DATA_DIR)
        assert kn != None

    def test_save_char_img_to_file(self, kaisetu):
        kn = KnPage(params=kaisetu)
        kn.clear_noise()
        kn.get_line_imgs()
        kn.get_chars()
        kn.check_chars()
        save_char_img_to_file(kn, DATA_DIR, 0, 0)
        save_char_img_to_file(kn, DATA_DIR, 1, 1)
        save_char_img_to_file(kn, DATA_DIR, line=2)
        save_char_img_to_file(kn, DATA_DIR, count=2)
        save_char_img_to_file(kn, DATA_DIR)
        assert kn != None
