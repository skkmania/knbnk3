# -*- coding: utf-8 -*-

import pytest
from unittest import TestCase as tc
from hamcrest import *
from classes.knpage import KnPage
from classes.knutil import DATA_DIR


#class TestKnPage(unittest.TestCase):
class TestKnPage():

    #@pytest.fixture
    def test_new(self, knp):
        kn = KnPage(params=knp)
        assert kn.img != None
        assert kn.img.shape != (100,100,3)
        assert kn.height == 2789
        assert kn.width == 3466
        assert kn.depth == 3
        return kn

    #@pytest.fixture
    #def test_write(self, tmpdir):
    def test_write(self, kn005):
        #dataDirectory = tmpdir.mkdir('data')
        #sampleFile = dataDirectory.join("sample.jpeg")
        kn = KnPage(params=kn005)
        #kn.write(sampleFile)
        kn.write("/tmp/outfile.jpeg")

    def test_getContours(self, kn005):
        kn = KnPage(params=kn005)
        kn.getContours()
        assert kn.gray != None
        assert kn.contours != None
        assert kn.hierarchy != None

    def test_getCentroids(self, kn005):
        kn = KnPage(params=kn005)
        kn.getCentroids()
        assert kn.centroids != None
        assert len(kn.centroids) == 3160

    def test_writeContour(self, kn005):
        kn = KnPage(params=kn005)
        kn.getContours()
        kn.writeContour()
        assert kn.img_of_contours != None

    def test_write_contours_bounding_rect_to_file(self, kn005):
        kn = KnPage(params=kn005)
        kn.write_contours_bounding_rect_to_file(DATA_DIR)
        assert kn != None

    def test_write_data_file(self, kn005):
        kn = KnPage(params=kn005)
        kn.write_data_file(DATA_DIR)

    def test_write_original_with_contour_file(self, kn005):
        kn = KnPage(params=kn005)
        kn.write_original_with_contour_file(DATA_DIR)

    def test_write_binarized_file(self, kn005):
        kn = KnPage(params=kn005)
        kn.write_binarized_file(DATA_DIR)

    def test_write_original_with_contour_and_rect_file(self, kn005):
        kn = KnPage(params=kn005)
        kn.write_original_with_contour_and_rect_file(DATA_DIR)

    def test_write_with_params(self, kn005):
        kn = KnPage(params=kn005)
        kn.write_data_file(DATA_DIR)
        kn.write_binarized_file(DATA_DIR)
        kn.write_contours_bounding_rect_to_file(DATA_DIR)
        kn.write_original_with_contour_file(DATA_DIR)
        kn.write_original_with_contour_and_rect_file(DATA_DIR)

    """
    def test_write_all(self):
        fname = DATA_DIR + '/twletters.jpg'
        for i in range(2,9):
          params = DATA_DIR + '/twletters_0' + str(i) + '.json'
          kn = KnPage(fname, params)
          kn.write_all(DATA_DIR)
    """

    def test_include(self, kn005):
        box1 = (20, 30, 10, 10)
        box2 = (25, 35, 15, 15)
        box3 = (35, 45, 10, 10)
        box4 = (35, 20, 20, 20)
        box5 = (10, 45, 20, 20)
        box6 = (27, 37, 10, 10)
        kn = KnPage(params=kn005)
        assert not kn.include(box1, box2)
        assert not kn.include(box1, box3)
        assert not kn.include(box1, box4)
        assert kn.include(box2, box6)

    def test_intersect(self, kn005):
        box1 = (20, 30, 10, 10)
        box2 = (25, 35, 15, 15)
        box3 = (35, 45, 10, 10)
        box4 = (35, 20, 20, 20)
        box5 = (10, 45, 20, 20)
        kn = KnPage(params=kn005)
        assert kn.intersect(box1, box2)
        assert kn.intersect(box1, box3)
        assert not kn.intersect(box1, box3, 0, 0)
        assert kn.intersect(box1, box3)
        assert kn.intersect(box1, box4)
        assert not kn.intersect(box1, box4, 0, 0)
        assert kn.intersect(box1, box5)
        assert not kn.intersect(box1, box5, 0, 0)
        assert kn.intersect(box2, box3)
        assert kn.intersect(box2, box4)
        assert kn.intersect(box2, box5)
        assert kn.intersect(box3, box4)
        assert not kn.intersect(box3, box4, 0, 0)
        assert kn.intersect(box3, box5)
        assert not kn.intersect(box3, box5, 0, 0)
        assert kn.intersect(box4, box5)
        assert not kn.intersect(box4, box5, 0, 0)

    def test_get_boundingBox(self, kn005):
        box1 = (20, 30, 10, 10)
        box2 = (25, 35, 15, 15)
        box3 = (35, 45, 10, 10)
        box4 = (35, 20, 20, 20)
        box5 = (10, 45, 20, 20)
        kn = KnPage(params=kn005)
        outer_box = kn.get_boundingBox([box1, box2,box3])
        assert outer_box == (20,30,25,25)
        outer_box = kn.get_boundingBox([box1, box2,box3,box4,box5])
        assert outer_box == (10,20,45,45)

    def test_sweep_included_boxes(self, kn005):
        box1 = (20, 30, 10, 10)
        box2 = (25, 35, 15, 15)
        box3 = (35, 45, 10, 10)
        box4 = (35, 20, 20, 20)
        box5 = (10, 45, 20, 20)
        box6 = (27, 37, 10, 10)
        kn = KnPage(params=kn005)
        result = kn.sweep_included_boxes([box1, box2, box3, box4, box5, box6])
        assert len(result) == 5

    def test_get_adj_boxes(self, kn005):
        box01 = (20, 30, 10, 10)
        box02 = (25, 35, 15, 15)
        box03 = (35, 45, 10, 10)
        box04 = (35, 20, 20, 20)
        box05 = (10, 45, 20, 20)
        box06 = (27, 37, 10, 10)
        box11 = (120, 30, 10, 10)
        box12 = (125, 35, 15, 15)
        box13 = (135, 45, 10, 10)
        box14 = (135, 20, 20, 20)
        box15 = (110, 45, 20, 20)
        box16 = (127, 37, 10, 10)
        boxes = [box01,box02,box03,box04,box05,box06, box11,box12,box13,box14,box15,box16]
        kn = KnPage(params=kn005)
        result = kn.get_adj_boxes(boxes, box01)
        assert list( set(result) - set([box01,box02,box03,box04,box05,box06]) )  == []

    def test_sweep_included_boxes_2(self, kn005):
        kn = KnPage(params=kn005)
        kn.sweep_included_boxes()
        kn.write_boxes_to_file(DATA_DIR)

    def test_write_self_boxes_to_file(self, kn005):
        kn = KnPage(params=kn005)
        kn.getCentroids()
        kn.write_self_boxes_to_file(DATA_DIR)

    # this test will take a long time(about 200 seconds)
    def test_collect_boxes(self, kn005):
        kn = KnPage(params=kn005)
        kn.collect_boxes()
        kn.write_collected_boxes_to_file(DATA_DIR)
        kn.write_original_with_collected_boxes_to_file(DATA_DIR)
