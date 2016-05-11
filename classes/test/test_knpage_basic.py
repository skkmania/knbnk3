# -*- coding: utf-8 -*-
import pytest
import hamcrest as h
from classes.knpage import KnPage
#from classes.knpage import KnPageException
#from classes.knpage import KnPageParamsException
import classes.knutil as ku
from classes.knutil import DATA_DIR

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


class TestFileName:
    def test_mkFilename(self, knp):
        kn = KnPage(knp)
        name = ku.mkFilename(kn, '_cont')
        expect = '/'.join([DATA_DIR,
                 '1091460/k001/can_50_200/hgh_1_2_100/right/001_0_cont.jpeg'])
        assert name == expect

    def test_write_data_file(self, knp):
        kn = KnPage(knp)
        kn.write_data_file(kn.pagedir)


class TestBoundingRect:
    def test_write_contours_bounding_rect_to_file(self, b1g101):
        knp = KnPage(b1g101)
        knp.write_contours_bounding_rect_to_file()

    def test_get_boundingBox(self, b1g101):
        knp = KnPage(b1g101)
        outer_box = knp.get_boundingBox([box01, box02, box03])
        assert outer_box == (20, 30, 25, 25)
        outer_box = knp.get_boundingBox([box01, box02, box03, box04, box05])
        assert outer_box == (10, 20, 45, 45)

    def test_include(self, b1g101):
        knp = KnPage(b1g101)
        assert not knp.include(box01, box02)
        assert not knp.include(box01, box03)
        assert not knp.include(box01, box04)
        assert knp.include(box02, box06)


class TestInterSect:
    def test_intersect(self, knp):
        box01 = (20, 30, 10, 10)
        box02 = (25, 35, 15, 15)
        box03 = (35, 45, 10, 10)
        box04 = (35, 20, 20, 20)
        box05 = (10, 45, 20, 20)
        kp = KnPage(params=knp)
        assert kp.intersect(box01, box02)
        assert kp.intersect(box01, box03)
        assert not kp.intersect(box01, box03, 0, 0)
        assert kp.intersect(box01, box03)
        assert kp.intersect(box01, box04)
        assert not kp.intersect(box01, box04, 0, 0)
        assert kp.intersect(box01, box05)
        assert not kp.intersect(box01, box05, 0, 0)
        assert kp.intersect(box02, box03)
        assert kp.intersect(box02, box04)
        assert kp.intersect(box02, box05)
        assert kp.intersect(box03, box04)
        assert not kp.intersect(box03, box04, 0, 0)
        assert kp.intersect(box03, box05)
        assert not kp.intersect(box03, box05, 0, 0)
        assert kp.intersect(box04, box05)
        assert not kp.intersect(box04, box05, 0, 0)


class TestSweepInPageMargin:
    def test_sweep_boxes_in_page_margin(self, kn005):
        kn = KnPage(params=kn005)
        kn.getBoxesAndCentroids()
        kn.sweep_boxes_in_page_margin()
        kn.sort_boxes()
        kn.write_boxes_to_file(fix='_trimed')


class TestSweepIncludedBoxes:
    def test_sweep_included_boxes(self, knp):
        kp = KnPage(knp)
        result = kp.sweep_included_boxes(
            [box01, box02, box03, box04, box05, box06])
        assert len(result) == 5

    def test_sweep_included_boxes_2(self, kn005):
        kn = KnPage(kn005)
        kn.getBoxesAndCentroids()
        kn.sweep_boxes_in_page_margin()
        kn.sweep_included_boxes()
        kn.sort_boxes()
        kn.write_boxes_to_file(fix='_no_inclusion')

    def test_sweep_included_boxes_3(self, b1g101):
        kn = KnPage(b1g101)
        kn.getBoxesAndCentroids()
        kn.sweep_boxes_in_page_margin()
        kn.sweep_included_boxes()
        kn.sort_boxes()
        kn.write_boxes_to_file(fix='_no_inclusion')

    def test_sweep_included_boxes_4(self, b1g102):
        kn = KnPage(b1g102)
        kn.getBoxesAndCentroids()
        kn.sweep_boxes_in_page_margin()
        kn.sweep_included_boxes()
        kn.sort_boxes()
        kn.write_boxes_to_file(fix='_no_inclusion')


class TestSortBoxes:
    def test_sort_boxes(self, kn005):
        kn = KnPage(kn005)
        kn.getBoxesAndCentroids()
        kn.sort_boxes()
        kn.write_boxes_to_file(target=[100, 200])


class TestSweepMaverickBoxes:
    def test_sweep_maverick_boxes(self, kn005):
        kn = KnPage(kn005)
        kn.getBoxesAndCentroids()
        kn.sweep_boxes_in_page_margin()
        kn.sweep_included_boxes()
        kn.sweep_maverick_boxes()
        kn.sort_boxes()
        kn.write_boxes_to_file(fix='_no_mavericks')
        kn.sweep_maverick_boxes()
        kn.write_boxes_to_file(fix='_no_mavericks_2')
        kn.sweep_maverick_boxes()
        kn.write_boxes_to_file(fix='_no_mavericks_3')


class TestCollectBoxes:
    def test_collect_boxes(self, kn005):
        kn = KnPage(kn005)
        kn.getBoxesAndCentroids()
        kn.collect_boxes()
        kn.write_collected_boxes_to_file()
        kn.write_original_with_collected_boxes_to_file()


class TestEstimateCharSize:
    def test_estimate_char_size(self, kn005):
        kn = KnPage(kn005)
        kn.getBoxesAndCentroids()
        kn.collect_boxes()
        kn.estimate_char_size()


class TestEstimateVerticalLines:
    def test_estimate_vertical_lines(self, kn005):
        kn = KnPage(kn005)
        kn.getBoxesAndCentroids()
        kn.collect_boxes()
        kn.estimate_char_size()
        kn.estimate_vertical_lines()


class TestEstimateRotateAngle:
    def test_estimate_rotate_angle(self, kn005):
        kn = KnPage(kn005)
        kn.getBoxesAndCentroids()
        kn.collect_boxes()
        kn.estimate_char_size()
        kn.estimate_vertical_lines()
        kn.estimate_rotate_angle()


class TestRotateImage:
    def test_rotate_image(self, kn005):
        kn = KnPage(kn005)
        kn.getBoxesAndCentroids()
        kn.collect_boxes()
        kn.estimate_char_size()
        kn.estimate_vertical_lines()
        kn.estimate_rotate_angle()
        kn.rotate_image()
        kn.write_rotated_img_to_file()


class TestRotateImageManyCase:
    def test_rotate_image_many_case(self, kn005):
        kn = KnPage(kn005)
        for x in [0.7, 0.75, 0.8, 0.85, 0.9]:
            kn.estimated_angle = x
            kn.rotate_image()
            kn.write_rotated_img_to_file(fix='_%f' % x)
        for x in [0.7, 0.75, 0.8, 0.85, 0.9]:
            kn.estimated_angle = -1 * x
            kn.rotate_image()
            kn.write_rotated_img_to_file(fix='_m%f' % x)


class TestManipulateBoxes:
    def test_get_adj_boxes(self, kn005):
        kn = KnPage(kn005)
        boxes = [box01, box02, box03, box04, box05, box06,
                 box11, box12, box13, box14, box15, box16]
        result = kn.get_adj_boxes(boxes, box01)
        assert list(set(result) -
                    set([box01, box02, box03, box04, box05, box06])) == []
