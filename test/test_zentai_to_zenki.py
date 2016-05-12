# -*- coding: utf-8 -*-

import pytest
import copy
from unittest import TestCase as tc
from hamcrest import *

import sys
import os.path
TEST_ROOT = os.path.dirname( os.path.abspath(__file__))
CLASSES_ROOT = os.path.split(TEST_ROOT)[0] + '/classes'
sys.path.append(TEST_ROOT)
sys.path.append(CLASSES_ROOT)
print(sys.path)

from knparam import KnParam
from knpage import KnPage
from knutil import DATA_DIR, check_test_environment
from output_samples import *

Default_Param = {
    "param": {
        "arcdir":      DATA_DIR,
        "topdir":      DATA_DIR
    },
    "book": {
        "height":       600,
        "width":        400,
        "pages_in_koma": 2,
        "dan":          1,
        "vorh":         "vert",
        "morc":         "mono",
        "keisen":       "no",
        "waku":         "yes"
    },
    "koma": {
        "scale_size":   [320.0, 640.0],
        "binarize":     ["canny"],
        "feature":      ["hough"],
        "hough":        [[1, 2, 100]],
        "canny":        [[50, 150, 3], [100, 200, 3]],
    },
    "page": {
        "pagedir":      "/".join(['can_50_200_3', 'hgh_1_2_100', 'right']),
        "lr":           "right",
        "mavstd":       10,
        "pgmgn":        [0.05, 0.05],
        "ismgn":        [15, 5],
        "toobig":       [200, 200],
        "boundingRect": [16, 32],
        "mode":         "EXTERNAL",
        "method":       "NONE",
        "canny":        [50, 200, 3]
    }
}

def pytest_funcarg__kn00900(request):
    param_dict = copy.deepcopy(Default_Param)
    spec = {
        "param": {
            "logfilename": "kn00900",
            "outdir":      "/".join([DATA_DIR, "1062973"]),
            "paramfdir":   "1062973/k009/00",
            "paramfname":  "knp.json",
            "balls":       ["1062973"]
        },
        "book": {
            "bookdir":      '1062973',
            "bookId":       "1062973"
        },
        "koma": {
            "komadir":      'k009',
            "komaId":       9,
            "komaIdStr":    "009",
            "imgfname":     "009.jpeg"
        },
        "page": {
            "pagedir":      '00',
            "imgfname":     "009_0.jpeg"
        }
    }
    for k, v in param_dict.items():
        v.update(spec[k])
    check_test_environment(param_dict, '1062973')
    knp = KnPage(params=KnParam(param_dict))
    return knp

class TestZentaiToZenki():

    def test_boxes_coodinates_data_to_textfile(self, kn00900):
        write_boxes_coordinates_data_to_textfile(kn00900, DATA_DIR, ext=".when_zero.txt")
        kn009.getContours()
        fpath1 = write_boxes_coordinates_data_to_textfile(kn00900, DATA_DIR+'/1062973/k009/00', ext=".after_getContour.txt")
        kn009.getCentroids()
        fpath2 = write_boxes_coordinates_data_to_textfile(kn00900, DATA_DIR+'/1062973/k009/00', ext=".after_getCentroids.txt")
        assert kn00900 != None
        assert os.path.exists(fpath1)
        assert os.path.exists(fpath2)

    def test_write_contours_and_hierarchy_data_to_textfile(self, kn00900):
        fpath = write_contours_and_hierarchy_data_to_textfile(kn00900, DATA_DIR+'/1062973/k009/00', ext='.txt')
        assert kn00900 != None
        assert os.path.exists(fpath)

    def test_write_boxes_to_file(self, kn00900):
        fpaths = write_boxes_to_file(kn00900, DATA_DIR+'/1062973/k009/00',(1, 100), '_1_100')
        assert kn00900 != None
        assert os.path.exists(fpaths[0])
        assert os.path.exists(fpaths[1])
        assert os.path.exists(fpaths[2])

    def test_write_collected_boxes_to_file(self, kn00900):
        fpath = write_collected_boxes_to_file(kn00900, DATA_DIR+'/1062973/k009/00')
        assert kn00900 != None
        assert os.path.exists(fpath)

    def test_write_original_with_collected_boxes_to_file(self, kn00900):
        fpath = write_original_with_collected_boxes_to_file(kn00900, DATA_DIR + '/1062973/k009/00')
        assert kn00900 != None
        assert os.path.exists(fpath)

class TestAllPageOutput():

    def test_all_page_out(self):
        for i in range(5, 45):
            for j in range(0, 2):
                param_dict = copy.deepcopy(Default_Param)
                spec = {
                    "param": {
                        "logfilename": "kn0{0:0>2}_0{1}".format(i, j),
                        "outdir": "/".join([DATA_DIR, "1062973"]),
                        "paramfdir": "1062973/k0{0:0>2}/0{1}".format(i, j),
                        "paramfname": "knp.json",
                        "balls": ["1062973"]
                    },
                    "book": {
                        "bookdir": '1062973',
                        "bookId": "1062973"
                    },
                    "koma": {
                        "komadir": "k0{0:0>2}".format(i),
                        "komaId": i,
                        "komaIdStr": "0{0:0>2}".format(i),
                        "imgfname": "0{0:0>2}.jpeg".format(i)
                    },
                    "page": {
                        "pagedir": "0{0}".format(j),
                        "imgfname": "0{0:0>2}_{1}.jpeg".format(i, j)
                    }
                }
                for k, v in param_dict.items():
                    v.update(spec[k])
                check_test_environment(param_dict, '1062973')
                knparam0 = KnParam(param_dict)
                assert knparam0 != None
                knpage0 = KnPage(params=knparam0)
                assert knpage0 != None

                fpath = write_collected_boxes_to_file(knpage0, "/".join([DATA_DIR, knparam0['param']['paramfdir']]))
                assert os.path.exists(fpath)

                fpath = write_original_with_collected_boxes_to_file(knpage0, "/".join([DATA_DIR, knparam0['param']['paramfdir']]))
                assert os.path.exists(fpath)

