# -*- coding: utf-8 -*-
#import os.path
#import pytest
import copy
import json
import shutil
import sys

from os import path
TEST_ROOT = path.dirname( path.abspath(__file__))
CLASSES_ROOT = path.split(TEST_ROOT)[0]
sys.path.append(TEST_ROOT)
sys.path.append(CLASSES_ROOT)
print(sys.path)

from classes.knparam  import KnParam
from classes.knutil import *

HOME_DIR = 'C:/Users/skkmania'
DATA_DIR = 'Z:/knbnk/data'

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


def pytest_funcarg__knp(request):
    param_dict = copy.deepcopy(Default_Param)
    spec = {
        "param": {
            "logfilename": "kn021",
            "outdir":      "/".join([DATA_DIR, "1091460"]),
            "paramfdir":   "1091460",
            "paramfname":  "knp.json",
            "balls":       ["1091460"]
        },
        "book": {
            "bookdir":      '1091460',
            "bookId":       "1091460"
        },
        "koma": {
            "komadir":      'k001',
            "komaId":       1,
            "komaIdStr":    "001",
            "imgfname":     "001.jpeg"
        },
        "page": {
            "imgfname":     "001_0.jpeg"
        }
    }
    for k, v in param_dict.items():
        v.update(spec[k])
    ku.check_test_environment(param_dict, '1091460')
    return KnParam(param_dict)


def pytest_funcarg__kn005(request):
    param_dict = copy.deepcopy(Default_Param)
    spec = {
        "param": {
            "logfilename": "kn005",
            "outdir":      "/".join([DATA_DIR, "twletters"]),
            "paramfdir":   "twletters",
            "paramfname":  "twlkn005.json",
            "balls":       ["twletters"]
        },
        "book": {
            "bookdir":      "twletters",
            "bookId":       "twletters"
        },
        "koma": {
            "komadir":      'k005',
            "komaId":       5,
            "komaIdStr":    "005",
            "imgfname":     "005.jpeg"
        },
        "page": {
            "imgfname":     "005_0.jpeg"
        }
    }
    for k, v in param_dict.items():
        v.update(spec[k])
    check_test_environment(param_dict, 'twletters')
    return KnParam(param_dict)


def pytest_funcarg__knManyLines(request):
    param_dict = copy.deepcopy(Default_Param)
    spec = {
        "param": {
            "logfilename": "knM007",
            "outdir":      "/".join([DATA_DIR, "1142178"]),
            "paramfdir":   "1142178",
            "paramfname":  "knMany007.json",
            "balls":       ["1142178"]
        },
        "book": {
            "bookdir":      "1142178",
            "bookId":       "1142178"
        },
        "koma": {
            "scale_size":   320.0,
            "hough": [1, 2, 80],
            "canny": [50, 150, 3],
            "komadir":      'k007',
            "komaId":       7,
            "komaIdStr":    "007",
            "imgfname":     "007.jpeg",
            "small_zone_levels": {'upper':  [0.03, 0.1],
                                  'lower':  [0.9, 0.97],
                                  'center': [0.45, 0.55],
                                  'left':   [0.03, 0.12],
                                  'right':  [0.88, 0.97]}
        },
        "page": {
            "imgfname":     "007_0.jpeg"
        }
    }
    for k, v in param_dict.items():
        v.update(spec[k])
    ku.check_test_environment(param_dict, '1142178')
    return KnParam(param_dict)


def pytest_funcarg__knFewLines(request):
    param_dict = copy.deepcopy(Default_Param)
    spec = {
        "param": {
            "logfilename": "knF006",
            "outdir":      "/".join([DATA_DIR, "1123003"]),
            "paramfdir":   "1123003",
            "paramfname":  "knMany006.json",
            "balls":       ["1123003"]
        },
        "book": {
            "bookdir":      "1123003",
            "bookId":       "1123003"
        },
        "koma": {
            "scale_size":   320.0,
            "hough": [1, 180, 200],
            "canny": [50, 150, 3],
            "komadir":      'k006',
            "komaId":       7,
            "komaIdStr":    "006",
            "imgfname":     "006.jpeg"
        },
        "page": {
            "imgfname":     "006_0.jpeg"
        }
    }
    for k, v in param_dict.items():
        v.update(spec[k])
    ku.check_test_environment(param_dict, '1123003')
    return KnParam(param_dict)


def pytest_funcarg__graph2(request):
    """
    両側とも全面挿絵のサンプル
    """
    param_dict = copy.deepcopy(Default_Param)
    spec = {
        "param": {
            "logfilename": "kngraph2_009",
            "outdir":      "/".join([DATA_DIR, "graph2"]),
            "paramfdir":   "graph2",
            "paramfname":  "kngraph2.json",
            "balls":       ["graph2"]
        },
        "book": {
            "bookdir":      "graph2",
            "bookId":       "graph2"
        },
        "koma": {
            "komadir":      'k009',
            "komaId":       9,
            "komaIdStr":    "009",
            "imgfname":     "009.jpeg"
        },
        "page": {
            "imgfname":     "009_0.jpeg",
        }
    }
    for k, v in param_dict.items():
        v.update(spec[k])
    ku.check_test_environment(param_dict, 'graph2')
    return KnParam(param_dict)


def pytest_funcarg__b1g101(request):
    """
    両側とも全面挿絵のサンプル
    """
    param_dict = copy.deepcopy(Default_Param)
    spec = {
        "param": {
            "logfilename": "kn021",
            "outdir":      "/".join([DATA_DIR, "b1g101"]),
            "paramfdir":   "b1g101",
            "paramfname":  "b1g101.json",
            "balls":       ["b1g101"]
        },
        "book": {
            "bookdir":      'b1g101',
            "bookId":       "b1g101",
        },
        "koma": {
            "komadir":      'k021',
            "komaId":       21,
            "komaIdStr":    "021",
            "imgfname":     "021.jpeg"
        },
        "page": {
            "imgfname":     "021_0.jpeg",
        }
    }
    for k, v in param_dict.items():
        v.update(spec[k])
    ku.check_test_environment(param_dict, 'b1g101')
    return KnParam(param_dict)


def pytest_funcarg__b1g1011(request):
    """
    両側とも全面挿絵のサンプル
    """
    param_dict = copy.deepcopy(Default_Param)
    spec = {
        "param": {
            "logfilename": "kn021",
            "outdir":      "/".join([DATA_DIR, "b1g101"]),
            "paramfdir":   "b1g101",
            "paramfname":  "b1g101.json",
            "balls":       ["b1g101"]
        },
        "book": {
            "bookdir":      'b1g101',
            "bookId":       "b1g101",
        },
        "koma": {
            "scale_size":   320.0,
            "komadir":      'k021',
            "komaId":       21,
            "komaIdStr":    "021",
            "imgfname":     "021.jpeg"
        },
        "page": {
            "imgfname":     "021_0.jpeg",
        }
    }
    for k, v in param_dict.items():
        v.update(spec[k])
    ku.check_test_environment(param_dict, 'b1g1011')
    return KnParam(param_dict)


def pytest_funcarg__b1g1012(request):
    """
    両側とも全面挿絵のサンプル
    """
    param_dict = copy.deepcopy(Default_Param)
    spec = {
        "param": {
            "logfilename": "kn021",
            "outdir":      "/".join([DATA_DIR, "b1g101"]),
            "paramfdir":   "b1g101",
            "paramfname":  "b1g101.json",
            "balls":       ["b1g101"]
        },
        "book": {
            "bookdir":      'b1g101',
            "bookId":       "b1g101",
        },
        "koma": {
            "scale_size":   960.0,
            "komadir":      'k021',
            "komaId":       21,
            "komaIdStr":    "021",
            "imgfname":     "021.jpeg"
        },
        "page": {
            "imgfname":     "021_0.jpeg",
        }
    }
    for k, v in param_dict.items():
        v.update(spec[k])
    ku.check_test_environment(param_dict, 'b1g1011')
    return KnParam(param_dict)


def pytest_funcarg__b1g102(request):
    """
    両側とも全面挿絵のサンプル
    """
    param_dict = copy.deepcopy(Default_Param)
    spec = {
        "param": {
            "logfilename": "kn106",
            "outdir":      "/".join([DATA_DIR, "b1g102"]),
            "paramfdir":   "b1g102",
            "paramfname":  "b1g102.json",
            "balls":       ["b1g102"]
        },
        "book": {
            "bookdir":      "b1g102",
            "bookId":       "b1g102"
        },
        "koma": {
            "komadir":      'k106',
            "komaId":       106,
            "komaIdStr":    "106",
            "imgfname":     "106.jpeg"
        },
        "page": {
            "imgfname":     "106_0.jpeg"
        }
    }
    for k, v in param_dict.items():
        v.update(spec[k])
    ku.check_test_environment(param_dict, 'b1g102')
    return KnParam(param_dict)


def pytest_funcarg__knbk1(request):
    param_dict = copy.deepcopy(Default_Param)
    spec = {
        "param": {
            "logfilename": "knbk1",
            "outdir":      "/".join([DATA_DIR, "1091460"]),
            "paramfdir":   "1091460",
            "paramfname":  "knbk1.json",
            "balls":       ["1091460"]
        },
        "book": {
            "bookdir":      "1091460",
            "bookId":       "1091460"
        },
        "koma": {
            "komadir":      'k001',
            "komaId":       1,
            "komaIdStr":    "001",
            "imgfname":     "001.jpeg"
        },
        "page": {
            "imgfname":     "001_0.jpeg"
        }
    }
    for k, v in param_dict.items():
        v.update(spec[k])
    ku.check_test_environment(param_dict, "1091460")
    return KnParam(param_dict)


def generate_param_dicts(param_dict, v_list):
    """
        既存のparam_dictをもとにして新しいparam_dictを複数つくり、そのリストを返す
    入力:
        param_dict : 既存のparam_dict.
                     これをもとにして新しいparam_dictを複数つくる
        v_list   : 適用したいlistを、
                      {"key1": {"key1_1" ： [values1],
                                "key1_2" ： [values2]},
                       "key2": {"key2_1" ： [values3}}
                   という形で与える
    戻り値:
        param_dictにv_listを反映させた新しいparam_dictからなるリスト。
        その長さは、len(values1) * len(values2) * ...となる
    使用例:
        param_dict = {
        "koma": {
            "komadir":      'k001',
            "komaId":       1,
            },
        "page": {
            "imgfname":     "001_0.jpeg"
        }}
        v_list = {
        "koma": {
            "komadir":      ['k001', 'k002', 'k003'],
            },
        "page": {
            "imgfname":     ["001_0.jpeg", "001_1.jpeg"]
        }}
        とすると戻り値は 長さが6(=3*2)のリストとなる。
     長くなるので、変化しているところだけ記すと次のようなイメージ。
        [ { "koma": {
                "komadir":      'k001', }
            "page": {
                "imgfname":     "001_0.jpeg" } },
          { "koma": {
                "komadir":      'k002', }
            "page": {
                "imgfname":     "001_0.jpeg" } },
            ------
            以下略
        ]

         ココロとしては、既存のparameter fileに対して
             対象画像を一気に増やしたい
             cannyのparameterをいろいろと変えてみたい
         というときに使う
    """
    ret = []
    for key1, dict1 in v_list.items():
        for key2, list1 in dict1.items():
            for v in list1:
                new_param = mydeepcopy(param_dict)
                new_param[key1][key2] = v
                ret.append(new_param)
    return ret


def mydeepcopy(param_dict):
    """
    deep copy recursively
    """
    ret = {}
    for k, d in param_dict.items():
        ret[k] = {}
        for k2, v in d.items():
            ret[k][k2] = copy.deepcopy(v)
    return ret


def edit_parms_file(pfbody=None, imfname=None, opts=None, data_dir=None):
    if data_dir:
        DATA_DIR = data_dir
    if imfname:
        img_fname = DATA_DIR + '/' + imfname
    if pfbody:
        pfname = DATA_DIR + '/%s.json' % pfbody
        with open(pfname) as f:
            lines = f.readlines()
            params = json.loads(''.join(lines))
        params['imgfname'] = img_fname
        params['komanumstr'] = imfname.split('.')[0]
        params['outfilename'] = DATA_DIR + '/' +\
            pfbody + '_' + imfname.split('.')[0]
        shutil.move(pfname, pfname + '.bak')
    else:
        params = {}

    if opts:
        for k in opts:
            params[k] = opts[k]

    with open(params["paramfname"], "w") as f:
        json.dump(params, f, sort_keys=False, indent=4)
