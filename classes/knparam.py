# -*- coding: utf-8 -*-
import logging
import copy
import os
import os.path
import json
import pprint
import knutil as ku
#import knbook as kb
from datetime import datetime
#import knkoma as kk
from classes.knutil import DATA_DIR

__all__ = ["KnParam", "KnParamException", "KnParamParamsException", "HOME_DIR", "DATA_DIR"]


# param_fname file :  parameterをjson形式であらわしたテキストファイル
# param_fname file の書式 :  json text
#     注意：commaの有無
#      文字列のquotation : 数字、配列以外は文字列なので""でくくること
#   {
#     以下は必須
#     "imgfname"     : "string"                 #  読み込む画像filename (full path)
#     "outdir"       : "string"                 #  出力するfileのdirectory
#     "paramfname"   : "string"                 # parameter file name
#                             (つまりこのfile自身のfull path)
#     以下は任意
#     "outfilename"  : "string",                # 出力するfileのbasenameを指定
#     "boundingRect" : [min, max],              # boundingRectの大きさ指定
#     "contour"      : [mode, method],
#         "mode"         : findContoursのmode,   # EXTERNAL, LIST, CCOMP, TREE
#         "method"       : findContoursのmethod, # NONE, SIMPLE, L1, KCOS

#   HoughLinesのparameter
#     "hough"        : [rho, theta, minimumVote]
#          rho : accuracy of rho.  integerを指定。1 など。
#          theta:  accuracy of theta. int(1 - 180)を指定。
#                  np.pi/180 などradianで考えるので、その分母を指定する。
#                  180なら1度ずつ、2なら水平と垂直の直線のみを候補とするという意味
#          minimumVote:
#          lineとみなすため必要な点の数。検出する線の長さに影響する。

#   以下の4つは排他。どれかひとつを指定。配列内の意味はopencvのdocを参照のこと
#     2値化のやりかたを決める重要な設定項目。
#     "canny"        : [threshold1, threshold2, apertureSize],
#     "threshold"    : [thresh, maxval, type],
#     "adaptive"     : [maxval, method, type, blockSize, C]
#     "harris"       : [blockSize, ksize, k]

#   以下の3つはgradientsのparameter。配列内の意味はopencvのdocを参照のこと
#     本プロジェクトには意味がない。
#     "scharr"       : [depth, dx, dy, scale, delta, borderType]
#     "sobel"        : [depth, dx, dy, ksize]
#     "laplacian"    : [depth]
#       これらのdepth は6 (=cv2.CV_64F)とするのが一般的
#
#   以下はpage_splitのparameter
# 処理するときの縦サイズ(px).
# 小さいほうが速いけど、小さすぎると小さい線が見つからなくなる.
#cvHoughLines2のパラメータもこれをベースに決める.
#     "scale_size"   : num  # 640.0 など対象画像の細かさに依存
# 最低オフセットが存在することを想定する(px).
# 真ん中にある謎の黒い線の上下をtop,bottomに選択しないためのテキトウな処理で使う.
#     "hard_offset"  : num  # 32
# ページ中心のズレの許容範囲(px / 2).
#  余白を切った矩形の中心からこの距離範囲の間でページの中心を決める.
#     "center_range" : num  # 64
# 中心を決める際に使う線の最大数.
#     "CENTER_SAMPLE_MAX" : num  # 1024
# 中心決めるときのクラスタ数
#     "CENTER_K" : num  # 3
#   }
#

MandatoryFields = {
    "param": ["arcdir", "paramfdir", "topdir", "outdir",
              "paramfname", "logfilename", "balls"],
    "book":  ["bookdir", "bookId"],
    "koma":  ["komadir", "komaId", "komaIdStr",
              "scale_size", "binarize", "feature", "hough", "imgfname"],
    "page":  ["pagedir", "imgfname", "lr", "boundingRect",
              "mode", "method", "mavstd", "pgmgn", "ismgn", "toobig", "canny"]
}


class KnParamException(Exception):
    def __init__(self, value):
        if value is None:
            self.initException()
        else:
            self.value = value

    def __str__(self):
        return repr(self.value)

    def printException(self):
        print("KnParam Exception.")

    @classmethod
    def paramsFileNotFound(self, value):
        print('%s not found.' % value)

    @classmethod
    def initException(self):
        print("parameter file name must be specified.")


class KnParamParamsException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        if self.value in MandatoryFields:
            return repr("param file lacks %s." % self.value)
        else:
            return repr(self.value)


class KnParam(dict):
    def __init__(self, param_dict=None, param_fname=None):
        dict.__init__(self)
        if param_dict is None and param_fname is None:
            raise KnParamParamsException(
                'param_dict or param_fname must be specified.')
        elif param_dict:
            if isinstance(param_dict, dict):
                for k in param_dict:
                    self[k] = param_dict[k]
            else:
                raise KnParamParamsException('param_dict must be dict object.')
        else:
            if isinstance(param_fname, str):
                self.read_paramf(param_fname)
            else:
                raise KnParamParamsException('param_fname must be string.')
        self.mandatory_check()
        self.requirement_check()

    def read_paramf(self, param_fname):
        if os.path.exists(param_fname):
            with open(param_fname) as f:
                lines = f.readlines()
                j = json.loads(''.join(lines))
            for k in j:
                self[k] = j[k]
        else:
            raise KnParamParamsException(param_fname + ' not found.')

    def mandatory_check(self):
        for k, v in MandatoryFields.items():
            if not k in self.keys():
                raise KnParamParamsException(k)
            else:
                for f in v:
                    if not f in self[k].keys():
                        raise KnParamParamsException(f)

    def requirement_check(self):
        for k in self["koma"]["binarize"]:
            if not k in self["koma"]:
                raise KnParamParamsException(k)
        for k in self["koma"]["feature"]:
            if not k in self["koma"]:
                raise KnParamParamsException(k)

    @ku.deblog
    def clone(self):
        tmp = {}
        for k in self:
            tmp[k] = copy.deepcopy(self[k])
        ret = KnParam(tmp)
        ret.set_logger(self['param']['loggername'])
        return ret

    @ku.deblog
    def start(self):
        self.check_environment()
        self.expand_tarballs()

    @ku.deblog
    def check_environment(self):
        pass

    """
    @ku.deblog
    def expand_tarballs(self):
        for ball in self.ball_list():
            self.logger.debug(ball)
            p = self.clone_for_book(ball)
            kb.KnBook(p).start()
    """


    def ball_list(self):
        return self['param']["balls"]


    def clone_for_book(self, ball):
        """
        自らをKnBookに渡すに当たって、対象のtarballやbookIdをトップレベルにもってくるなど
        自らの中身を調整する
        """
        ret = self.clone()
        ret['book']["bookdir"] = self['param']['workdir'] + '/' + ball
        ret['book']["bookId"] = ball
        return ret

    def isBook(self):
        """
        KnBookのparameterとしての必要条件を満たすか判定
        """
        if not "book" in self.keys():
            return False
        elif not "id" in self['book'].keys():
            return False
        else:
            return True

    def datadir(self):
        """
        出力: text : tarballが存在するdirectoryのfull path
        """
        return self['param']['datadir']

    def paramfdir(self):
        """
        出力: text : parameter json fileが存在するdirectoryのfull path
        """
        return self['param']['paramfdir']

    def workdir(self):
        """
        出力: text : tarballを展開し作成されるdirectoryのfull path
        """
        return self['param']['workdir']

    def outdir(self):
        """
        出力: text : 最終成果物を出力する先のdirectoryのfull path
        """
        return self['param']['outdir']

    def bookId(self):
        """
        出力: text : NDLの永続的識別子からとった数字の列
        """
        return self['book']['bookId']

    def mkPageParam(self, komanum):
        komanumstr = str(komanum).zfill(3)
        params = {}
        params['komanumstr'] = komanumstr
        params['paramfname'] = self.parameters['outdir']\
            + '/k_' + komanumstr + '.json'
        params['imgfname'] = self.parameters['outdir'] + '/'\
            + komanumstr + '.jpeg'
        params['outdir'] = self.parameters['outdir']
        params['outfilename'] = "auto"
        params['mode'] = "EXTERNAL"
        params['method'] = "NONE"
        params['hough'] = [1, 2, 100]
        params['canny'] = [50, 200, 3]
        params['scale_size'] = 640.0
        ku.print_params_files([params])
        return params['paramfname']

    def check_params(self):
        for k in MandatoryFields:
            if not k in self.raw.keys():
                raise KnParamParamsException(k)

    def get_numOfKoma(self):
        return self['book']['numOfKoma']

    def set_numOfKoma(self, n):
        self['book']['numOfKoma'] = n

    def get_komaIdStr(self):
        return self['koma']['komaIdStr']

    def set_komaId(self, current):
        komaIdStr = str(current).zfill(3)
        self['koma']['komaIdStr'] = komaIdStr
        self['koma']['komaId'] = current
        return komaIdStr

    def get_imgFullPath(self):
        fullpath = "/".join([self['param']['outdir'], self['koma']['komadir'], self['koma']['imgfname']])
        if os.path.exists(fullpath):
            return fullpath
        else:
            raise KnParamException(fullpath + ' does not exist.')

    def get_imgfname(self):
        if 'imgfname' in self['koma']:
            return self['koma']['imgfname']
        else:
            return self['koma']['imgfname']

    def set_lr(self, lr):
        self['page']['lr'] = lr

    def lrstr(self):
        return self['page']['lr']

    @ku.deblog
    def clone_for_page(self, page):
        self.logger.debug("page :\n" + pprint.pformat(page))
        ret = self.clone()
        ret['page'].update(page)
        self.logger.debug("self :\n" + pprint.pformat(self))
        return ret

    @ku.deblog
    def clone_for_koma(self, koma):
        ret = self.clone()
        ret['koma'].update(koma)
        return ret

    def set_logger(self, name, logfilename=None):
        nowstr = datetime.now().strftime("%Y%m%d_%H%M")
        logging.basicConfig()
        self['param']['loggername'] = name

        if logfilename is None:
            logfilename = self['param']['outdir'] + '/'\
                + self['param']['logfilename']
            file_handler = logging.FileHandler(
                filename=logfilename + '_' + nowstr + name + '.log')
        else:
            file_handler = logging.FileHandler(logfilename)

        file_handler.setFormatter(
            logging.Formatter('%(asctime)s %(name)s %(message)s',
                              datefmt='%H:%M:%S'))
        file_handler.level = logging.DEBUG

        self.logger = logging.getLogger(self['param']['loggername'])
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.warning("KnParam initialized :\n" + pprint.pformat(self))
