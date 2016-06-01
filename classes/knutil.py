# -*- coding: utf-8 -*-
import json
import os
import os.path
import time
import numpy as np
import itertools
import sys
import cv2

__all__ = ["print_params_files", "check_test_environment", "mkFilename",
           "deblog", "KnUtilException", "KnUtilParamsException"]

HOME_DIR = 'C:/Users/skkmania'
DATA_DIR = 'Z:/knbnk/data'


class KnUtilException(Exception):
    def __init__(self, value):
        if value is None:
            self.initException()
        else:
            self.value = value

    def __str__(self):
        return repr(self.value)

    def printException(self):
        print("KnUtil Exception.")

    @classmethod
    def paramsFileNotFound(self, value):
        print('%s not found.' % value)

    @classmethod
    def initException(self):
        print("parameter file name must be specified.")


class KnUtilParamsException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def get_range_list(hist, blanc_std):
    """
    数値のリストから、基準以上の値が連続する区間を探し、それらのリストを返す
    区間は、もとのリストのindexのリスト [start, end] で表現する
        例:
       　hist : [0,0,0,30,50,40,10,0,0,40,30,5,0],  blanc_std : 20 のとき、
        結果は  [[3,5],[9,10]] となる
    区間が見つからないときは、空のリストを返す
    :param hist:  list of number
    :param blanc_std: number
    :return: list of list of number
    """
    range_list = []  # 求める最終結果. 数字のリストのリスト
    range_flag = False  # rangeの中途にあるかどうか
    for i, v in enumerate(hist):
        if v >= blanc_std:
            if not range_flag:
                range_flag = True
                range_list.append([i])
        else:
            if range_flag:
                range_flag = False
                range_list[-1].append(i - 1)
    if range_list == [[0]]:  #  histの全要素がblanc_std以上のとき
        range_list = [[0, len(hist) - 1]]
    return range_list
    
def print_params_files(params_list):
    ret = []
    for params in params_list:
        topdir = params['param']['topdir']
        paramfdir = params['param']['paramfdir']
        paramfname = params['param']['paramfname']
        fname = "/".join([topdir, paramfdir, paramfname])
        with open(fname, 'w') as f:
            json.dump(params, f, sort_keys=False, indent=4)
            ret.append(fname)
    return ret


def check_test_environment(param_dict, bookId):
    """
    paramsに記述されたoutdirの存在確認
      なければ、tarballの展開とoutdirの作成
    parmsのtxt file( json format)の作成は常に行う
    (testのたびにそのtestの設定を使うこと。
    別のtestの影響を受けたくないので。)
    """
    if not os.path.exists(param_dict['param']['outdir']):
        cmd = 'tar jxf %s/%s.tar.bz2 -C %s' % (DATA_DIR, bookId, DATA_DIR)
        os.system(cmd)
        cmd = "find %s -type d -name '*%s*' -exec mv {} %s \\;" %\
            (DATA_DIR, bookId, param_dict['param']["outdir"])
        os.system(cmd)

    print_params_files([param_dict])


def mkFilename(obj, fix, outdir=None, ext=None):
    """
     obj : KnPageを想定(imgfullpath をpropertyにもつ)
     fix : file name の末尾に付加する
     outdir : 出力先directoryの指定
     ext : 拡張子の指定 .txt のように、. ではじめる
    """
    dirname = os.path.dirname(obj.imgfullpath)
    basename = os.path.basename(obj.imgfullpath)
    if fix == 'data':
        name, ext = os.path.splitext(basename)
        if hasattr(obj, 'outfilename'):
            name = obj.outfilename
        name = name + '_data'
        ext = '.txt'
    else:
        if ext is None:
            name, ext = os.path.splitext(basename)
        else:
            name = os.path.splitext(basename)[0]

        if hasattr(obj, 'outfilename'):
            name = obj.outfilename
        name = name + fix

    if outdir is None:
        return "/".join([dirname, name + ext])
    else:
        return "/".join([outdir, name + ext])
        # os.path.join(outdir, name = ext)
        # とするとバックスラッシュで結ばれる（Windows）
        # どちらが正しい？

def write(obj, outfilename=None, om=None):
    if om is None:
        om = obj.img
    if outfilename is None:
        if hasattr(obj, 'outfilename'):
            outfilename = obj.outfilename
        else:
            raise
    cv2.imwrite(outfilename, om)


def isVertical(line):
    """
    線分が水直であるかどうか判定する
    入力: line
     line = [[x1, y1],[x2, y2]]  直線を通る2点により線分を指定
     line = (rho, theta)         原点からの距離と角度で線分を指定
     を判別して対応
    戻り値: boolean : 水直ならTrue, 水直でなければ False
    """
    if isinstance(line[0], list) or isinstance(line[0], tuple):
        return (line[0][0] == line[1][0])
    else:
        return line[1] < 0.01


def isHorizontal(line):
    """
    線分が水平であるかどうか判定する
    入力: line
     line = [[x1, y1],[x2, y2]]  直線を通る2点により線分を指定
     line = (rho, theta)         原点からの距離と角度で線分を指定
     を判別して対応
    戻り値: boolean : 水平ならTrue, 水平でなければ False
    """
    if isinstance(line[0], list) or isinstance(line[0], tuple):
        return (line[0][1] == line[1][1])
    else:
        return abs(line[1] - np.pi / 2) < 0.01


def compLine(line0, line1, horv):
    """
    line0, line1 の関係を返す
    入力:
        line0, line1 の形式は2点指定式。[(x0,y0), (x1,y1)]
        horv :  "h" (水平)  or  "v" (水直) を指定
    戻り値: string :
        when horv == h
          "upper" : line0のy座標が
                    line1のy座標より大きい、つまりline1が上にある
          "lower" : line1のy座標が
                    line0のy座標より大きい、つまりline1が下にある
          "right" : line0のx座標が
                    line1のx座標より大きい、つまりline0が
                    右にある
          "left" :  line1のx座標が
                    line0のx座標より大きい、つまりline0が
                    左にある
    """
    if horv == 'h':
        if isHorizontal(line0) and isHorizontal(line1):
            if max(line0[0][1], line0[1][1]) >\
                    max(line1[0][1], line1[1][1]):
                return "upper"
            else:
                return "lower"
        else:
            raise ('wrong recognition of line')
    else:
        if isVertical(line0) and isVertical(line1):
            if max(line0[0][0], line0[1][0]) >\
                    max(line1[0][0], line1[1][0]):
                return "right"
            else:
                return "left"
        else:
            raise ('wrong recognition of line')


def findCornerLineP(linePoints):
    a = linePoints
    vlines = [vline for vline in a if abs(vline[0][0] - vline[1][0]) < 50]
    hlines = [hline for hline in a if abs(hline[0][1] - hline[1][1]) < 50]
    upper_hline = hlines[0]
    lower_hline = hlines[0]
    for line in hlines:
        if compLine(line, upper_hline, 'h') == "upper":
            upper_hline = line

        if compLine(line, lower_hline, 'h') == "lower":
            lower_hline = line

    right_vline = vlines[0]
    left_vline = vlines[0]
    for line in vlines:
        if compLine(line, right_vline, 'v') == "right":
            right_vline = line

        if compLine(line, left_vline, 'v') == "left":
            left_vline = line


def getIntersection(line1, line2):
    """
     Finds the intersection of two lines, or returns false.
     line1 = [[x1, y1],[x2, y2]]
     line2 = [[x1, y1],[x2, y2]]
    """
    s1 = np.array([float(x) for x in line1[0]])
    e1 = np.array([float(x) for x in line1[1]])

    s2 = np.array([float(x) for x in line2[0]])
    e2 = np.array([float(x) for x in line2[1]])

    if isVertical(line1):
        if isVertical(line2):
            return False
        else:
            a2 = (s2[1] - e2[1]) / (s2[0] - e2[0])
            b2 = s2[1] - (a2 * s2[0])
            x = line1[0][0]
            y = a2 * x + b2

    elif isVertical(line2):
        a1 = (s1[1] - e1[1]) / (s1[0] - e1[0])
        b1 = s1[1] - (a1 * s1[0])
        x = line2[0][0]
        y = a1 * x + b1

    elif isHorizontal(line1):
        if isHorizontal(line2):
            return False
        else:
            a2 = (s2[1] - e2[1]) / (s2[0] - e2[0])
            b2 = s2[1] - (a2 * s2[0])
            y = line1[0][1]
            x = (y - b2) / a2

    elif isHorizontal(line2):
        a1 = (s1[1] - e1[1]) / (s1[0] - e1[0])
        b1 = s1[1] - (a1 * s1[0])
        y = line1[1][1]
        x = (y - b1) / a1

    else:
        a1 = (s1[1] - e1[1]) / (s1[0] - e1[0])
        b1 = s1[1] - (a1 * s1[0])
        a2 = (s2[1] - e2[1]) / (s2[0] - e2[0])
        b2 = s2[1] - (a2 * s2[0])

        if abs(a1 - a2) < sys.float_info.epsilon:
            return False

        x = (b2 - b1) / (a1 - a2)
        y = a1 * x + b1

    return (int(round(x)), int(round(y)))


def deblog(func):
    def wrapper(*args, **kwargs):
        if "KNBNK_DEBUG" in os.environ:
            args[0].logger.debug('*IN* %s#%s.' %
                                 (args[0].__class__.__name__, func.__name__))
            if len(args) > 1:
                for arg in args[1:]:
                    args[0].logger.debug('with %s' % str(arg))
            if len(kwargs) > 0:
                for k, v in kwargs.items():
                    args[0].logger.debug('with %s=%s' % (k, str(v)))
        res = func(*args, **kwargs)
        if "KNBNK_DEBUG" in os.environ:
            args[0].logger.debug('*OUT* %s.' % func.__name__)
        print(func.__name__, args, kwargs)
        return res
    return wrapper

def mkOutFilename(params, str):
    """
    just a stub
    :param params:
    :param str:
    :return:
    """
    return "temporal"

def params_generator(source):
    """

    :param source:
    :return:
    """
    keys = source.keys()
    vals_products = map(list, itertools.product(*source.values()))
    todict = lambda a, b: dict(zip(a, b))
    metadict = lambda a: (lambda b: todict(a, b))
    temp = map(metadict(keys), vals_products)
    cnt = 0
    for params in temp:
        params["outfilename"] = mkOutFilename(params, '_' + str(cnt))
        cnt += 1
        params["paramfname"] = params['outdir'] + '/' +\
            params['outfilename'] + '.json'
    return temp


class ImageManager:
    """
    入力: obj : KnKoma, KnPageなど、obj.img というpropertyを持つもの
    戻り値: dict : obj.img から求めたcornerLine
    """
    def __init__(self, obj):
        if obj.img is None:
            self.initException()
        else:
            self.tgtObj = obj
            self.img = obj.img
            self.height, self.width, self.depth = self.img.shape
            self.candidates = {'upper': [], 'lower': [],
                               'center': [], 'left': [], 'right': []}
            self.logger = obj.logger
            self.p = obj.p
            self.parameters = obj.parameters
            self.complemented = False
            self.get_corner_lines()

    @deblog
    def find_pages_in_img(self):
        """
        KnKoma obj から問い合わせをうけ、画像のなかのページを探し、その数を答える
        出力: integer : page の数
        """
        if self.get_corner_lines() is False:
            # 1pageなのか無理に2pageにするのか判断が必要。
            return self.check_1or2()
        else:
            # cornerLines が５本あったので２ページと判断する
            self.pages_in_img = 2
            return self.pages_in_img

    @deblog
    def get_original_corner(self):
        self.originalCorner = {}
        for d in ['upper', 'lower', 'center', 'right', 'left']:
            self.originalCorner[d] = int(self.cornerLines[d][0] / self.scale)
        return self.originalCorner

    @deblog
    def get_corner_lines(self):
        """
        cornerLinesを算出する
        その過程でparameterを調整する
        副作用 : self.cornerLines を設定
        """
        self.cornerLines = {}
        try_count = 0
        while try_count < 5:
            self.get_binarized("small")
            self.getHoughLines()
            if self.lines is None:
                self.logger.debug(
                    'lines not found. try count : %d' % try_count)
                self.changeparam(try_count)
                try_count += 1
                continue
            elif self.enoughLines():
                self.findCornerLines()
                if self.isCenterAmbiguous():
                    self.findCenterLine()
                return self.cornerLines
            else:
                self.logger.debug(
                    'lines not enough. try count : %d' % try_count)
                self.changeparam(try_count)
                try_count += 1
        else:
            self.logger.debug('KnKoma#divide: retry over 5 times and gave up!')
            return False

    def check_1or2(self):
        """
        5本に満たないcornerLinesと、
        その他の情報から、
        この画像に2ページあるのか、1ページなのかを決定する
        """

    @deblog
    def get_binarized(self, flag):
        """
        binarized画像を作成
        hough parameters (rho, theta, minimumVote)をいったん確定しておく
        canny parameters (minval, maxval, apertureSize)をいったん確定しておく
        """
        if flag == "small":
            if 'scale_size' in self.parameters:
                self.scale_size = self.parameters['scale_size']
                if isinstance(self.scale_size, list):
                    self.scale_size = self.scale_size[0]
                self.scale = self.scale_size / self.width
            else:
                raise 'scale_size must be in param file'
            self.small_img = cv2.resize(self.img,
                                        (int(self.width * self.scale),
                                         int(self.height * self.scale)))
            self.small_height, self.small_width, self.small_depth =\
                self.small_img.shape
            self.small_img_gray = cv2.cvtColor(self.small_img,
                                               cv2.COLOR_BGR2GRAY)
            obj_img = self.small_img_gray
        elif flag == "original":
            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            obj_img = self.gray
        else:
            msg = 'ImageManager#get_binarized:' +\
                'flag must be "small" or "original"'
            self.logger.fatal(msg)
            raise msg

        if 'threshold' in self.parameters:
            thresh_low, thresh_high, typeval = self.parameters['threshold']
            ret, binarized_img =\
                cv2.threshold(obj_img, thresh_low, thresh_high, typeval)
            msg = 'binarized_img created by threshold : %s'\
                % str(self.parameters['threshold'])
        elif 'canny' in self.parameters:
            minval, maxval, apertureSize = self.parameters['canny']
            binarized_img = cv2.Canny(obj_img, minval, maxval, apertureSize)
            msg = 'binarized_img created by canny : %s'\
                % str(self.parameters['canny'])
        elif 'adaptive' in self.parameters:
            binarized_img =\
                cv2.adaptiveThreshold(obj_img, self.parameters['adaptive'])
            msg = 'self.binarized created by adaptive : %s'\
                % str(self.parameters['adaptive'])
        self.logger.debug(msg)

        if flag == "small":
            self.small_binarized = binarized_img
        else:
            self.binarized = binarized_img

    @deblog
    def getHoughLines(self):
        """
        small_binarized からHough lineを算出しておく
        戻り値: self.lines lineの配列
            この要素のlineは、(rho, theta). 2次元Hough space上の1点を指す
            OpenCVの戻り値は[[[0,1],[0,2],...,[]]]と外側に配列があるが、
            この関数の戻り値はそれをひとつ外して
            lineの配列としていることに注意。
            また、後々の処理の便宜のため、numpyのarrayからpythonのlistに変換し、
            theta, rhoの順に2段のkeyにもとづきsortしておく。
        """
        if 'hough' in self.parameters:
            rho, theta, minimumVote = self.parameters['hough']
            theta = np.pi / theta
        else:
            rho, theta, minimumVote = [1, np.pi / 180, 120]

        self.rho = rho
        self.theta = theta
        self.minimumVote = minimumVote

        tmplines = cv2.HoughLines(self.small_binarized,
                                  self.rho, self.theta,
                                  self.minimumVote)
        if tmplines is not None:
            self.lines = tmplines[0].tolist()
            self.lines.sort(key=lambda x: (x[1], x[0]))
        else:
            self.lines = None

    @deblog
    def changeparam(self, cnt):
        """
        HoughLinesを算出するためのparametersを10%小さくする
        これで、線分がより検出されやすくなることを期待している。
        """
        rho, theta, minimumVote = self.parameters['hough']
        minimumVote = int(0.9 * minimumVote)
        self.parameters['hough'] = [rho, theta, minimumVote]

        minval, maxval, apertureSize = self.parameters['canny']
        maxval = int(0.9 * maxval)
        self.parameters['canny'] = [minval, maxval, apertureSize]
        self.logger.debug('hough : %s' % str(self.parameters['hough']))
        self.logger.debug('canny : %s' % str(self.parameters['canny']))

    @deblog
    def enoughLines(self):
        """
        self.lines に十分な直線データがあるかどうか判定する
        戻り値: boolean : あれば True, なければ False
        """
        komanumstr = self.p['koma']['komaIdStr']
        self.logger.info('# of self.lines in %s : %s' %
                         (komanumstr, len(self.lines)))
        if len(self.lines) < 5:
            """
            5本無ければただちにFalseを返す
            """
            self.logger.debug('self.lines : %s' % str(self.lines))
            self.logger.debug('enoughLines returns *False*' +
                              ' because this poor Lines')
            return False
        else:
            self.partitionLines()
            # self.linesを水平線と垂直線とに分類し
            if len(self.horizLines) < 2 or len(self.vertLines) < 3:
                # 水平線が2本無ければ、あるいは垂直線が3本無ければFalseを返す
                self.logger.debug(
                    'self.horizLines : %s' % str(self.horizLines))
                self.logger.debug('self.vertLines : %s' % str(self.vertLines))
                self.logger.debug('enoughLines returns *False*' +
                                  ' because this poor (horiz|vert)Lines')
                return False
            else:
                self.makeSmallZone()
                for d in ['upper', 'lower', 'center', 'right', 'left']:
                    if not self.lineSeemsToExistInSmallZone(d):
                        self.logger.debug('enoughLines returns *False*' +
                                          ' because %s has 0 candidates' % d)
                        return False
        return True

    @deblog
    def partitionLines(self):
        """
        self.linesを水平線と垂直線とに分類
        副作用: 垂直線 self.vertLines
                水平線 self.horizLines
                の両property を設定
        """
        self.horizLines = filter(isHorizontal, self.lines)
        self.vertLines = filter(isVertical, self.lines)

    @deblog
    def makeSmallZone(self, levels=None):
        """
        small_zone
        とは、small_imgのある領域。周縁部。
        cornerLinesは経験上画像の4辺から、この程度離れたところに存在している
        はずであることから決め打ちしている。
        それを表現する数値のリスト。
        HoughLinesを取得したあと、cornerLinesを絞りこむために利用する。

        入力: dict : levels : default では、以下に定義した値。
                     KnParamにより外部から指定することもできる。
        副作用: self.small_zone を設定

        例
        'lower':  [0.9, 0.97]
        とは、画像の下辺を数字で表現している。
        画像の上端から90%から97%の領域という意味。

        なぜこんな数字か。
        画像の縁3%ぐらいにはノイズが多いので,それを無視したいから
        """
        if levels is None:
            if 'small_zone_levels' in self.parameters:
                levels = self.parameters['small_zone_levels']
            else:
                levels = {'upper':  [0.03, 0.1],
                          'lower':  [0.9, 0.97],
                          'center': [0.45, 0.55],
                          'left':   [0.03, 0.1],
                          'right':  [0.9, 0.97]}

        self.small_zone = {}
        for d in ['upper', 'lower']:
            self.small_zone[d] = [self.small_height * x for x in levels[d]]
        for d in ['center', 'left', 'right']:
            self.small_zone[d] = [self.small_width * x for x in levels[d]]

    @deblog
    def lineSeemsToExistInSmallZone(self, direction, umpire=None):
        """
        入力: string : direction
        を示す文字列。すなわち、
        'upper', 'lower', 'center', 'left', 'right'
        のいずれか。
        small_zone のうち、引数に指定したところに線分があるかどうかを報告する
        戻り値: boolean : あればTrue, 無ければFalse
        """
        if umpire is not None:
            pass
        else:
            if direction in ['upper', 'lower']:
                for line in self.horizLines:
                    if self.small_zone[direction][0] < line[0] <\
                            self.small_zone[direction][1]:
                        self.candidates[direction].append(line)
            else:
                for line in self.vertLines:
                    if self.small_zone[direction][0] < line[0] <\
                            self.small_zone[direction][1]:
                        self.candidates[direction].append(line)
        self.logger.debug('direction : %s' % direction)
        self.logger.debug('candidates: %s' % str(self.candidates[direction]))
        return len(self.candidates[direction]) > 0

    @deblog
    def findCornerLines(self):
        """
        candidates から選び出す実務を担当する
        (このやりかたが最善かどうか自信はない)
        """
        self.logger.debug('cornerLine: %s' % (str(self.cornerLines)))
        for (d, w) in [('upper', 'min'), ('lower', 'max'),
                       ('left', 'min'), ('right', 'max')]:
            lines = self.candidates[d]
            if len(lines) == 0:
                # candidate が無いなら無いままにしておく
                pass
            elif len(lines) == 1:
                # candidate が1本ならそれを選ぶしかない
                self.cornerLines[d] = lines[0]
            else:
                # candidate が2本以上ならそれをソートして適当なものを選択する
                self.cornerLines[d] = self.selectLine(w, lines)
        self.logger.debug('just before exitting findCornerLines:')
        self.logger.debug('cornerLine: %s' % (str(self.cornerLines)))

    def selectLine(self, way, lines):
        """
        candidates をソートして適当なものを選択する実務を担当する
        """
        if way == 'center':
            # 画像の中心に最も近いものを選んでいる
            lines.sort(key=lambda x: abs((self.small_width / 2) - x[0]))
            return lines[0]
        else:
            # 上下左右により、最小をとるか最大をとるかが異なるので、
            # こういう処理となる
            lines.sort(key=lambda x: x[0])
            if way == 'min':
                return lines[0]
            elif way == 'max':
                return lines[-1]

    @deblog
    def isCenterAmbiguous(self):
        """
        中心線が不定かどうかを判定する
        戻り値: boolean :
        cornerLine の候補が、left, center, rightともに１本以上あれば True,
        なければ False

        候補が複数あるときは、まだ慎重に真の中心線を決めねばならぬので
        こういう判定が必要になる
        """
        return len(self.candidates['left']) > 0 and\
            len(self.candidates['center']) > 0 and\
            len(self.candidates['right']) > 0

    @deblog
    def findCenterLine(self):
        """
        isCenterAmbiguous の判定を受け、曖昧な候補から中心線を選び出す
        そのやりかたは、
            それぞれの候補に基いて左右のページを決めたとすると
            そのページの幅は一致するか？
            を観察し、最も一致しそうな組み合わせを実現するものを中心線とする
        というもの。
        書籍の左右のページの幅は等しいはずであるという経験則に頼っている。
        """
        self.logger.debug('cornerLine: %s' % (str(self.cornerLines)))
        self.logger.debug('candidate: %s' % (str(self.candidates)))
        # ページの幅の差を求める関数
        diffOfPageWidth = lambda left, center, right:\
            abs((right[0] - center[0]) - (center[0] - left[0]))
        # 各候補のすべての組み合わせを列挙し、それを幅の差でソートする
        tuplesOfVertLines =\
            sorted(itertools.product(self.candidates['left'],
                                     self.candidates['center'],
                                     self.candidates['right']),
                   key=diffOfPageWidth)
        # 差が最小のものを中心線としている
        self.cornerLines['center'] = tuplesOfVertLines[0][1]
        self.logger.debug('just before exitting findCenterLine:')
        self.logger.debug('cornerLine: %s' % (str(self.cornerLines)))

    @deblog
    def complement_corner_lines(self):
        """
        cornerLine が見つけられなかったとき、画像のサイズをもとに機械的に
        cornerLineを決定する
        """
        half_pi = 1.5707963705062866
        self.logger.debug('cornerLine: %s' % (str(self.cornerLines)))
        self.findCornerLines()
        for d in ['upper', 'lower', 'left', 'right']:
            if (not (d in self.candidates)) or (len(self.candidates[d]) == 0):
                if d == 'upper':
                    self.cornerLines[d] = [
                        int(self.small_height * 0.15), half_pi]
                elif d == 'lower':
                    self.cornerLines[d] = [
                        int(self.small_height * 0.9), half_pi]
                elif d == 'left':
                    self.cornerLines[d] = [int(self.small_width * 0.1), 0]
                elif d == 'right':
                    self.cornerLines[d] = [int(self.small_width * 0.9), 0]
        self.cornerLines['center'] = [int(self.small_width * 0.5), 0]
        self.logger.debug('just before exitting complement_corner_lines:')
        self.logger.debug('cornerLine: %s' % (str(self.cornerLines)))
        self.complemented = True

    @deblog
    def getHoughLinesP(self):
        self.linesP = cv2.HoughLinesP(self.small_binarized,
                                      self.rho, self.theta, self.minimumVote)

    @deblog
    def get_small_img_with_lines(self):
        self.small_img_with_lines = self.small_img.copy()
        self.getLinePoints()
        for line in self.linePoints:
            cv2.line(self.small_img_with_lines,
                     line[0], line[1], (0, 0, 255), 2)

    @deblog
    def get_small_img_with_linesP(self):
        self.small_img_with_linesP = self.small_img.copy()
        if not hasattr(self, 'linesP'):
            self.getHoughLinesP()
        for line in self.linesP[0]:
            pt1 = tuple(line[:2])
            pt2 = tuple(line[-2:])
            cv2.line(self.small_img_with_linesP,
                     pt1, pt2, (0, 0, 255), 2)

    @deblog
    def write_linesP_to_file(self, outdir=None):
        if not hasattr(self, 'linesP'):
            self.getHoughLinesP()
        outfilename = mkFilename(self.tgtObj,
                                 '_linesP_data', outdir, ext='.txt')
        with open(outfilename, 'w') as f:
            f.write("stat\n")
            f.write("linesP\n")
            f.write("len of linesP[0] : " + str(len(self.linesP[0])) + "\n")
            f.write("\nlen of linesP: "
                    + str(len(self.linesP)) + "\n")
            f.write("[x1,y1,x2,y2]\n")
            for line in self.linesP:
                f.writelines(str(line))
                f.write("\n")

    @deblog
    def write_lines_to_file(self, outdir=None):
        if not hasattr(self, 'lines'):
            self.getHoughLines()
            if self.lines is None:
                return False
        if not hasattr(self, 'linePoints'):
            self.getLinePoints()
        outfilename = mkFilename(self.tgtObj,
                                 '_lines_data', outdir, ext='.txt')
        with open(outfilename, 'w') as f:
            f.write("stat\n")
            f.write("len of lines : " + str(len(self.lines)) + "\n")
            f.write("lines\n")
            f.write("[rho,  theta]\n")
            for line in self.lines:
                f.writelines(str(line))
                f.write("\n")
            f.write("\nlen of linePoints : "
                    + str(len(self.linePoints)) + "\n")
            f.write("linePoints\n")
            f.write("[(x1,y1), (x2,y2)]\n")
            for line in self.linePoints:
                f.writelines(str(line))
                f.write("\n")

    @deblog
    def getLinePoints(self):
        """
        HoughLinesで取得するlines は
            [[rho, theta],...]
        と表現される。 それを
            [[(x1,y1), (x2,y2)],...]
        に変換する
        """
        self.linePoints = []
        for rho, theta in self.lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            self.linePoints.append([(x1, y1), (x2, y2)])
        self.logger.debug('linePoints: # : %d' % len(self.linePoints))
        self.logger.debug('linePoints: %s' % str(self.linePoints))

    @deblog
    def getContours(self, thresh_low=50, thresh_high=255):
        """
        contourの配列を返す
        """
        self.contours, self.hierarchy =\
            cv2.findContours(self.binarized,
                             cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    @deblog
    def getBinarized(self):
        """
        binarize された配列を self.binarized にセットする
        parameters必須。
        """
        if 'threshold' in self.parameters:
            thresh_low, thresh_high, typeval = self.parameters['threshold']
            ret, self.binarized =\
                cv2.threshold(self.gray, thresh_low, thresh_high, typeval)
            self.logger.debug('self.binarized created by threshold : %s',
                              str(self.parameters['threshold']))
        elif 'canny' in self.parameters:
            minval, maxval, apertureSize = self.parameters['canny']
            self.binarized = cv2.Canny(self.gray, minval, maxval, apertureSize)
            self.logger.debug('self.binarized created by canny : %s',
                              str(self.parameters['canny']))
        elif 'adaptive' in self.parameters:
            self.binarized =\
                cv2.adaptiveThreshold(self.gray,
                                      self.parameters['adaptive'])
            self.logger.debug('self.binarized created by adaptive : %s',
                              str(self.parameters['adaptive']))


class Timer(object):
    def __init__(self, verbose=False):
        self.__verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.__verbose:
            print('elapsed time: %f ms' % self.msecs)
