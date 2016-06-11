# -*- coding: utf-8 -*-
import sys
import numpy as np
from scipy import ndimage, stats
import cv2
import json
import os.path
from operator import itemgetter
from functools import reduce
import classes.knchar as kc
import classes.knutil as ku
import classes.boxtools as bt
#from operator import itemgetter, attrgetter


class KnPageException(Exception):
    def __init__(self, value):
        if value is None:
            self.initException()
        else:
            self.value = value

    def __str__(self):
        return repr(self.value)

    def printException(self):
        print("KnPage Exception.")

    @classmethod
    def paramsFileNotFound(self, value):
        print(('%s not found.' % value))

    @classmethod
    def initException(self):
        print("parameter file name must be specified.")


class KnPageParamsException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class KnPage:
    """
    属性
    parameters
    img
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
    collect_boxes
    collected_boxes
    contours
    depth
    val
    """
    def __init__(self, fname=None, datadir=None, params=None, outdir=None):
        if params is None:
            raise KnPageException('params is None')
        # self.parameters = params
        self.read_parameter(params)
        self.imgfullpath = params.get_imgFullPath()
        if params['page']['imgfname'] and params['page']['pagedir']:
            self.imgfullpath = "/".join([params['param']['outdir'], params['koma']['komadir'], params['page']['pagedir'], params['page']['imgfname']])
        self.get_img()

    def read_params(self, params):
        with open(params) as f:
            lines = f.readlines()
        self.parameters = json.loads(''.join(lines))
        try:
            self.imgfname = self.parameters['imgfname']
            self.outdir = self.parameters['outdir']
            self.paramfname = self.parameters['paramfname']
        except KeyError as e:
            msg = 'key : %s must be in parameter file' % str(e)
            print(msg)
            raise KnPageParamsException(msg)
        self.outfilename = self.parameters['outfilename']

    def read_parameter(self, param):
        self.p = param
        self.parameters = param['page']
        if "mavstd" in self.parameters:
            self.mavstd = self.parameters['mavstd']
        else:
            self.mavstd = 10
        if "pgmgn" in self.parameters:
            self.pgmgn_x, self.pgmgn_y = self.parameters['pgmgn']
        else:
            self.pgmgn_x, self.pgmgn_y = [0.05, 0.05]
        # collectされたのに小さすぎるのはなにかの間違いとして排除
        #  mcbs : minimum collected box size
        if 'mcbs' in self.p['page']:
            self.mcbs = self.p['page']['mcbs']
        else:
            self.mcbs = 10
        self.pagedir = "/".join([self.p['param']['topdir'],
                                 self.p['book']['bookdir'],
                                 self.p['koma']['komadir'],
                                 self.p['page']['pagedir']])
        if "collected_box_min_size" in self.parameters:
            self.cb_min = self.parameters['collected_box_min_size']
        else:
            self.cb_min = 10
        if "collected_box_max_size" in self.parameters:
            self.cb_max = self.parameters['collected_box_max_size']
        else:
            self.cb_max = 100

    def get_img(self):
        if os.path.exists(self.imgfullpath):
            self.img = cv2.imread(self.imgfullpath)
            if self.img is None:
                raise KnPageException(self.imgfullpath + 'cannot be read')
            else:
                self.height, self.width, self.depth = self.img.shape
                self.centroids = []
                self.boxes = []
                self.candidates = {'upper': [], 'lower': [],
                                   'center': [], 'left': [], 'right': []}
                self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
                self.clear_noise()
                self.getBinarized()
        else:
            raise KnPageException('%s not found' % self.imgfullpath)

    def write(self, outfilename=None, om=None):
        if om is None:
            om = self.img
        #if hasattr(self, 'outfilename'):
        #    outfilename = self.outfilename
        #if outfilename:
        cv2.imwrite(outfilename, om)
        #else:
        #  raise

    def update(self, v):
        self.val = v

    def getCentroids(self, box_min=16, box_max=48):
        """
        contoursの重心（計算の簡便のためにcontourの外接方形の重心で代用）のリスト
        ただし、すべてのcontoursをカバーしていない
        このリストからは小さすぎるboxと大きすぎるboxは排除している
        :param box_min:　リストに含めるboxのサイズの下限
        :param box_max:　リストに含めるboxのサイズの上限
        :return:　戻り値はないが、このメソッドにより、self.centroids　が内容を持つ
        """
        if not hasattr(self, 'contours'):
            self.getContours()

        if hasattr(self, 'parameters'):
            # if self.parameters.has_key('boundingRect'):
            if 'boundingRect' in self.parameters:
                box_min, box_max = self.parameters['boundingRect']

        for cnt in self.contours:
            box = cv2.boundingRect(cnt)
            self.boxes.append(box)
            x, y, w, h = box
            if (int(w) in range(box_min, box_max)) or\
               (int(h) in range(box_min, box_max)):
                self.centroids.append((x + w / 2, y + h / 2))

    def getBinarized(self):
        """
        binarize された配列を self.binarized にセットする
        parameters必須。
        """
        if 'threshold' in self.p['koma']:
            thresh_low, thresh_high, typeval = self.p['koma']['threshold']
            ret, self.binarized =\
                cv2.threshold(self.gray, thresh_low, thresh_high, typeval)
        elif 'canny' in self.p['koma']:
            minval, maxval, apertureSize = self.p['koma']['canny'][0]
            self.binarized = cv2.Canny(self.gray, minval, maxval, apertureSize)
        elif 'adaptive' in self.p['koma']:
            self.binarized =\
                cv2.adaptiveThreshold(self.gray,
                                      self.p['koma']['adaptive'])

    def getGradients(self):
        """
        self.img のgradients を self.gradients_* にセットする
        parameters必須。
        """
        if 'sobel' in self.parameters:
            ddepth, dx, dy, ksize = self.parameters['sobel']
            self.gradients_sobel = cv2.Sobel(self.gray, ddepth, dx, dy, ksize)
        if 'scharr' in self.parameters:
            ddepth, dx, dy = self.parameters['scharr']
            self.gradients_scharr = cv2.Scharr(self.gray, ddepth, dx, dy)
        if 'laplacian' in self.parameters:
            ddepth = self.parameters['laplacian'][0]
            self.gradients_laplacian = cv2.Laplacian(self.gray, ddepth)

    def getContours(self, thresh_low=50, thresh_high=255):
        """
        contourの配列を返す
        """
        self.img2, self.contours, self.hierarchy =\
            cv2.findContours(self.binarized,
                             cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    def get_small_img_with_lines(self):
        self.small_img_with_lines = self.small_img.copy()
        self.getLinePoints()
        for line in self.linePoints:
            cv2.line(self.small_img_with_lines,
                     line[0], line[1], (0, 0, 255), 2)

    def get_small_img_with_linesP(self):
        self.small_img_with_linesP = self.small_img.copy()
        for line in self.linesP[0]:
            pt1 = tuple(line[:2])
            pt2 = tuple(line[-2:])
            cv2.line(self.small_img_with_linesP,
                     pt1, pt2, (0, 0, 255), 2)

    def write_small_img(self, outdir):
        outfilename = ku.mkFilename(self, '_small_img', outdir)
        cv2.imwrite(outfilename, self.small_img)
        outfilename = ku.mkFilename(self, '_small_img_gray', outdir)
        cv2.imwrite(outfilename, self.small_img_gray)
        outfilename = ku.mkFilename(self, '_small_img_canny', outdir)
        cv2.imwrite(outfilename, self.small_img_canny)

    def write_small_img_with_lines(self, outdir):
        outfilename = ku.mkFilename(self, '_small_img_with_lines', outdir)
        cv2.imwrite(outfilename, self.small_img_with_lines)

    def write_small_img_with_linesP(self, outdir):
        outfilename = ku.mkFilename(self, '_small_img_with_linesP', outdir)
        cv2.imwrite(outfilename, self.small_img_with_linesP)


    def intersect(self, box1, box2, x_margin=None, y_margin=None):
        """
        box1 と box2 が交わるか接するならtrueを返す。
        marginを指定することですこし離れていても接すると判定.
        """
        if 'ismgn' in self.parameters:
            xm, ym = self.parameters['ismgn']
        else:
            xm, ym = (20, 8)  # default

        if x_margin is not None:
            xm = x_margin
        if y_margin is not None:
            ym = y_margin
        ax1, ay1, w1, h1 = box1
        ax2 = ax1 + w1
        ay2 = ay1 + h1
        bx1, by1, w2, h2 = box2
        bx2 = bx1 + w2
        by2 = by1 + h2

        if self.h_apart(ax1, ax2, bx1, bx2, xm):
            return False
        elif self.v_apart(ay1, ay2, by1, by2, ym):
            return False
        else:
            return True

    def h_apart(self, ax1, ax2, bx1, bx2, xm):
        return ax2 < (bx1 - xm) or (bx2 + xm) < ax1

    def v_apart(self, ay1, ay2, by1, by2, ym):
        return ay2 < (by1 - ym) or (by2 + ym) < ay1

    def sweep_included_boxes(self, boxes=None):
        """
        他のboxに完全に包含されるboxをリストから排除する
        """
        flag = False
        if boxes is None:
            self.getContours()
            if len(self.boxes) == 0:
                self.getCentroids()
            boxes = self.boxes
            flag = True

        # w, h どちらかが200以上のboxは排除
        # boxes = [x for x in boxes if (x[2] < 200) and (x[3] < 200)]

        temp_boxes = []
        while len(boxes) > 0:
            abox = boxes.pop()
            boxes = [x for x in boxes if not bt.include(abox, x)]
            temp_boxes = [x for x in temp_boxes if not bt.include(abox, x)]
            temp_boxes.append(abox)

        if flag:
            self.boxes = temp_boxes
        return temp_boxes

    def flatten(self, i):
        return reduce(
            lambda a, b: a + (self.flatten(b) if hasattr(b, '__iter__')
                              else [b]),
            i, [])

    def show_message(f):
        def wrapper():
            print("function called")
            return f()
        return wrapper

    def get_adj_boxes(self, boxes, abox):
        """
        隣接するboxのリストを返す
        入力：
        boxes : boxのリスト。探索対象。
        abox : あるbox。探索の起点。このboxに隣接するboxからなるリストを返す。
        ここで隣接するとは、直接aboxに隣接するもの,
                            間接的にaboxに隣接するもの(隣の隣もそのまた隣もみな隣とみなす。)
        をどちらも含める。
        それゆえ、linked listを再帰でたどるのと同様に、この関数も再帰を利用している。
        出力: boxのリスト
        """
        if abox in boxes:
            boxes.remove(abox)

        if len(abox) > 0:
            ret = [x for x in boxes if self.intersect(abox, x)]
        else:
            return []

        if len(ret) > 0:
            for x in ret:
                boxes.remove(x)
            if len(boxes) > 0:
                for x in ret:
                    subs = self.get_adj_boxes(boxes, x)
                    ret += subs
            else:
                return ret
            return ret
        else:
            return []

    def collect_boxes_with_debug(self):
        """
        bounding boxを包含するboxに統合し、文字を囲むboxの取得を試みる
        """

        if len(self.boxes) == 0:
            self.getCentroids()

        # w, h どちらかが200以上のboxは排除
        self.boxes = [x for x in self.boxes if (x[2] < 200) and (x[3] < 200)]

        self.collected_boxes = []
        adjs = []

        #f = open('while_process.txt', 'w')    # for debug
        while len(self.boxes) > 0:
            # for debug
            #f.write('len of self.boxes : ' + str(len(self.boxes)) + "\n")
            abox = self.boxes.pop()
            #f.write('abox : ' + str(abox) + "\n")    # for debug
            adjs = self.get_adj_boxes(self.boxes, abox)
            #f.write('adjs : ' + str(adjs) + "\n")    # for debug
            for x in adjs:
                if x in self.boxes:
                    self.boxes.remove(x)
            #f.write('len of self.boxes after remove : '
            #        + str(len(self.boxes)) + "\n")    # for debug
            #f.write('self.boxes after remove: '
            #        + str(self.boxes) + "\n")    # for debug
            adjs.append(abox)
            #f.write('adjs after append: ' + str(adjs) + "\n")    # for debug
            if len(adjs) > 0:
                list_of_adjs = bt.recheck_adjs(adjs)
                for adj in list_of_adjs:
                    if len(adj) > 0:
                        boundingBox = bt.get_boundingBox(adj)
            #    f.write('boundingBox : '
            #            + str(boundingBox) + "\n")    # for debug
                        if self.qualify_collected_box(boundingBox):
                            self.collected_boxes.append(boundingBox)
            #    f.write('self.collected_boxes : '
            #            + str(self.collected_boxes) + "\n")    # for debug

        #f.close()    # for debug

    def collect_boxes(self):
        """
        bounding boxを包含するboxに統合し、文字を囲むboxの取得を試みる
        """

        if len(self.boxes) == 0:
            self.getCentroids()
        # self.dispose_boxes()

        # w, h どちらかが200以上のboxは排除
        #self.boxes = [x for x in self.boxes if (x[2] < 200) and (x[3] < 200)]

        self.collected_boxes = []
        adjs = []

        while len(self.boxes) > 0:
            abox = self.boxes.pop()
            adjs = self.get_adj_boxes(self.boxes, abox)
            for x in adjs:
                if x in self.boxes:
                    self.boxes.remove(x)
            adjs.append(abox)
            if len(adjs) > 0:
                list_of_adjs = bt.recheck_adjs(adjs)
                for adj in list_of_adjs:
                    if len(adj) > 0:
                        boundingBox = bt.get_boundingBox(adj)
                        if self.qualify_collected_box(boundingBox):
                            self.collected_boxes.append(boundingBox)

    def qualify_collected_box(self, box):
        """
        box がcollected_boxと認められるか判定する
        判断基準はboxの大きさ
        paramsに定義された page collected_boxに関するパラメータ
        "collected_box_min_size" : 数値 : collected_boxの面積がこれ以下のものは削除される
        "collected_box_max_size" : 数値 : collected_boxの面積がこれ以上のものは削除される

        :param box: (x, y, w, h)
        :return: boolean :
        """
        result = (self.cb_min < box[2] < self.cb_max) and\
                 (self.cb_min < box[3] < self.cb_max)
        return result

    def dispose_boxes(self, debug=False):
        """
        self.boxesから消せるものは消していく
        """
        # w, h どちらかが200以上のboxは排除
        # これはgraphの存在するページでは問題か？
        if "toobig" in self.parameters:
            toobig_w, toobig_h = self.parameters['toobig']
        else:
            toobig_w, toobig_h = [200, 200]
        self.boxes = [x for x in self.boxes
                      if (x[2] < toobig_w) and (x[3] < toobig_h)]

        self.sweep_boxes_in_page_margin()

        # 他のboxに包含されるboxは排除
        self.sweep_included_boxes()

        # 小さく、隣接するもののないboxは排除
        self.sweep_maverick_boxes()

    def get_stats(self):
        """
        求める項目
        collected_boxes関連（cb_で始まる)
        cb_num : collected_boxの個数
        cb_size_max : collected box の面積の最大値
        cb_size_min : collected box の面積の最小値
        cb_size_mean : collected box の面積の平均値

        raws_num :
        columns_num :

        :return:
        """

    @ku.deblog
    def estimate_char_size(self):
        self.logger.debug("# of collected_boxes: %d"
                          % len(self.collected_boxes))
        self.logger.debug("# of centroids: %d" % len(self.centroids))
        self.square_like_boxes = [x for x in self.collected_boxes if
                                  (x[2] * 0.8) < x[3] < (x[2] * 1.2)]
        self.logger.debug("# of square_like_boxes: %d"
                          % len(self.square_like_boxes))
        self.estimated_width = max(map(lambda x: x[2], self.square_like_boxes))
        self.estimated_height = max(map(lambda x: x[3],
                                        self.square_like_boxes))
        self.logger.debug('estimated_width: %d' % self.estimated_width)
        self.logger.debug('estimated_height: %d' % self.estimated_height)

    @ku.deblog
    def estimate_vertical_lines(self):
        """
        collected_boxesの重心をソートして、
        ｘ座標がジャンプしているところ
        (経験上、同じ行ならば20 pixel以上離れない）
        (ここは試行錯誤が必要か？ルビや句点を同じ行とするための工夫？）
        が行の切れ目だと判定し、
        collected_boxesをグループ分けする
        """
        self.centroids = map(lambda x: (x[0] + x[2] / 2, x[1] + x[3] / 2),
                             self.collected_boxes)
        self.square_centroids = map(lambda x:
                                    (x[0] + x[2] / 2, x[1] + x[3] / 2),
                                    self.square_like_boxes)
        self.logger.debug("# of square_centroids: %d"
                          % len(self.square_centroids))
        self.logger.debug("square_centroids: %s" % str(self.square_centroids))
        self.square_centroids.sort(key=itemgetter(0, 1))
        self.box_by_v_lines = {}
        self.box_by_v_lines[0] = [self.square_centroids[0]]
        line_idx = 0
        for c in self.square_centroids[1:]:
            if c[0] - self.box_by_v_lines[line_idx][-1][0] <= 20:

                self.box_by_v_lines[line_idx].append(c)
            else:
                line_idx += 1
                self.box_by_v_lines[line_idx] = [c]

        self.logger.debug('box_by_v_lines: %s' % str(self.box_by_v_lines))

    @ku.deblog
    def rotate_image(self):
        image_center = tuple(np.array(self.img.shape[0:2]) / 2)
        dsize = tuple(reversed(np.array(self.img.shape[0:2])))
        if self.estimated_angle > 0:
            degree = 180 * (np.pi / 2 -
                            np.arctan(self.estimated_angle)) / np.pi
            degree = degree * (-1.0)
        else:
            angle = (-1.0) * self.estimated_angle
            degree = 180 * (np.pi / 2 - np.arctan(angle)) / np.pi

        rot_mat = cv2.getRotationMatrix2D(image_center, degree, 1.0)
        self.rotated_img = cv2.warpAffine(self.img, rot_mat,
                                          dsize, flags=cv2.INTER_LINEAR)

    @ku.deblog
    def estimate_rotate_angle(self):
        slopes = []
        for k, v in self.box_by_v_lines.items():
            if len(v) > 10:
                xi = map(itemgetter(0), v)
                yi = map(itemgetter(1), v)
                results = stats.linregress(xi, yi)
                slopes.append(results[0])

        self.logger.debug("slopes: %s" % str(slopes))
        self.estimated_slope = np.mean(slopes)
        self.logger.debug("avg of slopes: %f" % self.estimated_slope)
        self.estimated_angle = np.arctan(self.estimated_slope)
        self.logger.debug("estimated_angle: %f" % self.estimated_angle)

    @ku.deblog
    def write_rotated_img_to_file(self, outdir=None, fix=None):
        if outdir is None:
            outdir = self.pagedir
        cv2.imwrite(ku.mkFilename(self, '_rotated%s' % fix, outdir),
                    self.rotated_img)

    def get_neighbors(self, box, x, y):
        """
        self.boxesから、boxの近所にあるboxを選びそのリストを返す
        近所とは、boxをx方向にx，y方向にy拡大した矩形と交わることとする
        """
        x0, y0, w, h = box
        x1 = max(0, x0 - x)
        y1 = max(0, y0 - y)
        w = w + 2 * x
        h = h + 2 * y
        newbox = (x1, y1, w, h)
        ret = [b for b in self.boxes if self.intersect(newbox, b, 0, 0)]
        if box in ret:
            ret.remove(box)
        return ret

    def sweep_maverick_boxes(self):
        """
        他のboxから離れて存在しているboxをself.boxesから排除する
        """
        boxes = self.boxes
        for box in boxes:
            neighbors = self.get_neighbors(box, 10, 20)
            # self.logger.debug('box: %s' % str(box))
            # self.logger.debug('# of neighbors: %d' % len(neighbors))
            if len(neighbors) == 0:
                self.boxes.remove(box)

    def flatten(self, i):
        return reduce(
            lambda a, b: a + (self.flatten(b) if hasattr(b, '__iter__')
                              else [b]),
            i, [])

    def show_message(f):
        def wrapper():
            print("function called")
            return f()
        return wrapper


    @ku.deblog
    def getBoxesAndCentroids(self, box_min=16, box_max=48):
        if not hasattr(self, 'contours'):
            self.getContours()

        if hasattr(self, 'parameters'):
            # if self.parameters.has_key('boundingRect'):
            if 'boundingRect' in self.parameters:
                box_min, box_max = self.parameters['boundingRect']

        for cnt in self.contours:
            box = cv2.boundingRect(cnt)
            self.boxes.append(box)
            x, y, w, h = box
            if (int(w) in range(box_min, box_max)) or \
                    (int(h) in range(box_min, box_max)):
                self.centroids.append((x + w / 2, y + h / 2))


    def in_margin(self, box, le, ri, up, lo):
        x1, y1 = box[0:2]
        x2, y2 = map(sum, zip(box[0:2], box[2:4]))
        return (y2 < up) or (y1 > lo) or (x2 < le) or (x1 > ri)


    def sweep_boxes_in_page_margin(self, mgn=None):
        """
        pageの余白に存在すると思われるboxは排除
        box: [x, y, w, h]
        """

        if mgn:
            self.pgmgn_x, self.pgmgn_y = mgn
        else:
            self.pgmgn_x, self.pgmgn_y = self.parameters['pgmgn']



        left_mgn = self.width * self.pgmgn_x
        right_mgn = self.width * (1 - self.pgmgn_x)
        upper_mgn = self.height * self.pgmgn_y
        lower_mgn = self.height * (1 - self.pgmgn_y)

        self.boxes = [x for x in self.boxes
                      if not self.in_margin(x, left_mgn,
                                            right_mgn, upper_mgn, lower_mgn)]

    def sort_boxes(self):
        """
        x_sorted : xの昇順、yの昇順に並べる
        y_sorted : yの昇順、xの昇順に並べる
        """
        if not hasattr(self, 'boxes'):
            self.getContours()
            if len(self.boxes) == 0:
                self.getBoxesAndCentroids()
        self.x_sorted_boxes = sorted(self.boxes, key=itemgetter(0, 1))
        self.y_sorted_boxes = sorted(self.boxes, key=itemgetter(1, 0))

    def clear_noise(self, max=None, under=None, type=None):
        """
        行間のノイズを消す
        self.img を白黒反転し、黒地に白い文字が浮かぶ画像となる self.bw_not をつくり、さらに
        しきい値処理して 55 より暗い点は全て０に、つまり真っ黒にする
        その画像を self.bw_not_tozero とする
        parameter の意味は　OpenCV threshold の文書を参照
        :param max:  Integer 0 - 255 : threshold max_value
        :param under: Integer 0 - 255 : threshold thresh_under
        :param type: Integer 0 - 5 : cv2.THRESH_xxx に相当する数値
        :return:
        """
        self.bw_not = cv2.bitwise_not(self.gray)
        if 'threshold' in self.parameters:
            max_value, thresh_under, thresh_type = self.parameters['threshold']
        else:
            max_value = max or 255
            thresh_under = under or 55
            thresh_type = type or cv2.THRESH_TOZERO
        ret, self.bw_not_tozero = cv2.threshold(self.bw_not, thresh_under, max_value, thresh_type)

    def get_line_imgs(self):
        """
        ページ画像を行ごとに分割する
        縦書き前提
        :return:
        """
        if not hasattr(self, 'bw_not_tozero'):
            self.clear_noise()
        self.hist_0 = np.sum(self.bw_not_tozero, axis=0)
        lines = ku.get_range_list(self.hist_0, 300)
        # 上のやりかただと、noise も行と認識してしまっているので
        # 幅の狭すぎる行と、ページの両端にかかるものは削除する
        lines = list(filter(lambda l: l[1] - l[0] > 10, lines))
        self.lines = list(filter(lambda l: l[0] is not 0, lines))
        # 上の結果に従って画像を行ごとに分割してみる
        self.line_imgs = []
        for i, gyou in enumerate(self.lines):
            self.line_imgs.append(self.bw_not_tozero[:, gyou[0]:gyou[1]])
        # 縦書き前提
        self.line_imgs.reverse()

    def get_hist_vs(self):
        """
        行画像の水平方向のヒストグラムを算出する
        :return:
        """
        if not hasattr(self, 'line_imgs'):
            self.get_line_imgs()
        self.hist_vs = []
        for i, line_img in enumerate(self.line_imgs):
            self.hist_vs.append(np.sum(line_img, axis=1))

    def get_chars(self, minimum=3):
        """
        行画像を文字画像に分割し,KnChar オブジェクトを得る
        :param minimum : char img の最低限の高さ
        :return:
        """
        if not hasattr(self, 'hist_vs'):
            self.get_hist_vs()
        self.chars = []
        for i, hist_v in enumerate(self.hist_vs):
            hist = ku.get_range_list(hist_v, 10)
            self.chars.append([])
            for j, char in enumerate(hist):
                if char[1] - char[0] >= minimum:
                    char_img = self.line_imgs[i][char[0]:char[1]+1, :]
                    self.chars[i].append(kc.KnChar(char_img, i, j))

    def estimate_upper_mergin(self):
        """
        縦書きを前提
        行の上の余白の広さを決めてしまう
        :return: integer : page 上部の余白の広さ
        """
        if not hasattr(self, 'hist_vs'):
            self.get_hist_vs()



    def get_x_zero(self, img):
        """
        img のヒストグラムをとり
        値が０である要素数を返す
        :param img : np.ndarray :
        :return: integer : 0の数
        """
        hist = np.sum(img, axis=0)
        return np.count_nonzero(hist == 0)

    def one_degree_plus(self, degree):
        """
        self.bw_not_tozero を degree + 1 度 回転した画像の
        x数を返す
        :param degree:
        :return:  integer : 0の数
        """
        r_img = ndimage.rotate(self.bw_not_tozero, degree + 1)
        return self.get_x_zero(r_img)

    def adjust_rotation(self):
        """
        ページ画像の回転を修正する角度を求める
        :return: float : 角度(in degree)
        """
        if not hasattr(self, 'bw_not_tozero'):
            self.clear_noise()
        x = self.get_x_zero(self.bw_not_tozero)
        tmp_degree = 0
        x_1 = self.one_degree_plus(tmp_degree)
        tmp_degree = tmp_degree + 1
        while x_1 > x:
            x = x_1
            x_1 = self.one_degree_plus(tmp_degree)
            tmp_degree = tmp_degree + 1

        result = self.get_tenth(tmp_degree)
        return result

    def get_tenth(self, degree):
        """
        degree より小刻みに小さい角度で回転して,最も成績の良い角度を返す
        :param degree:
        :return:
        """
        degrees = [degree - 0.2, degree - 0.4, degree - 0.6, degree - 0.8]
        xes = [self.get_x_zero(ndimage.rotate(self.bw_not_tozero, degree)) for degree in degrees]
        return degrees[xes.index(max(xes))]