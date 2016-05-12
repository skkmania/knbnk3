# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
import json
import os.path
from operator import itemgetter
from functools import reduce
import classes.knutil as ku
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
        self.parameters = params
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
        self.pagedir = "/".join([self.p['param']['workdir'],
                                 self.p['book']['bookdir'],
                                 self.p['koma']['komadir'],
                                 self.p['page']['pagedir']])

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
                self.getBinarized()
        else:
            raise KnPageException('%s not found' % self.imgfname)

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
        contoursの重心のリスト
        :param box_min:
        :param box_max:
        :return:
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
        if 'threshold' in self.parameters['koma']:
            thresh_low, thresh_high, typeval = self.parameters['koma']['threshold']
            ret, self.binarized =\
                cv2.threshold(self.gray, thresh_low, thresh_high, typeval)
        elif 'canny' in self.parameters['koma']:
            minval, maxval, apertureSize = self.parameters['koma']['canny'][0]
            self.binarized = cv2.Canny(self.gray, minval, maxval, apertureSize)
        elif 'adaptive' in self.parameters['koma']:
            self.binarized =\
                cv2.adaptiveThreshold(self.gray,
                                      self.parameters['koma']['adaptive'])

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

    def include(self, box1, box2):
        """
           box1 が box2 を包含するならtrueを返す。
        """

        ax1, ay1, w1, h1 = box1
        ax2 = ax1 + w1
        ay2 = ay1 + h1
        bx1, by1, w2, h2 = box2
        bx2 = bx1 + w2
        by2 = by1 + h2

        if (ax1 <= bx1) and (bx2 <= ax2) and (ay1 <= by1) and (by2 <= ay2):
            return True
        else:
            return False

    def intersect(self, box1, box2, x_margin=None, y_margin=None):
        """
        box1 と box2 が交わるか接するならtrueを返す。
        marginを指定することですこし離れていても接すると判定.
        """
        if 'ismgn' in self.parameters['page']:
            xm, ym = self.parameters['page']['ismgn']
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

    def get_boundingBox(self, boxes):
        """
        入力のboxの形式は(x,y,w,h)
        出力のboxの形式も(x,y,w,h)
        (x,y,w,h) -> (x,y,x+w,y+h)
        """
        target = [(x, y, x + w, y + h) for (x, y, w, h) in boxes]
        x1, y1, d1, d2 = list(map(min, list(zip(*target))))
        d1, d2, x2, y2 = list(map(max, list(zip(*target))))
        # (x,y,x+w,y+h) -> (x,y,x,y)
        return (x1, y1, x2 - x1, y2 - y1)

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
        boxes = [x for x in boxes if (x[2] < 200) and (x[3] < 200)]

        temp_boxes = []
        while len(boxes) > 0:
            abox = boxes.pop()
            boxes = [x for x in boxes if not self.include(abox, x)]
            temp_boxes = [x for x in temp_boxes if not self.include(abox, x)]
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

    def write_self_boxes_to_file(self, outdir):
        with open(ku.mkFilename(self, '_self_boxes', outdir, '.txt'), 'w') as f:
            f.write("self.boxes\n")
            for box in self.boxes:
                f.write(str(box) + "\n")
            f.write("\n")

    def collect_boxes(self):
        """
        bounding boxを包含するboxに統合し、文字を囲むboxの取得を試みる
        """

        if len(self.boxes) == 0:
            self.getCentroids()

        # w, h どちらかが200以上のboxは排除
        self.boxes = [x for x in self.boxes if (x[2] < 200) and (x[3] < 200)]

        # self.write_self_boxes_to_file('')    # for debug

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
                boundingBox = self.get_boundingBox(adjs)
            #    f.write('boundingBox : '
            #            + str(boundingBox) + "\n")    # for debug
                self.collected_boxes.append(boundingBox)
            #    f.write('self.collected_boxes : '
            #            + str(self.collected_boxes) + "\n")    # for debug

        #f.close()    # for debug

    def dispose_boxes(self, debug=False):
        """
        self.boxesから消せるものは消していく
        """
        # w, h どちらかが200以上のboxは排除
        # これはgraphの存在するページでは問題か？
        if "toobig" in self.parameters["page"]:
            toobig_w, toobig_h = self.parameters['page']['toobig']
        else:
            toobig_w, toobig_h = [200, 200]
        self.boxes = [x for x in self.boxes
                      if (x[2] < toobig_w) and (x[3] < toobig_h)]

        self.sweep_boxes_in_page_margin()

        # 他のboxに包含されるboxは排除
        self.sweep_included_boxes()

        # 小さく、隣接するもののないboxは排除
        self.sweep_maverick_boxes()

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


    def sweep_maverick_boxes(self):
        """
        他のboxから離れて存在しているboxをself.boxesから排除する
        """
        boxes = self.boxes
        for box in boxes:
            neighbors = self.get_neighbors(box, 10, 20)
            self.logger.debug('box: %s' % str(box))
            self.logger.debug('# of neighbors: %d' % len(neighbors))
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
            self.pgmgn_x, self.pgmgn_y = self.parameters['page']['pgmgn']



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