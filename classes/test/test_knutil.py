# -*- coding: utf-8 -*-
import knutil as ku
import knpage as kp
# import classes.knkoma as kk
import pytest


from classes.knutil import DATA_DIR
img_fname = DATA_DIR + '/twletters.jpg'


class TestCheckTestEnvironment:
    def test_check_test_environment1(self, b1g101):
        assert b1g101 is not None

    def test_check_test_environment2(self, b1g102):
        assert b1g102 is not None

class TestFileName:
    def test_mkFilename(self, knp):
        knp.set_logger("_mkFilename")
        knpage = kp.KnPage(params=knp)
        name = ku.mkFilename(knpage, '_cont')
        expect = DATA_DIR + '/1091460/k001/001_cont.jpeg'
        assert name == expect


class TestTmpDir:
    def test_write(self, tmpdir):
        dataDirectory = tmpdir.mkdir('data')
        sampleFile = str(dataDirectory.join("sample.jpeg"))
        # ku.write(sampleFile)
        assert 'sample.jpeg' in sampleFile
        assert sampleFile != '/tmp/pytest-skkmania/data/sample.jpeg'


class TestCompLine:
    @pytest.mark.parametrize("line0,line1,horv", [
        ([(4, 4), (8,  4)], [(1, 3), (7, 5)], 'h'),
        ([(4, 24), (8,  4)], [(0, 15), (0, 20)], 'h'),
        ([(10, 100), (20, 10)], [(100, 50), (1000, 5)], 'v'),
        ([(10, 1000), (50, 100)], [(7, 25), (35, 20)], 'v')
    ])
    def test_complLne_wrong_recognition(self, line0, line1, horv):
        with pytest.raises(TypeError) as e:
            ku.compLine(line0, line1, horv)
            assert 'wrong recognition' in str(e)
            ku.compLine(line1, line0, horv)
            assert 'wrong recognition' in str(e)

    @pytest.mark.parametrize("line0, line1, horv", [
        ([(4, 4), (8,  4)], [(1, 3), (7, 3)], 'h'),
        ([(4, 4), (8,  4)], [(10, 0), (20, 0)], 'h'),
        ([(10, 10), (20, 10)], [(100, 5), (1000, 5)], 'h'),
        ([(10, 100), (50, 100)], [(7, 20), (35, 20)], 'h')
    ])
    def test_complLne_horizontal(self, line0, line1, horv):
        result = ku.compLine(line0, line1, horv)
        assert result == "upper"
        result = ku.compLine(line1, line0, horv)
        assert result == "lower"

    @pytest.mark.parametrize("line0, line1, horv", [
        ([(4, 4), (8,  4)], [(1, 3), (7, 3)], 'v'),
        ([(4, 4), (8,  4)], [(10, 0), (20, 0)], 'v'),
        ([(10, 10), (20, 10)], [(100, 5), (1000, 5)], 'v'),
        ([(10, 100), (50, 100)], [(7, 20), (35, 20)], 'v')
    ])
    def test_complLne_vertical(self, line0, line1, horv):
        tr = lambda tup: (tup[1], tup[0])
        result = ku.compLine(list(map(tr, line0)), list(map(tr, line1)), horv)
        assert result == "right"
        result = ku.compLine(list(map(tr, line1)), list(map(tr, line0)), horv)
        assert result == "left"


class TestParamsGenerator:
    def test_params_generator(self, tmpdir):
        SRC_SAMPLE = {
            "scale_size": [640.0, 480.0, 320.0],
            "boundingRect": [[16, 32]],
            "imgfname": [DATA_DIR + '/007.jpeg', DATA_DIR + '/008.jpeg'],
            "mode": ["EXTERNAL"],
            "canny": [[50, 200, 3]],
            "hough": [[1, 180, 200]],
            "method": ["NONE"],
            "outdir": [DATA_DIR]
        }
        src = SRC_SAMPLE
        result = ku.params_generator(src)
        wrap = lambda d: d.get("scale_size")
#        assert set(map(wrap, result)) == set(src["scale_size"])
        dataDirectory = tmpdir.mkdir('data')
        sampleFile = dataDirectory.join("result.txt")
        with open(str(sampleFile), 'w') as f:
            f.write(str(result))
        ku.print_params_files(result)


class TestInterSection:
    slide = lambda self, x, y, line:\
        [(line[0][0] + x, line[0][1] + y),
         (line[1][0] + x, line[1][1] + y)]

    @pytest.mark.parametrize("line", [
        [(10, 10), (10, 3)],
        [(10, 10), (10, 7)],
        [(10, 10), (10, 8)],
        [(10, 10), (10, 53)]
    ])
    def test_isVertical_truth(self, line):
        result = ku.isVertical(line)
        assert result is True

    @pytest.mark.parametrize("line", [
        [(10, 10), (0, 3)],
        [(10, 10), (30, 7)],
        [(10, 10), (20, 8)],
        [(10, 10), (50, 53)]
    ])
    def test_isVertical_false(self, line):
        result = ku.isVertical(line)
        assert result is False

    @pytest.mark.parametrize("line", [
        [(10, 3), (100, 3)],
        [(10, 7), (30, 7)],
        [(10, 8), (20, 8)],
        [(10, 53), (50, 53)]
    ])
    def test_isHorizontal_truth(self, line):
        result = ku.isHorizontal(line)
        assert result is True

    @pytest.mark.parametrize("line", [
        [(10, 10), (0, 3)],
        [(10, 10), (30, 7)],
        [(10, 10), (20, 8)],
        [(10, 10), (50, 53)]
    ])
    def test_isHorizontal_false(self, line):
        result = ku.isHorizontal(line)
        assert result is False

    @pytest.mark.parametrize("line1, line2", [
        ([(10, 10), (0,  3)], [(10, 190), (0, 3)]),
        ([(10, 10), (30,  7)], [(10, 10), (0, 3)]),
        ([(10, 10), (20,  8)], [(10, 10), (0, 3)]),
        ([(10, 10), (50, 53)], [(20, 20), (5, 2)])
    ])
    def test_isHorizontal_false2(self, line1, line2):
        result = ku.getIntersection(line1, line2)
        assert result is not False

    @pytest.mark.parametrize("line1,line2", [
        ([(4, 2), (8, 4)], [(1, 3), (2, 6)]),
        ([(4, 2), (8, 4)], [(10, 10), (20, 20)]),
        ([(10, 10), (20, 20)], [(100, 10), (1000, 100)]),
        ([(10, 10), (50, 50)], [(7, 2), (35, 10)])
    ])
    def test_getIntersection(self, line1, line2):
        result = ku.getIntersection(line1, line2)
        assert result == (0, 0)
        l1 = self.slide(5, 8, line1)
        l2 = self.slide(5, 8, line2)
        result = ku.getIntersection(l1, l2)
        assert result == (5, 8)

    @pytest.mark.parametrize("line1,line2", [
        ([(4, 2), (4, 4)], [(10, 5), (8, 4)]),
        ([(4, 2), (4, 4)], [(5, 2), (8, 2)]),
        ([(4, 2), (4, 4)], [(0, 4), (8, 0)])
    ])
    def test_getIntersection_with_vertical(self, line1, line2):
        result = ku.getIntersection(line1, line2)
        assert result == (4, 2)
        l1 = self.slide(5, 8, line1)
        l2 = self.slide(5, 8, line2)
        result = ku.getIntersection(l1, l2)
        assert result == (9, 10)
