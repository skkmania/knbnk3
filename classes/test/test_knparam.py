# -*- coding: utf-8 -*-
import classes.knparam as kr
import pytest

from classes.knutil import DATA_DIR


class TestNew:
    def test_initialize_without_params(self):
        with pytest.raises(kr.KnParamParamsException) as e:
            kr.KnParam()
        assert 'must be specified' in str(e)

    def test_initialize_with_fname_not_existed(self):
        with pytest.raises(kr.KnParamParamsException) as e:
        #with pytest.raises(KnParamException) as e:
            kr.KnParam(param_fname='not_exist_file')
        assert 'not_exist_file' in str(e)

    def test_initialize_with_syntax_error_param_file(self):
        with pytest.raises(ValueError) as e:
            kr.KnParam(param_fname=DATA_DIR + '/params_with_syntax_error.json')
        assert 'Expecting' in str(e)

    def test_initialize_with_lack(self):
        with pytest.raises(kr.KnParamParamsException) as e:
            kr.KnParam(param_fname=DATA_DIR + '/params_lacks.json')
        assert 'lacks' in str(e)

    def test_new_from_file(self, knp):
        knp.set_logger("_new_from_file")
        knp.logger.debug(str(knp))
        assert knp['param']['topdir'] == DATA_DIR


class TestClone:
    def test_clone(self, knp):
        """
        cloneは新しい別個のobjectをつくるので
        clone先の部分的な変更は元のobjectに反映されない
        元のobjectの部分的な変更はclone先に反映されない
        """
        knp.set_logger("_clone")
        cl = knp.clone()
        assert cl['param']['balls'] == knp['param']['balls']
        cl['param']['balls'] = []
        assert cl['param']['balls'] != knp['param']['balls']

        assert cl['koma']['komaId'] == knp['koma']['komaId']
        knp['koma']['komaId'] = 3
        assert cl['koma']['komaId'] != knp['koma']['komaId']

"""
しばらくこのテストは保留する
class TestStart:
    def test_start(self, knp):
        knp.start()
        assert knp['param']['topdir'] == DATA_DIR
"""
