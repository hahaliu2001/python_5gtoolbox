# -*- coding: utf-8 -*-
import pytest
from scipy import io
import os
import numpy as np
import math
from zipfile import ZipFile 
import json

from tests.nr_pucch import nr_pucch_format0_testvectors
from py5gphy.nr_pucch import nr_pucch_format0
from py5gphy.common import nr_slot
from py5gphy.nr_waveform import nr_ul_waveform

def get_testvectors():
    path = "tests/nr_pucch/testvectors_format0"
    if len(os.listdir(path)) < 100: #desn't unzip testvectors
        zipfile_lists = []
        for f in os.listdir(path):
            if f.endswith(".zip"):
                zipfile_lists.append(path + '/' + f)

        for zipfile in zipfile_lists:
            zObject = ZipFile(zipfile, 'r')
            zObject.extractall(path)

    file_lists = []
    for f in os.listdir(path):
        if f.endswith(".mat") and f.startswith("nr_pucchformat0_testvec"):
        #if f.endswith(".mat") and f.startswith("nrSRS_testvec_1_scs_15khz_BW_20_NumSRSPorts_1_CSRS_0_BSRS_0"):
            file_lists.append(path + '/' + f)
                        
    return file_lists

@pytest.mark.parametrize('filename', get_testvectors())
def test_nr_pucchformat0_only(filename):
    #read data
    matfile = io.loadmat(filename)
    codeword_out, symbols_out, ref_fd_slot_data, pucchformat0_tvcfg = \
        nr_pucch_format0_testvectors.read_pucch_format0_testvec_matfile(matfile)
    carrier_config, pucch_format0_config = \
        nr_pucch_format0_testvectors.gen_pucchformat0_testvec_config(pucchformat0_tvcfg)
    
    path = "py5gphy/nr_default_config/"
    with open(path + "default_UL_waveform_config.json", 'r') as f:
        waveform_config = json.load(f)
    waveform_config['numofsubframes'] = 2
    waveform_config['samplerate_in_mhz'] = 122.88
    waveform_config['startSFN'] = 0
    waveform_config['startslot'] = 0

    pusch_config_list = []
    srs_config_list = []
    pucch_format0_config_list = []
    pucch_format0_config_list.append(pucch_format0_config)
    pucch_format1_config_list = []
    pucch_format2_config_list = []
    pucch_format3_config_list = []
    pucch_format4_config_list = []

    fd_waveform, td_waveform, ul_waveform = nr_ul_waveform.gen_ul_waveform(waveform_config, 
                    carrier_config, pusch_config_list, srs_config_list,
                    pucch_format0_config_list,pucch_format1_config_list,pucch_format2_config_list,
                    pucch_format3_config_list,pucch_format4_config_list)

    assert np.allclose(fd_waveform, ref_fd_slot_data, atol=1e-5)
    
    