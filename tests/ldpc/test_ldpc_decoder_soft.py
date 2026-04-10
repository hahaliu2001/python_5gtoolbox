# -*- coding: utf-8 -*-
import numpy as np
import pytest
from scipy import io
import os
import time
from zipfile import ZipFile 

from py5gphy.ldpc import nr_ldpc_decode

def get_testvectors():
    path = "tests/ldpc/testvectors"
    if len(os.listdir(path)) < 3: #desn't unzip testvectors
        zipfile_lists = []
        for f in os.listdir(path):
            if f.endswith(".zip"):
                zipfile_lists.append(path + '/' + f)

        for zipfile in zipfile_lists:
            zObject = ZipFile(zipfile, 'r')
            zObject.extractall(path)

    file_lists = []
    for f in os.listdir(path):
        if f.endswith(".mat") and f.startswith("ldpc_encode_testvec"):
            file_lists.append(path + '/' + f)
                        
    return file_lists

@pytest.mark.parametrize('filename', get_testvectors())
def test_nr_ldpc_decode_BP(filename):
    """ test LDPC BP decoder"""
    #ldpc decoder parameters
    snr_db = 30
    bler_target = 0.3 #very loose target to just make the test pass all the time
    max_count = 10
    L = 32

    matfile = io.loadmat(filename)
    #read data from mat file
    ck_ref = matfile['in'][0]
    dn = matfile['dn'][0]
    Zc = matfile['Zc'][0][0]
    bgn = matfile['bgn'][0][0]

    #test to verify bler <= bler_target
    failed_count = 0
    for _ in range(max_count):
        en = 1 - 2*dn.astype('i1') #BPSK modulation, 0 -> 1, 1 -> -1
        fn = en + np.random.normal(0, 10**(-snr_db/20), dn.size) #add noise
        #LLR is log(P(0)/P(1)) = (-(x-1)^2+(x+1)^2)/(2*noise_power) = 4x/(2*noise_power) = 2x/noise_power
        noise_power = 10**(-snr_db/10)
        LLRin = 2*fn/noise_power

        ck_decoded,dn_decoded, status = nr_ldpc_decode.nr_decode_ldpc(LLRin, Zc, bgn, L, algo='BP')
        if not np.array_equal(ck_ref,ck_decoded[0:ck_ref.size]):
            failed_count += 1
    
    bler = failed_count/max_count
    assert bler <= bler_target

@pytest.mark.parametrize('filename', get_testvectors())
def test_nr_ldpc_decode_min_sum(filename):
    """ test LDPC min-sum decoder"""
    #ldpc decoder parameters
    snr_db = 3
    bler_target = 0.3 #very loose target to just make the test pass all the time
    max_count = 10
    L = 32

    matfile = io.loadmat(filename)
    #read data from mat file
    ck_ref = matfile['in'][0]
    dn = matfile['dn'][0]
    Zc = matfile['Zc'][0][0]
    bgn = matfile['bgn'][0][0]

    #test to verify bler <= bler_target
    failed_count = 0
    for _ in range(max_count):
        en = 1 - 2*dn.astype('i1') #BPSK modulation, 0 -> 1, 1 -> -1
        fn = en + np.random.normal(0, 10**(-snr_db/20), dn.size) #add noise
        #LLR is log(P(0)/P(1)) = (-(x-1)^2+(x+1)^2)/(2*noise_power) = 4x/(2*noise_power) = 2x/noise_power
        noise_power = 10**(-snr_db/10)
        LLRin = 2*fn/noise_power

        ck_decoded,dn_decoded,tatus = nr_ldpc_decode.nr_decode_ldpc(LLRin, Zc, bgn, L, algo='min-sum')
        if not np.array_equal(ck_ref,dn_decoded[0:ck_ref.size]):
            failed_count += 1
    
    bler = failed_count/max_count
    assert bler <= bler_target

@pytest.mark.parametrize('filename', get_testvectors())
def test_nr_ldpc_decode_NMS(filename):
    """ test LDPC normalized min-sum decoder"""
    #ldpc decoder parameters
    snr_db = 30
    bler_target = 0.3 #very loose target to just make the test pass all the time
    max_count = 10
    L = 32
    alpha_v = 0.5

    matfile = io.loadmat(filename)
    #read data from mat file
    ck_ref = matfile['in'][0]
    dn = matfile['dn'][0]
    Zc = matfile['Zc'][0][0]
    bgn = matfile['bgn'][0][0]

    #test to verify bler <= bler_target
    failed_count = 0
    for _ in range(max_count):
        en = 1 - 2*dn.astype('i1') #BPSK modulation, 0 -> 1, 1 -> -1
        fn = en + np.random.normal(0, 10**(-snr_db/20), dn.size) #add noise
        #LLR is log(P(0)/P(1)) = (-(x-1)^2+(x+1)^2)/(2*noise_power) = 4x/(2*noise_power) = 2x/noise_power
        noise_power = 10**(-snr_db/10)
        LLRin = 2*fn/noise_power

        ck_decoded,dn_decoded, status = nr_ldpc_decode.nr_decode_ldpc(LLRin, Zc, bgn, L, algo='min-sum', alpha=alpha_v)
        if not np.array_equal(ck_ref,dn_decoded[0:ck_ref.size]):
            failed_count += 1
    
    bler = failed_count/max_count
    assert bler <= bler_target

@pytest.mark.parametrize('filename', get_testvectors())
def test_nr_ldpc_decode_OMS(filename):
    """ test LDPC offset min-sum decoder"""
    #ldpc decoder parameters
    snr_db = 3
    bler_target = 0.3 #very loose target to just make the test pass all the time
    max_count = 10
    L = 32
    alpha_v=1
    beta_v = 0.3

    matfile = io.loadmat(filename)
    #read data from mat file
    ck_ref = matfile['in'][0]
    dn = matfile['dn'][0]
    Zc = matfile['Zc'][0][0]
    bgn = matfile['bgn'][0][0]

    #test to verify bler <= bler_target
    failed_count = 0
    for _ in range(max_count):
        en = 1 - 2*dn.astype('i1') #BPSK modulation, 0 -> 1, 1 -> -1
        fn = en + np.random.normal(0, 10**(-snr_db/20), dn.size) #add noise
        #LLR is log(P(0)/P(1)) = (-(x-1)^2+(x+1)^2)/(2*noise_power) = 4x/(2*noise_power) = 2x/noise_power
        noise_power = 10**(-snr_db/10)
        LLRin = 2*fn/noise_power

        ck_decoded,dn_decoded, status = nr_ldpc_decode.nr_decode_ldpc(LLRin, Zc, bgn, L, algo='min-sum',alpha=alpha_v,beta=beta_v)
        if not np.array_equal(ck_ref,dn_decoded[0:ck_ref.size]):
            failed_count += 1
    
    bler = failed_count/max_count
    assert bler <= bler_target

@pytest.mark.parametrize('filename', get_testvectors())
def test_nr_ldpc_decode_mixed_min_sum(filename):
    """ test LDPC mixed min-sum decoder"""
    #ldpc decoder parameters
    snr_db = 3
    bler_target = 0.3 #very loose target to just make the test pass all the time
    max_count = 10
    L = 32
    alpha_v = 0.5
    beta_v = 0.3

    matfile = io.loadmat(filename)
    #read data from mat file
    ck_ref = matfile['in'][0]
    dn = matfile['dn'][0]
    Zc = matfile['Zc'][0][0]
    bgn = matfile['bgn'][0][0]

    #test to verify bler <= bler_target
    failed_count = 0
    for _ in range(max_count):
        en = 1 - 2*dn.astype('i1') #BPSK modulation, 0 -> 1, 1 -> -1
        fn = en + np.random.normal(0, 10**(-snr_db/20), dn.size) #add noise
        #LLR is log(P(0)/P(1)) = (-(x-1)^2+(x+1)^2)/(2*noise_power) = 4x/(2*noise_power) = 2x/noise_power
        noise_power = 10**(-snr_db/10)
        LLRin = 2*fn/noise_power

        ck_decoded,dn_decoded, status = nr_ldpc_decode.nr_decode_ldpc(LLRin, Zc, bgn, L, algo='min-sum',alpha=alpha_v,beta=beta_v)
        if not np.array_equal(ck_ref,dn_decoded[0:ck_ref.size]):
            failed_count += 1
    
    bler = failed_count/max_count
    assert bler <= bler_target

@pytest.mark.parametrize('filename', get_testvectors())
def test_nr_ldpc_decode_layed_mixed_min_sum(filename):
    """ test LDPC mixed min-sum decoder"""
    #ldpc decoder parameters
    snr_db = 3
    bler_target = 0.3 #very loose target to just make the test pass all the time
    max_count = 10
    L = 32
    alpha_v = 0.5
    beta_v = 0.3

    matfile = io.loadmat(filename)
    #read data from mat file
    ck_ref = matfile['in'][0]
    dn = matfile['dn'][0]
    Zc = matfile['Zc'][0][0]
    bgn = matfile['bgn'][0][0]

    #test to verify bler <= bler_target
    failed_count = 0
    for _ in range(max_count):
        en = 1 - 2*dn.astype('i1') #BPSK modulation, 0 -> 1, 1 -> -1
        fn = en + np.random.normal(0, 10**(-snr_db/20), dn.size) #add noise
        #LLR is log(P(0)/P(1)) = (-(x-1)^2+(x+1)^2)/(2*noise_power) = 4x/(2*noise_power) = 2x/noise_power
        noise_power = 10**(-snr_db/10)
        LLRin = 2*fn/noise_power

        ck_decoded,dn_decoded, status = nr_ldpc_decode.nr_decode_ldpc(LLRin, Zc, bgn, L, algo='layered_min-sum',alpha=alpha_v,beta=beta_v)
        if not np.array_equal(ck_ref,dn_decoded[0:ck_ref.size]):
            failed_count += 1
    
    bler = failed_count/max_count
    assert bler <= bler_target

@pytest.mark.parametrize('filename', get_testvectors())
def test_nr_ldpc_decode_layed_BP(filename):
    """ test LDPC mixed min-sum decoder"""
    #ldpc decoder parameters
    snr_db = 3
    bler_target = 0.3 #very loose target to just make the test pass all the time
    max_count = 10
    L = 32
    alpha_v = 0.5
    beta_v = 0.3

    matfile = io.loadmat(filename)
    #read data from mat file
    ck_ref = matfile['in'][0]
    dn = matfile['dn'][0]
    Zc = matfile['Zc'][0][0]
    bgn = matfile['bgn'][0][0]

    #test to verify bler <= bler_target
    failed_count = 0
    for _ in range(max_count):
        en = 1 - 2*dn.astype('i1') #BPSK modulation, 0 -> 1, 1 -> -1
        fn = en + np.random.normal(0, 10**(-snr_db/20), dn.size) #add noise
        #LLR is log(P(0)/P(1)) = (-(x-1)^2+(x+1)^2)/(2*noise_power) = 4x/(2*noise_power) = 2x/noise_power
        noise_power = 10**(-snr_db/10)
        LLRin = 2*fn/noise_power

        ck_decoded,dn_decoded, status = nr_ldpc_decode.nr_decode_ldpc(LLRin, Zc, bgn, L, algo='layered_BP',alpha=alpha_v,beta=beta_v)
        if not np.array_equal(ck_ref,dn_decoded[0:ck_ref.size]):
            failed_count += 1
    
    bler = failed_count/max_count
    assert bler <= bler_target
