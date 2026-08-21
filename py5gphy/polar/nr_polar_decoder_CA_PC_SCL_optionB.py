# -*- coding:utf-8 -*-

import numpy as np
import statistics 

from py5gphy.polar import polar_interleaver
from py5gphy.polar import polar_construct
from py5gphy.polar import polar_path_list
from py5gphy.crc import crc

def nr_decode_polar_SCL_optionB(LLRin, E, K, L, nMax, iIL, CRCLEN=24, padCRC=0, rnti=0 ):
    """ polar CRC aided and PC checked SCL decoder
        the idea refer to "List Decoding of Polar Codes" IV. SUCCESSIVE CANCELLATION LIST DECODER
        But the design is totally different
    input:
        LLRin: N length input LLR demodulation data
        N: length of polar decoder input data
        E: rate match length
        K: decoded data length
        (nMax, iIL,CRCLEN): (9, 1,24) apply for DL and (10, 0,11),(10, 0,6) apply for UL
        L: SCL polar decoder list length, shoule be even value
        CRCLEN: CRClength, CRC is used for CRC-aided SCL decoder
        padCRC: 1: padding 24 '1' ahead of CRC data for CRC encoder which is used by DL DCI
                0: no padding which is used by DL BCH
        rnti: used by DL DCI CRC mask
    output:
        decbits: L length bits
        status: True or False
    """
    
    # validate data
    assert [nMax, iIL,CRCLEN] in [[9,1,24], [10,0,6], [10,0,11]]
    
    #get F array, PC array and other values
    F, qPC, N, nPC, nPCwm = polar_construct.construct(K, E, nMax)
    assert (K + nPC) <= E #38.212 5.3.1
    assert (N <= 2**nMax) and (N >= 2**5) #38.212 5.3.1
    assert LLRin.size == N
       
    m = int(np.ceil(np.log2(N))) #m = log2(N)
    
    #initial a few tables
    #bit seqeucne ck is used to polar interleaving to get ckbar, ckbar map to u seq at non-frozen location
    #here need get u_seq to ckbar mapping table, 
    ckbar_indices = [idx for idx in range(N) if (F[idx]==0) and (idx not in qPC)]
    #deinterleave table, used for iIL == 1
    if iIL == 1:
        depitable = polar_interleaver.gen_deinterleave_table(K)
        crcidx_matrix = crc.gen_CRC24C_encoding_matrix(K-24)
        crc_mask = crc.gen_crc_mask(K-24, padCRC, rnti)

    #Bit-wise reverse LLRin to get layer0 LLR
    LLR0 = np.zeros(N)
    for branch in range(N):
        #path.setLLR(LLRin[branch], 0, branch, 0)
        #bit reverse LLRin
        br=int( '{:0{width}b}'.format(branch, width=m)[::-1],2)
        LLR0[branch] = LLRin[br]
    
    #init pathlist, active first path,
    pathlist = polar_path_list.Pathlist(LLR0, N, L)

    #SCL main loop
    for phase in range(N):
        active_paths = pathlist.get_active_paths()
        #recursivelyCalcLLR for each active path
        for _, path in active_paths:
            recursivelyCalcLLR(path,m,0,phase)
        
        if F[phase] == 1: #frozen bits
            for _, path in active_paths:
                path.update_BandPM(phase,0) #set B=0 and update PM
        else: #unfrozen bits
            continuePaths_UnfrozenBit(pathlist, phase)
                        
            active_paths = pathlist.get_active_paths() #get new active paths
              
            if nPC > 0 and (phase in qPC): #this this is PC bit
                # PC check
                for idx, path in active_paths:
                    u_seq = path.get_u_seq()
                    pc = u_seq[phase]
                    u_seq = u_seq[0:phase] #select u seq from 0 to phase -1
                    if pc != cal_polar_pc(u_seq, F, qPC, phase):
                        #bad path
                        pathlist.inactive_path(idx)
            elif iIL  : 
                #if polar interleaver = 1 
                # distributed CRC check, CRC poly must be 24C and CRC size = 24
                #get ckbar index for phase in u_seq
                ckbar_loc = ckbar_indices.index(phase)
                if ckbar_loc in depitable[-24:]: #this is CRC bits
                    for idx, path in active_paths:
                        u_seq = path.get_u_seq()
                        ckbar = u_seq[ckbar_indices] #get ckbar bit sequence
                        ck = ckbar[depitable] #deinterleave to get ck seq
                        crc_bit_loc = depitable[-24:].index(ckbar_loc)
                        if  crc.check_distributed_CRC24C(ck, crc_bit_loc, crcidx_matrix, crc_mask) == False:
                            #CRC bit can not match
                            pathlist.inactive_path(idx)

            if pathlist.get_total_active_paths() == 0:
                #early terminate
                return [-1], False
            
        active_paths = pathlist.get_active_paths() #get new active paths
        if (phase % 2) == 1:
            for _, path in active_paths:
                recursivelyUpdateB(path,m,0,phase)

    if pathlist.get_total_active_paths() == 0:
        #early terminate
        return [-1], False
    
    #sort active paths based on PM and then do CRC check for each active path until pass CRC check
    active_paths = pathlist.get_active_paths() #get new active paths
    PM_list = [[idx,path.PM] for idx, path in active_paths]
    PM_list.sort(key=lambda x: x[1]) #sort by the second element of sublist

    if iIL:
        #distributed CRC has passed during CA-SCL decoding, here return best active path
        path = pathlist.get_path(PM_list[0][0])
        decodedbits = path.get_u_seq()  
        ckbar = decodedbits[ckbar_indices] #get information bit and CRC bits only, not including pc bits
        ck = ckbar[depitable] #deinterleave to get ck seq
        return np.array(ck,'i1'), True
    else:
        for idx, _ in PM_list:
            path = pathlist.get_path(idx)
            #get decoded bits
            decodedbits = path.get_u_seq()  
            ckbar = decodedbits[ckbar_indices] #get information bit and CRC bits only, not including pc bits
            ck = ckbar
            poly = {6:'6', 11: '11', 24: '24C'}[CRCLEN]
            _, err = crc.nr_crc_decode(np.array(ck,'i1'), poly, rnti)
            if err == 0:
                return np.array(ck,'i1'), True

    #all path CRC failed
    return [-1], False

def continuePaths_UnfrozenBit(pathlist, phase):
    active_paths = pathlist.get_active_paths()
    #generate nextPM list for B=0 and B=1 for all active path
    #PM_list sublist [idx, B, PM] 
    PM_list = []
    for idx, path in active_paths:
        path.gen_nextPM(phase)
        PM_list.append([path.nextPM[0],idx,0])
        PM_list.append([path.nextPM[1],idx,1])

    L = pathlist.L            
    if pathlist.get_total_active_paths()*2 <= L:
        #split each active path to two,update current path with B=0 and new path with B=1
        for _, path in active_paths:
            #clone to one inactive path
            inactive_idx = pathlist.get_inactive_path_idx()
            clone2path = pathlist.get_path(inactive_idx)
            path.clone(clone2path)

            path.update_BandPM(phase,0) #update current path with B=0
            clone2path.update_BandPM(phase,1) #update new path with B=1
            pathlist.active_path(inactive_idx) #set new path to active
    else:
        PM_list.sort() #sort by PM
        tmp = [v[0] for v in PM_list]
        threshold = statistics.median(tmp)
                
        # inactive paths with both nextPM value that is >= threshold
        # to free bad path
        for idx, path in active_paths:
            if (path.nextPM[0] >= threshold) and (path.nextPM[1] >= threshold):
                pathlist.inactive_path(idx)
                
        active_paths = pathlist.get_active_paths() #get new active paths

        #if only one nextPM of the active path < threshod, update this path
        for _, path in active_paths:
            if (path.nextPM[0] < threshold) and (path.nextPM[1] >= threshold):
                path.update_BandPM(phase,0)
                    
            if (path.nextPM[0] >= threshold) and (path.nextPM[1] < threshold):
                path.update_BandPM(phase,1)
                    
        #duplicate active paths that both nextPM < threshold
        # it may be possible that multiple nextPM value is equal to threhold, 
        for _, path in active_paths:
            if (path.nextPM[0] < threshold) and (path.nextPM[1] < threshold):
                #clone to one inactive path
                inactive_idx = pathlist.get_inactive_path_idx()
                if inactive_idx == -1:
                    assert 0 #should not reach here                            
                else:
                    clone2path = pathlist.get_path(inactive_idx)
                    path.clone(clone2path)

                    path.update_BandPM(phase,0) #update current path with B=0
                    clone2path.update_BandPM(phase,1) #update new path with B=1
                    pathlist.active_path(inactive_idx) #set new path to active

def recursivelyUpdateB(path,layer,branch,phase):
    #update path.B: 
    newphase = phase // 2
    
    b1 = path.getB(layer,branch,phase-1)
    b2 = path.getB(layer,branch,phase)
    
    path.setB((b1+b2)%2, layer-1, 2*branch, newphase)
    
    if (newphase % 2):
        recursivelyUpdateB(path,layer-1,2*branch,newphase)

    path.setB(b2, layer-1, 2*branch+1, newphase)
    if (newphase % 2):
        recursivelyUpdateB(path,layer-1,2*branch+1,newphase)   

def recursivelyCalcLLR(path,layer, branch,phase):
    if layer == 0:
        return
    newphase = phase // 2
    
    if (phase % 2) == 0:
        recursivelyCalcLLR(path,layer-1, 2*branch, newphase)
        recursivelyCalcLLR(path,layer-1, 2*branch+1, newphase)
    
    LLR1 = path.getLLR(layer-1, 2*branch, newphase)
    LLR2 = path.getLLR(layer-1, 2*branch+1, newphase)
    
    if (phase % 2) == 0:
        value = np.sign(LLR1)*np.sign(LLR2)*min(abs(LLR1), abs(LLR2))
        path.setLLR(value, layer, branch, phase)
    else:
        B = path.getB(layer,branch,phase-1)
        value = LLR2 + (-1)**B * LLR1
        path.setLLR(value, layer, branch, phase)
        
def cal_polar_pc(u_seq, F, qPC, phase):
    """ cal pc value at phase"""
    y0 = y1 = y2 = y3 = y4 =0
    for idx in range(phase+1):
        yt = y0; y0 = y1; y1 = y2; y2 = y3; y3 = y4; y4 = yt
        if F[idx] == 0:  #information bit
            if idx in qPC:
                if idx == phase:
                    return y0           
            else:
                y0 = y0 ^ u_seq[idx]

if __name__ == "__main__":
    from tests.polar import test_nr_polar_decoder_CA_PC_SCL_optionB

    testlists = test_nr_polar_decoder_CA_PC_SCL_optionB.get_testvectors()
    for _ in range(2):
        for idx, testcase in enumerate(testlists):
            print("count= {}".format(idx))
            test_nr_polar_decoder_CA_PC_SCL_optionB.test_nr_polar_SCL_decoder(testcase)

    for _ in range(2):
        for idx, testcase in enumerate(testlists):
            print("count= {}".format(idx))
            test_nr_polar_decoder_CA_PC_SCL_optionB.test_nr_polar_SCL_decoder_no_noise(testcase)

    pass