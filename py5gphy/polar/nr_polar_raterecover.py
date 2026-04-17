# -*- coding:utf-8 -*-

import numpy as np
import math

def ratemrecover_polar(LLRin, K, N, iBIL, LLR_limit=20):
    """ does Polar block rate match recovery TS38.212 5.4.1
    outbits = ratemrecover_polar(inbits, K, N, iBIL)
    the processing:
        for puncturing bits, set LLR to 0
        for shorten bits, set LLR to very high value
        for repetition bits, add repetition LLR together
    input:
        inbits: E length input LLR
        K: information bits length
        N: rate match input length
        iBIL: interleaving coded bits or not. iBIL=1 for UL, 0 for DL
        LLR_limit: peak LLR, used for shorten bits, 20->10dB Eb/N0
    output:
        LLRout: N length LLR
    """
    E = LLRin.size

    assert E <= 8192
    if iBIL:  #UL
        assert not((K > 25) and (K <=30))  
        assert K > 18
    
    # Channel deinterleaving, Section 5.4.1.3
    if iBIL:  #Specified for uplink only
        inE = _iBIL_deinterl(LLRin, E)
    else:
        inE = LLRin

    # Bit selection, Section 5.4.1.2
    outN = np.zeros(N )
    if E >= N: #repetition, add repeted LLR together
        offset = 0
        while offset + N <= E:
            outN += inE[offset:offset+N]
            offset += N
        
        if offset < E:
            len = E - offset
            outN[0:len] += inE[offset:]
    else:
        if (K/E) <= (7/16): #puncturing, put at the end,puncturing bits LLR = 0
            outN[N-E:N] = inE
        else:  #shortening, put at the start,
            outN[0:E] = inE
            outN[E:] = LLR_limit #shorted bits LLR is highest
    
    #Sub-block deinterleaving, Section 5.4.1.1
    # Table 5.4.1.1-1: Sub-block interleaver pattern
    pi = [0,1,2,4, 3,5,6,7, 8,16,9,17, 10,18,11,19,
          12,20,13,21, 14,22,15,23, 24,25,26,28, 27,29,30,31]
    LLRout = np.zeros(N)
    for n in range(N):
        m = (32 * n) // N
        jn = pi[m] * (N // 32) + (n % (N // 32))
        LLRout[jn] = outN[n]
    
    return LLRout

def _iBIL_deinterl(LLRin, E):
    # T is smallest integer such that T(T+1)/2>=E
    T = math.ceil((-1 + math.sqrt(1+8*E))/2)

    # create table
    vtable = np.zeros((T,T),'i2')
    k = 0
    for m in range(T):
        for n in range(T-m):
            if k < E:
                vtable[m,n] = 1
            k +=1
    
    #Write input to buffer column-wise
    V = np.inf * np.ones((T, T)) 
    k = 0
    for n in range(T):
        for m in range(T-n):
            if (k < E) and vtable[m,n]:
                V[m,n] = LLRin[k]
                k += 1
    
    #read out row-wise
    outd = np.zeros(E)
    k = 0
    for m in range(T):
        for n in range(T-m):
            if V[m,n] != np.inf:
                outd[k] = V[m,n]
                k += 1
    
    return outd


if __name__ == "__main__":
    print("test polar rate recover")
    from tests.polar import test_polar_raterecover
    file_lists = test_polar_raterecover.get_testvectors()
    count = 1
    for filename in file_lists:
        print("count= {}, filename= {}".format(count, filename))
        count += 1
        test_polar_raterecover.test_nr_polar_raterecover(filename)

    pass
