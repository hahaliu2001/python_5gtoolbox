# -*- coding:utf-8 -*-

import numpy as np
import math

def PRACH_seq_gen(prach_RootSequenceIndex, LRA,zeroCorrelationZoneConfig,PreambleIndex):
    """generate PRACH sequence floolwingto 38.211 6.3.3.1
    """
    NCS = _get_Ncs(LRA, zeroCorrelationZoneConfig)

    # find logical root sequence index
    #num_of_preamblea_per_zc is number of preambles one ZC contain
    if NCS == 0:
        #when NCS = 0, one ZC contian only one preamble
        num_of_preamblea_per_zc = 1
    else:
        num_of_preamblea_per_zc = LRA // NCS

    if num_of_preamblea_per_zc >= 64:
        logical_root_seq = prach_RootSequenceIndex
        v = PreambleIndex
    else:
        numof_root_seq = math.ceil(64/num_of_preamblea_per_zc)
        for m in range(numof_root_seq):
            if num_of_preamblea_per_zc*(m+1) > PreambleIndex:
                logical_root_seq = prach_RootSequenceIndex + m
                #The logical root sequence order is cyclic; 
                #the logical index range is 0 - 837 for LRA = 839 and 0 - 137 for LRA = 139
                if logical_root_seq > LRA-2:
                    logical_root_seq -= LRA - 1
                v = PreambleIndex - num_of_preamblea_per_zc*m
                break
    
    #find sequency numer u
    u = _get_sequence_number(LRA, logical_root_seq)

    #generate xu sequence by 38.211 6.3.3.1
    xu = np.exp(-1j*np.pi*u*np.arange(LRA)*np.arange(1,LRA+1)/LRA)

    #get xuv
    Cv = v*NCS
    xuv = xu[list(range(Cv,LRA))+list(range(Cv))]

    #get frequency domain data yuv
    yuv = np.fft.fft(xuv)

    return yuv


def _get_Ncs(LRA,zeroCorrelationZoneConfig):
    """ get NCS value
    """
    #38.211 Table 6.3.3.1-5:
    longLRA_list = [0,13,15,18,22,26,32,38,46,59,76,93,119,167,279,419 ]
    #38.211 Table 6.3.3.1-7
    shortLRA_list = [0,2,4,6,8,10,12,13,15,17,19,23,27,34,46,69]

    assert zeroCorrelationZoneConfig in range(16)
    assert LRA in [139, 839]
    if LRA == 839:
        NCS = longLRA_list[zeroCorrelationZoneConfig]
    else:
        NCS = shortLRA_list[zeroCorrelationZoneConfig]
    return NCS

def _get_sequence_number(LRA, logical_root_seq):
    """get prach sequence number following to 38.211 Table 6.3.3.1-3 and Table 6.3.3.1-34
    """
    rootsTableShortPreamble=[1,138,2,137,3,136,4,135,5,134,6,133,7,132,8,131,9,130,10,129,11,128,12,127,13,126,14,125,15,124,16,123,17,122,18,121,19,120,20,119,21,118,22,117,23,116,24,115,25,114,26,113,27,112,28,111,29,110,30,109,31,108,32,107,33,106,34,105,35,104,36,103,37,102,38,101,39,100,40,99,41,98,42,97,43,96,44,95,45,94,46,93,47,92,48,91,49,90,50,89,51,88,52,87,53,86,54,85,55,84,56,83,57,82,58,81,59,80,60,79,61,78,62,77,63,76,64,75,65,74,66,73,67,72,68,71,69,70]
    rootsTableLongPreamble=[129,710,140,699,120,719,210,629,168,671,84,755,105,734,93,746,70,769,60,779,2,837,1,838,56,783,112,727,148,691,80,759,42,797,40,799,35,804,73,766,146,693,31,808,28,811,30,809,27,812,29,810,24,815,48,791,68,771,74,765,178,661,136,703,86,753,78,761,43,796,39,800,20,819,21,818,95,744,202,637,190,649,181,658,137,702,125,714,151,688,217,622,128,711,142,697,122,717,203,636,118,721,110,729,89,750,103,736,61,778,55,784,15,824,14,825,12,827,23,816,34,805,37,802,46,793,207,632,179,660,145,694,130,709,223,616,228,611,227,612,132,707,133,706,143,696,135,704,161,678,201,638,173,666,106,733,83,756,91,748,66,773,53,786,10,829,9,830,7,832,8,831,16,823,47,792,64,775,57,782,104,735,101,738,108,731,208,631,184,655,197,642,191,648,121,718,141,698,149,690,216,623,218,621,152,687,144,695,134,705,138,701,199,640,162,677,176,663,119,720,158,681,164,675,174,665,171,668,170,669,87,752,169,670,88,751,107,732,81,758,82,757,100,739,98,741,71,768,59,780,65,774,50,789,49,790,26,813,17,822,13,826,6,833,5,834,33,806,51,788,75,764,99,740,96,743,97,742,166,673,172,667,175,664,187,652,163,676,185,654,200,639,114,725,189,650,115,724,194,645,195,644,192,647,182,657,157,682,156,683,211,628,154,685,123,716,139,700,212,627,153,686,213,626,215,624,150,689,225,614,224,615,221,618,220,619,127,712,147,692,124,715,193,646,205,634,206,633,116,723,160,679,186,653,167,672,79,760,85,754,77,762,92,747,58,781,62,777,69,770,54,785,36,803,32,807,25,814,18,821,11,828,4,835,3,836,19,820,22,817,41,798,38,801,44,795,52,787,45,794,63,776,67,772,72,767,76,763,94,745,102,737,90,749,109,730,165,674,111,728,209,630,204,635,117,722,188,651,159,680,198,641,113,726,183,656,180,659,177,662,196,643,155,684,214,625,126,713,131,708,219,620,222,617,226,613,230,609,232,607,262,577,252,587,418,421,416,423,413,426,411,428,376,463,395,444,283,556,285,554,379,460,390,449,363,476,384,455,388,451,386,453,361,478,387,452,360,479,310,529,354,485,328,511,315,524,337,502,349,490,335,504,324,515,323,516,320,519,334,505,359,480,295,544,385,454,292,547,291,548,381,458,399,440,380,459,397,442,369,470,377,462,410,429,407,432,281,558,414,425,247,592,277,562,271,568,272,567,264,575,259,580,237,602,239,600,244,595,243,596,275,564,278,561,250,589,246,593,417,422,248,591,394,445,393,446,370,469,365,474,300,539,299,540,364,475,362,477,298,541,312,527,313,526,314,525,353,486,352,487,343,496,327,512,350,489,326,513,319,520,332,507,333,506,348,491,347,492,322,517,330,509,338,501,341,498,340,499,342,497,301,538,366,473,401,438,371,468,408,431,375,464,249,590,269,570,238,601,234,605,257,582,273,566,255,584,254,585,245,594,251,588,412,427,372,467,282,557,403,436,396,443,392,447,391,448,382,457,389,450,294,545,297,542,311,528,344,495,345,494,318,521,331,508,325,514,321,518,346,493,339,500,351,488,306,533,289,550,400,439,378,461,374,465,415,424,270,569,241,598,231,608,260,579,268,571,276,563,409,430,398,441,290,549,304,535,308,531,358,481,316,523,293,546,288,551,284,555,368,471,253,586,256,583,263,576,242,597,274,565,402,437,383,456,357,482,329,510,317,522,307,532,286,553,287,552,266,573,261,578,236,603,303,536,356,483,355,484,405,434,404,435,406,433,235,604,267,572,302,537,309,530,265,574,233,606,367,472,296,543,336,503,305,534,373,466,280,559,279,560,419,420,240,599,258,581,229,610]
    assert LRA in [139, 839]
    if LRA == 839:
        u = rootsTableLongPreamble[logical_root_seq]
    else:
        u = rootsTableShortPreamble[logical_root_seq]
    
    return u

if __name__ == "__main__":
    print("test nr PRACH sequence generation")

    from tests.nr_prach import test_nr_prach_seq
    file_lists = test_nr_prach_seq.get_nr_prach_seq_testvectors()
    count = 1
    for filename in file_lists:
        print("count= {}, filename= {}".format(count, filename))
        count += 1
        test_nr_prach_seq.test_nr_prach_seq(filename)

    aaaa=1