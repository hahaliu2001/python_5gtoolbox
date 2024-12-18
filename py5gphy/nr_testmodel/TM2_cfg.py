# -*- coding:utf-8 -*-

import numpy as np
import copy

def gen_TM2_pdsch_cfg_list(carrier_prb_size,Duplex_mode,scs,ref_pdsch_config):
    """ 
    """
    pdsch_config_list = []
    if Duplex_mode == 'FDD':
        
        pdsch_cfg = copy.deepcopy(ref_pdsch_config)
        pdsch_cfg['rnti'] = 2
        pdsch_cfg['mcs_table'] = "256QAM"
        pdsch_cfg['mcs_index'] = 11 #64QAM, TargetCodeRateby1024 = 466
        pdsch_cfg['rv'] = [0]
        #38.141-1 4.9.2.3.2 defined PN23 for PDSCH seq generation, here just choose random 
        #generate the sequence
        pdsch_cfg['data_source'] = []
        pdsch_cfg['num_of_layers'] = 1
        pdsch_cfg['VRBtoPRBMapping'] = "non-interleaved"
        pdsch_cfg['StartSymbolIndex'] = 2
        pdsch_cfg['NrOfSymbols'] = 12
        pdsch_cfg['ResAlloType1']['RBStart'] = 0
        pdsch_cfg['ResAlloType1']['RBSize'] = 1
        pdsch_cfg['period_in_slot'] = int(10*scs/15)
        pdsch_cfg['allocated_slots'] = list(range(0,int(10*scs/15),3))
        pdsch_config_list.append(pdsch_cfg)

        pdsch_cfg = copy.deepcopy(ref_pdsch_config)
        pdsch_cfg['rnti'] = 2
        pdsch_cfg['mcs_table'] = "256QAM"
        pdsch_cfg['mcs_index'] = 11 #64QAM, TargetCodeRateby1024 = 466
        pdsch_cfg['rv'] = [0]
        #38.141-1 4.9.2.3.2 defined PN23 for PDSCH seq generation, here just choose random 
        #generate the sequence
        pdsch_cfg['data_source'] = []
        pdsch_cfg['num_of_layers'] = 1
        pdsch_cfg['VRBtoPRBMapping'] = "non-interleaved"
        pdsch_cfg['StartSymbolIndex'] = 2
        pdsch_cfg['NrOfSymbols'] = 12
        pdsch_cfg['ResAlloType1']['RBStart'] = int(carrier_prb_size // 2)
        pdsch_cfg['ResAlloType1']['RBSize'] = 1
        pdsch_cfg['period_in_slot'] = int(10*scs/15)
        pdsch_cfg['allocated_slots'] = list(range(1,int(10*scs/15),3))
        pdsch_config_list.append(pdsch_cfg)

        pdsch_cfg = copy.deepcopy(ref_pdsch_config)
        pdsch_cfg['rnti'] = 2
        pdsch_cfg['mcs_table'] = "256QAM"
        pdsch_cfg['mcs_index'] = 11 #64QAM, TargetCodeRateby1024 = 466
        pdsch_cfg['rv'] = [0]
        #38.141-1 4.9.2.3.2 defined PN23 for PDSCH seq generation, here just choose random 
        #generate the sequence
        pdsch_cfg['data_source'] = []
        pdsch_cfg['num_of_layers'] = 1
        pdsch_cfg['VRBtoPRBMapping'] = "non-interleaved"
        pdsch_cfg['StartSymbolIndex'] = 2
        pdsch_cfg['NrOfSymbols'] = 12
        pdsch_cfg['ResAlloType1']['RBStart'] = carrier_prb_size - 1
        pdsch_cfg['ResAlloType1']['RBSize'] = 1
        pdsch_cfg['period_in_slot'] = int(10*scs/15)
        pdsch_cfg['allocated_slots'] = list(range(2,int(10*scs/15),3))
        pdsch_config_list.append(pdsch_cfg)
    else: 
        #TDD pattern is defined in 38.141-1 Table 4.9.2.2-1 
        #for 15khz scs, TDD pattern is 5ms period, DDDSUU, S slot(D/E/U): 10:2:2
        #for 30khz scs, TDD pattern is 5ms period, DDDDDDDSUUUU, S slot(D/E/U): 6:4:4
        pdsch_cfg = copy.deepcopy(ref_pdsch_config)
        pdsch_cfg['rnti'] = 2
        pdsch_cfg['mcs_table'] = "256QAM"
        pdsch_cfg['mcs_index'] = 11 #64QAM, TargetCodeRateby1024 = 466
        pdsch_cfg['rv'] = [0]
        #38.141-1 4.9.2.3.2 defined PN23 for PDSCH seq generation, here just choose random 
        #generate the sequence
        pdsch_cfg['data_source'] = []
        pdsch_cfg['num_of_layers'] = 1
        pdsch_cfg['VRBtoPRBMapping'] = "non-interleaved"
        pdsch_cfg['StartSymbolIndex'] = 2
        pdsch_cfg['NrOfSymbols'] = 12
        pdsch_cfg['ResAlloType1']['RBStart'] = 0
        pdsch_cfg['ResAlloType1']['RBSize'] = 1
        pdsch_cfg['period_in_slot'] = int(10*scs/15)
        slots = list(range(0,int(10*scs/15),3))
        if scs == 15:
            new_slots = [n for n in slots if (n % 5) in range(3)]
        else:
            new_slots = [n for n in slots if (n % 10) in range(7)]
        pdsch_cfg['allocated_slots'] = new_slots
        pdsch_config_list.append(pdsch_cfg)
        
        pdsch_cfg = copy.deepcopy(ref_pdsch_config)
        pdsch_cfg['rnti'] = 2
        pdsch_cfg['mcs_table'] = "256QAM"
        pdsch_cfg['mcs_index'] = 11 #64QAM, TargetCodeRateby1024 = 466
        pdsch_cfg['rv'] = [0]
        #38.141-1 4.9.2.3.2 defined PN23 for PDSCH seq generation, here just choose random 
        #generate the sequence
        pdsch_cfg['data_source'] = []
        pdsch_cfg['num_of_layers'] = 1
        pdsch_cfg['VRBtoPRBMapping'] = "non-interleaved"
        pdsch_cfg['StartSymbolIndex'] = 2
        pdsch_cfg['NrOfSymbols'] = 12
        pdsch_cfg['ResAlloType1']['RBStart'] = int(carrier_prb_size // 2)
        pdsch_cfg['ResAlloType1']['RBSize'] = 1
        pdsch_cfg['period_in_slot'] = int(10*scs/15)
        slots = list(range(1,int(10*scs/15),3))
        if scs == 15:
            new_slots = [n for n in slots if (n % 5) in range(3)]
        else:
            new_slots = [n for n in slots if (n % 10) in range(7)]
        pdsch_cfg['allocated_slots'] = new_slots
        pdsch_config_list.append(pdsch_cfg)

        pdsch_cfg = copy.deepcopy(ref_pdsch_config)
        pdsch_cfg['rnti'] = 2
        pdsch_cfg['mcs_table'] = "256QAM"
        pdsch_cfg['mcs_index'] = 11 #64QAM, TargetCodeRateby1024 = 466
        pdsch_cfg['rv'] = [0]
        #38.141-1 4.9.2.3.2 defined PN23 for PDSCH seq generation, here just choose random 
        #generate the sequence
        pdsch_cfg['data_source'] = []
        pdsch_cfg['num_of_layers'] = 1
        pdsch_cfg['VRBtoPRBMapping'] = "non-interleaved"
        pdsch_cfg['StartSymbolIndex'] = 2
        pdsch_cfg['NrOfSymbols'] = 12
        pdsch_cfg['ResAlloType1']['RBStart'] = carrier_prb_size - 1
        pdsch_cfg['ResAlloType1']['RBSize'] = 1
        pdsch_cfg['period_in_slot'] = int(10*scs/15)
        slots = list(range(2,int(10*scs/15),3))
        if scs == 15:
            new_slots = [n for n in slots if (n % 5) in range(3)]
        else:
            new_slots = [n for n in slots if (n % 10) in range(7)]
        pdsch_cfg['allocated_slots'] = new_slots
        pdsch_config_list.append(pdsch_cfg)

        pdsch_cfg = copy.deepcopy(ref_pdsch_config)
        pdsch_cfg['rnti'] = 2
        pdsch_cfg['mcs_table'] = "256QAM"
        pdsch_cfg['mcs_index'] = 11 #64QAM, TargetCodeRateby1024 = 466
        pdsch_cfg['rv'] = [0]
        #38.141-1 4.9.2.3.2 defined PN23 for PDSCH seq generation, here just choose random 
        #generate the sequence
        pdsch_cfg['data_source'] = []
        pdsch_cfg['num_of_layers'] = 1
        pdsch_cfg['VRBtoPRBMapping'] = "non-interleaved"
        pdsch_cfg['StartSymbolIndex'] = 2
        if scs == 15:
            pdsch_cfg['NrOfSymbols'] = 8
        else:
            pdsch_cfg['NrOfSymbols'] = 4
        pdsch_cfg['ResAlloType1']['RBStart'] = 0
        pdsch_cfg['ResAlloType1']['RBSize'] = 1
        pdsch_cfg['period_in_slot'] = int(10*scs/15)
        slots = list(range(0,int(10*scs/15),3))
        if scs == 15:
            new_slots = [n for n in slots if (n % 5) == 3]
        else:
            new_slots = [n for n in slots if (n % 10) == 7]
        pdsch_cfg['allocated_slots'] = new_slots
        pdsch_config_list.append(pdsch_cfg)
        
        pdsch_cfg = copy.deepcopy(ref_pdsch_config)
        pdsch_cfg['rnti'] = 2
        pdsch_cfg['mcs_table'] = "256QAM"
        pdsch_cfg['mcs_index'] = 11 #64QAM, TargetCodeRateby1024 = 466
        pdsch_cfg['rv'] = [0]
        #38.141-1 4.9.2.3.2 defined PN23 for PDSCH seq generation, here just choose random 
        #generate the sequence
        pdsch_cfg['data_source'] = []
        pdsch_cfg['num_of_layers'] = 1
        pdsch_cfg['VRBtoPRBMapping'] = "non-interleaved"
        pdsch_cfg['StartSymbolIndex'] = 2
        if scs == 15:
            pdsch_cfg['NrOfSymbols'] = 8
        else:
            pdsch_cfg['NrOfSymbols'] = 4
        pdsch_cfg['ResAlloType1']['RBStart'] = int(carrier_prb_size // 2)
        pdsch_cfg['ResAlloType1']['RBSize'] = 1
        pdsch_cfg['period_in_slot'] = int(10*scs/15)
        slots = list(range(1,int(10*scs/15),3))
        if scs == 15:
            new_slots = [n for n in slots if (n % 5) == 3]
        else:
            new_slots = [n for n in slots if (n % 10) == 7]
        pdsch_cfg['allocated_slots'] = new_slots
        pdsch_config_list.append(pdsch_cfg)

        pdsch_cfg = copy.deepcopy(ref_pdsch_config)
        pdsch_cfg['rnti'] = 2
        pdsch_cfg['mcs_table'] = "256QAM"
        pdsch_cfg['mcs_index'] = 11 #64QAM, TargetCodeRateby1024 = 466
        pdsch_cfg['rv'] = [0]
        #38.141-1 4.9.2.3.2 defined PN23 for PDSCH seq generation, here just choose random 
        #generate the sequence
        pdsch_cfg['data_source'] = []
        pdsch_cfg['num_of_layers'] = 1
        pdsch_cfg['VRBtoPRBMapping'] = "non-interleaved"
        pdsch_cfg['StartSymbolIndex'] = 2
        if scs == 15:
            pdsch_cfg['NrOfSymbols'] = 8
        else:
            pdsch_cfg['NrOfSymbols'] = 4
        pdsch_cfg['ResAlloType1']['RBStart'] = carrier_prb_size - 1
        pdsch_cfg['ResAlloType1']['RBSize'] = 1
        pdsch_cfg['period_in_slot'] = int(10*scs/15)
        slots = list(range(2,int(10*scs/15),3))
        if scs == 15:
            new_slots = [n for n in slots if (n % 5) == 3]
        else:
            new_slots = [n for n in slots if (n % 10) == 7]
        pdsch_cfg['allocated_slots'] = new_slots
        pdsch_config_list.append(pdsch_cfg)
            
    return pdsch_config_list




