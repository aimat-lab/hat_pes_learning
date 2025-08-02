#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import euclidean
from scipy.stats import chi2
import mol_data_generation.utils.xtb_fcts as xtb
import re
import os
import time

def read_out_bonds(path_to_folder):
    bond_idx_list_long = []
    bond_atm_list_long = []
    bond_lengths_list_long = []
    #print(os.getcwd())
    with open('{}/xtb.log'.format(path_to_folder), 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if 'Bond Distances' in lines[i]:

                for j in range(i+2, i+2+50):

                    if 'sigma' in lines[j]:
                        break

                    else:
                        line_j = lines[j].split()
                        for bonds in line_j:
                            # print(bonds)
                            two_atoms = bonds.split('-')
                            atom1 = two_atoms[0]
                            one_atom = two_atoms[1].split('=')
                            atom2 = one_atom[0]
                            bond_length = float(one_atom[1])
                            #print(atom1, atom2, bond_length)

                            element1 = atom1[0]
                            idx1 = atom1[1:]

                            element2 = atom2[0]
                            idx2 = atom2[1:]

                            bond_idx_i = [int(idx1), int(idx2)]
                            bond_atm_i = [element1, element2]

                            bond_idx_list_long.append(bond_idx_i)
                            bond_atm_list_long.append(bond_atm_i)
                            bond_lengths_list_long.append(bond_length)

    bond_idx_list, bond_atm_list, bond_lengths_list = [], [], []

    idx_list = []

    for idx, bond in enumerate(bond_idx_list_long):
        if [bond[0], bond[1]] not in bond_idx_list and [bond[1], bond[0]] not in bond_idx_list:
            bond_idx_list.append(bond)
            idx_list.append(idx)

    for idx in idx_list:
        bond_atm_list.append(bond_atm_list_long[idx])
        bond_lengths_list.append(bond_lengths_list_long[idx])

    return bond_idx_list, bond_atm_list, bond_lengths_list


def get_h_aa_caps_wrong(bond_idx_list, bond_atm_list, bond_lengths_list):

    tuple_caps = []

    atms_idx_tuples = list(
        zip(bond_idx_list, bond_atm_list, bond_lengths_list))
    print(len(atms_idx_tuples))
    n_acetyl_idx = []

    for idx, tuples in enumerate(atms_idx_tuples):

        if 1 in tuples[0]:
            tuple_caps.append(tuples)

        if 2 in tuples[0] and 1 not in tuples[0]:
            tuple_caps.append(tuples)

        if 3 in tuples[0] and 2 not in tuples[0] and 5 not in tuples[0]:
            tuple_caps.append(tuples)

        if 5 in tuples[0] and 'N' in tuples[1]:
            for idx_t in tuples[0]:
                if idx_t != 5:
                    n_acetyl_idx.append(idx_t)
        if n_acetyl_idx != []:
            if n_acetyl_idx[0] in tuples[0] and 5 not in tuples[0]:
                tuple_caps.append(tuples)

            if n_acetyl_idx[0]+1 in tuples[0] and n_acetyl_idx[0] not in tuples[0]:
                tuple_caps.append(tuples)

            if n_acetyl_idx[0]+2 in tuples[0] and n_acetyl_idx[0]+1 not in tuples[0]:
                tuple_caps.append(tuples)
    print('n acetyl', n_acetyl_idx)
    # print(len(tuple_caps))

    for tuples in tuple_caps:
        atms_idx_tuples.remove(tuples)

    rmv_tpl_wo_h = []

    for tpl in atms_idx_tuples:

        if 'H' not in tpl[1]:
            rmv_tpl_wo_h.append(tpl)

    # print(len(rmv_tpl_wo_h))

    for tpl in rmv_tpl_wo_h:
        atms_idx_tuples.remove(tpl)

    return atms_idx_tuples

def get_h_aa_caps(bond_idx_list, bond_atm_list, bond_lengths_list):

    tuple_caps = []

    atms_idx_tuples = list(
        zip(bond_idx_list, bond_atm_list, bond_lengths_list))
    #print(len(atms_idx_tuples))
    
    
    
    ch_idx_all = []

    for idx, tuples in enumerate(atms_idx_tuples):
        
        if 'C' in tuples[1] and 'H' in tuples[1]:
            if tuples[1][0] == 'C':
                #idx_c = tuples[0][0]
                ch_idx_all.append(tuples[0][0])
            if tuples[1][1] == 'C':
                #idx_c = tuples[0][1]
                ch_idx_all.append(tuples[0][1])

    ch3_check_idx = []
    
    for ch_idx in ch_idx_all:
        
        ch3_idx_count = []
        for idx, tuples in enumerate(atms_idx_tuples):
            
            if ch_idx in tuples[0] and 'H' in tuples[1]:
                ch3_idx_count.append(ch_idx)
        ch3_check_idx.append(ch3_idx_count)

    #print(ch3_check_idx)
    ch3_idx = []
    for li in ch3_check_idx:
        if len(li) == 3 and li[0] not in ch3_idx:
            ch3_idx.append(li[0])
    #print(ch3_idx)

    # ch3 idx: all c idx with 3 Hs

    # next: get NH idx and c idx
    
    cn_idx = []
    n_idx = []
    
    for idx, tuples in enumerate(atms_idx_tuples):
        
        for cidx in ch3_idx:
            if cidx in tuples[0] and 'N' in tuples[1]:
                
                if 'N' in tuples[1][0]:
                    n_idx.append(tuples[0][0])
                    cn_idx.append(cidx)
                if 'N' in tuples[1][1]:
                    n_idx.append(tuples[0][1])
                    cn_idx.append(cidx)
                    
    # check if specific
    #if len(cn_idx) == 1 and len(n_idx) = 1:
    #    pass
    
    #else:
    #    pass
    #print('cn_idx', cn_idx)
    #print('n_idx', n_idx)
    if len(cn_idx) == 1 and len(n_idx) == 1:
        
        
        co1_idx = []
        nc_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):

            if n_idx[0] in tuples[0]: # and 'C' in tuples[1]
                #print('tuples 0', tuples[0])
                if tuples[1][0] == 'C' and tuples[0][0] != cn_idx[0]:
                    co1_idx.append(tuples[0][0])
                    nc_idx.append(n_idx[0])
                    #print('C cn 0', tuples[0][0])
                if tuples[1][1] == 'C' and tuples[0][1] != cn_idx[0]:
                    co1_idx.append(tuples[0][1])
                    nc_idx.append(n_idx[0])
                    #print('C cn 0', tuples[0][1])
        
        #print('co1_idx', co1_idx)
        if len(co1_idx) == 1:
            
            cc_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if co1_idx[0] in tuples[0]:
                    if tuples[1][0] == 'C' and tuples[0][0] != co1_idx[0]:
                        cc_idx.append(tuples[0][0])
                    if tuples[1][1] == 'C' and tuples[0][1] != co1_idx[0]:
                        cc_idx.append(tuples[0][1])
            #print('cc_idx', cc_idx)
            
            cn_idx_1 = []
            
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if cc_idx[0] in tuples[0] and 'N' in tuples[1]:
                    if 'N' in tuples[1][0]:
                        cn_idx_1.append(tuples[0][0])
                    if 'N' in tuples[1][1]:
                        cn_idx_1.append(tuples[0][1])
            
            #print('cn_idx_1', cn_idx_1)            
            nc_idx_2 = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                for cn_i in cn_idx_1:
                    if cn_i in tuples[0] and 'C' in tuples[1]:
                        if 'C' in tuples[1][0]:
                            nc_idx_2.append(tuples[0][0])
                        if 'C' in tuples[1][1]:
                            nc_idx_2.append(tuples[0][1])
            
            co_idx_2 = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                for nc_idx_2_i in nc_idx_2:
                    if nc_idx_2_i in tuples[0] and 'O' in tuples[1]:
                        co_idx_2.append(nc_idx_2_i)
        
            if len(co_idx_2) == 1:
                '''
                nc_idx_2_1 = []
                for idx, tuples in enumerate(atms_idx_tuples):
                    
                    for co_idx_2_i in co_idx_2:
                        for nc_idx_2_i in nc_idx_2:
                            
                            if co_idx_2_i in tuples[0] and nc_idx_2_i in tuples[0]:
                                if nc_idx_2_i == tuples[0][0]:
                                    nc_idx_2_1.append(tuples[0][0])
                                if nc_idx_2_i == tuples[0][1]:
                                    nc_idx_2_1.append(tuples[0][1])
                print('co_idx_2', co_idx_2)
                print('nc_idx_2', nc_idx_2)
                
                print('nc_idx_2_1', nc_idx_2_1) 
                '''
                
                ch3_2_idx = []
                for idx, tuples in enumerate(atms_idx_tuples):
                    
                    if co_idx_2[0] in tuples[0]:
                        if tuples[1][0] == 'C' and tuples[0][0] != co_idx_2[0]:
                            ch3_2_idx.append(tuples[0][0])
                        if tuples[1][1] == 'C' and tuples[0][1] != co_idx_2[0]:
                            ch3_2_idx.append(tuples[0][1])
                            
                #print('ch3_2_idx', ch3_2_idx)
                
                for idx, tuples in enumerate(atms_idx_tuples):
                    
                    if cn_idx[0] in tuples[0] and 'H' in tuples[1]:
                        tuple_caps.append(tuples)
                    if n_idx[0] in tuples[0] and 'H' in tuples[1]:
                        tuple_caps.append(tuples)
                    if cn_idx_1[0] in tuples[0] and 'H' in tuples[1]:
                        tuple_caps.append(tuples)
                    if ch3_2_idx[0] in tuples[0] and 'H' in tuples[1]:
                        tuple_caps.append(tuples)
                        
                        
                        
                        
            else:
                print('sth went wrong, check co_idx_2')
            
            
        else:
            print('sth went wrong, check co1_idx')
        
        
        #for idx, tuples in enumerate(atms_idx_tuples):
                #tuple_caps
        
    else:
        print('sth went wrong, check n_idx')
               
                
    



    for tuples in tuple_caps:
        atms_idx_tuples.remove(tuples)

    rmv_tpl_wo_h = []

    for tpl in atms_idx_tuples:

        if 'H' not in tpl[1]:
            rmv_tpl_wo_h.append(tpl)

    # print(len(rmv_tpl_wo_h))

    for tpl in rmv_tpl_wo_h:
        atms_idx_tuples.remove(tpl)

    return atms_idx_tuples


def get_h_aa_dip_caps_old(bond_idx_list, bond_atm_list, bond_lengths_list):

    tuple_caps = []

    atms_idx_tuples = list(
        zip(bond_idx_list, bond_atm_list, bond_lengths_list))
    #print(len(atms_idx_tuples))
    
    
    
    ch_idx_all = []

    for idx, tuples in enumerate(atms_idx_tuples):
        
        if 'C' in tuples[1] and 'H' in tuples[1]:
            if tuples[1][0] == 'C':
                #idx_c = tuples[0][0]
                ch_idx_all.append(tuples[0][0])
            if tuples[1][1] == 'C':
                #idx_c = tuples[0][1]
                ch_idx_all.append(tuples[0][1])

    ch3_check_idx = []
    
    for ch_idx in ch_idx_all:
        
        ch3_idx_count = []
        for idx, tuples in enumerate(atms_idx_tuples):
            
            if ch_idx in tuples[0] and 'H' in tuples[1]:
                ch3_idx_count.append(ch_idx)
        ch3_check_idx.append(ch3_idx_count)

    #print(ch3_check_idx)
    ch3_idx = []
    for li in ch3_check_idx:
        if len(li) == 3 and li[0] not in ch3_idx:
            ch3_idx.append(li[0])
    #print(ch3_idx)

    # ch3 idx: all c idx with 3 Hs

    # next: get NH idx and c idx
    
    cn_idx = []
    n_idx = []
    
    for idx, tuples in enumerate(atms_idx_tuples):
        
        for cidx in ch3_idx:
            if cidx in tuples[0] and 'N' in tuples[1]:
                
                if 'N' in tuples[1][0]:
                    n_idx.append(tuples[0][0])
                    cn_idx.append(cidx)
                if 'N' in tuples[1][1]:
                    n_idx.append(tuples[0][1])
                    cn_idx.append(cidx)
                    
    # check if specific
    #if len(cn_idx) == 1 and len(n_idx) = 1:
    #    pass
    
    #else:
    #    pass
    #print('cn_idx', cn_idx)
    #print('n_idx', n_idx)
    if len(cn_idx) == 1 and len(n_idx) == 1:
        
        
        co1_idx = []
        nc_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):

            if n_idx[0] in tuples[0]: # and 'C' in tuples[1]
                #print('tuples 0', tuples[0])
                if tuples[1][0] == 'C' and tuples[0][0] != cn_idx[0]:
                    co1_idx.append(tuples[0][0])
                    nc_idx.append(n_idx[0])
                    #print('C cn 0', tuples[0][0])
                if tuples[1][1] == 'C' and tuples[0][1] != cn_idx[0]:
                    co1_idx.append(tuples[0][1])
                    nc_idx.append(n_idx[0])
                    #print('C cn 0', tuples[0][1])
        
        #print('co1_idx', co1_idx)
        if len(co1_idx) == 1:
            
            cc_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if co1_idx[0] in tuples[0]:
                    if tuples[1][0] == 'C' and tuples[0][0] != co1_idx[0]:
                        cc_idx.append(tuples[0][0])
                    if tuples[1][1] == 'C' and tuples[0][1] != co1_idx[0]:
                        cc_idx.append(tuples[0][1])
            #print('cc_idx', cc_idx)
            
            cn_idx_1 = []
            
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if cc_idx[0] in tuples[0] and 'N' in tuples[1]:
                    if 'N' in tuples[1][0]:
                        cn_idx_1.append(tuples[0][0])
                    if 'N' in tuples[1][1]:
                        cn_idx_1.append(tuples[0][1])
            
            #print('cn_idx_1', cn_idx_1)            
            
            
            nc_idx_2 = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                for cn_i in cn_idx_1:
                    if cn_i in tuples[0] and 'C' in tuples[1]:
                        if tuples[1][0] == 'C' :
                            nc_idx_2.append(tuples[0][0])
                        if tuples[1][1] == 'C':
                            nc_idx_2.append(tuples[0][1])
            
            co_idx_2 = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                for nc_idx_2_i in nc_idx_2:
                    if nc_idx_2_i in tuples[0] and 'O' in tuples[1]:
                        co_idx_2.append(nc_idx_2_i)
            
            #print('co_idx_2', co_idx_2)
            if len(co_idx_2) == 1:
                
                c_alpha_2 = []
                for idx, tuples in enumerate(atms_idx_tuples):
                    
                    if co_idx_2[0] in tuples[0]:
                        if tuples[1][0] == 'C' and tuples[0][0] != co_idx_2[0]:
                            c_alpha_2.append(tuples[0][0])
                        if tuples[1][1] == 'C' and tuples[0][1] != co_idx_2[0]:
                            c_alpha_2.append(tuples[0][1])
                
                cn_idx_2 = []
                
                for idx, tuples in enumerate(atms_idx_tuples):
                    if c_alpha_2[0] in tuples[0] and 'N' in tuples[1]:
                        if tuples[1][0] == 'N':
                            cn_idx_2.append(tuples[0][0])
                        if tuples[1][1] == 'N':
                            cn_idx_2.append(tuples[0][1])
                #print('cn_idx_2', cn_idx_2)
                co_idx_2_cap = []
                for idx, tuples in enumerate(atms_idx_tuples):
                    if cn_idx_2[0] in tuples[0]:
                        if tuples[1][0] == 'C' and tuples[0][0] != c_alpha_2[0]:
                            co_idx_2_cap.append(tuples[0][0])
                        if tuples[1][1] == 'C' and tuples[0][1] != c_alpha_2[0]:
                            co_idx_2_cap.append(tuples[0][1])
                
                
                
                ch3_2_idx = []
                for idx, tuples in enumerate(atms_idx_tuples):
                    
                    if co_idx_2_cap[0] in tuples[0]:
                        if tuples[1][0] == 'C' and tuples[0][0] != co_idx_2_cap[0]:
                            ch3_2_idx.append(tuples[0][0])
                        if tuples[1][1] == 'C' and tuples[0][1] != co_idx_2_cap[0]:
                            ch3_2_idx.append(tuples[0][1])
                            
                #print('ch3_2_idx', ch3_2_idx)
                
                for idx, tuples in enumerate(atms_idx_tuples):
                    
                    if cn_idx[0] in tuples[0] and 'H' in tuples[1]:
                        tuple_caps.append(tuples)
                    if n_idx[0] in tuples[0] and 'H' in tuples[1]:
                        tuple_caps.append(tuples)
                    if cn_idx_1[0] in tuples[0] and 'H' in tuples[1]:
                        tuple_caps.append(tuples)
                    if cn_idx_2[0] in tuples[0] and 'H' in tuples[1]:
                        tuple_caps.append(tuples)
                    if ch3_2_idx[0] in tuples[0] and 'H' in tuples[1]:
                        tuple_caps.append(tuples)
                        
                        
                        
                        
            else:
                print('sth went wrong, check co_idx_2')
            
            
        else:
            print('sth went wrong, check co1_idx')
        
        
        #for idx, tuples in enumerate(atms_idx_tuples):
                #tuple_caps
        
    else:
        print('sth went wrong, check n_idx')
               
                
    



    for tuples in tuple_caps:
        atms_idx_tuples.remove(tuples)

    rmv_tpl_wo_h = []

    for tpl in atms_idx_tuples:

        if 'H' not in tpl[1]:
            rmv_tpl_wo_h.append(tpl)

    # print(len(rmv_tpl_wo_h))

    for tpl in rmv_tpl_wo_h:
        atms_idx_tuples.remove(tpl)

    return atms_idx_tuples


def get_h_aa_dip_caps(bond_idx_list, bond_atm_list, bond_lengths_list):

    tuple_caps = []

    atms_idx_tuples = list(
        zip(bond_idx_list, bond_atm_list, bond_lengths_list))
    #print(len(atms_idx_tuples))
    
    
    
    ch_idx_all = []

    for idx, tuples in enumerate(atms_idx_tuples):
        
        if 'C' in tuples[1] and 'H' in tuples[1]:
            if tuples[1][0] == 'C':
                #idx_c = tuples[0][0]
                ch_idx_all.append(tuples[0][0])
            if tuples[1][1] == 'C':
                #idx_c = tuples[0][1]
                ch_idx_all.append(tuples[0][1])

    ch3_check_idx = []
    
    for ch_idx in ch_idx_all:
        
        ch3_idx_count = []
        for idx, tuples in enumerate(atms_idx_tuples):
            
            if ch_idx in tuples[0] and 'H' in tuples[1]:
                ch3_idx_count.append(ch_idx)
        ch3_check_idx.append(ch3_idx_count)

    #print(ch3_check_idx)
    ch3_idx = []
    for li in ch3_check_idx:
        if len(li) == 3 and li[0] not in ch3_idx:
            ch3_idx.append(li[0])
    #print(ch3_idx)

    # ch3 idx: all c idx with 3 Hs

    # next: get NH idx and c idx
    
    cn_idx = []
    n_idx = []
    
    for idx, tuples in enumerate(atms_idx_tuples):
        
        for cidx in ch3_idx:
            if cidx in tuples[0] and 'N' in tuples[1]:
                
                if 'N' in tuples[1][0]:
                    n_idx.append(tuples[0][0])
                    cn_idx.append(cidx)
                if 'N' in tuples[1][1]:
                    n_idx.append(tuples[0][1])
                    cn_idx.append(cidx)
                    
    # check if specific
    #if len(cn_idx) == 1 and len(n_idx) = 1:
    #    pass
    
    #else:
    #    pass
    #print('cn_idx', cn_idx)
    #print('n_idx', n_idx)
    if len(cn_idx) == 1 and len(n_idx) == 1:
        
        
        co1_idx = []
        nc_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):

            if n_idx[0] in tuples[0]: # and 'C' in tuples[1]
                #print('tuples 0', tuples[0])
                if tuples[1][0] == 'C' and tuples[0][0] != cn_idx[0]:
                    co1_idx.append(tuples[0][0])
                    nc_idx.append(n_idx[0])
                    #print('C cn 0', tuples[0][0])
                if tuples[1][1] == 'C' and tuples[0][1] != cn_idx[0]:
                    co1_idx.append(tuples[0][1])
                    nc_idx.append(n_idx[0])
                    #print('C cn 0', tuples[0][1])
        
        #print('co1_idx', co1_idx)
        if len(co1_idx) == 1:
            
            cc_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if co1_idx[0] in tuples[0]:
                    if tuples[1][0] == 'C' and tuples[0][0] != co1_idx[0]:
                        cc_idx.append(tuples[0][0])
                    if tuples[1][1] == 'C' and tuples[0][1] != co1_idx[0]:
                        cc_idx.append(tuples[0][1])
            #print('cc_idx', cc_idx)
            
            cn_idx_1 = []
            
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if cc_idx[0] in tuples[0] and 'N' in tuples[1]:
                    if 'N' in tuples[1][0]:
                        cn_idx_1.append(tuples[0][0])
                    if 'N' in tuples[1][1]:
                        cn_idx_1.append(tuples[0][1])
            
            #print('cn_idx_1', cn_idx_1)            
            
            
            nc_idx_2 = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                for cn_i in cn_idx_1:
                    if cn_i in tuples[0] and 'C' in tuples[1]:
                        if tuples[1][0] == 'C' :
                            nc_idx_2.append(tuples[0][0])
                        if tuples[1][1] == 'C':
                            nc_idx_2.append(tuples[0][1])
            
            co_idx_2 = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                for nc_idx_2_i in nc_idx_2:
                    if nc_idx_2_i in tuples[0] and 'O' in tuples[1]:
                        co_idx_2.append(nc_idx_2_i)
            
            #print('co_idx_2', co_idx_2)
            if len(co_idx_2) == 1:
                
                c_alpha_2 = []
                for idx, tuples in enumerate(atms_idx_tuples):
                    
                    if co_idx_2[0] in tuples[0]:
                        if tuples[1][0] == 'C' and tuples[0][0] != co_idx_2[0]:
                            c_alpha_2.append(tuples[0][0])
                        if tuples[1][1] == 'C' and tuples[0][1] != co_idx_2[0]:
                            c_alpha_2.append(tuples[0][1])
                
                cn_idx_2 = []
                
                for idx, tuples in enumerate(atms_idx_tuples):
                    if c_alpha_2[0] in tuples[0] and 'N' in tuples[1]:
                        if tuples[1][0] == 'N':
                            cn_idx_2.append(tuples[0][0])
                        if tuples[1][1] == 'N':
                            cn_idx_2.append(tuples[0][1])
                #print('cn_idx_2', cn_idx_2)
                
                
                ###
                co_idx_2_cap = []
                for idx, tuples in enumerate(atms_idx_tuples):
                    if cn_idx_2[0] in tuples[0]:
                        if tuples[1][0] == 'C' and tuples[0][0] != c_alpha_2[0]:
                            co_idx_2_cap.append(tuples[0][0])
                        if tuples[1][1] == 'C' and tuples[0][1] != c_alpha_2[0]:
                            co_idx_2_cap.append(tuples[0][1])
                
                #print('co_idx_2_cap', co_idx_2_cap)
                
                if len(co_idx_2_cap) == 1:
                    ch3_2_idx = []
                    for idx, tuples in enumerate(atms_idx_tuples):
                        
                        if co_idx_2_cap[0] in tuples[0]:
                            if tuples[1][0] == 'C' and tuples[0][0] != co_idx_2_cap[0]:
                                ch3_2_idx.append(tuples[0][0])
                            if tuples[1][1] == 'C' and tuples[0][1] != co_idx_2_cap[0]:
                                ch3_2_idx.append(tuples[0][1])
                                
                    #print('ch3_2_idx', ch3_2_idx)
                    
                    for idx, tuples in enumerate(atms_idx_tuples):
                        
                        if cn_idx[0] in tuples[0] and 'H' in tuples[1]:
                            tuple_caps.append(tuples)
                        if n_idx[0] in tuples[0] and 'H' in tuples[1]:
                            tuple_caps.append(tuples)
                        if cn_idx_1[0] in tuples[0] and 'H' in tuples[1]:
                            tuple_caps.append(tuples)
                        if cn_idx_2[0] in tuples[0] and 'H' in tuples[1]:
                            tuple_caps.append(tuples)
                        if ch3_2_idx[0] in tuples[0] and 'H' in tuples[1]:
                            tuple_caps.append(tuples)
                        
                else:
                    nco_idx = []

                    for idx, tuples in enumerate(atms_idx_tuples):
                        for co in co_idx_2_cap:
                            if co in tuples[0] and 'O' in tuples[1]:
                                nco_idx.append(co)
                    #print('nco_idx', nco_idx)
                    ch3_2_idx = []
                    for idx, tuples in enumerate(atms_idx_tuples):
                        
                        if nco_idx[0] in tuples[0]:
                            if tuples[1][0] == 'C' and tuples[0][0] != nco_idx[0]:
                                ch3_2_idx.append(tuples[0][0])
                            if tuples[1][1] == 'C' and tuples[0][1] != nco_idx[0]:
                                ch3_2_idx.append(tuples[0][1])
                                
                    #print('ch3_2_idx', ch3_2_idx)
                    
                    for idx, tuples in enumerate(atms_idx_tuples):
                        
                        if cn_idx[0] in tuples[0] and 'H' in tuples[1]:
                            tuple_caps.append(tuples)
                        if n_idx[0] in tuples[0] and 'H' in tuples[1]:
                            tuple_caps.append(tuples)
                        if cn_idx_1[0] in tuples[0] and 'H' in tuples[1]:
                            tuple_caps.append(tuples)
                        if cn_idx_2[0] in tuples[0] and 'H' in tuples[1]:
                            tuple_caps.append(tuples)
                        if ch3_2_idx[0] in tuples[0] and 'H' in tuples[1]:
                            tuple_caps.append(tuples)
                        
                        
                        
            else:
                print('sth went wrong, check co_idx_2')
            
            
        else:
            print('sth went wrong, check co1_idx')
        
        
        #for idx, tuples in enumerate(atms_idx_tuples):
                #tuple_caps
        
    else:
        print('sth went wrong, check n_idx')
               
                
    



    for tuples in tuple_caps:
        atms_idx_tuples.remove(tuples)

    rmv_tpl_wo_h = []

    for tpl in atms_idx_tuples:

        if 'H' not in tpl[1]:
            rmv_tpl_wo_h.append(tpl)

    # print(len(rmv_tpl_wo_h))

    for tpl in rmv_tpl_wo_h:
        atms_idx_tuples.remove(tpl)

    return atms_idx_tuples

def get_h_aa_dip_caps_wrong(bond_idx_list, bond_atm_list, bond_lengths_list):

    tuple_caps = []

    atms_idx_tuples = list(
        zip(bond_idx_list, bond_atm_list, bond_lengths_list))
    #print(len(atms_idx_tuples))
    #print(atms_idx_tuples)
    
    n_acetyl_idx = []
    for idx, tuples in enumerate(atms_idx_tuples):
        if 5 in tuples[0] and 'N' in tuples[1]:
            #print(tuples)
            if tuples[1][0] == 'N':
                n_acetyl_idx.append(tuples[0][0])
            if tuples[1][1] == 'N':
                n_acetyl_idx.append(tuples[0][1])

    print('nacetyl', n_acetyl_idx)
    
    co_idx = []
    c2_idx = []
    for idx, tuples in enumerate(atms_idx_tuples):
        if n_acetyl_idx[0] in tuples[0] and 'C' in tuples[1] and 5 not in tuples[0]:
            if n_acetyl_idx[0] == tuples[0][0]:
                co_idx.append(tuples[0][1])
            if n_acetyl_idx[0] == tuples[0][1]:
                co_idx.append(tuples[0][0])
                
    
    print('co idx', co_idx)
    for idx, tuples in enumerate(atms_idx_tuples):
        if co_idx[0] in tuples[0]:
            if tuples[1][0] == 'C' and tuples[0][0] != co_idx[0]:
                c2_idx.append(tuples[0][0])
            if tuples[1][1] == 'C' and tuples[0][1] != co_idx[0]:
                c2_idx.append(tuples[0][1])
    print('c2', c2_idx)
    
    n_idx = []
    for idx, tuples in enumerate(atms_idx_tuples):
        #print(c2_idx[0])
        if c2_idx[0] in tuples[0] and 'N' in tuples[1]:
            if tuples[1][0] == 'N':
                n_idx.append(tuples[0][0])
            if tuples[1][1] == 'N':
                n_idx.append(tuples[0][1])
    
    print('n idx', n_idx)
    
    ccap2 = []

    
    for idx, tuples in enumerate(atms_idx_tuples):
        if n_idx[0] in tuples[0] and 'C' in tuples[1] and c2_idx[0] not in tuples[0]:
            if tuples[1][0] == 'C':
                ccap2.append(tuples[0][0])
            if tuples[1][1] == 'C':
                ccap2.append(tuples[0][1])
    print('ccap2', ccap2)
    
    ch3_idx = []
    for idx, tuples in enumerate(atms_idx_tuples):
        if ccap2[0] in tuples[0]:
            if tuples[1][0] == 'C' and tuples[0][0] != ccap2[0]:
                ch3_idx.append(tuples[0][0])
            if tuples[1][1] == 'C' and tuples[0][1] != ccap2[0]:
                ch3_idx.append(tuples[0][1])
            
    print('ch3', ch3_idx)
    
    for idx, tuples in enumerate(atms_idx_tuples):
        
        if 1 in tuples[0]: #and 'H' in tuples[1]
            tuple_caps.append(tuples)
     
        if 2 in tuples[0] and 1 not in tuples[0]:
            tuple_caps.append(tuples)

        if 3 in tuples[0] and 2 not in tuples[0] and 5 not in tuples[0]:
            tuple_caps.append(tuples)


        
        if n_acetyl_idx[0] in tuples[0] and 'H' in tuples[1]:
            tuple_caps.append(tuples)
        # oder H von N aus Peptidbindung hinzufügen?
        


        
        if n_idx[0] in tuples[0] and 'H' in tuples[1]:
            tuple_caps.append(tuples)
        

        if ccap2[0] in tuples[0] and 'O' in tuples[1]:
            tuple_caps.append(tuples)
        

        
        if ch3_idx[0] in tuples[0] and 'H' in tuples[1]:
            tuple_caps.append(tuples)
        
        #c2_idx if tuples[1][0] == 'C' and tuples[0][0] != co_idx[0]:
            #cc = tuples[0][0]
        

    # print(len(tuple_caps))

    for tuples in tuple_caps:
        atms_idx_tuples.remove(tuples)

    rmv_tpl_wo_h = []

    for tpl in atms_idx_tuples:

        if 'H' not in tpl[1]:
            rmv_tpl_wo_h.append(tpl)

    print(len(rmv_tpl_wo_h))

    for tpl in rmv_tpl_wo_h:
        atms_idx_tuples.remove(tpl)

    return atms_idx_tuples


def get_h_dip_prot(bond_idx_list, bond_atm_list, bond_lengths_list):
    
    tuple_caps = []

    atms_idx_tuples = list(
        zip(bond_idx_list, bond_atm_list, bond_lengths_list))
    
    nh_idx_list = []
    for idx, tuples in enumerate(atms_idx_tuples):
        
        if 'N' in tuples[1][0] and 'H' in tuples[1][1]:
            nh_idx_list.append(tuples[0][0])
        if 'N' in tuples[1][1] and 'H' in tuples[1][0]:
            nh_idx_list.append(tuples[0][1])
    
    #print('nh list',nh_idx_list)
    
    nh3_list = []
    
    for nh_idx in nh_idx_list:
        
        num_nh = nh_idx_list.count(nh_idx)
        #print(num_nh)
        if num_nh == 3 and nh_idx not in nh3_list:
            nh3_list.append(nh_idx)
    
    #print('nh3',nh3_list)
    if len(nh3_list) == 1:
        # only one nh3 group
        for idx, tuples in enumerate(atms_idx_tuples):
            
            if nh3_list[0] in tuples[0] and 'H' in tuples[1]:
                tuple_caps.append(tuples)
    
        # find co
        nc_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            
            if nh3_list[0] in tuples[0] and 'C' in tuples[1]:
                if 'C' in tuples[1][0]:
                    nc_idx.append(tuples[0][0])
                if 'C' in tuples[1][1]:
                    nc_idx.append(tuples[0][1])
        
        cc_idx = []

        for idx, tuples in enumerate(atms_idx_tuples):
            
            for nc in nc_idx:
                if nc in tuples[0]:
                    if tuples[1][0] == 'C' and tuples[0][0] != nc:
                        cc_idx.append(tuples[0][0])
                    if tuples[1][1] == 'C' and tuples[0][1] != nc:
                        cc_idx.append(tuples[0][1])
        #print('cc', cc_idx)
        
        if len(cc_idx) == 1:
            n2h_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if cc_idx[0] in tuples[0] and 'N' in tuples[1]:
                    if 'N' in tuples[1][0]:
                        n2h_idx.append(tuples[0][0])
                    if 'N' in tuples[1][1]:
                        n2h_idx.append(tuples[0][1])
            #print('n2h', n2h_idx)
            
            if len(n2h_idx) == 1:
            
                for idx, tuples in enumerate(atms_idx_tuples):
                    
                    if n2h_idx[0] in tuples[0] and 'H' in tuples[1]:
                        tuple_caps.append(tuples)
                        
            else:
                
                nh2_list = []
                
                for nh_idx in nh_idx_list:
                    
                    num_nh = nh_idx_list.count(nh_idx)
                    #print(num_nh)
                    if num_nh == 2 and nh_idx not in nh2_list:
                        nh2_list.append(nh_idx)
                
                #print('nh2 list', nh2_list)
                
                if len(nh2_list) == 1:
                    
                    for idx, tuples in enumerate(atms_idx_tuples):
                        
                        if nh2_list[0] in tuples[0] and 'H' in tuples[1]:
                            tuple_caps.append(tuples)
                
                    # find co
                    nc_idx = []
                    for idx, tuples in enumerate(atms_idx_tuples):
                        
                        if nh2_list[0] in tuples[0] and 'C' in tuples[1]:
                            if 'C' in tuples[1][0]:
                                nc_idx.append(tuples[0][0])
                            if 'C' in tuples[1][1]:
                                nc_idx.append(tuples[0][1])
                    
                    cc3_idx = []

                    for idx, tuples in enumerate(atms_idx_tuples):
                        
                        for nc in nc_idx:
                            if nc in tuples[0]:
                                if tuples[1][0] == 'C' and tuples[0][0] != nc:
                                    cc3_idx.append(tuples[0][0])
                                if tuples[1][1] == 'C' and tuples[0][1] != nc:
                                    cc3_idx.append(tuples[0][1])
                    #print('cc', cc3_idx)
                    
                    cc3_single = []
                    if len(cc3_idx) != 1:
                        for idx, tuples in enumerate(atms_idx_tuples):
                            for cc3 in cc3_idx:
                                
                                if cc3 in tuples[0]:
                                    if tuples[1][0] == 'O' and tuples[0][0]:
                                        cc3_single.append(tuples[0][1])
                                    if tuples[1][1] == 'O' and tuples[0][1]:
                                        cc3_single.append(tuples[0][0])
                        
                        n2h_idx = []
                        for idx, tuples in enumerate(atms_idx_tuples):
                            
                            if cc3_single[0] in tuples[0] and 'N' in tuples[1]:
                                if 'N' in tuples[1][0]:
                                    n2h_idx.append(tuples[0][0])
                                if 'N' in tuples[1][1]:
                                    n2h_idx.append(tuples[0][1])
                        #print('n2h', n2h_idx)
                        
                        for idx, tuples in enumerate(atms_idx_tuples):
                            
                            if n2h_idx[0] in tuples[0] and 'H' in tuples[1]:
                                tuple_caps.append(tuples)
                                
                        # rmv nh3 idx tuple from tuple_caps
                        
                        for idx, tuples in enumerate(atms_idx_tuples):
                            #for tuple_i in tuple_caps:
                            if nh3_list[0] in tuples[0] and 'H' in tuples[1]:
                                tuple_caps.remove(tuples)
                        
                    
                    if len(cc3_idx) == 1:
                        n2h_idx = []
                        for idx, tuples in enumerate(atms_idx_tuples):
                            
                            if cc3_idx[0] in tuples[0] and 'N' in tuples[1]:
                                if 'N' in tuples[1][0]:
                                    n2h_idx.append(tuples[0][0])
                                if 'N' in tuples[1][1]:
                                    n2h_idx.append(tuples[0][1])
                        #print('n2h', n2h_idx)
                        
                        for idx, tuples in enumerate(atms_idx_tuples):
                            
                            if n2h_idx[0] in tuples[0] and 'H' in tuples[1]:
                                tuple_caps.append(tuples)
                                
                        # rmv nh3 idx tuple from tuple_caps
                        
                        for idx, tuples in enumerate(atms_idx_tuples):
                            #for tuple_i in tuple_caps:
                            if nh3_list[0] in tuples[0] and 'H' in tuples[1]:
                                tuple_caps.remove(tuples)
                            

                        #print('tuple_caps', tuple_caps)
                
            
            
        #else:
            #print('sth went wrong')
            
            
            
    
    if len(nh3_list) > 1:
        
        nc_idx = []
        
        for idx, tuples in enumerate(atms_idx_tuples):
            
            for nh3_idx in nh3_list:
                
                if nh3_idx in tuples[0] and 'C' in tuples[1]:
                    if 'C' in tuples[1][0]:
                        nc_idx.append(tuples[0][0])
                    if 'C' in tuples[1][1]:
                        nc_idx.append(tuples[0][1])
        #print('nc',nc_idx)
        cc_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            
            for nc in nc_idx:
                if nc in tuples[0]:
                    if tuples[1][0] == 'C' and tuples[0][0] != nc:
                        cc_idx.append(tuples[0][0])
                    if tuples[1][1] == 'C' and tuples[0][1] != nc:
                        cc_idx.append(tuples[0][1])
        #print('cc',cc_idx)
        
        co_idx = []

        for idx, tuples in enumerate(atms_idx_tuples):
            
            for cc in cc_idx:
                if cc in tuples[0] and 'O' in tuples[1]:
                    if tuples[1][0] == 'C': #and tuples[0][0] != cc:
                        co_idx.append(tuples[0][0])
                        #cn_idx.append(tuples[0][1])
                    if tuples[1][1] == 'C': #and tuples[0][1] != cc
                        co_idx.append(tuples[0][1])
                        #cn_idx.append(tuples[0][0])
        #print('co',co_idx)
        
        if len(co_idx) == 1:
        
            n2h_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if co_idx[0] in tuples[0] and 'N' in tuples[1]:
                    if 'N' in tuples[1][0]:
                        n2h_idx.append(tuples[0][0])
                    if 'N' in tuples[1][1]:
                        n2h_idx.append(tuples[0][1])
            #print('n2h', n2h_idx)
            
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if n2h_idx[0] in tuples[0] and 'H' in tuples[1]:
                    tuple_caps.append(tuples)
    
            if len(co_idx) == 1:
                cn_idx = []
                for idx, tuples in enumerate(atms_idx_tuples):
                    
                    if co_idx[0] in tuples[0]:
                        if 'C' in tuples[1][0] and tuples[0][0] != co_idx[0]:
                            cn_idx.append(tuples[0][0])
                        if 'C' in tuples[1][1] and tuples[0][1] != co_idx[0]:
                            cn_idx.append(tuples[0][1])
                    
                #print('cn', cn_idx)
                
                n3_idx = []
                for idx, tuples in enumerate(atms_idx_tuples):
                    
                    if cn_idx[0] in tuples[0] and 'N' in tuples[1]:
                        if 'N' in tuples[1][0]:
                            n3_idx.append(tuples[0][0])
                        if 'N' in tuples[1][1]:
                            n3_idx.append(tuples[0][1])
                            
                #print('n3', n3_idx)
                
                for idx, tuples in enumerate(atms_idx_tuples):
                    
                    if n3_idx[0] in tuples[0] and 'H' in tuples[1]:
                        tuple_caps.append(tuples)
                        
        if len(co_idx) > 1:
            co_pep_idx = []
            
            for idx, tuples in enumerate(atms_idx_tuples):
                
                for co in co_idx:
                    
                    if co in tuples[0] and 'N' in tuples[1]:
                        co_pep_idx.append(co)

            #print('co_pep_idx',co_pep_idx)
            
            n2h_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if co_pep_idx[0] in tuples[0] and 'N' in tuples[1]:
                    if 'N' in tuples[1][0]:
                        n2h_idx.append(tuples[0][0])
                    if 'N' in tuples[1][1]:
                        n2h_idx.append(tuples[0][1])
            #print('n2h', n2h_idx)
            
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if n2h_idx[0] in tuples[0] and 'H' in tuples[1]:
                    tuple_caps.append(tuples)
    
            if len(co_pep_idx) == 1:
                cn_idx = []
                for idx, tuples in enumerate(atms_idx_tuples):
                    
                    if co_pep_idx[0] in tuples[0]:
                        if 'C' in tuples[1][0] and tuples[0][0] != co_idx[0]:
                            cn_idx.append(tuples[0][0])
                        if 'C' in tuples[1][1] and tuples[0][1] != co_idx[0]:
                            cn_idx.append(tuples[0][1])
                    
                #print('cn', cn_idx)
                
                n3_idx = []
                for idx, tuples in enumerate(atms_idx_tuples):
                    
                    if cn_idx[0] in tuples[0] and 'N' in tuples[1]:
                        if 'N' in tuples[1][0]:
                            n3_idx.append(tuples[0][0])
                        if 'N' in tuples[1][1]:
                            n3_idx.append(tuples[0][1])
                            
                #print('n3', n3_idx)
                
                for idx, tuples in enumerate(atms_idx_tuples):
                    
                    if n3_idx[0] in tuples[0] and 'H' in tuples[1]:
                        tuple_caps.append(tuples)
            
                
        
        #else:
            #print('sth went wrong')
        
    
    if len(nh3_list) == 0:
        
        #print('len nh3 = 0')
        
        nh2_list = []
        
        for nh_idx in nh_idx_list:
            
            num_nh = nh_idx_list.count(nh_idx)
            #print(num_nh)
            if num_nh == 2 and nh_idx not in nh2_list:
                nh2_list.append(nh_idx)
        
        #print('nh2 list', nh2_list)
        
        if len(nh2_list) == 1:
            
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if nh2_list[0] in tuples[0] and 'H' in tuples[1]:
                    tuple_caps.append(tuples)
        
            # find co
            nc_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if nh2_list[0] in tuples[0] and 'C' in tuples[1]:
                    if 'C' in tuples[1][0]:
                        nc_idx.append(tuples[0][0])
                    if 'C' in tuples[1][1]:
                        nc_idx.append(tuples[0][1])
            
            cc_idx = []

            for idx, tuples in enumerate(atms_idx_tuples):
                
                for nc in nc_idx:
                    if nc in tuples[0]:
                        if tuples[1][0] == 'C' and tuples[0][0] != nc:
                            cc_idx.append(tuples[0][0])
                        if tuples[1][1] == 'C' and tuples[0][1] != nc:
                            cc_idx.append(tuples[0][1])
            #print('cc', cc_idx)
            
            if len(cc_idx) == 1:
                n2h_idx = []
                for idx, tuples in enumerate(atms_idx_tuples):
                    
                    if cc_idx[0] in tuples[0] and 'N' in tuples[1]:
                        if 'N' in tuples[1][0]:
                            n2h_idx.append(tuples[0][0])
                        if 'N' in tuples[1][1]:
                            n2h_idx.append(tuples[0][1])
                #print('n2h', n2h_idx)
                
                for idx, tuples in enumerate(atms_idx_tuples):
                    
                    if n2h_idx[0] in tuples[0] and 'H' in tuples[1]:
                        tuple_caps.append(tuples)
            
        if len(nh2_list) > 1:
            
            ncc_idx = []
            
            for idx, tuples in enumerate(atms_idx_tuples):
                
                for nh2_idx in nh2_list:
                    
                    if nh2_idx in tuples[0] and 'C' in tuples[1]:
                        #if 'C' in tuples[1][0]:
                        ncc_idx.append(nh2_idx)
                        #if 'C' in tuples[1][1]:
                            #nc_idx.append(tuples[0][1])
            
            #print('ncc',ncc_idx)
            
            nc2_list = []
            
            for nc_idx in ncc_idx:
                
                num_nh = ncc_idx.count(nc_idx)
                #print(num_nh)
                if num_nh == 2 and nc_idx not in nc2_list:
                    nc2_list.append(nc_idx)
            
            #print('nc2 list', nc2_list)
            
            
            if len(nc2_list) == 1:
                
                for idx, tuples in enumerate(atms_idx_tuples):
                    
                    if nc2_list[0] in tuples[0] and 'H' in tuples[1]:
                        tuple_caps.append(tuples)
                        
                # find co
                nc_idx = []
                for idx, tuples in enumerate(atms_idx_tuples):
                    
                    if nc2_list[0] in tuples[0] and 'C' in tuples[1]:
                        if 'C' in tuples[1][0]:
                            nc_idx.append(tuples[0][0])
                        if 'C' in tuples[1][1]:
                            nc_idx.append(tuples[0][1])
                
                cc_idx = []

                for idx, tuples in enumerate(atms_idx_tuples):
                    
                    for nc in nc_idx:
                        if nc in tuples[0]:
                            if tuples[1][0] == 'C' and tuples[0][0] != nc:
                                cc_idx.append(tuples[0][0])
                            if tuples[1][1] == 'C' and tuples[0][1] != nc:
                                cc_idx.append(tuples[0][1])
                #print('cc', cc_idx)
                
                if len(cc_idx) == 1:
                    n2h_idx = []
                    for idx, tuples in enumerate(atms_idx_tuples):
                        
                        if cc_idx[0] in tuples[0] and 'N' in tuples[1]:
                            if 'N' in tuples[1][0]:
                                n2h_idx.append(tuples[0][0])
                            if 'N' in tuples[1][1]:
                                n2h_idx.append(tuples[0][1])
                    #print('n2h', n2h_idx)
                    
                    for idx, tuples in enumerate(atms_idx_tuples):
                        
                        if n2h_idx[0] in tuples[0] and 'H' in tuples[1]:
                            tuple_caps.append(tuples)
            

        
        
    
    
    for tuples in tuple_caps:
        atms_idx_tuples.remove(tuples)

    rmv_tpl_wo_h = []

    for tpl in atms_idx_tuples:

        if 'H' not in tpl[1]:
            rmv_tpl_wo_h.append(tpl)

    # print(len(rmv_tpl_wo_h))

    for tpl in rmv_tpl_wo_h:
        atms_idx_tuples.remove(tpl)

    return atms_idx_tuples






def get_h_dip_prot_OLD(bond_idx_list, bond_atm_list, bond_lengths_list):
    
    tuple_caps = []

    atms_idx_tuples = list(
        zip(bond_idx_list, bond_atm_list, bond_lengths_list))
    
    nh_idx_list = []
    for idx, tuples in enumerate(atms_idx_tuples):
        
        if 'N' in tuples[1][0] and 'H' in tuples[1][1]:
            nh_idx_list.append(tuples[0][0])
        if 'N' in tuples[1][1] and 'H' in tuples[1][0]:
            nh_idx_list.append(tuples[0][1])
    
    #print(nh_idx_list)
    
    nh3_list = []
    
    for nh_idx in nh_idx_list:
        
        num_nh = nh_idx_list.count(nh_idx)
        
        if num_nh == 3 and nh_idx not in nh3_list:
            nh3_list.append(nh_idx)
    
    print(nh3_list)
    if len(nh3_list) == 1:
        # only one nh3 group
        for idx, tuples in enumerate(atms_idx_tuples):
            
            if nh3_list[0] in tuples[0] and 'H' in tuples[1]:
                tuple_caps.append(tuples)
    
        # find co
        nc_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            
            if nh3_list[0] in tuples[0] and 'C' in tuples[1]:
                if 'C' in tuples[1][0]:
                    nc_idx.append(tuples[0][0])
                if 'C' in tuples[1][1]:
                    nc_idx.append(tuples[0][1])
        
        cc_idx = []

        for idx, tuples in enumerate(atms_idx_tuples):
            
            for nc in nc_idx:
                if nc in tuples[0]:
                    if tuples[1][0] == 'C' and tuples[0][0] != nc:
                        cc_idx.append(tuples[0][0])
                    if tuples[1][1] == 'C' and tuples[0][1] != nc:
                        cc_idx.append(tuples[0][1])
        #print('cc', cc_idx)
        
        if len(cc_idx) == 1:
            n2h_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if cc_idx[0] in tuples[0] and 'N' in tuples[1]:
                    if 'N' in tuples[1][0]:
                        n2h_idx.append(tuples[0][0])
                    if 'N' in tuples[1][1]:
                        n2h_idx.append(tuples[0][1])
            #print('n2h', n2h_idx)
            
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if n2h_idx[0] in tuples[0] and 'H' in tuples[1]:
                    tuple_caps.append(tuples)
            
        #else:
            #print('sth went wrong')
            
            
            
    
    if len(nh3_list) > 1:
        
        nc_idx = []
        
        for idx, tuples in enumerate(atms_idx_tuples):
            
            for nh3_idx in nh3_list:
                
                if nh3_idx in tuples[0] and 'C' in tuples[1]:
                    if 'C' in tuples[1][0]:
                        nc_idx.append(tuples[0][0])
                    if 'C' in tuples[1][1]:
                        nc_idx.append(tuples[0][1])
        #print('nc',nc_idx)
        cc_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            
            for nc in nc_idx:
                if nc in tuples[0]:
                    if tuples[1][0] == 'C' and tuples[0][0] != nc:
                        cc_idx.append(tuples[0][0])
                    if tuples[1][1] == 'C' and tuples[0][1] != nc:
                        cc_idx.append(tuples[0][1])
        #print('cc',cc_idx)
        
        co_idx = []

        for idx, tuples in enumerate(atms_idx_tuples):
            
            for cc in cc_idx:
                if cc in tuples[0] and 'O' in tuples[1]:
                    if tuples[1][0] == 'C': #and tuples[0][0] != cc:
                        co_idx.append(tuples[0][0])
                        #cn_idx.append(tuples[0][1])
                    if tuples[1][1] == 'C': #and tuples[0][1] != cc
                        co_idx.append(tuples[0][1])
                        #cn_idx.append(tuples[0][0])
        #print('co',co_idx)
        
        n2h_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            
            if co_idx[0] in tuples[0] and 'N' in tuples[1]:
                if 'N' in tuples[1][0]:
                    n2h_idx.append(tuples[0][0])
                if 'N' in tuples[1][1]:
                    n2h_idx.append(tuples[0][1])
        #print('n2h', n2h_idx)
        
        for idx, tuples in enumerate(atms_idx_tuples):
            
            if n2h_idx[0] in tuples[0] and 'H' in tuples[1]:
                tuple_caps.append(tuples)

        if len(co_idx) == 1:
            cn_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if co_idx[0] in tuples[0]:
                    if 'C' in tuples[1][0] and tuples[0][0] != co_idx[0]:
                        cn_idx.append(tuples[0][0])
                    if 'C' in tuples[1][1] and tuples[0][1] != co_idx[0]:
                        cn_idx.append(tuples[0][1])
                
            #print('cn', cn_idx)
            
            n3_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if cn_idx[0] in tuples[0] and 'N' in tuples[1]:
                    if 'N' in tuples[1][0]:
                        n3_idx.append(tuples[0][0])
                    if 'N' in tuples[1][1]:
                        n3_idx.append(tuples[0][1])
                        
            #print('n3', n3_idx)
            
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if n3_idx[0] in tuples[0] and 'H' in tuples[1]:
                    tuple_caps.append(tuples)
            
                
        
        #else:
            #print('sth went wrong')
        
    for tuples in tuple_caps:
        atms_idx_tuples.remove(tuples)

    rmv_tpl_wo_h = []

    for tpl in atms_idx_tuples:

        if 'H' not in tpl[1]:
            rmv_tpl_wo_h.append(tpl)

    # print(len(rmv_tpl_wo_h))

    for tpl in rmv_tpl_wo_h:
        atms_idx_tuples.remove(tpl)

    return atms_idx_tuples

def get_h_dip_old(bond_idx_list, bond_atm_list, bond_lengths_list):
    
    tuple_caps = []
    
    atms_idx_tuples = list(
        zip(bond_idx_list, bond_atm_list, bond_lengths_list))
    
    oc_idx = []
    for idx, tuples in enumerate(atms_idx_tuples):
        if 'O' in tuples[1]:      
            if 'C' in tuples[1]:
                #nc_tuples.append(tuples)            
                if tuples[1][0] == 'C':
                    oc_idx.append(tuples[0][0])                  
                if tuples[1][1] == 'C':
                    oc_idx.append(tuples[0][1])
                
    #print('oc_idx', oc_idx)
    coo_idx_list = []
    for idx_c in oc_idx:
        
        num_oc = oc_idx.count(idx_c)
        
        if num_oc == 2 and idx_c not in coo_idx_list:
            coo_idx_list.append(idx_c)
        

    #print('coo idx', coo_idx_list)
    
    if len(coo_idx_list) == 1:
        # only one carboxyl group
        oh_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            if coo_idx_list[0] in tuples[0] and 'O' in tuples[1]:
                if 'O' in tuples[1][0]:
                    oh_idx.append(tuples[0][0])
                if 'O' in tuples[1][1]:
                    oh_idx.append(tuples[0][1])
        
        #ho_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            for oh in oh_idx:                
                if oh in tuples[0] and 'H' in tuples[1]:
                    #print(tuples)
                    tuple_caps.append(tuples)
                    
        cc_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            if coo_idx_list[0] in tuples[0]:
                if tuples[1][0] == 'C' and tuples[0][0] != coo_idx_list[0]:
                    cc_idx.append(tuples[0][0])
                if tuples[1][1] == 'C' and tuples[0][1] != coo_idx_list[0]:
                    cc_idx.append(tuples[0][1])

        #print('cc', cc_idx)
        
        nc_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            if cc_idx[0] in tuples[0] and 'N' in tuples[1]:
                if 'N' in tuples[1][0]:
                    nc_idx.append(tuples[0][0])
                if 'N' in tuples[1][1]:
                    nc_idx.append(tuples[0][1])
        #print('nc', nc_idx)
        for idx, tuples in enumerate(atms_idx_tuples):
            if nc_idx[0] in tuples[0] and 'H' in tuples[1]:
                tuple_caps.append(tuples)
        co_idx = []       
        for idx, tuples in enumerate(atms_idx_tuples):
            if nc_idx[0] in tuples[0]:
                if 'C' in tuples[1][0] and tuples[0][0] != cc_idx[0]:
                    co_idx.append(tuples[0][0])
                if 'C' in tuples[1][1] and tuples[0][1] != cc_idx[0]:
                    co_idx.append(tuples[0][1])
        #print('co', co_idx)
            
        # ccn 
        ccn_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            
            if co_idx[0] in tuples[0]:
                if 'C' in tuples[1][0] and tuples[0][0] != co_idx[0]:
                    ccn_idx.append(tuples[0][0])
                if 'C' in tuples[1][1] and tuples[0][1] != co_idx[0]:
                    ccn_idx.append(tuples[0][1])
        #print('ccn', ccn_idx)
        
        n2c_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            if ccn_idx[0] in tuples[0] and 'N' in tuples[1]:
                if 'N' in tuples[1][0]:
                    n2c_idx.append(tuples[0][0])
                if 'N' in tuples[1][1]:
                    n2c_idx.append(tuples[0][1])
        #print('n2c', n2c_idx)
        
        for idx, tuples in enumerate(atms_idx_tuples):
            if n2c_idx[0] in tuples[0] and 'H' in tuples[1]:
                tuple_caps.append(tuples)

            
            
    
    else:
        # need to find carboxyl group
        
        cc_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            for coo_idx in coo_idx_list:
                if coo_idx in tuples[0]:
                    if 'C' in tuples[1][0] and tuples[0][0] != coo_idx:
                        cc_idx.append(tuples[0][0])
                    if 'C' in tuples[1][1] and tuples[0][1] != coo_idx:
                        cc_idx.append(tuples[0][1])
        #print('cc', cc_idx)
        
        nc_idx = []
        cn_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            for cc in cc_idx:
                if cc in tuples[0] and 'N' in tuples[1]:
                    cn_idx.append(cc)
                    if 'N' in tuples[1][0]:
                        nc_idx.append(tuples[0][0])
                    if 'N' in tuples[1][1]:
                        nc_idx.append(tuples[0][1])
        #print('nc', nc_idx)
        #print('cn', cn_idx)
        
        # add NH peptide bond to remove
        for idx, tuples in enumerate(atms_idx_tuples):
            if nc_idx[0] in tuples[0] and 'H' in tuples[1]:
                tuple_caps.append(tuples)
                
        # find oh
        cco_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            if cn_idx[0] in tuples[0]:
                if 'C' in tuples[1][0] and tuples[0][0] != cn_idx[0]:
                    cco_idx.append(tuples[0][0])
                if 'C' in tuples[1][1] and tuples[0][1] != cn_idx[0]:
                    cco_idx.append(tuples[0][1])
                
        #print('cco', cco_idx)
        
        cco = []
        for idx, tuples in enumerate(atms_idx_tuples):
            for cco_id in cco_idx:
                if cco_id in tuples[0] and 'O' in tuples[1]:
                    cco.append(cco_id)
        
        #print('cco s', cco)
        
        oc_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            if cco[0] in tuples[0] and 'O' in tuples[1]:
                if 'O' in tuples[1][0]:
                    oc_idx.append(tuples[0][0])
                if 'O' in tuples[1][1]:
                    oc_idx.append(tuples[0][1])
        #print('oc', oc_idx)
        
        oh_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            for oc in oc_idx:
                if oc in tuples[0] and 'H' in tuples[1]:
                    tuple_caps.append(tuples)
                    
        
        co_idx = []       
        for idx, tuples in enumerate(atms_idx_tuples):
            if nc_idx[0] in tuples[0]:
                if 'C' in tuples[1][0] and tuples[0][0] != cc_idx[0]:
                    co_idx.append(tuples[0][0])
                if 'C' in tuples[1][1] and tuples[0][1] != cc_idx[0]:
                    co_idx.append(tuples[0][1])
        #print('co', co_idx)
            
        # ccn 
        ccn_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            
            if co_idx[0] in tuples[0]:
                if 'C' in tuples[1][0] and tuples[0][0] != co_idx[0]:
                    ccn_idx.append(tuples[0][0])
                if 'C' in tuples[1][1] and tuples[0][1] != co_idx[0]:
                    ccn_idx.append(tuples[0][1])
        #print('ccn', ccn_idx)
        
        n2c_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            if ccn_idx[0] in tuples[0] and 'N' in tuples[1]:
                if 'N' in tuples[1][0]:
                    n2c_idx.append(tuples[0][0])
                if 'N' in tuples[1][1]:
                    n2c_idx.append(tuples[0][1])
        #print('n2c', n2c_idx)
        
        for idx, tuples in enumerate(atms_idx_tuples):
            if n2c_idx[0] in tuples[0] and 'H' in tuples[1]:
                tuple_caps.append(tuples)
        
    
    for tuples in tuple_caps:
        atms_idx_tuples.remove(tuples)

    rmv_tpl_wo_h = []

    for tpl in atms_idx_tuples:

        if 'H' not in tpl[1]:
            rmv_tpl_wo_h.append(tpl)

    # print(len(rmv_tpl_wo_h))

    for tpl in rmv_tpl_wo_h:
        atms_idx_tuples.remove(tpl)

    return atms_idx_tuples

def get_h_dip(bond_idx_list, bond_atm_list, bond_lengths_list):
    
    tuple_caps = []
    
    atms_idx_tuples = list(
        zip(bond_idx_list, bond_atm_list, bond_lengths_list))
    
    oc_idx = []
    for idx, tuples in enumerate(atms_idx_tuples):
        if 'O' in tuples[1]:      
            if 'C' in tuples[1]:
                #nc_tuples.append(tuples)            
                if tuples[1][0] == 'C':
                    oc_idx.append(tuples[0][0])                  
                if tuples[1][1] == 'C':
                    oc_idx.append(tuples[0][1])
                
    #print('oc_idx', oc_idx)
    coo_idx_list = []
    for idx_c in oc_idx:
        
        num_oc = oc_idx.count(idx_c)
        
        if num_oc == 2 and idx_c not in coo_idx_list:
            coo_idx_list.append(idx_c)
        

    #print('coo idx', coo_idx_list)
    
    if len(coo_idx_list) == 1:
        # only one carboxyl group
        oh_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            if coo_idx_list[0] in tuples[0] and 'O' in tuples[1]:
                if 'O' in tuples[1][0]:
                    oh_idx.append(tuples[0][0])
                if 'O' in tuples[1][1]:
                    oh_idx.append(tuples[0][1])
        
        #ho_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            for oh in oh_idx:                
                if oh in tuples[0] and 'H' in tuples[1]:
                    #print(tuples)
                    tuple_caps.append(tuples)
                    
        cc_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            if coo_idx_list[0] in tuples[0]:
                if tuples[1][0] == 'C' and tuples[0][0] != coo_idx_list[0]:
                    cc_idx.append(tuples[0][0])
                if tuples[1][1] == 'C' and tuples[0][1] != coo_idx_list[0]:
                    cc_idx.append(tuples[0][1])

        #print('cc', cc_idx)
        
        nc_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            if cc_idx[0] in tuples[0] and 'N' in tuples[1]:
                if 'N' in tuples[1][0]:
                    nc_idx.append(tuples[0][0])
                if 'N' in tuples[1][1]:
                    nc_idx.append(tuples[0][1])
        #print('nc', nc_idx)
        for idx, tuples in enumerate(atms_idx_tuples):
            if nc_idx[0] in tuples[0] and 'H' in tuples[1]:
                tuple_caps.append(tuples)
        co_idx = []       
        for idx, tuples in enumerate(atms_idx_tuples):
            if nc_idx[0] in tuples[0]:
                if 'C' in tuples[1][0] and tuples[0][0] != cc_idx[0]:
                    co_idx.append(tuples[0][0])
                if 'C' in tuples[1][1] and tuples[0][1] != cc_idx[0]:
                    co_idx.append(tuples[0][1])
        #print('co', co_idx)
        #### part for PROLINE
        if len(co_idx) == 1:
           
                
            # ccn 
            ccn_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if co_idx[0] in tuples[0]:
                    if 'C' in tuples[1][0] and tuples[0][0] != co_idx[0]:
                        ccn_idx.append(tuples[0][0])
                    if 'C' in tuples[1][1] and tuples[0][1] != co_idx[0]:
                        ccn_idx.append(tuples[0][1])
            #print('ccn', ccn_idx)
            
            n2c_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                if ccn_idx[0] in tuples[0] and 'N' in tuples[1]:
                    if 'N' in tuples[1][0]:
                        n2c_idx.append(tuples[0][0])
                    if 'N' in tuples[1][1]:
                        n2c_idx.append(tuples[0][1])
            #print('n2c', n2c_idx)
            
            for idx, tuples in enumerate(atms_idx_tuples):
                if n2c_idx[0] in tuples[0] and 'H' in tuples[1]:
                    tuple_caps.append(tuples)
                   
        else:
            #### part for PROLINE start
            nco_idx = []

            for idx, tuples in enumerate(atms_idx_tuples):
                for co in co_idx:
                    if co in tuples[0] and 'O' in tuples[1]:
                        nco_idx.append(co)

            #print('nco', nco_idx)
            # ccn 
            ccn_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if nco_idx[0] in tuples[0]:
                    if 'C' in tuples[1][0] and tuples[0][0] != nco_idx[0]:
                        ccn_idx.append(tuples[0][0])
                    if 'C' in tuples[1][1] and tuples[0][1] != nco_idx[0]:
                        ccn_idx.append(tuples[0][1])
            #print('ccn', ccn_idx)
            
            n2c_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                if ccn_idx[0] in tuples[0] and 'N' in tuples[1]:
                    if 'N' in tuples[1][0]:
                        n2c_idx.append(tuples[0][0])
                    if 'N' in tuples[1][1]:
                        n2c_idx.append(tuples[0][1])
            #print('n2c', n2c_idx)
            
            for idx, tuples in enumerate(atms_idx_tuples):
                if n2c_idx[0] in tuples[0] and 'H' in tuples[1]:
                    tuple_caps.append(tuples)

            #### part for PROLINE end  
            
    
    else:
        # need to find carboxyl group
        
        cc_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            for coo_idx in coo_idx_list:
                if coo_idx in tuples[0]:
                    if 'C' in tuples[1][0] and tuples[0][0] != coo_idx:
                        cc_idx.append(tuples[0][0])
                    if 'C' in tuples[1][1] and tuples[0][1] != coo_idx:
                        cc_idx.append(tuples[0][1])
        #print('cc', cc_idx)
        
        nc_idx = []
        cn_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            for cc in cc_idx:
                if cc in tuples[0] and 'N' in tuples[1]:
                    cn_idx.append(cc)
                    if 'N' in tuples[1][0]:
                        nc_idx.append(tuples[0][0])
                    if 'N' in tuples[1][1]:
                        nc_idx.append(tuples[0][1])
        #print('nc', nc_idx)
        #print('cn', cn_idx)
        
        # add NH peptide bond to remove
        for idx, tuples in enumerate(atms_idx_tuples):
            if nc_idx[0] in tuples[0] and 'H' in tuples[1]:
                tuple_caps.append(tuples)
                
        # find oh
        cco_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            if cn_idx[0] in tuples[0]:
                if 'C' in tuples[1][0] and tuples[0][0] != cn_idx[0]:
                    cco_idx.append(tuples[0][0])
                if 'C' in tuples[1][1] and tuples[0][1] != cn_idx[0]:
                    cco_idx.append(tuples[0][1])
                
        #print('cco', cco_idx)
        
        cco = []
        for idx, tuples in enumerate(atms_idx_tuples):
            for cco_id in cco_idx:
                if cco_id in tuples[0] and 'O' in tuples[1]:
                    cco.append(cco_id)
        
        #print('cco s', cco)
        
        oc_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            if cco[0] in tuples[0] and 'O' in tuples[1]:
                if 'O' in tuples[1][0]:
                    oc_idx.append(tuples[0][0])
                if 'O' in tuples[1][1]:
                    oc_idx.append(tuples[0][1])
        #print('oc', oc_idx)
        
        oh_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            for oc in oc_idx:
                if oc in tuples[0] and 'H' in tuples[1]:
                    tuple_caps.append(tuples)
                    
        
        co_idx = []       
        for idx, tuples in enumerate(atms_idx_tuples):
            if nc_idx[0] in tuples[0]:
                if 'C' in tuples[1][0] and tuples[0][0] != cc_idx[0]:
                    co_idx.append(tuples[0][0])
                if 'C' in tuples[1][1] and tuples[0][1] != cc_idx[0]:
                    co_idx.append(tuples[0][1])
        #print('co', co_idx)
            
        # ccn ###
        if len(co_idx) == 1:
            ccn_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if co_idx[0] in tuples[0]:
                    if 'C' in tuples[1][0] and tuples[0][0] != co_idx[0]:
                        ccn_idx.append(tuples[0][0])
                    if 'C' in tuples[1][1] and tuples[0][1] != co_idx[0]:
                        ccn_idx.append(tuples[0][1])
            #print('ccn', ccn_idx)
            
            n2c_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                if ccn_idx[0] in tuples[0] and 'N' in tuples[1]:
                    if 'N' in tuples[1][0]:
                        n2c_idx.append(tuples[0][0])
                    if 'N' in tuples[1][1]:
                        n2c_idx.append(tuples[0][1])
            #print('n2c', n2c_idx)
            
            for idx, tuples in enumerate(atms_idx_tuples):
                if n2c_idx[0] in tuples[0] and 'H' in tuples[1]:
                    tuple_caps.append(tuples)
                    
        else:
            #print('len co idx != 1')
            cn_idx2 = []
            for idx, tuples in enumerate(atms_idx_tuples):
                for co2 in co_idx:
                    if co2 in tuples[0]:
                        if 'O' in tuples[1][0]:
                            cn_idx2.append(tuples[0][1])
                        if 'O' in tuples[1][1]:
                            cn_idx2.append(tuples[0][0])
            #print('cc', cn_idx2)
            
            ccn_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if cn_idx2[0] in tuples[0]:
                    if 'C' in tuples[1][0] and tuples[0][0] != cn_idx2[0]:
                        ccn_idx.append(tuples[0][0])
                    if 'C' in tuples[1][1] and tuples[0][1] != cn_idx2[0]:
                        ccn_idx.append(tuples[0][1])
            #print('ccn', ccn_idx)
            
            n2c_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                if ccn_idx[0] in tuples[0] and 'N' in tuples[1]:
                    if 'N' in tuples[1][0]:
                        n2c_idx.append(tuples[0][0])
                    if 'N' in tuples[1][1]:
                        n2c_idx.append(tuples[0][1])
            #print('n2c', n2c_idx)
            
            for idx, tuples in enumerate(atms_idx_tuples):
                if n2c_idx[0] in tuples[0] and 'H' in tuples[1]:
                    tuple_caps.append(tuples)
                    
           
            
            
        
    
    for tuples in tuple_caps:
        atms_idx_tuples.remove(tuples)

    rmv_tpl_wo_h = []

    for tpl in atms_idx_tuples:

        if 'H' not in tpl[1]:
            rmv_tpl_wo_h.append(tpl)

    # print(len(rmv_tpl_wo_h))

    for tpl in rmv_tpl_wo_h:
        atms_idx_tuples.remove(tpl)

    return atms_idx_tuples

def get_h_dip_old2(bond_idx_list, bond_atm_list, bond_lengths_list):
    
    tuple_caps = []
    
    atms_idx_tuples = list(
        zip(bond_idx_list, bond_atm_list, bond_lengths_list))
    
    oc_idx = []
    for idx, tuples in enumerate(atms_idx_tuples):
        if 'O' in tuples[1]:      
            if 'C' in tuples[1]:
                #nc_tuples.append(tuples)            
                if tuples[1][0] == 'C':
                    oc_idx.append(tuples[0][0])                  
                if tuples[1][1] == 'C':
                    oc_idx.append(tuples[0][1])
                
    #print('oc_idx', oc_idx)
    coo_idx_list = []
    for idx_c in oc_idx:
        
        num_oc = oc_idx.count(idx_c)
        
        if num_oc == 2 and idx_c not in coo_idx_list:
            coo_idx_list.append(idx_c)
        

    #print('coo idx', coo_idx_list)
    
    if len(coo_idx_list) == 1:
        # only one carboxyl group
        oh_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            if coo_idx_list[0] in tuples[0] and 'O' in tuples[1]:
                if 'O' in tuples[1][0]:
                    oh_idx.append(tuples[0][0])
                if 'O' in tuples[1][1]:
                    oh_idx.append(tuples[0][1])
        
        #ho_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            for oh in oh_idx:                
                if oh in tuples[0] and 'H' in tuples[1]:
                    #print(tuples)
                    tuple_caps.append(tuples)
                    
        cc_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            if coo_idx_list[0] in tuples[0]:
                if tuples[1][0] == 'C' and tuples[0][0] != coo_idx_list[0]:
                    cc_idx.append(tuples[0][0])
                if tuples[1][1] == 'C' and tuples[0][1] != coo_idx_list[0]:
                    cc_idx.append(tuples[0][1])

        #print('cc', cc_idx)
        
        nc_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            if cc_idx[0] in tuples[0] and 'N' in tuples[1]:
                if 'N' in tuples[1][0]:
                    nc_idx.append(tuples[0][0])
                if 'N' in tuples[1][1]:
                    nc_idx.append(tuples[0][1])
        #print('nc', nc_idx)
        for idx, tuples in enumerate(atms_idx_tuples):
            if nc_idx[0] in tuples[0] and 'H' in tuples[1]:
                tuple_caps.append(tuples)
        co_idx = []       
        for idx, tuples in enumerate(atms_idx_tuples):
            if nc_idx[0] in tuples[0]:
                if 'C' in tuples[1][0] and tuples[0][0] != cc_idx[0]:
                    co_idx.append(tuples[0][0])
                if 'C' in tuples[1][1] and tuples[0][1] != cc_idx[0]:
                    co_idx.append(tuples[0][1])
        #print('co', co_idx)
        #### part for PROLINE
        if len(co_idx) == 1:
           
                
            # ccn 
            ccn_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if co_idx[0] in tuples[0]:
                    if 'C' in tuples[1][0] and tuples[0][0] != co_idx[0]:
                        ccn_idx.append(tuples[0][0])
                    if 'C' in tuples[1][1] and tuples[0][1] != co_idx[0]:
                        ccn_idx.append(tuples[0][1])
            #print('ccn', ccn_idx)
            
            n2c_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                if ccn_idx[0] in tuples[0] and 'N' in tuples[1]:
                    if 'N' in tuples[1][0]:
                        n2c_idx.append(tuples[0][0])
                    if 'N' in tuples[1][1]:
                        n2c_idx.append(tuples[0][1])
            #print('n2c', n2c_idx)
            
            for idx, tuples in enumerate(atms_idx_tuples):
                if n2c_idx[0] in tuples[0] and 'H' in tuples[1]:
                    tuple_caps.append(tuples)
                   
        else:
            #### part for PROLINE start
            nco_idx = []

            for idx, tuples in enumerate(atms_idx_tuples):
                for co in co_idx:
                    if co in tuples[0] and 'O' in tuples[1]:
                        nco_idx.append(co)

            #print('nco', nco_idx)
            # ccn 
            ccn_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if nco_idx[0] in tuples[0]:
                    if 'C' in tuples[1][0] and tuples[0][0] != nco_idx[0]:
                        ccn_idx.append(tuples[0][0])
                    if 'C' in tuples[1][1] and tuples[0][1] != nco_idx[0]:
                        ccn_idx.append(tuples[0][1])
            #print('ccn', ccn_idx)
            
            n2c_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                if ccn_idx[0] in tuples[0] and 'N' in tuples[1]:
                    if 'N' in tuples[1][0]:
                        n2c_idx.append(tuples[0][0])
                    if 'N' in tuples[1][1]:
                        n2c_idx.append(tuples[0][1])
            #print('n2c', n2c_idx)
            
            for idx, tuples in enumerate(atms_idx_tuples):
                if n2c_idx[0] in tuples[0] and 'H' in tuples[1]:
                    tuple_caps.append(tuples)

            #### part for PROLINE end  
            
    
    else:
        # need to find carboxyl group
        
        cc_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            for coo_idx in coo_idx_list:
                if coo_idx in tuples[0]:
                    if 'C' in tuples[1][0] and tuples[0][0] != coo_idx:
                        cc_idx.append(tuples[0][0])
                    if 'C' in tuples[1][1] and tuples[0][1] != coo_idx:
                        cc_idx.append(tuples[0][1])
        #print('cc', cc_idx)
        
        nc_idx = []
        cn_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            for cc in cc_idx:
                if cc in tuples[0] and 'N' in tuples[1]:
                    cn_idx.append(cc)
                    if 'N' in tuples[1][0]:
                        nc_idx.append(tuples[0][0])
                    if 'N' in tuples[1][1]:
                        nc_idx.append(tuples[0][1])
        #print('nc', nc_idx)
        #print('cn', cn_idx)
        
        # add NH peptide bond to remove
        for idx, tuples in enumerate(atms_idx_tuples):
            if nc_idx[0] in tuples[0] and 'H' in tuples[1]:
                tuple_caps.append(tuples)
                
        # find oh
        cco_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            if cn_idx[0] in tuples[0]:
                if 'C' in tuples[1][0] and tuples[0][0] != cn_idx[0]:
                    cco_idx.append(tuples[0][0])
                if 'C' in tuples[1][1] and tuples[0][1] != cn_idx[0]:
                    cco_idx.append(tuples[0][1])
                
        #print('cco', cco_idx)
        
        cco = []
        for idx, tuples in enumerate(atms_idx_tuples):
            for cco_id in cco_idx:
                if cco_id in tuples[0] and 'O' in tuples[1]:
                    cco.append(cco_id)
        
        #print('cco s', cco)
        
        oc_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            if cco[0] in tuples[0] and 'O' in tuples[1]:
                if 'O' in tuples[1][0]:
                    oc_idx.append(tuples[0][0])
                if 'O' in tuples[1][1]:
                    oc_idx.append(tuples[0][1])
        #print('oc', oc_idx)
        
        oh_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            for oc in oc_idx:
                if oc in tuples[0] and 'H' in tuples[1]:
                    tuple_caps.append(tuples)
                    
        
        co_idx = []       
        for idx, tuples in enumerate(atms_idx_tuples):
            if nc_idx[0] in tuples[0]:
                if 'C' in tuples[1][0] and tuples[0][0] != cc_idx[0]:
                    co_idx.append(tuples[0][0])
                if 'C' in tuples[1][1] and tuples[0][1] != cc_idx[0]:
                    co_idx.append(tuples[0][1])
        #print('co', co_idx)
            
        # ccn 
        ccn_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            
            if co_idx[0] in tuples[0]:
                if 'C' in tuples[1][0] and tuples[0][0] != co_idx[0]:
                    ccn_idx.append(tuples[0][0])
                if 'C' in tuples[1][1] and tuples[0][1] != co_idx[0]:
                    ccn_idx.append(tuples[0][1])
        #print('ccn', ccn_idx)
        
        n2c_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            if ccn_idx[0] in tuples[0] and 'N' in tuples[1]:
                if 'N' in tuples[1][0]:
                    n2c_idx.append(tuples[0][0])
                if 'N' in tuples[1][1]:
                    n2c_idx.append(tuples[0][1])
        #print('n2c', n2c_idx)
        
        for idx, tuples in enumerate(atms_idx_tuples):
            if n2c_idx[0] in tuples[0] and 'H' in tuples[1]:
                tuple_caps.append(tuples)
        
    
    for tuples in tuple_caps:
        atms_idx_tuples.remove(tuples)

    rmv_tpl_wo_h = []

    for tpl in atms_idx_tuples:

        if 'H' not in tpl[1]:
            rmv_tpl_wo_h.append(tpl)

    # print(len(rmv_tpl_wo_h))

    for tpl in rmv_tpl_wo_h:
        atms_idx_tuples.remove(tpl)

    return atms_idx_tuples

def get_h_aa_prot(bond_idx_list, bond_atm_list, bond_lengths_list):
    
    tuple_caps = []

    atms_idx_tuples = list(
        zip(bond_idx_list, bond_atm_list, bond_lengths_list))
    
    nh_idx_list = []
    for idx, tuples in enumerate(atms_idx_tuples):
        
        if 'N' in tuples[1][0] and 'H' in tuples[1][1]:
            nh_idx_list.append(tuples[0][0])
        if 'N' in tuples[1][1] and 'H' in tuples[1][0]:
            nh_idx_list.append(tuples[0][1])
    
    #print(nh_idx_list)
    
    nh3_list = []
    
    for nh_idx in nh_idx_list:
        
        num_nh = nh_idx_list.count(nh_idx)
        
        if num_nh == 3 and nh_idx not in nh3_list:
            nh3_list.append(nh_idx)
    
    #print(nh3_list)
    if len(nh3_list) == 1:
        # only one nh3 group
        for idx, tuples in enumerate(atms_idx_tuples):
            
            if nh3_list[0] in tuples[0] and 'H' in tuples[1]:
                tuple_caps.append(tuples)
                                 

    
    if len(nh3_list) > 1:
        # Lysine nh3_group idx = 10
        for idx, tuples in enumerate(atms_idx_tuples):
            if 10 in tuples[0] and 'H' in tuples[1]:
                tuple_caps.append(tuples)

    
    if len(nh3_list) == 0:
        #fehler oder Proline
        for idx, tuples in enumerate(atms_idx_tuples):
            if 8 in tuples[0] and 'H' in tuples[1]:
                tuple_caps.append(tuples)
                
    for tuples in tuple_caps:
        atms_idx_tuples.remove(tuples)

    rmv_tpl_wo_h = []

    for tpl in atms_idx_tuples:

        if 'H' not in tpl[1]:
            rmv_tpl_wo_h.append(tpl)

    # print(len(rmv_tpl_wo_h))

    for tpl in rmv_tpl_wo_h:
        atms_idx_tuples.remove(tpl)

    return atms_idx_tuples

def get_h_aa_amide(bond_idx_list, bond_atm_list, bond_lengths_list):
    atms_idx_tuples = list(zip(bond_idx_list, bond_atm_list, bond_lengths_list))
    #print(atms_idx_tuples)
    #print(len(atms_idx_tuples))
    

    #nc_tuples = []
    
    
    
    c_idx = []
   
    for idx, tuples in enumerate(atms_idx_tuples):
        
        if 'N' in tuples[1]:
            
            
            if 'C' in tuples[1]:
                #nc_tuples.append(tuples)
                
                if tuples[1][0] == 'C':
                    c_idx.append(tuples[0][0])
                    
                if tuples[1][1] == 'C':
                    c_idx.append(tuples[0][1])
                
    #print(c_idx)
    co_idx = []   
   
    for idx, tuples in enumerate(atms_idx_tuples):
        
        for idx_c in c_idx:
            
            if idx_c in tuples[0]:
            
                if 'O' in tuples[1]:
                    co_idx.append(idx_c)
    #print('co_idx', co_idx)
    if len(co_idx) == 1:
        
        n_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            
            if co_idx[0] in tuples[0] and 'N' in tuples[1]:
                
                if tuples[1][0] == 'N':
                    n_idx.append(tuples[0][0])
                if tuples[1][1] == 'N':
                    n_idx.append(tuples[0][1])
                    
       #             
            if co_idx[0] in tuples[0]:
                #print(tuples[0])
                if 'C' in tuples[1]:
                    if tuples[1][0] == 'C' and tuples[0][0] != co_idx[0]:
                        cc = tuples[0][0]
                        #print('cc', cc)
                    if tuples[1][1] == 'C' and tuples[0][1] != co_idx[0]:
                        cc = tuples[0][1]
                        #print('cc', cc)
        for idx, tuples in enumerate(atms_idx_tuples):
            
            if cc in tuples[0] and 'N' in tuples[1]:
                
                if tuples[1][0] == 'N':
                    n_idx.append(tuples[0][0])
                if tuples[1][1] == 'N':
                    n_idx.append(tuples[0][1])
       #
       
        #print('n idx', n_idx)
        nh_tuples = []
        
        for idx, tuples in enumerate(atms_idx_tuples):
            for idx_nn in n_idx:
                if idx_nn in tuples[0] and 'H' in tuples[1]:
                    nh_tuples.append(tuples)
        
        #print(len(nh_tuples))
        #print(nh_tuples)
        for nh_tuple in nh_tuples:
            atms_idx_tuples.remove(nh_tuple)
            
        #print(len(atms_idx_tuples))
        
        
        rmv_tpl_wo_h = []
        
        for tpl in atms_idx_tuples:
            
            if 'H' not in tpl[1]:
                rmv_tpl_wo_h.append(tpl)
        
        #print(len(rmv_tpl_wo_h))
        
        for tpl in rmv_tpl_wo_h:
            atms_idx_tuples.remove(tpl)
        
        return atms_idx_tuples
            

        
        
        
    
    else:    
        '''
        for idx, tuples in enumerate(nc_tuples):
            
            for idx_co in co_idx:
                if idx_co not in tuples[0]:
                    nc_tuples.remove(tuples)
        '''
    
        ###
        cc_idx = []                
        for idx, tuples in enumerate(atms_idx_tuples):
            
            for idx_co in co_idx:
                
                if idx_co in tuples[0]:
                
                    if 'C' in tuples[1]:
                        if tuples[1][0] == 'C' and tuples[0][0] != idx_co:
                            cc_idx.append([idx_co,tuples[0][0]])
                        if tuples[1][1] == 'C' and tuples[0][1] != idx_co:
                            cc_idx.append([idx_co, tuples[0][1]])
        '''                
        for idx, tuples in enumerate(nc_tuples):
            
            for idx_cc in cc_idx:
                if idx_cc not in tuples[0]:
                    nc_tuples.remove(tuples)
        '''        
        #print('cc_idx', cc_idx)
        
        cn_idx = []                
        for idx, tuples in enumerate(atms_idx_tuples):
            
            for idx_cc in cc_idx:
                
                if idx_cc[1] in tuples[0]:
                
                    if 'N' in tuples[1]:
                        
                        cn_idx.append(idx_cc[0])       
        #print(cn_idx) # correct cn
        
        if len(cn_idx) == 1:
            
            
            n_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if cn_idx[0] in tuples[0] and 'N' in tuples[1]:
                    
                    if tuples[1][0] == 'N':
                        n_idx.append(tuples[0][0])
                    if tuples[1][1] == 'N':
                        n_idx.append(tuples[0][1])
                
                if cn_idx[0] in tuples[0]:
                    print(tuples[0])
                    if 'C' in tuples[1]:
                        if tuples[1][0] == 'C' and tuples[0][0] != cn_idx[0]:
                            cc = tuples[0][0]
                            print('cc', cc)
                        if tuples[1][1] == 'C' and tuples[0][1] != cn_idx[0]:
                            cc = tuples[0][1]
                            print('cc', cc)
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if cc in tuples[0] and 'N' in tuples[1]:
                    
                    if tuples[1][0] == 'N':
                        n_idx.append(tuples[0][0])
                    if tuples[1][1] == 'N':
                        n_idx.append(tuples[0][1])
                
            #print('n idx', n_idx)    
            #if len(n_idx) ==1:
            nh_tuples = []
            
            for idx, tuples in enumerate(atms_idx_tuples):
                for idx_nn in n_idx:
                    if idx_nn in tuples[0] and 'H' in tuples[1]:
                        nh_tuples.append(tuples)
            
            #print(len(nh_tuples))
            #print(nh_tuples)
            for nh_tuple in nh_tuples:
                atms_idx_tuples.remove(nh_tuple)
            
            rmv_tpl_wo_h = []
            
            for tpl in atms_idx_tuples:
                
                if 'H' not in tpl[1]:
                    rmv_tpl_wo_h.append(tpl)
            
            #print(len(rmv_tpl_wo_h))
            
            for tpl in rmv_tpl_wo_h:
                atms_idx_tuples.remove(tpl)
            
            #print(len(atms_idx_tuples))
            
            return atms_idx_tuples

            

        
        else:
           return print('something went wrong')



def get_h_aa(bond_idx_list, bond_atm_list, bond_lengths_list):
    atms_idx_tuples = list(zip(bond_idx_list, bond_atm_list, bond_lengths_list))
    #print(len(atms_idx_tuples))
    
    c_idx = []
   
    for idx, tuples in enumerate(atms_idx_tuples):
        
        if 'O' in tuples[1]:
            
            if 'C' in tuples[1]:
                #nc_tuples.append(tuples)
                
                if tuples[1][0] == 'C':
                    c_idx.append(tuples[0][0])
                    
                if tuples[1][1] == 'C':
                    c_idx.append(tuples[0][1])
                
    #print(c_idx)
    c_s_idx = []
    for idx_c in c_idx:
        if idx_c not in c_s_idx:
            c_s_idx.append(idx_c)
    #print('cs idx', c_s_idx)
    if len(c_s_idx) == 1:
        # kein andere oc Bindung, carboxyl gruppe eindeutig
        
        rm_tuple = []
        # need oh
        co_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            if c_s_idx[0] in tuples[0] and 'O' in tuples[1]:
                if tuples[1][0] == 'O':
                    co_idx.append(tuples[0][0])
                if tuples[1][1] == 'O':
                    co_idx.append(tuples[0][1])
        
        for idx, tuples in enumerate(atms_idx_tuples):
            for idx_co in co_idx:
                if idx_co in tuples[0] and 'H' in tuples[1]:
                    rm_tuple.append(tuples)
        
        
        # need nh2
        for idx, tuples in enumerate(atms_idx_tuples):
            
            if c_s_idx[0] in tuples[0]:
                if 'C' in tuples[1]:
                    if tuples[1][0] == 'C' and tuples[0][0] != c_s_idx[0]:
                        cc = tuples[0][0]
                    if tuples[1][1] == 'C' and tuples[0][1] != c_s_idx[0]:
                        cc = tuples[0][1]
        
        n_idx = []
        for idx, tuples in enumerate(atms_idx_tuples):
            if cc in tuples[0] and 'N' in tuples[1]:
                if tuples[1][0] == 'N':
                    n_idx.append(tuples[0][0])
                if tuples[1][1] == 'N':
                    n_idx.append(tuples[0][1])
        
        if len(n_idx) == 1:
                          
            for idx, tuples in enumerate(atms_idx_tuples):

                    if n_idx[0] in tuples[0] and 'H' in tuples[1]:
                        rm_tuple.append(tuples)
            
        else:
            print('sth went wrong')
        
        #print(len(rm_tuple))
        #print(rm_tuple)
        
        for xh_tuples in rm_tuple:
            atms_idx_tuples.remove(xh_tuples)
        
        #print(len(atms_idx_tuples))
        rmv_tpl_wo_h = []
        
        for tpl in atms_idx_tuples:
            
            if 'H' not in tpl[1]:
                rmv_tpl_wo_h.append(tpl)
        
        #print(len(rmv_tpl_wo_h))
        
        for tpl in rmv_tpl_wo_h:
            atms_idx_tuples.remove(tpl)
        
        #print(len(atms_idx_tuples))
        
        return atms_idx_tuples
        
        
    else:
        # check c mit 2 O gebunden c_idx

        coo_idx = []

        for cs in c_s_idx:
            num_oc = c_idx.count(cs)
            if num_oc == 2:
                coo_idx.append(cs)
        #print('coo idx', coo_idx) 
        
        if len(coo_idx) == 1:
            # carboxyl gruppe eindeutig
            rm_tuple = []
            # need oh
            co_idx = []
            
            for idx, tuples in enumerate(atms_idx_tuples):
                if coo_idx[0] in tuples[0] and 'O' in tuples[1]:
                    if tuples[1][0] == 'O':
                        co_idx.append(tuples[0][0])
                    if tuples[1][1] == 'O':
                        co_idx.append(tuples[0][1])
            
            for idx, tuples in enumerate(atms_idx_tuples):
                for idx_co in co_idx:
                    if idx_co in tuples[0] and 'H' in tuples[1]:
                        rm_tuple.append(tuples)
            
            
            # need nh2
            for idx, tuples in enumerate(atms_idx_tuples):
                
                if coo_idx[0] in tuples[0]:
                    if 'C' in tuples[1]:
                        if tuples[1][0] == 'C' and tuples[0][0] != coo_idx[0]:
                            cc = tuples[0][0]
                        if tuples[1][1] == 'C' and tuples[0][1] != coo_idx[0]:
                            cc = tuples[0][1]
            
            n_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                if cc in tuples[0] and 'N' in tuples[1]:
                    if tuples[1][0] == 'N':
                        n_idx.append(tuples[0][0])
                    if tuples[1][1] == 'N':
                        n_idx.append(tuples[0][1])
            
            if len(n_idx) == 1:
                              
                for idx, tuples in enumerate(atms_idx_tuples):

                        if n_idx[0] in tuples[0] and 'H' in tuples[1]:
                            rm_tuple.append(tuples)
                
            else:
                print('sth went wrong')
            
            #print(len(rm_tuple))
            #print(rm_tuple)
            
            for xh_tuples in rm_tuple:
                atms_idx_tuples.remove(xh_tuples)
            
            #print(len(atms_idx_tuples))
            rmv_tpl_wo_h = []
            
            for tpl in atms_idx_tuples:
                
                if 'H' not in tpl[1]:
                    rmv_tpl_wo_h.append(tpl)
            
            #print(len(rmv_tpl_wo_h))
            
            for tpl in rmv_tpl_wo_h:
                atms_idx_tuples.remove(tpl)
            
            #print(len(atms_idx_tuples))
            
            return atms_idx_tuples
            
            
        
        else:
            # 2 cooh groups
            # check next neighbours, 2nd neighbour N
            # len(coo_idx) > 1
            # need next c idx
            
            c2_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                for idx_coo in coo_idx:
                    if idx_coo in tuples[0]:
                        if 'C' in tuples[1]:
                            if tuples[1][0] == 'C' and tuples[0][0] != idx_coo:
                                c2_idx.append([idx_coo,tuples[0][0]])
                            if tuples[1][1] == 'C' and tuples[0][1] != idx_coo:
                                c2_idx.append([idx_coo,tuples[0][1]])
            
            #print('c2 idx', c2_idx)
            
            co_cor_idx = []
            for idx, tuples in enumerate(atms_idx_tuples):
                
                for idx_c2 in c2_idx:
                    
                    if idx_c2[1] in tuples[0] and 'N' in tuples[1]:
                        co_cor_idx.append(idx_c2[0])
                        
            #print('coo idx cor', co_cor_idx)
            
            if len(co_cor_idx) == 1:
                
                rm_tuple = []
                # need oh
                co_idx = []
                
                for idx, tuples in enumerate(atms_idx_tuples):
                    if co_cor_idx[0] in tuples[0] and 'O' in tuples[1]:
                        if tuples[1][0] == 'O':
                            co_idx.append(tuples[0][0])
                        if tuples[1][1] == 'O':
                            co_idx.append(tuples[0][1])
                
                for idx, tuples in enumerate(atms_idx_tuples):
                    for idx_co in co_idx:
                        if idx_co in tuples[0] and 'H' in tuples[1]:
                            rm_tuple.append(tuples)
                
                
                # need nh2
                for idx, tuples in enumerate(atms_idx_tuples):
                    
                    if co_cor_idx[0] in tuples[0]:
                        if 'C' in tuples[1]:
                            if tuples[1][0] == 'C' and tuples[0][0] != co_cor_idx[0]:
                                cc = tuples[0][0]
                            if tuples[1][1] == 'C' and tuples[0][1] != co_cor_idx[0]:
                                cc = tuples[0][1]
                
                n_idx = []
                for idx, tuples in enumerate(atms_idx_tuples):
                    if cc in tuples[0] and 'N' in tuples[1]:
                        if tuples[1][0] == 'N':
                            n_idx.append(tuples[0][0])
                        if tuples[1][1] == 'N':
                            n_idx.append(tuples[0][1])
                
                if len(n_idx) == 1:
                                  
                    for idx, tuples in enumerate(atms_idx_tuples):

                            if n_idx[0] in tuples[0] and 'H' in tuples[1]:
                                rm_tuple.append(tuples)
                    
                else:
                    print('sth went wrong')
                
                #print(len(rm_tuple))
                #print(rm_tuple)
                
                for xh_tuples in rm_tuple:
                    atms_idx_tuples.remove(xh_tuples)
                
                #print(len(atms_idx_tuples))
                rmv_tpl_wo_h = []
                
                for tpl in atms_idx_tuples:
                    
                    if 'H' not in tpl[1]:
                        rmv_tpl_wo_h.append(tpl)
                
                #print(len(rmv_tpl_wo_h))
                
                for tpl in rmv_tpl_wo_h:
                    atms_idx_tuples.remove(tpl)
                
                #print(len(atms_idx_tuples))
                
                return atms_idx_tuples
            
            else:
                print('sth went wrong.')

def get_aa_h(mol_name, bond_idx_list, bond_atm_list, bond_lengths_list, protonated=False):
    """
    Robustly determine which helper function to use for hydrogen removal based on molecule name.
    Works for single amino acids, dipeptides, and is extensible.

    Parameters
    ----------
    mol_name : str
        Molecule name, e.g. 'Alanine_nms_samples_num20_T50_Emax5'
    bond_idx_list, bond_atm_list, bond_lengths_list : list
        Bond information from the structure
    protonated : bool
        Whether to use the protonated logic

    Returns
    -------
    atms_idx_h_tuples : list
        List of tuples describing H atom indices for radicalization
    """
    # --- Find the chemistry-relevant part of the name (the residues)
    if "_nms_samples_" in mol_name:
        prefix = mol_name.split("_nms_samples_")[0]
    elif "_samples_" in mol_name:
        prefix = mol_name.split("_samples_")[0]
    else:
        prefix = mol_name  # fallback for unusual cases

    residue_names = prefix.split("_")
    n_residues = len(residue_names)

    # Uncomment for debugging:
    # print(f"DEBUG: {mol_name} => {n_residues} residues ({residue_names})")

    # --- If it's a capped peptide or amide, use existing logic
    if 'cap' in mol_name or 'amide' in mol_name:
        num_ = mol_name.count('_')
        if 'cap' in mol_name:
            if num_ == 1:
                # Capped single AA
                return get_h_aa_caps(bond_idx_list, bond_atm_list, bond_lengths_list)
            if num_ == 2:
                # Capped dipeptide
                return get_h_aa_dip_caps(bond_idx_list, bond_atm_list, bond_lengths_list)
        if 'amide' in mol_name:
            return get_h_aa_amide(bond_idx_list, bond_atm_list, bond_lengths_list)
    else:
        if not protonated:
            if n_residues == 1:
                return get_h_aa(bond_idx_list, bond_atm_list, bond_lengths_list)
            elif n_residues == 2:
                return get_h_dip(bond_idx_list, bond_atm_list, bond_lengths_list)
            else:
                raise ValueError(
                    f"[get_aa_h] Unrecognized residue count for '{mol_name}': found {n_residues} in {residue_names}"
                )
        else:
            if n_residues == 1:
                return get_h_aa_prot(bond_idx_list, bond_atm_list, bond_lengths_list)
            elif n_residues == 2:
                return get_h_dip_prot(bond_idx_list, bond_atm_list, bond_lengths_list)
            else:
                raise ValueError(
                    f"[get_aa_h] Unrecognized residue count for protonated '{mol_name}': found {n_residues} in {residue_names}"
                )


def get_aa_h2(mol_name, bond_idx_list, bond_atm_list, bond_lengths_list, protonated = False):
    # added dip with caps 
    # need to adapt to dip wo caps and tri
    
    if 'cap' in mol_name or 'amide' in mol_name:
        
        num_ = mol_name.count('_')
        
        if 'cap' in mol_name:
            if num_ == 1:
                # amino acids caps protonated (and not protonated)
                atms_idx_h_tuples = get_h_aa_caps(bond_idx_list, bond_atm_list, bond_lengths_list)
            if num_ == 2:
                # dipeptides caps protonated (and not protonated)
                
                atms_idx_h_tuples = get_h_aa_dip_caps(bond_idx_list, bond_atm_list, bond_lengths_list)
                
        if 'amide' in mol_name:
            atms_idx_h_tuples = get_h_aa_amide(
                bond_idx_list, bond_atm_list, bond_lengths_list)

    else:
        num_ = mol_name.count('_')
        
        if protonated == False:
            if num_ == 0:
                # amino acids not protonated
                atms_idx_h_tuples = get_h_aa(bond_idx_list, bond_atm_list, bond_lengths_list)
            if num_ >= 1: # ==
                # dipeptides not protonated
                atms_idx_h_tuples = get_h_dip(bond_idx_list, bond_atm_list, bond_lengths_list)
        if protonated:
            if num_ == 0:
                # amino acids protonated
                atms_idx_h_tuples = get_h_aa_prot(bond_idx_list, bond_atm_list, bond_lengths_list)
            if num_ == 1:
                # dipeptides protonated
                atms_idx_h_tuples = get_h_dip_prot(bond_idx_list, bond_atm_list, bond_lengths_list)
        
    return atms_idx_h_tuples


# choose random tuple to remove, read out H idx

def get_h_idx(atms_idx_tuples):
    rm_h_tuple = random.choice(atms_idx_tuples)
    h_idx = []
    rad_idx = []
    if rm_h_tuple[1][0] == 'H':
        h_idx.append(rm_h_tuple[0][0])
        rad_idx.append(rm_h_tuple[0][1])
    if rm_h_tuple[1][1] == 'H':
        h_idx.append(rm_h_tuple[0][1])
        rad_idx.append(rm_h_tuple[0][0])
    return h_idx[0]-1, rad_idx[0]-1


# remove element, coords, bond at this idx

def rmv_h_from_mol(h_idx, rad_idx, coords, elements, bond_idx_list, bond_atm_list, bond_lengths_list):

    coords_new = coords.copy()
    elements_new = elements.copy()
    bond_idx_list_before = bond_idx_list.copy()
    bond_atm_list_before = bond_atm_list.copy()
    bond_lengths_list_before = bond_lengths_list.copy()

    if h_idx < rad_idx:
        rad_idx_new = rad_idx-1
        
    else:
        rad_idx_new = rad_idx
    
    coords_new.pop(h_idx)
    elements_new.pop(h_idx)

    # need to adjust bond idx!!
    # cannot do bond_idx_list.pop(h_idx)
    bond_idx_list_new = []
    bond_atm_list_new = []
    bond_lenghts_list_new = []

    for i in range(len(bond_idx_list)):
        if bond_idx_list[i][0] == h_idx+1 or bond_idx_list[i][1] == h_idx+1:
            continue
        else:
            if bond_idx_list[i][0] > h_idx+1:
                bond0 = bond_idx_list[i][0]-1
            else:
                bond0 = bond_idx_list[i][0]
            if bond_idx_list[i][1] > h_idx+1:
                bond1 = bond_idx_list[i][1]-1
            else:
                bond1 = bond_idx_list[i][1]
            bond_idx_list_new.append([bond0, bond1])
            bond_atm_list_new.append(bond_atm_list[i])
            bond_lenghts_list_new.append(bond_lengths_list[i])

    return rad_idx_new, coords_new, elements_new, bond_idx_list_new, bond_atm_list_new, bond_lenghts_list_new, coords, elements, bond_idx_list_before, bond_atm_list_before, bond_lengths_list_before

def rmv_h_from_mol_intra(h_idx_2, rad_idx, h1_idx,r2_idx, coords, elements, bond_idx_list, bond_atm_list, bond_lengths_list):

    coords_before_1 = coords.copy()
    elements_before_1 = elements.copy()
    bond_idx_list_before_1 = bond_idx_list.copy()
    bond_atm_list_before_1 = bond_atm_list.copy()
    bond_lengths_list_before_1 = bond_lengths_list.copy()

    ## state 1
    if h_idx_2 < rad_idx:
        rad_idx_new = rad_idx-1
        
    else:
        rad_idx_new = rad_idx

    
    if h_idx_2 < h1_idx:
        h1_idx_new = h1_idx-1
        
    else:
        h1_idx_new = h1_idx
        
    if h_idx_2 < r2_idx:
        r2_idx_new = r2_idx-1
        
    else:
        r2_idx_new = r2_idx
    
    coords_before_1.pop(h_idx_2)
    elements_before_1.pop(h_idx_2)

    # need to adjust bond idx!!
    # cannot do bond_idx_list.pop(h_idx)
    bond_idx_list_new = []
    bond_atm_list_new = []
    bond_lenghts_list_new = []

    for i in range(len(bond_idx_list)):
        if bond_idx_list[i][0] == h_idx_2+1 or bond_idx_list[i][1] == h_idx_2+1:
            continue
        else:
            if bond_idx_list[i][0] > h_idx_2+1:
                bond0 = bond_idx_list[i][0]-1
            else:
                bond0 = bond_idx_list[i][0]
            if bond_idx_list[i][1] > h_idx_2+1:
                bond1 = bond_idx_list[i][1]-1
            else:
                bond1 = bond_idx_list[i][1]
            bond_idx_list_new.append([bond0, bond1])
            bond_atm_list_new.append(bond_atm_list[i])
            bond_lenghts_list_new.append(bond_lengths_list[i])


    ## state 2
    coords_before_2 = coords.copy()
    elements_before_2 = elements.copy()
    bond_idx_list_before = bond_idx_list_before_1.copy()
    bond_atm_list_before = bond_atm_list_before_1.copy()
    bond_lengths_list_before = bond_lengths_list_before_1.copy()
    
    if h1_idx < r2_idx:
        rad_idx_new_2 = r2_idx-1
        
    else:
        rad_idx_new_2 = r2_idx

    
    if h1_idx < h_idx_2:
        h1_idx_new_2 = h_idx_2-1
        
    else:
        h1_idx_new_2 = h_idx_2
        
    if h1_idx < rad_idx:
        r2_idx_new_2 = rad_idx-1
        
    else:
        r2_idx_new_2 = rad_idx
    
    coords_before_2.pop(h1_idx)
    elements_before_2.pop(h1_idx)

    # need to adjust bond idx!!
    # cannot do bond_idx_list.pop(h_idx)
    bond_idx_list_new_2 = []
    bond_atm_list_new_2 = []
    bond_lenghts_list_new_2 = []

    for i in range(len(bond_idx_list)):
        if bond_idx_list[i][0] == h1_idx+1 or bond_idx_list[i][1] == h1_idx+1:
            continue
        else:
            if bond_idx_list[i][0] > h1_idx+1:
                bond0 = bond_idx_list[i][0]-1
            else:
                bond0 = bond_idx_list[i][0]
            if bond_idx_list[i][1] > h1_idx+1:
                bond1 = bond_idx_list[i][1]-1
            else:
                bond1 = bond_idx_list[i][1]
            bond_idx_list_new_2.append([bond0, bond1])
            bond_atm_list_new_2.append(bond_atm_list[i])
            bond_lenghts_list_new_2.append(bond_lengths_list[i])


    return rad_idx_new,h1_idx_new, r2_idx_new, coords_before_1, elements_before_1, bond_idx_list_new, bond_atm_list_new, bond_lenghts_list_new, bond_idx_list_before, bond_atm_list_before, bond_lengths_list_before, coords[h_idx_2],  rad_idx_new_2,h1_idx_new_2, r2_idx_new_2, coords_before_2, elements_before_2, bond_idx_list_new_2, bond_atm_list_new_2, bond_lenghts_list_new_2, coords, elements, coords[h1_idx]


def rmv_h_from_mol_inter_V2(h_idx_2, rad_idx, h1_idx,r2_idx, coords, elements, bond_idx_list):

    coords_before_1 = coords.copy()
    elements_before_1 = elements.copy()
    bond_idx_list_before_1 = bond_idx_list.copy()


    ## state 1
    if h_idx_2 < rad_idx:
        rad_idx_new = rad_idx-1
        
    else:
        rad_idx_new = rad_idx

    
    if h_idx_2 < h1_idx:
        h1_idx_new = h1_idx-1
        
    else:
        h1_idx_new = h1_idx
        
    if h_idx_2 < r2_idx:
        r2_idx_new = r2_idx-1
        
    else:
        r2_idx_new = r2_idx
    
    coords_before_1.pop(h_idx_2)
    elements_before_1.pop(h_idx_2)

    # need to adjust bond idx!!
    # cannot do bond_idx_list.pop(h_idx)
    bond_idx_list_new = []


    for i in range(len(bond_idx_list)):
        if bond_idx_list[i][0] == h_idx_2+1 or bond_idx_list[i][1] == h_idx_2+1:
            continue
        else:
            if bond_idx_list[i][0] > h_idx_2+1:
                bond0 = bond_idx_list[i][0]-1
            else:
                bond0 = bond_idx_list[i][0]
            if bond_idx_list[i][1] > h_idx_2+1:
                bond1 = bond_idx_list[i][1]-1
            else:
                bond1 = bond_idx_list[i][1]
            bond_idx_list_new.append([bond0, bond1])
 


    ## state 2
    coords_before_2 = coords.copy()
    elements_before_2 = elements.copy()
    bond_idx_list_before = bond_idx_list_before_1.copy()

    
    if h1_idx < r2_idx:
        rad_idx_new_2 = r2_idx-1
        
    else:
        rad_idx_new_2 = r2_idx

    
    if h1_idx < h_idx_2:
        h1_idx_new_2 = h_idx_2-1
        
    else:
        h1_idx_new_2 = h_idx_2
        
    if h1_idx < rad_idx:
        r2_idx_new_2 = rad_idx-1
        
    else:
        r2_idx_new_2 = rad_idx
    
    coords_before_2.pop(h1_idx)
    elements_before_2.pop(h1_idx)

    # need to adjust bond idx!!
    # cannot do bond_idx_list.pop(h_idx)
    bond_idx_list_new_2 = []


    for i in range(len(bond_idx_list)):
        if bond_idx_list[i][0] == h1_idx+1 or bond_idx_list[i][1] == h1_idx+1:
            continue
        else:
            if bond_idx_list[i][0] > h1_idx+1:
                bond0 = bond_idx_list[i][0]-1
            else:
                bond0 = bond_idx_list[i][0]
            if bond_idx_list[i][1] > h1_idx+1:
                bond1 = bond_idx_list[i][1]-1
            else:
                bond1 = bond_idx_list[i][1]
            bond_idx_list_new_2.append([bond0, bond1])



    return rad_idx_new,h1_idx_new, r2_idx_new, coords_before_1, elements_before_1, bond_idx_list_new,  bond_idx_list_before,  coords[h_idx_2],  rad_idx_new_2,h1_idx_new_2, r2_idx_new_2, coords_before_2, elements_before_2, bond_idx_list_new_2, coords, elements, coords[h1_idx]





# get 1st and 2nd nearest neighbors (and H of 2nd nn)
def get_neighbors(atm_idx, bond_idx_list, bond_atm_list, num_atoms):
    
    # maybe adjust levels of nn
    
    atms_idx_tuples = list(zip(bond_idx_list, bond_atm_list))
    #print(len(atms_idx_tuples))
    
    atm_idx = atm_idx+1
    nn = []
   
    for idx, tuples in enumerate(atms_idx_tuples):
        
        if atm_idx == tuples[0][0]:
            nn.append(tuples[0][1])
        if atm_idx == tuples[0][1]:
            nn.append(tuples[0][0])
    #rint(nn)
    
    nn_2 = []
    for idx, tuples in enumerate(atms_idx_tuples):
        
        for idx_atm in nn:
            
            if idx_atm == tuples[0][0] and tuples[0][1] != atm_idx:
                nn_2.append(tuples[0][1])
            if idx_atm == tuples[0][1] and tuples[0][0] != atm_idx:
                nn_2.append(tuples[0][0])
    #print(nn_2)
    
    nn_H_3 = []
    # check H
    for idx, tuples in enumerate(atms_idx_tuples):
        
        for idx_atm in nn_2:
            
            if idx_atm == tuples[0][0] and 'H' in tuples[1][1]:
                nn_H_3.append(tuples[0][1])
            if idx_atm == tuples[0][1] and 'H' in tuples[1][0]:
                nn_H_3.append(tuples[0][0])
    #print(nn_H_3)
    
    not_freeze = []
    
    for i in nn:
        not_freeze.append(i)
    for i in nn_2:
        not_freeze.append(i)
    for i in nn_H_3:
        not_freeze.append(i)
        
    #print(not_freeze)
    freeze_list = []
    for idx in range(1,num_atoms+1):
        if idx not in not_freeze:
            freeze_list.append(idx)
            
    #print(freeze_list)
    
    return freeze_list


# function for translating H1 and R to (0,0)

def translate_to_center(coords, atm_idx_0):
    
    atm_center = np.array(coords[atm_idx_0])
    
    coords_x,coords_y,coords_z = [],[],[]
    for i in range(len(coords)):
        coords_x.append(coords[i][0])
        coords_y.append(coords[i][1])
        coords_z.append(coords[i][2])
        
    coords_x_new = coords_x-atm_center[0]
    coords_y_new = coords_y-atm_center[1]
    coords_z_new = coords_z-atm_center[2]
    
    zip_new = zip(coords_x_new,coords_y_new,coords_z_new)
    coords_new = np.array(list(zip_new))
    
    return coords_new


# mol 2 um X-H1 Achse zufällig rotieren

def rotate_radical(coords_1, coords_2, idx_H1, idx_h_bond_1):
    
    # get axis 
    coords_1_h = coords_1[idx_H1]
    coords_1_h_bond = coords_1[idx_h_bond_1]
    
    coords_axis = []
    for i in range(len(coords_1_h)):
        coords_axis_i = coords_1_h[i] - coords_1_h_bond[i]
        coords_axis.append(coords_axis_i)
    
    rot_axis_norm = coords_axis/ np.linalg.norm(coords_axis)
    
    
    rotation_degrees = random.randrange(360)
    #print(rotation_degrees)
    rotation_radians = np.radians(rotation_degrees)
    
    rotation_vector = rotation_radians * rot_axis_norm
    
    rotation = R.from_rotvec(rotation_vector)
    
    rotated_mol = rotation.apply(np.array(coords_2))
    
    return rotated_mol

#rotated_mol = rotate_radical(coords_1_new, coords_2_new, h_idx_1, h_bond_1)

# Punkt auf Sphäre um H1 mit Radius dh ziehen

def random_point_on_sphere_surface(radius):
    # Generate uniformly distributed values of u and v
    u = random.uniform(0, 1)
    v = random.uniform(0, 1)
    
    # Convert u and v to spherical coordinates
    theta = 2 * math.pi * u
    phi = math.acos(2 * v - 1)
    
    # Convert spherical coordinates to Cartesian coordinates
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)
    
    return [x, y, z]



# Mol 2 translatieren, sodass R == Psmit clash Funktion checken

def translate_radical(coords, idx_radical, rad_radius):
    
    if type(rad_radius) == list:
        rad_min = rad_radius[0]
        rad_max = rad_radius[1]
    
        radius = np.random.uniform(rad_min, rad_max) #1.0, 4.2
    #radius = 6.0
    
    if rad_radius == 'chi2':
        k = 5
        loc = 1.0
        scale_chi = 0.2
        cut = 4.0
        radius = sample_scaled_chi_square_scipy_cutoff(k, loc, scale=scale_chi, size=1, cutoff = cut)[0]
        #print('chi2 radius', radius, type(radius))
    
    point_on_sphere = random_point_on_sphere_surface(radius)
    #print('point sphere', point_on_sphere, type(point_on_sphere))
    radical_point = np.array(coords[idx_radical])
    #print('rad point', radical_point, type(radical_point))
    coords_ar = np.array(coords)
    #print(coords_ar.shape)
    displacement_vector = point_on_sphere-radical_point
    #print('displ', displacement_vector, type(displacement_vector))
    #print(displacement_vector.shape)
    coords_shifted = coords_ar+ displacement_vector
    
    return coords_shifted.tolist(), radius, point_on_sphere


def translate_radical_inter_V2(coords, idx_radical, rad_radius):
    
    if type(rad_radius) == list:
        rad_min = rad_radius[0]
        rad_max = rad_radius[1]
    
        radius = np.random.uniform(rad_min, rad_max) #1.0, 4.2
    #radius = 6.0
    
    if rad_radius == 'chi2':
        k = 5
        loc = 1.0
        scale_chi = 0.2
        cut = 4.0
        radius = sample_scaled_chi_square_scipy_cutoff(k, loc, scale=scale_chi, size=1, cutoff = cut)[0]
        #print('chi2 radius', radius, type(radius))
        #print('radius', radius)
    
    point_on_sphere = random_point_on_sphere_surface(radius)
    #print('point sphere', point_on_sphere, type(point_on_sphere))
    radical_point = np.array(coords[idx_radical])
    #print('rad point', radical_point, type(radical_point))
    coords_ar = np.array(coords)
    #print(coords_ar.shape)
    displacement_vector = point_on_sphere-radical_point
    #print('displ', displacement_vector, type(displacement_vector))
    #print(displacement_vector.shape)
    coords_shifted = coords_ar+ displacement_vector
    
    return coords_shifted.tolist(), radius, point_on_sphere


def check_coordinates_distance(coords1, coords2, skip_indices1=None, skip_indices2=None):
    """
    Function to calculate distances between all pairs of 3D coordinates, with the option to skip certain
    indices for coords1 and coords2.

    Args:
        coords1 (list): List of 3D coordinates of first set of points. Each coordinate is a list [x, y, z].
        coords2 (list): List of 3D coordinates of second set of points. Each coordinate is a list [x, y, z].
        skip_indices1 (list): List of indices of coordinates to skip for coords1. Default is None.
        skip_indices2 (list): List of indices of coordinates to skip for coords2. Default is None.

    Returns:
        list: List of distance values between all pairs of coordinates.
    """
    distances = []
    for i, coord1 in enumerate(coords1):
        for j, coord2 in enumerate(coords2):
            if skip_indices2 and j in skip_indices2 and skip_indices1 and i in skip_indices1:
                continue
            distance = euclidean(coord1, coord2)
            distances.append(distance)
    return distances


def check_H_radical_distances(coords1, coords2, elements1, h_idx, rad_idx):
    
    rad_coords = coords2[rad_idx]
    
    #h_coords1 = []
    h_rad_distances = []
    for i, coords in enumerate(coords1):
        if i == h_idx:
            continue
        
        else:
            if elements1[i] == 'H':
                distance = euclidean(rad_coords, coords)
                h_rad_distances.append(distance)
           
    return h_rad_distances


# Funktion, die geo Schritte vorher macht und distances checkt in loop bis distance >min

def find_system(coords1, coords2, h1_idx, h1_bond_idx, rad_idx, elements_1, elements_2, bond_idx_list_1, bond_idx_list_2, rad_radius, chrg, solve, rad_dist):
    
    clash = True
    
    while clash:
        
        # rotate molecule 2
        rotated_mol = rotate_radical(coords1, coords2, h1_idx, h1_bond_idx)
    
        # shift molecule 2
        coords_2_shifted, radius,point_on_sphere = translate_radical(rotated_mol, rad_idx, rad_radius) #rad_radius[0], rad_radius[1]
        
        
        # concatenate coords
        coords_plot,elements_plot, bonds_plot, rad_idx_new = shift_bonds_for_saving(coords1, coords_2_shifted,elements_1, elements_2, bond_idx_list_1, bond_idx_list_2,rad_idx)
        
        # distances
        distances = check_coordinates_distance(coords1, coords_2_shifted, skip_indices1=[h1_idx], skip_indices2=[rad_idx])

        if min(distances) >= rad_dist:
            h_rad_distances = check_H_radical_distances(coords1, coords_2_shifted, elements_1, h1_idx, rad_idx)
            #print(h_rad_distances) 
            #print(radius)
            closest_other_h = min(h_rad_distances)
            #print('found system without clash')
            if closest_other_h >= radius:
                #print('Found System without other close H')
                #print('radius=', radius)
                e_system = xtb.single_point_energy(coords_plot, elements_plot, charge = chrg, unp_e = 1, solvent = solve)
                f_system = xtb.single_force(coords_plot, elements_plot, charge = chrg, unp_e = 1, solvent = solve)
                
                try:
                    if f_system.all() != None:
                        forces = True
                except:
                    if f_system == None:
                        #print('forces none')
                        forces = False
                    
                if e_system != None and forces:
                    clash = False
                


    # if found
    #plot_molecule_sphere(coords_plot, elements_plot, bonds_plot, radius, point_on_sphere)
    #fu.exportXYZ(coords_plot, elements_plot, '{}/test_system_6.xyz'.format(outdir))

    return coords_plot,elements_plot, bonds_plot, radius, rad_idx_new, e_system, f_system

def find_system_inter_V2(coords1, coords2, h1_idx, h1_bond_idx, rad_idx, h2_idx, elements_1, elements_2, bond_idx_list_1, bond_idx_list_2, rad_radius, chrg, solve, rad_dist):
    
    clash = True
    st0 = time.time()
    while clash:
        
        elapsed_time = time.time() - st0
        if elapsed_time <= 30:
            # rotate molecule 2
            rotated_mol = rotate_radical(coords1, coords2, h1_idx, h1_bond_idx)
        
            # shift molecule 2
            coords_2_shifted, radius,point_on_sphere = translate_radical_inter_V2(rotated_mol, h2_idx, rad_radius) #rad_radius[0], rad_radius[1]
            
            
            # concatenate coords
            coords_plot,elements_plot, bonds_plot, rad_idx_new, h2_idx_new = shift_bonds_for_saving_V2(coords1, coords_2_shifted,elements_1, elements_2, bond_idx_list_1, bond_idx_list_2,rad_idx,h2_idx)
            
            # distances
            distances = check_coordinates_distance(coords1, coords_2_shifted, skip_indices1=[h1_idx], skip_indices2=[h2_idx])
    
            if min(distances) >= 1.7:
                h_rad_distances = check_H_radical_distances(coords1, coords_2_shifted, elements_1, h1_idx, h2_idx)
                #print(h_rad_distances) 
                #print(radius)
                closest_other_h = min(h_rad_distances)
                #print('found system without clash')
                if closest_other_h >= radius:
                    #print('Found System without other close H')
                    #print('radius=', radius)
                    e_system = xtb.single_point_energy(coords_plot, elements_plot, charge = chrg, unp_e = 1, solvent = solve)
                    f_system = xtb.single_force(coords_plot, elements_plot, charge = chrg, unp_e = 1, solvent = solve)
                    
                    try:
                        if f_system.all() != None:
                            forces = True
                    except:
                        if f_system == None:
                            print('forces none')
                            forces = False
                        
                    if e_system != None and forces:
                        found_system = True
                        clash = False
                        
                        
        else:
            print('find system timeout')
            found_system = False
            coords_plot,elements_plot, bonds_plot, radius, rad_idx_new, h2_idx_new, e_system, f_system = [],[],[],[],[],[],[],[]
            clash = False

                


    # if found
    #plot_molecule_sphere(coords_plot, elements_plot, bonds_plot, radius, point_on_sphere)
    #fu.exportXYZ(coords_plot, elements_plot, '{}/test_system_6.xyz'.format(outdir))

    return coords_plot,elements_plot, bonds_plot, radius, rad_idx_new, h2_idx_new, e_system, f_system, found_system




def find_system_V2(coords1, coords2, h1_idx, h1_bond_idx, rad_idx, elements_1, elements_2, bond_idx_list_1, bond_idx_list_2, rad_radius, chrg, solve):
    
    clash = True
    
    while clash:
        
        # rotate molecule 2
        rotated_mol = rotate_radical(coords1, coords2, h1_idx, h1_bond_idx)
    
        # shift molecule 2
        coords_2_shifted, radius,point_on_sphere = translate_radical(rotated_mol, rad_idx, rad_radius) #rad_radius[0], rad_radius[1]
        
        
        # concatenate coords
        coords_plot,elements_plot, bonds_plot, rad_idx_new = shift_bonds_for_saving(coords1, coords_2_shifted,elements_1, elements_2, bond_idx_list_1, bond_idx_list_2,rad_idx)
        
        distances = check_coordinates_distance(coords1, coords_2_shifted, skip_indices1=[h1_idx], skip_indices2=[rad_idx])
        if min(distances) >= 1.5:
            # test:
            # check H0-R0 distance
            r0_coords = coords_2_shifted[rad_idx]
            
            distance_h0_r0 = euclidean(coords1[h1_idx], r0_coords)
            print('h0 r0 distance', distance_h0_r0)
            #dist_h0_mol2 = check_coordinates_distance(coords1[h1_idx], coords_2_shifted,skip_indices2=[rad_idx] )
            
            dist_h0_mol2 = []
            for i, coord2 in enumerate(coords_2_shifted):
                if i == rad_idx:
                    continue
                else:
                    distance = euclidean(coords1[h1_idx], coord2)
                    dist_h0_mol2.append(distance)
    
            
            #print('h0 mol2 distance', dist_h0_mol2)
            print('min h0 mol2', min(dist_h0_mol2))
            if min(dist_h0_mol2) >= distance_h0_r0:
                e_system = xtb.single_point_energy(coords_plot, elements_plot, charge = chrg, unp_e = 1, solvent = solve)
                f_system = xtb.single_force(coords_plot, elements_plot, charge = chrg, unp_e = 1, solvent = solve)
                
                try:
                    if f_system.all() != None:
                        forces = True
                except:
                    if f_system == None:
                        #print('forces none')
                        forces = False
                    
                if e_system != None and forces:
                    clash = False
            
        '''
        # distances
        #distances = check_coordinates_distance(coords1, coords_2_shifted, skip_indices1=[h1_idx], skip_indices2=[rad_idx])

        #if min(distances) >= 2.0:
            #h_rad_distances = check_H_radical_distances(coords1, coords_2_shifted, elements_1, h1_idx, rad_idx)
            #print(h_rad_distances) 
            #print(radius)
            #closest_other_h = min(h_rad_distances)
            #print('found system without clash')
            #if closest_other_h >= radius:
                #print('Found System without other close H')
                #print('radius=', radius)
                #e_system = xtb.single_point_energy(coords_plot, elements_plot, charge = chrg, unp_e = 1, solvent = solve)
                #f_system = xtb.single_force(coords_plot, elements_plot, charge = chrg, unp_e = 1, solvent = solve)
                
                try:
                    if f_system.all() != None:
                        forces = True
                except:
                    if f_system == None:
                        #print('forces none')
                        forces = False
                    
                if e_system != None and forces:
                    clash = False
          '''      


    # if found
    #plot_molecule_sphere(coords_plot, elements_plot, bonds_plot, radius, point_on_sphere)
    #fu.exportXYZ(coords_plot, elements_plot, '{}/test_system_6.xyz'.format(outdir))

    return coords_plot,elements_plot, bonds_plot, radius, rad_idx_new, e_system, f_system



def shift_bonds_for_saving(coords1, coords2,elements1,elements2, bond_idx1, bond_idx2, rad_idx):
    coords_plot = list(coords1) + coords2
    elements_plot = elements1+elements2
    bonds1_flat = []
    for bond in bond_idx1:
        bonds1_flat.append(bond[0])
        bonds1_flat.append(bond[1])
    numb_bonds1 = max(bonds1_flat)
    
    bonds2_new = []
    for bond in bond_idx2:
        bond0 = bond[0] + numb_bonds1
        bond1 = bond[1] + numb_bonds1
        bonds2_new.append([bond0, bond1])
        
    bonds_plot = bond_idx1 + bonds2_new
    
    rad_idx_new = numb_bonds1 + rad_idx 
    
    return coords_plot, elements_plot, bonds_plot, rad_idx_new

def shift_bonds_for_saving_V2(coords1, coords2,elements1,elements2, bond_idx1, bond_idx2, rad_idx, h2_idx): #bond_idx_list_2,rad_idx, h1_idx, h1_bond_idx,h2_idx)
    coords_plot = list(coords1) + coords2
    elements_plot = elements1+elements2
    bonds1_flat = []
    for bond in bond_idx1:
        bonds1_flat.append(bond[0])
        bonds1_flat.append(bond[1])
    numb_bonds1 = max(bonds1_flat)
    
    bonds2_new = []
    for bond in bond_idx2:
        bond0 = bond[0] + numb_bonds1
        bond1 = bond[1] + numb_bonds1
        bonds2_new.append([bond0, bond1])
        
    bonds_plot = bond_idx1 + bonds2_new
    
    rad_idx_new = numb_bonds1 + rad_idx 
    h2_idx_new = numb_bonds1 + h2_idx
    
    return coords_plot, elements_plot, bonds_plot, rad_idx_new, h2_idx_new


def remove_variable_substring(main_string, pattern_to_remove):
    pattern = pattern_to_remove.replace('*', '.*')  # Convert '*' to '.*' for regex
    regex = re.compile(pattern)
    return regex.sub('', main_string)


def sample_scaled_chi_square_scipy(k, loc, scale=1.0, size=1):
    """
    Generate scaled samples from a chi-square distribution using scipy.stats.

    Parameters:
    - k: Degrees of freedom.
    - loc: Location parameter to shift the distribution.
    - scale: Scale parameter to scale the distribution (default is 1.0).
    - size: Number of samples to generate (default is 1).

    Returns:
    - samples: Array of scaled samples from the chi-square distribution.
    """
    if k <= 0:
        raise ValueError("Degrees of freedom (k) must be greater than 0.")

    chi_square_dist = chi2(df=k, loc=loc, scale=scale)
    samples = chi_square_dist.rvs(size=size)
    return samples

def sample_scaled_chi_square_scipy_cutoff(k, loc, scale=1.0, size=1, cutoff=None):
    """
    Generate scaled samples from a chi-square distribution using scipy.stats.

    Parameters:
    - k: Degrees of freedom.
    - loc: Location parameter to shift the distribution.
    - scale: Scale parameter to scale the distribution (default is 1.0).
    - size: Number of samples to generate (default is 1).
    - cutoff: Cutoff value for the generated samples (default is None, meaning no cutoff).

    Returns:
    - samples: Array of scaled samples from the chi-square distribution.
    """
    if k <= 0:
        raise ValueError("Degrees of freedom (k) must be greater than 0.")

    chi_square_dist = chi2(df=k, loc=loc, scale=scale)
    if cutoff is None:
        samples = chi_square_dist.rvs(size=size)
    else:
        # Generate more samples than needed to ensure we have enough after filtering
        extra_samples = chi_square_dist.rvs(size=size * 10)
        # Filter out values above the cutoff
        filtered_samples = extra_samples[extra_samples <= cutoff]
        # If we have more samples than needed, trim down to the required size
        samples = filtered_samples[:size]
    
    while samples.size == 0:
        print('samples empty')
        extra_samples = chi_square_dist.rvs(size=size * 10)
        # Filter out values above the cutoff
        filtered_samples = extra_samples[extra_samples <= cutoff]
        # If we have more samples than needed, trim down to the required size
        samples = filtered_samples[:size]
        

    return samples

## intra 

def get_H_dist_intra(coords, atms_idx_h_tuples_1):
    
    dist_H_idx = []
    #print(atms_idx_h_tuples_1)
    for i, tuples1 in enumerate(atms_idx_h_tuples_1):
        #print('i',i,  tuples1)
        atm_idx1 = tuples1[0][0]-1
        hi_idx = tuples1[0][1]-1
        for j, tuples2 in enumerate(atms_idx_h_tuples_1):    
            #print('j', j, tuples2)
            #print('j',j, atms_idx_h_tuples_1[j][0][0])
            atm_idx2 = tuples2[0][0]-1
            #print(atm_idx1, atm_idx2)
            if atm_idx1 == atm_idx2:
                #print('continue')
                continue
            #print(atm_idx1, atm_idx2)
            # H atom idx
            hj_idx = tuples2[0][1]-1
            #print(hi_idx, hj_idx)
            try:
                distance = euclidean(coords[hi_idx], coords[hj_idx])
                dist_tuple = [distance, hj_idx, hi_idx]
                if dist_tuple not in dist_H_idx:
                    dist_H_idx.append([distance, hi_idx, hj_idx, atm_idx1, atm_idx2])
                
                dist_er = False
                
            except:
                dist_er = True
                break

            #if j == 6:
               # break
        #break
    return dist_H_idx,dist_er


def get_H_tuple_intra(dist_H_idx_all, max_H_dist_intra):
    
    dist_tuple = []
    
    
    for i, tuples in enumerate(dist_H_idx_all):
        
        if tuples[0] <= max_H_dist_intra:
            dist_tuple.append(tuples)
    
    if dist_tuple == []:
        #print('dist tuple empty, add 0.5')
        #print('len idx', len(dist_H_idx_all))
        #print(dist_H_idx_all)
        
        for i, tuples in enumerate(dist_H_idx_all):
            
            if tuples[0] <= max_H_dist_intra+0.5:
                dist_tuple.append(tuples)
                
        
    try:
        tuple_chosen_i = np.random.choice(list(range(len(dist_tuple))), 1, replace=True)
        
        tuple_er = False
    
    except:
        tuple_er = True
        h1, h2, r1, r2 = [],[],[],[]
        return h1, h2, r1, r2, tuple_er
        
    #print(tuple_chosen_i)
    tuple_chosen = dist_tuple[tuple_chosen_i[0]]
    #print(tuple_chosen)
    hi = tuple_chosen[1]
    hj = tuple_chosen[2]

    rand_num = np.random.uniform(size = 1)
    
    if rand_num[0] > 0.5:
        h1 = hi
        h2 = hj
        r1 = tuple_chosen[4]
        r2 = tuple_chosen[3]
    else:
        h1 = hj
        h2 = hi
        r1 = tuple_chosen[3]
        r2 = tuple_chosen[4]   

    return h1, h2, r1, r2, tuple_er

