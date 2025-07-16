import csv
import math
import os
import random
import shutil
import statistics
#import os
#os.environ['R_HOME'] = r"C:\Program Files\R\R-4.4.3"

from PIL import Image

import nibabel as nib
import gzip

import struct

import numpy as np
import matplotlib.pyplot as plt
#import rpy2
#import rpy2.robjects as ro
from scipy import stats
from scipy.odr import quadratic
from scipy.stats import linregress
import sympy as sp
from sympy import *
import json

import magic


from sympy import Eq
from sympy import diff, integrals, integrate
import pandas as pd  # Pandas for handling tabular data

from joblib import Parallel, delayed


internal_gradiant_log_lb=0

def find_diag_freq(l1,num):
    freq=0
    for element in l1:
        if element ==num:
            freq+=1
    return freq

def convert_l_int(l1):
    list_int = []
    for element in l1:
        list_int.append(int(element))
    return list_int

def test_pval(t_list):
    if t_list[0]< 0.05:
        return t_list[1]
    else:
        return clear_set

def random_color():
  r = random.randint(0, 255)
  g = random.randint(0, 255)
  b = random.randint(0, 255)
  return (r, g, b)





e_constant= math.e

og_csv_file = r"C:\PYTHONPROJALL\Psych_Data_MHCL\mhcld_puf_2022.csv"

mh1_filtered =[]
mh2_filtered =[]
mh3_filtered =[]

clear_set =[0,0,0]

clear_coeff = []
for i in range(0,9):
    clear_coeff.append(0)



trauma_total= anxiety_total= \
adhd_total= conduct_total= \
dementia_total=bipolar_total =\
depress_total = odd_total = \
pervasive_dev_total= personality_d_total = \
schizo_psych_total =substance_total =other_total \
=clear_set

#print(trauma_total)



# with open(og_csv_file) as og_csv_data:
#     read_og_csv = csv.DictReader(og_csv_data)
#     for line in read_og_csv:
#         if not line['MH1']==-9:
#             mh1_filtered.append(line['MH1'])
#         if not line['MH2'] ==-9:
#             mh2_filtered.append(line['MH2'])
#         if not line['MH3'] ==9:
#             mh3_filtered.append(line['MH3'])
#
# mh1_filtered=convert_l_int(mh1_filtered)
# mh2_filtered=convert_l_int(mh2_filtered)
# mh3_filtered=convert_l_int(mh3_filtered)

# trauma_total[0],trauma_total[1],trauma_total[2] =  (find_diag_freq(mh1_filtered,1),
#                                                     find_diag_freq(mh2_filtered,1),
#                                                     find_diag_freq(mh3_filtered,1))
#
# anxiety_total[0],anxiety_total[1],anxiety_total[2] =  (find_diag_freq(mh1_filtered,2),
#                                                     find_diag_freq(mh2_filtered,2),
#                                                     find_diag_freq(mh3_filtered,2))
# adhd_total[0],adhd_total[1],adhd_total[2] =  (find_diag_freq(mh1_filtered,3),
#                                                     find_diag_freq(mh2_filtered,3),
#                                                     find_diag_freq(mh3_filtered,3))
# conduct_total[0],conduct_total[1],conduct_total[2] =  (find_diag_freq(mh1_filtered,4),
#                                                     find_diag_freq(mh2_filtered,4),
#                                                     find_diag_freq(mh3_filtered,4))
# dementia_total[0],dementia_total[1],dementia_total[2] =  (find_diag_freq(mh1_filtered,5),
#                                                     find_diag_freq(mh2_filtered,5),
#                                                     find_diag_freq(mh3_filtered,5))
# bipolar_total[0],bipolar_total[1],bipolar_total[2] =  (find_diag_freq(mh1_filtered,6),
#                                                     find_diag_freq(mh2_filtered,6),
#                                                     find_diag_freq(mh3_filtered,6))
# depress_total[0],depress_total[1],depress_total[2] =  (find_diag_freq(mh1_filtered,7),
#                                                     find_diag_freq(mh2_filtered,7),
#                                                     find_diag_freq(mh3_filtered,7))
# odd_total[0],odd_total[1],odd_total[2] = (find_diag_freq(mh1_filtered,8),
#                                                     find_diag_freq(mh2_filtered,8),
#                                                     find_diag_freq(mh3_filtered,8))
# pervasive_dev_total[0],pervasive_dev_total[1],pervasive_dev_total[2] =  (find_diag_freq(mh1_filtered,9),
#                                                     find_diag_freq(mh2_filtered,9),
#                                                     find_diag_freq(mh3_filtered,9))
# personality_d_total[0],personality_d_total[1],personality_d_total[2] =  (find_diag_freq(mh1_filtered,10),
#                                                     find_diag_freq(mh2_filtered,10),
#                                                     find_diag_freq(mh3_filtered,10))
# schizo_psych_total[0],schizo_psych_total[1],schizo_psych_total[2] =  (find_diag_freq(mh1_filtered,11),
#                                                     find_diag_freq(mh2_filtered,11),
#                                                     find_diag_freq(mh3_filtered,11))
# substance_total[0],substance_total[1],substance_total[2] =  (find_diag_freq(mh1_filtered,12),
#                                                     find_diag_freq(mh2_filtered,12),
#                                                     find_diag_freq(mh3_filtered,12))
# other_total[0],other_total[1],other_total[2] =  (find_diag_freq(mh1_filtered,13),
#                                                     find_diag_freq(mh2_filtered,13),
#                                                     find_diag_freq(mh3_filtered,13))
#
# print(f"Trauma: {trauma_total}")


# x_time_periods =[0,1,2]
#
# trauma_total=[934642,332512,110796]
# anxiety_total=[890929,648743,130307]
# adhd_total=[392393,173723,65059]
# conduct_total=[61726,24678,9177]
# dementia_total=[16216,8737,2257]
# bipolar_total=[579510,90730,23904]
# depress_total=[1433443,344609,73243]
# odd_total=[76569,54463,17261]
# pervasive_dev_total=[68282,31670,10949]
# personality_d_total=[47741,72528,39687]
# schizo_psych_total=[662497,64975,13752]
# substance_total=[236450,93840,28639]
# other_total=[499300,299461,81697]
#
# (trauma_t_pearson_pval, anxiety_t_pearson_pval, adhd_t_pearson_pval,
#  conduct_t_pearson_pval,dementia_t_pearson_pval, bipolar_t_pearson_pval,
#  depress_t_pearson_pval,odd_t_pearson_pval, pervasive_dev_t_pval,
#  personality_d_t_pval, schizo_psych_t_pval, substance_t_pval, other_t_pval)= (
#     stats.pearsonr(x_time_periods,trauma_total).pvalue,
#     stats.pearsonr(x_time_periods,anxiety_total).pvalue,
#     stats.pearsonr(x_time_periods,adhd_total).pvalue,
#     stats.pearsonr(x_time_periods,conduct_total).pvalue,
#     stats.pearsonr(x_time_periods,dementia_total).pvalue,
#     stats.pearsonr(x_time_periods,bipolar_total).pvalue,
#     stats.pearsonr(x_time_periods,depress_total).pvalue,
#     stats.pearsonr(x_time_periods,odd_total).pvalue,
#     stats.pearsonr(x_time_periods,pervasive_dev_total).pvalue,
#     stats.pearsonr(x_time_periods, personality_d_total).pvalue,
#     stats.pearsonr(x_time_periods, schizo_psych_total).pvalue,
#     stats.pearsonr(x_time_periods, substance_total).pvalue,
#     stats.pearsonr(x_time_periods, other_total).pvalue
# )
#
# diagnosis_matrix = [[trauma_t_pearson_pval,trauma_total],[anxiety_t_pearson_pval,anxiety_total], [adhd_t_pearson_pval,adhd_total],
#                     [conduct_t_pearson_pval,conduct_total], [dementia_t_pearson_pval,dementia_total], [bipolar_t_pearson_pval,bipolar_total],
#                     [depress_t_pearson_pval, depress_total] , [odd_t_pearson_pval,odd_total], [pervasive_dev_t_pval,pervasive_dev_total],
#                     [personality_d_t_pval, personality_d_total],[schizo_psych_t_pval,schizo_psych_total], [substance_t_pval, substance_total],
#                     [other_t_pval,other_total]]
#
# mh_val=1
# for row in diagnosis_matrix:
#     test_result = test_pval(row)
#     if not test_result == clear_set:
#         print(f"{mh_val}: {row}\n")
#         plt.plot(x_time_periods,test_result,random_color())
#     mh_val+=1
#
# plt.show()



################################## CHI SQUARED TEST FOR INDEPENDENCE ##################################
print("\n\nCHI_SQUARED TESTS:  \n")



def chi_squared_observed(list_data,categories):
    observed_result=[]

    for category in categories:
        observed_result.append(find_diag_freq(list_data,category))

    return observed_result

def totals_matrix(matrix):
    r_total = []
    c_total = []

    row = 0
    column = 0

    for i in range(0, len(matrix[0])):
        for j in range(0, len(matrix)):
            column += matrix[j][i]
        c_total.append(column)
        column=0




    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[0])):
            row += matrix[i][j]
        r_total.append(row)
        row=0

    return [r_total, c_total]


def expected_matrix(r_param_total, c_param_total):
    expec_matrix =[]
    epsilon = 10e-6
    obsv_matrix_total = sum(r_param_total) + epsilon
    for i in range(0,len(r_param_total)):
        expec_row =[]
        for j in range(0,len(c_param_total)):
            expec = c_param_total[j]*r_param_total[i]
            expec = expec/obsv_matrix_total
            expec_row.append(expec)
        expec_matrix.append(expec_row)
    return expec_matrix

def chi_squared_test(matrix_obsv,matrix_expec):
    chi_squared =0
    chi_compart=[]

    df = (len(matrix_obsv)-1)*(len(matrix_obsv[0])-1)

    for i in range(0,len(matrix_obsv)):
        for j in range(0,len(matrix_obsv[0])):
            chi_part_numerator = math.pow((matrix_obsv[i][j]-matrix_expec[i][j]),2)
            chi_part = chi_part_numerator/(matrix_expec[i][j])
            #print(f"O:{matrix_obsv[i][j]} E:{matrix_expec[i][j]} chi_part:{chi_part}")
            chi_compart.append(chi_part)

    #print(chi_compart)

    for compart in chi_compart:
        chi_squared+=compart

    p_value = 1-stats.chi2.cdf(chi_squared,df)

    if p_value < 0.05:
        return "REJECT Ho. Pval < 0.05"
    else:
        return "FAIL TO REJECT Ho. Pval >= 0.05"


age_obsv_matrix =[]
sex_obsv_matrix=[]
educ_obsv_matrix=[]
race_obsv_matrix=[]


with open(og_csv_file) as og_csv_data:
    og_csv_reader = csv.DictReader(og_csv_data)

    age_f1 = []
    sex_f1 = []
    educ_f1 = []
    race_f1= []
    age_f2 = []
    sex_f2 = []
    educ_f2 =[]
    race_f2= []


    for line in og_csv_reader:
        if line['MH1'] =='6' or line['MH2'] == '6' or line['MH3'] == '6':
            age_f1.append(line['AGE'])
            sex_f1.append(line['GENDER'])
            educ_f1.append(line['EDUC'])
            race_f1.append(line['RACE'])
        elif not(line['MH1'] =='-9' or line['MH2'] == '-9' or line['MH3'] == '-9'):
            age_f2.append(line['AGE'])
            sex_f2.append(line['GENDER'])
            educ_f2.append(line['EDUC'])
            race_f2.append(line['RACE'])


    for param_list in [age_f1,age_f2]:
        age_obsv_matrix.append(chi_squared_observed(param_list,['1','2','3','4','5','6','7','8','9','10','11','12','13','14']))

    for param_list in [sex_f1, sex_f2]:
        sex_obsv_matrix.append(chi_squared_observed(param_list, ['1','2']))

    for param_list in [educ_f1, educ_f2]:
        educ_obsv_matrix.append(chi_squared_observed(param_list, ['1','2','3','4','5']))

    for param_list in [race_f1, race_f2]:
        race_obsv_matrix.append(chi_squared_observed(param_list, ['1','2','3','4','5','6']))


    #print(f"AGE OBSERVED: {age_obsv_matrix}")

    age_obsv_totals = totals_matrix(age_obsv_matrix)
    sex_obsv_totals= totals_matrix(sex_obsv_matrix)
    educ_obsv_totals = totals_matrix(educ_obsv_matrix)
    race_obsv_totals = totals_matrix(race_obsv_matrix)

    age_expec_matrix = expected_matrix(age_obsv_totals[0],age_obsv_totals[1])
    sex_expec_matrix = expected_matrix(sex_obsv_totals[0], sex_obsv_totals[1])
    educ_expec_matrix = expected_matrix(educ_obsv_totals[0], educ_obsv_totals[1])
    race_expec_matrix = expected_matrix(race_obsv_totals[0], race_obsv_totals[1])

    #print(f"AGE EXPECTED: {age_expec_matrix}")


    print(f"AGE: {chi_squared_test(age_obsv_matrix,age_expec_matrix)}")
    print(f"SEX: {chi_squared_test(sex_obsv_matrix, sex_expec_matrix)}")
    print(f"EDUC: {chi_squared_test(educ_obsv_totals, educ_expec_matrix)}")
    print(f"RACE: {chi_squared_test(race_obsv_matrix, race_expec_matrix)}")




################################## CMH TEST ##################################
print("\n\n\nCMH TESTS:  \n")
def var_cmh(r_total,c_total):
    var_table_cmh = 1
    for r_t in r_total:
        var_table_cmh=var_table_cmh*r_t
    for c_t in c_total:
        var_table_cmh=var_table_cmh*c_t

    epsilon = 10 ** -6
    var_table_cmh = var_table_cmh/(sum(r_total)+epsilon)

    return var_table_cmh

def cmh_difference(matrix_obsv, matrix_expec):
    cmh_strata =0
    cmh_diff = matrix_obsv[0][0]-matrix_expec[0][0]

    return cmh_diff

def cmh_test(cmh_diff_list,var_cmh_list):
    sum_cmh_diff = sum(cmh_diff_list)
    sum_cmh_diff_sq=sum_cmh_diff**2
    sum_cmh_var = sum(var_cmh_list)

    epsilon = 10**-6

    chi_sq_chm = sum_cmh_diff_sq/(sum_cmh_var+epsilon)

    p_value = 1 - stats.chi2.cdf(chi_sq_chm,1)

    if p_value < 0.05:
        return "REJECT Ho. Pval < 0.05"
    else :
        return "FAIL TO REJECT Ho. Pval >= 0.05"



bpd_stratum1=[] # => bpd mh1 mh2 and mh3
bpd_stratum2=[] # => bpd mh1 but no bpd mh2 and mh3
bpd_stratum3=[] # => no bpd mh1 but bpd mh2 and mh3
bpd_stratum4=[] # => no bpd


s_1_substance=[0,0]
s_1_no_substance=[0,0]
s_2_substance=[0,0]
s_2_no_substance=[0,0]
s_3_substance=[0,0]
s_3_no_substance=[0,0]
s_4_substance=[0,0]
s_4_no_substance=[0,0]

with open(og_csv_file) as og_csv_data:
   og_csv_reader = csv.DictReader(og_csv_data)


   for line in og_csv_reader:

       #Strata 1
       if line['MH1']=='6' and line['MH2']=='6' and line['MH3']=='6':
           if line['SAP'] =='1':
               if line['SMISED'] =='1' or line['SMISED'] =='2':
                   s_1_substance[0]+=1
               elif line['SMISED'] =='3':
                   s_1_substance[1]+=1
           if line['SAP'] == '2':
               if line['SMISED'] == '1' or line['SMISED'] == '2':
                   s_1_no_substance[0] += 1
               elif line['SMISED'] == '3':
                   s_1_no_substance[1] += 1

       # Strata 2
       elif line['MH1']=='-6' and line['MH2']!='-9' and line['MH3'] !='-9' and line['MH2']!='6' and line['MH3']!='6':
           if line['SAP'] == '1':
               if line['SMISED'] == '1' or line['SMISED'] == '2':
                   s_2_substance[0] += 1
               elif line['SMISED'] == '3':
                   s_2_substance[1] += 1
           if line['SAP'] == '2':
               if line['SMISED'] == '1' or line['SMISED'] == '2':
                   s_2_no_substance[0] += 1
               elif line['SMISED'] == '3':
                   s_2_no_substance[1] += 1

       # Strata 3
       elif line['MH1'] !='-9' and line['MH1'] !='6' and (line['MH3']=='6' or line['MH2']== '6'):
           if line['SAP'] == '1':
               if line['SMISED'] == '1' or line['SMISED'] == '2':
                   s_3_substance[0] += 1
               elif line['SMISED'] == '3':
                   s_3_substance[1] += 1
           if line['SAP'] == '2':
               if line['SMISED'] == '1' or line['SMISED'] == '2':
                   s_3_no_substance[0] += 1
               elif line['SMISED'] == '3':
                   s_3_no_substance[1] += 1

       # Strata 4
       elif line['MH1']!='6' and line['MH2']!='6' and line['MH3']!='6' and line['MH1']!='-9' and line['MH2']!='-9' and line['MH3']!='-9':
           if line['SAP'] =='1':
               if line['SMISED'] =='1' or line['SMISED'] =='2':
                   s_4_substance[0]+=1
               elif line['SMISED'] =='3':
                   s_4_substance[1]+=1
           if line['SAP'] == '2':
               if line['SMISED'] == '1' or line['SMISED'] == '2':
                   s_4_no_substance[0] += 1
               elif line['SMISED'] == '3':
                   s_4_no_substance[1] += 1

bpd_stratum1.append(s_1_substance)
bpd_stratum1.append(s_1_no_substance)
bpd_stratum2.append(s_2_substance)
bpd_stratum2.append(s_2_no_substance)
bpd_stratum3.append(s_3_substance)
bpd_stratum3.append(s_3_no_substance)
bpd_stratum4.append(s_4_substance)
bpd_stratum4.append(s_4_substance)

#print(f"OBSV_CMH: {bpd_stratum1}")

t_1_strata = totals_matrix(bpd_stratum1)
t_2_strata = totals_matrix(bpd_stratum2)
t_3_strata = totals_matrix(bpd_stratum3)
t_4_strata = totals_matrix(bpd_stratum4)

bpd_s1_expected_matrix = expected_matrix(t_1_strata[0],t_1_strata[1])
bpd_s2_expected_matrix = expected_matrix(t_2_strata[0],t_2_strata[1])
bpd_s3_expected_matrix = expected_matrix(t_3_strata[0],t_3_strata[1])
bpd_s4_expected_matrix = expected_matrix(t_4_strata[0],t_4_strata[1])

#print(f"EXPEC_CMH: {bpd_s1_expected_matrix}")


cmh_diff_test_1 = [cmh_difference(bpd_stratum1,bpd_s1_expected_matrix),
            cmh_difference(bpd_stratum2,bpd_s2_expected_matrix),
            cmh_difference(bpd_stratum3,bpd_s3_expected_matrix),
            cmh_difference(bpd_stratum4,bpd_s4_expected_matrix),
            ]
var_strata_test_1 =[var_cmh(t_1_strata[0],t_1_strata[1]),
             var_cmh(t_2_strata[0],t_2_strata[1]),
             var_cmh(t_3_strata[0],t_3_strata[1]),
             var_cmh(t_4_strata[0],t_4_strata[1])
            ]

print(f"CMH TEST 1: {cmh_test(cmh_diff_test_1,var_strata_test_1)}")


def initalize_strata():
    rows, cols = (2, 2)
    arr = [[0]*cols]*rows

    return arr


stratum1=initalize_strata()
stratum2=initalize_strata()
stratum3=initalize_strata()
stratum4=initalize_strata()
stratum5=initalize_strata()
stratum6=initalize_strata()
stratum7=initalize_strata()
stratum8=initalize_strata()
stratum9=initalize_strata()
stratum10=initalize_strata()
stratum11=initalize_strata()
stratum12=initalize_strata()
stratum13=initalize_strata()


def create_stratum(mh1_param, simsed_param, sap_param):
    s_total = [stratum1,stratum2,stratum3,stratum4,stratum5,stratum6,
               stratum7,stratum8,stratum9,stratum10,stratum11,stratum11,
               stratum12,stratum13]

    s_strata = s_total[(int)(mh1_param)]

    if sap_param == '1':
        if simsed_param == '1' or simsed_param == '2':
            s_strata[0][0] += 1

        elif simsed_param == '3':
            s_strata[0][1] += 1

    elif sap_param == '2':
        if simsed_param or simsed_param == '2':
            s_strata[1][0] += 1

        elif simsed_param == '3':
            s_strata[1][1] += 1


with open(og_csv_file) as og_csv_data:
   og_csv_reader = csv.DictReader(og_csv_data)

   for line in og_csv_reader:

      create_stratum(line['MH1'],line['SMISED'],line['SAP'])



   t_strata_1 = totals_matrix(stratum1)
   t_strata_2 = totals_matrix(stratum2)
   t_strata_3 = totals_matrix(stratum3)
   t_strata_4 = totals_matrix(stratum4)
   t_strata_5 = totals_matrix(stratum5)
   t_strata_6 = totals_matrix(stratum6)
   t_strata_7 = totals_matrix(stratum7)
   t_strata_8 = totals_matrix(stratum8)
   t_strata_9 = totals_matrix(stratum9)
   t_strata_10 = totals_matrix(stratum10)
   t_strata_11 = totals_matrix(stratum11)
   t_strata_12 = totals_matrix(stratum12)
   t_strata_13 = totals_matrix(stratum13)

   expec_strata_1 = expected_matrix(t_strata_1[0],t_strata_1[1])
   expec_strata_2 = expected_matrix(t_strata_2[0],t_strata_2[1])
   expec_strata_3 = expected_matrix(t_strata_3[0],t_strata_3[1])
   expec_strata_4 = expected_matrix(t_strata_4[0],t_strata_4[1])
   expec_strata_5 = expected_matrix(t_strata_5[0],t_strata_5[1])
   expec_strata_6 = expected_matrix(t_strata_6[0],t_strata_6[1])
   expec_strata_7 = expected_matrix(t_strata_7[0],t_strata_7[1])
   expec_strata_8 = expected_matrix(t_strata_8[0],t_strata_8[1])
   expec_strata_9 = expected_matrix(t_strata_9[0],t_strata_9[1])
   expec_strata_10 = expected_matrix(t_strata_10[0],t_strata_10[1])
   expec_strata_11 = expected_matrix(t_strata_11[0],t_strata_11[1])
   expec_strata_12 = expected_matrix(t_strata_12[0],t_strata_12[1])
   expec_strata_13 = expected_matrix(t_strata_13[0],t_strata_13[1])


cmh_diff_test_2 = [cmh_difference(stratum1,expec_strata_1),
            cmh_difference(stratum2,expec_strata_2),
            cmh_difference(stratum3,expec_strata_3),
            cmh_difference(stratum4,expec_strata_4),
            cmh_difference(stratum5,expec_strata_5),
            cmh_difference(stratum6,expec_strata_6),
            cmh_difference(stratum7,expec_strata_7),
            cmh_difference(stratum8,expec_strata_8),
            cmh_difference(stratum9,expec_strata_9),
            cmh_difference(stratum10,expec_strata_10),
            cmh_difference(stratum11, expec_strata_11),
            cmh_difference(stratum12, expec_strata_12),
            cmh_difference(stratum13, expec_strata_13),
            ]
var_strata_test_2 =[var_cmh(t_strata_1[0],t_strata_1[1]),
             var_cmh(t_strata_2[0],t_strata_2[1]),
             var_cmh(t_strata_3[0],t_strata_3[1]),
             var_cmh(t_strata_4[0],t_strata_4[1]),
             var_cmh(t_strata_5[0],t_strata_5[1]),
             var_cmh(t_strata_6[0],t_strata_6[1]),
             var_cmh(t_strata_7[0],t_strata_7[1]),
             var_cmh(t_strata_8[0],t_strata_8[1]),
             var_cmh(t_strata_9[0],t_strata_9[1]),
             var_cmh(t_strata_10[0],t_strata_10[1]),
             var_cmh(t_strata_11[0],t_strata_11[1]),
             var_cmh(t_strata_12[0],t_strata_12[1]),
             var_cmh(t_strata_13[0],t_strata_13[1]),
            ]

print(f"CMH TEST 2: {cmh_test(cmh_diff_test_2,var_strata_test_2)}")












################################## Logistic Regression Model ##################################
print("\n\n\nLOGISTIC REGRESSION MODEL:  \n")

def initialize_pi_n(l_coef, l_var):

    intercept = l_coef[0]

    ni = intercept

    for i in range(1,len(l_coef)):
       ni+=l_coef[i]*l_var[i-1]

    eq_denominator = 1 + e_constant**(-1*ni)
    eq_total = 1/eq_denominator

    return [ni, eq_total]


def gradiant_matrix(param_loglb, param_coeff):
    gradiant_matrix =[]
    for i in range(0,len(param_coeff)):
        partial_b = diff(param_loglb,param_coeff[i])
        gradiant_matrix.append(partial_b)

    return gradiant_matrix


def check_convergence(prev_coeff,new_coeff):
    if not prev_coeff == new_coeff:
        epsilon = 10**-8
        diff  = []
        for i in range(0,len(prev_coeff)):
            diff.append(abs(new_coeff[i]-prev_coeff[i]))

        relative_change  = []

        for i in range(0,len(diff)):
            relative_change.append(diff[i]/(prev_coeff[i]+epsilon))

        # print(max(relative_change))
        return max(relative_change) < 0.001

    return false

def return_race(dummy_var):
    race_array =[]
    if int(dummy_var)==5:
        for i in range(0,5):
            race_array.append(0)
    else:
        dummy_var_index =0
        if not int(dummy_var) ==6: dummy_var_index= int(dummy_var)-1
        else: dummy_var_index = 4
        for i in range(0,5):
            if i == dummy_var_index:
                race_array.append(1)
            else :
                race_array.append(0)
    return race_array

def return_age(dummy_var):
    age_array =[]
    if 12 <= int(dummy_var) <= 17:
        age_array = [1,0,0,0,0]
    elif 18 <= int(dummy_var) <= 24:
        age_array = [0,1,0,0,0]
    elif 25 <= int(dummy_var) <= 34:
        age_array = [0,0,1,0,0]
    elif 35 <= int(dummy_var) <= 54:
        age_array = [0,0,0,1,0]
    elif 55 <=int(dummy_var):
        age_array = [0,0,0,0,1]
    else:
        age_array=[0,0,0,0,0]

    return age_array





def logistic_fdata(n_sample):
    fdata=[]
    f_age_12_17=[]
    f_age_18_24 = []
    f_age_25_34 = []
    f_age_35_54 = []
    f_age_55_plus = []
    f_sex=[]
    f_prev_diagnosis=[]
    f_race_is_native=[]
    f_race_is_asian=[]
    f_race_is_black=[]
    f_race_is_hawaii=[]
    f_race_is_other=[]
    y_mh2_data =[]
    with open(og_csv_file) as og_csv_data :
        og_csv_reader = csv.DictReader(og_csv_data)
        n_count =1
        for line in og_csv_reader:
            if n_count<n_sample:
                if  line['AGE']!= '-9' and line['RACE']!= '-9' and line['GENDER']!= '-9' and line['MH1']!= '-9' and line['MH2']!= '-9':
                    f_race_is_native.append(return_race(line['RACE'])[0])
                    f_race_is_asian.append(return_race(line['RACE'])[1])
                    f_race_is_black.append(return_race(line['RACE'])[2])
                    f_race_is_hawaii.append(return_race(line['RACE'])[3])
                    f_race_is_other.append(return_race(line['RACE'])[4])
                    f_age_12_17.append(return_age(line['AGE'])[0])
                    f_age_18_24.append(return_age(line['AGE'])[1])
                    f_age_25_34.append(return_age(line['AGE'])[2])
                    f_age_35_54.append(return_age(line['AGE'])[3])
                    f_age_55_plus.append(return_age(line['AGE'])[4])
                    f_sex.append((int)(line['GENDER'])-1)

                    if line['MH1']=='6': f_prev_diagnosis.append(1)
                    else: f_prev_diagnosis.append(0)

                    if line['MH2']=='6': y_mh2_data.append(1)
                    else: y_mh2_data.append(0)
            else: break

        return {
            "x_age_12_17": f_age_12_17,
            "x_age_18_24": f_age_18_24,
            "x_age_25_34": f_age_25_34,
            "x_age_35_54": f_age_35_54,
            "x_age_55_plus": f_age_55_plus,
            "x_sex":f_sex,
            "x_prev_diagnosis":f_prev_diagnosis,
            "x_is_native":f_race_is_native,
            "x_is_asian":f_race_is_asian,
            "x_is_black":f_race_is_black,
            "x_is_hawaii":f_race_is_hawaii,
            "x_is_other":f_race_is_other,
            "y_mh2":y_mh2_data
        }

def hessian_matrix(interal_gradient: object, coef_vars: object):
    hess_mtrx = []
    for coeff in coef_vars:
        hessian_row =[]
        for j in range(0,len(interal_gradient)):
            term = diff(interal_gradient[j],coeff)
            hessian_row.append(term)
        hess_mtrx.append(hessian_row)

    return hess_mtrx


def compute_hessian_vectorized(coeff_sample, x_array, y_array, h_matrix_functions):
    n = len(h_matrix_functions)
    hessian_eval = np.zeros((n, n))


    all_evals = np.zeros((n, n, len(y_array)))


    for i in range(n):
        for j in range(n):
            all_evals[i, j] = h_matrix_functions[i][j](coeff_sample, x_array.T, y_array)


    hessian_eval = np.sum(all_evals, axis=2)
    return hessian_eval


def compute_hessian_element(i, j, coeff_sample, x_array, y_array, h_matrix_functions):
    hessian_values = h_matrix_functions[i][j](coeff_sample, x_array.T, y_array)
    return i, j, np.sum(hessian_values)

def compute_hessian_parallel(coeff_sample, x_array, y_array, h_matrix_functions, n_jobs=-1):
    n = len(h_matrix_functions)
    hessian_eval = np.zeros((n, n))


    indices = []
    for i in range(n):
        for j in range(n):
            indices.append((i, j))


    tasks = []
    for i, j in indices:
        task = delayed(compute_hessian_element)(i, j, coeff_sample, x_array, y_array, h_matrix_functions)
        tasks.append(task)

    results = Parallel(n_jobs=n_jobs)(tasks)

    for i, j, val in results:
        hessian_eval[i, j] = val

    return hessian_eval


def compute_covariance(hessian_matrix, regularization):
    try:

        rank = np.linalg.matrix_rank(hessian_matrix)
        full_rank = hessian_matrix.shape[0]
        print(f"Rank of Hessian matrix: {rank}/{full_rank}")


        if rank < full_rank:
            print("Hessian matrix is singular or nearly singular. Regularizing...")
            hessian_matrix += regularization * np.eye(hessian_matrix.shape[0])
        cov_matrix = np.linalg.inv(hessian_matrix)
        print("Covariance matrix computed successfully using the inverse.")
        return cov_matrix

    except np.linalg.LinAlgError:
        print("Hessian matrix is singular. Using pseudo-inverse.")
        cov_matrix = np.linalg.pinv(hessian_matrix)
        print("Covariance matrix computed using the pseudo-inverse.")
        return cov_matrix


def standard_eror(internal_gradiant, param_coeff_list, param_covar_list, coeff_sample, n_size):
    h_matrix  = hessian_matrix(internal_gradiant,param_coeff_list)
    fdata = logistic_fdata(n_size)
    stdv_error=[]

    print("LINE 509")

    h_matrix_functions=[]
    for i in range(0,len(h_matrix)):
        h_func_row =[]
        for j in range(0,len(h_matrix[0])):
            function = lambdify(
                [param_coeff_list, param_covar_list, y_i],
                h_matrix[i][j],
                "numpy"
            )
            h_func_row.append(function)
        h_matrix_functions.append(h_func_row)

    print("LINE 523")

    x_array=[]
    y_array=[]

    all_keys = fdata.keys()
    feature_columns = []
    for key in all_keys:
        if key != 'y_mh2':
            feature_values = fdata[key]
            feature_columns.append(feature_values)

    x_array  = np.column_stack(feature_columns)


    mh2_val = fdata['y_mh2']
    y_array = np.array(mh2_val)

    print("LINE 541")

    h_matrix_eval = compute_hessian_parallel(coeff_sample, x_array, y_array, h_matrix_functions)

    print(f"H_MATRIX: {h_matrix_eval}")
    epsilon = 1e-6
    cov_matrix = compute_covariance(h_matrix_eval,epsilon)

    print(f"COV_MATRIX: {cov_matrix}")


    for i in range(0,len(cov_matrix)):
        for j in range(0,len(cov_matrix[0])):
            if i == j:
                stdv_error.append(math.sqrt(abs(cov_matrix[i][j])))

    return stdv_error




    #print(compute_hessian_vectorized(coeff_sample, x_array, y_array, h_matrix_functions))



    # hessian_eval = np.zeros((len(param_coeff_list), len(param_coeff_list)))
    # for i in range(0,len(h_matrix)):
    #     for j in range(0,len(h_matrix[0])):
    #         hessian_values = h_matrix_functions[i][j](coeff_sample, x_array.T, y_array)
    #         hessian_eval[i, j] = np.sum(hessian_values)
    #
    # print(hessian_eval)


    # for i in range(0,len(h_matrix)):
    #     h_row=[]
    #     for j in range(0,len(h_matrix[0])):
    #         term =0
    #         for n in range(0,n_size):
    #             term += h_matrix[i][j].subs({
    #                 param_coeff_list[0]:coeff_sample[0],param_coeff_list[1]:coeff_sample[1],
    #                 param_coeff_list[2]:coeff_sample[2],param_coeff_list[3]:coeff_sample[3],
    #                 param_coeff_list[4]:coeff_sample[4],param_coeff_list[5]:coeff_sample[5],
    #                 param_coeff_list[6]:coeff_sample[6],param_coeff_list[7]:coeff_sample[7],
    #                 param_coeff_list[8]:coeff_sample[8],param_coeff_list[9]:coeff_sample[9],
    #                 param_coeff_list[10]:coeff_sample[10],param_coeff_list[11]:coeff_sample[11],
    #                 param_coeff_list[12]:coeff_sample[12],
    #                 param_covar_list[0]:fdata["x_age_12_17"][n],param_covar_list[1]:fdata["x_age_18_24"][n],
    #                 param_covar_list[2]:fdata["x_age_25_34"][n],param_covar_list[3]:fdata["x_age_35_54"][n],
    #                 param_covar_list[4]:fdata["x_age_55_plus"][n],param_covar_list[5]:fdata["x_sex"][n],
    #                 param_covar_list[6]:fdata["x_prev_diagnosis"][n],param_covar_list[7]:fdata["x_is_native"][n],
    #                 param_covar_list[8]:fdata["x_is_asian"][n],param_covar_list[9]:fdata["x_is_black"][n],
    #                 param_covar_list[10]:fdata["x_is_hawaii"][n],param_covar_list[11]:fdata["x_is_other"][n],
    #                 y_i:fdata["y_mh2"][n]
    #             }).evalf()
    #         h_row.append(term)
    #     h_matrix_eval.append(h_row)

    # print(f"H MATRIX: {h_matrix}")
    # print("\n\n")
    # print(f"H MATRIX EVAL: {h_matrix_eval}")


def coeff_confident_intervals(std_error_coeff, sample_coeff):
    coeff_c_intervals = []
    for i in range(0,len(sample_coeff)):
        margin_error = std_error_coeff[i]*abs(stats.norm.ppf(0.025))
        coeff_interval = [sample_coeff[i]-margin_error,sample_coeff[i]+margin_error]
        coeff_c_intervals.append(coeff_interval)
    return coeff_c_intervals



def gradiant_ascent(internal_gradiant, learning_curve, pi_ni, n_size, param_coeff_list, param_covar_list):

    b_test_coeff = [0,0,0,0,0,0,0,0,0,0,0,0,0]

    old_b_test_coeff = b_test_coeff.copy()
    fdata = logistic_fdata(n_size)
    coeff = 0
    partial_coeff=[]

    while not check_convergence(old_b_test_coeff,b_test_coeff) :
        for partial in internal_gradiant:
            for i in range(0,n_size):
                #print("line 457")
                obsv_grad_term = partial.subs({param_coeff_list[0]:b_test_coeff[0],param_coeff_list[1]:b_test_coeff[1],
                                               param_coeff_list[2]:b_test_coeff[2],param_coeff_list[3]:b_test_coeff[3],
                                               param_coeff_list[4]:b_test_coeff[4],param_coeff_list[5]:b_test_coeff[5],
                                               param_coeff_list[6]:b_test_coeff[6],param_coeff_list[7]:b_test_coeff[7],
                                               param_coeff_list[8]:b_test_coeff[8],param_coeff_list[9]:b_test_coeff[9],
                                               param_coeff_list[10]:b_test_coeff[10],param_coeff_list[11]:b_test_coeff[11],
                                               param_coeff_list[12]:b_test_coeff[12],
                                               param_covar_list[0]:fdata["x_age_12_17"][i],param_covar_list[1]:fdata["x_age_18_24"][i],
                                               param_covar_list[2]:fdata["x_age_25_34"][i],param_covar_list[3]:fdata["x_age_35_54"][i],
                                               param_covar_list[4]:fdata["x_age_55_plus"][i],param_covar_list[5]:fdata["x_sex"][i],
                                               param_covar_list[6]:fdata["x_prev_diagnosis"][i],param_covar_list[7]:fdata["x_is_native"][i],
                                               param_covar_list[8]:fdata["x_is_asian"][i],param_covar_list[9]:fdata["x_is_black"][i],
                                               param_covar_list[10]:fdata["x_is_hawaii"][i],param_covar_list[11]:fdata["x_is_other"][i],
                                               y_i:fdata["y_mh2"][i]
                                               }).evalf()

                #print(obsv_grad_term)
                try : math.isnan(coeff)
                    # print(partial)
                    # print(partial.subs({param_coeff_list[0]:b_test_coeff[0],param_coeff_list[1]:b_test_coeff[1],
                    #                            param_coeff_list[2]:b_test_coeff[2],param_coeff_list[3]:b_test_coeff[3],
                    #                            param_coeff_list[4]:b_test_coeff[4],param_coeff_list[5]:b_test_coeff[5],
                    #                            param_coeff_list[6]:b_test_coeff[6],param_coeff_list[7]:b_test_coeff[7],
                    #                            param_coeff_list[8]:b_test_coeff[8],
                    #                            param_covar_list[0]:fdata["x_age"][i], param_covar_list[1]:fdata["x_sex"][i],
                    #                            param_covar_list[2]:fdata["x_prev_diagnosis"][i],param_covar_list[3]:fdata["x_is_native"][i],
                    #                            param_covar_list[4]:fdata["x_is_asian"][i], param_covar_list[5]:fdata["x_is_black"][i],
                    #                            param_covar_list[6]:fdata["x_is_hawaii"][i],param_covar_list[7]:fdata["x_is_other"][i],
                    #                            y_i:fdata["y_mh2"][i]
                    #                            }))
                    # print(obsv_grad_term)
                except:
                    print(f"{param_coeff_list[0]}:{b_test_coeff[0]}\n{param_coeff_list[1]}:{b_test_coeff[1]}\n{param_coeff_list[2]}:{b_test_coeff[2]}\n{param_coeff_list[3]}:{b_test_coeff[3]}\n"
                          f"{param_coeff_list[4]}:{b_test_coeff[4]}\n{param_coeff_list[5]}:{b_test_coeff[5]}\n{param_coeff_list[6]}:{b_test_coeff[6]}\n{param_coeff_list[7]}:{b_test_coeff[7]}\n"
                          f"{param_coeff_list[8]}:{b_test_coeff[8]}\n"
                          f"{param_coeff_list[9]}:{b_test_coeff[9]}\n"
                          f"{param_coeff_list[10]}:{b_test_coeff[10]}\n"
                          f"{param_coeff_list[11]}:{b_test_coeff[11]}\n"
                          f"{param_coeff_list[12]}:{b_test_coeff[12]}")
                    print(f"{param_covar_list[0]}:{fdata['x_age_12_17'][i]}\n"
                          f"{param_covar_list[1]}:{fdata['x_age_18_24'][i]}\n"
                          f"{param_covar_list[2]}:{fdata['x_age_25_34'][i]}\n"
                          f"{param_covar_list[3]}:{fdata['x_age_35_54'][i]}\n"
                          f"{param_covar_list[4]}:{fdata['x_age_55_plus'][i]}\n"
                          f"{param_covar_list[5]}:{fdata['x_sex'][i]}\n"
                          f"{param_covar_list[6]}:{fdata['x_prev_diagnosis'][i]}\n"
                          f"{param_covar_list[7]}:{fdata['x_is_native'][i]}\n"
                          f"{param_covar_list[8]}:{fdata['x_is_asian'][i]}\n"
                          f"{param_covar_list[9]}:{fdata['x_is_black'][i]}\n"
                          f"{param_covar_list[10]}:{fdata['x_is_hawaii'][i]}\n"
                          f"{param_covar_list[11]}:{fdata['x_is_other'][i]}\n"
                          f"{y_i}:{fdata['y_mh2'][i]}")
                coeff +=obsv_grad_term

            partial_coeff.append(coeff)
            coeff =0
        old_b_test_coeff = b_test_coeff.copy()
        b_test_coeff.clear()
        for i in range (0,len(partial_coeff)):
            b_test_coeff.append(partial_coeff[i]*learning_curve+old_b_test_coeff[i])
        print(b_test_coeff)
        partial_coeff.clear()

    return b_test_coeff





def getInputs(pInputStrings):
    count = 0
    answers = []
    while count < len(pInputStrings):
        userInput = input(pInputStrings[count])
        if userInput == "0":
            answers.append(0)
        else:
            answers.append(userInput)
        count += 1
    return answers









coeff_list =symbols(
    'B_0 B_1 B_2 B_3 B_4 B_5 B_6 B_7 B_8,B_9 B_10 B_11 B_12'
)

pi_ni, y_i, x_i1, x_i2, x_i3 , x_i4 , x_i5 , x_i6 , x_i7 , x_i8, ni, B_j, ni_b = symbols(
    'pi_ni y_i x_i1 x_i2 x_i3 x_i4 x_i5 x_i6 x_i7 x_i8 ni B_j ni_b',
    real = True
)


#covar_list = [x_i1,x_i2,x_i3,x_i4,x_i5,x_i6,x_i7,x_i8]
covar_list  = symbols(
    'x_i1 x_i2 x_i3 x_i4 x_i5 x_i6 x_i7 x_i8 x_i9 x_i10 x_i11 x_i12',
    real = True
)

ni = initialize_pi_n(coeff_list,covar_list)[0]

#print(ni)

pi_ni = initialize_pi_n(coeff_list,covar_list)[1]

#print(pi_ni)

internal_log_lb = y_i*log(pi_ni,e_constant) + (1-y_i)*log(1-pi_ni,e_constant)

#print(internal_log_lb)

internal_gradiant_log_lb = gradiant_matrix(internal_log_lb,coeff_list)

#print(internal_gradiant_log_lb)

# print(len(clear_coeff))
# print(len(internal_gradiant_log_lb))
b_sample_coeff = gradiant_ascent(internal_gradiant_log_lb,0.001,pi_ni,1000,coeff_list,covar_list)
print(f"FINAL: {b_sample_coeff}")
#b_sample_coeff =[-0.793374930915092, -0.112855595501805, 0, 0, 0, 0, -0.445334975242641, -0.115378905934215, -0.00498571931986853, -0.00243750129403487, -0.265257191082972, -0.000881425992504944, -0.0392934534273202]
std_error= standard_eror(internal_gradiant_log_lb,coeff_list,covar_list,b_sample_coeff,1000)
print(f"STD ERRORS: {std_error}")
coeff_c_intervals = coeff_confident_intervals(std_error,b_sample_coeff)
print(f"Confidence Intervals: {coeff_c_intervals}")

# user_info = getInputs(["AGE: ",
#                        "SEX(0-Male/ 1-Female): ",
#                        "RACE: \n(pls choose one of the following\n1= Native \n2= Asian\n3= Black/African\n4= Hawaiian \n5= White\n6= Two or more/other\nEnter:  ",
#                        "PREV Diagnosis(0-No Bpd/ 1-BPD): "])
# user_b_var=[]
#
# age = return_age((int)(user_info[0]))
# sex= (int)(user_info[1])
# race = return_race((int)(user_info[2]))
# prev_diagnosis = (int)(user_info[3])
#
# for  a in age:
#     user_b_var.append(a)
#
# user_b_var.append(prev_diagnosis)
# user_b_var.append(sex)
#
# for  r in race:
#     user_b_var.append(r)
#
# pi_eval_lower = pi_ni.subs(
#     {
#         coeff_list[0]:coeff_c_intervals[0][0],
#         coeff_list[0]:coeff_c_intervals[1][0],
#         coeff_list[0]:b_sample_coeff[2],
#         coeff_list[0]:b_sample_coeff[3],
#         coeff_list[0]:b_sample_coeff[4],
#         coeff_list[0]:b_sample_coeff[5],
#         coeff_list[0]:coeff_c_intervals[6][0],
#         coeff_list[0]:coeff_c_intervals[7][0],
#         coeff_list[0]:coeff_c_intervals[8][0],
#         coeff_list[0]:coeff_c_intervals[9][0],
#         coeff_list[0]:coeff_c_intervals[10][0],
#         coeff_list[0]:coeff_c_intervals[11][0],
#         coeff_list[0]:coeff_c_intervals[12][0],
#         covar_list[0]:user_b_var[0],
#         covar_list[1]:user_b_var[0],
#         covar_list[2]:user_b_var[0],
#         covar_list[3]:user_b_var[0],
#         covar_list[4]:user_b_var[0],
#         covar_list[5]:user_b_var[0],
#         covar_list[6]:user_b_var[0],
#         covar_list[7]:user_b_var[0],
#         covar_list[8]:user_b_var[0],
#         covar_list[9]:user_b_var[0],
#         covar_list[10]:user_b_var[0],
#         covar_list[11]:user_b_var[0],
#
#     }).evalf()
#
# pi_eval_upper = pi_ni.subs(
#     {
#         coeff_list[0]: coeff_c_intervals[0][1],
#         coeff_list[0]: coeff_c_intervals[1][1],
#         coeff_list[0]: b_sample_coeff[2],
#         coeff_list[0]: b_sample_coeff[3],
#         coeff_list[0]: b_sample_coeff[4],
#         coeff_list[0]: b_sample_coeff[5],
#         coeff_list[0]: coeff_c_intervals[6][1],
#         coeff_list[0]: coeff_c_intervals[7][1],
#         coeff_list[0]: coeff_c_intervals[8][1],
#         coeff_list[0]: coeff_c_intervals[9][1],
#         coeff_list[0]: coeff_c_intervals[10][1],
#         coeff_list[0]: coeff_c_intervals[11][1],
#         coeff_list[0]: coeff_c_intervals[12][1],
#         covar_list[0]: user_b_var[0],
#         covar_list[1]: user_b_var[0],
#         covar_list[2]: user_b_var[0],
#         covar_list[3]: user_b_var[0],
#         covar_list[4]: user_b_var[0],
#         covar_list[5]: user_b_var[0],
#         covar_list[6]: user_b_var[0],
#         covar_list[7]: user_b_var[0],
#         covar_list[8]: user_b_var[0],
#         covar_list[9]: user_b_var[0],
#         covar_list[10]: user_b_var[0],
#         covar_list[11]: user_b_var[0],
#
#     }).evalf()
#
# print(f"Probability of BPD during MH2 is ( {pi_eval_lower}, {pi_eval_upper} )")





# fdata = logistic_fdata(1000)
# all_keys = fdata.keys()
# print(len(all_keys))
# feature_columns = []
# for key in all_keys:
#     if key != 'y_mh2':
#         feature_values = fdata[key]
#         feature_columns.append(feature_values)
#
# x_array = np.column_stack(feature_columns)
# print(len(x_array[0]))


























