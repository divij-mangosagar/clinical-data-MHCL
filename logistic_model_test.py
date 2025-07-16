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

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5 import uic  # For loading the UI dynamically
import sys


user_answ=[]
user_name=""


from sympy import Eq
from sympy import diff, integrals, integrate
import pandas as pd  # Pandas for handling tabular data

from joblib import Parallel, delayed


# class MyApp(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         uic.loadUi("main.ui", self)
#
#         self.submit_button.clicked.connect(self.get_reponse)
#
#     def get_response(self):
#         user_answ.append((int)(self.age_enter.text()))
#
#         if(self.native_race.isChecked())
#
#         elif (self.native_race.isChecked())
#
#         elif (self.native_race.isChecked())
#
#         elif (self.native_race.isChecked())







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

def initialize_pi_n(l_coef, l_var):

    intercept = l_coef[0]

    ni = intercept

    for i in range(1,len(l_coef)):
       ni+=l_coef[i]*l_var[i-1]

    e_constant = math.e

    eq_denominator = 1 + e_constant**(-1*ni)
    eq_total = 1/eq_denominator

    return [ni, eq_total]

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



b_sample_coeff =[-0.793374930915092, -0.112855595501805, 0, 0, 0, 0, -0.445334975242641, -0.115378905934215, -0.00498571931986853, -0.00243750129403487, -0.265257191082972, -0.000881425992504944, -0.0392934534273202]
coeff_c_intervals = [[np.float64(-0.7987422694797139), np.float64(-0.7880075923504701)], [np.float64(-0.12169264528154798), np.float64(-0.10401854572206203)], [np.float64(-1959.9639845400545), np.float64(1959.9639845400545)], [np.float64(-1959.9639845400545), np.float64(1959.9639845400545)], [np.float64(-1959.9639845400545), np.float64(1959.9639845400545)], [np.float64(-1959.9639845400545), np.float64(1959.9639845400545)], [np.float64(-0.45179819347791206), np.float64(-0.4388717570073699)], [np.float64(-0.12564449258155985), np.float64(-0.10511331928687015)], [np.float64(-0.02542300707091542), np.float64(0.01545156843117836)], [np.float64(-0.03110037342907133), np.float64(0.02622537084100159)], [np.float64(-0.27440240948140654), np.float64(-0.2561119726845375)], [np.float64(-0.060320923790297645), np.float64(0.058558071805287756)], [np.float64(-0.050527626398937465), np.float64(-0.028059280455702933)]]




# class MyApp(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         uic.loadUi("main.ui", self)




while(true):

    user_info = getInputs(["AGE: ",
                           "SEX(0-Male/ 1-Female): ",
                           "RACE: \n(pls choose one of the following\n1= Native \n2= Asian\n3= Black/African\n4= Hawaiian \n5= White\n6= Two or more/other\nEnter:  ",
                           "PREV Diagnosis(0-No Bpd/ 1-BPD): "])
    user_b_var=[]

    age = return_age((int)(user_info[0]))
    sex= (int)(user_info[1])
    race = return_race((int)(user_info[2]))
    prev_diagnosis = (int)(user_info[3])

    for  a in age:
        user_b_var.append(a)

    user_b_var.append(prev_diagnosis)
    user_b_var.append(sex)

    for  r in race:
        user_b_var.append(r)


    print(user_b_var)

    pi_eval_lower = pi_ni.subs(
        {
            coeff_list[0]:coeff_c_intervals[0][0],
            coeff_list[0]:coeff_c_intervals[1][0],
            coeff_list[0]:b_sample_coeff[2],
            coeff_list[0]:b_sample_coeff[3],
            coeff_list[0]:b_sample_coeff[4],
            coeff_list[0]:b_sample_coeff[5],
            coeff_list[0]:coeff_c_intervals[6][0],
            coeff_list[0]:coeff_c_intervals[7][0],
            coeff_list[0]:coeff_c_intervals[8][0],
            coeff_list[0]:coeff_c_intervals[9][0],
            coeff_list[0]:coeff_c_intervals[10][0],
            coeff_list[0]:coeff_c_intervals[11][0],
            coeff_list[0]:coeff_c_intervals[12][0],
            covar_list[0]: user_b_var[0],
            covar_list[1]: user_b_var[1],
            covar_list[2]: user_b_var[2],
            covar_list[3]: user_b_var[3],
            covar_list[4]: user_b_var[4],
            covar_list[5]: user_b_var[5],
            covar_list[6]: user_b_var[6],
            covar_list[7]: user_b_var[7],
            covar_list[8]: user_b_var[8],
            covar_list[9]: user_b_var[9],
            covar_list[10]: user_b_var[10],
            covar_list[11]: user_b_var[11],

        }).evalf()

    pi_eval_upper = pi_ni.subs(
        {
            coeff_list[0]: coeff_c_intervals[0][1],
            coeff_list[0]: coeff_c_intervals[1][1],
            coeff_list[0]: b_sample_coeff[2],
            coeff_list[0]: b_sample_coeff[3],
            coeff_list[0]: b_sample_coeff[4],
            coeff_list[0]: b_sample_coeff[5],
            coeff_list[0]: coeff_c_intervals[6][1],
            coeff_list[0]: coeff_c_intervals[7][1],
            coeff_list[0]: coeff_c_intervals[8][1],
            coeff_list[0]: coeff_c_intervals[9][1],
            coeff_list[0]: coeff_c_intervals[10][1],
            coeff_list[0]: coeff_c_intervals[11][1],
            coeff_list[0]: coeff_c_intervals[12][1],
            covar_list[0]: user_b_var[0],
            covar_list[1]: user_b_var[1],
            covar_list[2]: user_b_var[2],
            covar_list[3]: user_b_var[3],
            covar_list[4]: user_b_var[4],
            covar_list[5]: user_b_var[5],
            covar_list[6]: user_b_var[6],
            covar_list[7]: user_b_var[7],
            covar_list[8]: user_b_var[8],
            covar_list[9]: user_b_var[9],
            covar_list[10]: user_b_var[10],
            covar_list[11]: user_b_var[11],

        }).evalf()

    print(f"Probability of BPD during the second reporting period is ( {pi_eval_lower}, {pi_eval_upper} )")