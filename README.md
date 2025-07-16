# clinical_data_MHCL
A manual statistical analysis on a clinical dataset of patients (https://shorturl.at/XRWwW) to help reinforce statistical concepts. Done in python. 
WARNING: This code is inefficient and overly computationally heavy (the actual MLE and coeff CI code) due to an inefficient use of for loops and sympy library. 
Experimented with 2 main inferential analysis methods:
Hypothesis tests: cmh (Cochran-Mantel-Haenszel test), chi-squared test
, and Binary logistic regression model, bernoullis distribution: MLE manual implementation (gradient ascent) and coefficient CI (hessian matrix) => using to estimate 
bernoullis pi_n parameter 
.This project was very fun and I definetely enjoyed doing it! There are clear inaccuracies and inefficiencies; however, it helped me reinforce concepts and 
helped me grow my skills for future projects. :) 
* Will try to see if I can expand later on *

  
My Log:

'''

1) Cleaned data by removing all -9(unidentified/bad) diagnosis from mh1, mh2, mh3 => stored in list
2) Found frequency for each diagnosis => matched w/ data 
3) Did the pearson's-r test (used method) to see which data had association
    4) Bipolar disorder and other(unspecified) disorder failed Ho at alpha = 0.05 (significant association)

5) Doing chi-squared test for bipolar disorder  => FAILED CONDITION NOT SATISFIED

7) Regression model + ci
    a) Binary Logistic Model => probability of bpd diagnosis during second diagnosis period given previous diagnosis + patient info

8) chi-squared => did a different contingent table to ensure independence 
    a) Bipolar Disorder => Number of bpd diagnosis vs variables

9) CHM test
    a) SMI/SED vs Substance Abuse strata by diagnosis
    b) Substance Abuse vs Employment strata by SMI/SED
   
'''
