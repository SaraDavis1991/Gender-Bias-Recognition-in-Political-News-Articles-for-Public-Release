#################################
# run_pretrain_and_finetune.py:
# This file runs pretrain_and_finetune.py
#################################

from pretrain_and_finetune import *
pf = pretrain()
'''
Uncomment the option you wish to run; our results were obtained with  option 1
dirtyNewsBias is a bool and refers to whether the news bias dataset has been cleaned or not
cleanatn is a bool and refers to whether all the news should be cleaned before pretraining; NOTE: this takes A LONG TIME

NOTE: to run this file, articles must have been collected and run_preprocessor.py must have been run
'''

#OPTION 1: run pretrain and fineTune on cleanatn, then on cleaned newsbias dataset
fine_tuned_model = pf.pretrain_and_fineTune(atnPortion = 0.2, dirtyNewsBias = False, cleanatn = True)

#OPTION 2: run pretain and fineTune on dirtyatn, then on dirty newbias dataset
#fine_tuned_model = pf.pretrain_and_fineTune(atnPortion = 0.2, dirtyNewsBias = True, cleanatn = False)


