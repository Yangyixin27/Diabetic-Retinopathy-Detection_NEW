import glob
import os
import sys
import json
import time
import numpy as np
import shutil
import SimpleITK as sitk
import random
from sklearn.model_selection import KFold

from project.settings import cases_root, slicer_dir, intermediate_dir
from project.utils import IntermediateUtil


if __name__ == '__main__':
    NUM_TEST = 12
    SPLIT = 5
    FOLDDIR = intermediate_dir + "fold/"
    if not os.path.exists(FOLDDIR):
        os.makedirs(FOLDDIR)
    
    iu = IntermediateUtil(intermediate_dir, "nrrd/")
    caselist = iu.get_case_number()
    random.shuffle(caselist)
    print(caselist, len(caselist))
    
    testlist = caselist[0:NUM_TEST]
    trainvallist = caselist[NUM_TEST:]
    #Create Test TXT
    with open(FOLDDIR + "test.txt", 'w+') as file:
        for test in testlist:
            file.write("{}\n".format(test))
    
    assert len(trainvallist)%SPLIT == 0, "Not Dividable"
    
    kf = KFold(n_splits=SPLIT, shuffle=False)
    x = np.arange(0, len(trainvallist))
    i =0
    for train_index, val_index in kf.split(x):
        print(val_index)
        vallist = [trainvallist[idx] for idx in val_index]
        trainlist = [trainvallist[idx] for idx in train_index]
        i = i + 1
        print("Fold {}: {}".format(i, vallist))
        
        validation_txt = FOLDDIR + "validation_{}.txt".format(i)
        #Create Validation TXT
        with open(FOLDDIR + "validation_{}.txt".format(i), 'w+') as file:
            for val in vallist:
                file.write("{}\n".format(val))
        #Create Training TXT 
        with open(FOLDDIR + "train_{}.txt".format(i), 'w+') as file:
            for train in trainlist:
                file.write("{}\n".format(train))
                
    