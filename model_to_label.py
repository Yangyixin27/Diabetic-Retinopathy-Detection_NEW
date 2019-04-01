import numpy as np
import os
import glob
import shutil

from project.utils import DataUtil
from project.settings import cases_root, slicer_dir, intermediate_dir


def convert_model_to_label(input_volume, input_model, output_file):
    command = slicer_dir + 'Slicer --launch ' + \
                slicer_dir + 'lib/Slicer-4.7/cli-modules/ModelToLabelMap '+ \
                """ --distance 1 --labelValue 255 """ + \
                input_volume + ' ' + input_model + ' ' + output_file
    os.system(command)
    

if __name__ == '__main__':
    du = DataUtil(cases_root)
    volumes = du.get_needle_vol_files_with_manual_needle_segmentation()
    print("total number of needles: {}".format(len(volumes)))
    caselist = []
    
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)
        
    nrrd_dir = intermediate_dir + "nrrd/"
    if not os.path.exists(nrrd_dir):
        os.makedirs(nrrd_dir)
    
    for volume in volumes:
        head, volume_file = os.path.split(volume)
        model = os.path.join(head + '/' + 'manual-seg_0.vtk')
        if not os.path.isfile(model):
            model = os.path.join(head + '/' + 'manual-seg_1.vtk')
            print("manual-seg_0.vtk not in {}".format(head))
            if not os.path.isfile(model):
                raise ValueError("{} doesn't include any vtk".format(model))
            else:
                print("but manual-seg_1.vtk is in {}".format(head))
        needle_number = volume_file.split('-')[0]
        head, needle_num = os.path.split(head)
        assert needle_number == needle_num, "Needle Number Doesn't Match"
        if len(needle_num) == 1:
            needle_num = "0" + needle_num
        casepath = os.path.split(head)[0]
        case_num = os.path.split(casepath)[1]
        
        caselist.append(case_num)
        
        #copy *-Neddle.nrrd
        output = nrrd_dir + case_num + '_' + "Needle" + needle_num + ".nrrd"
        shutil.copyfile(volume, output)
        
        #generate *-Needle_labelmap.nrrd
        output_labelmap = nrrd_dir + case_num + '_' + "Needle" + needle_num + "_labelmap" ".nrrd"
        convert_model_to_label(volume, model, output_labelmap)
        
    caselist = list(set(caselist))
    caselist.sort()
    print("Unique Case Number : {}".format(caselist))
    print("Number Of Segmented Cases : {}".format(len(caselist)))