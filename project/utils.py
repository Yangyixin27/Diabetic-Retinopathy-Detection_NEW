import glob
import os
from random import shuffle


class DataUtil(object):
    def __init__(self, cases_root, file_type='nrrd'):
        self.intraop_folders = sorted([x[0] for x in os.walk(cases_root) if "IntraopImages" in x[0]])
        self.manual_az_folders = sorted([x[0] for x in os.walk(cases_root) if "Manual_AZ" in x[0] and not "Manual_AZ/" in x[0]])
        self.file_type = file_type

    def __get_cover_prostate_from_label(self, label_file):
        dir_name, base_name = os.path.split(label_file)
        series_number = base_name.split('-')[0]
        return os.path.join(dir_name, series_number + '-CoverProstate.' + self.file_type)

    @staticmethod
    def __shuffle_pair_lists(list1, list2):
        list1_shuffled = []
        list2_shuffled = []
        index_shuffle = range(len(list1))
        shuffle(index_shuffle)
        for i in index_shuffle:
            list1_shuffled.append(list1[i])
            list2_shuffled.append(list2[i])
        return list1_shuffled, list2_shuffled

    def get_prostate_vol_label_file_pairs(self, shuffle_pairs=True):
        label_files_query = [glob.glob(intraop_folder + '/*-label.' + self.file_type) for intraop_folder in self.intraop_folders]
        label_files = [label[0] for label in label_files_query if label]
        volume_files = [self.__get_cover_prostate_from_label(label_file) for label_file in label_files if
                        os.path.exists(self.__get_cover_prostate_from_label(label_file))]
        if len(volume_files) != len(label_files):
            raise Exception("Number of volume files and labels mismatch.")
        if shuffle_pairs:
            volume_files, label_files = self.__shuffle_pair_lists(volume_files, label_files)
        return volume_files, label_files
            

    def get_needle_vol_files(self):
        needle_files_query = [glob.glob(intraop_folder + '/*-Needle.' + self.file_type) for intraop_folder in self.intraop_folders]
        needle_files = [needle_file for needle_files in needle_files_query for needle_file in needle_files if
                        needle_files]
        return needle_files

    def get_needle_vol_files_with_manual_needle_segmentation(self):
        needle_files_query = [glob.glob(intraop_folder + '/*/*-Needle.' + self.file_type) for intraop_folder in self.manual_az_folders]
        needle_files = [needle_file for needle_files in needle_files_query for needle_file in needle_files if
                        needle_files]
        return needle_files

    def get_all_nrrd_files(self):
        nrrd_files_query = [glob.glob(intraop_folder + '/*.' + self.file_type) for intraop_folder in self.intraop_folders]
        nrrd_files = [nrrd_file for nrrd_files in nrrd_files_query for nrrd_file in nrrd_files if
                        nrrd_files]
        return nrrd_files

    
class IntermediateUtil(object):
    def __init__(self, intermediate_dir, sub_dir = "nrrd/"):
        self.intermediate_dir = intermediate_dir
        self.nrrd_dir = intermediate_dir + sub_dir
        self.filenames = os.listdir(self.nrrd_dir)
        self.filenames.sort()
    
    def get_case_number(self):
        case_nums = []
        for file_ in self.filenames:
            case_nums.append(file_[:7])
        case_nums = list(set(case_nums))
        case_nums.sort()
        return case_nums
        
    def get_needle_vol_files(self):
        vol_files = [self.nrrd_dir + x for x in self.filenames if 'labelmap' not in x and 'mask' not in x]
        return vol_files
    
    def get_needle_map_files(self):
        map_files = [self.nrrd_dir + x for x in self.filenames if 'labelmap' in x]
        return map_files
    
    def get_prostate_mask_files(self):
        mask_files = [self.nrrd_dir + x for x in self.filenames if 'mask' in x]
        return mask_files
        
    def get_needle_vol_files_with_case(self, case_num):
        vol_files = self.get_needle_vol_files()
        volumes = [x for x in vol_files if case_num in x]
        volumes.sort()
        return  volumes
    
    def get_needle_map_files_with_case(self, case_num):
        map_files = self.get_needle_map_files()
        labelmaps = [x for x in map_files if case_num in x]
        labelmaps.sort()
        return  labelmaps
    
    def get_prostate_mask_files_with_case(self, case_num):
        mask_files = self.get_prostate_mask_files()
        masks = [x for x in mask_files if case_num in x]
        masks.sort()
        return  masks