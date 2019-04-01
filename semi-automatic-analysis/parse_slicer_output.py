import pandas as pd
if __name__ == '__main__':
    file_name = "/home/mehrtash/Dropbox_Partners/Prostate Needle Finder AM/log3.txt"
    with open(file_name) as f:
        lines = f.readlines()
    results = {}
    i = 0
    for index, line in enumerate(lines):
        if "Processing on:" in line:
            case_line_no = index
        if "New HD Validation Results:" in line:
            val_results = lines[index + 2].split('\t')
            validation = dict()
            path_split = lines[case_line_no].split('/')
            case_id = path_split[-3] + '_needle' + path_split[-1][:-1]
            # validation["case_id"] = case_id
            validation["tipHD"] = val_results[2]
            validation["HD"] = val_results[3][:-2]
            results[case_id] = validation
            i +=1
    print "number of cases: ", i
    results_df = pd.DataFrame.from_dict(results).transpose()
                 #columns=["case_id_needle_id","tipHD [mm]", "HD [mm]"])
    results_df.to_csv("results4.csv")
    with pd.option_context('display.max_rows', 1000, 'display.max_columns', 3):
        print results_df



