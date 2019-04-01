'''
To be run in Slicer python interactor
gist address: https://gist.githubusercontent.com/mehrtash/b57985d81a6ef907590497a04b5acf8e/raw/69f65c896704e16ee3b4a1da2ee00738425417b7/needle_finder_script.py
'''


def delayDisplay(message, msec=1000):
    print(message)
    info = qt.QDialog()
    infoLayout = qt.QVBoxLayout()
    info.setLayout(infoLayout)
    label = qt.QLabel(message, info)
    infoLayout.addWidget(label)
    qt.QTimer.singleShot(msec, info.close)
    info.exec_()


# set startup project as needlefinder
import os, glob

s = slicer.util.moduleSelector()
s.selectModule("needlefinder")
# cases_root = "C:\\D\\Datasets\\Prostate Registration Analysis\\Cases"
# cases_root = "/home/mehrtash/Dropbox (Partners HealthCare)/Prostate Needle Finder AM"
cases_root = "/home/mehrtash/Dropbox_Partners/Prostate Needle Finder"
manual_az_folders = sorted([x[0] for x in os.walk(cases_root) if "Manual_AZ" in x[0]])

for inputdirectory in manual_az_folders:
    scenes_files_query = glob.glob(inputdirectory + '/*.mrml')
    if len(scenes_files_query) == 1:
        print "+" * 100
        print "+" * 100
        print "+" * 100
        # delayDisplay("Processing on: %s\n" % inputdirectory)
        print "Processing on: ", inputdirectory
        scene_file = scenes_files_query[0]
        loadScene(scene_file)
        nf = slicer.modules.NeedleFinderInstance
        logic = nf.logic
        # TODO: comment the messagebox line
        logic.startValidation(script=False)
        # replace manual_az with semi_auto_am
        path_list = inputdirectory.split(os.sep)
        path_list[-2] = 'Semi_Auto_AM'
        path_list[0] = os.path.join(path_list[0], os.sep)
        outputdirectory = os.path.join(*path_list)
        # create directory tree if not exists
        if not os.path.exists(outputdirectory):
            os.makedirs(outputdirectory)
        scene = slicer.mrmlScene
        # rename
        node_names = getNodes('*auto-seg*')
        # ASSUMPTION: we have one needle per volume!
        for node_name in node_names:
            name_sp = node_name.split('-')
            node = getNode(node_name)
            node_new_name = name_sp[0]
            node.SetName(node_new_name)
        # save the mrml scene to a temp directory, then zip it
        #
        applicationLogic = slicer.app.applicationLogic()
        sceneSaveDirectory = outputdirectory
        # delayDisplay("Saving scene to: %s\n" % sceneSaveDirectory)
        applicationLogic.SaveSceneToSlicerDataBundleDirectory(sceneSaveDirectory, None)
        # save MRB
        case_number = path_list
        mrb_path = os.path.join(sceneSaveDirectory, path_list[-3] + '-' + path_list[-1] + '.mrb')
        applicationLogic.Zip(mrb_path, sceneSaveDirectory)
        # delayDisplay("Finished saving scene")
        scene.Clear(0)
        slicer.app.processEvents()
