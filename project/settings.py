import socket

if socket.gethostname() == 'winterfell':
    cases_root = '/media/mehrtash/mehrtash/Dropbox (Partners HealthCare)/Prostate/Prostate Needle Finder/' \
                 'Prostate Needle Finder/Cases/'

if socket.gethostname() == 'theBeast':
    cases_root = '/home/deepinfer/Desktop/ProstateNeedleFinder/Cases/'
    slicer_dir = '/home/deepinfer/Slicer-4.7.0-2017-03-27-linux-amd64/'
    intermediate_dir = '/home/deepinfer/Desktop/Intermediate/'
    
if socket.gethostname() == 'cdsclen2':
    cases_root = '/home/administrator/ProstateNeedleFinder/Cases/'
    slicer_dir = '/home/administrator/Slicer-4.7.0-2017-03-29-linux-amd64/'
    intermediate_dir = '/home/administrator/ProstateNeedleFinder/Intermediate/'