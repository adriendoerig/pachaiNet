import os
import sys

# TO CHANGE OUTPUT SIZE: cf line 45/46.
# TO CHANGE OUTPUT FILE NAMES cf also down there.

if  os.path.exists('lmdb/trainLmdb') or os.path.exists('lmdb/meanConfig.binaryproto') or os.path.exists('lmdb/meanConfig.npy') or os.path.exists('labels/trainSet.txt'):
    print '\n' + r"trainLmdb and/or meanConfig.binaryproto and/or meanConfig.npy already exist in file lmdb, or trainSet.txt exists in file labels. Remove them." + '\n'
    sys.exit()

if not os.path.exists('labels'):
	os.mkdir('labels')
if not os.path.exists('lmdb'):
	os.mkdir('lmdb')
	os.system(r'chmod -R a+w lmdb')

os.chdir('labels')
# 5/6 images go to the train set
with open("trainSet.txt", "w") as my_file:
    for file in os.listdir(r'../jpgs/'):
        if file[-4:] == ".jpg":
            if float(file[7:-3])%6 != 0:
                if float(file[7:-3]) <= 1716:
                    my_file.write(file + " 0\n")
                else:
                    my_file.write(file + " 1\n")

# 1/6 images go to the validation set
with open("valSet.txt", "w") as my_file:
    for file in os.listdir(r'../jpgs/'):
        if file[-4:] == ".jpg":
            if float(file[7:-3])%6 == 0:
                if float(file[7:-3]) <= 1716:
                    my_file.write(file + " 0\n")
                else:
                    my_file.write(file + " 1\n")

os.chdir('..')
gray = False
if gray == True:
    os.system(r'GLOG_logtostderr=1 ../caffe/build/tools/convert_imageset --resize_height=60 --resize_width=100 --shuffle --gray jpgs/ labels/trainSet.txt lmdb/trainLmdb')  # create trainLmdb
    os.system(r'GLOG_logtostderr=1 ../caffe/build/tools/convert_imageset --resize_height=60 --resize_width=100 --shuffle --gray jpgs/ labels/valSet.txt lmdb/valLmdb')  # create valLmdb
else:
    os.system(r'GLOG_logtostderr=1 ../caffe/build/tools/convert_imageset --resize_height=60 --resize_width=100 --shuffle jpgs/ labels/trainSet.txt lmdb/trainLmdb') # create trainLmdb
    os.system(r'GLOG_logtostderr=1 ../caffe/build/tools/convert_imageset --resize_height=60 --resize_width=100 --shuffle jpgs/ labels/valSet.txt lmdb/valLmdb') # create valLmdb

os.system(r'GLOG_logtostderr=1 ../caffe/build/tools/compute_image_mean lmdb/trainLmdb lmdb/meanConfig.binaryproto') # create meanConfigs.binaryproto
os.system(r'python caffe-compute-image-mean.py lmdb/meanConfig.binaryproto lmdb/meanConfig.npy') # create meanConfigs.npy
