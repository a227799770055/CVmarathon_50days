# CVmarathon_50days
### Computer Vision and Deep Learning Marathson

#### VOC convert to YOLO type####
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import random



# the classes you might predict
classes = ["kangaroo", 'raccoon']


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)



def convert_annotation(image_id):
    in_file = open(xml_path.format(image_id))
    out_file = open(txt_path.format(image_id), 'w')
    print(in_file)
    print(out_file)
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def take_image_id(path):
    print('Enter take_image_id function')
    image_id=[i[:-4] for i in os.listdir(path)]
    return image_id

def train_vali_split(train_rate, img_id_path):
    all_file = []
    with open (img_id_path, 'r') as f:
        for i in f:
            all_file.append(i[:-1])
    random.shuffle(all_file)
    array_len = len(all_file)
    train_size=int(array_len*train_rate)
    vali_size=array_len-train_size
    print('training size = {};\nvalidation size = {}'.format(train_size, vali_size))
    #save training_set path
    with open(img_id_path[:-13]+'training_set.txt','w') as f:
        for i in all_file[:train_size]:
            f.write(i+'\n')
    #save validation set path
    with open(img_id_path[:-13]+'validation_set.txt','w') as f:
        for i in all_file[train_size:]:
            f.write(i+'\n')



def main():
    #get image id txt file
    num_list=take_image_id(image_id_path)

    #file for all image path 
    with open (main_path+'test_list.txt','w') as f:
        for i in num_list:
            # num = i.split(' ')[0][:-1]
            file_name = image_jpg_path+'{}.jpg'.format(i)
            f.write(file_name+'\n')

    #convert VOC type to YOLO type
    for img_num in num_list:
        convert_annotation(img_num)

    # split data to training set and validation set
    train_vali_split(0.7, main_path+'test_list.txt')


# kangaroo set
main_path = '/home/gapcmgr/kt_privacy/pySourceCode/darknet/Racoon_Kangaroo/kangaroo/'
image_jpg_path = '/home/gapcmgr/kt_privacy/pySourceCode/darknet/Racoon_Kangaroo/kangaroo/images/'
image_id_path = '/home/gapcmgr/kt_privacy/pySourceCode/darknet/Racoon_Kangaroo/kangaroo/annots'
xml_path='/home/gapcmgr/kt_privacy/pySourceCode/darknet/Racoon_Kangaroo/kangaroo/annots/{}.xml'
txt_path='/home/gapcmgr/kt_privacy/pySourceCode/darknet/Racoon_Kangaroo/kangaroo/images/{}.txt'

# raconn set
# main_path = '/home/gapcmgr/kt_privacy/pySourceCode/darknet/Racoon_Kangaroo/racoon/'
# image_jpg_path = '/home/gapcmgr/kt_privacy/pySourceCode/darknet/Racoon_Kangaroo/racoon/images/'
# image_id_path = '/home/gapcmgr/kt_privacy/pySourceCode/darknet/Racoon_Kangaroo/racoon/annotations/'
# xml_path='/home/gapcmgr/kt_privacy/pySourceCode/darknet/Racoon_Kangaroo/racoon/annotations/{}.xml'
# txt_path='/home/gapcmgr/kt_privacy/pySourceCode/darknet/Racoon_Kangaroo/racoon/images/{}.txt'


main()
########



            

