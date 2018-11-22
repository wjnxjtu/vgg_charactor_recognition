#coding=utf-8


import os

import cv2
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
PATH='./out3_num_abc'
#PATH='./ocrchar'
def key_value(filepath='/home/jnw/ocrmingpai/TextRecognitionDataGenerator-master/TextRecognitionDataGenerator/dicts/cn.txt'):
    f=open(filepath,'r')
    label_no={}
    i=0;
    line=f.readline()
    while line:
        line=line.strip('\n')
#        print line
        if line not in label_no.keys():
#            print i
            label_no[line]=i;
            i=i+1
        line=f.readline()
    return label_no

def load_data():
    label_no=key_value()
    train=[]
    labels=[]
    step=0
    for root ,dirs,files in os.walk(PATH):
        for file in files:
            train.append(os.path.join(PATH,file))
            labels.append(label_no[file.split('_')[0]])
    return train,labels
CHAR_LENGTH=len(key_value())
#train,label=load_data()
#for i,j in zip(train,label):
#    print i,j
def load_img(path):
    image=cv2.imread(path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.resize(image,(28,28))
    # b,g,r=cv2.split(image)
    # rgb_image=cv2.merge([r,g,b])
    return image

#def addWord(theIndex,word,pagenumber):
#    theIndex.setdefault(word, [ ]).append(pagenumber)#存在就在基础上加入列表，不存在就新建个字典key 


#def load_classes_list():
#    files=[]
#    for root,dirs,file in os.walk('/home/jnw/ocrmingpai/ocrtf/out'):
#        for i in file:
#            files.append(i)
#    classes_jpg={}
#    for file in files:
#        tmp=file.split('_')[0]
#        addWord(classes_jpg,tmp,file)
#    return classes_jpg
#def load_data():
#    classes_jpg=load_classes_list()
#    key_list=[]
#    for i in classes_jpg:
#        key_list.append(i)
#    train=[]
#    labels=[]
#    step=0
#    for keys in key_list:
#        for img_path in classes_jpg[keys]:        
#            path = os.path.join(PATH,img_path)
#            train.append(path)
#            labels.append(step)
#        step+=1
#    return train,labels
#train,labels=load_data()
#label=[]
#cnt=1
#lastlabel=-1
#for i,j in zip(train,labels):
##    print i,j
#    if j!=lastlabel:
#        print i,j
#        lastlabel=j
#    else:
#        continue
#image=load_img('./ocrchar/1_123.jpg')
import tensorflow as tf

def trans2tfRecord():
    train,labels = load_data()
    writer = tf.python_io.TFRecordWriter('ocrchar_abcnum_argu.tfrecords')
    cnt=1
    for examples,label in zip(train,labels):
        print("NO{}".format(cnt)),
        cnt+=1  
        #need to convert the example(bytes) to utf-8
        example1 = examples.encode("UTF-8")
        print example1,label
        image = load_img(example1)
        image_raw = image.tostring()
        
        example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw':_bytes_feature(image_raw),
                 'label': _int64_feature(label)                        
                }))
        writer.write(example.SerializeToString())
    writer.close()
def _int64_feature(value):  
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))    
def _bytes_feature(value):  
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) 
def read_and_decode(filename,batch_size,num_epoch=None):
    filename_queue = tf.train.string_input_producer([filename],shuffle=True,num_epochs=num_epoch)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [28, 28, 1])
    #image = tf.cast(image, tf.float32) * (1. / 255)
    label=tf.cast(features['label'],tf.int32)
    min_after_dequeue=4000
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([image, label],  
                                                batch_size= batch_size,  
                                                num_threads= 4, 
                                                min_after_dequeue=min_after_dequeue,
                                                capacity = capacity)
    #return image, label
    return image_batch, tf.reshape(label_batch, [batch_size]) 
trans2tfRecord()

# tfrecords_file = './ocrchar_test.tfrecords'
# image1,label=read_and_decode(filename=tfrecords_file,batch_size=1,num_epoch=1)
# # image_batch1=tf.cast(image1,tf.float32)
# image_batch1=image1
# label_batch=tf.cast(label,tf.int64)
# with tf.Session() as sess:
#     init_op = tf.group(tf.global_variables_initializer(),
#                        tf.local_variables_initializer())
#     sess.run(init_op)
#     coord=tf.train.Coordinator()
#     threads= tf.train.start_queue_runners(coord=coord)
#     i=0
#     try:
#         while not coord.should_stop():
#             example1,l = sess.run([image_batch1,label_batch])#在会话中取出image和label

#             img1=example1[0]
#             # cv2.imwrite(str(l[0])+'_'+str(i)+'.jpg',img1)
#             i=i+1
#             print i
# #            if i==3199:
# #                print label
#             # print l
#             break
            
#     except tf.errors.OutOfRangeError:
#         print 'Done records'
#     finally:
#         coord.request_stop()
        # coord.join(threads)
