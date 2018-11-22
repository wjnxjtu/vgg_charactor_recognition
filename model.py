from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys,os,glob
import argparse
from official.utils.arg_parsers import parsers
from official.utils.logging import hooks_helper


def key_value(filepath='/home/jnw/ocrmingpai/TextRecognitionDataGenerator-master/TextRecognitionDataGenerator/dicts/cn.txt'):
    f=open(filepath,'r')
    label_no={}
    i=0;
    line=f.readline()
    while line:
        line=line.strip('\n')
        if line not in label_no.keys():
            label_no[line]=i;
            i=i+1
        line=f.readline()
    f.close()
    return label_no
CHAR_LENGTH=len(key_value())

def parse(record):
    features=tf.parse_single_example(record,
                                     features={
                                        'image_raw':tf.FixedLenFeature([], tf.string),
                                        'label':tf.FixedLenFeature([], tf.int64)
                                     }
                                     )
    decoded_image=tf.decode_raw(features['image_raw'],tf.uint8)
    decoded_image=tf.reshape(decoded_image,[28,28,1])
    decoded_image=tf.cast(decoded_image,tf.float32)
    label=features['label']
    label=tf.cast(label,tf.int64)
    return decoded_image,label

def input_fn_general(is_training,
                     filename,
                     batch_size=64,
                     num_epoches=1,
                     num_parallel_calls=1,
                     ):
    dataset=tf.data.TFRecordDataset(['ocrchar_train.tfrecords','ocrchar_train_abcnum_argu.tfrecords','ocrchar_train_initial_argu.tfrecords'])
    dataset=dataset.map(parse,num_parallel_calls=num_parallel_calls)
    if is_training:
        # buffer_size=len(get_filenames())
        dataset=dataset.shuffle(buffer_size=40000)#buffer_size=len(get_filenames())
        dataset=dataset.repeat(num_epoches)
    dataset=dataset.batch(batch_size)
    iterator=dataset.make_one_shot_iterator()
    batch_image,batch_label=iterator.get_next()
    return batch_image,batch_label

class DANArgParser(argparse.ArgumentParser):
  """Arguments for configuring and running a Resnet Model.
  """

  def __init__(self):
    super(DANArgParser, self).__init__(parents=[
        parsers.BaseParser(),
        parsers.PerformanceParser(),
        parsers.ImageModelParser(),
    ])

    self.add_argument(
        "--data_dir_test", "-ddt", default='./ocrchar_test.tfrecords',
        help="[default: %(default)s] The location of the test data.",
        metavar="<DD>",
    )

    self.add_argument(
        '--mode','-mode',type=str,default='train',
        choices=['train','eval','infer']
    )


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
def batch_norm(inputs,training,data_format):
    return tf.layers.batch_normalization(inputs=inputs, axis=1 if data_format == 'channels_first' else -1,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)

def vgg_block(inputs,filters,num_convs,training,kernel_size,maxpool,data_format):
    for i in range(num_convs):
        inputs = tf.layers.conv2d(inputs,filters,kernel_size,1,
                         padding='same',activation=tf.nn.relu,
                         kernel_initializer=tf.glorot_uniform_initializer(),
                         data_format=data_format)
        # inputs = batch_norm(inputs,training=training,data_format=data_format)
    if maxpool:
        inputs = tf.layers.max_pooling2d(inputs,2,2)

    return inputs

class Model(object):

    def __init__(self,
                 img_size,
                 filter_sizes,
                 num_convs,
                 kernel_size,
                 data_format=None):
        if not data_format :
            data_format=('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
        self.data_format=data_format
        self.img_size=img_size
        self.num_convs=num_convs;
        self.kernel_size=kernel_size
        self.filter_sizes=filter_sizes

    def __call__(self,
                 inputs_imgs,
                 training):
        inputs_imgs = tf.reshape(inputs_imgs, [-1, self.img_size, self.img_size, 1])
        tf.summary.image('image', inputs_imgs, max_outputs=6)

        # Convert the inputs from channels_last (NHWC) to channels_first
        # (NCHW).
        # This provides a large performance boost on GPU.  See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        with tf.variable_scope('vgg16'):
            inputs = inputs_imgs

            if self.data_format == 'channels_first':#should there be channels_last?to be checked
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            for i, num_filter in enumerate(self.filter_sizes):
                inputs = vgg_block(inputs=inputs,filters=num_filter,num_convs=self.num_convs,
                                  training=training,kernel_size=self.kernel_size,maxpool=True,
                                  data_format=self.data_format)
        
            inputs = tf.contrib.layers.flatten(inputs)
            #inputs = tf.layers.dropout(inputs,0.5,training=training)#remove dropout layer, it is said dropout isn't as usefuel as bn
            fc1 = tf.layers.dense(inputs,4096,activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer())
            # fc1 = batch_norm(fc1,training,data_format=self.data_format)

            fc2 = tf.layers.dense(fc1,4096,activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer())
            # fc2 = batch_norm(fc1,training,data_format=self.data_format)

            s1_fc2 = tf.layers.dense(fc2, CHAR_LENGTH,activation=None)
            rd = tf.identity(s1_fc2,name='output_label')
        return rd

class VGG16Model(Model):
    def __init__(self,data_format=None):
        
        img_size=28
        filter_sizes=[64,128,256,512]
        num_convs=2
        kernel_size=3

        super(VGG16Model,self).__init__(
            img_size=img_size,
            filter_sizes=filter_sizes,
            num_convs=num_convs,
            kernel_size=kernel_size,
            data_format=data_format
        )

def model_fn_vgg16(features,
                   groundtruth,
                   mode,
                   model_class,
                   data_format, 
                   multi_gpu=False):

    if isinstance(features, dict):
        print('features is dict')
        features = features['image_raw']

    model = model_class(data_format)
    result = model(features,mode==tf.estimator.ModeKeys.TRAIN )
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=result
        )

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=groundtruth)
    loss = tf.reduce_mean(loss)

    # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'vgg16')):
    #     optimizer = tf.train.AdamOptimizer(0.0001)
    #     if multi_gpu:
    #         optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
    #     train_op = optimizer.minimize(loss,global_step=tf.train.get_or_create_global_step(),
    #                                         var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'vgg16'))
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    if multi_gpu:
        optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
    train_op = optimizer.minimize(loss,global_step=tf.train.get_or_create_global_step(),
                                        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'vgg16'))
    
    # correct = tf.nn.in_top_k(result,groundtruth,1)
    # correct = tf.cast(correct,tf.float16)
    # acc=tf.reduce_mean(correct)
    eval_metric_ops={'acc':tf.metrics.accuracy(labels=groundtruth,predictions=tf.argmax(input=result,axis=1))}
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=result,
            loss=loss,
            train_op=train_op
            )
    if mode==tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=result,
            loss=loss,
            train_op=None,
            eval_metric_ops=eval_metric_ops
            )

def model_fn(features,labels,mode,params):
    return model_fn_vgg16(features=features,
                          groundtruth=labels,
                          mode=mode,
                          model_class=VGG16Model,
                          data_format=params['data_format'],
                          multi_gpu=params['multi_gpu']
                          )


def main(argv):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    parser=DANArgParser()
    parser.set_defaults(data_dir=None,#'ocrchar_train.tfrecords','ocrchar_train_abcnum_argu.tfrecords','ocrchar_train_initial_argu.tfrecords',
                        model_dir='./model',
                        data_format='channels_last',
                        train_epochs=10,
                        batch_size=64)
    flags=parser.parse_args(args=argv[1:])
    flags_trans = { 
        'train':tf.estimator.ModeKeys.TRAIN,
        'eval':tf.estimator.ModeKeys.EVAL,
        'infer':tf.estimator.ModeKeys.PREDICT
              }
    
    model_function=model_fn
    if flags.multi_gpu:
        validate_batch_size_for_multi_gpu(flags.batch_size)
        model_function = tf.contrib.estimator.replicate_model_fn(model_function,loss_reduction=tf.losses.Reduction.MEAN)

    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=flags.inter_op_parallelism_threads,
        intra_op_parallelism_threads=flags.intra_op_parallelism_threads,
        allow_soft_placement=True)
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=240,
                                                session_config=session_config)


    classifier=tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=flags.model_dir,
        config=run_config,
        params={
            'data_format':flags.data_format,
            'multi_gpu':flags.multi_gpu,
            'batch_size':flags.batch_size
        }
        )
    def input_fn_train():
        print('--------input_fn_train: flags.data_dir----------\n:', flags.data_dir)
        return input_fn_general(True,flags.data_dir,flags.batch_size,flags.epochs_per_eval,flags.num_parallel_calls)
    def input_fn_eval():
        print('--------input_fn_eval: flags.data_dir_test----------\n:', flags.data_dir_test)
        return input_fn_general(False,flags.data_dir_test,256,1,flags.num_parallel_calls)

    if flags.mode==tf.estimator.ModeKeys.TRAIN:
        i=1
        for _ in range(flags.train_epochs//flags.epochs_per_eval):
            print ('-------------starting a training cycle-------------',i)
            i+=1
            # try:
            classifier.train(input_fn=input_fn_train,max_steps=flags.max_train_steps)
            # except tf.errors.OutOfRangeError:
                # print(i-1,'iterator(s)')

            print ('-------------starting  evaluation------------------')
            # try:
            eval_results=classifier.evaluate(input_fn=input_fn_eval,steps=flags.max_train_steps)
            # except tf.errors.OutOfRangeError:
                # print('evaluation end')
            print(eval_results)

    if flags.mode==tf.estimator.ModeKeys.EVAL:
        eval_results=classifier.evaluate(input_fn=input_fn_eval,steps=flags.max_train_steps)
        print(eval_results['acc'])
    if flags.mode==tf.estimator.ModeKeys.PREDICT:
        label_no=key_value()
        no_label={}
        for i in label_no:
            no_label[label_no[i]]=i
        import cv2
        def input_fn_predict():
            def decode(image_path):
                image=cv2.imread(image_path)
                image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)    
                image=cv2.resize(image,(28,28))
                image=np.reshape(image,(28,28,1))
                # image=tf.image.convert_image_dtype(image,dtype=tf.float32)
                return image.astype(np.float32)

            image_path=get_filenames()
            image_list=[]
            for name in image_path:
                image_list.append(decode(name))
            dataset=tf.data.Dataset.from_tensor_slices(np.array(image_list))
            dataset=dataset.batch(1)
            return dataset.make_one_shot_iterator().get_next()
        def argmaxN(label):
            res=[0,1]
            for i in range(2,len(label)):
                if label[i]>label[res[0]]:
                    res[0]=i
                elif label[i]>label[res[1]]:
                    res[1]=i
            if label[res[0]]<label[res[1]]:
                res[0],res[1]=res[1],res[0]
            return res

        def get_filenames():
            listext=['*.png','*.jpg']
            imagelist=[]
            for ext in listext:
                p=os.path.join('.',ext)
                imagelist.extend(glob.glob(p))
            return imagelist
        predict_results=classifier.predict(input_fn=input_fn_predict)
        filenames=get_filenames()
        cnt=0
        while True:
            try:
                label=next(predict_results)
                print('groundtruth: ',filenames[cnt],'\nprediction: ',no_label[np.argmax(label,0)])
                cnt+=1
                index=argmaxN(label)
                for i in index:
                    print(no_label[i],)
                print('\n')
            except StopIteration:
                print("End of dataset")
                break
                

if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(argv=sys.argv)