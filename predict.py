import time
from multiprocessing import Process, Queue

import argparse
import numpy as np
import pandas as pd
import skimage.transform
import matplotlib.pyplot as plt

import lasagne
import theano
import pickle

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax, sigmoid
from lasagne.utils import floatX

def build_net(weights_path, target_size):
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_4'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_4'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_4'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['fc7_dropout'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)
    del net['fc7'], net['fc7_dropout'], net['fc8'], net['prob']
    
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096, name='tower_fc7')
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5, name='tower_fc7_dropout')
    net['fc8'] = DenseLayer(net['fc7_dropout'], num_units=2048, name='tower_fc8')
    net['fc8_dropout'] = DropoutLayer(net['fc8'], p=0.5, name='tower_fc8_dropout')
    net['fc9'] = DenseLayer(net['fc8_dropout'], num_units=target_size, nonlinearity=None, name='tower_fc9')
    net['prob'] = NonlinearityLayer(net['fc9'], sigmoid, name='tower_prob')
    
    values = pickle.load(open(weights_path, 'r'))['param values']    
    lasagne.layers.set_all_param_values(net['prob'], values)    
    return net

MEAN_VALUES = np.array([103.939, 116.779, 123.68]).reshape((3,1,1))
IMAGE_W = 224

def process_image(im):
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (IMAGE_W, w*IMAGE_W/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*IMAGE_W/w, IMAGE_W), preserve_range=True)

    # Central crop
    h, w, _ = im.shape
    im = im[h//2-IMAGE_W//2:h//2+IMAGE_W//2, w//2-IMAGE_W//2:w//2+IMAGE_W//2]
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # Convert RGB to BGR
    im = im[::-1, :, :]

    im = im - MEAN_VALUES
    return floatX(im[np.newaxis])

def deprocess(x):
    x = np.copy(x[0])
    x += MEAN_VALUES

    x = x[::-1]
    x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)
    
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def batch_reader(queue, users, batch_size, folder):
    indices = np.arange(len(users))
    for start_idx in xrange(0, len(users) - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        ims = []
        targets = []
        for u in users[excerpt, :]:
            try:
                im = plt.imread(folder + '%d_face.jpg' % u[0])
                ims.append(process_image(im))
                targets.append(u[1:])
            except Exception as e:
                continue
        queue.put((np.vstack(ims), np.vstack(targets)))
        while True:
            if queue.qsize() < 20:
                break
            time.sleep(1)
    queue.put('POISON')
    return
        
def iterate_minibatches(users, folder, batch_size):
    q = Queue()
    p = Process(target=batch_reader, args=(q,users,batch_size,folder))
    p.start()
    while True:
        message = q.get(timeout=5)
        if message == 'POISON':
            break
        yield message

def main():
    parser = argparse.ArgumentParser(description='run VGG 19 learning', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights', help='VGG 19 initial weights path', default='vgg19.pkl')
    parser.add_argument('--folder', help='images folder path', required=True)
    parser.add_argument('--targets', help='targets csv file path', required=True)
    parser.add_argument('--batch_size', help='batch size', default='32')
    args = parser.parse_args()
    
    users = pd.read_csv(args.targets, sep='\t', header=None, index_col=None).values
    net = build_net(args.weights, users.shape[1] - 1)
    
    input_im = theano.tensor.tensor4()
    target = theano.tensor.matrix()
    det_prediction = lasagne.layers.get_output(net['prob'], input_im, deterministic=True)
    det_loss = lasagne.objectives.binary_crossentropy(det_prediction, target).mean()
    eval_fun = theano.function([input_im, target],[det_loss,det_prediction], allow_input_downcast=True)
    
    im_count = 0
    with open('predictions.tsv', 'w') as preds_file, open('answers.tsv', 'w') as answers_file:
        for images, targets in iterate_minibatches(users, args.folder, int(args.batch_size)):
            loss, preds = eval_fun(images, targets)
            for i in xrange(len(preds)):
                preds_file.write('\t'.join(map(lambda val: str(val), preds[i,:])) + '\n')
                answers_file.write('\t'.join(map(lambda val: str(val), targets[i,:])) + '\n')
            preds_file.flush()
            answers_file.flush()
            im_count += len(images)
            print '%d / %d' % (im_count, users.shape[0])
    
if  __name__ == '__main__':
    main()
