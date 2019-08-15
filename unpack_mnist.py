'''
Created on Jun 27, 2018

@author: david
'''

import struct
import numpy as np
import gzip

import dapnn.data as data

def read_imagedata(file):
    
    print('reading: ' + file)
    
    with gzip.open(file, 'rb') as f:
        magic = f.read(4)
        assert struct.unpack('>i', magic)[0] == 2051
        
        lines = struct.unpack('>i', f.read(4))[0]
        rows = struct.unpack('>i', f.read(4))[0]
        cols = struct.unpack('>i', f.read(4))[0]
        
        images = list()
        
        for _ in range(0, lines):
            image = f.read(rows * cols)
            b_image = struct.unpack('>' + str(rows*cols) + 'B', image)
            images.append(b_image)
            
        return np.array(images, dtype=np.float32)
            

def read_labeldata(file):
    
    print('reading: ' + file)
    
    with gzip.open(file, 'rb') as f:
        magic = f.read(4)
        assert struct.unpack('>i', magic)[0] == 2049
        
        lines = struct.unpack('>i', f.read(4))[0]

        labels = list()
        
        for _ in range(0, lines):
            label = np.zeros(10)
            label[struct.unpack('>B', f.read(1))[0]] = 1
            labels.append(label)
            
        return np.array(labels, dtype=np.float32)

def load_traindata():
    images = read_imagedata('./raw_data/train-images-idx3-ubyte.gz')
    labels = read_labeldata('./raw_data/train-labels-idx1-ubyte.gz')
    
    train = data.dataset([images[0:55000], labels[0:55000]])
    val = data.dataset([images[55000:60000], labels[55000:60000]])
    
    return train, val

def load_testdata():
    images = read_imagedata('./raw_data/t10k-images-idx3-ubyte.gz')
    labels = read_labeldata('./raw_data/t10k-labels-idx1-ubyte.gz')
    
    return data.dataset([images, labels])



if __name__ == '__main__':
    train_data, val_data = load_traindata()
    test_data = load_testdata()
    train_data.save('./datasets/mnist_train.data')
    val_data.save('./datasets/mnist_val.data')
    test_data.save('./datasets/mnist_test.data')

