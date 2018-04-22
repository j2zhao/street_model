import numpy
import scipy.io
import scipy.misc
import tensorflow as tf
from enum import Enum

ROOT = "../../data/"

class Size(Enum):
    S_100 = '100'
    S_500 = '500'
    S_1000 = '1000'
    S_5000 = '5000'
    S_10000 = '10000'
    FULL = 'full'
    VALIDATION = 'validation'
    TEST = 'test'

class Data(Enum):
    CUSTOM = "custom"
    STREET = "street"
    MNIST = "MNIST"

#return shuffled data and labels
def shuffle_data(data, labels):
    randomize = numpy.arange(data.shape[0])
    numpy.random.shuffle(randomize)
    data = data[randomize]
    labels = labels[randomize]
    return (data, labels)

def load_data(name, label = False):
    data = scipy.io.loadmat(name)
    if (label):
        data = data['labels']
        data = numpy.squeeze(tf.Session().run(tf.one_hot(data, 10)))
    else:
        data = data['images']
    return data

def generate_data_path(size, type, label = False):
    path = ROOT
    if (label):
        path = path + size.value + '/' + type.value + "_labels_" + size.value + '.mat'
    else:
        path = path + size.value + '/' + type.value + "_images_" + size.value + '.mat'
    return path

def generate_random_order(size):
    x = numpy.array([Data.CUSTOM, Data.STREET,Data.MNIST])
    x = numpy.repeat(x, size)
    numpy.random.shuffle(x)
    return x

# DEPRECATED
#def test_street_data():
    # data
#    train_data = load_data(DATA_TRAIN_IMAGES)
#    train_labels = load_data(DATA_TRAIN_LABELS, label = True)
#    test_data = load_data(DATA_TEST_IMAGES)
#    test_labels = load_data(DATA_TEST_LABELS, label = True)
#    [test_data, test_labels] = shuffle_data(test_data, test_labels)
#    [train_data, train_labels] = shuffle_data(train_data, train_labels)
    #image show
#    im = scipy.misc.toimage(numpy.squeeze(train_data[0,:,:,:]))
#    im.save('tmp.png')
#    print(train_labels[0,:])
#   im = scipy.misc.toimage(numpy.squeeze(test_data[0,:,:,:]))
#    im.save('tmp2.png')
#    print(test_labels[0,:])

# check on data tests
def main():
    test_street_data()
    
if __name__ == '__main__':
    main()