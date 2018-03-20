"""
A pure TensorFlow implementation of a neural network. This can be
used as a drop-in replacement for a Keras model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import math
from AE_attacks.model import Model


class MLP(Model):
    """
    An example of a bare bones multilayer perceptron (MLP) class.
    """
    # layers: [conv2D , relu ...] , input_shape (batch_size,28,28,1) 
    def __init__(self, layers, input_shape):
        super(MLP, self).__init__()
        self.image_size = 32
        self.num_channels =3
        self.num_labels=10
        self.layer_names = []
        self.layers = layers
        self.input_shape = input_shape
        if isinstance(layers[-1], Softmax):
            # add name attribute to softmax
            layers[-1].name = 'probs'
            layers[-2].name = 'logits'
        else:
            layers[-1].name = 'logits'
        for i, layer in enumerate(self.layers):
            # build-in function hashattr, object.__class__.__name__ : get class name
            if hasattr(layer, 'name'):
                name = layer.name
            else:
                name = layer.__class__.__name__ + str(i)
            #  append : append value for list class 
            self.layer_names.append(name)

            layer.set_input_shape(input_shape)
            input_shape = layer.get_output_shape()

    def fprop(self, x, set_ref=False,training=True,restore_parameters=False,hash_retraining=False):
        states = []

        for layer in self.layers:
            #print(layer)
            if set_ref:
                layer.ref = x
            if hasattr(layer, 'is_BN')==True and restore_parameters==False and hash_retraining==False:
                x=layer.compute_bn(x,training)
            elif hasattr(layer, 'is_BN')==True and restore_parameters==False and hash_retraining==True:
                x=layer.fprop(x,training)
            elif hasattr(layer, 'is_BN')==False and hasattr(layer, 'is_dropout')==False:
                x = layer.fprop(x)	
            elif hasattr(layer, 'is_dropout')==True or (hasattr(layer, 'is_BN')==True and restore_parameters==True):
                x=layer.fprop(x,training)
            assert x is not None
            states.append(x)
        states = dict(zip(self.get_layer_names(), states))
        return states


class Layer(object):

    def get_output_shape(self):
        return self.output_shape


class Linear(Layer):
    # num_hid : the number of neurals
    def __init__(self, num_hid):
        self.num_hid = num_hid

    def set_input_shape(self, input_shape):
        batch_size, dim = input_shape
        self.input_shape = [batch_size, dim]
        self.output_shape = [batch_size, self.num_hid]
        init = tf.random_normal([dim, self.num_hid], dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                                   keep_dims=True))
        self.W = tf.Variable(init)
        self.b = tf.Variable(np.zeros((self.num_hid,)).astype('float32'))

        #print(self.W.shape)
    def fprop(self, x):
        return tf.matmul(x, self.W) + self.b

class HashLinear(Layer):
    # num_hid : the number of neurals
    def __init__(self, num_hid,compression):
        self.compression=compression
        self.num_hid = num_hid
        self.is_hash=True
    def get_index_weight(self,size_h_W,weight_lenth,weight_width):
        count = -1
        idx=np.zeros(weight_lenth*weight_width)
        extra_str='h_weight'
        rep=3
        for i in range(weight_lenth):
            for j in range(weight_width):
                count,key_i,key_j = count+1,'',''
                for r in range(rep):
                    key_i=key_i + str(i+r)
                    key_j=key_j + str(j+r)
                key = key_i + '_' + key_j + '_' +extra_str
                idx[count]=hash(key)%size_h_W

        return idx.reshape(weight_lenth,weight_width)
    def get_weight(self,h_W,index_weight):
        weight=np.matrix(index_weight)
        weight.fill(0)
        for i in range(len(h_W)):
            mask=index_weight==i
            weight_filler=np.ma.array(weight,mask=mask,fill_value=h_W[i])
            weight=weight_filler.filled()

        return weight		
	
    def set_input_shape(self, input_shape):
        batch_size, dim = input_shape
        # such as input 50*1000  output 50*10 weight 1000*10(it is different with torch)
        self.input_shape = [batch_size, dim]
        self.output_shape = [batch_size, self.num_hid]
		
        start_vars = set(x.name for x in tf.global_variables())	
		
        self.b = tf.Variable(np.zeros((self.num_hid,)).astype('float32'),name='hash_b')
        self.size_h_W=math.ceil(self.compression*dim*self.num_hid)
		
        # normal weight init
        init = np.random.randn(self.size_h_W)
        self.h_W = init / np.sqrt(1e-7 + np.sum(np.square(init)))
		
        # uniform weight init
        #stdv=1/np.sqrt(dim)
        #self.h_W = np.random.uniform(-stdv,stdv,self.size_h_W)	

        self.index_weight=self.get_index_weight(self.size_h_W, dim, self.num_hid)
        self.xi=self.get_index_weight(2, dim, self.num_hid)*2-1
        self.W=self.get_weight(self.h_W,self.index_weight)
        #print(np.shape(self.W))
        self.W = tf.Variable(self.W.astype('float32'),name='hash_W')
        self.xi = tf.constant(self.xi.astype('float32'))
		
		
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        self.init = tf.variables_initializer(var_list=new_vars)
		
    def fprop(self, x):
        return tf.matmul(x, tf.multiply(self.xi,self.W)) + self.b

class Conv2D(Layer):
    # output_channels: the number of output feature map  such as: (64,(5,5),(1,1),'SAME')   
    # for padding 'SMAE' means the size of output_feature_map is as same as input_feature_map 'VALID' means no padding
    def __init__(self, output_channels, kernel_shape, strides, padding):
        self.__dict__.update(locals())
        del self.self
    # as first layer, input_shape such as [128,32,32,3]
    def set_input_shape(self, input_shape):
        batch_size, rows, cols, input_channels = input_shape
        # kernel_shape  (5,5 ,3,64)
        kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                                   self.output_channels)												   
        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape
        # init weights 
        init = tf.random_normal(kernel_shape, dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init),
                                                   axis=(0, 1, 2)))
        self.kernels = tf.Variable(init)
        self.b = tf.Variable(
            np.zeros((self.output_channels,)).astype('float32'))
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = 1
        self.output_shape = tuple(output_shape)
        #print(output_shape)
        #print(np.shape(self.kernels))
    def fprop(self, x):
        #print('conv')
        #print(x)
        return tf.nn.conv2d(x, self.kernels, (1,) + tuple(self.strides) + (1,),
                            self.padding) + self.b


class ReLU(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def get_output_shape(self):
        return self.output_shape

    def fprop(self, x):
        return tf.nn.relu(x)

class MaxPooling(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        
        self.output_shape = tuple([shape[0],int(shape[1]/2),int(shape[2]/2),shape[3]])
        #print(self.output_shape)

    def fprop(self, x):
        return tf.layers.max_pooling2d(x, 2, 2)		

						  
class Dropout(Layer):

    def __init__(self,rate):
	    
        self.is_dropout=True
        self.rate=rate

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x,training):  
		
        return tf.layers.dropout(x, rate=self.rate, training=training)	

class BatchNormalization_nn(Layer):

    def __init__(self):
	    
        self.is_BN=True

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape
        self.epsilon=0.001
        self.n_out=self.input_shape[3]
        #print(self.n_out)
			
    def compute_bn(self,x,training):
        #print(training)
        self.beta = tf.Variable(tf.constant(0.0, shape=[self.n_out]),
                                     name='beta', trainable=True)
        self.gamma = tf.Variable(tf.constant(1.0, shape=[self.n_out]),
                                      name='gamma', trainable=True)
        self.batch_mean, self.batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        self.ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = self.ema.apply([self.batch_mean, self.batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(self.batch_mean), tf.identity(self.batch_var)

        self.mean, self.var = tf.cond(training,
                            mean_var_with_update,
                            lambda: (self.ema.average(self.batch_mean), self.ema.average(self.batch_var)))
        normed = tf.nn.batch_normalization(x,self.mean,self.var,self.beta,self.gamma,self.epsilon)
        return normed
    def fprop(self, x,training):

        return tf.nn.batch_normalization(x,self.mean,self.var,self.beta,self.gamma,self.epsilon)	

		
class Softmax(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.softmax(x)


class Flatten(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        output_width = 1
        for factor in shape[1:]:
            output_width *= factor
        self.output_width = output_width
        self.output_shape = [None, output_width]

    def fprop(self, x):
        return tf.reshape(x, [-1, self.output_width])


	
def make_DNN_model(input_shape=(None, 28, 28, 1)):

    layers = [Conv2D(192, (3, 3), (1, 1), "SAME"),
	          BatchNormalization_nn(),
              ReLU(),
			  Conv2D(128, (3, 3), (1,1), "SAME"),
			  BatchNormalization_nn(),
              ReLU(),
			  MaxPooling(),
              Conv2D(256, (3, 3), (1, 1), "VALID"),
			  BatchNormalization_nn(),
              ReLU(),
			  Conv2D(64, (5, 5), (1, 1), "VALID"),
			  BatchNormalization_nn(),
              ReLU(),
			  MaxPooling(),
			  Flatten(),
			  Dropout(0.5),
			  Linear(512),
			  ReLU(),
			  Dropout(0.5),
			  Linear(512),
			  ReLU(),
			  Dropout(0.5),
			  Linear(10),
              Softmax()]

    model = MLP(layers, input_shape)
    return model
		
def make_vgg19_model(input_shape=(None, 32, 32, 3)):

    layers = [Conv2D(64, (3, 3), (1, 1), "SAME"),
	          BatchNormalization_nn(),
              ReLU(),
			  Conv2D(64, (3, 3), (1,1), "SAME"),
			  BatchNormalization_nn(),
              ReLU(),
			  MaxPooling(),
			  
              Conv2D(128, (3, 3), (1, 1), "SAME"),
			  BatchNormalization_nn(),
              ReLU(),
			  Conv2D(128, (3, 3), (1, 1), "SAME"),
			  BatchNormalization_nn(),
              ReLU(),
			  MaxPooling(),
			  
              Conv2D(256, (3, 3), (1, 1), "SAME"),
			  BatchNormalization_nn(),
              ReLU(),
			  Conv2D(256, (3, 3), (1, 1), "SAME"),
			  BatchNormalization_nn(),
              ReLU(),
              Conv2D(256, (3, 3), (1, 1), "SAME"),
			  BatchNormalization_nn(),
              ReLU(),
			  Conv2D(256, (3, 3), (1, 1), "SAME"),
			  BatchNormalization_nn(),
              ReLU(),
			  MaxPooling(),

	          Conv2D(512, (3, 3), (1, 1), "SAME"),
			  BatchNormalization_nn(),
              ReLU(),
			  Conv2D(512, (3, 3), (1, 1), "SAME"),
			  BatchNormalization_nn(),
              ReLU(),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
			  BatchNormalization_nn(),
              ReLU(),
			  Conv2D(512, (3, 3), (1, 1), "SAME"),
			  BatchNormalization_nn(),
              ReLU(),
			  MaxPooling(),
			  
	          Conv2D(512, (3, 3), (1, 1), "SAME"),
			  BatchNormalization_nn(),
              ReLU(),
			  Conv2D(512, (3, 3), (1, 1), "SAME"),
			  BatchNormalization_nn(),
              ReLU(),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
			  BatchNormalization_nn(),
              ReLU(),
			  Conv2D(512, (3, 3), (1, 1), "SAME"),
			  BatchNormalization_nn(),
              ReLU(),
			  MaxPooling(),	

			  
			  Flatten(),
			  Dropout(0.5),
			  Linear(512),
			  ReLU(),
			  Dropout(0.5),
	          Linear(512),
			  ReLU(),
			  Dropout(0.5),
              Linear(10),
              Softmax()]

    model = MLP(layers, input_shape)
    #print(model.layers[3].W)
    return model

def get_hash_retraining_model(model,nb_classes=10):
    compression_rate1=1/4192
    compression_rate2=1/512

    hash_classifier_layer = [HashLinear(512,compression_rate1),
	                         HashLinear(nb_classes,compression_rate2)
					        ]
    batchsize=None
    input_shape=(batchsize, 512)
    hash_classifier = MLP(hash_classifier_layer, input_shape)
    #print(hash_classifier)
    model.layers[-5] = hash_classifier.layers[0]
    model.layers[-2] = hash_classifier.layers[1]

    #print(model.layers)
    return model
