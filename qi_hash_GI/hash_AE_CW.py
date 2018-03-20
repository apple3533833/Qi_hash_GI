from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
import math
import time
import logging
import os
from AE_attacks.attacks import CarliniWagnerL2,CarliniWagnerL0,FastGradientMethod,BasicIterativeMethod,SaliencyMapMethod
from AE_attacks.utils import pair_visual, grid_visual, AccuracyReport
from AE_attacks.utils import set_log_level,other_classes
from utils_mnist import data_mnist
from utils_cifar10 import data_cifar10
from AE_attacks.utils import AccuracyReport,batch_indices, _ArgsWrapper, create_logger
from utils_tf import tf_model_load,model_argmax,generate_data
from hash_utils import model_loss,hash_gradient,hash_gradient_computation_handy_update,model_train_evaluation,model_eval
from hash_models import make_DNN_model,make_vgg19_model,get_hash_retraining_model
from setup_cifar import CIFAR
from AE_attacks.l2_attack import CarliniL2
from AE_attacks.l0_attack import CarliniL0
from AE_attacks.li_attack import CarliniLi

from scipy import ndimage
FLAGS = flags.FLAGS

		  
def test_GSM_attack(sess,model,x,y,training,X_test,type='fgsm',eval_params=None,gsm_params=None):
    if type=='fgsm':
        fgsm = FastGradientMethod(model, sess=sess,x_clean=x,training_BN=training)
        adv_x_gsm = fgsm.generate(x, **gsm_params)
    elif type=='igsm':	
        igsm = BasicIterativeMethod(model, sess=sess,x_clean=x,training_BN=training)	
        adv_x_gsm = igsm.generate(x, **gsm_params)

	
    adv_x_gsm = tf.stop_gradient(adv_x_gsm)
	
    preds_gsm = model.get_probs_BN(adv_x_gsm,training=training,restore_parameters=True)		
    preds_model = model.get_probs_BN(x,training=training,restore_parameters=True)
    preds_max = tf.reduce_max(preds_model, 1, keep_dims=True)
    model_Y = tf.to_float(tf.equal(preds_model, preds_max))
    model_Y = tf.stop_gradient(model_Y)          
    model_Y = sess.run(model_Y,feed_dict = {x: X_test[:1000],training:False})
	
    success_rate = 1.0 - model_eval(sess, x, y, training,preds_gsm, X_test[:1000],model_Y, args=eval_params)
    print('success rate on adversarial examples: %0.4f' % success_rate)
    return success_rate

def test_JSMA_attack(sess,model,preds,x,y,training,X_test,Y_test,source_samples=None): 

    nb_classes=10
    # Keep track of success (adversarial example classified in target)
    results = np.zeros((nb_classes, source_samples), dtype='i')
    results_acc = np.zeros((nb_classes, source_samples), dtype='i')
    # Rate of perturbed features for each test set example and target class
    perturbations = np.zeros((nb_classes, source_samples), dtype='f')

    # Instantiate a SaliencyMapMethod attack object
    jsma = SaliencyMapMethod(model, back='tf', sess=sess,x_clean = x,training_BN=training)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}

    figure = None
    # Loop over the samples we want to perturb into adversarial examples
    for sample_ind in xrange(0, source_samples):
        print('--------------------------------------')
        print('Attacking input %i/%i' % (sample_ind + 1, source_samples))
        sample = X_test[sample_ind:(sample_ind+1)]
        #sample = X_test[10+sample_ind:10+(sample_ind+1)]
        # We want to find an adversarial example for each possible target class
        # (i.e. all classes that differ from the label given in the dataset)
        current_class = int(np.argmax(Y_test[sample_ind]))
        target_classes = other_classes(nb_classes, current_class)


        # Loop over all target classes
        for target in target_classes:
            print('Generating adv. example for target class %i' % target)

            # This call runs the Jacobian-based saliency map approach
            one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
            one_hot_target[0, target] = 1
            jsma_params['y_target'] = one_hot_target
            #print(jsma_params)
            adv_x = jsma.generate_np(sample, **jsma_params)
            feed = {x:adv_x,training:False}
            # Check if success was achieved
            res = int(model_argmax(sess, x, preds, adv_x,feed) == target)
            res_acc = int(model_argmax(sess, x, preds, adv_x,feed) == current_class)
            # Computer number of modified features
            adv_x_reshape = adv_x.reshape(-1)
            test_in_reshape = X_test[sample_ind].reshape(-1)
            nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
            percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]

            # Update the arrays for later analysis
            results[target, sample_ind] = res
            results_acc[target, sample_ind] = res_acc
            print(results)
            perturbations[target, sample_ind] = percent_perturb

    print('--------------------------------------')

    # Compute the number of adversarial examples that were successfully found
    nb_targets_tried = ((nb_classes - 1) * source_samples)
    succ_rate = float(np.sum(results)) / nb_targets_tried
    print('Avg. rate of successful adv. examples {0:.4f}'.format(succ_rate))


    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(perturbations)
    print('Avg. rate of perturbed features {0:.4f}'.format(percent_perturbed))
	
def hash_GI_defense_tutorial(nb_epochs=50, nb_hash_retraining_epochs=3,batch_size=128,targeted=True,hash_retraining=False,
                   learning_rate=0.001,
				   source_samples=10,train_dir="/tmp",
                   filename="mnist.ckpt", load_model=False,
				   nb_classes=10):



    report = AccuracyReport()
    tf.set_random_seed(1234)
    set_log_level(logging.DEBUG)


    sess = tf.Session()

    # Get MNIST test data  60000*28*28*1   10000*28*28*1  60000*10 10000*10
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=0,train_end=60000,test_start=0,test_end=10000)

    #X_train, Y_train, X_test, Y_test = data_cifar10()

	
    # mnist
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    #x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    #y = tf.placeholder(tf.float32, shape=(None, 10))
	# for BN
    training = tf.placeholder(tf.bool)


    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
		'train_dir': train_dir,
        'filename': filename
    }
 
    rng = np.random.RandomState([2017, 8, 30])

    # mnist
    model = make_DNN_model(input_shape=(None, 28, 28, 1))
    # cifar10	
    #model = make_DNN_model(input_shape=(None, 32, 32, 3))
    #model = make_vgg19_cifar10_nn()

    preds = model.get_probs_BN(x,training=training)

    def evaluate():

        eval_params = {'batch_size': batch_size}
        acc= model_eval(
                sess, x, y, training, preds, X_test, Y_test, args=eval_params)
        report.clean_train_clean_eval = acc
        print('Test accuracy on legitimate examples: %0.4f' % acc)	

    ckpt = tf.train.get_checkpoint_state(train_dir)
    ckpt_path = False if ckpt is None else ckpt.model_checkpoint_path

    rng = np.random.RandomState([2017, 8, 30])

    if load_model and ckpt_path:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)
        print("Model loaded from: {}".format(ckpt_path))
        evaluate()
    else:
        print("Model was not loaded, training from scratch.")
        model_train_evaluation(sess, x, y, training, preds, X_train, Y_train,evaluate=evaluate,save=True,hash_retraining=False,
                    args=train_params, rng=rng,model=model)
    def AE_test():
     

        eval_params = {'batch_size': batch_size}    
        fgsm_params = {'eps':0.3,'clip_min': 0.,'clip_max': 1.} 
        igsm_params = {'eps': 0.3,'eps_iter':0.03,'clip_min': 0.,'clip_max': 1.,'nb_iter':40}
        print("--------------FGSM attack-------------")
        test_GSM_attack(sess,model,x,y,training,X_test,type='fgsm',eval_params=eval_params,gsm_params=fgsm_params)
        print("--------------BIM attack-------------")
        test_GSM_attack(sess,model,x,y,training,X_test,type='igsm',eval_params=eval_params,gsm_params=igsm_params)  
        
        print("--------------JSMA-Z attack-------------") 
        preds = model.get_probs_BN(x,training=training,restore_parameters=True)
        test_JSMA_attack(sess,model,preds,x,y,training,X_test,Y_test,source_samples=source_samples)	
        
        print("-------------CW attack-----------------")
        print("This could take some time ...")
        print('--------------CW L2------------------')	
        cw = CarliniWagnerL2(model, back='tf', sess=sess,x_clean=x,training_BN=training)

        if targeted:       
            adv_inputs, adv_ys = generate_data(X_test,Y_test, samples=source_samples, targeted=targeted,start=0, inception=False)
            yname = "y_target"
        else:   
            adv_inputs = X_test[:source_samples]
            adv_ys = None
            yname = "y"

        cw_params = {'binary_search_steps' : 9,
                 yname: adv_ys,
                 'max_iterations': 1000,
		         'learning_rate': 1e-2,
				 'batch_size':9,
				 'initial_const':1e-3
				 }

        adv = cw.generate_np(adv_inputs,**cw_params)
	

        if targeted:  
            adv_success_rate = model_eval(sess, x, y, training ,preds, adv, adv_ys, args=eval_params)
        else:      
            adv_success_rate = 1 - model_eval(sess, x, y, preds, adv, Y_test[:source_samples], args=eval_params)

        # Compute the number of adversarial examples that were successfully found
        print('Avg. rate of successful adv. examples {0:.4f}'.format(adv_success_rate))
        percent_perturbed = np.mean(np.sum((adv - adv_inputs)**2,axis=(1, 2, 3))**.5)
        print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))

	    
        print('---------------- CW L0 ------------------')
        eval_params = {'batch_size': 1}
        attack = CarliniL0(sess , model, x_clean=x,training_BN=training)

        adv_inputs, adv_ys = generate_data(X_test,Y_test, samples=source_samples, targeted=targeted,start=0, inception=False)
        adv = attack.attack(adv_inputs, adv_ys)
	
        if targeted:   
            adv_success_rate = model_eval(sess, x, y, training ,preds, adv, adv_ys, args=eval_params)
        else:      
            adv_success_rate = 1 - model_eval(sess, x, y, preds, adv, Y_test[:source_samples], args=eval_params)

    
        # Compute the number of adversarial examples that were successfully found
        print('Avg. rate of successful adv. examples {0:.4f}'.format(adv_success_rate))
        percent_perturbed = np.mean(np.sum((adv - adv_inputs)**2,axis=(1, 2, 3))**.5)
        print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))

	
        print('---------------- CW Li ------------------')
        attack = CarliniLi(sess , model, x_clean=x,training_BN=training)
        adv_inputs, adv_ys = generate_data(X_test,Y_test, samples=source_samples, targeted=targeted,start=0, inception=False)
        adv = attack.attack(adv_inputs, adv_ys)


        if targeted:
            adv_success_rate = model_eval(sess, x, y, training ,preds, adv, adv_ys, args=eval_params)
        else:      
            adv_success_rate = 1 - model_eval(sess, x, y, preds, adv, Y_test[:source_samples], args=eval_params)

        # Compute the number of adversarial examples that were successfully found
        print('Avg. rate of successful adv. examples {0:.4f}'.format(adv_success_rate))
        percent_perturbed = np.mean(np.sum((adv - adv_inputs)**2,axis=(1, 2, 3))**.5)
        print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))	
	

    def gradient_inhibition(type=None,constant_uniform=0.2,constant_w_multiple=3,constant=0.1):	

        if type=='constant':
            model.layers[-8].W=model.layers[-8].W+constant*tf.sign(model.layers[-8].W)
            model.layers[-5].W=model.layers[-5].W+constant*tf.sign(model.layers[-5].W)
            model.layers[-2].W=model.layers[-2].W+constant*tf.sign(model.layers[-2].W)			
        elif type=='w_multiple':		
            model.layers[-8].W=model.layers[-8].W+constant_w_multiple*tf.multiply(tf.abs(model.layers[-8].W),tf.sign(model.layers[-8].W))
            model.layers[-5].W=model.layers[-5].W+constant_w_multiple*tf.multiply(tf.abs(model.layers[-5].W),tf.sign(model.layers[-5].W))
            model.layers[-2].W=model.layers[-2].W+constant_w_multiple*tf.multiply(tf.abs(model.layers[-2].W),tf.sign(model.layers[-2].W))
        elif type=='uniform':
            uniform_w1_pert=tf.random_uniform(tf.shape(model.layers[-8].W),0,1)
            uniform_w2_pert=tf.random_uniform(tf.shape(model.layers[-5].W),0,1)
            uniform_w3_pert=tf.random_uniform(tf.shape(model.layers[-2].W),0,1)
            model.layers[-8].W=model.layers[-8].W+constant_uniform*tf.multiply(uniform_w1_pert,tf.sign(model.layers[-8].W))
            model.layers[-5].W=model.layers[-5].W+constant_uniform*tf.multiply(uniform_w2_pert,tf.sign(model.layers[-5].W))
            model.layers[-2].W=model.layers[-2].W+constant_uniform*tf.multiply(uniform_w3_pert,tf.sign(model.layers[-2].W))	

			
    print("AE attack -> DNN")   
    #AE_test()
	# hash retraining
    if hash_retraining == True:	
        print("hash retraining:")
        model = get_hash_retraining_model(model)
        preds = model.get_probs_BN_hash_retraining(x,training=training,hash_retraining=True)
        hash_retrain_params = {
        'nb_epochs': nb_hash_retraining_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate/10,
		'train_dir': train_dir,
        'filename': filename
        }
        model_train_evaluation(sess, x, y, training, preds, X_train, Y_train,evaluate=evaluate,save=False,hash_retraining=True,
                    args=hash_retrain_params, rng=rng,model=model)	

    print("GI processing")
    gradient_inhibition(type='uniform',constant_uniform=0.2)

    preds = model.get_probs_BN(x,training=training,restore_parameters=True)
    evaluate()
    print("AE attack -> hash+GI model")   
    AE_test()    
    sess.close()
    return report


def main(argv=None):

    hash_GI_defense_tutorial(nb_epochs=FLAGS.nb_original_epochs,
                      nb_hash_retraining_epochs=FLAGS.nb_hash_retraining_epochs,
                      batch_size=FLAGS.batch_size,
                      nb_classes=FLAGS.nb_classes,
                      source_samples=FLAGS.source_samples,
                      learning_rate=FLAGS.learning_rate,
					  train_dir=FLAGS.train_dir,
                      filename=FLAGS.filename,
                      load_model=FLAGS.load_model,
					  hash_retraining=FLAGS.hash_retraining
					  )


if __name__ == '__main__':
    flags.DEFINE_integer('nb_original_epochs', 50, 'Number of epochs to train DNN model')
    flags.DEFINE_integer('nb_hash_retraining_epochs', 3, 'Number of epochs to train DNN model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_integer('nb_classes', 10, 'Number of output classes')
    flags.DEFINE_integer('source_samples', 10, 'Nb of test inputs to taggeted attack(JSMA and CW)')
    flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
    flags.DEFINE_string('train_dir', 'model/mnist_DNN_model', 'Directory where to save model.')
    flags.DEFINE_string('filename', 'mnist_DNN_model.ckpt', 'Checkpoint filename.')
    #flags.DEFINE_string('train_dir', 'model/cifar10_DNN_model', 'Directory where to save model.')
    #flags.DEFINE_string('filename', 'cifar10_DNN_model.ckpt', 'Checkpoint filename.')
    #flags.DEFINE_string('train_dir', 'model/cifar_nn_vgg_BN_model', 'Directory where to save model.')
    #flags.DEFINE_string('filename', 'cifar_nn_vgg_BN_model.ckpt', 'Checkpoint filename.')
    flags.DEFINE_boolean('load_model', True, 'Load saved model or train.')
    flags.DEFINE_boolean('hash_retraining', True, 'if it use the hash retraining')


    tf.app.run()
