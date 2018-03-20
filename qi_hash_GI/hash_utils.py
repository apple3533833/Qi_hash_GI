

import tensorflow as tf
from AE_attacks.utils import AccuracyReport,batch_indices, _ArgsWrapper, create_logger
import math
import time
import numpy as np
import logging
import os
def model_loss(y, model, mean=True):

    op = model.op
    if "softmax" in str(op).lower():
        logits, = op.inputs
    else:
        logits = model

    out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

    if mean:
        out = tf.reduce_mean(out)
    return out

	
def hash_gradient(grads,vars,layer):

    old_grad_weight=np.matrix(grads)
    index_weight=layer.index_weight
    grad_weight=np.matrix(old_grad_weight)
    grad_weight.fill(0)
    for i in range(len(layer.index_weight)):

        mask=index_weight==i
        grad=np.mean(old_grad_weight[index_weight==i])
        grad_weight_filler=np.ma.array(grad_weight,mask=mask,fill_value=grad)
        grad_weight=grad_weight_filler.filled()

    grads=grad_weight

    return grads,vars

def hash_gradient_computation_handy_update(x,y,feed_dict,optimizer,loss,model,learning_rate):
    grads_and_vars = optimizer.compute_gradients(loss)
    grads_and_vars_new=[]

    grads1,vars1=hash_gradient(grads_and_vars[-3][0].eval(feed_dict=feed_dict),grads_and_vars[-3][1],model.layers[-5])	
    model.layers[-5].W=model.layers[-5].W-tf.constant(grads1)*learning_rate

    grads2,vars2=hash_gradient(grads_and_vars[-1][0].eval(feed_dict=feed_dict),grads_and_vars[-1][1],model.layers[-2])	
    model.layers[-2].W=model.layers[-2].W-tf.constant(grads2)*learning_rate	


	
	
def model_train_evaluation(sess, x, y, training,predictions, X_train, Y_train, save=False,evaluate=None,hash_retraining=False,args=None, rng=None,model=None):

    args = _ArgsWrapper(args or {})
    training_BN=True 

    if rng is None:
        rng = np.random.RandomState()

    # Define loss
    loss = model_loss(y, predictions)


    if hash_retraining==False:
        optimizer =  tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)				
        optimizer = optimizer.minimize(loss)
    #  the with keyword is used when working with unmanaged resources
    with sess.as_default():
        if hash_retraining==True:
            sess.run([model.layers[-2].init,model.layers[-5].init])
            batch_size_hash=4096
        else:
            tf.global_variables_initializer().run()

        # training epochs
        for epoch in range(args.nb_epochs):
            # Compute number of batches   60000/128=468.75=469
            if (epoch+1)%25==0:
                args.learning_rate=args.learning_rate/2    
            nb_batches = int(math.ceil(float(len(X_train)) / args.batch_size))
            # Indices to shuffle training set
            index_shuf = list(range(len(X_train)))
            rng.shuffle(index_shuf)
            prev = time.time()
            if hash_retraining==True:
                nb_batches_hash = int(math.ceil(float(len(X_train)) / batch_size_hash))
			    # mini-batch traning
                for batch in range(nb_batches_hash):

                    optimizer =  tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
                    start, end = batch_indices(batch, len(X_train), args.batch_size)

                    feed_dict = {x: X_train[index_shuf[start:end]],
                                 y: Y_train[index_shuf[start:end]],
							     training:training_BN}
                    hash_gradient_computation_handy_update(x,y,feed_dict,optimizer,loss,model,args.learning_rate)

                    optimizer = optimizer.minimize(loss)
 
                for batch in range(nb_batches):
                    start, end = batch_indices(batch, len(X_train), args.batch_size)
                    feed_dict = {x: X_train[index_shuf[start:end]],
                                 y: Y_train[index_shuf[start:end]],
							     training:training_BN}
                    optimizer.run(feed_dict=feed_dict)
            else:
                for batch in range(nb_batches):
                    start, end = batch_indices(batch, len(X_train), args.batch_size)
                    feed_dict = {x: X_train[index_shuf[start:end]],
                                 y: Y_train[index_shuf[start:end]],
							     training:training_BN}
                    optimizer.run(feed_dict=feed_dict)

				
            cur = time.time()
            
			# evaluation for trained model
            evaluate()
        	
        if save:
            save_path = os.path.join(args.train_dir, args.filename)
            saver = tf.train.Saver()
            saver.save(sess, save_path)



    return True

def model_eval(sess, x, y, training, predictions=None, X_test=None, Y_test=None,
                args=None):

    args = _ArgsWrapper(args or {})
    training_BN=False
    # Define accuracy symbolically

    correct_preds = tf.equal(tf.argmax(y, axis=-1),
                                 tf.argmax(predictions, axis=-1))

    acc_value = tf.reduce_mean(tf.to_float(correct_preds))  

    # Init result var
    accuracy = 0.0

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))

        for batch in range(nb_batches):
            if batch % 100 == 0 and batch > 0:
                _logger.debug("Batch " + str(batch))

            # Must not use the `batch_indices` function here, because it repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * args.batch_size
            end = min(len(X_test), start + args.batch_size)
            cur_batch_size = end - start
            #feed_dict = {x: X_test[start:end], y: Y_test[start:end]}
            feed_dict = {x: X_test[start:end], y: Y_test[start:end],training:training_BN}

            cur_acc = acc_value.eval(feed_dict=feed_dict)

            accuracy += (cur_batch_size * cur_acc)
        # Divide by number of examples to get final value(average)
        accuracy /= len(X_test)

    return accuracy