import numpy as np
import tensorflow as tf
import os
import model_comparison.models.transformations as iv
import model_comparison.models.networks as nets
import tensorflow.contrib.slim as slim
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util
from tensorflow.keras import layers
import gin
import matplotlib.pyplot as plt
float_type=np.float64

@gin.configurable
class FAModel():
    def __init__(self, cnn_fn, X, Y, Xtest, Ytest,
                averaging=None, optimizer=tf.train.AdamOptimizer,
                batch_size=32, learning_rate = 0.0001, name="model",
                loss_fn=tf.nn.softmax_cross_entropy_with_logits_v2, ckpt_dir = None):
        self.cnn_fn = cnn_fn
        self.output_dim = Y.shape[1]
        print("Output dim: ", self.output_dim)
        self.averaging=averaging
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.name = name
        self.optimizer = optimizer(learning_rate=learning_rate)
        self.image_shape = X.shape[1:4]
        self.network = self.build_network(self.averaging)
        self.X = X
        self.Y = Y
        self.Xtest = Xtest
        self.Ytest = Ytest
        #self.loss_fn = loss_fn
        self.ckpt_dir = ckpt_dir

    def loss_fn(self, predictions, targets):
        return tf.reduce_sum((tf.cast(tf.reshape(targets,
                        [-1, self.output_dim]), float_type) - tf.cast(tf.reshape(predictions,
                        [-1, self.output_dim]), float_type))**2)
    
    def build_network(self, averaging):
        
        def f(x):
            if averaging=='p4':
                n_samples = 4
                d = lambda x, i : iv.c4_rotate(x, i)
                df = lambda x : tf.map_fn(lambda i : d(x, i), tf.constant([1,2,3,4]), dtype=tf.float32)
                transformed_x = x #tf.reshape(tf.map_fn(df, x), [-1]+list(self.image_shape))
                print(transformed_x.shape, "Transformed shape")
            elif averaging==None:
                print("NO averaging")
                n_samples = 1
                transformed_x = x #tf.Variable(x, trainable=False)
                # print(x.shape, transformed_x.shape.ndims)
            else:
                n_samples = 10
                d = lambda x, i : averaging(x)
                df = lambda x : tf.map_fn(lambda i : d(x, i), tf.constant(np.ones((10))), dtype=tf.float32)
                transformed_x = tf.reshape(tf.map_fn(df, x), [-1] + list(self.image_shape))
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                y = self.cnn_fn(transformed_x, self.output_dim)
                # if averaging == None:
                #    return y
                print("Done cnn")
                print(y.shape)
                y = tf.reduce_mean(tf.reshape(y, [32, n_samples, self.output_dim]), axis=1)
                print("Y shape: ", y.shape)
            return y

        return f

    def predict(self, x):
        return self.network(x)

    def optimize(self, steps=1000):
        num_indices = self.Y.shape[0]
        train_indices = tf.random_uniform(
        [self.batch_size], 0, num_indices, tf.int64)
        num_test_indices = self.Ytest.shape[0]
        test_indices = tf.random_uniform(
        [self.batch_size], 0, num_test_indices, tf.int64)
        train_indices2 = tf.map_fn(lambda x : [0,1,2,3] + x - (x % 4), train_indices)
        train_indices2 = tf.reshape(train_indices2, [-1])
        print(train_indices2.shape, train_indices.shape)
        train_data_node = tf.gather(self.X, train_indices2)
        
        train_labels_node = tf.gather(self.Y, train_indices)
        test_indices2 = tf.map_fn(lambda x : [0,1,2,3] + x - (x % 4), test_indices)
        test_indices2 = tf.reshape(test_indices2, [-1])
        test_data_node = tf.gather(self.Xtest, test_indices2)
        test_labels_node = tf.gather(self.Ytest, test_indices)

        train_preds = self.predict(train_data_node)
        test_preds = tf.stop_gradient(self.predict(test_data_node))
        loss_op = self.loss_fn( train_labels_node, train_preds)
        test_loss_op = self.loss_fn(test_labels_node, test_preds)

        # Add summaries
        tf.summary.scalar('Train loss', loss_op)
        tf.summary.scalar('Test loss', test_loss_op)
        summary_op = tf.summary.merge_all()
        train_op = self.optimizer.minimize(loss_op)
        train_op = slim.learning.create_train_op(loss_op, self.optimizer)
        print("Starting training for steps: %s" % steps)
        builder = tf.profiler.ProfileOptionBuilder
        opts = builder(builder.time_and_memory()).order_by('micros').build()
        opts2 = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        # with tf.contrib.tfprof.ProfileContext('/tmp/train_dir',
        #                              trace_steps=range(10, 20),
        #                              dump_steps=[20]) as pctx:
        #        # Run online profiling with 'op' view and 'opts' options at step 15, 18, 20.
        #        pctx.add_auto_profiling('op', opts, [15, 18, 20])
        #        # Run online profiling with 'scope' view and 'opts2' options at step 20.
        #        pctx.add_auto_profiling('scope', opts2, [20])
        print("Finished setting up profiler.")
        final_loss = slim.learning.train(
                  train_op,
                  logdir=self.ckpt_dir,
                  number_of_steps=steps,
                  save_summaries_secs=1,
                  log_every_n_steps=500)
                #  print("done")
        print("Finished training. Last batch loss:", final_loss)
        print("Checkpoint saved in %s" % self.ckpt_dir)
        # s = tf.Session()
        # s.run(tf.global_variables_initializer())
        # print(tf.trainable_variables())
        # for i in range(50000):
        #     to, loss, ps, ls = s.run([train_op, loss_op, test_preds, test_labels_node])
        #     if i % 100 == 0:
        #         plt.imshow(s.run(train_data_node)[0, :, :, 0])
        #         plt.show()
        #         print(loss)
        #         print(np.argmax(ps, axis=1), np.argmax(ls, axis=1))

        



@gin.configurable
class VariationalModel():

# TODO: update KL to accommodate prior mu, sigma
    def __init__(self, num_features, output_dim, prior_mu=None, prior_sigma=None, batch_size=32, training_size=60000):
        self.num_features = num_features
    # Note -- num_features will be second axis in feature matrix
        self.output_dim = output_dim
        self.optimizer = tf.train.AdamOptimizer(learning_rate =0.01)
        self.KL_scale = batch_size/training_size
        if prior_mu:

            self.mu = prior_mu
            self.prior_mu = prior_mu
        else:
            self.prior_mu = np.zeros((self.num_features, self.output_dim))
            self.mu = tf.Variable(self.prior_mu, name="mu")

        # Note, sigma is parametrized by sqrt(sigma) (we square to enforce positivity)
        if prior_sigma:
            self.sigma = prior_sigma
            self.prior_sigma = prior_sigma
        else:
            # Assume diagonal covariance that's constant across output dim
            self.prior_sigma = np.ones((self.num_features, 1))
            self.sigma = tf.Variable(0.001*np.ones((self.num_features, 1)), name="sigma")

    # KL from mean zero var 1 gaussian
    def KL(self):
        return 0.5 * (tf.reduce_sum(
                    self.sigma**2) + tf.reduce_sum(self.mu**2) - self.num_features - tf.reduce_sum(
                        tf.log(self.sigma**2))) 

    def predict(self, features):
        return features # tf.matmul(features, 1.0)#self.mu)

    def model_fit(self, predictions, targets):
        return tf.reduce_sum((tf.cast(tf.reshape(targets,
                        [-1, self.output_dim]), float_type) - tf.cast(tf.reshape(predictions,
                        [-1, self.output_dim]), float_type))**2)

    def variance(self, features):
        feature_norms = tf.reshape(tf.norm(features, axis=1), [-1, 1])
        return self.output_dim * tf.reduce_sum(tf.matmul(self.sigma**(2), feature_norms,  transpose_b=True))

    def lower_bound(self, features, targets):
        # Assume prior is normal with mean zero var one
        KL = self.KL_scale * self.KL()
        predictions = self.predict(features)

        model_fit = self.model_fit(predictions, targets)

        feature_norms = tf.reshape(tf.norm(features, axis=1), [-1, 1])

        variance = self.output_dim * tf.reduce_sum(tf.matmul(self.sigma**(2), feature_norms,  transpose_b=True))

        return tf.cast(KL, tf.float64) + tf.cast(model_fit, tf.float64) + tf.cast(variance, tf.float64)

    def optimizeLB(self, features, targets, iterations=100, s=tf.Session(), writer=None):
        train_op = self.optimizer.minimize(self.lower_bound(features, targets, s))
        s.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        for i in range(iterations):
            summary, _ = s.run([merged, train_op])
            if writer:
                writer.add_summary(summary, i)


@gin.configurable
class DeepVM(VariationalModel):
    def __init__(self, num_features, output_dim, prior_mu=None, prior_sigma=None,
        averaging=None, name='deepVM', num_samples=10, ckpt_dir=None, input_size=None, batch_size=None,
        train_features=True, equivariant=None):
        self.name = name
        self.num_features = num_features
        self.ckpt_dir = ckpt_dir
        self.group = equivariant
        if not equivariant:
            self.cnn_fn = nets.cnn_fn
        else:
            self.cnn_fn = nets.eq_cnn_fn
        if not averaging:
            self.feature_map = lambda x : tf.cast(self.cnn_fn(tf.cast(x, tf.float32), self.num_features,
                trainable=train_features, group=self.group), float_type)
        elif averaging =='p4': 
            d = lambda x, i : iv.c4_rotate(x, i)
            df = lambda x : tf.cast(tf.map_fn(lambda i : self.cnn_fn(tf.cast(d(x, i), tf.float32),
                self.num_features, trainable=train_features, group=self.group), tf.constant([1,2,3,4]), dtype=tf.float32),  float_type)
            self.feature_map = lambda x : tf.cast(tf.reduce_sum(df(x), axis=0), float_type)
        elif averaging == 'uniform':
            self.theta_lower = tf.Variable(-np.pi/30, name="thetaLower")
            self.theta_upper = tf.Variable(np.pi/30, name="thetaUpper")
            tf.summary.scalar('theta_upper', self.theta_upper)
            tf.summary.scalar('theta_lower', self.theta_lower)    
            e = lambda x : iv.rotate(x, 20*self.theta_lower, 20*self.theta_upper)
            f = lambda x : tf.cast(self.cnn_fn(tf.cast(e(x), tf.float32), self.num_features,
                trainable=train_features, group=self.group), float_type) 
            self.feature_map = lambda x : tf.reduce_sum(
                tf.map_fn(lambda y : f(x), tf.ones(num_samples, dtype=tf.float64)),axis=0)/num_samples
        with tf.variable_scope(self.name, tf.AUTO_REUSE):
            # Build the feature map
            # inputs = tf.placeholder(float_type)
            # features = self.feature_map(inputs)
            # Initialize variational parameters
            super(DeepVM, self).__init__(num_features, output_dim, prior_mu, prior_sigma)
        tf.summary.scalar('sigma', tf.norm(self.sigma))    
    # def predict(self, inputs):
    #     with tf.variable_scope(self.name, tf.AUTO_REUSE):
    #         return tf.matmul(self.feature_map(inputs), self.mu)



    def optimizeDataFit(self, input_data, targets, batch_size=32, steps=1500 , test_data=None, test_targets=None, loss=None):
        num_indices = targets.shape[0]
        if not loss:
            loss = self.model_fit
        train_indices = tf.random_uniform(
        [batch_size], 0, num_indices, tf.int64)
        train_data_node = tf.gather(input_data, train_indices)
        train_labels_node = tf.gather(targets, train_indices)
        # Variable sharing is on, so make sure every instantiation of feature map is under
        # the name of the variational model
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE): 
            features = self.feature_map(train_data_node)
        predictions = self.predict(features)
        loss_op = tf.reduce_mean(loss(predictions, train_labels_node))
        if test_data is not None:
            num_test_indices = test_targets.shape[0]
            test_indices = tf.random_uniform([batch_size], 0, num_test_indices, tf.int64)
            test_data_node = tf.gather(test_data, test_indices)
            test_labels_node = tf.gather(test_targets, test_indices)
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                test_features = self.feature_map(test_data_node)
            test_preds = self.predict(test_features)
            test_loss = tf.reduce_mean(loss(test_preds, test_labels_node))
            tf.summary.scalar('test perf', test_loss)
        else:
            test_loss = 0
        #
        aux_loss =tf.cast(tf.stop_gradient(test_loss), tf.float64) +  tf.cast(self.variance(tf.stop_gradient(features)), tf.float64)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        # train_op = slim.learning.create_train_op(loss_op + aux_loss, optimizer) 
        kl = self.KL()
        train_op = slim.learning.create_train_op(loss_op, optimizer) # + aux_loss + self.KL_scale * kl, optimizer) 
        datafit = tf.reduce_mean(loss(predictions, train_labels_node))
        var = self.variance(features)
        elbo = tf.reduce_mean(self.lower_bound(features, train_labels_node))
        tf.summary.scalar('KL', kl)
        tf.summary.scalar('data fit', datafit)
        tf.summary.scalar('variance', var)
        tf.summary.scalar('elbo', elbo)
        summary_op = tf.summary.merge_all()
        final_loss = slim.learning.train(
          train_op,
          logdir=self.ckpt_dir,
          number_of_steps=steps,
          save_summaries_secs=1,
          log_every_n_steps=100)
        print("Finished training. Last batch loss:", final_loss)
        print("Checkpoint saved in %s" % self.ckpt_dir)

    def optimizeLB(self, input_data, targets, batch_size=32, steps=1500, test_data=None, test_targets=None):
        num_indices = targets.shape[0]
        train_indices = tf.random_uniform(
        [batch_size], 0, num_indices, tf.int64)
        train_data_node = tf.gather(input_data, train_indices)
        train_labels_node = tf.gather(targets, train_indices)
        if test_data is not None:
            num_test_indices = test_targets.shape[0]
            test_indices = tf.random_uniform([batch_size], 0, num_test_indices, tf.int64)
            test_data_node = tf.gather(test_data, test_indices)
            test_labels_node = tf.gather(test_targets, test_indices)
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                test_features = self.feature_map(test_data_node)
            test_preds = self.predict(test_features)
            test_loss = tf.reduce_mean(self.model_fit(test_preds, test_labels_node))
            tf.summary.scalar('test perf', test_loss)
        else:
            test_loss = 0.0
        # Variable sharing is on, so make sure every instantiation of feature map is under
        # the name of the variational model
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE): 
            features = self.feature_map(train_data_node)
        predictions = self.predict(features)
        loss_op = tf.reduce_mean(self.lower_bound(features, train_labels_node)) + 0.000000001 *  tf.stop_gradient(test_loss)
        # loss_op = test_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train_op = slim.learning.create_train_op(loss_op, optimizer) 
        
        # Save summaries of the different components of the loss function
        kl = self.KL()
        datafit = tf.reduce_mean(self.model_fit(predictions, train_labels_node))
        var = self.variance(features)
        
        tf.summary.scalar('KL', kl)
        tf.summary.scalar('elbo', loss_op)
        tf.summary.scalar('data fit', datafit)
        tf.summary.scalar('variance', var)
        tf.summary.scalar('elbo', loss_op)
        summary_op = tf.summary.merge_all()
        final_loss = slim.learning.train(
          train_op,
          logdir=self.ckpt_dir,
          number_of_steps=steps,
          save_summaries_secs=1,
          log_every_n_steps=500)
        print("Finished training. Last batch loss:", final_loss)
        print("Checkpoint saved in %s" % self.ckpt_dir)




