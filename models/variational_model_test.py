import numpy as np
import tensorflow as tf
import os
import variational_model as vm


s = tf.Session()
v = vm.VariationalModel(10,2)
f = np.random.rand(20,10)
f2 = np.random.rand(20,2)
f2 = 10*f[:, :2]
# f[0,0] = 1
features = tf.Variable(f)
tf.summary.histogram('features', features)
tf.summary.histogram('mu', v.mu)
tf.summary.histogram('sigma', v.sigma)
tf.summary.scalar('mu_max', tf.reduce_max(v.mu))

feature_norms = tf.reshape(tf.norm(features, axis=1), [-1])
print(f[0])
targets = tf.constant(f2)
l = v.lower_bound(features, targets)
tf.summary.scalar('lower bound', l)
tf.summary.scalar('KL', v.KL())
tf.summary.scalar('data fit', v.model_fit(v.predict(features), targets))
tf.summary.scalar('feature penalty', v.variance(features))


train_writer = tf.summary.FileWriter('.'+ '/train/3',
                                      s.graph)
test_writer = tf.summary.FileWriter('.'+ '/test')


s.run(tf.global_variables_initializer())
preds = s.run(v.predict(features))
print(s.run([v.mu, v.sigma]))
print(s.run(v.lower_bound(features, targets, s)))
print("Model fit: ", s.run(v.model_fit(preds, targets)))
print("KL", s.run(v.KL()))
print("Variance", s.run(v.variance(features)))
v.optimize(features, targets, iterations=5000, s=s)
preds = s.run(v.predict(features))
print("Model fit: ", s.run(v.model_fit(preds, targets)))
print(preds, s.run(targets))
print("KL", s.run(v.KL()))
print("Variance", s.run(v.variance(features)))
print(s.run([v.mu, v.sigma]))
print(s.run(v.lower_bound(features, targets, s)))
print(s.run(features)[0])
print(s.run(targets)[0])

print("done")