import os, sys, shutil, argparse
sys.path.append(os.getcwd())
import tensorflow as tf, baselines.common.tf_util as U, numpy as np

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-2, shape=()):

        self._sum = tf.get_variable(
            dtype=tf.float32,
            shape=shape,
            initializer=tf.constant_initializer(0.0),
            name="runningsum", trainable=False)
        self._sumsq = tf.get_variable(
            dtype=tf.float32,
            shape=shape,
            initializer=tf.constant_initializer(epsilon),
            name="runningsumsq", trainable=False)
        self._count = tf.get_variable(
            dtype=tf.float32,
            shape=(),
            initializer=tf.constant_initializer(epsilon),
            name="count", trainable=False)
        self.shape = shape

        self.mean = tf.to_float(self._sum / self._count)
        self.std = tf.sqrt( tf.maximum( tf.to_float(self._sumsq / self._count) - tf.square(self.mean) , 1e-2 ))

        self.newsum = tf.placeholder(shape=self.shape, dtype=tf.float32, name='sum')
        self.newsumsq = tf.placeholder(shape=self.shape, dtype=tf.float32, name='var')
        self.newcount = tf.placeholder(shape=[], dtype=tf.float32, name='count')

        self.incsum = tf.assign_add(self._sum, self.newsum)
        self.incsumsq = tf.assign_add(self._sumsq, self.newsumsq)
        self.inccount = tf.assign_add(self._count, self.newcount)

        self.incfiltparams = tf.group(*[self.incsum, self.incsumsq, self.inccount])

    def update(self, sess, x):
        x = x.astype('float32')
        sess.run(self.incfiltparams, feed_dict={self.newsum:x.sum(axis=0).ravel(), self.newsumsq:np.square(x).sum(axis=0).ravel(), self.newcount:np.array(len(x),dtype='float32')})

@U.in_session
def test_runningmeanstd():
    for (x1, x2, x3) in [
        #(np.random.randn(3), np.random.randn(4), np.random.randn(5)),
        (np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2)),
        ]:

        print (x1.shape)
        print (x1.shape[1:])
        rms = RunningMeanStd(epsilon=0.0, shape=x1.shape[1:])
        U.initialize()

        x = np.concatenate([x1, x2, x3], axis=0)
        ms1 = [x.mean(axis=0), x.std(axis=0)]
        rms.update(x1)
        rms.update(x2)
        rms.update(x3)
        ms2 = U.eval([rms.mean, rms.std])

        assert np.allclose(ms1, ms2)

if __name__ == "__main__":
    test_runningmeanstd()
