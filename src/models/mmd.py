import tensorflow as tf
import numpy as np
import sys 

seen = tf.placeholder(tf.float32,shape=[None,1024])
unseen = tf.placeholder(tf.float32,shape=[None,1024])


""" MMD functions
"""

def _mix_rbf_kernel(X, Y, gammas, wts=None):
    if wts is None:
        wts = [1] * len(gammas)

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0
    for gamma_t, wt in zip(gammas, wts):
        gamma = 1/ gamma_t
        #gamma = 1 / (2 * sigma**2)
        K_XX += wt * tf.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY += wt * tf.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY += wt * tf.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))

    return K_XX, K_XY, K_YY, tf.reduce_sum(wts)


def rbf_mmd2(X, Y, gammas=1, biased=True):
    return mix_rbf_mmd2(X, Y, gammas=[gammas], biased=biased)

def mix_rbf_mmd2(X, Y, gammas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, gammas, wts)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = tf.cast(tf.shape(K_XX)[0], tf.float32)
    n = tf.cast(tf.shape(K_YY)[0], tf.float32)

    if biased:
        mmd2 = (tf.reduce_sum(K_XX) / (m * m)
              + tf.reduce_sum(K_YY) / (n * n)
              - 2 * tf.reduce_sum(K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = tf.trace(K_XX)
            trace_Y = tf.trace(K_YY)

        mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
              + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
              - 2 * tf.reduce_sum(K_XY) / (m * n))

    return mmd2,n
def main():
    seen = tf.placeholder(tf.float32,shape=[None,1024])
    unseen = tf.placeholder(tf.float32,shape=[None,1024])


    mmd,n = rbf_mmd2(seen,unseen)
    mmd,n = mix_rbf_mmd2(seen,unseen,gammas=[10.0,1.0,0.1,0.01,0.001])

    source_numpy = np.load(sys.argv[1])
    target_numpy = np.load(sys.argv[2])
    source_numpy_labels = np.load(sys.argv[3])
    target_numpy_labels = np.load(sys.argv[4])


    with tf.Session() as sess:
        print("Total", sess.run(mmd,feed_dict={seen:source_numpy,unseen:target_numpy}))
        for i in np.unique(source_numpy_labels):
            print(i,sess.run(mmd,feed_dict={seen:source_numpy[source_numpy_labels==i],unseen:target_numpy[target_numpy_labels==i]}))

if __name__ == "__main__":
    main()
