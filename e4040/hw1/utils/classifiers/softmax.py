import numpy as np
from random import shuffle

from numpy.ma.extras import compress_cols


def softmax(f):
    # to make the result numerical stale
    f -= np.max(f,axis=0)
    return np.exp(f)/np.sum(np.exp(f),axis=0)


def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)
      This adjusts the weights to minimize loss.

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength. For regularization, we use L2 norm.

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                                    #
    #############################################################################

    dW = np.zeros_like(W)
    # softmax function for all xi. shape: C*N
    softmat = softmax(np.transpose(np.dot(X,W)))
    cross_entropy = np.zeros((W.shape[1],1))
    
    # for each sample
    for i in range(X.shape[0]):
          # xi is the i-th sample in X, has shape D*1
          xi = X[i,:].reshape(-1,1)
          # softmax(xi) C*1
          softmax_xi = softmat[:,i].reshape(-1,1)
          # one-hot y with shape 1*C
          y_onehot = np.eye(W.shape[1])[y[i]]
          # here we only calculate the Li, which is cross entropy.
          cross_entropy[y[i]] -= np.log(y_onehot.dot(softmax_xi))
          # D*1 . 1*C = D*C
          dW -= xi.dot(y_onehot-softmax_xi.T)

    dW = dW/X.shape[0] + 2*reg*W
    loss = np.sum(cross_entropy)/X.shape[0] + reg*np.sum(np.linalg.norm(W,ord=2,axis=0))
    #############################################################################
    #                     END OF YOUR CODE                                      #
    #############################################################################


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    This adjusts the weights to minimize loss.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                                    #
    #############################################################################
    
    # softmax function for all xi. shape: C*N
    softmat = softmax(np.dot(X,W).T)
    
    # one-hot y with shape N*C
    y_onehot = np.eye(W.shape[1])[y]

    cross_entropy = np.zeros((W.shape[1],1))

    cross_entropy = -np.sum(y_onehot*np.log(softmat).T,axis=0)

    loss = np.sum(cross_entropy)/X.shape[0] + reg*np.sum(np.linalg.norm(W,ord=2,axis=0))

    # X has shape N*D   D*N . N*C = D*C
    dW = -(X.T.dot(y_onehot-softmat.T)/X.shape[0]) + 2*reg*W



    #############################################################################
    #                     END OF YOUR CODE                                      #
    #############################################################################
    

    return loss, dW
