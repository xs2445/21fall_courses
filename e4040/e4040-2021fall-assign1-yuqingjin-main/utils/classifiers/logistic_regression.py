import numpy as np
from random import shuffle
from tensorflow.keras.utils import to_categorical

def sigmoid(x):
    """Sigmoid function implementation"""
    h = np.zeros_like(x)
    
    #############################################################################
    # TODO: Implement sigmoid function.                                         #         
    #############################################################################
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    
    bottom = 1 + np.exp(-x) 
    h = 1 / bottom

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return h 

def logistic_regression_loss_naive(W, X, y, reg):
    """
      Logistic regression loss function, naive implementation (with loops)
      Use this linear classification method to find optimal decision boundary.

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c can be either 0 or 1.
      - reg: (float) regularization strength. For regularization, we use L2 norm.

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W) # to inform the dW is the same shape as W

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    N = X.shape[0] # num of examples
    C = W.shape[1] # num of classes
    
    Li = 0.0
    z = np.dot(X, W)
    h = sigmoid(z)
    df = h
    
    for i in range(N):
        for j in range(C):
            Li -= y[i] * np.log(h[i][j]) + (1 - y[i]) * np.log(1 - h[i][j])
            df[i][j] = h[i][j] - y[i]
            
    # add regularization
    loss = Li / N + reg * (np.sum(W*W))
    dW = np.dot(X.T, df) / N + 2 * reg * W
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW



def logistic_regression_loss_vectorized(W, X, y, reg):
    """
    Logistic regression loss function, vectorized version.
    Use this linear classification method to find optimal decision boundary.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the logistic regression loss and its gradient using no     # 
    # explicit loops.                                                          #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the       #
    # regularization!                                                          #
    ############################################################################
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    N = X.shape[0] # num of examples
    C = W.shape[1] # num of classes
    
    # converts a class vector (integers) to binary class matrix.
    y_tran = to_categorical(y) 
    
    z = np.dot(X, W)
    h = sigmoid(z)
    
    ones_y = np.ones_like(y)
    ones_ytran = np.ones_like(y_tran)
    ones_h = np.ones_like(h)
    
    loss = - np.dot(y, np.log(h)) - np.dot((ones_y - y), np.log(ones_h - h)) 
    loss = np.sum(loss) / N + reg * (np.sum(W) ** 2)
    
    h1 = h + y_tran - ones_ytran #
    h11 = h1[:, 0]
    h2 = h - y_tran #
    h22 = h2[:, 1]
    df = np.column_stack((h11, h22))
    dW = np.dot(X.T, df) / N + 2 * reg * W    
         
        
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    

    return loss, dW
