import numpy as np
from random import shuffle

def sigmoid(x):
    """Sigmoid function implementation"""
    h = np.zeros_like(x)
    
    #############################################################################
    # TODO: Implement sigmoid function.                                         #         
    #############################################################################
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################

    h = 1/(1+np.exp(-x))
    
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
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    
    # For binary classification, there are two classes, the probability for each 
    # classes is p and 1-p. So the W has two coloums. After optimization, 
    # sigmoid(W.t*x) have 2 colomns as well, representing the probability of y=1 
    # and y=0 wchich is [[p],[1-p]].

    # Here, for simplicity, let w be 1 dimensional array, which means C = 1. 
    # Then we only need to calculate the derivative of L wrt w when y=1.
    
    # for every sample Xi in X
    for i in range(X.shape[0]):
        temp_x = X[i,:].reshape(-1,1)
        temp_sig = sigmoid(np.dot(np.transpose(W), temp_x))
        # the loss for this iteration
        loss +=  -1*(y[i]*np.log(temp_sig)+(1-y[i])*np.log(1-temp_sig))
        # since here we only have one class which is y = 1, don't need to calculate
        # for every classes
        dW += (temp_sig-y[i])*temp_x
    
    loss /= X.shape[0] 
    loss += reg*np.linalg.norm(W,ord=2)
    dW += 2*reg*W
        
    
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
    
    sig = sigmoid(np.dot(X,W))

    loss = reg*np.linalg.norm(W,2) - (np.dot(np.transpose(y),np.log(sig))+np.dot(np.transpose(1-y),1-sig))/X.shape[0]
    dW = np.dot(np.transpose(X), sig-y)
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    

    return loss, dW
