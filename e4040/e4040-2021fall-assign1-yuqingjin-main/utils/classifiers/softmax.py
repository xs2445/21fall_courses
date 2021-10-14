import numpy as np
from random import shuffle

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
    N = X.shape[0] # num of examples
    C = W.shape[1] # num of classes
  
    
    for i in range (N):
        
        score = X[i].dot(W) # prediction of each example
        log_C = np.max(score)
        score -= log_C
        
        probability = np.exp(score)/np.sum(np.exp(score))
        # loss is the log of the probability of the correct class
        loss += -np.log(probability[y[i]])# add up loss
        probability[y[i]] -= 1 # calculate p-1 and later we'll put the negative back
        
        for j in range(C):
            dW[:,j] += X[i,:] * probability[j]
        
    loss /= N
    dW /= N
    
    loss += reg * np.sum(W*W)
    dW += 2*reg*W
        
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
    
    N = X.shape[0]
    C = X.dot(W) # NxD * DxC = NxC
    
    score = X.dot(W)
    score -= np.max(score,axis=1,keepdims=True)
    probabilities = np.exp(score) / np.sum(np.exp(score),axis=1,keepdims=True)
    
    correct_class_probabilities = probabilities[range(N),y] # 把yi中0的部分去除掉，range(N)=0,1,2,...N-1;
    # probability[a,b]相当于筛选了原probability矩阵中的部分元素
    loss = np.sum(-np.log(correct_class_probabilities)) / N
    # it is to summarize across classes that aren't classified correctly
    # so now we need to subtract 1 class for each case (a total of N) that are correctly classified
    
    loss += 2 * reg * np.sum(W*W) 

    probabilities[range(N),y] -= 1
    dW = X.T.dot(probabilities) / N + 2 * reg * W
    
    #############################################################################
    #                     END OF YOUR CODE                                      #
    #############################################################################
    

    return loss, dW
