from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine transformation function.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: a numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: a numpy array of weights, of shape (D, M)
    - b: a numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    """
    ############################################################################
    # TODO: Implement the affine forward pass. Store the result in 'out'. You  #
    # will need to reshape the input into rows.                                #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################

    # print(b.shape)
    out = x.dot(w)+b.reshape(-1)
    # print(out.shape)

    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return out


def affine_backward(dout, x, w, b):
    """
    Computes the backward pass of an affine transformation function.

    Inputs:
    - dout: upstream derivative, of shape (N, M)
    - x: input data, of shape (N, d_1, ... d_k)
    - w: weights, of shape (D, M)
    - b: bias, of shape (M,)

    Returns a tuple of:
    - dx: gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: gradient with respect to w, of shape (D, M)
    - db: gradient with respect to b, of shape (M,)
    """
    ############################################################################
    # TODO: Implement the affine backward pass.                                #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################


    # df_dx = np.tile(np.sum(w,axis=0),(x.shape[0],1))
    # df_dw = np.tile(np.sum(x,axis=1).T,(1,w.shape[1]))
    # df_db = np.ones_like(b)

    dx = dout.dot(w.T)
    dw = x.T.dot(dout)
    db = dout.sum(axis=1)

    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for rectified linear units (ReLUs) activation function.

    Input:
    - x: inputs, of any shape

    Returns a tuple of:
    - out: output, of the same shape as x
    """
    ############################################################################
    # TODO: Implement the ReLU forward pass.                                   #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################
    
    # for xi in x, xi = 0 if xi < 0
    out = np.asarray(x)
    out[out<0]=0
    # print(out)
    
    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return out


def relu_backward(dout, x):
    """
    Computes the backward pass for rectified linear units (ReLUs) activation function.

    Input:
    - dout: upstream derivatives, of any shape

    Returns:
    - dx: gradient with respect to x
    """
    ############################################################################
    # TODO: Implement the ReLU backward pass.                                  #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################
    
    drelu_dx = np.asarray(x)
    drelu_dx[drelu_dx>=0]=1
    drelu_dx[drelu_dx<0]=0
    # print(drelu_dx.shape)
    # print(dout.shape)
    dx = dout*drelu_dx
    
    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return dx


def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    This adjusts the weights to minimize loss.
    y_prediction = argmax(softmax(x))

    Inputs:
    - x: (float) a tensor of shape (N, #classes)
    - y: (int) ground truth label, a array of length N

    Returns:
    - loss: the cross-entropy loss
    - dx: gradients wrt input x
    """
    # Initialize the loss.
    loss = 0.0
    dx = np.zeros_like(x)

    # When calculating the cross entropy,
    # you may meet another problem about numerical stability, log(0)
    # to avoid this, you can add a small number to it, log(0+epsilon)
    epsilon = 1e-15


    ############################################################################
    # TODO: You can use the previous softmax loss function here.               #
    # Hint: Be careful on overflow problem                                     #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################

    softmat = softmax(x)
    
    # one-hot encoded y 
    y_onehot = np.eye(x.shape[1])[y]
    
    loss = -np.sum(y_onehot*np.log(softmat))/x.shape[0] 
    
    dx = (softmat-y_onehot)/x.shape[0]
    
    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return loss, dx

def softmax(f):
    # to make the result numerical stale
    f_temp = f - np.max(f,axis=1).reshape(-1,1)
    return np.exp(f_temp)/np.sum(np.exp(f_temp),axis=1).reshape(-1,1)
