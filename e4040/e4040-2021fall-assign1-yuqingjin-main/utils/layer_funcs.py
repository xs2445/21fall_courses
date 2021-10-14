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
    
    N = x.shape[0]
    D = np.prod(x.shape[1:]) #计算所有元素的乘积 D = d_1*d_2*d_3*...*d_k
    x2 = np.reshape(x,(N,D)) #把x的元素按照（N，D）重新排列
    out = np.dot(x2, w) + b
    
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
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x2 = np.reshape(x,(N,D))
    
    #dx = dout/w
    dx = dout.dot(w.T) # (N,M)dot(M,D)——>(N,D)
    dx = np.reshape(dx, x.shape)
    
    #dw = dout/x
    dw = x2.T.dot(dout) #(D,N)dot(N,M)——>(D,M)
    
    #db = dout
    db = np.sum(dout, axis=0) #(N,)
    
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
    
    # pick those xi >=0
    out = np. maximum(x,0)
    
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

    #dout=dx(x>=0),dout=0(x<0)
    dx=dout
    dx[x<0]=0

    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return dx


def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    This adjusts the weights to minimize loss.
    y_prediction = argmax(softmax(x)) # 最大值的index

    Inputs:
    - x: (float) a tensor of shape (N, #classes)
    - y: (int) ground truth label, a array of length N

    Returns:
    - loss: the cross-entropy loss
    - dx: gradients wrt input x
    """
    # Initialize the loss.
    #loss = 0.0
    #dx = np.zeros_like(x)

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
   
    N = x.shape[0]
    D = np.prod(x.shape[1:])#D=d_1*..*d_k
    x2 = np.reshape(x,(N,D))
    epsilon = 1e-15
    
    max_score = np.max(x2,axis = 1,keepdims = True)#get max score for 0~N
    score = x2 - max_score
    
    p = np.exp(score)
    p = p/np.sum(p,axis = 1,keepdims = True)
    # p(0,y1), p(1,y2),...p(N-1,yn)
    loss_matrix = -np.log(p[np.arange(N),y] + epsilon)# p(0,y1), p(1,y2),...p(N-1,yn)
    # np.arrange(起始值（default:0），终止值，步长(defalut:1)) 左闭右开区间  
    loss = np.sum(loss_matrix)/N
    
    # dx为p的数值
    dx = p.copy()
    dx[np.arange(N),y] -=1
    dx = dx/N
    #dx=np.reshape(dx,[N,x.shape[1:]])
    
    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return loss, dx
