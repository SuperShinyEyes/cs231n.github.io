from builtins import range
import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros_like(W) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss_contribution = np.zeros([1, num_classes])
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            # Hinge loss, https://en.wikipedia.org/wiki/Hinge_loss
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW = dW / num_train  + 2 * reg * W 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    # D X C
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_classes = W.shape[1]
    num_train = X.shape[0]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # scores: N x C
    scores = X.dot(W)
    # N x 1
    correct_class_score = scores[np.arange(len(y)), y].reshape(-1, 1)
    # N x C
    correct_class_mask = np.full(scores.shape, True)
    correct_class_mask[np.arange(len(y)), y] = False
    # N x C
    margin = scores - correct_class_score + 1
    negative_mask = margin > 0
    loss = margin[correct_class_mask & negative_mask]
    loss = np.sum(loss) / num_train + reg * np.sum(W * W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # *****Semi-vectorized*****
    # for i in range(num_train):
    #     row_correct_class_mask = np.full(num_classes, True)
    #     row_correct_class_mask[y[i]] = False
    #     dW[:, row_correct_class_mask & negative_mask[i]] += X[i].reshape(-1, 1)
    #     dW[:, ~row_correct_class_mask] -= X[i].reshape(-1, 1) * sum(negative_mask[i] & row_correct_class_mask)

    # *****Vectorized*****
    N_C = np.zeros((num_train, num_classes))
    N_C[correct_class_mask & negative_mask] = 1

    N_C2 = np.zeros((num_train, num_classes))
    N_C2[~correct_class_mask] = 1

    # import ipdb; ipdb.set_trace()
    dW = X.T @ N_C \
        - ( X * np.sum(negative_mask & correct_class_mask, axis=1 )[:,None] ).T @ N_C2 
    
    dW = dW / num_train  + 2 * reg * W 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
