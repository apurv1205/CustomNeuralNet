import numpy as np
from numpy import array
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import pickle

def rectifier(x):
    return np.log(1.+1./np.exp(-x))
    #return ( 1./(1. + np.exp(-x)) )

class Config:
    nn_input_dim = 6  # input layer dimensionality
    nn_output_dim = 3  # output layer dimensionality
    # Gradient descent parameters (I picked these by hand)
    epsilon = 0.00001  # learning rate for gradient descent
    reg_lambda = 0.013  # regularization strength


# Helper function to evaluate the total loss on the dataset
def calculate_loss(model, X, y):
    W, b = model['W'], model['b']
    # Forward propagation
    num_layers = len(W)-1
    num_examples = len(X)

    z={}
    a={}
    z[0] = X.dot(W[0]) + b[0]
    a[0] = rectifier(z[0])

    for i in range(num_layers-1) :
        z[i+1] = a[i].dot(W[i+1]) + b[i+1]
        a[i+1] = rectifier(z[i+1])

    z[num_layers] = a[num_layers-1].dot(W[num_layers]) + b[num_layers]

    #print z2
    exp_scores = np.exp(z[num_layers])
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    sum1 = np.sum(np.square(W[0]))
    for i in range(num_layers-1) :
        sum1 += np.sum(np.square(W[i+1]))
    sum1 += np.sum(np.square(W[num_layers]))
    data_loss += Config.reg_lambda / 2 * ( sum1 )
    return 1. / num_examples * data_loss


def predict(model, x):
    W, b = model['W'], model['b']
    # Forward propagation
    num_layers = len(W)-1
    z={}
    a={}
    z[0] = x.dot(W[0]) + b[0]
    a[0] = rectifier(z[0])


    for i in range(num_layers-1) :
        z[i+1] = a[i].dot(W[i+1]) + b[i+1]
        a[i+1] = rectifier(z[i+1])

    z[num_layers] = a[num_layers-1].dot(W[num_layers]) + b[num_layers]

    #print z2
    exp_scores = np.exp(z[num_layers])
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(X, y, nn_hdim, num_layers = 1, num_passes=20000,print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    num_examples = len(X)
    np.random.seed(0)
    W=[]
    b=[]
    W1 = np.random.randn(Config.nn_input_dim, nn_hdim) / np.sqrt(Config.nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_hdim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_hdim))
    W3 = np.random.randn(nn_hdim, Config.nn_output_dim) / np.sqrt(nn_hdim)
    b3 = np.zeros((1, Config.nn_output_dim))

    W.append(W1)
    b.append(b1)
    for i in range(num_layers-1) :
        W.append(W2)
        b.append(b2)
    W.append(W3)
    b.append(b3)

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for j in range(0, num_passes):
        z={}
        a={}

        # Forward propagation
        z[0] = X.dot(W[0]) + b[0]
        a[0] = rectifier(z[0])
        
        for i in range(num_layers-1) :
            z[i+1] = a[i].dot(W[i+1]) + b[i+1]
            a[i+1] = rectifier(z[i+1])

        z[num_layers] = a[num_layers-1].dot(W[num_layers]) + b[num_layers]

        #print z2
        exp_scores = np.exp(z[num_layers])
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


        delta = {}
        dW = {}
        db = {}
        # Backpropagation
        delta[num_layers] = probs
        delta[num_layers][range(num_examples), y] -= 1
        dW[num_layers] = (a[num_layers-1].T).dot(delta[num_layers])
        db[num_layers] = np.sum(delta[num_layers], axis=0, keepdims=True)
        
        for i in reversed(range(num_layers-1) ) :
            delta[i+1] = delta[i+2].dot(W[i+2].T) * (1./(1. + np.exp(-z[i+1])))#(1 - np.power(a[i+1], 2))
            dW[i+1] = np.dot(a[i].T, delta[i+1])
            db[i+1] = np.sum(delta[i+1], axis=0)

        delta[0] = delta[1].dot(W[1].T) * (1./(1. + np.exp(-z[0])))#(1 - np.power(a[0], 2))
        dW[0] = np.dot(X.T, delta[0])
        db[0] = np.sum(delta[0], axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)

        for i in range(num_layers+1) :
            dW[i] += Config.reg_lambda * W[i]
            W[i] += -Config.epsilon * dW[i]
            b[i] += -Config.epsilon * db[i]

        # Assign new parameters to the model
        model = {'W': W, 'b': b}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and j % 1000 == 0:
            print("Loss after iteration %i: %f" % (j, calculate_loss(model, X, y)))

    return model

def main():
    X=[]
    y=[]
    with open("data-train.txt","r") as infile :
        for lines in infile :
            part = lines.split()
            lst=[]
            for j,i in enumerate(part) :
                if (j==0) : continue
                if (j==len(part)-1 ) :
                    y.append(int(i)-1)
                else : lst.append(float(i))
            X.append(lst)

    X1=array(X)
    #X1=array(X)
    X2=array(X[4000:])
    y1=array(y)
    #y1=array(y)
    y2=array(y[4000:])

    print (len(X),len(y))
    #building the model using the entire train dataset
    model = build_model(X1, y1, 15, num_layers=2, print_loss=True)

    Y_pred = predict(model,X2)
    print accuracy_score(y2,Y_pred)
    print classification_report(y2,Y_pred)

    Y_pred1 = predict(model,X1)
    print classification_report(y1,Y_pred1)

    print "Writing model to pickle file"
    with open('model0', 'wb') as outfile:
        pickle.dump(model, outfile)

if __name__ == "__main__":
    main()