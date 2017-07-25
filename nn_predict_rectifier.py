import numpy as np
from numpy import array
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import pickle

def rectifier(x):
    return np.log(1.+1./np.exp(-x))
    #return ( 1./(1. + np.exp(-x)) )

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


def main():
    X=[]
    y=[]
    with open("data-test.txt","r") as infile :
        for lines in infile :
            part = lines.split()
            lst=[]
            for j,i in enumerate(part) :
                if (j==0) : continue
                if (j==len(part)-1 ) :
                    y.append(int(i)-1)
                else : lst.append(float(i))
            X.append(lst)

    print (len(X),len(y))
    
    X=array(X)
    y=array(y)

    print "Reading model from json file"

    with open("model0","rb") as infile:
        model = pickle.load(infile)

    Y_pred = predict(model,X)
    print accuracy_score(y,Y_pred)
    print classification_report(y,Y_pred)

if __name__ == "__main__":
    main()