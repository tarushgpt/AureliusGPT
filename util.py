import numpy as np

class Util:
    def softmax(self, n, axis):
        max_val = np.max(n, axis, keepdims=True)
        exp = np.exp(n-max_val)
        return exp / np.sum(exp, axis=1, keepdims=True)

    #relu
    def relu(self, n):
        return np.maximum(0, n) #no need for nonvectorized, slow time/space complex double loop nightmare version

    #derivative of relu for backpropogation
    def reluprime(self, n):
        return (n > 0).astype(float) #again, no need for manual reimplementation


    def mask(self, n, m):
        return np.triu(np.full((n,m), -np.inf), k=1)