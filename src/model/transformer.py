import numpy as np
from tokenizer.tokenizer import Tokenizer
from config import PROJECT_ROOT, d_k, d_v, d_model, d_ff, h, num_blocks, epsilon, vocab_length
from util import Util
import os

class Transformer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.util = Util()
        self.relpath = os.path.join(PROJECT_ROOT, "data", "weights")
        self.blocks = [TransformerBlock(num, self.relpath) for num in range(num_blocks)]
        self.Wo_path = os.path.join(self.relpath, "Wo.npy")
        if os.path.exists(self.Wo_path):
            self.Wo = np.load(self.Wo_path)
        else:
            self.Wo = np.random.normal(0, 0.02, (vocab_length, d_model))
            np.save(self.Wo_path, self.Wo)

    def forward(self, encoded):
        X = self.tokenizer.embed(encoded) + self.tokenizer.positional(encoded)

        Y = self.blocks[0].forward(X)

        for pos in range(len(self.blocks)-1):
            Y = self.blocks[pos+1].forward(Y)

        logits = Y @ self.Wo
    
        return self.util.softmax(logits)


    def ce_loss(self, probability, actual):
        return -np.log(probability[actual])

class TransformerBlock:
    def __init__(self, num, path):
        block = "block" + str(num)
        self.relpath = os.path.join(path, block)
        self.attentionblock = AttentionBlock(self.relpath)
        self.normone = LayerNorm(1, self.relpath)
        self.ffn = FFN(self.relpath)
        self.normtwo = LayerNorm(2, self.relpath)

    def forward(self, X):
        MHA = self.attentionblock.forward(X)
        AddNorm = self.normone.forward(MHA, X)
        FFN = self.ffn.forward(AddNorm)
        output = self.normtwo.forward(FFN, AddNorm)
        return output

class AttentionBlock:
    def __init__(self, path):
        self.relpath = os.path.join(path, "attention")

        self.heads = [AttentionHead() for i in range(h)]

        self.d_head = int(d_model / h)

        self.Wq_path = os.path.join(self.relpath, "Wq.npy")
        self.Wk_path = os.path.join(self.relpath, "Wk.npy")
        self.Wv_path = os.path.join(self.relpath, "Wv.npy")
        self.Wo_path = os.path.join(self.relpath,  "Wo.npy")
        
        if os.path.exists(self.Wq_path):
            self.Wq = np.load(self.Wq_path)
        else:
            self.Wq = np.random.normal(0, 0.02, (d_model, d_k))
            np.save(self.Wq_path, self.Wq)
        
        if os.path.exists(self.Wk_path):
            self.Wk = np.load(self.Wk_path)
        else:
            self.Wk = np.random.normal(0, 0.02, (d_model, d_k))
            np.save(self.Wk_path, self.Wk)
        
        if os.path.exists(self.Wv_path):
            self.Wv = np.load(self.Wv_path)
        else:
            self.Wv = np.random.normal(0, 0.02, (d_model, d_v))
            np.save(self.Wv_path, self.Wv)
        
        if os.path.exists(self.Wo_path):
            self.Wo = np.load(self.Wo_path)
        else:
            self.Wo = np.random.normal(0, 0.02, (d_model, d_model))
            np.save(self.Wo_path, self.Wo)

    def forward(self, X):
        
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
    
        passes = []
        for i, head in enumerate(self.heads):
            startindex = i * self.d_head
            endindex = (i+1) * self.d_head 
            Qh = Q[:, startindex:endindex]
            Kh = K[:, startindex:endindex]
            Vh = V[:, startindex:endindex]

            passes.append(head.forward(Qh, Kh, Vh))
        
        MHA = np.concatenate(passes, axis=1)

        MHA = MHA @ self.Wo
        return MHA
        #add in shape assertions later down the line; ensures that training is going smoothly

class LayerNorm:
    def __init__(self, num, path):
        numpath = "norm" + str(num)
        self.relpath = os.path.join(path, numpath)
        self.gamma_path = os.path.join(self.relpath, "gamma.npy")
        self.beta_path = os.path.join(self.relpath, "beta.npy")

        if os.path.exists(self.gamma_path):
            self.gamma = np.load(self.gamma_path)
        else:
            self.gamma = np.ones(d_model) 
            np.save(self.gamma_path, self.gamma)
        if os.path.exists(self.beta_path):
            self.beta = np.load(self.beta_path)
        else:
            self.beta = np.zeros(d_model)
            np.save(self.beta_path, self.beta)
    
    def forward(self, pred, X):
        X = pred + X
        lnorm = []
        for xt in X:
            mean = np.mean(xt)
            stddev = np.std(xt)
            norm = self.gamma * (xt - mean) / np.sqrt(stddev ** 2 + epsilon) + self.beta
            lnorm.append(norm)
        return np.array(lnorm)
        
class FFN:
    def __init__(self, path):
        self.relpath = os.path.join(path, "ffn")

        self.util = Util()

        self.W1path = os.path.join(self.relpath, "W1.npy")
        self.b1path = os.path.join(self.relpath, "b1.npy")
        self.W2path = os.path.join(self.relpath, "W2.npy")
        self.b2path = os.path.join(self.relpath, "b2.npy")

        if os.path.exists(self.W1path):
            self.W1 = np.load(self.W1path)
        else:
            self.W1 = np.random.normal(0, 0.02, (d_model, d_ff))
            np.save(self.W1path, self.W1)
        if os.path.exists(self.b1path):
            self.b1 = np.load(self.b1path)
        else:
            self.b1 = np.random.normal(0, 0.02, (d_ff,))
            np.save(self.b1path, self.b1)
        if os.path.exists(self.W2path):
            self.W2 = np.load(self.W2path)
        else:
            self.W2 = np.random.normal(0, 0.02, (d_ff, d_model))
            np.save(self.W2path, self.W2)
        if os.path.exists(self.b2path):
            self.b2 = np.load(self.b2path)
        else:
            self.b2 = np.random.normal(0, 0.02, (d_model,))
            np.save(self.b2path, self.b2)


    def forward(self, X):
        h1 = X @ self.W1 + self.b1
        h2 = self.util.relu(h1)
        h3 = h2 @ self.W2 + self.b2
        return h3

class AttentionHead:
    def __init__(self):
        self.util = Util()
    
    def forward(self, Qh, Kh, Vh):
        scores = Qh @ Kh.T
        d_head = int(d_model / h)
        scores /= np.sqrt(d_head)
        scores += self.util.mask(len(scores), len(scores[0]))

        attentionh = self.util.softmax(scores) @ Vh
        return attentionh    
