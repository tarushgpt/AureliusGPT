import numpy as np
from src.tokenizer.tokenizer import Tokenizer
from config import PROJECT_ROOT, d_k, d_v, d_model, d_ff, h, num_blocks, epsilon, vocab_length, lr
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
            self.Wo = np.random.normal(0, 0.02, (d_model, vocab_length))
            np.save(self.Wo_path, self.Wo)
        self.lr = lr

    def forward(self, encoded, apply_soft=False):
        self.emb = self.tokenizer.embed(encoded)
        self.pos = self.tokenizer.positional(encoded)
        X = self.emb + self.pos
        self.X = X
        Y = X
        for block in self.blocks:
           Y = block.forward(Y)

        logits = Y @ self.Wo
        if apply_soft:
            return self.util.softmax(logits, -1) #not training
        return logits

    def backward(self, dL_dlogits, input_tokens):
        dL_dWo = self.blocks[-1].output.T @ dL_dlogits
        self.Wo -= self.lr * dL_dWo
        
        dL = dL_dlogits @ self.Wo.T
        for i in range(len(self.blocks)-1, -1, -1):
            dL = self.blocks[i].backward(dL)
        dL_dX = dL
        dL_dE = np.zeros_like(self.tokenizer.E)
        for i in range(len(self.X)):
            dL_dE[input_tokens[i]] += dL_dX[i]
        
        self.tokenizer.E -= lr * dL_dE
    
    def save_weights(self):
        np.save(self.Wo_path, self.Wo)
        for block in self.blocks:
            block.save_weights()
        
    def ce_loss(self, targets, logits):
            logits_max = np.max(logits, axis=-1, keepdims=True)
            logits_shifted = logits - logits_max
            
            log_sum_exp = np.log(np.sum(np.exp(logits_shifted), axis=-1, keepdims=True))
            log_probs = logits_shifted - log_sum_exp
            
            target_log_probs = log_probs[np.arange(len(targets)), targets]
            
            return -np.mean(target_log_probs)
    
    def ce_loss_gradient(self, output_tokens, logits):
        probs = self.util.softmax(logits, axis=-1)

        grad = probs.copy()
        grad[np.arange(len(output_tokens)), output_tokens] -= 1
        
        grad = grad / len(output_tokens)
        return grad

class TransformerBlock:
    def __init__(self, num, path):
        block = "block" + str(num)
        self.relpath = os.path.join(path, block)
        self.attentionblock = AttentionBlock(self.relpath)
        self.normone = LayerNorm(1, self.relpath)
        self.ffn = FFN(self.relpath)
        self.normtwo = LayerNorm(2, self.relpath)

    def forward(self, X):
        self.X = X
        MHA = self.attentionblock.forward(X)
        AddNorm = self.normone.forward(MHA, X)
        FFN = self.ffn.forward(AddNorm)
        output = self.normtwo.forward(FFN, AddNorm)
        self.output = output
        return output

    def backward(self, dL):
        dL_dffn, dL_daddnorm_norm2 = self.normtwo.backward(dL)
        dL_daddnorm_ffn = self.ffn.backward(dL_dffn)
        dL_daddnorm = dL_daddnorm_norm2 + dL_daddnorm_ffn
        dL_dmha, dL_dX_norm1 = self.normone.backward(dL_daddnorm)
        dL_dX_attention = self.attentionblock.backward(dL_dmha)
        
        dL_dX = dL_dX_norm1 + dL_dX_attention
        return dL_dX
    
    def save_weights(self):
        self.attentionblock.save_weights()
        self.normone.save_weights()
        self.ffn.save_weights()
        self.normtwo.save_weights()


class AttentionBlock:
    def __init__(self, path):
        self.relpath = os.path.join(path, "attention")
        os.makedirs(self.relpath, exist_ok=True)

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
        self.X = X
        Q = X @ self.Wq #t * d_k
        K = X @ self.Wk #t * d_k
        V = X @ self.Wv #t * d_v
        
        self.Q = Q
        self.K = K
        self.V = V
    
        passes = []
        for i, head in enumerate(self.heads):
            startindex = i * self.d_head
            endindex = (i+1) * self.d_head 
            Qh = Q[:, startindex:endindex]
            Kh = K[:, startindex:endindex]
            Vh = V[:, startindex:endindex]

            passes.append(head.forward(Qh, Kh, Vh))
        
        self.A = np.concatenate(passes, axis=1) #t * d_model

        MHA = self.A @ self.Wo
        return MHA
        #add in shape assertions later down the line; ensures that training is going smoothly

    def backward(self, dL_dmha):
        dL_dWo = self.A.T @ dL_dmha
        self.Wo -= lr * dL_dWo
        
        dL_dA = dL_dmha @ self.Wo.T
        
        dL_dheads = []
        for i in range(h):
            startindex = i * self.d_head
            endindex = (i+1) * self.d_head
            dL_dheads.append(dL_dA[:, startindex:endindex])
        
        dL_dQ = np.zeros_like(self.Q)
        dL_dK = np.zeros_like(self.K)
        dL_dV = np.zeros_like(self.V)
        
        for i, head in enumerate(self.heads):
            startindex = i * self.d_head
            endindex = (i+1) * self.d_head
            
            Qh = self.Q[:, startindex:endindex]
            Kh = self.K[:, startindex:endindex]
            Vh = self.V[:, startindex:endindex]
            
            dL_dQh, dL_dKh, dL_dVh = head.backward(dL_dheads[i], Qh, Kh, Vh)
            
            dL_dQ[:, startindex:endindex] = dL_dQh
            dL_dK[:, startindex:endindex] = dL_dKh
            dL_dV[:, startindex:endindex] = dL_dVh
        
        dL_dWq = self.X.T @ dL_dQ
        dL_dWk = self.X.T @ dL_dK
        dL_dWv = self.X.T @ dL_dV
        
        self.Wq -= lr * dL_dWq
        self.Wk -= lr * dL_dWk
        self.Wv -= lr * dL_dWv
        
        dL_dX = dL_dQ @ self.Wq.T + dL_dK @ self.Wk.T + dL_dV @ self.Wv.T
        
        return dL_dX
    
    def save_weights(self):
        np.save(self.Wq_path, self.Wq)
        np.save(self.Wk_path, self.Wk)
        np.save(self.Wv_path, self.Wv)
        np.save(self.Wo_path, self.Wo)

class LayerNorm:
    def __init__(self, num, path):
        numpath = "norm" + str(num)
        self.relpath = os.path.join(path, numpath)
        os.makedirs(self.relpath, exist_ok=True)
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
        self.pred = pred
        self.X_in = X
        X = pred + X
        self.X = X
        
        self.means = []
        self.stddevs = []
        self.normalized = []
        
        lnorm = []
        for xt in X:
            mean = np.mean(xt)
            stddev = np.std(xt)
            norm_xt = (xt - mean) / np.sqrt(stddev ** 2 + epsilon)
            
            self.means.append(mean)
            self.stddevs.append(stddev)
            self.normalized.append(norm_xt)
            
            norm = self.gamma * norm_xt + self.beta
            lnorm.append(norm)
        return np.array(lnorm)

    def backward(self, dL):
        dL_dgamma = np.sum(dL * np.array(self.normalized), axis=0)
        dL_dbeta = np.sum(dL, axis=0)
        
        self.gamma -= lr * dL_dgamma
        self.beta -= lr * dL_dbeta
        
        dL_dnorm = dL * self.gamma
        
        dL_dX = np.zeros_like(self.X)
        for i in range(len(self.X)):
            mean = self.means[i]
            stddev = self.stddevs[i]
            var = stddev ** 2
            
            N = d_model
            x_centered = self.X[i] - mean
            
            dL_dvar = np.sum(dL_dnorm[i] * x_centered * -0.5 * (var + epsilon) ** (-1.5))
            dL_dmean = np.sum(dL_dnorm[i] * -1.0 / np.sqrt(var + epsilon)) + dL_dvar * np.sum(-2.0 * x_centered) / N
            
            dL_dX[i] = dL_dnorm[i] / np.sqrt(var + epsilon) + dL_dvar * 2.0 * x_centered / N + dL_dmean / N
        
        dL_dpred = dL_dX
        dL_dX_in = dL_dX
        
        return dL_dpred, dL_dX_in
    
    def save_weights(self):
        np.save(self.gamma_path, self.gamma)
        np.save(self.beta_path, self.beta)
        
        
class FFN:
    def __init__(self, path):
        self.relpath = os.path.join(path, "ffn")
        os.makedirs(self.relpath, exist_ok=True)

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
        self.X = X
        h1 = X @ self.W1 + self.b1
        self.h1 = h1
        h2 = self.util.relu(h1)
        self.h2 = h2
        h3 = h2 @ self.W2 + self.b2
        return h3
    
    def backward(self, dL_dh3):
        dL_dW2 = self.h2.T @ dL_dh3
        dL_db2 = np.sum(dL_dh3, axis=0)
        
        self.W2 -= lr * dL_dW2
        self.b2 -= lr * dL_db2
        
        dL_dh2 = dL_dh3 @ self.W2.T
        
        dL_dh1 = dL_dh2 * self.util.reluprime(self.h1)
        
        dL_dW1 = self.X.T @ dL_dh1
        dL_db1 = np.sum(dL_dh1, axis=0)
        
        self.W1 -= lr * dL_dW1
        self.b1 -= lr * dL_db1
        
        dL_dX = dL_dh1 @ self.W1.T
        
        return dL_dX
    
    def save_weights(self):
        np.save(self.W1path, self.W1)
        np.save(self.b1path, self.b1)
        np.save(self.W2path, self.W2)
        np.save(self.b2path, self.b2)
class AttentionHead:
    def __init__(self):
        self.util = Util()
    
    def forward(self, Qh, Kh, Vh):
        scores = Qh @ Kh.T
        d_head = int(d_model / h)
        scores /= np.sqrt(d_head)
        self.scores_pre_mask = scores.copy()
        scores += self.util.mask(len(scores), len(scores[0]))
        self.scores = scores
        
        self.attention_weights = self.util.softmax(scores, -1)
        attentionh = self.attention_weights @ Vh
        return attentionh
    
    def backward(self, dL_dattentionh, Qh, Kh, Vh):
        dL_dVh = self.attention_weights.T @ dL_dattentionh
        
        dL_dweights = dL_dattentionh @ Vh.T
        
        dL_dscores = np.zeros_like(self.scores)
        for i in range(len(self.attention_weights)):
            s = self.attention_weights[i:i+1, :]
            ds = dL_dweights[i:i+1, :]
            dL_dscores[i] = (ds - np.sum(ds * s, axis=1, keepdims=True)) * s
        
        d_head = int(d_model / h)
        dL_dscores /= np.sqrt(d_head)
        
        dL_dQh = dL_dscores @ Kh
        dL_dKh = dL_dscores.T @ Qh
        
        return dL_dQh, dL_dKh, dL_dVh    
