from collections import Counter
import numpy as np
import re, json, os
from config import d_model, max_seq_length, vocab_length, PROJECT_ROOT, min_freq, n, greek_transliteration
from scripts.preprocess import Preprocessor

class Tokenizer:
    def __init__(self):
        self.preprocess = Preprocessor()
        self.protected_tokens = set(greek_transliteration)
        self.meditations = self.preprocess.meditations

        self.token_to_id_path = os.path.join(PROJECT_ROOT, "data", "vocabulary", "token_to_id.json")
        self.id_to_token_path = os.path.join(PROJECT_ROOT, "data", "vocabulary", "id_to_token.json")
        self.rules_path = os.path.join(PROJECT_ROOT, "data", "vocabulary", "rules.json")
        self.vocab_path = os.path.join(PROJECT_ROOT, "data", "vocabulary", "vocab.json")
        self.embeddings_path = os.path.join(PROJECT_ROOT, "data", "vocabulary", "embeddings.npy")

        
        #identified the OS hell. this is the biggest problem in the current code
        if not (os.path.exists(self.token_to_id_path) and os.path.exists(self.id_to_token_path) and os.path.exists(self.rules_path) and os.path.exists(self.vocab_path) and os.path.exists(self.embeddings_path)):
            self.tokenize_train()
        with open(self.token_to_id_path, "r") as f:
            self.token_to_id = json.load(f)
        with open(self.id_to_token_path, "r") as f:
            self.id_to_token = json.load(f)
        with open(self.rules_path, "r") as f:
            self.rules = json.load(f)
        with open(self.vocab_path, "r") as f:
            self.vocab = json.load(f)
        self.E = np.load(self.embeddings_path)
            
    def words(self, text):
        newline = r"[\n]"        
        punctuation = r"([.,;:!?\"'])"
        words = re.sub(newline, " ", text)
        words = re.sub(punctuation, r" \1 ", words)
        words = words.split()
        return words

    def characterize(self, text, protected = None):
        words = self.words(text)

        characters = []

        for word in words: 
            if protected and word.lower() not in protected:
                characters.append("_")
                characters.extend([char for char in word])
            elif protected and word.lower() in protected:
                characters.append("_")
                characters.append(word.lower())
            else:
                characters.append("_")
                characters.extend([char for char in word])

        return characters 
    
    def bpe_train(self):
        characters = self.characterize(self.meditations, self.protected_tokens)
        vocab = sorted(list(set(characters)))
        vocab.append("<UNK>")
        vocab.append("<EOS>")
        vocab.append("<BEGIN>")
        
        self.rules = []

        while len(vocab) < vocab_length:
            
            a = 0
            pairs = []

            while a < len(characters) - 1:
                if not "_" in characters[a+1]:
                    pair = (characters[a], characters[a+1])
                    pairs.append(pair)
                a += 1

        
            rules = Counter(pairs).most_common()
            
            #rule selection

            rule = None

            for i in rules:
                
                #vocab based check
                rulestr = ""
                for char in i[0]: rulestr += char

                #gain based check
                frequency = i[1]

                if rulestr in vocab or frequency < min_freq:
                    continue
                else:
                    rule = i
                    vocab.append(rulestr)
                    break

            if rule is None:
                break

            i = 0
            characters2 = []
            while i < len(characters):
                if rule and i < len(characters) - 1 and characters[i] == rule[0][0] and characters[i+1] == rule[0][1]:
                    
                    merged = rule[0][0] + rule[0][1]
                    characters2.append(merged)
                    i += 2
                else:    
                    characters2.append(characters[i])
                    i += 1


            characters = characters2

            if rule: self.rules.append((rule[0])) 

        self.vocab = vocab

    def tokenize_train(self):
        self.bpe_train()
        token_to_id = {}
        id_to_token = {}
        for i in range(len(self.vocab)):
            token_to_id[self.vocab[i]] =  i
            id_to_token[i] = self.vocab[i]

        with open(self.token_to_id_path, 'w') as file:
            json.dump(token_to_id, file) 
        self.token_to_id = token_to_id

        with open(self.id_to_token_path, 'w') as file:
            json.dump(id_to_token, file)
        self.id_to_token = id_to_token

        if os.path.exists(self.embeddings_path):
            self.E = np.load(self.embeddings_path)
        else:
            self.E = np.random.normal(0, 0.02, (len(self.vocab), d_model))
            np.save(self.embeddings_path, self.E)
        
        with open(self.rules_path, "w") as f:
            json.dump(self.rules, f)

        with open(self.vocab_path, "w") as f:
            json.dump(self.vocab, f)

    def tokenize(self, input_str):
        tokens = self.characterize(input_str, self.protected_tokens)

        #I will allow this to throw an exception manually, meaning it is out of sequence
        
        for rule in self.rules:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == rule[0] and tokens[i+1] == rule[1]:
                    merged = tokens[i] + tokens[i+1]
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens
        return tokens

    def encode(self, input_str):
        tokens = self.tokenize(input_str)
        
        encoded = []

        for token in tokens:
            try:
                encoded.append(self.token_to_id[token])
            except Exception:
                encoded.append(self.token_to_id["<UNK>"])
                
        return encoded

    def decode(self, input_nums):
        return [self.id_to_token[str(i)] for i in input_nums]
    
    def embed(self, encoded):
        X = np.array([self.E[i] for i in encoded])
        return X

    def positional(self, encoded):
        positionals = np.zeros((len(encoded), d_model))
        for pos in range(len(encoded)):
            for i in range(0, d_model, 2):
                denominator = n ** (i / d_model)
                positionals[pos, i] = np.sin(pos / denominator)
                if i + 1 < d_model:
                    positionals[pos, i + 1] = np.cos(pos / denominator)
        return positionals

    def tpw_ratio(self):
        words = self.words(self.meditations)
        tokens = self.tokenize(self.meditations)

        if len(words) == 0:
            return 0
        
        tokens = [t for t in tokens if t != "_"]

        return len(tokens) / len(words)
        
    def tokenize_and_chunk_corpus(self):
        chunked = []
        encoded = self.encode(self.meditations)

        encoded = [self.token_to_id["<BEGIN>"]] + encoded

        remainder = len(encoded) % max_seq_length
        for i in range(0, len(encoded), max_seq_length):
            chunk = [0] + encoded[i:i+max_seq_length-1]
            chunked.append(chunk)
        if remainder != 0:
            chunked.append(encoded[len(encoded)-max_seq_length : len(encoded)])
        return chunked
    
    def save_embeddings(self):
        np.save(self.embeddings_path, self.E)