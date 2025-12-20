from collections import Counter
import numpy as np
import re, json, os
from config import d_model, max_seq_length, vocab_length, PROJECT_ROOT


class Tokenizer:
    def __init__(self):
        self.protected_tokens = set(["emphron", "sumphron", "huperphron", "nomos", "nemon", "axiopistos", "theophoretos", "oikonomian", "touto", "eferen", "auto", "eumoiros", "eudaimonia", "eupatridai", "kathoti", "katorthoseos", "kosmos", "melos", "meros", "pareilephamen", "symbainein", "tasis", "agathos", "aktines", "ekteinesthai", "daimon", "katorthoseis", "auto"])
        self.PROJECT_ROOT = PROJECT_ROOT
        with open(os.path.join(self.PROJECT_ROOT, "data/processed/meditations.txt"), "r") as f:
            self.meditations = f.read()
        self.vocab_length = vocab_length
        self.d_model = d_model

        self.token_to_id_path = os.path.join(self.PROJECT_ROOT, "data/vocabulary/token_to_id.npy")
        self.id_to_token_path = os.path.join(self.PROJECT_ROOT, "data/vocabulary/id_to_token.npy")
        self.rules_path = os.path.join(self.PROJECT_ROOT, "data/vocabulary/rules.json")

        if os.path.exists(self.token_to_id_path) and os.path.exists(self.id_to_token_path) and os.path.exists(self.rules_path):
            with open(self.token_to_id_path, "r") as f:
                self.token_to_id = json.load(f)
            with open(self.id_to_token_path, "r") as f:
                self.id_to_token = json.load(f)
            with open(self.rules_path, "r") as f:
                self.rules = json.load(f)

    def words(self, text):
        newline = r"[\n]"        
        words = re.sub(newline, " ", text)
        words = words.split(" ")
        return words

    def characterize(self, text, protected = None):
        words = self.words(text)

        characters = []

        for word in words: 
            if protected and word not in protected:
                characters.extend([char for char in word])
            elif protected and word in protected:
                characters.append(word)
            else:
                characters.extend([char for char in word])
            characters.append("</w>")            
        
        return characters
    
    def bpe_train(self):
        characters = self.characterize(self.meditations, self.protected_tokens)

        vocab = sorted(list(set(characters)))
        
        self.rules = []
        while len(vocab) < self.vocab_length:
            
            a = 0
            pairs = []

            while a < len(characters) - 1:
                if not characters[a+1] == "</w>":
                    pair = (characters[a], characters[a+1])
                    pairs.append(pair)
                    a += 1
                else:
                    a += 2

        
            rules = Counter(pairs).most_common()
            
            for i in rules:
                rulestr = ""
                for char in i[0]: rulestr += char
                if rulestr in vocab: continue
                else:
                    rule = i
                    vocab.append(rulestr)
                    break

            i = 0
            characters2 = []
            while i < len(characters):
                if i < len(characters) - 1 and characters[i] == rule[0][0] and characters[i+1] == rule[0][1]:
                    merged = rule[0][0] + rule[0][1]
                    characters2.append(merged)
                    i += 2
                else:    
                    characters2.append(characters[i])
                    i += 1


            characters = characters2

            self.rules.append((rule[0]))

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

        if os.path.exists(os.path.join(self.PROJECT_ROOT, 'data/vocabulary/embeddings.npy')):
            E = np.load(os.path.join(self.PROJECT_ROOT, 'data/vocabulary/embeddings.npy'))
        else:
            E = np.random.normal(0, 0.02, (self.vocab_length, self.d_model))
        self.E = E

        if not os.path.exists(os.path.join(self.PROJECT_ROOT, "data/vocabulary/embeddings.npy")):
            np.save(os.path.join(self.PROJECT_ROOT, "data/vocabulary/embeddings.npy"), E)
        
        with open(os.path.join(self.PROJECT_ROOT, "data/vocabulary/rules.json"), "w") as f:
            json.dump(self.rules, f)

    def tokenize(self, input_str):
        tokens = self.characterize(input_str, self.protected_tokens)


        with open(os.path.join(self.PROJECT_ROOT, "data/vocabulary/rules.json"), "r") as f:
            self.rules = json.load(f)

        #I will allow this to throw an exception manally, meaning it is out of sequence

        i = 0
        
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
        return [self.token_to_id[token] for token in tokens]
    
    def decode(self, input_nums):
        return [self.id_to_token[int(i)] for i in input_nums]
    
    def tpw_ratio(self, text):
        unique_words = self.tokenize(text)
        return len(unique_words) / self.vocab_length
        
