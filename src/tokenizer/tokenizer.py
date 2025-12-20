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

        self.token_to_id_path = os.path.join(self.PROJECT_ROOT, "data/vocabulary/token_to_id.json")
        self.id_to_token_path = os.path.join(self.PROJECT_ROOT, "data/vocabulary/id_to_token.json")
        self.rules_path = os.path.join(self.PROJECT_ROOT, "data/vocabulary/rules.json")
        self.vocab_path = os.path.join(self.PROJECT_ROOT, "data/vocabulary/vocab.json")

        if not (os.path.exists(self.token_to_id_path) and os.path.exists(self.id_to_token_path) and os.path.exists(self.rules_path) and os.path.exists(self.vocab_path)):
            self.tokenize_train()
        with open(self.token_to_id_path, "r") as f:
            self.token_to_id = json.load(f)
        with open(self.id_to_token_path, "r") as f:
            self.id_to_token = json.load(f)
        with open(self.rules_path, "r") as f:
            self.rules = json.load(f)
        with open(self.vocab_path, "r") as f:
            self.vocab = json.load(f)
            

    def words(self, text):
        newline = r"[\n]"        
        words = re.sub(newline, " ", text)
        words = words.split()
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
        vocab.append("<UNK>")

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

            if not rules:
                print("Rules not found")
                break
            
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

        self.embeddings_path = os.path.join(self.PROJECT_ROOT, 'data/vocabulary/embeddings.npy')

        if os.path.exists(self.embeddings_path):
            self.E = np.load(self.embeddings_path)
        else:
            self.E = np.random.normal(0, 0.02, (self.vocab_length, self.d_model))
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
            if token in self.vocab:
                encoded.append(self.token_to_id[token])

            else:
                encoded.append(self.token_to_id["<UNK>"])
        return encoded

    def decode(self, input_nums):
        return [self.id_to_token[str(i)] for i in input_nums]
    
    def tpw_ratio(self):
        unique_words = len(self.words(self.meditations))
        unique_tokens = len(self.tokenize(self.meditations))

        if unique_words == 0:
            return 0
        return unique_tokens / unique_words
        

tokenizer = Tokenizer()
print(tokenizer.tpw_ratio())