from tokenizer.tokenizer import Tokenizer
from model.transformer import Transformer
import concurrent.futures
import numpy as np
from config import max_tokens_inference

class Train:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.transformer = Transformer(self.tokenizer)
        self.chunked = self.tokenizer.tokenize_and_chunk_corpus()

    #sequential training takes ~5min by my calculations as opposed to using ProcessPoolExecutor / changing plumbing
    #v0 will focus on sequential; future training strategies will have concurrent (as a practice / future CUDA drilling, no significant engineering/latency impact)
    def train(self):
        for i in range(10):
            for chunk in self.chunked:
                self.train_chunk(chunk)

    def train_chunk(self, chunk):
        input_tokens = chunk[:-1]
        output_tokens = chunk[1:]


        probability_distribution = self.transformer.forward(input_tokens)
        
        losses = []

        for i in range(len(probability_distribution)):
            predicted_distribution = probability_distribution[i]
            actual_token = output_tokens[i]
            losses.append(self.transformer.ce_loss(predicted_distribution, actual_token))
        
        dL = np.mean(losses)

        gradients = self.transformer.backward(dL)

        #update all gradients; that is one chunk completed





class Test:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.transformer = Transformer(self.tokenizer)
    
    def main(self):
        print("Welcome to AureliusGPT! Please press 'Enter' to break.\n\n")
        while True:
            user_input = input("User: ")
            if user_input == " ":
                break
            self.run(user_input)


    def run(self, user_input):
        
        print("\n")

        tokens = self.tokenizer.encode(user_input)
        eos_token_id = self.tokenizer.token_to_id("<EOS>")

        while len(tokens) < max_tokens_inference:

            output = self.transformer.forward(tokens)[-1]

            next_token = np.argmax(output)

            if next_token == eos_token_id:
                break

            tokens.append(next_token)

        response = self.tokenizer.decode(tokens)
        print("AureliusGPT: " + response)



if __name__ == "__train__":
    train = Train()
    train.train()


elif __name__ == "__test__":
    test = Test()
    test.main()