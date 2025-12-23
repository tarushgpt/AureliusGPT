from src.tokenizer.tokenizer import Tokenizer
from src.model.transformer import Transformer
import concurrent.futures
import numpy as np
from config import max_tokens_inference, temperature, epsilon
import argparse
from util import Util

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

        logits = self.transformer.forward(input_tokens)
        loss = self.transformer.ce_loss(output_tokens, logits)        

        dL_dlogits = self.transformer.ce_loss_gradient(output_tokens, logits)
        
        self.transformer.backward(dL_dlogits, input_tokens)
        return loss


class Test:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.transformer = Transformer(self.tokenizer)
        self.util = Util()
    
    def main(self):
        print("Welcome to AureliusGPT! Please press 'Enter' to break.\n\n")
        while True:
            user_input = input("User: ")
            if user_input == "":
                break
            self.run(user_input)


    def run(self, user_input):
        
        print("\n")

        tokens = self.tokenizer.encode(user_input)
        eos_token_id = self.tokenizer.token_to_id["<EOS>"]

        while len(tokens) < max_tokens_inference:

            logits = self.transformer.forward(tokens)[-1]
            
            logits = logits / temperature
            probs = self.util.softmax(logits, -1)
            next_token = np.random.choice(len(probs), p=probs)

            if next_token == eos_token_id:
                break

            tokens.append(next_token)

        response = self.tokenizer.decode(tokens)
        
        output = ""

        for token in response:
            output += str(token)

        space = "_"
        output = output.replace("_", " ")

        print("AureliusGPT: " + output)


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train")
    subparsers.add_parser("run")

    args = parser.parse_args()

    if args.command == "train":
        train = Train()
        train.train()
    elif args.command == "run":
        test = Test()
        test.main()


if __name__ == "__main__":
    main()