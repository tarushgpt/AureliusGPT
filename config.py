import os

d_model = 256
d_k = 64
d_v = 64
h = 4
d_ff = 1024
max_seq_length = 512
lr = 0.001
num_blocks = 3
vocab_length = 1000
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
min_freq = 30