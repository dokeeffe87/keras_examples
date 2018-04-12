"""
Basic sequence to sequence example for translating English to French.  This does character to character translation, so
it won't be great.  We'll see if we can improve on this with a word level type model. Compare this with the original
RNN encoder-decoder method of 1406.1078
"""

from __future__ import print_function
from __future__ import division

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

batch_size = 64  # Batch size for training
epochs = 100  # Number of epochs to train for (should change this).
latent_dim = 256  # Latent dimensionality of the encoding space (what happens if I make this smaller?).
num_samples = 10000  # Number of samples to train on.
# Define the path to the data
data_path = 'data/fra-eng/fra.txt'

# Vectorize the data
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We need start of sequence character to identify the inputs (this is a tab)
    # Also need an end of sequence character to identify the target (this is \n)
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples: {0}'.format(len(input_texts)))
print('Number of unique input tokens: {0}'.format(num_encoder_tokens))
print('Number of unique output tokens: {0}'.format(num_decoder_tokens))
print('Max sequence length of inputs: {0}'.format(max_encoder_seq_length))
print('Max sequence length of outputs: {0}'.format(max_decoder_seq_length))

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dytpe='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dytpe='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dytpe='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is going to be ahead of decoder_input_data by one timestep.
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data is ahead by one timestep and will not include the start character
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.





