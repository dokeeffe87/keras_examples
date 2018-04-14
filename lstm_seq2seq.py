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
import io

batch_size = 64  # Batch size for training
# For testing
# epochs = 5
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
with io.open(data_path, 'r', encoding='utf-8') as f:
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

encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is going to be ahead of decoder_input_data by one timestep.
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data is ahead by one timestep and will not include the start character
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

# Process an input sequence
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Now we can set up the decoder.  The initial state is then 'encoder_states', the last state of the encoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Now define the model to turn encoder_input_data and decoder_input_data into decoder_target_data
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Start the training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size, epochs=epochs, validation_split=0.2)
# Save the trained model
model.save('s2s.h5')

# Build out the inference mode (sampling).  There are 3 steps:
# 1) encode input and retrieve initial decoder state.
# 2) run one step of decoder with this initial state and a start of sequence token as a target. The output will be the
#    next target token.
# 3) repeat with the current target token and current states

# Define the sampling model
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Reverse-lookup token to decode sequence back to something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # First, encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences. Assume batch of size 1.
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length of find stop character.
        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (length 1)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (which is in the training set) and see how it fares in terms of decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print(u'Input sentence: {0}'.format(input_texts[seq_index]))
    print(u'Translated sentence: {0}'.format(decoded_sentence))







