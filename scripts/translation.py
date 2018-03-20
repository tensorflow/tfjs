# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================
"""Train a simple LSTM model for character-level language translation.

This is based on the Keras example at:
  https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py

The training data can be downloaded with a command like the following example:
  wget http://www.manythings.org/anki/fra-eng.zip
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import json
import os

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import tensorflowjs as tfjs


def read_data():
  # Vectorize the data.
  input_texts = []
  target_texts = []
  input_characters = set()
  target_characters = set()
  lines = io.open(FLAGS.data_path, 'r', encoding='utf-8').read().split('\n')
  for line in lines[: min(FLAGS.num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character for the targets, and "\n"
    # as "end sequence" character.
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

  print('Number of samples:', len(input_texts))
  print('Number of unique input tokens:', num_encoder_tokens)
  print('Number of unique output tokens:', num_decoder_tokens)
  print('Max sequence length for inputs:', max_encoder_seq_length)
  print('Max sequence length for outputs:', max_decoder_seq_length)

  input_token_index = dict(
      [(char, i) for i, char in enumerate(input_characters)])
  target_token_index = dict(
      [(char, i) for i, char in enumerate(target_characters)])

  # Save the token indices to file.
  metadata_json_path = os.path.join(
      FLAGS.artifacts_dir, 'translation.metadata.json')
  if not os.path.isdir(os.path.dirname(metadata_json_path)):
    os.makedirs(os.path.dirname(metadata_json_path))
  with io.open(metadata_json_path, 'w', encoding='utf-8') as f:
    metadata = {
        'input_token_index': input_token_index,
        'target_token_index': target_token_index,
        'max_encoder_seq_length': max_encoder_seq_length,
        'max_decoder_seq_length': max_decoder_seq_length
    }
    f.write(json.dumps(metadata, ensure_ascii=False))
  print('Saved metadata at: %s' % metadata_json_path)

  encoder_input_data = np.zeros(
      (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
      dtype='float32')
  decoder_input_data = np.zeros(
      (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
      dtype='float32')
  decoder_target_data = np.zeros(
      (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
      dtype='float32')

  for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
      encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
      # decoder_target_data is ahead of decoder_input_data by one timestep
      decoder_input_data[i, t, target_token_index[char]] = 1.
      if t > 0:
        # decoder_target_data will be ahead by one timestep
        # and will not include the start character.
        decoder_target_data[i, t - 1, target_token_index[char]] = 1.

  return (input_texts, max_encoder_seq_length, max_decoder_seq_length,
          num_encoder_tokens, num_decoder_tokens,
          input_token_index, target_token_index,
          encoder_input_data, decoder_input_data, decoder_target_data)


def seq2seq_model(num_encoder_tokens, num_decoder_tokens, latent_dim):
  """Create a Keras model for the seq2seq translation.

  Args:
    num_encoder_tokens: Total number of distinct tokens in the inputs
      to the encoder.
    num_decoder_tokens: Total number of distinct tokens in the outputs
      to/from the decoder
    latent_dim: Number of latent dimensions in the LSTMs.

  Returns:
    encoder_inputs: Instance of `keras.Input`, symbolic tensor as input to
      the encoder LSTM.
    encoder_states: Instance of `keras.Input`, symbolic tensor for output
      states (h and c) from the encoder LSTM.
    decoder_inputs: Instance of `keras.Input`, symbolic tensor as input to
      the decoder LSTM.
    decoder_lstm: `keras.Layer` instance, the decoder LSTM.
    decoder_dense: `keras.Layer` instance, the Dense layer in the decoder.
    model: `keras.Model` instance, the entire translation model that can be
      used in training.
  """
  # Define an input sequence and process it.
  encoder_inputs = Input(shape=(None, num_encoder_tokens))
  encoder = LSTM(latent_dim,
                 return_state=True,
                 recurrent_initializer=FLAGS.recurrent_initializer)
  _, state_h, state_c = encoder(encoder_inputs)
  # We discard `encoder_outputs` and only keep the states.
  encoder_states = [state_h, state_c]

  # Set up the decoder, using `encoder_states` as initial state.
  decoder_inputs = Input(shape=(None, num_decoder_tokens))
  # We set up our decoder to return full output sequences,
  # and to return internal states as well. We don't use the
  # return states in the training model, but we will use them in inference.
  decoder_lstm = LSTM(FLAGS.latent_dim,
                      return_sequences=True,
                      return_state=True,
                      recurrent_initializer=FLAGS.recurrent_initializer)
  decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                       initial_state=encoder_states)
  decoder_dense = Dense(num_decoder_tokens, activation='softmax')
  decoder_outputs = decoder_dense(decoder_outputs)

  # Define the model that will turn
  # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
  model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
  return (encoder_inputs, encoder_states, decoder_inputs, decoder_lstm,
          decoder_dense, model)


def decode_sequence(input_seq,
                    encoder_model,
                    decoder_model,
                    num_decoder_tokens,
                    target_begin_index,
                    reverse_target_char_index,
                    max_decoder_seq_length):
  """Decode (i.e., translate) an encoded sentence.

  Args:
    input_seq: A `numpy.ndarray` of shape
      `(1, max_encoder_seq_length, num_encoder_tokens)`.
    encoder_model: A `keras.Model` instance for the encoder.
    decoder_model: A `keras.Model` instance for the decoder.
    num_decoder_tokens: Number of unique tokens for the decoder.
    target_begin_index: An `int`: the index for the beginning token of the
      decoder.
    reverse_target_char_index: A lookup table for the target characters, i.e.,
      a map from `int` index to target character.
    max_decoder_seq_length: Maximum allowed sequence length output by the
      decoder.

  Returns:
    The result of the decoding (i.e., translation) as a string.
  """

  # Encode the input as state vectors.
  states_value = encoder_model.predict(input_seq)

  # Generate empty target sequence of length 1.
  target_seq = np.zeros((1, 1, num_decoder_tokens))
  # Populate the first character of target sequence with the start character.
  target_seq[0, 0, target_begin_index] = 1.

  # Sampling loop for a batch of sequences
  # (to simplify, here we assume a batch of size 1).
  stop_condition = False
  decoded_sentence = ''
  while not stop_condition:
    output_tokens, h, c = decoder_model.predict(
        [target_seq] + states_value)

    # Sample a token
    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    sampled_char = reverse_target_char_index[sampled_token_index]
    decoded_sentence += sampled_char

    # Exit condition: either hit max length
    # or find stop character.
    if (sampled_char == '\n' or
        len(decoded_sentence) > max_decoder_seq_length):
      stop_condition = True

    # Update the target sequence (of length 1).
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, sampled_token_index] = 1.

    # Update states
    states_value = [h, c]

  return decoded_sentence


def main():
  (input_texts, _, max_decoder_seq_length,
   num_encoder_tokens, num_decoder_tokens,
   __, target_token_index,
   encoder_input_data, decoder_input_data, decoder_target_data) = read_data()

  (encoder_inputs, encoder_states, decoder_inputs, decoder_lstm,
   decoder_dense, model) = seq2seq_model(
       num_encoder_tokens, num_decoder_tokens, FLAGS.latent_dim)

  # Run training.
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
  model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
            batch_size=FLAGS.batch_size,
            epochs=FLAGS.epochs,
            validation_split=0.2)

  tfjs.converters.save_keras_model(model, FLAGS.artifacts_dir)

  # Next: inference mode (sampling).
  # Here's the drill:
  # 1) encode input and retrieve initial decoder state
  # 2) run one step of decoder with this initial state
  # and a "start of sequence" token as target.
  # Output will be the next target token
  # 3) Repeat with the current target token and current states

  # Define sampling models
  encoder_model = Model(encoder_inputs, encoder_states)

  decoder_state_input_h = Input(shape=(FLAGS.latent_dim,))
  decoder_state_input_c = Input(shape=(FLAGS.latent_dim,))
  decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
  decoder_outputs, state_h, state_c = decoder_lstm(
      decoder_inputs, initial_state=decoder_states_inputs)
  decoder_states = [state_h, state_c]
  decoder_outputs = decoder_dense(decoder_outputs)
  decoder_model = Model(
      [decoder_inputs] + decoder_states_inputs,
      [decoder_outputs] + decoder_states)

  # Reverse-lookup token index to decode sequences back to
  # something readable.
  reverse_target_char_index = dict(
      (i, char) for char, i in target_token_index.items())

  target_begin_index = target_token_index['\t']

  for seq_index in range(FLAGS.num_test_sentences):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(
        input_seq, encoder_model, decoder_model, num_decoder_tokens,
        target_begin_index, reverse_target_char_index, max_decoder_seq_length)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      'Keras seq2seq translation model training and serialization')
  parser.add_argument(
      'data_path',
      type=str,
      help='Path to the training data, e.g., ~/ml-data/fra-eng/fra.txt')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=64,
      help='Training batch size.')
  parser.add_argument(
      '--epochs',
      type=int,
      default=100,
      help='Number of training epochs.')
  parser.add_argument(
      '--latent_dim',
      type=int,
      default=256,
      help='Latent dimensionality of the encoding space.')
  parser.add_argument(
      '--num_samples',
      type=int,
      default=10000,
      help='Number of samples to train on.')
  parser.add_argument(
      '--num_test_sentences',
      type=int,
      default=100,
      help='Number of example sentences to test at the end of the training.')
  # TODO(cais): This is a workaround for the limitation in TF.js Layers that the
  # default recurrent initializer "Orthogonal" is currently not supported.
  # Remove this once "Orthogonal" becomes available.
  parser.add_argument(
      '--recurrent_initializer',
      type=str,
      default='orthogonal',
      help='Custom initializer for recurrent kernels of LSTMs (e.g., '
      'glorot_uniform)')
  parser.add_argument(
      '--artifacts_dir',
      type=str,
      default='/tmp/translation.keras',
      help='Local path for saving the TensorFlow.js artifacts.')

  FLAGS, _ = parser.parse_known_args()
  main()
