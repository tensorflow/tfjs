# Learning digits of pi using an LSTM

Demonstrates training a simple autoregressive LSTM network in Tensorflow and
then porting that model to deeplearn.js.

This network uses two ``BasicLSTMCell``s combined with ``MultiRNNCell``. The
network is trained to memorize the first few digits of pi.

First, train the LSTM network with Tensorflow:

```
python demos/lstm/train.py
```

Next, export the weights to be used by deeplearn.js:

```
python scripts/dump_checkpoints/dump_checkpoint_vars.py --model_type=tensorflow --output_dir=demos/lstm/ --checkpoint_file=/tmp/simple_lstm-1000 --remove_variables_regex=".*Adam.*|.*beta.*"
```

Finally, start the demo:

```
./scripts/watch-demo demos/lstm
```
