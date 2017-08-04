---
layout: page
order: 4
---

# Port TensorFlow models
This tutorial demonstrates training and porting a TensorFlow model to **deeplearn.js**.
The code and all the necessary resources used in this tutorial are stored in
`demos/mnist`.

We will use a fully connected neural network that predicts hand-written digits
from the MNIST dataset. The code is forked from the official
[TensorFlow MNIST tutorial](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/fully_connected_feed.py).

> NOTE: We will refer to the base directory of the **deeplearn.js** repo as `$BASE`.

First, we clone the **deeplearn.js** repository and make sure we have TensorFlow
installed. We cd into `$BASE` and train the model by running:

```bash
python demos/mnist/fully_connected_feed.py
```

The training should take ~1 minute and will store a model checkpoint in
`/tmp/tensorflow/mnist/tensorflow/mnist/logs/fully_connected_feed/`.

Next, we need to port the weights from the TensorFlow checkpoint to **deeplearn.js**.
We provide a script that does this. We run it from the `$BASE` directory:

```bash
python scripts/dump_checkpoint_vars.py --output_dir=demos/mnist/ --checkpoint_file=/tmp/tensorflow/mnist/logs/fully_connected_feed/model.ckpt-1999
```

The script will save a set of files (one file per variable, and a
`manifest.json`) in the `demos/mnist` directory. The `manifest.json` is a simple
dictionary that maps variable names to files and their shapes:

```json
{
  ...,
  "hidden1/weights": {
    "filename": "hidden1_weights",
    "shape": [784, 128]
  },
  ...
}
```

One last thing before we start coding - we need to run a static HTTP server from
the `$BASE` directory:

```bash
npm run prep
./node_modules/.bin/http-server
>> Starting up http-server, serving ./
>> Available on:
>>   http://127.0.0.1:8080
>> Hit CTRL-C to stop the server
```

Make sure you can access `manifest.json` via HTTP by visiting
`http://localhost:8080/demos/mnist/manifest.json` in the browser.

We are ready to write some **deeplearn.js** code!

> NOTE: If you choose to write in TypeScript,
make sure you compile the code to JavaScript and serve it via the static HTTP
server.


To read the weights, we need to create a `CheckpointLoader` and point it to the
manifest file. We then call `loader.getAllVariables()` which returns a
dictionary that maps variable names to `NDArray`s. At that point, we are ready
to write our model. Here is a snippet demonstrating the use of
`CheckpointLoader`:

```ts
import {CheckpointLoader, Graph} from 'deeplearnjs';
// manifest.json is in the same dir as index.html.
const reader = new CheckpointReader('.');
reader.getAllVariables().then(vars => {
  // Write your model here.
  const g = new Graph();
  const input = g.placeholder('input', [784]);
  const hidden1W = g.constant(vars['hidden1/weights']);
  const hidden1B = g.constant(vars['hidden1/biases']);
  const hidden1 = g.relu(g.add(g.matmul(input, hidden1W), hidden1B));
  ...
  ...
  const math = new NDArrayMathGPU();
  const sess = new Session(g, math);
  math.scope(() => {
    const result = sess.eval(...);
    console.log(result.getValues());
  });
});
```

For details regarding the full model code see `demos/mnist/mnist.ts`. The demo
provides the exact implementation of the MNIST model using 3 different API:

- `buildModelGraphAPI()` uses the Graph API which mimics the TensorFlow API,
providing a lazy execution with feeds and fetches. Users do not need to worry
about GPU-related memory leaks, other than their input data.
- `buildModelLayerAPI()` uses the Graph API in conjuction with `Graph.layers`,
which mimics the Keras layers API.
- `buildModelMathAPI()` uses the Math API. This is the lowest level API in
**deeplearn.js** giving the most control to the user. Math commands execute immediately,
like numpy. Math commands are wrapped in math.scope() so that NDArrays created
by intermediate math commands are automatically cleaned up.

To run the mnist demo, we provide a `watch-demo` script that watches and
recompiles the typescript code when it changes. In addition, the script runs a
simple HTTP server on 8080 that serves the static html/js files. Before you run
`watch-demo`, make sure you kill the HTTP server we started earlier in the
tutorial in order to free up the 8080 port. Then run `watch-demo` from `$BASE`
pointed to the entry-point of the web app demo, `demos/mnist/mnist.ts`:

```bash
./scripts/watch-demo demos/mnist/mnist.ts
>> Starting up http-server, serving ./
>> Available on:
>>   http://127.0.0.1:8080
>>   http://192.168.1.5:8080
>> Hit CTRL-C to stop the server
>> 1410084 bytes written to demos/mnist/bundle.js (0.91 seconds) at 5:17:45 PM
```

Visit `http://localhost:8080/demos/mnist/` and you should see a simple page
showing test accuracy of ~90% measured using a test set of 50 mnist images
stored in `demos/mnist/sample_data.json`. Feel free to play with the demo
(e.g. make it interactive) and send us a pull request!
