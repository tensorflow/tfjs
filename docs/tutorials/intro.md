---
layout: page
order: 1
---
# Introduction

**deeplearn.js** is an open source WebGL-accelerated JavaScript library for machine
intelligence. **deeplearn.js** brings highly performant machine learning
building blocks to your fingertips, allowing you to train neural networks
in a browser or run pre-trained models in inference mode.

Lets take a look at some of the core concepts in deeplearn.js

## Tensors

The central unit of data in **deeplearn.js** is the `Tensor`. A `Tensor`
consists of a set of numerical values shaped into an array of an arbitrary
number of dimensions. Tensors have a `shape` attribute to define
their shape. The library provides sugar subclasses for low-rank Tensors:
`Scalar`, `Tensor1D`, `Tensor2D`, `Tensor3D` and `Tensor4D`, as well as helper functions to construct them.

Example usage with a 2x3 matrix:

```js
let shape = [2, 3]; // 2 rows, 3 columns
let a = dl.tensor2d([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], shape);
// deeplearn.js can also infer the shape
let b = dl.tensor2d([[0.0, 2.0], [4.0, 6.0]]);  // 2 rows, 2 columns
```

`Tensor` can store data either on the GPU as a `WebGLTexture`, or on the CPU as
a JavaScript `TypedArray`. Most of the time, the user should not think about
the storage, as it is an implementation detail.

One place you do want to think about understand this difference is when pulling
data out of a Tensor, for example when debugging.

```js
let a = dl.tensor2d([[0.0, 2.0], [4.0, 6.0]]);
a = a.square();

// Async call to get the data from the tensor
a.data().then(data => console.log('The data TypedArray', data));

// Alternatively we can also call this synchronously
let data = a.dataSync();
console.log('The data TypedArray', data);
```

In the example above we first create a tensor then call a math operation on it. This will
_upload_ that tensor to the GPU automatically. When we want to use it in out JavaScript context
(e.g. to print it out), we call `data()` or `dataSync()` to _download_ it to the CPU memory. Note that
this is a relatively expensive operation, so you would likely want to call the async version.


### Operations (Ops)

While Tensors allow us to store data, ops allow us to manipulate data. __deeplearn.js__ comes with a wide variety of mathematical opearations suitable for linear algebra and machine learning. These include unary ops like `square()` and binary ops like `add()` and `mul()` Generally speaking an op will do some transformation on one of more tensors and return a new tensor as a result.

```js
let a = dl.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
let b = dl.tensor2d([[0.0, 2.0], [4.0, 6.0]]);

// The library has a chainable API allowing you to call operations
// directly as methods on Tensors.
let average = a.sub(b).square().mean();

// All operations are also exposed as functions in the main namespace
// so we could also do.
let avg = dl.mean(dl.square(dl.sub(a, b)));
```

#### Tidy Operations

Because deeplearn.js uses the GPU to accelerate math operations, there is a need
to manage GPU memory. While in regular JavaScript this is handled with scopes, we
provide a convenience function to clean up intermediate memory created when performing
operations on tensors.

We call this function `dl.tidy`.

```js
let a = dl.tensor2d([1.0, 2.0, 3.0, 4.0]);

// dl.tidy takes a function to tidy up after
let average = dl.tidy(() => {
  // dl.tidy will clean up all the GPU memory used by tensors inside
  // this function, other than the tensor that is returned.
  //
  // Even in a short sequence of operations like the one below, a number
  // of intermediate tensors get created. So it is a good practice to
  // put your math ops in a tidy!
  return a.sub(b).square().mean();
});
```

Using `dl.tidy()` will help prevent memory leaks in your application, and can be used to more carefully control when memory is reclaimed.

The manual way to clean up a tensor's backing memory is the dispose method.

```js
let a = dl.tensor2d([[0.0, 2.0], [4.0, 6.0]]);
a = a.square();
a.dispose(); // Clean up GPU buffer
```

But using tidy functions is much more convenient!

## Training

At the heart of many machine learning problems is the question of actually _training_ the machine to do some task. In deeplearn.js this process is encapsulated by  _Optimizers_. Optimizers are strategies to progressively tune the variables of your model in order to reduce the error (or _loss_ in ML parlance) in your model's predictions.

We cover training and optimizers [in this tutorial](ml_beginners.md), but here is an outline of what the training process in deeplearn.js looks like.

```js
import * as dl from 'deeplearn';

// Initialize the models variables
const weights = dl.variable(dl.randomNormal([10, 64]));
const biases = dl.variable(dl.zeros([64]));

// Set a learning rate and create an optimizer.
const LEARNING_RATE = .1;
const optimizer = dl.train.sgd(LEARNING_RATE)

/**
 * Perform inference return a prediction
 */
function inference(input) { }

/**
 * Compute the loss of the model by comparing the prediction
 * and ground truth.
 *
 * Return a Scalar (i.e. single number tensor) loss value.
 */
function loss(predictions, labels) { }

/**
 * Train the model a *single* step.
 */
function trainStep(data, labels, returnCost = true) {
  // Calling optimizer.minimize will adjust the variables in the
  // model based on the loss value returned by your loss function.
  // It handles all the backpropogation and weight updates.
  const cost = optimizer.minimize(() => {
    // Any variables used in this inference function will be optimized
    // by the optimizer.

    // Make a prediction using the current state of the model
    const prediction = inference(data);

    // Compute loss of the current model and return it. Calculating this loss
    // should involve the variables we are trying to optimize.
    //
    // Once we return the less the optimizer will adjust the network
    // weights for our next iteration.
    return loss(prediction, labels);
  }, returnCost);

  // return the current loss/cost so that we can visualize it
  return cost;
}

/**
 * Train the model.
 *
 * Calls trainstep in a loop. Use await dl.nextFrame() to avoid
 * stalling the browser.
 *
 * You can load, batch and shuffle your data here.
 */
function train(data) { }
```

## Backends

The library provides a number of _backends_ which implement the core mathematics of the library, currently we have a __CPU__ backend and a __WebGL__ backend. Deeplearn.js will use the __WebGL__ backend by default whenever the browser supports it. The __WebGL__ backend uses the computers' __GPU__, to perform fast and highly optimized linear algebra kernels.

To force the use of the CPU backend, you can call `dl.setBackend('cpu')` at the start of your program

To check which backend is being used call `dl.getBackend()`.


### WebGL backend

When using the WebGL backend, mathematical
operations like `dl.add` enqueue shader programs to be executed on the GPU. Unlike in CPU backend, **these operations are not blocking** (though there is some overhead in moving data from main memory to GPU memory).

These shader programs read and write from `WebGLTexture`s. When chaining mathematical operations, textures can stay in GPU memory, which is critical for performance.

You can periodically _download_ data from the gpu by calling `data()` on a `Tensor`, this allows you to read that data in your main javascript thread.

Example of taking the mean squared difference between two matrices:

```js
const a = dl.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
const b = dl.tensor2d([[0.0, 2.0], [4.0, 6.0]]);

// All these operations will execute on the GPU (if available)
// without blocking the main thread.
const diff = dl.sub(a, b);

// Calling .data returns a promise that resolves to a TypedArray that holds
// the tensor data downloaded from the GPU.
diff.data().then(d => console.log('difference: ' + d));
// We could also use dataSync to do this synchronously.
console.log('difference: ' + diff.dataSync());
```

> TIP: Avoid calling `data()/dataSync()` between mathematical GPU
operations unless you are debugging. This forces a texture download, and
subsequent operation calls will have to re-upload the data to a new
texture.

### CPU Backend

When using CPU implementations, these mathematical
operations are blocking and get executed immediately on the underlying
`TypedArray`s in the main JavaScript thread.

The same operations are implemented on both so your code doesn't have to change based on which backend is used on the client.

## Graphs
**Note: the following sections describe using the deeplearn.js graph API. We have deprecated this API in support of a new 'eager' mode after research and community feedback. It will be removed in future versions of deeplearn.js. Eager Mode supports training**

Differentiable data flow graphs in **deeplearn.js** use a delayed execution model,
just like in TensorFlow. Users construct a graph and then train or
infer on them by providing input `NDArray`s through `FeedEntry`s.

> NOTE: NDArrayMath and NDArrays are sufficient for inference mode. You only need a
graph if you want to train.

The `Graph` object is the core class for constructing data flow graphs.
`Graph` objects don't actually hold `NDArray` data, only connectivity
between operations.

The `Graph` class has differentiable operations as top level member
functions. When you call a graph method to add an operation, you get back a
`Tensor` object which only holds connectivity and shape information.

An example graph that multiplies an input by a variable:

```js
const g = new dl.Graph();

// Placeholders are input containers. This is the container for where we will
// feed an input NDArray when we execute the graph.
const inputShape = [3];
const inputTensor = g.placeholder('input', inputShape);

const labelShape = [1];
const labelTensor = g.placeholder('label', labelShape);

// Variables are containers that hold a value that can be updated from
// training.
// Here we initialize the multiplier variable randomly.
const multiplier = g.variable('multiplier', dl.randNormal([1, 3]));

// Top level graph methods take Tensors and return Tensors.
const outputTensor = g.matmul(multiplier, inputTensor);
const costTensor = g.meanSquaredCost(outputTensor, labelTensor);

// Tensors, like NDArrays, have a shape attribute.
console.log(outputTensor.shape);
```

### Session and FeedEntry

Session objects are what drive the execution of `Graph`s. `FeedEntry`
(similar to TensorFlow `feed_dict`) are what provide data for the run,
feeding a value to a `Tensor` from a given NDArray.

> A quick note on batching: **deeplearn.js** hasn't yet implemented batching as an outer
dimension for operations. This means every top level graph op, as well as math
function, operate on single examples. However, batching is important so that
weight updates operate on the average of gradients over a batch. **deeplearn.js**
simulates batching by using an `InputProvider` in train `FeedEntry`s to
provide inputs, rather than `NDArray`s directly. The `InputProvider`
will get called for each item in a batch. We provide a
`InMemoryShuffledInputProviderBuilder` for shuffling a set of inputs and
keeping them in sync.

Training with the `Graph` object from above:

```js
const learningRate = .00001;
const batchSize = 3;
const math = dl.ENV.math;

const session = new dl.Session(g, math);
const optimizer = new dl.SGDOptimizer(learningRate);

const inputs: dl.Array1D[] = [
  dl.Array1D.new([1.0, 2.0, 3.0]),
  dl.Array1D.new([10.0, 20.0, 30.0]),
  dl.Array1D.new([100.0, 200.0, 300.0])
];

const labels: dl.Array1D[] = [
  dl.Array1D.new([4.0]),
  dl.Array1D.new([40.0]),
  dl.Array1D.new([400.0])
];

// Shuffles inputs and labels and keeps them mutually in sync.
const shuffledInputProviderBuilder =
  new dl.InCPUMemoryShuffledInputProviderBuilder([inputs, labels]);
const [inputProvider, labelProvider] =
  shuffledInputProviderBuilder.getInputProviders();

// Maps tensors to InputProviders.
const feedEntries: dl.FeedEntry[] = [
  {tensor: inputTensor, data: inputProvider},
  {tensor: labelTensor, data: labelProvider}
];

const NUM_BATCHES = 10;
for (let i = 0; i < NUM_BATCHES; i++) {
  // Train takes a cost tensor to minimize. Trains one batch. Returns the
  // average cost as a Scalar.
  const cost = session.train(
      costTensor, feedEntries, batchSize, optimizer, CostReduction.MEAN);

  console.log('last average cost (' + i + '): ' + await cost.val());
}
```

After training, we can infer through the graph:

```js
const testInput = dl.Array1D.new([0.1, 0.2, 0.3]);

// session.eval can take NDArrays as input data.
const testFeedEntries: dl.FeedEntry[] = [
  {tensor: inputTensor, data: testInput}
];

const testOutput = session.eval(outputTensor, testFeedEntries);

console.log('---inference output---');
console.log('shape: ' + testOutput.shape);
console.log('value: ' + await testOutput.data());
```

Want to learn more? Read [these tutorials](index.md).
