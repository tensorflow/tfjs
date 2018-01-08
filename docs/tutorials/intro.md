---
layout: page
order: 1
---
# Introduction

**deeplearn.js** is an open source WebGL-accelerated JavaScript library for machine
intelligence. **deeplearn.js** brings highly performant machine learning
building blocks to your fingertips, allowing you to train neural networks
in a browser or run pre-trained models in inference mode. It provides an API for
constructing differentiable data flow graphs, as well as a set of mathematical
functions that can be used directly.

* TOC
{:toc}

You can find the code that supplements this tutorial
[here](https://github.com/PAIR-code/deeplearnjs/tree/master/demos/intro).

Run it yourself with:
```ts
./scripts/watch-demo demos/intro
```

And visit `http://localhost:8080/demos/intro/`.

Or just view the demo we have hosted [here](https://deeplearnjs.org/demos/intro/).

For the purposes of the documentation, we will use TypeScript code examples.
For vanilla JavaScript, you may need to remove the occasional TypeScript type annotation or definition.

This includes `console.log(await ndarray.data())`, which in ES5 would be written as:
`ndarray.data().then(function(data) { console.log(data); });`.

## Core concepts

### NDArrays

The central unit of data in **deeplearn.js** is the `NDArray`. An `NDArray`
consists of a set of floating point values shaped into an array of an arbitrary
number of dimensions. `NDArray`s have a `shape` attribute to define
their shape. The library provides sugar subclasses for low-rank `NDArray`s:
`Scalar`, `Array1D`, `Array2D`, `Array3D` and `Array4D`.

Example usage with a 2x3 matrix:

```js
const shape = [2, 3];  // 2 rows, 3 columns
const a = Array2D.new(shape, [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
```

`NDArray`s can store data either on the GPU as a `WebGLTexture`, where each
pixel stores a floating point value, or on the CPU as a vanilla JavaScript
`TypedArray`. Most of the time, the user should not think about the storage,
as it is an implementation detail.

If `NDArray` data is
stored on the CPU, the first time a GPU mathematical operation is called the
data will be uploaded to a texture automatically. If you call
`NDArray.data()` on a GPU-resident `NDArray`, the
library will download the texture to the CPU and delete the texture.

### NDArrayMath

The library provides a `NDArrayMath` base class which defines a set of
mathematical functions that operate on `NDArray`s.

#### NDArrayMath with WebGL backend

When using the `NDArrayMath` with the WebGL backend, these mathematical
operations enqueue shader programs to be executed on the GPU. Unlike in
CPU backend, **these operations are not blocking**, but the user can
synchronize the cpu with the gpu by calling `data()` on
the `NDArray`, as we describe in detail below.

These shaders read and write from `WebGLTexture`s which are owned by
`NDArray`s. When chaining mathematical operations, textures can stay in GPU
memory (not downloaded to the CPU between operations), which is critical for
performance.

Example of taking the mean squared difference between two matrices:

```js
const math = ENV.math;

const a = Array2D.new([2, 2], [1.0, 2.0, 3.0, 4.0]);
const b = Array2D.new([2, 2], [0.0, 2.0, 4.0, 6.0]);

// Non-blocking math calls.
const diff = math.sub(a, b);
const squaredDiff = math.elementWiseMul(diff, diff);
const sum = math.sum(squaredDiff);
const size = Scalar.new(a.size);
const average = math.divide(sum, size);

console.log('mean squared difference: ' + await average.val());
```

> TIP: Avoid calling `data()/dataSync()` between mathematical GPU
operations unless you are debugging. This forces a texture download, and
subsequent operation calls will have to re-upload the data to a new
texture.

#### NDArrayMathCPU

When using CPU implementations, these mathematical
operations are blocking and get executed immediately on the underlying
`TypedArray`s with vanilla JavaScript.

### Training

Differentiable data flow graphs in **deeplearn.js** use a delayed execution model,
just like in TensorFlow. Users construct a graph and then train or
infer on them by providing input `NDArray`s through `FeedEntry`s.

> NOTE: NDArrayMath and NDArrays are sufficient for inference mode. You only need a
graph if you want to train.

#### Graphs and Tensors
The `Graph` object is the core class for constructing data flow graphs.
`Graph` objects don't actually hold `NDArray` data, only connectivity
between operations.

The `Graph` class has differentiable operations as top level member
functions. When you call a graph method to add an operation, you get back a
`Tensor` object which only holds connectivity and shape information.

An example graph that multiplies an input by a variable:

```js
const g = new Graph();

// Placeholders are input containers. This is the container for where we will
// feed an input NDArray when we execute the graph.
const inputShape = [3];
const inputTensor = g.placeholder('input', inputShape);

const labelShape = [1];
const labelTensor = g.placeholder('label', labelShape);

// Variables are containers that hold a value that can be updated from
// training.
// Here we initialize the multiplier variable randomly.
const multiplier = g.variable('multiplier', Array2D.randNormal([1, 3]));

// Top level graph methods take Tensors and return Tensors.
const outputTensor = g.matmul(multiplier, inputTensor);
const costTensor = g.meanSquaredCost(outputTensor, labelTensor);

// Tensors, like NDArrays, have a shape attribute.
console.log(outputTensor.shape);
```

#### Session and FeedEntry

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
const math = ENV.math;

const session = new Session(g, math);
const optimizer = new SGDOptimizer(learningRate);

const inputs: Array1D[] = [
  Array1D.new([1.0, 2.0, 3.0]),
  Array1D.new([10.0, 20.0, 30.0]),
  Array1D.new([100.0, 200.0, 300.0])
];

const labels: Array1D[] = [
  Array1D.new([4.0]),
  Array1D.new([40.0]),
  Array1D.new([400.0])
];

// Shuffles inputs and labels and keeps them mutually in sync.
const shuffledInputProviderBuilder =
  new InCPUMemoryShuffledInputProviderBuilder([inputs, labels]);
const [inputProvider, labelProvider] =
  shuffledInputProviderBuilder.getInputProviders();

// Maps tensors to InputProviders.
const feedEntries: FeedEntry[] = [
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
const testInput = Array1D.new([0.1, 0.2, 0.3]);

// session.eval can take NDArrays as input data.
const testFeedEntries: FeedEntry[] = [
  {tensor: inputTensor, data: testInput}
];

const testOutput = session.eval(outputTensor, testFeedEntries);

console.log('---inference output---');
console.log('shape: ' + testOutput.shape);
console.log('value: ' + await testOutput.data());
```

Want to learn more? Read [these tutorials](index.md).
