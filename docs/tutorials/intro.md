---
layout: page
order: 2
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
./scripts/watch-demo demos/intro/intro.ts
```

And visit `http://localhost:8080/demos/intro/`.

Or just view the demo we have hosted [here](https://pair-code.github.io/deeplearnjs/demos/intro/).

For the purposes of the documentation, we will use TypeScript code examples.
For vanilla JavaScript, you may need to remove TypeScript syntax like
`const`, `let`, or other type definitions.

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
`NDArray.getValues()` on a GPU-resident `NDArray`, the
library will download the texture to the CPU and delete the texture.

### NDArrayMath

The library provides a `NDArrayMath` base class which defines a set of
mathematical functions that operate on `NDArray`s.

#### NDArrayMathGPU

When using the `NDArrayMathGPU` implementation, these mathematical
operations enqueue shader programs to be executed on the GPU. Unlike in
`NDArrayMathCPU`, **these operations are not blocking**, but the user can
synchronize the cpu with the gpu by calling `get()` or `getValues()` on
the `NDArray`, as we describe in detail below.

These shaders read and write from `WebGLTexture`s which are owned by
`NDArray`s. When chaining mathematical operations, textures can stay in GPU
memory (not downloaded to the CPU between operations), which is critical for
performance.

Example of taking the mean squared difference between two matrices (more
details about `math.scope`, `keep`, and `track` below):

```js
const math = new NDArrayMathGPU();

math.scope((keep, track) => {
  const a = track(Array2D.new([2, 2], [1.0, 2.0, 3.0, 4.0]));
  const b = track(Array2D.new([2, 2], [0.0, 2.0, 4.0, 6.0]));

  // Non-blocking math calls.
  const diff = math.sub(a, b);
  const squaredDiff = math.elementWiseMul(diff, diff);
  const sum = math.sum(squaredDiff);
  const size = Scalar.new(a.size);
  const average = math.divide(sum, size);

  // Blocking call to actually read the values from average. Waits until the
  // GPU has finished executing the operations before returning values.
  // average is a Scalar so we use .get()
  console.log(average.get());
});
```

> NOTE: `NDArray.get()` and `NDArray.getValues()` are blocking calls.
There is no need to register callbacks after performing chained math functions,
just call `getValues()` to synchronize the CPU & GPU.

> TIP: Avoid calling `get()` or `getValues()` between mathematical GPU
operations unless you are debugging. This forces a texture download, and
subsequent `NDArrayMathGPU` calls will have to re-upload the data to a new
texture.

##### math.scope()

When math operations are used, you must wrap them in a math.scope() function
closure as shown in the example above. The results of math operations in this
scope will get disposed at the end of the scope, unless they are the value
returned in the scope.

Two functions are passed to the function closure, `keep()` and `track()`.

`keep()` ensures that the NDArray passed to keep will not be cleaned up
automatically when the scope ends.

`track()` tracks any NDArrays that you may construct directly inside of a
scope. When the scope ends, any manually tracked `NDArray`s will get
cleaned up. Results of all `math.method()` functions, as well as results of
many other core library functions are automatically cleaned up, so you don't
have to manually track them.

```ts
const math = new NDArrayMathGPU();

let output;

// You must have an outer scope, but don't worry, the library will throw an
// error if you don't have one.
math.scope((keep, track) => {
  // CORRECT: By default, math wont track NDArrays that are constructed
  // directly. You can call track() on the NDArray for it to get tracked and
  // cleaned up at the end of the scope.
  const a = track(Scalar.new(2));

  // INCORRECT: This is a texture leak!!
  // math doesn't know about b, so it can't track it. When the scope ends, the
  // GPU-resident NDArray will not get cleaned up, even though b goes out of
  // scope. Make sure you call track() on NDArrays you create.
  const b = Scalar.new(2);

  // CORRECT: By default, math tracks all outputs of math functions.
  const c = math.neg(math.exp(a));

  // CORRECT: d is tracked by the parent scope.
  const d = math.scope(() => {
    // CORRECT: e will get cleaned up when this inner scope ends.
    const e = track(Scalar.new(3));

    // CORRECT: The result of this math function is tracked. Since it is the
    // return value of this scope, it will not get cleaned up with this inner
    // scope. However, the result will be tracked automatically in the parent
    // scope.
    return math.elementWiseMul(e, e);
  });

  // CORRECT, BUT BE CAREFUL: The output of math.tanh will be tracked
  // automatically, however we can call keep() on it so that it will be kept
  // when the scope ends. That means if you are not careful about calling
  // output.dispose() some time later, you might introduce a texture memory
  // leak. A better way to do this would be to return this value as a return
  // value of a scope so that it gets tracked in a parent scope.
  output = keep(math.tanh(d));
});
```

> More technical details: When WebGL textures go out of scope in JavaScript,
they don't get cleaned up automatically by the browser's garbage collection
mechanism. This means when you are done with an NDArray that is GPU-resident,
it must manually be disposed some time later. If you forget to manually call
`ndarray.dispose()` when you are done with an NDArray, you will introduce
a texture memory leak, which will cause serious performance issues.
If you use `math.scope()`, any NDArrays created by `math.method()` and
any other method that returns the result through a scope will automatically
get cleaned up.


> If you want to do manual memory management and not use math.scope(), you can
construct a `NDArrayMath` object with safeMode = false. This is not
recommended, but is useful for `NDArrayMathCPU` since CPU-resident memory
will get cleaned up automatically by the JavaScript garbage collector.


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
const math = new NDArrayMathGPU();

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
  // Wrap session.train in a scope so the cost gets cleaned up automatically.
  math.scope(() => {
    // Train takes a cost tensor to minimize. Trains one batch. Returns the
    // average cost as a Scalar.
    const cost = session.train(
        costTensor, feedEntries, batchSize, optimizer, CostReduction.MEAN);

    console.log('last average cost (' + i + '): ' + cost.get());
  });
}
```

After training, we can infer through the graph:

```js
// Wrap session.eval in a scope so the intermediate values get cleaned up
// automatically.
math.scope((keep, track) => {
  const testInput = track(Array1D.new([0.1, 0.2, 0.3]));

  // session.eval can take NDArrays as input data.
  const testFeedEntries: FeedEntry[] = [
    {tensor: inputTensor, data: testInput}
  ];

  const testOutput = session.eval(outputTensor, testFeedEntries);

  console.log('---inference output---');
  console.log('shape: ' + testOutput.shape);
  console.log('value: ' + testOutput.get(0));
});

// Cleanup training data.
inputs.forEach(input => input.dispose());
labels.forEach(label => label.dispose());
```

Want to learn more? Read [these tutorials](index.md).
