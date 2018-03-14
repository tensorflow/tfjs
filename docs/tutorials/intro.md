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

While Tensors allow us to store data, ops allow us to manipulate data. __deeplearn.js__ comes with a wide variety of mathematical operations suitable for linear algebra and machine learning. These include unary ops like `square()` and binary ops like `add()` and `mul()` Generally speaking an op will do some transformation on one of more tensors and return a new tensor as a result.

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
    // Once we return the loss the optimizer will adjust the network
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

Want to learn more? Read [these tutorials](index.md).
