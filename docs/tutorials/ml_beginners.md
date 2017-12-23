---
layout: page
order: 3
---
# Guide for non-ML experts

* TOC
{:toc}

You can find the code that supplements this tutorial
[here](https://github.com/PAIR-code/deeplearnjs/tree/master/demos/ml_beginners).

Run it yourself with:
```ts
./scripts/watch-demo demos/ml_beginners
```

And visit `http://localhost:8080/demos/ml_beginners/`.

Or just view the demo we have hosted [here](https://deeplearnjs.org/demos/ml_beginners/).

For the purposes of the documentation, we will use TypeScript code examples.
For vanilla JavaScript, you may need to remove the occasional TypeScript type annotation or definition.

This includes `console.log(await ndarray.data())`, which in ES5 would be written as:
`ndarray.data().then(data => console.log(data));`.

### NDArrays, Tensors, and numbers

#### Mathematical tensors

Mathematically, a "tensor" is the most basic object of linear algebra, a
generalization of numbers, vectors, and matrices. A vector can be thought of as
a 1-dimensional list of numbers; a matrix as a 2-dimensional list of numbers. A
tensor simply generalizes that concept to n-dimensional lists of numbers. It is
any arrangement of component numbers (or even strings or other data types)
arranged in any multi-dimensional rectangular array.

A tensor has several properties:

*   It has a <b>type</b>, which describes the types of each of its components,
    for instance `integer`, `float`, etc. **deeplearn.js** only supports
    `float32` for now.
*   It has a <b>shape</b>, a list of integers which describes the shape of the
    rectangular array of components. When you say a matrix is "4 by 4", you are
    describing the shape of the matrix.
*   It has a <b>rank</b>, which is the length of its shape; the dimension of the
    array of components. A vector has rank 1; a matrix has rank 2.

Example tensor     | Type  | Shape    | Rank
------------------ | ----- | -------- | ----
Scalar: 3.0        | float | \[\]     | 0
Vector: (1, 5, -2) | int   | \[3\]    | 1
2x2 Matrix         | int   | \[2, 2\] | 2

#### number[], NDArray, Tensor: Three data types for mathematical tensors

This same object a mathematician would call a "tensor" is represented in three
different ways in **deeplearn.js**. The above discussion (ranks, shapes, and
types), applies to all of them, but they are different and it is important to
keep them straight:

*   <b>`number[]`</b> is the underlying javascript type corresponding to an
    array of numbers. Really it is `number` for a 0-rank tensor, `number[]` for
    a 1-rank tensor, `number[][]` for 2-rank, etc. You won't use it much within
    **deeplearn.js**, but it is the way you will get things in and out of
    **deeplearn.js**.
*   <b>`NDArray`</b> is **deeplearn.js**'s more powerful implementation.
    Calculations involving NDArrays can be performed on the client's GPU, which
    is the fundamental advantage of **deeplearn.js**. This is the format most
    actual tensor data will be in when the calculations ultimately happen. When
    you call `Session.eval`, for instance, this is what you get back (and what
    the `FeedEntry` inputs should be). You can convert between `NDArray` and
    `number[]` using `NDArray.new(number[])` and `NDArray.get([indices])`
*   <b>`Tensor`</b> is an empty bag; it has no actual data inside. It is a
    placeholder used when a Graph is constructed, which records the shape and
    type of the data that will ultimately fit inside it. It contains no actual
    values in its components. Just by knowing the shape and type, however, it
    can do important error-catching at Graph-construction time; if you want to
    multiply a 2x3 matrix by a 10x10 matrix, the graph can yell at you when you
    create the node, before you ever give it input data. It does not make sense
    to convert directly between a `Tensor` and an `NDArray` or `number[]`; if
    you find you are trying to do that, one of the following is probably true:
    *   You have a static `NDArray` that you want to use in a graph; use
        graph.constant() to create a constant ```Tensor``` node.
    *   You have an `NDArray` that you want to feed in as an input to the Graph.
        Create a Placeholder with ```graph.placeholder``` in the graph, and then
        send your input to the graph in the `FeedEntry`.
    *   You have an output tensor and you want the session to evaluate and
        return its value. Call `Session.eval(tensor)`.


In general, you should only use a `Graph` when you want automatic
differentiation (training). If you just want to use the library for forward
mode inference, or just general numeric computation, using `NDArray`s with
`NDArrayMath` will suffice.

If you are interested in training, you must use a `Graph`. When you
construct a graph you will be working with `Tensor`s, and when you execute it
with `Session.eval`, the result will be `NDArray`s.

### Forward mode inference / numeric computation

If you just want to perform mathematical operations on NDArrays, you can simply
construct `NDArray`s with your data, and perform operations on them with
a `NDArrayMath` object.

For example, if you want to compute a matrix times a vector on the GPU:
```ts
const math = ENV.math;

const matrixShape = [2, 3];  // 2 rows, 3 columns.
const matrix = Array2D.new(matrixShape, [10, 20, 30, 40, 50, 60]);
const vector = Array1D.new([0, 1, 2]);
const result = math.matrixTimesVector(matrix, vector);

console.log("result shape:", result.shape);
console.log("result", await result.data());
```

For more information on `NDArrayMath`, see [Introduction and core concepts](intro.md).

The `NDArray`/`NDArrayMath` layer can be thought of as analogous to
[NumPy](http://www.numpy.org/).

### Training: delayed execution, Graphs, and Sessions

The most important thing to understand about training
(automatic differentiation) in **deeplearn.js** is that it uses a delayed
execution model. Your code will contain two separate stages: first you will
build a `Graph`, the object representing the calculation you want to do, then
later you will execute the graph and get the results.

Most of the time, your `Graph` will transform some input(s) to some output(s).
In general, the architecture of your `Graph` will stay fixed, but it will
contain parameters that will be automatically updated.

When you execute a `Graph`, there are two modes: training, and inference.

Inference is the act of providing a `Graph` an input to produce an output.

Training a graph involves providing the `Graph` many examples of labelled
input / output pairs, and automatically updating parameters of the `Graph`
so that the output of the `Graph` when evaluating (inferring) an input is
closer to the labelled output. The function that gives a `Scalar` representing
how close labelled output to a generated output is called the "cost function"
(also known as a "loss function"). The loss function should output close to
zero when the model is performing well. You must provide a cost function when
training.

**deeplearn.js** is structured very similarly to
[Tensorflow](https://www.tensorflow.org), Google's python-based machine
learning language. If you know TensorFlow, the concepts of `Tensor`s, `Graph`s,
and `Session`s are all almost the same, however we assume no knowledge of
TensorFlow here.

#### Graphs as Functions

The difference can be understood by analogy to regular JavaScript code. For the
rest of this tutorial, we will be working with this quadratic equation:

```ts
// y = a * x^2 + b * x + c
const x = 4;
const a = Math.random();
const b = Math.random();
const c = Math.random();

const order2 = a * Math.pow(x, 2);
const order1 = b * x;
const y = order2 + order1 + c;
```

In this original code, the mathematical calculations are evaluated in each line
as it's processed immediately.

Contrast that with the following code, analogous to how **deeplearn.js**
`Graph` inference works.

```ts
function graph(x, a, b, c) {
  const order2 = a * Math.pow(x, 2);
  const order1 = b * x;
  return order2 + order1 + c;
}

const a = Math.random();
const b = Math.random();
const c = Math.random();
const y = graph(4, a, b, c);
```

This code is in two blocks: first setting up the graph function, and then
calling it. The code to set up the graph function in the first few lines does
not do any actual mathematical operations until the function is called in the
final line. During the setup, basic compiler type-safety errors can be caught
even though the calculations are not yet performed.

This is exactly analogous to how `Graph`s work in **deeplearn.js**. The first
part of your code will set up the graph, describing:

 * Inputs, in our case "x". Inputs are represented as placeholders
   (e.g. `graph.placeholder()`).
 * Outputs, in our case "order1", "order2", and the final output "y".
 * Operations to produce outputs, in our case the decomposed functions of the
   quadratic (x^2, multiplication, addition).
 * Updatable parameters, in our case "a", "b", "c". Updatable parameters are
   represented as variables (e.g. `graph.variable()`)


Then in a later part of your code you will "call" (`Session.eval`) the graph's
function on certain inputs, and you will learn the values for "a", "b", and
"c", for some data with `Session.train`.

One minor difference between the above function analogy and **deeplearn.js**
`Graph`s is that the `Graph` does not specify its output. Instead, the caller
of the `Graph` function specifies which of the tensors they want to be
returned. This allows different calls to the same `Graph` to execute different
parts of it. Only the parts necessary to obtain the results demanded by the
caller will be evaluated.

Inference and training of a `Graph` is driven by a `Session` object. This
object contains runtime state, weights, activations, and gradients
(derivatives), whereas the `Graph` object only holds connectivity information.

So the function above would be implemented in **deeplearn.js** as follows:

```ts
const graph = new Graph();
// Make a new input in the graph, called 'x', with shape [] (a Scalar).
const x: Tensor = graph.placeholder('x', []);
// Make new variables in the graph, 'a', 'b', 'c' with shape [] and random
// initial values.
const a: Tensor = graph.variable('a', Scalar.new(Math.random()));
const b: Tensor = graph.variable('b', Scalar.new(Math.random()));
const c: Tensor = graph.variable('c', Scalar.new(Math.random()));
// Make new tensors representing the output of the operations of the quadratic.
const order2: Tensor = graph.multiply(a, graph.square(x));
const order1: Tensor = graph.multiply(b, x);
const y: Tensor = graph.add(graph.add(order2, order1), c);

// When training, we need to provide a label and a cost function.
const yLabel: Tensor = graph.placeholder('y label', []);
// Provide a mean squared cost function for training. cost = (y - yLabel)^2
const cost: Tensor = graph.meanSquaredCost(y, yLabel);

// At this point the graph is set up, but has not yet been evaluated.
// **deeplearn.js** needs a Session object to evaluate a graph.
const math = ENV.math;
const session = new Session(graph, math);

// For more information on scope, check out the [tutorial on performance](performance.md).
await math.scope(async () => {
  /**
   * Inference
   */
  // Now we ask the graph to evaluate (infer) and give us the result when
  // providing a value 4 for "x".
  // NOTE: "a", "b", and "c" are randomly initialized, so this will give us
  // something random.
  let result: NDArray =
      session.eval(y, [{tensor: x, data: Scalar.new(4)}]);
  console.log(result.shape);
  console.log('result', await result.data());

  /**
   * Training
   */
  // Now let's learn the coefficients of this quadratic given some data.
  // To do this, we need to provide examples of x and y.
  // The values given here are for values a = 3, b = 2, c = 1, with random
  // noise added to the output so it's not a perfect fit.
  const xs: Scalar[] = [
    Scalar.new(0),
    Scalar.new(1),
    Scalar.new(2),
    Scalar.new(3)
  ];
  const ys: Scalar[] = [
    Scalar.new(1.1),
    Scalar.new(5.9),
    Scalar.new(16.8),
    Scalar.new(33.9)
  ];
  // When training, it's important to shuffle your data!
  const shuffledInputProviderBuilder =
      new InCPUMemoryShuffledInputProviderBuilder([xs, ys]);
  const [xProvider, yProvider] =
      shuffledInputProviderBuilder.getInputProviders();

  // Training is broken up into batches.
  const NUM_BATCHES = 20;
  const BATCH_SIZE = xs.length;
  // Before we start training, we need to provide an optimizer. This is the
  // object that is responsible for updating weights. The learning rate param
  // is a value that represents how large of a step to make when updating
  // weights. If this is too big, you may overstep and oscillate. If it is too
  // small, the model may take a long time to train.
  const LEARNING_RATE = .01;
  const optimizer = new SGDOptimizer(LEARNING_RATE);
  for (let i = 0; i < NUM_BATCHES; i++) {
    // Train takes a cost tensor to minimize; this call trains one batch and
    // returns the average cost of the batch as a Scalar.
    const costValue = session.train(
        cost,
        // Map input providers to Tensors on the graph.
        [{tensor: x, data: xProvider}, {tensor: yLabel, data: yProvider}],
        BATCH_SIZE, optimizer, CostReduction.MEAN);

    console.log('average cost: ' + await costValue.data());
  }

  // Now print the value from the trained model for x = 4, should be ~57.0.
  result = session.eval(y, [{tensor: x, data: Scalar.new(4)}]);
  console.log('result should be ~57.0:');
  console.log(result.shape);
  console.log(await result.data());
});
```

After training the model, you can infer through the graph again to get a
value for "y" given an "x".

Of course, in practice, you will not want to just use `Scalar` values.
**deeplearn.js** provides powerful hardware-accelerated linear algebra which
you can use for everything from image recognition to text generation. See other
[tutorials](index.md) for more!
