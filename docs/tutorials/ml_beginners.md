---
layout: page
order: 2
---
# Training for ML Beginners

In order to get a model to learn to do the things we would like, we have to train it. There are various styles of learning, but this guide will primarily focus on supervised learning. This is where we build a model and give it some examples of inputs and __desired outputs__ (aka labels), we then adjust our model variables to learn to correctly 'map' from these inputs to outputs and hopefully generalize to unseen inputs.

Lets look at a toy example to get a feel for what training a model looks like. Note that this tutorial _does not_ include machine learning best practices that deeplearn.js supports such as batching or shuffling training examples.

In this guide we will build a very simple model to learn the coefficients for a quadratic equation given a set of **existing** `x` and `y` observations. This is the equation we want to fit to our data

```
y = a * x^2 + b * x + c
```

To do this we will use the *optimizer* API. We will set up some variables (or weights) to represent the coefficients (a, b and c), and the optimizer will automatically adjust the values of these coefficients. Let's take a look at the code, it is heavily commented so feel free to read through it slowly.

```js
import * as dl from 'deeplearn';

/**
 * We want to learn the coefficients that give correct solutions to the
 * following quadratic equation:
 *      y = a * x^2 + b * x + c
 * In other words we want to learn values for:
 *      a
 *      b
 *      c
 * Such that this function produces 'desired outputs' for y when provided
 * with x. We will provide some examples of xs and ys to allow this model
 * to learn what we mean by desired outputs and then use it to produce new
 * values of y that fit the curve implied by our example.
 */


// Step 1. Set up variables, these are the things we want the model
// to learn in order to do prediction accurately. We will initialize
// them with random values.
const a = dl.variable(dl.scalar(Math.random()));
const b = dl.variable(dl.scalar(Math.random()));
const c = dl.variable(dl.scalar(Math.random()));


// Step 2. Create an optimizer, we will use this later
const learningRate = 0.01;
const optimizer = dl.train.sgd(learningRate);

// Step 3. Write our training process functions.


/*
 * This function represents our 'model'. Given an input 'x' it will try and predict
 * the appropriate output 'y'.
 *
 * This could be as complicated a 'neural net' as we would like, but we can just
 * directly model the quadratic equation we are trying to model.
 *
 * It is also sometimes referred to as the 'forward' step of our training process.
 * Though we will use the same function for predictions later.
 *
 *
 * @return number predicted y value
 */
function predict(input) {
  // y = a * x ^ 2 + b * x + c
  return dl.tidy(() => {
    const x = dl.scalar(input);

    const ax2 = a.mul(x.square());
    const bx = b.mul(x);
    const y = ax2.add(bx).add(c);

    return y;
  });
}

/*
 * This will tell us how good the 'prediction' is given what we actually expected.
 *
 * prediction is a tensor with our predicted y value.
 * actual number is a number with the y value the model should have predicted.
 */
function loss(prediction, actual) {
  // Having a good error metric is key for training a machine learning model
  const error = dl.scalar(actual).sub(prediction).square();
  return error.asScalar();
}

/*
 * This will iteratively train our model. We test how well it is doing
 * after numIterations by calculating the mean error over all the given
 * samples after our training.
 *
 * xs - training data x values
 * ys — training data y values
 */
async function train(xs, ys, numIterations, done) {
  for (let iter = 0; iter < numIterations; iter++) {
    for (let i = 0; i < xs.length; i++) {
      // Minimize is where the magic happens, we must return a
      // numerical estimate (i.e. loss) of how well we are doing using the
      // current state of the variables we created at the start.

      // This optimizer does the 'backward' step of our training data
      // updating variables defined previously in order to minimize the
      // loss.
      optimizer.minimize(() => {
        // Feed the examples into the model
        const pred = predict(xs[i]);
        const predLoss = loss(pred, ys[i]);

        return predLoss;
      });
    }

    // Use dl.nextFrame to not block the browser.
    await dl.nextFrame();
  }

  done();
}
/*
 * This function compare expected results with the predicted results from
 * our model.
 */
function test(xs, ys) {
  dl.tidy(() => {
    const predictedYs = xs.map(predict);
    console.log('Expected', ys);
    console.log('Got', predictedYs.map((p) => p.dataSync()[0]));
  })
}


const data = {
  xs: [0, 1, 2, 3],
  ys: [1.1, 5.9, 16.8, 33.9]
};

// Lets see how it does before training.
console.log('Before training: using random coefficients')
test(data.xs, data.ys);
train(data.xs, data.ys, 50, () => {
  console.log(
      `After training: a=${a.dataSync()}, b=${b.dataSync()}, c=${c.dataSync()}`)
  test(data.xs, data.ys);
});

// Huzzah we have trained a simple machine learning model!
```


After training the model, you can infer through the graph again to get a
value for "y" given an "x".

This is a very simple model and training setup. Too simple for most challenges we would want to use neural networks for. In practice we would likely not just use `Scalar` variables, but higher-dimensional Tensors, including images.. We would likely want to use techniques like batching our training data, which allows us to run prediction over a number of inputs in parallel, and computing error over a batch. We would also often shuffle our training data as we train, so that the model doesn't learn something dependent on the order of examples. Future tutorials will explore common techniques used in building more robust training pipelines.
