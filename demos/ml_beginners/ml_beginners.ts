/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as dl from 'deeplearn';

async function mlBeginners() {
  const math = dl.ENV.math;

  // This file parallels (some of) the code in the ML Beginners tutorial.
  {
    const matrixShape: [number, number] = [2, 3];  // 2 rows, 3 columns.
    const matrix = dl.tensor2d([10, 20, 30, 40, 50, 60], matrixShape);
    const vector = dl.tensor1d([0, 1, 2]);
    const result = dl.matrixTimesVector(matrix, vector);

    console.log('result shape:', result.shape);
    console.log('result', await result.data());
  }

  {
    const g = new dl.Graph();
    // Make a new input in the dl.Graph, called 'x', with shape [] (a
    // dl.Scalar).
    const x = g.placeholder('x', []);
    // Make new variables in the dl.Graph, 'a', 'b', 'c' with shape [] and
    // random initial values.
    const a = g.variable('a', dl.scalar(Math.random()));
    const b = g.variable('b', dl.scalar(Math.random()));
    const c = g.variable('c', dl.scalar(Math.random()));
    // Make new tensors representing the output of the operations of the
    // quadratic.
    const order2 = g.multiply(a, g.square(x));
    const order1 = g.multiply(b, x);
    const y = g.add(g.add(order2, order1), c);

    // When training, we need to provide a label and a cost function.
    const yLabel = g.placeholder('y label', []);
    // Provide a mean squared cost function for training. cost = (y - yLabel)^2
    const cost = g.meanSquaredCost(y, yLabel);

    // At this point the dl.Graph is set up, but has not yet been evaluated.
    // **deeplearn.js** needs a dl.Session object to evaluate a dl.Graph.
    const session = new dl.Session(g, math);

    /**
     * Inference
     */
    // Now we ask the dl.Graph to evaluate (infer) and give us the result when
    // providing a value 4 for "x".
    // NOTE: "a", "b", and "c" are randomly initialized, so this will give us
    // something random.
    let result = session.eval(y, [{tensor: x, data: dl.scalar(4)}]);
    console.log(await result.data());

    /**
     * Training
     */
    // Now let's learn the coefficients of this quadratic given some data.
    // To do this, we need to provide examples of x and y.
    // The values given here are for values a = 3, b = 2, c = 1, with random
    // noise added to the output so it's not a perfect fit.
    const xs = [dl.scalar(0), dl.scalar(1), dl.scalar(2), dl.scalar(3)];
    const ys =
        [dl.scalar(1.1), dl.scalar(5.9), dl.scalar(16.8), dl.scalar(33.9)];
    // When training, it's important to shuffle your data!
    const shuffledInputProviderBuilder =
        new dl.InCPUMemoryShuffledInputProviderBuilder([xs, ys]);
    const [xProvider, yProvider] =
        shuffledInputProviderBuilder.getInputProviders();

    // Training is broken up into batches.
    const NUM_BATCHES = 20;
    const BATCH_SIZE = xs.length;
    // Before we start training, we need to provide an optimizer. This is the
    // object that is responsible for updating weights. The learning rate
    // param is a value that represents how large of a step to make when
    // updating weights. If this is too big, you may overstep and oscillate.
    // If it is too small, the model may take a long time to train.
    const LEARNING_RATE = .01;
    const optimizer = dl.train.sgd(LEARNING_RATE);
    for (let i = 0; i < NUM_BATCHES; i++) {
      // Train takes a cost dl.Tensor to minimize; this call trains one batch
      // and returns the average cost of the batch as a dl.Scalar.
      const costValue = session.train(
          cost,
          // Map input providers to Tensors on the dl.Graph.
          [{tensor: x, data: xProvider}, {tensor: yLabel, data: yProvider}],
          BATCH_SIZE, optimizer, dl.CostReduction.MEAN);

      console.log(`average cost: ${await costValue.data()}`);
    }

    // Now print the value from the trained model for x = 4, should be ~57.0.
    result = session.eval(y, [{tensor: x, data: dl.scalar(4)}]);
    console.log('result should be ~57.0:');
    console.log(await result.data());
  }
}

mlBeginners();
