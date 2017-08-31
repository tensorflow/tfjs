/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// tslint:disable-next-line:max-line-length
import {NDArrayMathGPU, Scalar, NDArray, Array1D, Array2D, Graph, Session, Tensor, SGDOptimizer, InCPUMemoryShuffledInputProviderBuilder, CostReduction} from '../deeplearnjs';

// This file parallels (some of) the code in the ML Beginners tutorial.
{
  const math = new NDArrayMathGPU();

  math.scope((keep, track) => {
    const matrixShape: [number, number] = [2, 3];  // 2 rows, 3 columns.
    const matrix = track(Array2D.new(matrixShape, [10, 20, 30, 40, 50, 60]));
    const vector = track(Array1D.new([0, 1, 2]));
    const result = math.matrixTimesVector(matrix, vector);

    console.log("result shape:", result.shape);
    console.log("result", result.getValues());
  });
}

{
  const graph = new Graph();
  // Make a new input in the graph, called 'x', with shape [] (a Scalar).
  const x: Tensor = graph.placeholder('x', []);
  // Make new variables in the graph, 'a', 'b', 'c' with shape [] and random
  // initial values.
  const a: Tensor = graph.variable('a', Scalar.new(Math.random()));
  const b: Tensor = graph.variable('b', Scalar.new(Math.random()));
  const c: Tensor = graph.variable('c', Scalar.new(Math.random()));
  // Make new tensors representing the output of the operations of the
  // quadratic.
  const order2: Tensor = graph.multiply(a, graph.square(x));
  const order1: Tensor = graph.multiply(b, x);
  const y: Tensor = graph.add(graph.add(order2, order1), c);
  
  // When training, we need to provide a label and a cost function.
  const yLabel: Tensor = graph.placeholder('y label', []);
  // Provide a mean squared cost function for training. cost = (y - yLabel)^2
  const cost: Tensor = graph.meanSquaredCost(y, yLabel);

  // At this point the graph is set up, but has not yet been evaluated.
  // **deeplearn.js** needs a Session object to evaluate a graph.
  const math = new NDArrayMathGPU();
  const session = new Session(graph, math);
  
  math.scope((keep, track) => {
    /**
     * Inference
     */
    // Now we ask the graph to evaluate (infer) and give us the result when
    // providing a value 4 for "x".
    // NOTE: "a", "b", and "c" are randomly initialized, so this will give us
    // something random.
    let result: NDArray =
        session.eval(y, [{tensor: x, data: track(Scalar.new(4))}]);
    console.log(result.shape);
    console.log(result.getValues());
  
    /**
     * Training
     */
    // Now let's learn the coefficients of this quadratic given some data.
    // To do this, we need to provide examples of x and y.
    // The values given here are for values a = 3, b = 2, c = 1, with random
    // noise added to the output so it's not a perfect fit.
    const xs: Scalar[] = [
      track(Scalar.new(0)),
      track(Scalar.new(1)),
      track(Scalar.new(2)),
      track(Scalar.new(3))
    ];
    const ys: Scalar[] = [
      track(Scalar.new(1.1)),
      track(Scalar.new(5.9)),
      track(Scalar.new(16.8)),
      track(Scalar.new(33.9))
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
  
      console.log('average cost: ' + costValue.get());
    }

    // Now print the value from the trained model for x = 4, should be ~57.0.
    result = session.eval(y, [{tensor: x, data: track(Scalar.new(4))}]);
    console.log('result should be ~57.0:');
    console.log(result.shape);
    console.log(result.getValues());
  });
}
