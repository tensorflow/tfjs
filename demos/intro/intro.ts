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

// tslint:disable-next-line:max-line-length
import {Array1D, Array2D, CostReduction, ENV, FeedEntry, Graph, InCPUMemoryShuffledInputProviderBuilder, Scalar, Session, SGDOptimizer} from 'deeplearn';

// This file parallels (some of) the code in the introduction tutorial.

/**
 * 'NDArrayMath with WebGL backend' section of tutorial
 */
async function intro() {
  const math = ENV.math;

  const a = Array2D.new([2, 2], [1.0, 2.0, 3.0, 4.0]);
  const b = Array2D.new([2, 2], [0.0, 2.0, 4.0, 6.0]);

  // Non-blocking math calls.
  const diff = math.sub(a, b);
  const squaredDiff = math.elementWiseMul(diff, diff);
  const sum = math.sum(squaredDiff);
  const size = Scalar.new(a.size);
  const average = math.divide(sum, size);

  console.log(`mean squared difference: ${await average.val()}`);

  /**
   * 'Graphs and Tensors' section of tutorial
   */

  const g = new Graph();

  // Placeholders are input containers. This is the container for where we
  // will feed an input NDArray when we execute the graph.
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
  const costTensor = g.meanSquaredCost(labelTensor, outputTensor);

  // Tensors, like NDArrays, have a shape attribute.
  console.log(outputTensor.shape);

  /**
   * 'Session and FeedEntry' section of the tutorial.
   */

  const learningRate = .00001;
  const batchSize = 3;

  const session = new Session(g, math);
  const optimizer = new SGDOptimizer(learningRate);

  const inputs: Array1D[] = [
    Array1D.new([1.0, 2.0, 3.0]), Array1D.new([10.0, 20.0, 30.0]),
    Array1D.new([100.0, 200.0, 300.0])
  ];

  const labels: Array1D[] =
      [Array1D.new([4.0]), Array1D.new([40.0]), Array1D.new([400.0])];

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
    // Wrap session.train in a scope so the cost gets cleaned up
    // automatically.
    await math.scope(async () => {
      // Train takes a cost tensor to minimize. Trains one batch. Returns the
      // average cost as a Scalar.
      const cost = session.train(
          costTensor, feedEntries, batchSize, optimizer, CostReduction.MEAN);

      console.log(`last average cost (${i}): ${await cost.val()}`);
    });
  }

  const testInput = Array1D.new([0.1, 0.2, 0.3]);

  // session.eval can take NDArrays as input data.
  const testFeedEntries: FeedEntry[] = [{tensor: inputTensor, data: testInput}];

  const testOutput = session.eval(outputTensor, testFeedEntries);

  console.log('---inference output---');
  console.log(`shape: ${testOutput.shape}`);
  console.log(`value: ${await testOutput.val(0)}`);
}

intro();
