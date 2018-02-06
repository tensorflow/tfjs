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

// This file parallels (some of) the code in the introduction tutorial.

/**
 * 'NDArrayMath with WebGL backend' section of tutorial
 */
async function intro() {
  const a = dl.tensor2d([1.0, 2.0, 3.0, 4.0], [2, 2]);
  const b = dl.tensor2d([0.0, 2.0, 4.0, 6.0], [2, 2]);

  const size = dl.scalar(a.size);

  // Non-blocking math calls.
  const average = a.sub(b).square().sum().div(size);

  console.log(`mean squared difference: ${await average.val()}`);

  /**
   * 'Graphs and Tensors' section of tutorial
   */

  const g = new dl.Graph();

  // Placeholders are input containers. This is the container for where we
  // will feed an input Tensor when we execute the graph.
  const inputShape = [3];
  const inputTensor = g.placeholder('input', inputShape);

  const labelShape = [1];
  const labelTensor = g.placeholder('label', labelShape);

  // Variables are containers that hold a value that can be updated from
  // training.
  // Here we initialize the multiplier variable randomly.
  const multiplier = g.variable('multiplier', dl.randomNormal([1, 3]));

  // Top level graph methods take Tensors and return Tensors.
  const outputTensor = g.matmul(multiplier, inputTensor);
  const costTensor = g.meanSquaredCost(labelTensor, outputTensor);

  // Tensors, like Tensors, have a shape attribute.
  console.log(outputTensor.shape);

  /**
   * 'dl.Session and dl.FeedEntry' section of the tutorial.
   */

  const learningRate = .00001;
  const batchSize = 3;

  const session = new dl.Session(g, dl.ENV.math);
  const optimizer = new dl.SGDOptimizer(learningRate);

  const inputs: dl.Tensor1D[] = [
    dl.tensor1d([1.0, 2.0, 3.0]), dl.tensor1d([10.0, 20.0, 30.0]),
    dl.tensor1d([100.0, 200.0, 300.0])
  ];

  const labels: dl.Tensor1D[] =
      [dl.tensor1d([4.0]), dl.tensor1d([40.0]), dl.tensor1d([400.0])];

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
    // Wrap session.train in a scope so the cost gets cleaned up
    // automatically.
    await dl.tidy(async () => {
      // Train takes a cost tensor to minimize. Trains one batch. Returns the
      // average cost as a dl.Scalar.
      const cost = session.train(
          costTensor, feedEntries, batchSize, optimizer, dl.CostReduction.MEAN);

      console.log(`last average cost (${i}): ${await cost.val()}`);
    });
  }

  const testInput = dl.tensor1d([0.1, 0.2, 0.3]);

  // session.eval can take Tensors as input data.
  const testFeedEntries: dl.FeedEntry[] =
      [{tensor: inputTensor, data: testInput}];

  const testOutput = session.eval(outputTensor, testFeedEntries);

  console.log('---inference output---');
  console.log(`shape: ${testOutput.shape}`);
  console.log(`value: ${await testOutput.val(0)}`);
}

intro();
