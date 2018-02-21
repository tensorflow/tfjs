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
'use strict';

import * as dl from 'deeplearn';

const EPSILON = 1e-7;

const getRandomIntegerInRange = (min: number, max: number) => {
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

export const learnXOR = async () => {
  const iterations = getRandomIntegerInRange(800, 1000);
  const timeStart: number = performance.now();
  let loss: number;
  let cost: dl.Scalar;

  const graph = new dl.Graph();

  const input = graph.placeholder('input', [2]);
  const y = graph.placeholder('y', [1]);

  const hiddenLayer = graph.layers.dense(
      'hiddenLayer', input, 10, (x: dl.SymbolicTensor) => graph.relu(x), true);
  const output = graph.layers.dense(
      'outputLayer', hiddenLayer, 1, (x: dl.SymbolicTensor) => graph.sigmoid(x),
      true);

  const costTensor = graph.reduceSum(graph.add(
      graph.multiply(
          graph.constant([-1]),
          graph.multiply(
              y, graph.log(graph.add(output, graph.constant([EPSILON]))))),
      graph.multiply(
          graph.constant([-1]),
          graph.multiply(
              graph.subtract(graph.constant([1]), y),
              graph.log(graph.add(
                  graph.subtract(graph.constant([1]), output),
                  graph.constant([EPSILON])))))));

  const session = new dl.Session(graph, dl.ENV.math);
  const optimizer = new dl.SGDOptimizer(0.2);

  const inputArray = [
    dl.tensor1d([0, 0]), dl.tensor1d([0, 1]), dl.tensor1d([1, 0]),
    dl.tensor1d([1, 1])
  ];

  const targetArray =
      [dl.tensor1d([0]), dl.tensor1d([1]), dl.tensor1d([1]), dl.tensor1d([0])];

  const shuffledInputProviderBuilder =
      new dl.InCPUMemoryShuffledInputProviderBuilder([inputArray, targetArray]);

  const [inputProvider, targetProvider] =
      shuffledInputProviderBuilder.getInputProviders();

  const feedEntries =
      [{tensor: input, data: inputProvider}, {tensor: y, data: targetProvider}];

  /**
   * Train the model
   */
  for (let i = 0; i < iterations; i += 1) {
    cost = session.train(
        costTensor, feedEntries, 4, optimizer, dl.CostReduction.MEAN);
  }
  loss = await cost.val();

  const result = [];

  /**
   * Test the model
   */
  for (let i = 0; i < 4; i += 1) {
    const inputData = inputArray[i];
    const expectedOutput = targetArray[i];

    const val = session.eval(output, [{tensor: input, data: inputData}]);

    result.push({
      input: await inputData.data(),
      expected: await expectedOutput.data(),
      output: await val.data()
    });
  }

  const timeEnd: number = performance.now();
  const time = timeEnd - timeStart;

  return {iterations, loss, time, result};
};
