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

// tslint:disable:restrict-plus-operands
// tslint:disable-next-line:max-line-length
import {AdagradOptimizer, Array2D, CostReduction, FeedEntry, Graph, InGPUMemoryShuffledInputProviderBuilder, NDArray, NDArrayMath, Session, Tensor} from 'deeplearn';

/** Generates GameOfLife sequence pairs (current sequence + next sequence) */
export class GameOfLife {
  math: NDArrayMath;
  size: number;

  constructor(size: number, math: NDArrayMath) {
    this.math = math;
    this.size = size;
  }

  setSize(size: number) {
    this.size = size;
  }

  generateGolExample(): [NDArray, NDArray] {
    let world: NDArray;
    let worldNext: NDArray;
    this.math.scope(keep => {
      const randWorld =
          Array2D.randUniform([this.size - 2, this.size - 2], 0, 2, 'int32');
      const worldPadded = GameOfLife.padArray(randWorld);
      // TODO(kreeger): This logic can be vectorized and kept on the GPU with a
      // logical_or() and where() implementations.
      const numNeighbors =
          this.countNeighbors(this.size, worldPadded).dataSync();
      const worldValues = randWorld.dataSync();
      const nextWorldValues = [];
      for (let i = 0; i < numNeighbors.length; i++) {
        const value = numNeighbors[i];
        let nextVal = 0;
        if (value === 3) {
          // Cell rebirths
          nextVal = 1;
        } else if (value === 2) {
          // Cell survives
          nextVal = worldValues[i];
        } else {
          // Cell dies
          nextVal = 0;
        }
        nextWorldValues.push(nextVal);
      }
      world = keep(worldPadded);
      worldNext = keep(GameOfLife.padArray(
          Array2D.new(randWorld.shape, nextWorldValues, 'int32')));
    });
    return [world, worldNext];
  }

  /** Counts total sum of neighbors for a given world. */
  private countNeighbors(size: number, worldPadded: Array2D): Array2D {
    let neighborCount = this.math.add(
        this.math.slice2D(worldPadded, [0, 0], [size - 2, size - 2]),
        this.math.slice2D(worldPadded, [0, 1], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount,
        this.math.slice2D(worldPadded, [0, 2], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount,
        this.math.slice2D(worldPadded, [1, 0], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount,
        this.math.slice2D(worldPadded, [1, 2], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount,
        this.math.slice2D(worldPadded, [2, 0], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount,
        this.math.slice2D(worldPadded, [2, 1], [size - 2, size - 2]));
    neighborCount = this.math.add(
        neighborCount,
        this.math.slice2D(worldPadded, [2, 2], [size - 2, size - 2]));
    return neighborCount as Array2D;
  }

  /* Helper method to pad an array until the op is ready. */
  // TODO(kreeger, #409): Drop this when math.pad() is ready.
  private static padArray(array: NDArray): Array2D<'int32'> {
    const x1 = array.shape[0];
    const x2 = array.shape[1];
    const pad = 1;

    const oldValues = array.dataSync();
    const shape = [x1 + pad * 2, x2 + pad * 2];
    const values = [];

    let z = 0;
    for (let i = 0; i < shape[0]; i++) {
      let rangeStart = -1;
      let rangeEnd = -1;
      if (i > 0 && i < shape[0] - 1) {
        rangeStart = i * shape[1] + 1;
        rangeEnd = i * shape[1] + x2;
      }
      for (let j = 0; j < shape[1]; j++) {
        const v = i * shape[0] + j;
        if (v >= rangeStart && v <= rangeEnd) {
          values[v] = oldValues[z++];
        } else {
          values[v] = 0;
        }
      }
    }
    return Array2D.new(shape as [number, number], values, 'int32');
  }
}

/**
 * Main class for running a deep-neural network of training for Game-of-life
 * next sequence.
 */
export class GameOfLifeModel {
  session: Session;
  math: NDArrayMath;

  optimizer: AdagradOptimizer;
  inputTensor: Tensor;
  targetTensor: Tensor;
  costTensor: Tensor;
  predictionTensor: Tensor;

  size: number;
  batchSize: number;
  step = 0;

  // Maps tensors to InputProviders
  feedEntries: FeedEntry[];

  constructor(math: NDArrayMath) {
    this.math = math;
  }

  setupSession(
      boardSize: number, batchSize: number, initialLearningRate: number,
      numLayers: number, useLogCost: boolean): void {
    this.optimizer = new AdagradOptimizer(initialLearningRate);

    this.size = boardSize;
    this.batchSize = batchSize;
    const graph = new Graph();
    const shape = this.size * this.size;

    this.inputTensor = graph.placeholder('input', [shape]);
    this.targetTensor = graph.placeholder('target', [shape]);

    let hiddenLayer = GameOfLifeModel.createFullyConnectedLayer(
        graph, this.inputTensor, 0, shape);
    for (let i = 1; i < numLayers; i++) {
      // Last layer will use a sigmoid:
      hiddenLayer = GameOfLifeModel.createFullyConnectedLayer(
          graph, hiddenLayer, i, shape, i < numLayers - 1);
    }

    this.predictionTensor = hiddenLayer;

    if (useLogCost) {
      this.costTensor =
          this.logLoss(graph, this.targetTensor, this.predictionTensor);
    } else {
      this.costTensor =
          graph.meanSquaredCost(this.targetTensor, this.predictionTensor);
    }
    this.session = new Session(graph, this.math);
  }

  trainBatch(fetchCost: boolean, worlds: Array<[NDArray, NDArray]>): number {
    this.setTrainingData(worlds);

    let costValue = -1;
    this.math.scope(() => {
      const cost = this.session.train(
          this.costTensor, this.feedEntries, this.batchSize, this.optimizer,
          fetchCost ? CostReduction.MEAN : CostReduction.NONE);
      costValue = cost.get();
    });
    return costValue;
  }

  predict(world: NDArray): Array2D {
    let values = null;
    this.math.scope(() => {
      const mapping =
          [{tensor: this.inputTensor, data: world.flatten().asType('float32')}];

      const evalOutput = this.session.eval(this.predictionTensor, mapping);
      values = evalOutput.dataSync();
    });
    return Array2D.new([this.size, this.size], values);
  }

  private setTrainingData(worlds: Array<[NDArray, NDArray]>): void {
    const inputs = [];
    const outputs = [];
    for (let i = 0; i < worlds.length; i++) {
      const example = worlds[i];
      inputs.push(example[0].flatten().asType('float32'));
      outputs.push(example[1].flatten().asType('float32'));
    }

    // TODO(kreeger): Don't really need to shuffle these.
    const inputProviderBuilder =
        new InGPUMemoryShuffledInputProviderBuilder([inputs, outputs]);
    const [inputProvider, targetProvider] =
        inputProviderBuilder.getInputProviders();

    this.feedEntries = [
      {tensor: this.inputTensor, data: inputProvider},
      {tensor: this.targetTensor, data: targetProvider}
    ];
  }

  /* Helper method for creating a fully connected layer. */
  private static createFullyConnectedLayer(
      graph: Graph, inputLayer: Tensor, layerIndex: number,
      sizeOfThisLayer: number, includeRelu = true, includeBias = true): Tensor {
    return graph.layers.dense(
        'fully_connected_' + layerIndex, inputLayer, sizeOfThisLayer,
        includeRelu ? (x) => graph.relu(x) : (x) => graph.sigmoid(x),
        includeBias);
  }

  /* Helper method for calculating loss. */
  private logLoss(graph: Graph, labelTensor: Tensor, predictionTensor: Tensor):
      Tensor {
    const epsilon = graph.constant(1e-7);
    const one = graph.constant(1);
    const negOne = graph.constant(-1);
    const predictionsPlusEps = graph.add(predictionTensor, epsilon);

    const left = graph.multiply(
        negOne, graph.multiply(labelTensor, graph.log(predictionsPlusEps)));
    const right = graph.multiply(
        graph.subtract(one, labelTensor),
        graph.log(graph.add(graph.subtract(one, predictionTensor), epsilon)));

    const losses = graph.subtract(left, right);
    const totalLosses = graph.reduceSum(losses);
    return graph.reshape(
        graph.divide(totalLosses, graph.constant(labelTensor.shape)), []);
  }
}
