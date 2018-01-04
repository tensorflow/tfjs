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
import {Array1D, CostReduction, ENV, FeedEntry, Graph, InCPUMemoryShuffledInputProviderBuilder, Session, SGDOptimizer, Tensor} from 'deeplearn';

class ComplementaryColorModel {
  // Runs training.
  session: Session;

  // Encapsulates math operations on the CPU and GPU.
  math = ENV.math;

  // An optimizer with a certain learning rate. Used for training.
  learningRate = 0.1;
  optimizer: SGDOptimizer;

  // Each training batch will be on this many examples.
  batchSize = 50;

  inputTensor: Tensor;
  targetTensor: Tensor;
  costTensor: Tensor;
  predictionTensor: Tensor;

  // Maps tensors to InputProviders.
  feedEntries: FeedEntry[];

  constructor() {
    this.optimizer = new SGDOptimizer(this.learningRate);
  }

  /**
   * Constructs the graph of the model. Call this method before training.
   */
  setupSession(): void {
    const graph = new Graph();

    // This tensor contains the input. In this case, it is a scalar.
    this.inputTensor = graph.placeholder('input RGB value', [3]);

    // This tensor contains the target.
    this.targetTensor = graph.placeholder('output RGB value', [3]);

    // Create 3 fully connected layers, each with half the number of nodes of
    // the previous layer. The first one has 64 nodes.
    let fullyConnectedLayer =
        this.createFullyConnectedLayer(graph, this.inputTensor, 0, 64);

    // Create fully connected layer 1, which has 32 nodes.
    fullyConnectedLayer =
        this.createFullyConnectedLayer(graph, fullyConnectedLayer, 1, 32);

    // Create fully connected layer 2, which has 16 nodes.
    fullyConnectedLayer =
        this.createFullyConnectedLayer(graph, fullyConnectedLayer, 2, 16);
    this.predictionTensor =
        this.createFullyConnectedLayer(graph, fullyConnectedLayer, 3, 3);

    // We will optimize using mean squared loss.
    this.costTensor =
        graph.meanSquaredCost(this.targetTensor, this.predictionTensor);

    // Create the session only after constructing the graph.
    this.session = new Session(graph, this.math);

    // Generate the data that will be used to train the model.
    this.generateTrainingData(1e5);
  }

  /**
   * Trains one batch for one iteration. Call this method multiple times to
   * progressively train. Calling this function transfers data from the GPU in
   * order to obtain the current loss on training data.
   *
   * If shouldFetchCost is true, returns the mean cost across examples in the
   * batch. Otherwise, returns -1. We should only retrieve the cost now and then
   * because doing so requires transferring data from the GPU.
   */
  train1Batch(shouldFetchCost: boolean): number {
    // Every 42 steps, lower the learning rate by 15%.
    const learningRate =
        this.learningRate * Math.pow(0.85, Math.floor(step / 42));
    this.optimizer.setLearningRate(learningRate);

    // Train 1 batch.
    let costValue = -1;
    this.math.scope(() => {
      const cost = this.session.train(
          this.costTensor, this.feedEntries, this.batchSize, this.optimizer,
          shouldFetchCost ? CostReduction.MEAN : CostReduction.NONE);

      if (!shouldFetchCost) {
        // We only train. We do not compute the cost.
        return;
      }

      // Compute the cost (by calling get), which requires transferring data
      // from the GPU.
      costValue = cost.get();
    });
    return costValue;
  }

  normalizeColor(rgbColor: number[]): number[] {
    return rgbColor.map(v => v / 255);
  }

  denormalizeColor(normalizedRgbColor: number[]): number[] {
    return normalizedRgbColor.map(v => v * 255);
  }

  predict(rgbColor: number[]): number[] {
    let complementColor: number[] = [];
    this.math.scope(() => {
      const mapping = [{
        tensor: this.inputTensor,
        data: Array1D.new(this.normalizeColor(rgbColor)),
      }];
      const evalOutput = this.session.eval(this.predictionTensor, mapping);
      const values = evalOutput.dataSync();
      const colors = this.denormalizeColor(Array.prototype.slice.call(values));

      // Make sure the values are within range.
      complementColor =
          colors.map(v => Math.round(Math.max(Math.min(v, 255), 0)));
    });
    return complementColor;
  }

  private createFullyConnectedLayer(
      graph: Graph, inputLayer: Tensor, layerIndex: number,
      sizeOfThisLayer: number) {
    return graph.layers.dense(
        `fully_connected_${layerIndex}`, inputLayer, sizeOfThisLayer,
        (x) => graph.relu(x));
  }

  /**
   * Generates data used to train. Creates a feed entry that will later be used
   * to pass data into the model. Generates `exampleCount` data points.
   */
  private generateTrainingData(exampleCount: number) {
    const rawInputs = new Array(exampleCount);

    for (let i = 0; i < exampleCount; i++) {
      rawInputs[i] = [
        this.generateRandomChannelValue(), this.generateRandomChannelValue(),
        this.generateRandomChannelValue()
      ];
    }

    // Store the data within Array1Ds so that learnjs can use it.
    const inputArray: Array1D[] =
        rawInputs.map(c => Array1D.new(this.normalizeColor(c)));
    const targetArray: Array1D[] = rawInputs.map(
        c => Array1D.new(
            this.normalizeColor(this.computeComplementaryColor(c))));

    // This provider will shuffle the training data (and will do so in a way
    // that does not separate the input-target relationship).
    const shuffledInputProviderBuilder =
        new InCPUMemoryShuffledInputProviderBuilder([inputArray, targetArray]);
    const [inputProvider, targetProvider] =
        shuffledInputProviderBuilder.getInputProviders();

    // Maps tensors to InputProviders.
    this.feedEntries = [
      {tensor: this.inputTensor, data: inputProvider},
      {tensor: this.targetTensor, data: targetProvider}
    ];
  }

  private generateRandomChannelValue() {
    return Math.floor(Math.random() * 256);
  }

  /**
   * This implementation of computing the complementary color came from an
   * answer by Edd https://stackoverflow.com/a/37657940
   */
  computeComplementaryColor(rgbColor: number[]): number[] {
    let r = rgbColor[0];
    let g = rgbColor[1];
    let b = rgbColor[2];

    // Convert RGB to HSL
    // Adapted from answer by 0x000f http://stackoverflow.com/a/34946092/4939630
    r /= 255.0;
    g /= 255.0;
    b /= 255.0;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h = (max + min) / 2.0;
    let s = h;
    const l = h;

    if (max === min) {
      h = s = 0;  // achromatic
    } else {
      const d = max - min;
      s = (l > 0.5 ? d / (2.0 - max - min) : d / (max + min));

      if (max === r && g >= b) {
        h = 1.0472 * (g - b) / d;
      } else if (max === r && g < b) {
        h = 1.0472 * (g - b) / d + 6.2832;
      } else if (max === g) {
        h = 1.0472 * (b - r) / d + 2.0944;
      } else if (max === b) {
        h = 1.0472 * (r - g) / d + 4.1888;
      }
    }

    h = h / 6.2832 * 360.0 + 0;

    // Shift hue to opposite side of wheel and convert to [0-1] value
    h += 180;
    if (h > 360) {
      h -= 360;
    }
    h /= 360;

    // Convert h s and l values into r g and b values
    // Adapted from answer by Mohsen http://stackoverflow.com/a/9493060/4939630
    if (s === 0) {
      r = g = b = l;  // achromatic
    } else {
      const hue2rgb = (p: number, q: number, t: number) => {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 1 / 2) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
      };

      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;

      r = hue2rgb(p, q, h + 1 / 3);
      g = hue2rgb(p, q, h);
      b = hue2rgb(p, q, h - 1 / 3);
    }

    return [r, g, b].map(v => Math.round(v * 255));
  }
}

const complementaryColorModel = new ComplementaryColorModel();

// Create the graph of the model.
complementaryColorModel.setupSession();

// On every frame, we train and then maybe update the UI.
let step = 0;
function trainAndMaybeRender() {
  if (step > 4242) {
    // Stop training.
    return;
  }

  // Schedule the next batch to be trained.
  requestAnimationFrame(trainAndMaybeRender);

  // We only fetch the cost every 5 steps because doing so requires a transfer
  // of data from the GPU.
  const localStepsToRun = 5;
  let cost;
  for (let i = 0; i < localStepsToRun; i++) {
    cost = complementaryColorModel.train1Batch(i === localStepsToRun - 1);
    step++;
  }

  // Print data to console so the user can inspect.
  console.log('step', step - 1, 'cost', cost);

  // Visualize the predicted complement.
  const colorRows = document.querySelectorAll('tr[data-original-color]');
  for (let i = 0; i < colorRows.length; i++) {
    const rowElement = colorRows[i];
    const tds = rowElement.querySelectorAll('td');
    const originalColor =
        (rowElement.getAttribute('data-original-color') as string)
            .split(',')
            .map(v => parseInt(v, 10));

    // Visualize the predicted color.
    const predictedColor = complementaryColorModel.predict(originalColor);
    populateContainerWithColor(
        tds[2], predictedColor[0], predictedColor[1], predictedColor[2]);
  }
}

function populateContainerWithColor(
    container: HTMLElement, r: number, g: number, b: number) {
  const originalColorString = 'rgb(' + [r, g, b].join(',') + ')';
  container.textContent = originalColorString;

  const colorBox = document.createElement('div');
  colorBox.classList.add('color-box');
  colorBox.style.background = originalColorString;
  container.appendChild(colorBox);
}

function initializeUi() {
  const colorRows = document.querySelectorAll('tr[data-original-color]');
  for (let i = 0; i < colorRows.length; i++) {
    const rowElement = colorRows[i];
    const tds = rowElement.querySelectorAll('td');
    const originalColor =
        (rowElement.getAttribute('data-original-color') as string)
            .split(',')
            .map(v => parseInt(v, 10));

    // Visualize the original color.
    populateContainerWithColor(
        tds[0], originalColor[0], originalColor[1], originalColor[2]);

    // Visualize the complementary color.
    const complement =
        complementaryColorModel.computeComplementaryColor(originalColor);
    populateContainerWithColor(
        tds[1], complement[0], complement[1], complement[2]);
  }
}

// Kick off training.
initializeUi();
trainAndMaybeRender();
