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
import * as nn_art_util from './nn_art_util';

const MAX_LAYERS = 10;
const math = dl.ENV.math;

export type ActivationFunction = 'tanh'|'sin'|'relu'|'step';
const activationFunctionMap: {
  [activationFunction in ActivationFunction]: (ndarray: dl.Array2D) =>
      dl.Array2D
} = {
  'tanh': (x: dl.Array2D) => math.tanh(x),
  'sin': (x: dl.Array2D) => math.sin(x),
  'relu': (x: dl.Array2D) => math.relu(x),
  'step': (x: dl.Array2D) => math.step(x)
};

const NUM_IMAGE_SPACE_VARIABLES = 3;  // x, y, r
const NUM_LATENT_VARIABLES = 2;

export class CPPN {
  private inputAtlas: dl.Array2D;
  private ones: dl.Array2D<'float32'>;

  private firstLayerWeights: dl.Array2D;
  private intermediateWeights: dl.Array2D[] = [];
  private lastLayerWeights: dl.Array2D;

  private z1Counter = 0;
  private z2Counter = 0;
  private z1Scale: number;
  private z2Scale: number;
  private numLayers: number;

  private selectedActivationFunctionName: ActivationFunction;

  private isInferring = false;

  constructor(private inferenceCanvas: HTMLCanvasElement) {
    const canvasSize = 128;
    this.inferenceCanvas.width = canvasSize;
    this.inferenceCanvas.height = canvasSize;

    this.inputAtlas = nn_art_util.createInputAtlas(
        canvasSize, NUM_IMAGE_SPACE_VARIABLES, NUM_LATENT_VARIABLES);
    this.ones = dl.Array2D.ones([this.inputAtlas.shape[0], 1]);
  }

  generateWeights(neuronsPerLayer: number, weightsStdev: number) {
    for (let i = 0; i < this.intermediateWeights.length; i++) {
      this.intermediateWeights[i].dispose();
    }
    this.intermediateWeights = [];
    if (this.firstLayerWeights != null) {
      this.firstLayerWeights.dispose();
    }
    if (this.lastLayerWeights != null) {
      this.lastLayerWeights.dispose();
    }

    this.firstLayerWeights = dl.Array2D.randTruncatedNormal(
        [NUM_IMAGE_SPACE_VARIABLES + NUM_LATENT_VARIABLES, neuronsPerLayer], 0,
        weightsStdev);
    for (let i = 0; i < MAX_LAYERS; i++) {
      this.intermediateWeights.push(dl.Array2D.randTruncatedNormal(
          [neuronsPerLayer, neuronsPerLayer], 0, weightsStdev));
    }
    this.lastLayerWeights = dl.Array2D.randTruncatedNormal(
        [neuronsPerLayer, 3 /** max output channels */], 0, weightsStdev);
  }

  setActivationFunction(activationFunction: ActivationFunction) {
    this.selectedActivationFunctionName = activationFunction;
  }

  setNumLayers(numLayers: number) {
    this.numLayers = numLayers;
  }

  setZ1Scale(z1Scale: number) {
    this.z1Scale = z1Scale;
  }

  setZ2Scale(z2Scale: number) {
    this.z2Scale = z2Scale;
  }

  start() {
    this.isInferring = true;
    this.runInferenceLoop();
  }

  private async runInferenceLoop() {
    if (!this.isInferring) {
      return;
    }

    this.z1Counter += 1 / this.z1Scale;
    this.z2Counter += 1 / this.z2Scale;

    const lastOutput = math.scope(() => {
      const z1 = dl.Scalar.new(Math.sin(this.z1Counter));
      const z2 = dl.Scalar.new(Math.cos(this.z2Counter));

      const concatAxis = 1;
      const latentVars = math.concat2D(
          math.multiply(z1, this.ones) as dl.Array2D,
          math.multiply(z2, this.ones) as dl.Array2D, concatAxis);

      const activation = (x: dl.Array2D) =>
          activationFunctionMap[this.selectedActivationFunctionName](x);

      let lastOutput: dl.NDArray =
          math.concat2D(this.inputAtlas, latentVars, concatAxis);
      lastOutput = activation(
          math.matMul(lastOutput as dl.Array2D, this.firstLayerWeights));

      for (let i = 0; i < this.numLayers; i++) {
        const matmulResult =
            math.matMul(lastOutput as dl.Array2D, this.intermediateWeights[i]);

        lastOutput = activation(matmulResult);
      }

      return math
          .sigmoid(math.matMul(lastOutput as dl.Array2D, this.lastLayerWeights))
          .reshape(
              [this.inferenceCanvas.height, this.inferenceCanvas.width, 3]);
    });

    await renderToCanvas(lastOutput as dl.Array3D, this.inferenceCanvas);
    await dl.util.nextFrame();
    this.runInferenceLoop();
  }

  stopInferenceLoop() {
    this.isInferring = false;
  }
}

// TODO(nsthorat): Move this to a core library util.
async function renderToCanvas(a: dl.Array3D, canvas: HTMLCanvasElement) {
  const [height, width, ] = a.shape;
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = await a.data();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    const k = i * 3;
    imageData.data[j + 0] = Math.round(255 * data[k + 0]);
    imageData.data[j + 1] = Math.round(255 * data[k + 1]);
    imageData.data[j + 2] = Math.round(255 * data[k + 2]);
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}
