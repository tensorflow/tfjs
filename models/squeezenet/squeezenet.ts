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
import {Array1D, Array3D, Array4D, CheckpointLoader, Model, NDArray, NDArrayMath} from 'deeplearn';
import * as model_util from '../util';
import {IMAGENET_CLASSES} from './imagenet_classes';

const GOOGLE_CLOUD_STORAGE_DIR =
    'https://storage.googleapis.com/learnjs-data/checkpoint_zoo/';

export type ActivationName = 'conv_1'|'maxpool_1'|'fire2'|'fire3'|'maxpool_2'|
    'fire4'|'fire5'|'maxpool_3'|'fire6'|'fire7'|'fire8'|'fire9'|'conv10';

export class SqueezeNet implements Model {
  private variables: {[varName: string]: NDArray};

  private preprocessOffset = Array1D.new([103.939, 116.779, 123.68]);

  constructor(private math: NDArrayMath) {}

  /**
   * Loads necessary variables for SqueezeNet.
   */
  async load(): Promise<void> {
    const checkpointLoader =
        new CheckpointLoader(GOOGLE_CLOUD_STORAGE_DIR + 'squeezenet1_1/');
    this.variables = await checkpointLoader.getAllVariables();
  }

  /**
   * Infer through SqueezeNet, assumes variables have been loaded. This does
   * standard ImageNet pre-processing before inferring through the model. This
   * method returns named activations as well as pre-softmax logits.
   *
   * @param input un-preprocessed input Array.
   * @return The pre-softmax logits.
   */
  predict(input: Array3D): Array1D {
    return this.predictWithActivation(input).logits;
  }

  /**
   * Infer through SqueezeNet, assumes variables have been loaded. This does
   * standard ImageNet pre-processing before inferring through the model. This
   * method returns named activations as well as pre-softmax logits.
   *
   * @param input un-preprocessed input Array.
   * @return A requested activation and the pre-softmax logits.
   */
  predictWithActivation(input: Array3D, activationName?: ActivationName):
      {logits: Array1D, activation: Array3D} {
    const [logits, activation] = this.math.scope(() => {
      let activation: Array3D;
      // Preprocess the input.
      const preprocessedInput =
          this.math.subtract(input.asType('float32'), this.preprocessOffset) as
          Array3D;
      const conv1 = this.math.conv2d(
          preprocessedInput, this.variables['conv1_W:0'] as Array4D,
          this.variables['conv1_b:0'] as Array1D, 2, 0);
      const conv1relu = this.math.relu(conv1);
      if (activationName === 'conv_1') {
        activation = conv1relu;
      }

      const pool1 = this.math.maxPool(conv1relu, 3, 2, 0);
      if (activationName === 'maxpool_1') {
        activation = pool1;
      }

      const fire2 = this.fireModule(pool1, 2);
      if (activationName === 'fire2') {
        activation = fire2;
      }

      const fire3 = this.fireModule(fire2, 3);
      if (activationName === 'fire3') {
        activation = fire3;
      }

      const pool2 = this.math.maxPool(fire3, 3, 2, 'valid');
      if (activationName === 'maxpool_2') {
        activation = pool2;
      }

      const fire4 = this.fireModule(pool2, 4);
      if (activationName === 'fire4') {
        activation = fire4;
      }

      const fire5 = this.fireModule(fire4, 5);
      if (activationName === 'fire5') {
        activation = fire5;
      }

      const pool3 = this.math.maxPool(fire5, 3, 2, 0);
      if (activationName === 'maxpool_3') {
        activation = pool3;
      }

      const fire6 = this.fireModule(pool3, 6);
      if (activationName === 'fire6') {
        activation = fire6;
      }

      const fire7 = this.fireModule(fire6, 7);
      if (activationName === 'fire7') {
        activation = fire7;
      }

      const fire8 = this.fireModule(fire7, 8);
      if (activationName === 'fire8') {
        activation = fire8;
      }

      const fire9 = this.fireModule(fire8, 9);
      if (activationName === 'fire9') {
        activation = fire9;
      }

      const conv10 = this.math.conv2d(
          fire9, this.variables['conv10_W:0'] as Array4D,
          this.variables['conv10_b:0'] as Array1D, 1, 0);
      if (activationName === 'conv10') {
        activation = conv10;
      }
      return [
        this.math.avgPool(conv10, conv10.shape[0], 1, 0).as1D(), activation
      ];
    });
    return {activation: activation as Array3D, logits: logits as Array1D};
  }

  private fireModule(input: Array3D, fireId: number) {
    const y1 = this.math.conv2d(
        input, this.variables[`fire${fireId}/squeeze1x1_W:0`] as Array4D,
        this.variables[`fire${fireId}/squeeze1x1_b:0`] as Array1D, 1, 0);
    const y2 = this.math.relu(y1);
    const left1 = this.math.conv2d(
        y2, this.variables[`fire${fireId}/expand1x1_W:0`] as Array4D,
        this.variables[`fire${fireId}/expand1x1_b:0`] as Array1D, 1, 0);
    const left2 = this.math.relu(left1);

    const right1 = this.math.conv2d(
        y2, this.variables[`fire${fireId}/expand3x3_W:0`] as Array4D,
        this.variables[`fire${fireId}/expand3x3_b:0`] as Array1D, 1, 1);
    const right2 = this.math.relu(right1);

    return this.math.concat3D(left2, right2, 2);
  }

  /**
   * Get the topK classes for pre-softmax logits. Returns a map of className
   * to softmax normalized probability.
   *
   * @param logits Pre-softmax logits array.
   * @param topK How many top classes to return.
   */
  async getTopKClasses(logits: Array1D, topK: number):
      Promise<{[className: string]: number}> {
    const predictions = this.math.scope(() => {
      return this.math.softmax(logits).asType('float32');
    });
    const topk = model_util.topK(await predictions.data(), topK);
    predictions.dispose();
    const topkIndices = topk.indices;
    const topkValues = topk.values;

    const topClassesToProbability: {[className: string]: number} = {};
    for (let i = 0; i < topkIndices.length; i++) {
      topClassesToProbability[IMAGENET_CLASSES[topkIndices[i]]] = topkValues[i];
    }
    return topClassesToProbability;
  }

  dispose() {
    this.preprocessOffset.dispose();
    for (const varName in this.variables) {
      this.variables[varName].dispose();
    }
  }
}
