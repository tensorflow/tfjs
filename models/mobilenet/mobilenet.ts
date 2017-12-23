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
import {Array1D, Array3D, Array4D, CheckpointLoader, Model, NDArray, NDArrayMath, Scalar} from 'deeplearn';
import * as model_util from '../util';
import {IMAGENET_CLASSES} from './imagenet_classes';

const GOOGLE_CLOUD_STORAGE_DIR =
    'https://storage.googleapis.com/learnjs-data/checkpoint_zoo/';

export class MobileNet implements Model {
  private variables: {[varName: string]: NDArray};

  // yolo variables
  private PREPROCESS_DIVISOR = Scalar.new(255.0 / 2);
  private ONE = Scalar.new(1);

  constructor(private math: NDArrayMath) {}

  /**
   * Loads necessary variables for MobileNet.
   */
  async load(): Promise<void> {
    const checkpointLoader = new CheckpointLoader(
        GOOGLE_CLOUD_STORAGE_DIR + 'mobilenet_v1_1.0_224/');
    this.variables = await checkpointLoader.getAllVariables();
  }

  /**
   * Infer through MobileNet, assumes variables have been loaded. This does
   * standard ImageNet pre-processing before inferring through the model. This
   * method returns named activations as well as pre-softmax logits.
   *
   * @param input un-preprocessed input Array.
   * @return Named activations and the pre-softmax logits.
   */
  predict(input: Array3D): Array1D {
    // Keep a map of named activations for rendering purposes.
    const netout = this.math.scope((keep) => {
      // Preprocess the input.
      const preprocessedInput =
          this.math.subtract(
              this.math.arrayDividedByScalar(input, this.PREPROCESS_DIVISOR),
              this.ONE) as Array3D;

      const x1 = this.convBlock(preprocessedInput, [2, 2]);
      const x2 = this.depthwiseConvBlock(x1, [1, 1], 1);

      const x3 = this.depthwiseConvBlock(x2, [2, 2], 2);
      const x4 = this.depthwiseConvBlock(x3, [1, 1], 3);

      const x5 = this.depthwiseConvBlock(x4, [2, 2], 4);
      const x6 = this.depthwiseConvBlock(x5, [1, 1], 5);

      const x7 = this.depthwiseConvBlock(x6, [2, 2], 6);
      const x8 = this.depthwiseConvBlock(x7, [1, 1], 7);
      const x9 = this.depthwiseConvBlock(x8, [1, 1], 8);
      const x10 = this.depthwiseConvBlock(x9, [1, 1], 9);
      const x11 = this.depthwiseConvBlock(x10, [1, 1], 10);
      const x12 = this.depthwiseConvBlock(x11, [1, 1], 11);

      const x13 = this.depthwiseConvBlock(x12, [2, 2], 12);
      const x14 = this.depthwiseConvBlock(x13, [1, 1], 13);

      const x15 = this.math.avgPool(x14, x14.shape[0], 1, 0);
      const x16 = this.math.conv2d(
          x15,
          this.variables['MobilenetV1/Logits/Conv2d_1c_1x1/weights'] as Array4D,
          this.variables['MobilenetV1/Logits/Conv2d_1c_1x1/biases'] as Array1D,
          1, 'same');

      return x16.as1D();
    });

    return netout;
  }

  private convBlock(inputs: Array3D, strides: [number, number]) {
    const convPadding = 'MobilenetV1/Conv2d_0';

    const x1 = this.math.conv2d(
        inputs, this.variables[convPadding + '/weights'] as Array4D,
        null,  // this convolutional layer does not use bias
        strides, 'same');

    const x2 = this.math.batchNormalization3D(
        x1, this.variables[convPadding + '/BatchNorm/moving_mean'] as Array1D,
        this.variables[convPadding + '/BatchNorm/moving_variance'] as Array1D,
        .001, this.variables[convPadding + '/BatchNorm/gamma'] as Array1D,
        this.variables[convPadding + '/BatchNorm/beta'] as Array1D);

    return this.math.clip(x2, 0, 6);  // simple implementation of Relu6
  }

  private depthwiseConvBlock(
      inputs: Array3D, strides: [number, number], blockID: number) {
    const dwPadding = 'MobilenetV1/Conv2d_' + String(blockID) + '_depthwise';
    const pwPadding = 'MobilenetV1/Conv2d_' + String(blockID) + '_pointwise';

    const x1 =
        this.math.depthwiseConv2D(
            inputs, this.variables[dwPadding + '/depthwise_weights'] as Array4D,
            strides, 'same') as Array3D;

    const x2 = this.math.batchNormalization3D(
        x1, this.variables[dwPadding + '/BatchNorm/moving_mean'] as Array1D,
        this.variables[dwPadding + '/BatchNorm/moving_variance'] as Array1D,
        .001, this.variables[dwPadding + '/BatchNorm/gamma'] as Array1D,
        this.variables[dwPadding + '/BatchNorm/beta'] as Array1D);

    const x3 = this.math.clip(x2, 0, 6);

    const x4 = this.math.conv2d(
        x3, this.variables[pwPadding + '/weights'] as Array4D,
        null,  // this convolutional layer does not use bias
        [1, 1], 'same');

    const x5 = this.math.batchNormalization3D(
        x4, this.variables[pwPadding + '/BatchNorm/moving_mean'] as Array1D,
        this.variables[pwPadding + '/BatchNorm/moving_variance'] as Array1D,
        .001, this.variables[pwPadding + '/BatchNorm/gamma'] as Array1D,
        this.variables[pwPadding + '/BatchNorm/beta'] as Array1D);

    return this.math.clip(x5, 0, 6);
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
    const predictions = this.math.softmax(logits).asType('float32');
    const topk = model_util.topK(await predictions.data(), topK);
    const topkIndices = topk.indices;
    const topkValues = topk.values;

    const topClassesToProbability: {[className: string]: number} = {};
    for (let i = 0; i < topkIndices.length; i++) {
      topClassesToProbability[IMAGENET_CLASSES[topkIndices[i]]] = topkValues[i];
    }
    return topClassesToProbability;
  }

  dispose() {
    for (const varName in this.variables) {
      this.variables[varName].dispose();
    }
  }
}
