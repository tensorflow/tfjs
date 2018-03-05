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
import * as model_util from '../util';
import {IMAGENET_CLASSES} from './imagenet_classes';

const GOOGLE_CLOUD_STORAGE_DIR =
    'https://storage.googleapis.com/learnjs-data/checkpoint_zoo/';

export class MobileNet {
  private variables: {[varName: string]: dl.Tensor};

  // yolo variables
  private PREPROCESS_DIVISOR = dl.scalar(255.0 / 2);
  private ONE = dl.scalar(1);

  /**
   * Loads necessary variables for MobileNet.
   */
  async load(): Promise<void> {
    const checkpointLoader = new dl.CheckpointLoader(
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
  predict(input: dl.Tensor3D): dl.Tensor1D {
    return dl.tidy(() => {
      // Normalize the pixels [0, 255] to be between [-1, 1].
      const preprocessedInput =
          input.div(this.PREPROCESS_DIVISOR).sub(this.ONE) as dl.Tensor3D;
      const x1 = this.convBlock(preprocessedInput, 2);
      const x2 = this.depthwiseConvBlock(x1, 1, 1);

      const x3 = this.depthwiseConvBlock(x2, 2, 2);
      const x4 = this.depthwiseConvBlock(x3, 1, 3);

      const x5 = this.depthwiseConvBlock(x4, 2, 4);
      const x6 = this.depthwiseConvBlock(x5, 1, 5);

      const x7 = this.depthwiseConvBlock(x6, 2, 6);
      const x8 = this.depthwiseConvBlock(x7, 1, 7);
      const x9 = this.depthwiseConvBlock(x8, 1, 8);
      const x10 = this.depthwiseConvBlock(x9, 1, 9);
      const x11 = this.depthwiseConvBlock(x10, 1, 10);
      const x12 = this.depthwiseConvBlock(x11, 1, 11);

      const x13 = this.depthwiseConvBlock(x12, 2, 12);
      const x14 = this.depthwiseConvBlock(x13, 1, 13);

      const x15 = x14.avgPool(x14.shape[0], 1, 0);
      const x16Filter =
          this.variables['MobilenetV1/Logits/Conv2d_1c_1x1/weights'] as
          dl.Tensor4D;
      const x16Bias =
          this.variables['MobilenetV1/Logits/Conv2d_1c_1x1/biases'] as
          dl.Tensor1D;
      const x16 = x15.conv2d(x16Filter, 1, 'same').add(x16Bias);
      return x16.as1D();
    });
  }

  private convBlock(inputs: dl.Tensor3D, stride: number) {
    const convPadding = 'MobilenetV1/Conv2d_0';

    const x1 = inputs.conv2d(
        this.variables[convPadding + '/weights'] as dl.Tensor4D, stride,
        'same');
    const x2 = x1.batchNormalization(
        this.variables[convPadding + '/BatchNorm/moving_mean'] as dl.Tensor1D,
        this.variables[convPadding + '/BatchNorm/moving_variance'] as
            dl.Tensor1D,
        .001, this.variables[convPadding + '/BatchNorm/gamma'] as dl.Tensor1D,
        this.variables[convPadding + '/BatchNorm/beta'] as dl.Tensor1D);
    const res = x2.clipByValue(0, 6);  // simple implementation of Relu6
    return res;
  }

  private depthwiseConvBlock(
      inputs: dl.Tensor3D, stride: number, blockID: number) {
    const dwPadding = 'MobilenetV1/Conv2d_' + String(blockID) + '_depthwise';
    const pwPadding = 'MobilenetV1/Conv2d_' + String(blockID) + '_pointwise';

    const x1 =
        inputs.depthwiseConv2D(
            this.variables[dwPadding + '/depthwise_weights'] as dl.Tensor4D,
            stride, 'same') as dl.Tensor3D;

    const x2 = x1.batchNormalization(
        this.variables[dwPadding + '/BatchNorm/moving_mean'] as dl.Tensor1D,
        this.variables[dwPadding + '/BatchNorm/moving_variance'] as dl.Tensor1D,
        .001, this.variables[dwPadding + '/BatchNorm/gamma'] as dl.Tensor1D,
        this.variables[dwPadding + '/BatchNorm/beta'] as dl.Tensor1D);

    const x3 = x2.clipByValue(0, 6);

    const x4 = x3.conv2d(
        this.variables[pwPadding + '/weights'] as dl.Tensor4D, [1, 1], 'same');

    const x5 = x4.batchNormalization(
        this.variables[pwPadding + '/BatchNorm/moving_mean'] as dl.Tensor1D,
        this.variables[pwPadding + '/BatchNorm/moving_variance'] as dl.Tensor1D,
        .001, this.variables[pwPadding + '/BatchNorm/gamma'] as dl.Tensor1D,
        this.variables[pwPadding + '/BatchNorm/beta'] as dl.Tensor1D);

    return x5.clipByValue(0, 6);
  }

  /**
   * Get the topK classes for pre-softmax logits. Returns a map of className
   * to softmax normalized probability.
   *
   * @param logits Pre-softmax logits array.
   * @param topK How many top classes to return.
   */
  async getTopKClasses(logits: dl.Tensor1D, topK: number):
      Promise<{[className: string]: number}> {
    const predictions = logits.softmax().asType('float32');
    const topk =
        model_util.topK(await predictions.data() as Float32Array, topK);
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
