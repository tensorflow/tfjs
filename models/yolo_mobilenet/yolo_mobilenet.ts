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
import * as dl from 'deeplearn';
import {Tensor1D, Tensor3D, Tensor4D} from 'deeplearn';

import {BoundingBox} from './mobilenet_utils';

const GOOGLE_CLOUD_STORAGE_DIR =
    'https://storage.googleapis.com/learnjs-data/checkpoint_zoo/';

export class YoloMobileNetDetection implements dl.Model {
  private variables: {[varName: string]: dl.Tensor};

  // yolo variables
  private PREPROCESS_DIVISOR = dl.scalar(255.0 / 2);
  private ONE = dl.scalar(1);
  private THRESHOLD = 0.3;
  private THRESHOLD_SCALAR = dl.scalar(this.THRESHOLD);
  private ANCHORS: number[] = [
    0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778,
    9.77052, 9.16828
  ];

  /**
   * Loads necessary variables for MobileNet.
   */
  async load(): Promise<void> {
    const checkpointLoader = new dl.CheckpointLoader(
        GOOGLE_CLOUD_STORAGE_DIR + 'yolo_mobilenet_v1_1.0_416/');
    this.variables = await checkpointLoader.getAllVariables();
  }

  /**
   * Infer through MobileNet, assumes variables have been loaded. This does
   * standard ImageNet pre-processing before inferring through the model. This
   * method returns named activations as well as pre-softmax logits.
   *
   * @param input un-preprocessed input Tensor.
   * @return Named activations and the pre-softmax logits.
   */
  predict(input: Tensor3D): Tensor4D {
    return dl.tidy(() => {
      // Preprocess the input.
      const preprocessedInput =
          input.div(this.PREPROCESS_DIVISOR).sub(this.ONE) as Tensor3D;

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

      const x15 =
          x14.conv2d(
                 this.variables['conv_23/kernel'] as Tensor4D, [1, 1], 'same')
              .add(this.variables['conv_23/bias'] as Tensor1D);

      return x15.as4D(13, 13, 5, 6);
    });
  }

  private convBlock(inputs: Tensor3D, strides: [number, number]) {
    const x1 = inputs.conv2d(
        this.variables['conv1/kernel'] as Tensor4D, strides, 'same');

    const x2 = x1.batchNormalization(
        this.variables['conv1_bn/moving_mean'] as Tensor1D,
        this.variables['conv1_bn/moving_variance'] as Tensor1D, .001,
        this.variables['conv1_bn/gamma'] as Tensor1D,
        this.variables['conv1_bn/beta'] as Tensor1D);

    return x2.clipByValue(0, 6);  // simple implementation of Relu6
  }

  private depthwiseConvBlock(
      inputs: Tensor3D, strides: [number, number], blockID: number) {
    const dwPadding = 'conv_dw_' + String(blockID) + '';
    const pwPadding = 'conv_pw_' + String(blockID) + '';

    const x1 = dl.depthwiseConv2d(
        inputs, this.variables[dwPadding + '/depthwise_kernel'] as Tensor4D,
        strides, 'same');

    const x2 = x1.batchNormalization(
        this.variables[dwPadding + '_bn/moving_mean'] as Tensor1D,
        this.variables[dwPadding + '_bn/moving_variance'] as Tensor1D, .001,
        this.variables[dwPadding + '_bn/gamma'] as Tensor1D,
        this.variables[dwPadding + '_bn/beta'] as Tensor1D);

    const x3 = x2.clipByValue(0, 6);

    const x4 = x3.conv2d(
        this.variables[pwPadding + '/kernel'] as Tensor4D, [1, 1], 'same');

    const x5 = x4.batchNormalization(
        this.variables[pwPadding + '_bn/moving_mean'] as Tensor1D,
        this.variables[pwPadding + '_bn/moving_variance'] as Tensor1D, .001,
        this.variables[pwPadding + '_bn/gamma'] as Tensor1D,
        this.variables[pwPadding + '_bn/beta'] as Tensor1D);

    return x5.clipByValue(0, 6);
  }

  async interpretNetout(netout: Tensor4D): Promise<BoundingBox[]> {
    // interpret the output by the network
    const GRID_H = netout.shape[0];
    const GRID_W = netout.shape[1];
    const BOX = netout.shape[2];
    const CLASS = netout.shape[3] - 5;
    const boxes: BoundingBox[] = [];

    // adjust confidence predictions
    const confidence =
        netout.slice([0, 0, 0, 4], [GRID_H, GRID_W, BOX, 1]).sigmoid();

    // adjust class prediction
    let classes = netout.slice([0, 0, 0, 5], [GRID_H, GRID_W, BOX, CLASS])
                      .softmax()
                      .mul(confidence);

    const mask = classes.sub(this.THRESHOLD_SCALAR).relu().step();
    classes = classes.mul(mask);
    mask.dispose();

    const objectLikelihood = classes.sum(3);

    const objectLikelihoodValues = await objectLikelihood.data();

    for (let i = 0; i < objectLikelihoodValues.length; i++) {
      if (objectLikelihoodValues[i] > 0) {
        const [row, col, box] = objectLikelihood.indexToLoc(i) as number[];

        const conf = confidence.get(row, col, box, 0);
        const probs =
            await classes.slice([row, col, box, 0], [1, 1, 1, CLASS]).data();
        const xywh =
            await netout.slice([row, col, box, 0], [1, 1, 1, 4]).data();

        let x = xywh[0];
        let y = xywh[1];
        let w = xywh[2];
        let h = xywh[3];
        x = (col + this.sigmoid(x)) / GRID_W;
        y = (row + this.sigmoid(y)) / GRID_H;
        w = this.ANCHORS[2 * box + 0] * Math.exp(w) / GRID_W;
        h = this.ANCHORS[2 * box + 1] * Math.exp(h) / GRID_H;

        boxes.push(new BoundingBox(x, y, w, h, conf, probs as Float32Array));
      }
    }

    confidence.dispose();
    classes.dispose();
    objectLikelihood.dispose();

    // suppress nonmaximal boxes
    for (let cls = 0; cls < CLASS; cls++) {
      const allProbs = boxes.map((box) => box.probs[cls]);
      const indices = new Array(allProbs.length);

      for (let i = 0; i < allProbs.length; ++i) {
        indices[i] = i;
      }

      indices.sort((a, b) => allProbs[a] > allProbs[b] ? 1 : 0);

      for (let i = 0; i < allProbs.length; i++) {
        const indexI = indices[i];

        if (boxes[indexI].probs[cls] === 0) {
          continue;
        } else {
          for (let j = i + 1; j < allProbs.length; j++) {
            const indexJ = indices[j];

            if (boxes[indexI].iou(boxes[indexJ]) > 0.4) {
              boxes[indexJ].probs[cls] = 0;
            }
          }
        }
      }
    }

    // obtain the most likely boxes
    const likelyBoxes = [];

    for (const box of boxes) {
      if (box.getMaxProb() > this.THRESHOLD) {
        likelyBoxes.push(box);
      }
    }

    return likelyBoxes;
  }

  private sigmoid(x: number): number {
    return 1. / (1. + Math.exp(-x));
  }

  dispose() {
    for (const varName in this.variables) {
      this.variables[varName].dispose();
    }
  }
}
