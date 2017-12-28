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
import {BoundingBox} from './mobilenet_utils';

const GOOGLE_CLOUD_STORAGE_DIR =
    'https://storage.googleapis.com/learnjs-data/checkpoint_zoo/';

export class YoloMobileNetDetection implements Model {
  private variables: {[varName: string]: NDArray};

  // yolo variables
  private PREPROCESS_DIVISOR = Scalar.new(255.0 / 2);
  private ONE = Scalar.new(1);
  private THRESHOLD = 0.3;
  private THRESHOLD_SCALAR = Scalar.new(this.THRESHOLD);
  private ANCHORS: number[] = [
    0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778,
    9.77052, 9.16828
  ];

  constructor(private math: NDArrayMath) {}

  /**
   * Loads necessary variables for MobileNet.
   */
  async load(): Promise<void> {
    const checkpointLoader = new CheckpointLoader(
        GOOGLE_CLOUD_STORAGE_DIR + 'yolo_mobilenet_v1_1.0_416/');
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
  predict(input: Array3D): Array4D {
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

      const x15 = this.math.conv2d(
          x14, this.variables['conv_23/kernel'] as Array4D,
          this.variables['conv_23/bias'] as Array1D, [1, 1], 'same');

      return x15.as4D(13, 13, 5, 6);
    });

    return netout;
  }

  private convBlock(inputs: Array3D, strides: [number, number]) {
    const x1 = this.math.conv2d(
        inputs, this.variables['conv1/kernel'] as Array4D,
        null,  // this convolutional layer does not use bias
        strides, 'same');

    const x2 = this.math.batchNormalization3D(
        x1, this.variables['conv1_bn/moving_mean'] as Array1D,
        this.variables['conv1_bn/moving_variance'] as Array1D, .001,
        this.variables['conv1_bn/gamma'] as Array1D,
        this.variables['conv1_bn/beta'] as Array1D);

    return this.math.clip(x2, 0, 6);  // simple implementation of Relu6
  }

  private depthwiseConvBlock(
      inputs: Array3D, strides: [number, number], blockID: number) {
    const dwPadding = 'conv_dw_' + String(blockID) + '';
    const pwPadding = 'conv_pw_' + String(blockID) + '';

    const x1 =
        this.math.depthwiseConv2D(
            inputs, this.variables[dwPadding + '/depthwise_kernel'] as Array4D,
            strides, 'same') as Array3D;

    const x2 = this.math.batchNormalization3D(
        x1, this.variables[dwPadding + '_bn/moving_mean'] as Array1D,
        this.variables[dwPadding + '_bn/moving_variance'] as Array1D, .001,
        this.variables[dwPadding + '_bn/gamma'] as Array1D,
        this.variables[dwPadding + '_bn/beta'] as Array1D);

    const x3 = this.math.clip(x2, 0, 6);

    const x4 = this.math.conv2d(
        x3, this.variables[pwPadding + '/kernel'] as Array4D,
        null,  // this convolutional layer does not use bias
        [1, 1], 'same');

    const x5 = this.math.batchNormalization3D(
        x4, this.variables[pwPadding + '_bn/moving_mean'] as Array1D,
        this.variables[pwPadding + '_bn/moving_variance'] as Array1D, .001,
        this.variables[pwPadding + '_bn/gamma'] as Array1D,
        this.variables[pwPadding + '_bn/beta'] as Array1D);

    return this.math.clip(x5, 0, 6);
  }

  async interpretNetout(netout: Array4D): Promise<BoundingBox[]> {
    // interpret the output by the network
    const GRID_H = netout.shape[0];
    const GRID_W = netout.shape[1];
    const BOX = netout.shape[2];
    const CLASS = netout.shape[3] - 5;
    const boxes: BoundingBox[] = [];

    // adjust confidence predictions
    const confidence = this.math.sigmoid(
        this.math.slice4D(netout, [0, 0, 0, 4], [GRID_H, GRID_W, BOX, 1]));

    // adjust class prediction
    let classes = this.math.softmax(
        this.math.slice4D(netout, [0, 0, 0, 5], [GRID_H, GRID_W, BOX, CLASS]));
    classes = this.math.multiply(classes, confidence) as Array4D;
    const mask = this.math.step(
        this.math.relu(this.math.subtract(classes, this.THRESHOLD_SCALAR)));
    classes = this.math.multiply(classes, mask) as Array4D;

    const objectLikelihood = this.math.sum(classes, 3);
    const objectLikelihoodValues = await objectLikelihood.data();

    for (let i = 0; i < objectLikelihoodValues.length; i++) {
      if (objectLikelihoodValues[i] > 0) {
        const [row, col, box] = objectLikelihood.indexToLoc(i) as number[];

        const conf = confidence.get(row, col, box, 0);
        const probs =
            await this.math
                .slice4D(classes, [row, col, box, 0], [1, 1, 1, CLASS])
                .data();
        const xywh =
            await this.math.slice4D(netout, [row, col, box, 0], [1, 1, 1, 4])
                .data();

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
