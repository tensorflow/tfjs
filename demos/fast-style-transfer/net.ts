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
import {Array1D, Array3D, Array4D, CheckpointLoader, ENV, Model, NDArray, NDArrayMath, Scalar} from 'deeplearn';

const GOOGLE_CLOUD_STORAGE_DIR =
    'https://storage.googleapis.com/learnjs-data/checkpoint_zoo/transformnet/';

export class TransformNet implements Model {
  private variables: {[varName: string]: NDArray};
  private variableDictionary:
      {[styleName: string]: {[varName: string]: NDArray}};
  private timesScalar: Scalar;
  private plusScalar: Scalar;
  private epsilonScalar: NDArray;
  private math: NDArrayMath;

  constructor(private style: string) {
    this.math = ENV.math;
    this.variableDictionary = {};
    this.timesScalar = Scalar.new(150);
    this.plusScalar = Scalar.new(255. / 2);
    this.epsilonScalar = Scalar.new(1e-3);
  }

  setStyle(style: string) {
    this.style = style;
  }

  /**
   * Loads necessary variables for SqueezeNet. Resolves the promise when the
   * variables have all been loaded.
   */
  async load(): Promise<void> {
    if (this.variableDictionary[this.style] == null) {
      const checkpointLoader =
          new CheckpointLoader(GOOGLE_CLOUD_STORAGE_DIR + this.style + '/');
      this.variableDictionary[this.style] =
          await checkpointLoader.getAllVariables();
    }
    this.variables = this.variableDictionary[this.style];
  }

  /**
   * Infer through TransformNet, assumes variables have been loaded.
   * Original Tensorflow version of model can be found at
   * https://github.com/lengstrom/fast-style-transfer
   *
   * @param preprocessedInput preprocessed input Array.
   * @return Array3D containing pixels of output img
   */
  predict(preprocessedInput: Array3D): Array3D {
    const img = this.math.scope((keep, track) => {
      const conv1 = this.convLayer(preprocessedInput, 1, true, 0);
      const conv2 = this.convLayer(conv1, 2, true, 3);
      const conv3 = this.convLayer(conv2, 2, true, 6);
      const resid1 = this.residualBlock(conv3, 9);
      const resid2 = this.residualBlock(resid1, 15);
      const resid3 = this.residualBlock(resid2, 21);
      const resid4 = this.residualBlock(resid3, 27);
      const resid5 = this.residualBlock(resid4, 33);
      const convT1 = this.convTransposeLayer(resid5, 64, 2, 39);
      const convT2 = this.convTransposeLayer(convT1, 32, 2, 42);
      const convT3 = this.convLayer(convT2, 1, false, 45);
      const outTanh = this.math.tanh(convT3);
      const scaled = this.math.scalarTimesArray(this.timesScalar, outTanh);
      const shifted = this.math.scalarPlusArray(this.plusScalar, scaled);
      const clamped = this.math.clip(shifted, 0, 255);
      const normalized = this.math.divide(clamped, Scalar.new(255.)) as Array3D;

      return normalized;
    });

    return img;
  }

  private convLayer(
      input: Array3D, strides: number, relu: boolean, varId: number): Array3D {
    const y = this.math.conv2d(
        input, this.variables[this.varName(varId)] as Array4D, null,
        [strides, strides], 'same');

    const y2 = this.instanceNorm(y, varId + 1);

    if (relu) {
      return this.math.relu(y2);
    }

    return y2;
  }

  private convTransposeLayer(
      input: Array3D, numFilters: number, strides: number,
      varId: number): Array3D {
    const [height, width, ]: [number, number, number] = input.shape;
    const newRows = height * strides;
    const newCols = width * strides;
    const newShape: [number, number, number] = [newRows, newCols, numFilters];

    const y = this.math.conv2dTranspose(
        input, this.variables[this.varName(varId)] as Array4D, newShape,
        [strides, strides], 'same');

    const y2 = this.instanceNorm(y, varId + 1);

    const y3 = this.math.relu(y2);

    return y3;
  }

  private residualBlock(input: Array3D, varId: number): Array3D {
    const conv1 = this.convLayer(input, 1, true, varId);
    const conv2 = this.convLayer(conv1, 1, false, varId + 3);
    return this.math.addStrict(conv2, input);
  }

  private instanceNorm(input: Array3D, varId: number): Array3D {
    const [height, width, inDepth]: [number, number, number] = input.shape;
    const moments = this.math.moments(input, [0, 1]);
    const mu = moments.mean;
    const sigmaSq = moments.variance as Array3D;
    const shift = this.variables[this.varName(varId)] as Array1D;
    const scale = this.variables[this.varName(varId + 1)] as Array1D;
    const epsilon = this.epsilonScalar;
    const normalized = this.math.divide(
        this.math.sub(input.asType('float32'), mu),
        this.math.sqrt(this.math.add(sigmaSq, epsilon)));
    const shifted = this.math.add(this.math.multiply(scale, normalized), shift);
    return shifted.as3D(height, width, inDepth);
  }

  private varName(varId: number): string {
    if (varId === 0) {
      return 'Variable';
    } else {
      return 'Variable_' + varId.toString();
    }
  }

  dispose() {
    for (const styleName in this.variableDictionary) {
      for (const varName in this.variableDictionary[styleName]) {
        this.variableDictionary[styleName][varName].dispose();
      }
    }
  }
}
