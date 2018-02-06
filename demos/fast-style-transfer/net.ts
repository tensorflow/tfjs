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

const GOOGLE_CLOUD_STORAGE_DIR =
    'https://storage.googleapis.com/learnjs-data/checkpoint_zoo/transformnet/';

export class TransformNet implements dl.Model {
  private variables: {[varName: string]: dl.Tensor};
  private variableDictionary:
      {[styleName: string]: {[varName: string]: dl.Tensor}};
  private timesScalar: dl.Scalar;
  private plusScalar: dl.Scalar;
  private epsilonScalar: dl.Tensor;

  constructor(private style: string) {
    this.variableDictionary = {};
    this.timesScalar = dl.scalar(150);
    this.plusScalar = dl.scalar(255. / 2);
    this.epsilonScalar = dl.scalar(1e-3);
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
          new dl.CheckpointLoader(GOOGLE_CLOUD_STORAGE_DIR + this.style + '/');
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
   * @return dl.Tensor3D containing pixels of output img
   */
  predict(preprocessedInput: dl.Tensor3D): dl.Tensor3D {
    const img = dl.tidy(() => {
      const conv1 = this.convLayer(preprocessedInput.toFloat(), 1, true, 0);
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

      return convT3.tanh()
                 .mul(this.timesScalar)
                 .add(this.plusScalar)
                 .clip(0, 255)
                 .div(dl.scalar(255)) as dl.Tensor3D;
    });

    return img;
  }

  private convLayer(
      input: dl.Tensor3D, strides: number, relu: boolean,
      varId: number): dl.Tensor3D {
    const y = input.conv2d(
        this.variables[this.varName(varId)] as dl.Tensor4D, null,
        [strides, strides], 'same');

    const y2 = this.instanceNorm(y, varId + 1);

    if (relu) {
      return y2.relu();
    }

    return y2;
  }

  private convTransposeLayer(
      input: dl.Tensor3D, numFilters: number, strides: number,
      varId: number): dl.Tensor3D {
    const [height, width, ]: [number, number, number] = input.shape;
    const newRows = height * strides;
    const newCols = width * strides;
    const newShape: [number, number, number] = [newRows, newCols, numFilters];

    const y = input.conv2dTranspose(
        this.variables[this.varName(varId)] as dl.Tensor4D, newShape,
        [strides, strides], 'same');

    return this.instanceNorm(y, varId + 1).relu();
  }

  private residualBlock(input: dl.Tensor3D, varId: number): dl.Tensor3D {
    const conv1 = this.convLayer(input, 1, true, varId);
    const conv2 = this.convLayer(conv1, 1, false, varId + 3);
    return conv2.addStrict(input);
  }

  private instanceNorm(input: dl.Tensor3D, varId: number): dl.Tensor3D {
    const [height, width, inDepth]: [number, number, number] = input.shape;
    const moments = dl.moments(input, [0, 1]);
    const mu = moments.mean;
    const sigmaSq = moments.variance as dl.Tensor3D;
    const shift = this.variables[this.varName(varId)] as dl.Tensor1D;
    const scale = this.variables[this.varName(varId + 1)] as dl.Tensor1D;
    const epsilon = this.epsilonScalar;

    const normalized = input.sub(mu).div(sigmaSq.add(epsilon).sqrt());

    const shifted = scale.mul(normalized).add(shift);

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
