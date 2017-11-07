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
import {Array1D, Array2D, Array4D, conv_util, Graph, Initializer, NDArrayInitializer, Tensor, util, VarianceScalingInitializer, ZerosInitializer} from 'deeplearn';

/**
 * Classes that specify operation parameters, how they affect output shape,
 * and methods for building the operations themselves. Any new ops to be added
 * to the model builder UI should be added here.
 */

export type LayerName =
    'Fully connected'|'ReLU'|'Convolution'|'Max pool'|'Reshape'|'Flatten';

/**
 * Creates a layer builder object.
 *
 * @param layerName The name of the layer to build.
 * @param layerBuilderJson An optional LayerBuilder JSON object. This doesn't
 *     have the prototype methods on them as it comes from serialization. This
 *     method creates the object with the necessary prototype methods.
 */
export function getLayerBuilder(
    layerName: LayerName, layerBuilderJson?: LayerBuilder): LayerBuilder {
  let layerBuilder: LayerBuilder;
  switch (layerName) {
    case 'Fully connected':
      layerBuilder = new FullyConnectedLayerBuilder();
      break;
    case 'ReLU':
      layerBuilder = new ReLULayerBuilder();
      break;
    case 'Convolution':
      layerBuilder = new Convolution2DLayerBuilder();
      break;
    case 'Max pool':
      layerBuilder = new MaxPoolLayerBuilder();
      break;
    case 'Reshape':
      layerBuilder = new ReshapeLayerBuilder();
      break;
    case 'Flatten':
      layerBuilder = new FlattenLayerBuilder();
      break;
    default:
      throw new Error(`Layer builder for ${layerName} not found.`);
  }

  // For layer builders passed as serialized objects, we create the objects and
  // set the fields.
  if (layerBuilderJson != null) {
    for (const prop in layerBuilderJson) {
      if (layerBuilderJson.hasOwnProperty(prop)) {
        // tslint:disable-next-line:no-any
        (layerBuilder as any)[prop] = (layerBuilderJson as any)[prop];
      }
    }
  }
  return layerBuilder;
}

export interface LayerParam {
  label: string;
  initialValue(inputShape: number[]): number|string;
  type: 'number'|'text';
  min?: number;
  max?: number;
  setValue(value: number|string): void;
  getValue(): number|string;
}

export type LayerWeightsDict = {
  [name: string]: number[]
};

export interface LayerBuilder {
  layerName: LayerName;
  getLayerParams(): LayerParam[];
  getOutputShape(inputShape: number[]): number[];
  addLayer(
      g: Graph, network: Tensor, inputShape: number[], index: number,
      weights?: LayerWeightsDict|null): Tensor;
  // Return null if no errors, otherwise return an array of errors.
  validate(inputShape: number[]): string[]|null;
}

export class FullyConnectedLayerBuilder implements LayerBuilder {
  layerName: LayerName = 'Fully connected';
  hiddenUnits: number;

  getLayerParams(): LayerParam[] {
    return [{
      label: 'Hidden units',
      initialValue: (inputShape: number[]) => 10,
      type: 'number',
      min: 1,
      max: 1000,
      setValue: (value: number) => this.hiddenUnits = value,
      getValue: () => this.hiddenUnits
    }];
  }

  getOutputShape(inputShape: number[]): number[] {
    return [this.hiddenUnits];
  }

  addLayer(
      g: Graph, network: Tensor, inputShape: number[], index: number,
      weights: LayerWeightsDict|null): Tensor {
    const inputSize = util.sizeFromShape(inputShape);
    const wShape: [number, number] = [this.hiddenUnits, inputSize];

    let weightsInitializer: Initializer;
    let biasInitializer: Initializer;
    if (weights != null) {
      weightsInitializer =
          new NDArrayInitializer(Array2D.new(wShape, weights['W']));
      biasInitializer = new NDArrayInitializer(Array1D.new(weights['b']));
    } else {
      weightsInitializer = new VarianceScalingInitializer();
      biasInitializer = new ZerosInitializer();
    }

    const useBias = true;
    return g.layers.dense(
        'fc1', network, this.hiddenUnits, null, useBias, weightsInitializer,
        biasInitializer);
  }

  validate(inputShape: number[]) {
    if (inputShape.length !== 1) {
      return ['Input shape must be a Array1D.'];
    }
    return null;
  }
}

export class ReLULayerBuilder implements LayerBuilder {
  layerName: LayerName = 'ReLU';
  getLayerParams(): LayerParam[] {
    return [];
  }

  getOutputShape(inputShape: number[]): number[] {
    return inputShape;
  }

  addLayer(
      g: Graph, network: Tensor, inputShape: number[], index: number,
      weights: LayerWeightsDict|null): Tensor {
    return g.relu(network);
  }

  validate(inputShape: number[]): string[]|null {
    return null;
  }
}

export class Convolution2DLayerBuilder implements LayerBuilder {
  layerName: LayerName = 'Convolution';
  fieldSize: number;
  stride: number;
  zeroPad: number;
  outputDepth: number;

  getLayerParams(): LayerParam[] {
    return [
      {
        label: 'Field size',
        initialValue: (inputShape: number[]) => 3,
        type: 'number',
        min: 1,
        max: 100,
        setValue: (value: number) => this.fieldSize = value,
        getValue: () => this.fieldSize
      },
      {
        label: 'Stride',
        initialValue: (inputShape: number[]) => 1,
        type: 'number',
        min: 1,
        max: 100,
        setValue: (value: number) => this.stride = value,
        getValue: () => this.stride
      },
      {
        label: 'Zero pad',
        initialValue: (inputShape: number[]) => 0,
        type: 'number',
        min: 0,
        max: 100,
        setValue: (value: number) => this.zeroPad = value,
        getValue: () => this.zeroPad
      },
      {
        label: 'Output depth',
        initialValue: (inputShape: number[]) =>
            this.outputDepth != null ? this.outputDepth : 1,
        type: 'number',
        min: 1,
        max: 1000,
        setValue: (value: number) => this.outputDepth = value,
        getValue: () => this.outputDepth
      }
    ];
  }

  getOutputShape(inputShape: number[]): number[] {
    return conv_util.computeOutputShape3D(
        inputShape as [number, number, number], this.fieldSize,
        this.outputDepth, this.stride, this.zeroPad);
  }

  addLayer(
      g: Graph, network: Tensor, inputShape: number[], index: number,
      weights: LayerWeightsDict|null): Tensor {
    const wShape: [number, number, number, number] =
        [this.fieldSize, this.fieldSize, inputShape[2], this.outputDepth];
    let w: Array4D;
    let b: Array1D;
    if (weights != null) {
      w = Array4D.new(wShape, weights['W']);
      b = Array1D.new(weights['b']);
    } else {
      w = Array4D.randTruncatedNormal(wShape, 0, 0.1);
      b = Array1D.zeros([this.outputDepth]);
    }
    const wTensor = g.variable(`conv2d-${index}-w`, w);
    const bTensor = g.variable(`conv2d-${index}-b`, b);
    return g.conv2d(
        network, wTensor, bTensor, this.fieldSize, this.outputDepth,
        this.stride, this.zeroPad);
  }

  validate(inputShape: number[]) {
    if (inputShape.length !== 3) {
      return ['Input shape must be a Array3D.'];
    }
    return null;
  }
}

export class MaxPoolLayerBuilder implements LayerBuilder {
  layerName: LayerName = 'Max pool';
  fieldSize: number;
  stride: number;
  zeroPad: number;

  getLayerParams(): LayerParam[] {
    return [
      {
        label: 'Field size',
        initialValue: (inputShape: number[]) => 3,
        type: 'number',
        min: 1,
        max: 100,
        setValue: (value: number) => this.fieldSize = value,
        getValue: () => this.fieldSize
      },
      {
        label: 'Stride',
        initialValue: (inputShape: number[]) => 1,
        type: 'number',
        min: 1,
        max: 100,
        setValue: (value: number) => this.stride = value,
        getValue: () => this.stride
      },
      {
        label: 'Zero pad',
        initialValue: (inputShape: number[]) => 0,
        type: 'number',
        min: 0,
        max: 100,
        setValue: (value: number) => this.zeroPad = value,
        getValue: () => this.zeroPad
      }
    ];
  }

  getOutputShape(inputShape: number[]): number[] {
    return conv_util.computeOutputShape3D(
        inputShape as [number, number, number], this.fieldSize, inputShape[2],
        this.stride, this.zeroPad);
  }

  addLayer(
      g: Graph, network: Tensor, inputShape: number[], index: number,
      weights: LayerWeightsDict|null): Tensor {
    return g.maxPool(network, this.fieldSize, this.stride, this.zeroPad);
  }

  validate(inputShape: number[]) {
    if (inputShape.length !== 3) {
      return ['Input shape must be a Array3D.'];
    }
    return null;
  }
}

export class ReshapeLayerBuilder implements LayerBuilder {
  layerName: LayerName = 'Reshape';
  outputShape: number[];
  getLayerParams() {
    return [{
      label: 'Shape (comma separated)',
      initialValue: (inputShape: number[]) => inputShape.join(', '),
      type: 'text' as 'text',
      setValue: (value: string) => this.outputShape =
          value.split(',').map((value) => +value),
      getValue: () => this.outputShape.join(', ')
    }];
  }

  getOutputShape(inputShape: number[]): number[] {
    return this.outputShape;
  }

  addLayer(
      g: Graph, network: Tensor, inputShape: number[], index: number,
      weights: LayerWeightsDict|null): Tensor {
    return g.reshape(network, this.outputShape);
  }

  validate(inputShape: number[]) {
    const inputSize = util.sizeFromShape(inputShape);
    const outputSize = util.sizeFromShape(this.outputShape);
    if (inputSize !== outputSize) {
      return [
        `Input size (${inputSize}) must match output size (${outputSize}).`
      ];
    }
    return null;
  }
}

export class FlattenLayerBuilder implements LayerBuilder {
  layerName: LayerName = 'Flatten';

  getLayerParams(): LayerParam[] {
    return [];
  }

  getOutputShape(inputShape: number[]): number[] {
    return [util.sizeFromShape(inputShape)];
  }

  addLayer(
      g: Graph, network: Tensor, inputShape: number[], index: number,
      weights: LayerWeightsDict|null): Tensor {
    return g.reshape(network, this.getOutputShape(inputShape));
  }

  validate(inputShape: number[]): string[]|null {
    return null;
  }
}
