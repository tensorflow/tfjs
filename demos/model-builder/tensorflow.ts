/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// tslint:disable-next-line:max-line-length
import {Convolution2DLayerBuilder, LayerBuilder, MaxPoolLayerBuilder} from './layer_builder';

export enum Normalization {
  NORMALIZATION_NEGATIVE_ONE_TO_ONE,
  NORMALIZATION_ZERO_TO_ONE,
  NORMALIZATION_NONE
}

export function generatePython(
    datasetName: string, normalizationStrategy: number, inputShape: number[],
    modelLayers: LayerBuilder[]): string {
  const loadData = generateLoadData(datasetName, normalizationStrategy);
  const buildModel = generateBuildModel(inputShape, modelLayers);
  const captureWeights = generateCaptureWeights(modelLayers);
  return [loadData, buildModel, captureWeights].join('\n\n');
}

function generateLoadData(
    datasetName: string, normalizationStrategy: number): string {
  let loadFunction: string;
  switch (datasetName) {
    case 'CIFAR 10': {
      loadFunction = 'cifar10';
      break;
    }
    case 'MNIST': {
      loadFunction = 'mnist';
      break;
    }
    default: {
      throw new Error('datasetName must be \'CIFAR 10\' or \'MNIST\'');
    }
  }

  let normString: string;
  switch (normalizationStrategy) {
    case Normalization.NORMALIZATION_NEGATIVE_ONE_TO_ONE: {
      normString = 'NORMALIZATION_NEGATIVE_ONE_TO_ONE';
      break;
    }
    case Normalization.NORMALIZATION_ZERO_TO_ONE: {
      normString = 'NORMALIZATION_ZERO_TO_ONE';
      break;
    }
    case Normalization.NORMALIZATION_NONE: {
      normString = 'NORMALIZATION_NONE';
      break;
    }
    default: { throw new Error('invalid normalizationStrategy value'); }
  }

  return `def load_data():
  return learnjs_colab.load_${loadFunction}(learnjs_colab.${normString})
`;
}

function generateBuildModelLayer(
    layerIndex: number, inputShape: number[], layer: LayerBuilder): string {
  let src = '';
  const W = 'W_' + layerIndex;
  const b = 'b_' + layerIndex;
  const outputShape = layer.getOutputShape(inputShape);
  switch (layer.layerName) {
    case 'Fully connected': {
      const shape = [inputShape[0], outputShape].join(', ');

      src = `  ${W} = tf.Variable(tf.truncated_normal([${shape}],
                    stddev = 1.0 / math.sqrt(${outputShape[0]})))
  ${b} = tf.Variable(tf.truncated_normal([${outputShape[0]}], stddev = 0.1))
  layers.append({ 'x': layers[-1]['y'],
                  'W': ${W},
                  'b': ${b},
                  'y': tf.add(tf.matmul(layers[-1]['y'], ${W}), ${b}) })`;
      break;
    }

    case 'ReLU': {
      src = `  layers.append({ 'x': layers[-1]['y'],
                  'y': tf.nn.relu(layers[-1]['y']) })`;
      break;
    }

    case 'Convolution': {
      const conv = layer as Convolution2DLayerBuilder;
      const f = conv.fieldSize;
      const d1 = inputShape[inputShape.length - 1];
      const d2 = outputShape[outputShape.length - 1];
      const wShape = '[' + f + ', ' + f + ', ' + d1 + ', ' + d2 + ']';
      const stride = '[1, ' + conv.stride + ', ' + conv.stride + ', 1]';
      src = `  ${W} = tf.Variable(tf.truncated_normal(${wShape}, stddev = 0.1))
  ${b} = tf.Variable(tf.truncated_normal([${d2}], stddev = 0.1))
  layers.append({ 'x': layers[-1]['y'],
                  'W': ${W},
                  'b': ${b},
                  'y': tf.add(tf.nn.conv2d(layers[-1]['y'],
                                           ${W},
                                           strides = ${stride},
                                           padding = 'SAME'), ${b}) })`;
      break;
    }

    case 'Max pool': {
      const mp = layer as MaxPoolLayerBuilder;
      const field = '[1, ' + mp.fieldSize + ', ' + mp.fieldSize + ', 1]';
      const stride = '[1, ' + mp.stride + ', ' + mp.stride + ', 1]';
      src = `  layers.append({ 'x': layers[-1]['y'],
                  'y': tf.nn.max_pool(layers[-1]['y'],
                                      ${field},
                                      ${stride},
                                      padding = 'SAME') })`;
      break;
    }

    case 'Reshape': {
      break;
    }

    case 'Flatten': {
      src = `  layers.append({ 'x': layers[-1]['y'],
                  'y': tf.reshape(layers[-1]['y'], [-1, ${outputShape[0]}]) })`;
      break;
    }

    default: {
      throw new Error('unknown layer type \'' + layer.layerName + '\'');
    }
  }

  return src;
}

function generateBuildModel(
    inputShape: number[], modelLayers: LayerBuilder[]): string {
  const inputShapeStr = inputShape.join(', ');
  const sources: string[] = [];

  sources.push(`def build_model():
  layers = []

  layers.append({ 'y': tf.placeholder(tf.float32, [None, ${inputShapeStr}]),
                  'y_label': tf.placeholder(tf.float32, [None, 10]) })`);

  for (let i = 0; i < modelLayers.length; ++i) {
    sources.push(generateBuildModelLayer(i + 1, inputShape, modelLayers[i]));
    inputShape = modelLayers[i].getOutputShape(inputShape);
  }

  sources.push('  return layers\n');
  return sources.join('\n\n');
}

function generateCaptureWeights(modelLayers: LayerBuilder[]): string {
  const sources: string[] = [];
  sources.push(`def capture_weights():
  weights = []`);

  for (let i = 0; i < modelLayers.length; ++i) {
    const layer = modelLayers[i];
    const index = i + 1;
    let src = '';
    const W = '\'W\': model[' + index + '][\'W\']';
    const b = '\'b\': model[' + index + '][\'b\']';
    switch (layer.layerName) {
      case 'Fully connected': {
        src = `  weights.append({ ${W}.eval().flatten().tolist(),
                   ${b}.eval().flatten().tolist() })`;
        break;
      }

      case 'Convolution': {
        src = `  weights.append({ ${W}.eval().transpose().flatten().tolist(),
                   ${b}.eval().flatten().tolist() })`;
        break;
      }

      default: { src = '  weights.append({})'; }
    }

    src += ' # ' + layer.layerName;
    sources.push(src);
  }

  sources.push('  return weights');
  return sources.join('\n');
}
