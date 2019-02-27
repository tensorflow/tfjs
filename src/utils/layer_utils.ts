/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {Container} from '../engine/container';
import {Layer, Node} from '../engine/topology';
import {countParamsInWeights} from './variable_utils';

/**
 * Print the summary of a LayersModel object.
 *
 * @param model tf.LayersModel instance.
 * @param lineLength Total length of printed lines. Set this to adapt to the
 *   display to different terminal or console sizes.
 * @param positions Relative or absolute positions of log elements in each
 *   line. Each number corresponds to right-most (i.e., ending) position of a
 *   column.
 *   If not provided, defaults to `[0.45, 0.85, 1]` for sequential-like
 *   models and `[0.33, 0.55, 0.67, 1]` for non-sequential like models.
 * @param printFn Print function to use.
 *   It will be called on each line of the summary. You can provide a custom
 *   function in order to capture the string summary. Defaults to `console.log`.
 */
export function printSummary(
    model: Container, lineLength?: number, positions?: number[],
    // tslint:disable-next-line:no-any
    printFn: (message?: any, ...optionalParams: any[]) => void =
        console.log): void {
  const sequentialLike = isModelSequentialLike(model);

  // Header names for different log elements.
  const toDisplay: string[] = ['Layer (type)', 'Output shape', 'Param #'];
  if (sequentialLike) {
    lineLength = lineLength || 65;
    positions = positions || [0.45, 0.85, 1];
  } else {
    lineLength = lineLength || 98;
    positions = positions || [0.33, 0.55, 0.67, 1];
    // Header names for different log elements.
  }

  if (positions[positions.length - 1] <= 1) {
    // `positions` is relative. Convert it to absolute positioning.
    positions = positions.map(p => Math.floor(lineLength * p));
  }

  let relevantNodes: Node[];
  if (!sequentialLike) {
    toDisplay.push('Receives inputs');
    relevantNodes = [];
    for (const depth in model.nodesByDepth) {
      relevantNodes.push(...model.nodesByDepth[depth]);
    }
  }

  printFn('_'.repeat(lineLength));
  printRow(toDisplay, positions, printFn);
  printFn('='.repeat(lineLength));

  const layers = model.layers;
  for (let i = 0; i < layers.length; ++i) {
    if (sequentialLike) {
      printLayerSummary(layers[i], positions, printFn);
    } else {
      printLayerSummaryWithConnections(
          layers[i], positions, relevantNodes, printFn);
    }
    printFn((i === layers.length - 1 ? '=' : '_').repeat(lineLength));
  }

  // tslint:disable-next-line:no-any
  (model as any).checkTrainableWeightsConsistency();

  const trainableCount = countTrainableParams(model);
  const nonTrainableCount = countParamsInWeights(model.nonTrainableWeights);

  printFn(`Total params: ${trainableCount + nonTrainableCount}`);
  printFn(`Trainable params: ${trainableCount}`);
  printFn(`Non-trainable params: ${nonTrainableCount}`);
  printFn('_'.repeat(lineLength));
}

function countTrainableParams(model: Container): number {
  let trainableCount: number;
  // tslint:disable:no-any
  if ((model as any).collectedTrainableWeights != null) {
    trainableCount =
        countParamsInWeights((model as any).collectedTrainableWeights);
  } else {
    trainableCount = countParamsInWeights(model.trainableWeights);
  }
  // tslint:enable:no-any
  return trainableCount;
}

function isModelSequentialLike(model: Container): boolean {
  let sequentialLike = true;
  const nodesByDepth: Node[][] = [];
  const nodes: Node[] = [];
  for (const depth in model.nodesByDepth) {
    nodesByDepth.push(model.nodesByDepth[depth]);
  }
  for (const depthNodes of nodesByDepth) {
    if (depthNodes.length > 1 ||
        depthNodes.length === 1 && depthNodes[0].inboundLayers.length > 1) {
      sequentialLike = false;
      break;
    }
    nodes.push(...depthNodes);
  }
  if (sequentialLike) {
    // Search for shared layers.
    for (const layer of model.layers) {
      let flag = false;
      for (const node of layer.inboundNodes) {
        if (nodes.indexOf(node) !== -1) {
          if (flag) {
            sequentialLike = false;
            break;
          } else {
            flag = true;
          }
        }
      }
      if (!sequentialLike) {
        break;
      }
    }
  }
  return sequentialLike;
}

function printRow(
    fields: string[], positions: number[],
    // tslint:disable-next-line:no-any
    printFn: (message?: any, ...optionalParams: any[]) => void = console.log) {
  let line = '';
  for (let i = 0; i < fields.length; ++i) {
    if (i > 0) {
      line = line.slice(0, line.length - 1) + ' ';
    }
    line += fields[i];
    line = line.slice(0, positions[i]);
    line += ' '.repeat(positions[i] - line.length);
  }
  printFn(line);
}

/**
 * Prints a summary for a single Layer, without connectivity information.
 *
 * @param layer: Layer instance to print.
 */
function printLayerSummary(
    layer: Layer, positions: number[],
    // tslint:disable-next-line:no-any
    printFn: (message?: any, ...optionalParams: any[]) => void) {
  let outputShape: string;
  try {
    outputShape = JSON.stringify(layer.outputShape);
  } catch (err) {
    outputShape = 'multiple';
  }

  const name = layer.name;
  const className = layer.getClassName();
  const fields: string[] =
      [`${name} (${className})`, outputShape, layer.countParams().toString()];
  printRow(fields, positions, printFn);
}

/**
 * Prints a summary for a single Layer, with connectivity information.
 */
function printLayerSummaryWithConnections(
    layer: Layer, positions: number[], relevantNodes: Node[],
    // tslint:disable-next-line:no-any
    printFn: (message?: any, ...optionalParams: any[]) => void) {
  let outputShape: string;
  try {
    outputShape = JSON.stringify(layer.outputShape);
  } catch (err) {
    outputShape = 'multiple';
  }

  const connections: string[] = [];
  for (const node of layer.inboundNodes) {
    if (relevantNodes != null && relevantNodes.length > 0 &&
        relevantNodes.indexOf(node) === -1) {
      continue;
    }
    for (let i = 0; i < node.inboundLayers.length; ++i) {
      const inboundLayer = node.inboundLayers[i].name;
      const inboundLayerIndex = node.nodeIndices[i];
      const inboundTensorIndex = node.tensorIndices[i];
      connections.push(
          `${inboundLayer}[${inboundLayerIndex}][${inboundTensorIndex}]`);
    }
  }
  const name = layer.name;
  const className = layer.getClassName();
  const firstConnection = connections.length === 0 ? '' : connections[0];
  const fields: string[] = [
    `${name} (${className})`, outputShape, layer.countParams().toString(),
    firstConnection
  ];

  printRow(fields, positions, printFn);
  for (let i = 1; i < connections.length; ++i) {
    printRow(['', '', '', connections[i]], positions, printFn);
  }
}
