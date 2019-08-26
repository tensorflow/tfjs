/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {dispose, mul, Tensor, Tensor1D, tensor1d, tidy} from '@tensorflow/tfjs-core';

/**
 * For multi-class classification problems, this object is designed to store a
 * mapping from class index to the "weight" of the class, where higher weighted
 * classes have larger impact on loss, accuracy, and other metrics.
 *
 * This is useful for cases in which you want the model to "pay more attention"
 * to examples from an under-represented class, e.g., in unbalanced datasets.
 */
export type ClassWeight = {
  [classIndex: number]: number
};

/**
 * Class weighting for a model with multiple outputs.
 *
 * This object maps each output name to a class-weighting object.
 */
export type ClassWeightMap = {
  [outputName: string]: ClassWeight
};

function standardizeSampleOrClassWeights(
    xWeight: ClassWeight|ClassWeight[]|ClassWeightMap, outputNames: string[],
    weightType: 'sampleWeight'|'classWeight'): ClassWeight[] {
  const numOutputs = outputNames.length;
  if (xWeight == null || (Array.isArray(xWeight) && xWeight.length === 0)) {
    return outputNames.map(name => null);
  }
  if (numOutputs === 1) {
    if (Array.isArray(xWeight) && xWeight.length === 1) {
      return xWeight;
    } else if (typeof xWeight === 'object' && outputNames[0] in xWeight) {
      return [(xWeight as ClassWeightMap)[outputNames[0]]];
    } else {
      return [xWeight as ClassWeight];
    }
  }
  if (Array.isArray(xWeight)) {
    if (xWeight.length !== numOutputs) {
      throw new Error(
          `Provided ${weightType} is an array of ${xWeight.length} ` +
          `element(s), but the model has ${numOutputs} outputs. ` +
          `Make sure a set of weights is provided for each model output.`);
    }
    return xWeight;
  } else if (
      typeof xWeight === 'object' && Object.keys(xWeight).length > 0 &&
      typeof (xWeight as ClassWeightMap)[Object.keys(xWeight)[0]] ===
          'object') {
    const output: ClassWeight[] = [];
    outputNames.forEach(outputName => {
      if (outputName in xWeight) {
        output.push((xWeight as ClassWeightMap)[outputName]);
      } else {
        output.push(null);
      }
    });
    return output;
  } else {
    throw new Error(
        `The model has multiple (${numOutputs}) outputs, ` +
        `so ${weightType} must be either an array with ` +
        `${numOutputs} elements or an object with ${outputNames} keys. ` +
        `Provided ${weightType} not understood: ${JSON.stringify(xWeight)}`);
  }
}

/**
 * Standardize class weighting objects.
 *
 * This function takes a single class-weighting object, an array of them,
 * or a map from output name to class-weighting object. It compares it to the
 * output name(s) of the model, base on which it outputs an array of
 * class-weighting objects of which the length matches the number of outputs.
 *
 * @param classWeight Input class-weighting object(s).
 * @param outputNames All output name(s) of the model.
 * @return An array of class-weighting objects. The length of the array matches
 *   the model's number of outputs.
 */
export function standardizeClassWeights(
    classWeight: ClassWeight|ClassWeight[]|ClassWeightMap,
    outputNames: string[]): ClassWeight[] {
  return standardizeSampleOrClassWeights(
      classWeight, outputNames, 'classWeight');
}

export function standardizeSampleWeights(
    classWeight: ClassWeight|ClassWeight[]|ClassWeightMap,
    outputNames: string[]): ClassWeight[] {
  return standardizeSampleOrClassWeights(
      classWeight, outputNames, 'sampleWeight');
}

/**
 * Standardize by-sample and/or by-class weights for training.
 *
 * Note that this function operates on one model output at a time. For a model
 * with multiple outputs, you must call this function multiple times.
 *
 * @param y The target tensor that the by-sample and/or by-class weight is for.
 *     The values of y are assumed to encode the classes, either directly
 *     as an integer index, or as one-hot encoding.
 * @param sampleWeight By-sample weights.
 * @param classWeight By-class weights: an object mapping class indices
 *     (integers) to a weight (float) to apply to the model's loss for the
 *     samples from this class during training. This can be useful to tell the
 *     model to "pay more attention" to samples from an under-represented class.
 * @param sampleWeightMode The mode for the sample weights.
 * @return A Promise of weight tensor, of which the size of the first dimension
 *     matches that of `y`.
 */
export async function standardizeWeights(
    y: Tensor, sampleWeight?: Tensor, classWeight?: ClassWeight,
    sampleWeightMode?: 'temporal'): Promise<Tensor> {
  if (sampleWeight != null || sampleWeightMode != null) {
    // TODO(cais): Once 'temporal' mode is implemented, document it in the doc
    // string.
    throw new Error('Support sampleWeight is not implemented yet');
  }

  if (classWeight != null) {
    // Apply class weights per sample.
    const yClasses: Tensor1D = tidy(() => {
      if (y.shape.length === 1) {
        // Assume class indices.
        return y.clone() as Tensor1D;
      } else if (y.shape.length === 2) {
        if (y.shape[1] > 1) {
          // Assume one-hot encoding of classes.
          const axis = 1;
          return y.argMax(axis);
        } else if (y.shape[1] === 1) {
          // Class index.
          return y.reshape([y.shape[0]]);
        } else {
          throw new Error(
              `Encountered unexpected last-dimension size (${y.shape[1]}) ` +
              `during handling of class weights. The size is expected to be ` +
              `>= 1.`);
        }
      } else {
        throw new Error(
            `Unexpected rank of target (y) tensor (${y.rank}) during ` +
            `handling of class weights. The rank is expected to be 1 or 2.`);
      }
    });

    const yClassIndices = Array.from(await yClasses.data());
    dispose(yClasses);
    const classSampleWeight: number[] = [];
    yClassIndices.forEach(classIndex => {
      if (classWeight[classIndex] == null) {
        throw new Error(
            `classWeight must contain all classes in the training data. ` +
            `The class ${classIndex} exists in the data but not in ` +
            `classWeight`);
      } else {
        classSampleWeight.push(classWeight[classIndex]);
      }
    });

    return tensor1d(classSampleWeight, 'float32');
  } else {
    return null;
  }
}

/**
 * Apply per-sample weights on the loss values from a number of samples.
 *
 * @param losses Loss tensor of shape `[batchSize]`.
 * @param sampleWeights Per-sample weight tensor of shape `[batchSize]`.
 * @returns Tensor of the same shape as`losses`.
 */
export function computeWeightedLoss(losses: Tensor, sampleWeights: Tensor) {
  return mul(losses, sampleWeights);
}
