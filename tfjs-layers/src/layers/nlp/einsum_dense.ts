/**
 * @license
 * Copyright 2023 Google LLC.
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

/**
 *  TFJS-based einsum dense layer.
 */

/* Original source: keras/layers/core/einsum_dense.py */
import { Tensor, Tensor2D, einsum, serialization, tidy } from '@tensorflow/tfjs-core';

import { Activation, getActivation, serializeActivation } from '../../activations';
import { Constraint, ConstraintIdentifier, getConstraint, serializeConstraint } from '../../constraints';
import { Layer, LayerArgs } from '../../engine/topology';
import { ValueError } from '../../errors';
import { Initializer, InitializerIdentifier, getInitializer, serializeInitializer } from '../../initializers';
import { ActivationIdentifier } from '../../keras_format/activation_config';
import { Shape } from '../../keras_format/common';
import { Regularizer, RegularizerIdentifier, getRegularizer, serializeRegularizer } from '../../regularizers';
import { Kwargs } from '../../types';
import { LayerVariable } from '../../variables';

/**
 * Analyzes an einsum string to determine the required weight shape.
 */
export function analyzeEinsumString(
  equation: string,
  biasAxes: string,
  inputShape: Shape,
  outputShape: Shape
): [Shape, Shape, Shape] {
  const dotReplacedString = equation.replace(/\.\.\./g, '0');

  // This is the case where no ellipses are present in the string.
  let splitString =
    dotReplacedString.match(/([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)/);
  if (splitString) {
    return analyzeSplitString(
      splitString, biasAxes, inputShape, outputShape);
  }

  // This is the case where ellipses are present on the left.
  splitString =
    dotReplacedString.match(/0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)/);
  if (splitString) {
    return analyzeSplitString(
      splitString, biasAxes, inputShape, outputShape, true);
  }

  // This is the case where ellipses are present on the right.
  splitString =
    dotReplacedString.match(/([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0/);
  if (splitString) {
    return analyzeSplitString(
      splitString, biasAxes, inputShape, outputShape);
  }

  throw new ValueError(
    `Invalid einsum equation '${equation}'. Equations must be in the form ` +
    '[X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]....'
  );
}

/**
 * Analyze an pre-split einsum string to find the weight shape.
 */
export function analyzeSplitString(
  splitString: RegExpMatchArray,
  biasAxes: string,
  inputShape: Shape,
  outputShape: Shape|number,
  leftElided = false
): [Shape, Shape, Shape] {
  const inputSpec = splitString[1];
  const weightSpec = splitString[2];
  const outputSpec = splitString[3];
  const elided = inputShape.length - inputSpec.length;

  const newOutputShape: Shape = Array.isArray(outputShape) ?
    outputShape.slice() : [outputShape];
  newOutputShape.unshift(inputShape[0]);

  if (elided > 0 && leftElided) {
    for(let i = 1; i < elided; i++) {
      // We already inserted the 0th input dimension at dim 0, so we need
      // to start at location 1 here.
      newOutputShape.splice(1, 0, inputShape[i]);
    }
  } else if (elided > 0 && !leftElided) {
    for(let i = inputShape.length - elided; i < inputShape.length; i++) {
      newOutputShape.push(inputShape[i]);
    }
  }

  const inputSpecArr = Array.from(inputSpec);
  const outputSpecArr = Array.from(outputSpec);
  let inputDimMap, outputDimMap;

  if (leftElided) {
    // If we have beginning dimensions elided, we need to use negative
    // indexing to determine where in the input dimension our values are.
    inputDimMap = new Map<string, number>(
      inputSpecArr.map((dim, i) => {
        // This converts any negative indices to positive ones.
        const idx = i + elided - inputShape.length;
        const positiveIdx =
          ((idx % inputShape.length) + inputShape.length) % inputShape.length;
        return [dim, positiveIdx];
      })
    );

    // Because we've constructed the full output shape already, we don't need
    // to do negative indexing.
    outputDimMap = new Map<string, number>(
      outputSpecArr.map((dim, i) => [dim, i + elided])
    );
  } else {
    inputDimMap = new Map<string, number>(
      inputSpecArr.map((dim, i) => [dim, i])
    );
    outputDimMap = new Map<string, number>(
      outputSpecArr.map((dim, i) => [dim, i])
    );
  }

  for (const dim of inputSpec) {
    const inputShapeAtDim = inputShape[inputDimMap.get(dim)];
    if (outputDimMap.has(dim)) {
      const outputShapeAtDim = newOutputShape[outputDimMap.get(dim)];
      if (outputShapeAtDim !== null && outputShapeAtDim !== inputShapeAtDim) {
        throw new ValueError(
          `Input shape and output shape do not match at shared dimension `+
          `'${dim}'. Input shape is ${inputShapeAtDim}, and output shape ` +
          `is ${outputShapeAtDim}.`
        );
      }
    }
  }

  for (const dim of outputSpec) {
    if (!inputSpec.includes(dim) && !weightSpec.includes(dim)) {
      throw new ValueError(
        `Dimension '${dim}' was specified in the output '${outputSpec}' ` +
        `but has no corresponding dimension in the input spec ` +
        `'${inputSpec}' or weight spec '${weightSpec}'`
      );
    }
  }

  const weightShape: Shape = [];
  for (const dim of weightSpec) {
    if (inputDimMap.has(dim)) {
      weightShape.push(inputShape[inputDimMap.get(dim)]);
    } else if (outputDimMap.has(dim)) {
      weightShape.push(newOutputShape[outputDimMap.get(dim)]);
    } else {
      throw new ValueError(
        `Weight dimension '${dim}' did not have a match in either the ` +
        `input spec '${inputSpec}' or the output spec '${outputSpec}'. For ` +
        `this layer, the weight must be fully specified.`
      );
    }
  }

  let biasShape: Shape;
  if (biasAxes != null) {
    const numLeftElided = leftElided ? elided : 0;
    const idxMap: { [char: string]: number } = {};
    for (let i = 0; i < outputSpec.length; i++) {
      idxMap[outputSpec[i]] = newOutputShape[i + numLeftElided];
    }

    for (const char of biasAxes) {
      if (!outputSpec.includes(char)) {
        throw new ValueError(
          `Bias dimension '${char}' was requested, but is not part of the ` +
          `output spec '${outputSpec}'`
        );
      }
    }

    const firstBiasLocation = Math.min(
      ...biasAxes.split('').map(char => outputSpec.indexOf(char))
    );
    const biasOutputSpec = outputSpec.slice(firstBiasLocation);

    biasShape = biasOutputSpec.split('').map(char =>
      biasAxes.includes(char) ? idxMap[char] : 1
    );

    if (!leftElided) {
      for (let i = 0; i < elided; i++) {
        biasShape.push(1);
      }
    }
  } else {
    biasShape = null;
  }
  return [weightShape, biasShape, newOutputShape];
}

export declare interface EinsumDenseArgs extends LayerArgs {
  /**
   * An equation describing the einsum to perform. This equation must be a
   * valid einsum string of the form `ab,bc->ac`, `...ab,bc->...ac`, or
   * `ab...,bc->ac...` where 'ab', 'bc', and 'ac' can be any valid einsum
   * axis expression sequence.
   */
  equation: string;

  /**
   * The expected shape of the output tensor (excluding the batch dimension and
   * any dimensions represented by ellipses). You can specify None for any
   * dimension that is unknown or can be inferred from the input shape.
   */
  outputShape: Shape|number;

  /**
   * Activation function to use. If you don't specify anything, no activation
   * is applied (that is, a "linear" activation: `a(x) = x`).
   */
  activation?: ActivationIdentifier;

  /**
   * A string containing the output dimension(s) to apply a bias to. Each
   * character in the `biasAxes` string should correspond to a character
   * in the output portion of the `equation` string.
   */
  biasAxes?: string;

  /**
   * Initializer for the `kernel` weights matrix.
   * Defaults to `"glorotUniform"`.
   */
  kernelInitializer?: InitializerIdentifier;

  /**
   * Initializer for the bias vector.
   * Defaults to `"zeros"`.
   */
  biasInitializer?: InitializerIdentifier;

  /**
   * Regularizer function applied to the `kernel` weights matrix.
   */
  kernelRegularizer?: RegularizerIdentifier;

  /**
   * Regularizer function applied to the bias vector.
   */
  biasRegularizer?: RegularizerIdentifier;

  /**
   * Regularizer function applied to the output of the layer (its "activation").
   */
  activityRegularizer?: RegularizerIdentifier;

  /**
   * Constraint function applied to the `kernel` weights matrix.
   */
  kernelConstraint?: ConstraintIdentifier;

  /**
   * Constraint function applied to the bias vector.
   */
  biasConstraint?: ConstraintIdentifier;
}

/**
 * A layer that uses `tf.einsum` as the backing computation.
 *
 * This layer can perform einsum calculations of arbitrary dimensionality.
 *
 * Examples:
 *
 * **Biased dense layer with einsums**
 *
 * This example shows how to instantiate a standard Keras dense layer using
 * einsum operations. This example is equivalent to
 * tf.layers.Dense({units: 64, useBias: true})`.
 *
 * const layer = new EinsumDense({
 *    equation: "ab,bc->ac", outputShape: 4, biasAxes: "c"});
 * const inputTensor = tf.input({shape: [32]});
 * const outputTensor = layer.call(inputTensor);
 * console.log(outputTensor);  // [null, 64]
 *
 * **Applying a dense layer to a sequence**
 *
 * This example shows how to instantiate a layer that applies the same dense
 * operation to every element in a sequence. Here, the `outputShape` has two
 * values (since there are two non-batch dimensions in the output); the first
 * dimension in the `outputShape` is `null`, because the sequence dimension
 * `b` has an unknown shape.
 *
 * ```js
 * const layer = new EinsumDense({
 *    equation: "abc,cd->abd", outputShape: [null, 64], biasAxes: "d"});
 * const inputTensor = tf.input({shape: [32, 128]});
 * const outputTensor = layer.call(inputTensor);
 * console.log(outputTensor);  // [null, 32, 64]
 * ```
 *
 * **Applying a dense layer to a sequence using ellipses**
 *
 * This example shows how to instantiate a layer that applies the same dense
 * operation to every element in a sequence, but uses the ellipsis notation
 * instead of specifying the batch and sequence dimensions.
 *
 * Because we are using ellipsis notation and have specified only one axis, the
 * `outputShape` arg is a single value. When instantiated in this way, the
 * layer can handle any number of sequence dimensions - including the case
 * where no sequence dimension exists.
 *
 * ```js
 * const layer = new EinsumDense({
 *    equation: "...x,xy->...y", outputShape: 64, biasAxes: "y"});
 * const inputTensor = tf.input({shape: [32, 128]});
 * const outputTensor = layer.call(inputTensor);
 * console.log(outputTensor);  // [null, 32, 64]
 * ``
 */
export class EinsumDense extends Layer {
  /** @nocollapse */
  static readonly className = 'EinsumDense';
  private readonly equation: string;
  private readonly biasAxes: string;
  private readonly partialOutputShape: Shape;
  private readonly activation: Activation;
  private readonly kernelInitializer: Initializer;
  private readonly biasInitializer: Initializer;
  private readonly kernelRegularizer: Regularizer;
  private readonly biasRegularizer: Regularizer;
  private readonly kernelConstraint: Constraint;
  private readonly biasConstraint: Constraint;
  private fullOutputShape: Shape;
  private _kernel: LayerVariable;
  private _bias: LayerVariable;

  constructor(args: EinsumDenseArgs) {
    super(args);
    this.equation = args.equation;
    this.biasAxes = args.biasAxes;
    this.partialOutputShape =
      Array.isArray(args.outputShape) ? args.outputShape : [args.outputShape];
    this.activation = getActivation(args.activation);
    this.kernelInitializer = getInitializer(
      args.kernelInitializer ?? 'glorotUniform');
    this.biasInitializer = getInitializer(args.biasInitializer ?? 'zeros');
    this.kernelRegularizer = getRegularizer(args.kernelRegularizer);
    this.biasRegularizer = getRegularizer(args.biasRegularizer);
    this.kernelConstraint = getConstraint(args.kernelConstraint);
    this.biasConstraint = getConstraint(args.biasConstraint);
  }

  get kernel(): LayerVariable {
    return this._kernel;
  }

  get bias(): LayerVariable {
    return this._bias;
  }

  override build(inputShape: Shape): void {
    const [kernelShape, biasShape, fullOutputShape] = analyzeEinsumString(
      this.equation,
      this.biasAxes,
      inputShape,
      this.partialOutputShape
    );
    this.fullOutputShape = fullOutputShape;
    this._kernel = this.addWeight(
      'kernel',
      kernelShape,
      this.dtype,
      this.kernelInitializer,
      this.kernelRegularizer,
      true,
      this.kernelConstraint,
    );

    if (biasShape != null) {
      this._bias = this.addWeight(
        'bias',
        biasShape,
        this.dtype,
        this.biasInitializer,
        this.biasRegularizer,
        true,
        this.biasConstraint,
      );
    } else {
      this._bias = null;
    }
    super.build(inputShape);
  }

  override computeOutputShape(_: Shape): Shape {
    return this.fullOutputShape;
  }

  override getConfig(): serialization.ConfigDict {
    const config = {
      outputShape: this.partialOutputShape,
      equation: this.equation,
      activation: serializeActivation(this.activation),
      biasAxes: this.biasAxes,
      kernelInitializer: serializeInitializer(this.kernelInitializer),
      biasInitializer: serializeInitializer(this.biasInitializer),
      kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
      biasRegularizer: serializeRegularizer(this.biasRegularizer),
      kernelConstraint: serializeConstraint(this.kernelConstraint),
      biasConstraint: serializeConstraint(this.biasConstraint),
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  override call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor2D {
    return tidy(() => {
      inputs = Array.isArray(inputs) ? inputs : [inputs];
      let ret = einsum(this.equation, ...inputs, this.kernel.read());
      if (this.bias != null) {
        ret = ret.add(this.bias.read());
      }
      if (this.activation != null) {
        ret = this.activation.apply(ret);
      }
      return ret;
    });
  }
}
serialization.registerClass(EinsumDense);
