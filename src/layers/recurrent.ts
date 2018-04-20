/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * TensorFlow.js Layers: Recurrent Neural Network Layers.
 */

import {doc, Tensor, util} from '@tensorflow/tfjs-core';

// tslint:disable:max-line-length
import {ActivationFn, ActivationIdentifier, getActivation, serializeActivation} from '../activations';
import * as K from '../backend/tfjs_backend';
import {Constraint, ConstraintIdentifier, getConstraint, serializeConstraint} from '../constraints';
import {InputSpec} from '../engine/topology';
import {Layer, LayerConfig} from '../engine/topology';
import {AttributeError, NotImplementedError, ValueError} from '../errors';
import {getInitializer, Initializer, InitializerIdentifier, Ones, serializeInitializer} from '../initializers';
import {getRegularizer, Regularizer, RegularizerIdentifier, serializeRegularizer} from '../regularizers';
import {DType, Shape, SymbolicTensor} from '../types';
import {ConfigDict, LayerVariable} from '../types';
import * as generic_utils from '../utils/generic_utils';
import * as math_utils from '../utils/math_utils';

import {deserialize} from './serialization';

// tslint:enable:max-line-length

export interface BaseRNNLayerConfig extends LayerConfig {
  /**
   * A RNN cell instance. A RNN cell is a class that has:
   *   - a `call()` method, which takes `[Tensor, Tensor]` as the
   *     first input argument. The first item is the input at time t, and
   *     second item is the cell state at time t.
   *     The `call()` method returns `[outputAtT, statesAtTPlus1]`.
   *     The `call()` method of the cell can also take the argument `constants`,
   *     see section "Note on passing external constants" below.
   *     Porting Node: PyKeras overrides the `call()` signature of RNN cells,
   *       which are Layer subtypes, to accept two arguments. tfjs-layers does
   *       not do such overriding. Instead we preseve the `call()` signature,
   *       which due to its `Tensor|Tensor[]` argument and return value, is
   *       flexible enough to handle the inputs and states.
   *   - a `stateSize` attribute. This can be a single integer (single state)
   *     in which case it is the size of the recurrent state (which should be
   *     the same as the size of the cell output). This can also be an Array of
   *     integers (one size per state). In this case, the first entry
   *     (`stateSize[0]`) should be the same as the size of the cell output.
   * It is also possible for `cell` to be a list of RNN cell instances, in which
   * case the cells get stacked on after the other in the RNN, implementing an
   * efficient stacked RNN.
   */
  cell?: RNNCell|RNNCell[];

  /**
   * Whether to return the last output in the output sequence, or the full
   * sequence.
   */
  returnSequences?: boolean;

  /**
   * Whether to return the last state in addition to the output.
   */
  returnState?: boolean;

  /**
   * If `true`, process the input sequence backwards and return the reversed
   * sequence (default: `false`).
   */
  goBackwards?: boolean;

  /**
   * If `true`, the last state for each sample at index i in a batch will be
   * used as initial state of the sample of index i in the following batch
   * (default: `false`).
   */
  stateful?: boolean;

  /**
   * If `true`, the network will be unrolled, else a symbolic loop will be
   * used. Unrolling can speed-up a RNN, although it tends to be more memory-
   * intensive. Unrolling is only suitable for short sequences (default:
   * `false`).
   * Porting Note: tfjs-layers has an imperative backend. RNNs are executed with
   *   normal TypeScript control flow. Hence this property is inapplicable and
   *   ignored in tfjs-layers.
   */
  unroll?: boolean;

  /**
   * Dimensionality of the input (integer).
   *   This option (or alternatively, the option `inputShape`) is required when
   *   this layer is used as the first layer in a model.
   */
  inputDim?: number;

  /**
   * Length of the input sequences, to be specified when it is constant.
   * This argument is required if you are going to connect `Flatten` then
   * `Dense` layers upstream (without it, the shape of the dense outputs cannot
   * be computed). Note that if the recurrent layer is not the first layer in
   * your model, you would need to specify the input length at the level of the
   * first layer (e.g., via the `inputShape` option).
   */
  inputLength?: number;
}

/**
 * RNNLayerConfig is identical to BaseRNNLayerConfig, except it makes the
 * `cell` property required. This interface is  to be used with constructors
 * of concrete RNN layer sbutypes.
 */
export interface RNNLayerConfig extends BaseRNNLayerConfig {
  cell: RNNCell|RNNCell[];
}

/**
 * Base class for recurrent layers.
 *
 * Input shape:
 *   3D tensor with shape `[batchSize, timeSteps, inputDim]`.
 *
 * Output shape:
 *   - if `returnState`, an Array of tensors (i.e., `Tensor`s). The first
 *     tensor is the output. The remaining tensors are the states at the
 *     last time step, each with shape `[batchSize, units]`.
 *   - if `returnSequences`, the output will have shape
 *     `[batchSize, timeSteps, units]`.
 *   - else, the output will have shape `[batchSize, units]`.
 *
 * Masking:
 *   This layer supports masking for input data with a variable number
 *   of timesteps. To introduce masks to your data,
 *   use an embedding layer with the `mask_zero` parameter
 *   set to `True`.
 *
 * Notes on using statefulness in RNNs:
 *   You can set RNN layers to be 'stateful', which means that the states
 *   computed for the samples in one batch will be reused as initial states
 *   for the samples in the next batch. This assumes a one-to-one mapping
 *   between samples in different successive batches.
 *
 *   To enable statefulness:
 *     - specify `stateful: true` in the layer constructor.
 *     - specify a fixed batch size for your model, by passing
 *       if sequential model:
 *         `batchInputShape=[...]` to the first layer in your model.
 *       else for functional model with 1 or more Input layers:
 *         `batchShape=[...]` to all the first layers in your model.
 *       This is the expected shape of your inputs *including the batch size*.
 *       It should be a tuple of integers, e.g. `(32, 10, 100)`.
 *     - specify `shuffle=False` when calling fit().
 *
 *   To reset the states of your model, call `.reset_states()` on either
 *   a specific layer, or on your entire model.
 *
 * Note on specifying the initial state of RNNs
 *   You can specify the initial state of RNN layers symbolically by
 *   calling them with the option `initialState`. The value of
 *   `initialState` should be a tensor or list of tensors representing
 *   the initial state of the RNN layer.
 *
 *   You can specify the initial state of RNN layers numerically by
 *   calling `resetStates` with the keyword argument `states`. The value of
 *   `states` should be a numpy array or list of numpy arrays representing
 *   the initial state of the RNN layer.
 *
 * Note on passing external constants to RNNs
 *   You can pass "external" constants to the cell using the `constants`
 *   keyword argument of `RNN.call` method. This requires that the `cell.call`
 *   method accepts the same keyword argument `constants`. Such constants
 *   can be used to conditon the cell transformation on additional static inputs
 *   (not changing over time), a.k.a an attention mechanism.
 */
export class RNN extends Layer {
  public readonly cell: RNNCell;
  public readonly returnSequences: boolean;
  public readonly returnState: boolean;
  public readonly goBackwards: boolean;
  public readonly unroll: boolean;

  public stateSpec: InputSpec[];
  public states: Tensor[];

  private numConstants: number;

  constructor(config: RNNLayerConfig) {
    super(config);
    let cell: RNNCell;
    if (config.cell == null) {
      throw new ValueError(
          'cell property is missing for the constructor of RNN.');
    } else if (Array.isArray(config.cell)) {
      cell = new StackedRNNCells({cells: config.cell});
    } else {
      cell = config.cell;
    }
    if ((cell as RNNCell).stateSize == null) {
      throw new ValueError(
          'The RNN cell should have an attribute `stateSize` (tuple of ' +
          'integers, one integer per RNN state).');
    }
    this.cell = cell;
    this.returnSequences =
        config.returnSequences == null ? false : config.returnSequences;
    this.returnState = config.returnState == null ? false : config.returnState;
    this.goBackwards = config.goBackwards == null ? false : config.goBackwards;
    this._stateful = config.stateful == null ? false : config.stateful;
    this.unroll = config.unroll == null ? false : config.unroll;

    this.supportsMasking = true;
    this.inputSpec = [new InputSpec({ndim: 3})];
    this.stateSpec = null;
    this.states = null;
    // TODO(cais): Add constantsSpec and numConstants.
    this.numConstants = null;
    // TODO(cais): Look into the use of initial_state in the kwargs of the
    //   constructor.
  }

  // Porting Note: This is the equivalent of `RNN.states` property getter in
  //   PyKeras.
  getStates(): Tensor[] {
    if (this.states == null) {
      const numStates =
          Array.isArray(this.cell.stateSize) ? this.cell.stateSize.length : 1;
      return math_utils.range(0, numStates).map(x => null);
    } else {
      return this.states;
    }
  }

  // Porting Note: This is the equivalent of the `RNN.states` property setter in
  //   PyKeras.
  setStates(states: Tensor[]): void {
    this.states = states;
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    if (generic_utils.isArrayOfShapes(inputShape)) {
      inputShape = (inputShape as Shape[])[0];
    }
    inputShape = inputShape as Shape;

    // TODO(cais): Remove the casting once stacked RNN cells become supported.
    let stateSize = this.cell.stateSize;
    if (!Array.isArray(stateSize)) {
      stateSize = [stateSize];
    }
    const outputDim = stateSize[0];
    let outputShape: Shape|Shape[];
    if (this.returnSequences) {
      outputShape = [inputShape[0], inputShape[1], outputDim];
    } else {
      outputShape = [inputShape[0], outputDim];
    }

    if (this.returnState) {
      const stateShape: Shape[] = [];
      for (const dim of stateSize) {
        stateShape.push([inputShape[0], dim]);
      }
      return [outputShape].concat(stateShape);
    } else {
      return outputShape;
    }
  }

  computeMask(inputs: Tensor|Tensor[], mask?: Tensor|Tensor[]): Tensor {
    throw new NotImplementedError(
        'computeMask has not been implemented for RNN yet');
  }

  public build(inputShape: Shape|Shape[]): void {
    // Note inputShape will be an Array of Shapes of initial states and
    // constants if these are passed in apply().
    const constantShape: Shape[] = null;
    if (this.numConstants != null) {
      throw new NotImplementedError(
          'Constants support is not implemented in RNN yet.');
    }

    if (generic_utils.isArrayOfShapes(inputShape)) {
      inputShape = (inputShape as Shape[])[0];
    }
    inputShape = inputShape as Shape;

    const batchSize: number = this.stateful ? inputShape[0] : null;
    const inputDim = inputShape[inputShape.length - 1];
    this.inputSpec[0] = new InputSpec({shape: [batchSize, null, inputDim]});

    // Allow cell (if RNNCell Layer) to build before we set or validate
    // stateSpec.
    const stepInputShape = [inputShape[0]].concat(inputShape.slice(2));
    if (constantShape != null) {
      throw new NotImplementedError(
          'Constants support is not implemented in RNN yet.');
    } else {
      this.cell.build(stepInputShape);
    }

    // Set or validate stateSpec.
    let stateSize: number[];
    if (Array.isArray((this.cell as RNNCell).stateSize)) {
      stateSize = this.cell.stateSize as number[];
    } else {
      stateSize = [this.cell.stateSize as number];
    }

    if (this.stateSpec != null) {
      if (!util.arraysEqual(
              this.stateSpec.map(spec => spec.shape[spec.shape.length - 1]),
              stateSize)) {
        throw new ValueError(
            `An initialState was passed that is not compatible with ` +
            `cell.stateSize. Received stateSpec=${this.stateSpec}; ` +
            `However cell.stateSize is ${this.cell.stateSize}`);
      }
    } else {
      this.stateSpec =
          stateSize.map(dim => new InputSpec({shape: [null, dim]}));
    }
    if (this.stateful) {
      throw new NotImplementedError(
          'stateful RNN layer is not implemented yet');
      // TODO(cais): Uncomment the following line once stateful = true is
      //   implemented.
      // this.resetStates();
    }
  }

  resetStates(states?: Tensor|Tensor[]): void {
    if (!this.stateful) {
      throw new AttributeError(
          'Cannot call resetState() on an RNN Layer that is not stateful.');
    }
    const batchSize = this.inputSpec[0].shape[0];
    if (batchSize == null) {
      throw new ValueError(
          'If an RNN is stateful, it needs to know its batch size. Specify ' +
          'the batch size of your input tensors: \n' +
          '- If using a Sequential model, specify the batch size by passing ' +
          'a `batchInputShape` option to your first layer.\n' +
          '- If using the functional API, specify the batch size by ' +
          'passing a `batchShape` option to your Input layer.');
    }
    // Initialize state if null.
    if (this.states == null) {
      if (Array.isArray(this.cell.stateSize)) {
        this.states = this.cell.stateSize.map(dim => K.zeros([batchSize, dim]));
      } else {
        this.states = [K.zeros([batchSize, this.cell.stateSize])];
      }
    } else if (states == null) {
      if (Array.isArray(this.cell.stateSize)) {
        this.states = this.cell.stateSize.map(dim => K.zeros([batchSize, dim]));
      } else {
        this.states[0] = K.zeros([batchSize, this.cell.stateSize]);
      }
    } else {
      if (!Array.isArray(states)) {
        states = [states];
      }
      if (states.length !== this.states.length) {
        throw new ValueError(
            `Layer ${this.name} expects ${this.states.length} state(s), ` +
            `but it received ${states.length} state value(s). Input ` +
            `received: ${states}`);
      }
      for (let index = 0; index < this.states.length; ++index) {
        const value = states[index];
        const dim = Array.isArray(this.cell.stateSize) ?
            this.cell.stateSize[index] :
            this.cell.stateSize;
        const expectedShape = [batchSize, dim];
        if (!util.arraysEqual(value.shape, expectedShape)) {
          throw new ValueError(
              `State ${index} is incompatible with layer ${this.name}: ` +
              `expected shape=${expectedShape}, received shape=${value.shape}`);
        }
        this.states[index] = value;
      }
    }
  }

  /**
   * Standardize `apply()` args to a single list of tensor inputs.
   *
   * When running a model loaded from file, the input tensors `initialState` and
   * `constants` are passed to `RNN.apply()` as part of `inputs` instead of the
   * dedicated kwargs fields. `inputs` consists of
   * `[inputs, initialState0, initialState1, ..., constant0, constant1]` in this
   * case.
   * This method makes sure that arguments are
   * separated and that `initialState` and `constants` are `Array`s of tensors
   * (or None).
   *
   * @param inputs Tensor or `Array` of  tensors.
   * @param initialState Tensor or `Array` of tensors or `null`/`undefined`.
   * @param constants Tensor or `Array` of tensors or `null`/`undefined`.
   * @returns An object consisting of
   *   inputs: A tensor.
   *   initialState: `Array` of tensors or `null`.
   *   constants: `Array` of tensors or `null`.
   * @throws ValueError, if `inputs` is an `Array` but either `initialState` or
   *   `constants` is provided.
   */
  protected standardizeArgs(
      inputs: Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[],
      initialState: Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[],
      constants: Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[]): {
    inputs: Tensor|SymbolicTensor,
    initialState: Tensor[]|SymbolicTensor[],
    constants: Tensor[]|SymbolicTensor[]
  } {
    if (Array.isArray(inputs)) {
      if (initialState != null || constants != null) {
        throw new ValueError(
            'When inputs is an array, neither initialState or constants ' +
            'should be provided');
      }
      if (this.numConstants != null) {
        constants =
            inputs.slice(inputs.length - this.numConstants, inputs.length);
        inputs = inputs.slice(0, inputs.length - this.numConstants);
      }
      if (inputs.length > 1) {
        initialState = inputs.slice(1, inputs.length);
      }
      inputs = inputs[0];
    }

    function toListOrNull(x: Tensor|Tensor[]|SymbolicTensor|
                          SymbolicTensor[]): Tensor[]|SymbolicTensor[] {
      if (x == null || Array.isArray(x)) {
        return x as Tensor[] | SymbolicTensor[];
      } else {
        return [x] as Tensor[] | SymbolicTensor[];
      }
    }

    initialState = toListOrNull(initialState);
    constants = toListOrNull(constants);

    return {inputs, initialState, constants};
  }

  apply(
      inputs: Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[],
      // tslint:disable-next-line:no-any
      kwargs?: any): Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[] {
    // TODO(cais): Figure out whether initialState is in kwargs or inputs.
    let initialState: Tensor[]|SymbolicTensor[] =
        kwargs == null ? null : kwargs['initialState'];
    let constants: Tensor[]|SymbolicTensor[] =
        kwargs == null ? null : kwargs['constants'];
    if (kwargs == null) {
      kwargs = {};
    }

    const standardized = this.standardizeArgs(inputs, initialState, constants);
    inputs = standardized.inputs;
    initialState = standardized.initialState;
    constants = standardized.constants;

    // If any of `initial_state` or `constants` are specified and are
    // `SymbolicTensor`s, then add them to the inputs and temporarily modify the
    // input_spec to include them.

    let additionalInputs: Array<Tensor|SymbolicTensor> = [];
    let additionalSpecs: InputSpec[] = [];
    if (initialState != null) {
      kwargs['initialState'] = initialState;
      additionalInputs = additionalInputs.concat(initialState);
      this.stateSpec = [];
      for (const state of initialState) {
        this.stateSpec.push(new InputSpec({shape: state.shape}));
      }
      // TODO(cais): Use the following instead.
      // this.stateSpec = initialState.map(state => new InputSpec({shape:
      // state.shape}));
      additionalSpecs = additionalSpecs.concat(this.stateSpec);
    }
    if (constants != null) {
      kwargs['constants'] = constants;
      additionalInputs = additionalInputs.concat(constants);
      // TODO(cais): Add this.constantsSpec.
      this.numConstants = constants.length;
    }

    const isTensor = additionalInputs[0] instanceof SymbolicTensor;
    if (isTensor) {
      // Compute full input spec, including state and constants.
      const fullInput =
          [inputs].concat(additionalInputs) as Tensor[] | SymbolicTensor[];
      const fullInputSpec = this.inputSpec.concat(additionalSpecs);
      // Perform the call with temporarily replaced inputSpec.
      const originalInputSpec = this.inputSpec;
      this.inputSpec = fullInputSpec;
      const output = super.apply(fullInput, kwargs);
      this.inputSpec = originalInputSpec;
      return output;
    } else {
      return super.apply(inputs, kwargs);
    }
  }

  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    // Input shape: `[samples, time (padded with zeros), input_dim]`.
    // Note that the .build() method of subclasses **must** define
    // this.inputSpec and this.stateSpec owith complete input shapes.
    const mask = kwargs == null ? null : kwargs['mask'];
    const training = kwargs == null ? null : kwargs['training'];
    let initialState: Tensor[] = kwargs == null ? null : kwargs['initialState'];

    inputs = generic_utils.getExactlyOneTensor(inputs);
    if (initialState == null) {
      if (this.stateful) {
        throw new NotImplementedError(
            'stateful RNN layer is not implemented yet.');
      } else {
        initialState = this.getInitialState(inputs);
      }
    }

    if (mask != null) {
      throw new NotImplementedError('Masking is not implemented for RNN yet');
    }

    const numStates =
        Array.isArray(this.cell.stateSize) ? this.cell.stateSize.length : 1;
    if (initialState.length !== numStates) {
      throw new ValueError(
          `RNN Layer has ${numStates} state(s) but was passed ` +
          `${initialState.length} initial state(s).`);
    }
    const inputShape = inputs.shape;
    const timesteps = inputShape[1];
    if (this.unroll) {
      console.warn(
          'Ignoring unroll = true for RNN layer, due to imperative backend.');
    }

    // tslint:disable-next-line:no-any
    const cellCallKwargs: {[key: string]: any} = {training};

    // TODO(cais): Add support for constants.
    const step = (inputs: Tensor, states: Tensor[]) => {
      // `inputs` and `states` are concatenated to form a single `Array` of
      // `Tensor`s as the input to `cell.call()`.
      const outputs =
          this.cell.call([inputs].concat(states), cellCallKwargs) as Tensor[];
      // Marshall the return value into output and new states.
      return [outputs[0], outputs.slice(1)] as [Tensor, Tensor[]];
    };

    // TODO(cais): Add support for constants.
    // TODO(cais): Add support for masks.

    const rnnOutputs = K.rnn(
        step, inputs, initialState, this.goBackwards, null, null, this.unroll,
        timesteps);
    const lastOutput = rnnOutputs[0];
    const outputs = rnnOutputs[1];
    const states = rnnOutputs[2];

    if (this.stateful) {
      throw new NotImplementedError(
          'stateful RNN layer is not implemented yet');
    }

    const output = this.returnSequences ? outputs : lastOutput;

    // TODO(cais): Porperty set learning phase flag.

    if (this.returnState) {
      return [output].concat(states);
    } else {
      return output;
    }
  }

  getInitialState(inputs: Tensor): Tensor[] {
    // Build an all-zero tensor of shape [samples, outputDim].
    // [Samples, timeSteps, inputDim].
    let initialState = K.zeros(inputs.shape);
    // [Samples].
    initialState = K.sum(initialState, [1, 2]);
    initialState = K.expandDims(initialState);  // [Samples, 1].

    if (Array.isArray(this.cell.stateSize)) {
      return this.cell.stateSize.map(
          dim => dim > 1 ? K.tile(initialState, [1, dim]) : initialState);
    } else {
      return this.cell.stateSize > 1 ?
          [K.tile(initialState, [1, this.cell.stateSize])] :
          [initialState];
    }
  }

  get trainableWeights(): LayerVariable[] {
    if (!this.trainable) {
      return [];
    }
    // Porting Note: In TypeScript, `this` is always an instance of `Layer`.
    return this.cell.trainableWeights;
  }

  get nonTrainableWeights(): LayerVariable[] {
    // Porting Note: In TypeScript, `this` is always an instance of `Layer`.
    if (!this.trainable) {
      return this.cell.weights;
    }
    return this.cell.nonTrainableWeights;
  }

  getClassName(): string {
    return 'RNN';
  }

  getConfig(): ConfigDict {
    const config: ConfigDict = {
      returnSequences: this.returnSequences,
      returnState: this.returnState,
      goBackwards: this.goBackwards,
      stateful: this.stateful,
      unroll: this.unroll,
    };
    if (this.numConstants != null) {
      config.numConstants = this.numConstants;
    }
    const cellConfig = this.cell.getConfig();
    config.cell = {
      className: this.cell.getClassName(),
      config: cellConfig,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
generic_utils.ClassNameMap.register('RNN', RNN);

/**
 * An RNNCell layer.
 */
// Porting Note: This is a common parent class for RNN cells. There is no
// equivalent of this in PyKeras. Having a common parent class forgoes the
//  need for `has_attr(cell, ...)` checks or its TypeScript equivalent.
@doc({heading: 'Layers', subheading: 'Classes'})
export abstract class RNNCell extends Layer {
  /**
   * Size(s) of the states.
   * For RNN cells with only a single state, this is a single integer.
   */
  public stateSize: number|number[];
}

export interface SimpleRNNCellLayerConfig extends LayerConfig {
  /**
   * units: Positive integer, dimensionality of the output space.
   */
  units: number;

  /**
   * Activation function to use.
   * Default: hyperbolic tangent ('tanh').
   * If you pass `null`,  'linear' activation will be applied.
   */
  activation?: ActivationIdentifier;

  /**
   * Whether the layer uses a bias vector.
   */
  useBias?: boolean;

  /**
   * Initializer for the `kernel` weights matrix, used for the linear
   * transformation of the inputs.
   */
  kernelInitializer?: InitializerIdentifier|Initializer;

  /**
   * Initializer for the `recurrentKernel` weights matrix, used for
   * linear transformation of the recurrent state.
   */
  recurrentInitializer?: InitializerIdentifier|Initializer;

  /**
   * Initializer for the bias vector.
   */
  biasInitializer?: InitializerIdentifier|Initializer;

  /**
   * Regularizer function applied to the `kernel` weights matrix.
   */
  kernelRegularizer?: RegularizerIdentifier|Regularizer;

  /**
   * Regularizer function applied to the `recurrent_kernel` weights matrix.
   */
  recurrentRegularizer?: RegularizerIdentifier|Regularizer;

  /**
   * Regularizer function applied to the bias vector.
   */
  biasRegularizer?: RegularizerIdentifier|Regularizer;

  /**
   * Constraint function applied to the `kernel` weights matrix.
   */
  kernelConstraint?: ConstraintIdentifier|Constraint;

  /**
   * Constraint function applied to the `recurrentKernel` weights matrix.
   */
  recurrentConstraint?: ConstraintIdentifier|Constraint;

  /**
   * Constraintfunction applied to the bias vector.
   */
  biasConstraint?: ConstraintIdentifier|Constraint;

  /**
   * Float number between 0 and 1. Fraction of the units to drop for the linear
   * transformation of the inputs.
   */
  dropout?: number;

  /**
   * Float number between 0 and 1. Fraction of the units to drop for the linear
   * transformation of the recurrent state.
   */
  recurrentDropout?: number;
}

/**
 * Cell class for `SimpleRNN`.
 *
 * `SimpleRNNCell` is distinct from the `RNN` subclass `SimpleRNN` in that its
 * `apply` method takes the input data of only a single time step and returns
 * the cell's output at the time step, while `SimpleRNN` takes the input data
 * over a number of time steps. For example:
 *
 * ```js
 * const cell = tf.layers.simpleRNNCell({units: 2});
 * const input = tf.input({shape: [10]});
 * const output = cell.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10]: This is the cell's output at a single time step. The 1st
 * // dimension is the unknown batch size.
 * ```
 *
 * Instance(s) of `SimpleRNNCell` can be used to construct `RNN` layers. The
 * most typical use of this workflow is to combine a number of cells into a
 * stacked RNN cell (i.e., `StackedRNNCell` internally) and use it to create an
 * RNN. For example:
 *
 * ```js
 * const cells = [
 *   tf.layers.simpleRNNCell({units: 4}),
 *   tf.layers.simpleRNNCell({units: 8}),
 * ];
 * const rnn = tf.layers.rnn({cell: cells, returnSequences: true});
 *
 * // Create an input with 10 time steps and a length-20 vector at each step.
 * const input = tf.input({shape: [10, 20]});
 * const output = rnn.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the last `SimpleRNNCell`'s number of units.
 * ```
 *
 * To create an `RNN` consisting of only *one* `SimpleRNNCell`, use the
 * `tf.layers.simpleRNN`.
 */
export class SimpleRNNCell extends RNNCell {
  readonly units: number;
  readonly activation: ActivationFn;
  readonly useBias: boolean;

  readonly kernelInitializer: Initializer;
  readonly recurrentInitializer: Initializer;
  readonly biasInitializer: Initializer;

  readonly kernelConstraint: Constraint;
  readonly recurrentConstraint: Constraint;
  readonly biasConstraint: Constraint;

  readonly kernelRegularizer: Regularizer;
  readonly recurrentRegularizer: Regularizer;
  readonly biasRegularizer: Regularizer;

  readonly dropout: number;
  readonly recurrentDropout: number;

  readonly stateSize: number;

  kernel: LayerVariable;
  recurrentKernel: LayerVariable;
  bias: LayerVariable;

  readonly DEFAULT_ACTIVATION = 'tanh';
  readonly DEFAULT_KERNEL_INITIALIZER = 'glorotNormal';
  readonly DEFAULT_RECURRENT_INITIALIZER = 'orthogonal';
  readonly DEFAULT_BIAS_INITIALIZER: InitializerIdentifier = 'zeros';

  constructor(config: SimpleRNNCellLayerConfig) {
    super(config);
    this.units = config.units;
    this.activation = getActivation(
        config.activation == null ? this.DEFAULT_ACTIVATION :
                                    config.activation);
    this.useBias = config.useBias == null ? true : config.useBias;

    this.kernelInitializer = getInitializer(
        config.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
    this.recurrentInitializer = getInitializer(
        config.recurrentInitializer || this.DEFAULT_RECURRENT_INITIALIZER);

    this.biasInitializer =
        getInitializer(config.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);

    this.kernelRegularizer = getRegularizer(config.kernelRegularizer);
    this.recurrentRegularizer = getRegularizer(config.recurrentRegularizer);
    this.biasRegularizer = getRegularizer(config.biasRegularizer);

    this.kernelConstraint = getConstraint(config.kernelConstraint);
    this.recurrentConstraint = getConstraint(config.recurrentConstraint);
    this.biasConstraint = getConstraint(config.biasConstraint);

    this.dropout = math_utils.min(
        [1, math_utils.max([0, config.dropout == null ? 0 : config.dropout])]);
    this.recurrentDropout = math_utils.min([
      1,
      math_utils.max(
          [0, config.recurrentDropout == null ? 0 : config.recurrentDropout])
    ]);
    this.stateSize = this.units;
  }

  build(inputShape: Shape|Shape[]): void {
    inputShape = generic_utils.getExactlyOneShape(inputShape);
    // TODO(cais): Use regularizer.
    this.kernel = this.addWeight(
        'kernel', [inputShape[inputShape.length - 1], this.units], null,
        this.kernelInitializer, this.kernelRegularizer, true,
        this.kernelConstraint);
    this.recurrentKernel = this.addWeight(
        'recurrent_kernel', [this.units, this.units], null,
        this.recurrentInitializer, this.recurrentRegularizer, true,
        this.recurrentConstraint);
    if (this.useBias) {
      this.bias = this.addWeight(
          'bias', [this.units], null, this.biasInitializer,
          this.biasRegularizer, true, this.biasConstraint);
    } else {
      this.bias = null;
    }
    this.built = true;
  }

  // Porting Note: PyKeras' equivalent of this method takes two tensor inputs:
  //   `inputs` and `states`. Here, the two tensors are combined into an
  //   `Tensor[]` Array as the first input argument.
  //   Similarly, PyKeras' equivalent of this method returns two values:
  //    `output` and `[output]`. Here the two are combined into one length-2
  //    `Tensor[]`, consisting of `output` repeated.
  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    inputs = inputs as Tensor[];
    if (inputs.length !== 2) {
      throw new ValueError(
          `SimpleRNNCell expects 2 input Tensors, got ${inputs.length}.`);
    }
    const prevOutput = inputs[1];
    inputs = inputs[0];
    // TODO(cais): Uncomment the following when implementing the logic for
    //   dropout and training.
    // const training = kwargs['training'] == null ? false : kwargs['training'];
    if (this.dropout !== 0 || this.recurrentDropout !== 0) {
      throw new NotImplementedError(
          'Dropout is not implemented for SimpleRNNCell yet');
    }

    // TODO(cais): Handle dropout.
    let h = K.dot(inputs, this.kernel.read());
    if (this.bias != null) {
      h = K.biasAdd(h, this.bias.read());
    }
    let output = K.add(h, K.dot(prevOutput, this.recurrentKernel.read()));
    if (this.activation != null) {
      output = this.activation(output);
    }

    // TODO(cais): Properly set learning phase on output tensor?
    return [output, output];
  }

  getClassName(): string {
    return 'SimpleRNNCell';
  }

  getConfig(): ConfigDict {
    const config: ConfigDict = {
      units: this.units,
      activation: serializeActivation(this.activation),
      useBias: this.useBias,
      kernelInitializer: serializeInitializer(this.kernelInitializer),
      recurrentInitializer: serializeInitializer(this.recurrentInitializer),
      biasInitializer: serializeInitializer(this.biasInitializer),
      kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
      recurrentRegularizer: serializeRegularizer(this.recurrentRegularizer),
      biasRegularizer: serializeRegularizer(this.biasRegularizer),
      activityRegularizer: serializeRegularizer(this.activityRegularizer),
      kernelConstraint: serializeConstraint(this.kernelConstraint),
      recurrentConstraint: serializeConstraint(this.recurrentConstraint),
      biasConstraint: serializeConstraint(this.biasConstraint),
      dropout: this.dropout,
      recurrentDropout: this.recurrentDropout,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
generic_utils.ClassNameMap.register('SimpleRNNCell', SimpleRNNCell);

export interface SimpleRNNLayerConfig extends BaseRNNLayerConfig {
  /**
   * Positive integer, dimensionality of the output space.
   */
  units: number;

  /**
   * Activation function to use.
   *
   * Defaults to  hyperbolic tangent (`tanh`)
   *
   * If you pass `null`, no activation will be applied.
   */
  activation?: ActivationIdentifier;

  /**
   * Whether the layer uses a bias vector.
   */
  useBias?: boolean;

  /**
   * Initializer for the `kernel` weights matrix, used for the linear
   * transformation of the inputs.
   */
  kernelInitializer?: InitializerIdentifier|Initializer;

  /**
   * Initializer for the `recurrentKernel` weights matrix, used for
   * linear transformation of the recurrent state.
   */
  recurrentInitializer?: InitializerIdentifier|Initializer;

  /**
   * Initializer for the bias vector.
   */
  biasInitializer?: InitializerIdentifier|Initializer;

  /**
   * Regularizer function applied to the kernel weights matrix.
   */
  kernelRegularizer?: RegularizerIdentifier|Regularizer;

  /**
   * Regularizer function applied to the recurrentKernel weights matrix.
   */
  recurrentRegularizer?: RegularizerIdentifier|Regularizer;

  /**
   * Regularizer function applied to the bias vector.
   */
  biasRegularizer?: RegularizerIdentifier|Regularizer;

  /**
   * Constraint function applied to the kernel weights matrix.
   */
  kernelConstraint?: ConstraintIdentifier|Constraint;

  /**
   * Constraint function applied to the recurrentKernel weights matrix.
   */
  recurrentConstraint?: ConstraintIdentifier|Constraint;

  /**
   * Constraint function applied to the bias vector.
   */
  biasConstraint?: ConstraintIdentifier|Constraint;

  /**
   * Number between 0 and 1. Fraction of the units to drop for the linear
   * transformation of the inputs.
   */
  dropout?: number;

  /**
   * Number between 0 and 1. Fraction of the units to drop for the linear
   * transformation of the recurrent state.
   */
  recurrentDropout?: number;
}

/**
 * Fully-connected RNN where the output is to be fed back to input.
 *
 * This is an `RNN` layer consisting of one `SimpleRNNCell`. However, unlike
 * the underlying `SimpleRNNCell`, the `apply` method of `SimpleRNN` operates
 * on a sequence of inputs. The shape of the input (not including the first,
 * batch dimension) needs to be at least 2-D, with the first dimension being
 * time steps. For example:
 *
 * ```js
 * const rnn = tf.layers.simpleRNN({units: 8, returnSequences: true});
 *
 * // Create an input with 10 time steps.
 * const input = tf.input({shape: [10, 20]});
 * const output = rnn.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the `SimpleRNNCell`'s number of units.
 * ```
 */
export class SimpleRNN extends RNN {
  constructor(config: SimpleRNNLayerConfig) {
    config.cell = new SimpleRNNCell(config);
    super(config as RNNLayerConfig);
    // TODO(cais): Add activityRegularizer.
  }

  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    // TODO(cais): Add dropoutMask and recurrentDropoutMask.
    const mask = kwargs == null ? null : kwargs['mask'];
    const training = kwargs == null ? null : kwargs['training'];
    const initialState: Tensor[] =
        kwargs == null ? null : kwargs['initialState'];
    return super.call(inputs, {mask, training, initialState});
  }

  // TODO(cais): Research possibility of refactoring out the tedious all
  //   the getters that delegate to `this.cell` below.
  get units(): number {
    return (this.cell as SimpleRNNCell).units;
  }

  get activation(): ActivationFn {
    return (this.cell as SimpleRNNCell).activation;
  }

  get useBias(): boolean {
    return (this.cell as SimpleRNNCell).useBias;
  }

  get kernelInitializer(): Initializer {
    return (this.cell as SimpleRNNCell).kernelInitializer;
  }

  get recurrentInitializer(): Initializer {
    return (this.cell as SimpleRNNCell).recurrentInitializer;
  }

  get biasInitializer(): Initializer {
    return (this.cell as SimpleRNNCell).biasInitializer;
  }

  get kernelRegularizer(): Regularizer {
    return (this.cell as SimpleRNNCell).kernelRegularizer;
  }

  get recurrentRegularizer(): Regularizer {
    return (this.cell as SimpleRNNCell).recurrentRegularizer;
  }

  get biasRegularizer(): Regularizer {
    return (this.cell as SimpleRNNCell).biasRegularizer;
  }

  get kernelConstraint(): Constraint {
    return (this.cell as SimpleRNNCell).kernelConstraint;
  }

  get recurrentConstraint(): Constraint {
    return (this.cell as SimpleRNNCell).recurrentConstraint;
  }

  get biasConstraint(): Constraint {
    return (this.cell as SimpleRNNCell).biasConstraint;
  }

  get dropout(): number {
    return (this.cell as SimpleRNNCell).dropout;
  }

  get recurrentDropout(): number {
    return (this.cell as SimpleRNNCell).recurrentDropout;
  }

  getClassName(): string {
    return 'SimpleRNN';
  }

  getConfig(): ConfigDict {
    const config: ConfigDict = {
      units: this.units,
      activation: serializeActivation(this.activation),
      useBias: this.useBias,
      kernelInitializer: serializeInitializer(this.kernelInitializer),
      recurrentInitializer: serializeInitializer(this.recurrentInitializer),
      biasInitializer: serializeInitializer(this.biasInitializer),
      kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
      recurrentRegularizer: serializeRegularizer(this.recurrentRegularizer),
      biasRegularizer: serializeRegularizer(this.biasRegularizer),
      activityRegularizer: serializeRegularizer(this.activityRegularizer),
      kernelConstraint: serializeConstraint(this.kernelConstraint),
      recurrentConstraint: serializeConstraint(this.recurrentConstraint),
      biasConstraint: serializeConstraint(this.biasConstraint),
      dropout: this.dropout,
      recurrentDropout: this.recurrentDropout,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
generic_utils.ClassNameMap.register('SimpleRNN', SimpleRNN);

// Porting Note: Since this is a superset of SimpleRNNLayerConfig, we extend
//   that interface instead of repeating the fields.
export interface GRUCellLayerConfig extends SimpleRNNCellLayerConfig {
  /**
   * Activation function to use for the recurrent step.
   *
   * Defaults to hard sigmoid (`hardSigomid`).
   *
   * If `null`, no activation is applied.
   */
  recurrentActivation?: string;

  /**
   * Implementation mode, either 1 or 2.
   *
   * Mode 1 will structure its operations as a larger number of
   *   smaller dot products and additions.
   *
   * Mode 2 will batch them into fewer, larger operations. These modes will
   * have different performance profiles on different hardware and
   * for different applications.
   */
  implementation?: number;
}

/**
 * Cell class for `GRU`.
 *
 * `GRUCell` is distinct from the `RNN` subclass `GRU` in that its
 * `apply` method takes the input data of only a single time step and returns
 * the cell's output at the time step, while `GRU` takes the input data
 * over a number of time steps. For example:
 *
 * ```js
 * const cell = tf.layers.gruCell({units: 2});
 * const input = tf.input({shape: [10]});
 * const output = cell.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10]: This is the cell's output at a single time step. The 1st
 * // dimension is the unknown batch size.
 * ```
 *
 * Instance(s) of `GRUCell` can be used to construct `RNN` layers. The
 * most typical use of this workflow is to combine a number of cells into a
 * stacked RNN cell (i.e., `StackedRNNCell` internally) and use it to create an
 * RNN. For example:
 *
 * ```js
 * const cells = [
 *   tf.layers.gruCell({units: 4}),
 *   tf.layers.gruCell({units: 8}),
 * ];
 * const rnn = tf.layers.rnn({cell: cells, returnSequences: true});
 *
 * // Create an input with 10 time steps and a length-20 vector at each step.
 * const input = tf.input({shape: [10, 20]});
 * const output = rnn.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the last `gruCell`'s number of units.
 * ```
 *
 * To create an `RNN` consisting of only *one* `GRUCell`, use the
 * `tf.layers.gru`.
 */
export class GRUCell extends RNNCell {
  readonly units: number;
  readonly activation: ActivationFn;
  readonly recurrentActivation: ActivationFn;
  readonly useBias: boolean;

  readonly kernelInitializer: Initializer;
  readonly recurrentInitializer: Initializer;
  readonly biasInitializer: Initializer;

  readonly kernelRegularizer: Regularizer;
  readonly recurrentRegularizer: Regularizer;
  readonly biasRegularizer: Regularizer;

  readonly kernelConstraint: Constraint;
  readonly recurrentConstraint: Constraint;
  readonly biasConstraint: Constraint;

  readonly dropout: number;
  readonly recurrentDropout: number;

  readonly stateSize: number;
  readonly implementation: number;

  readonly DEFAULT_ACTIVATION = 'tanh';
  readonly DEFAULT_RECURRENT_ACTIVATION = 'hardSigmoid';

  readonly DEFAULT_KERNEL_INITIALIZER = 'glorotNormal';
  readonly DEFAULT_RECURRENT_INITIALIZER = 'orthogonal';
  readonly DEFAULT_BIAS_INITIALIZER: InitializerIdentifier = 'zeros';

  kernel: LayerVariable;
  recurrentKernel: LayerVariable;
  bias: LayerVariable;

  constructor(config: GRUCellLayerConfig) {
    super(config);

    this.units = config.units;
    this.activation = getActivation(
        config.activation === undefined ? this.DEFAULT_ACTIVATION :
                                          config.activation);
    this.recurrentActivation = getActivation(
        config.activation === undefined ? this.DEFAULT_RECURRENT_ACTIVATION :
                                          config.recurrentActivation);
    this.useBias = config.useBias == null ? true : config.useBias;

    this.kernelInitializer = getInitializer(
        config.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
    this.recurrentInitializer = getInitializer(
        config.recurrentInitializer || this.DEFAULT_RECURRENT_INITIALIZER);

    this.biasInitializer =
        getInitializer(config.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);

    this.kernelRegularizer = getRegularizer(config.kernelRegularizer);
    this.recurrentRegularizer = getRegularizer(config.recurrentRegularizer);
    this.biasRegularizer = getRegularizer(config.biasRegularizer);

    this.kernelConstraint = getConstraint(config.kernelConstraint);
    this.recurrentConstraint = getConstraint(config.recurrentConstraint);
    this.biasConstraint = getConstraint(config.biasConstraint);

    this.dropout = math_utils.min(
        [1, math_utils.max([0, config.dropout == null ? 0 : config.dropout])]);
    this.recurrentDropout = math_utils.min([
      1,
      math_utils.max(
          [0, config.recurrentDropout == null ? 0 : config.recurrentDropout])
    ]);
    this.implementation = config.implementation;
    this.stateSize = this.units;
  }

  public build(inputShape: Shape|Shape[]): void {
    inputShape = generic_utils.getExactlyOneShape(inputShape);
    const inputDim = inputShape[inputShape.length - 1];
    this.kernel = this.addWeight(
        'kernel', [inputDim, this.units * 3], null, this.kernelInitializer,
        this.kernelRegularizer, true, this.kernelConstraint);
    this.recurrentKernel = this.addWeight(
        'recurrent_kernel', [this.units, this.units * 3], null,
        this.recurrentInitializer, this.recurrentRegularizer, true,
        this.recurrentConstraint);
    if (this.useBias) {
      this.bias = this.addWeight(
          'bias', [this.units * 3], null, this.biasInitializer,
          this.biasRegularizer, true, this.biasConstraint);
    } else {
      this.bias = null;
    }
    // Porting Notes: Unlike the PyKeras implementation, we perform slicing
    //   of the weights and bias in the call() method, at execution time.
    this.built = true;
  }

  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    // TODO(cais): Implement dropout.
    if (this.dropout !== 0 || this.recurrentDropout !== 0) {
      throw new NotImplementedError(
          'Dropout is not implemented for GRUCell yet');
    }

    inputs = inputs as Tensor[];
    if (inputs.length !== 2) {
      throw new ValueError(
          `GRUCell expects 2 input Tensors (inputs, h, c), got ` +
          `${inputs.length}.`);
    }
    const hTMinus1 = inputs[1];  // Previous memory state.
    inputs = inputs[0];

    let z: Tensor;
    let r: Tensor;
    let hh: Tensor;
    if (this.implementation === 1) {
      const kernelZ = K.sliceAlongLastAxis(this.kernel.read(), 0, this.units);
      const kernelR =
          K.sliceAlongLastAxis(this.kernel.read(), this.units, this.units);
      const kernelH =
          K.sliceAlongLastAxis(this.kernel.read(), this.units * 2, this.units);
      const recurrentKernelZ =
          K.sliceAlongLastAxis(this.recurrentKernel.read(), 0, this.units);
      const recurrentKernelR = K.sliceAlongLastAxis(
          this.recurrentKernel.read(), this.units, this.units);
      const recurrentKernelH = K.sliceAlongLastAxis(
          this.recurrentKernel.read(), this.units * 2, this.units);

      // TODO(cais): Add input dropout.
      const inputsZ = inputs;
      const inputsR = inputs;
      const inputsH = inputs;

      let xZ = K.dot(inputsZ, kernelZ);
      let xR = K.dot(inputsR, kernelR);
      let xH = K.dot(inputsH, kernelH);
      if (this.useBias) {
        const biasZ = K.sliceAlongFirstAxis(this.bias.read(), 0, this.units);
        const biasR =
            K.sliceAlongFirstAxis(this.bias.read(), this.units, this.units);
        const biasH =
            K.sliceAlongFirstAxis(this.bias.read(), this.units * 2, this.units);
        xZ = K.biasAdd(xZ, biasZ);
        xR = K.biasAdd(xR, biasR);
        xH = K.biasAdd(xH, biasH);
      }

      // TODO(cais): Add recurrent dropout.
      const hTMinus1Z = hTMinus1;
      const hTMinus1R = hTMinus1;
      const hTMinus1H = hTMinus1;
      z = this.recurrentActivation(
          K.add(xZ, K.dot(hTMinus1Z, recurrentKernelZ)));
      r = this.recurrentActivation(
          K.add(xR, K.dot(hTMinus1R, recurrentKernelR)));
      hh = this.activation(
          K.add(xH, K.dot(K.multiply(r, hTMinus1H), recurrentKernelH)));
    } else {
      // TODO(cais): Add input dropout.
      let matrixX = K.dot(inputs, this.kernel.read());
      if (this.useBias) {
        matrixX = K.biasAdd(matrixX, this.bias.read());
      }
      // TODO(cais): Add recurrent dropout.
      const matrixInner = K.dot(
          hTMinus1,
          K.sliceAlongLastAxis(this.recurrentKernel.read(), 0, 2 * this.units));

      const xZ = K.sliceAlongLastAxis(matrixX, 0, this.units);
      const xR = K.sliceAlongLastAxis(matrixX, this.units, this.units);
      const recurrentZ = K.sliceAlongLastAxis(matrixInner, 0, this.units);
      const recurrentR =
          K.sliceAlongLastAxis(matrixInner, this.units, this.units);

      z = this.recurrentActivation(K.add(xZ, recurrentZ));
      r = this.recurrentActivation(K.add(xR, recurrentR));

      const xH = K.sliceAlongLastAxis(matrixX, 2 * this.units, this.units);
      const recurrentH = K.dot(
          K.multiply(r, hTMinus1),
          K.sliceAlongLastAxis(
              this.recurrentKernel.read(), 2 * this.units, this.units));
      hh = this.activation(K.add(xH, recurrentH));
    }

    const h = K.add(
        K.multiply(z, hTMinus1),
        K.multiply(K.scalarPlusArray(K.getScalar(1), K.neg(z)), hh));
    // TODO(cais): Add use_learning_phase flag properly.
    return [h, h];
  }

  getClassName(): string {
    return 'GRUCell';
  }

  getConfig(): ConfigDict {
    const config: ConfigDict = {
      units: this.units,
      activation: serializeActivation(this.activation),
      useBias: this.useBias,
      kernelInitializer: serializeInitializer(this.kernelInitializer),
      recurrentInitializer: serializeInitializer(this.recurrentInitializer),
      biasInitializer: serializeInitializer(this.biasInitializer),
      kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
      recurrentRegularizer: serializeRegularizer(this.recurrentRegularizer),
      biasRegularizer: serializeRegularizer(this.biasRegularizer),
      activityRegularizer: serializeRegularizer(this.activityRegularizer),
      kernelConstraint: serializeConstraint(this.kernelConstraint),
      recurrentConstraint: serializeConstraint(this.recurrentConstraint),
      biasConstraint: serializeConstraint(this.biasConstraint),
      dropout: this.dropout,
      recurrentDropout: this.recurrentDropout,
      implementation: this.implementation,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
generic_utils.ClassNameMap.register('GRUCell', GRUCell);

// Porting Note: Since this is a superset of SimpleRNNLayerConfig, we inherit
//   from that interface instead of repeating the fields here.
export interface GRULayerConfig extends SimpleRNNLayerConfig {
  /**
   * Implementation mode, either 1 or 2.
   *
   * Mode 1 will structure its operations as a larger number of
   * smaller dot products and additions.
   *
   * Mode 2 will batch them into fewer, larger operations. These modes will
   * have different performance profiles on different hardware and
   * for different applications.
   */
  implementation?: number;
}

/**
 * Gated Recurrent Unit - Cho et al. 2014.
 *
 * This is an `RNN` layer consisting of one `GRUCell`. However, unlike
 * the underlying `GRUCell`, the `apply` method of `SimpleRNN` operates
 * on a sequence of inputs. The shape of the input (not including the first,
 * batch dimension) needs to be at least 2-D, with the first dimension being
 * time steps. For example:
 *
 * ```js
 * const rnn = tf.layers.gru({units: 8, returnSequences: true});
 *
 * // Create an input with 10 time steps.
 * const input = tf.input({shape: [10, 20]});
 * const output = rnn.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the `GRUCell`'s number of units.
 */
export class GRU extends RNN {
  constructor(config: GRULayerConfig) {
    if (config.implementation === 0) {
      console.warn(
          '`implementation=0` has been deprecated, and now defaults to ' +
          '`implementation=1`. Please update your layer call.');
    }
    config.cell = new GRUCell(config);
    super(config as RNNLayerConfig);
    // TODO(cais): Add activityRegularizer.
  }

  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    // TODO(cais): Add dropoutMask and recurrentDropoutMask.
    const mask = kwargs == null ? null : kwargs['mask'];
    const training = kwargs == null ? null : kwargs['training'];
    const initialState: Tensor[] =
        kwargs == null ? null : kwargs['initialState'];
    return super.call(inputs, {mask, training, initialState});
  }

  get units(): number {
    return (this.cell as GRUCell).units;
  }

  get activation(): ActivationFn {
    return (this.cell as GRUCell).activation;
  }

  get useBias(): boolean {
    return (this.cell as GRUCell).useBias;
  }

  get kernelInitializer(): Initializer {
    return (this.cell as GRUCell).kernelInitializer;
  }

  get recurrentInitializer(): Initializer {
    return (this.cell as GRUCell).recurrentInitializer;
  }

  get biasInitializer(): Initializer {
    return (this.cell as GRUCell).biasInitializer;
  }

  get kernelRegularizer(): Regularizer {
    return (this.cell as GRUCell).kernelRegularizer;
  }

  get recurrentRegularizer(): Regularizer {
    return (this.cell as GRUCell).recurrentRegularizer;
  }

  get biasRegularizer(): Regularizer {
    return (this.cell as GRUCell).biasRegularizer;
  }

  get kernelConstraint(): Constraint {
    return (this.cell as GRUCell).kernelConstraint;
  }

  get recurrentConstraint(): Constraint {
    return (this.cell as GRUCell).recurrentConstraint;
  }

  get biasConstraint(): Constraint {
    return (this.cell as GRUCell).biasConstraint;
  }

  get dropout(): number {
    return (this.cell as GRUCell).dropout;
  }

  get recurrentDropout(): number {
    return (this.cell as GRUCell).recurrentDropout;
  }

  get implementation(): number {
    return (this.cell as GRUCell).implementation;
  }

  getClassName(): string {
    return 'GRU';
  }

  getConfig(): ConfigDict {
    const config: ConfigDict = {
      units: this.units,
      activation: serializeActivation(this.activation),
      useBias: this.useBias,
      kernelInitializer: serializeInitializer(this.kernelInitializer),
      recurrentInitializer: serializeInitializer(this.recurrentInitializer),
      biasInitializer: serializeInitializer(this.biasInitializer),
      kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
      recurrentRegularizer: serializeRegularizer(this.recurrentRegularizer),
      biasRegularizer: serializeRegularizer(this.biasRegularizer),
      activityRegularizer: serializeRegularizer(this.activityRegularizer),
      kernelConstraint: serializeConstraint(this.kernelConstraint),
      recurrentConstraint: serializeConstraint(this.recurrentConstraint),
      biasConstraint: serializeConstraint(this.biasConstraint),
      dropout: this.dropout,
      recurrentDropout: this.recurrentDropout,
      implementation: this.implementation,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  static fromConfig<T>(cls: generic_utils.Constructor<T>, config: ConfigDict):
      T {
    if (config['implmentation'] === 0) {
      config['implementation'] = 1;
    }
    return new cls(config);
  }
}
generic_utils.ClassNameMap.register('GRU', GRU);

// Porting Note: Since this is a superset of SimpleRNNLayerConfig, we extend
//   that interface instead of repeating the fields.
export interface LSTMCellLayerConfig extends SimpleRNNCellLayerConfig {
  /**
   * Activation function to use for the recurrent step.
   *
   * Defaults to hard sigmoid (`hardSigomid`).
   *
   * If `null`, no activation is applied.
   */
  recurrentActivation?: ActivationIdentifier;

  /**
   * If `true`, add 1 to the bias of the forget gate at initialization.
   * Setting it to `true` will also force `biasInitializer = 'zeros'`.
   * This is recommended in
   * [Jozefowicz et
   * al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
   */
  unitForgetBias?: boolean;

  /**
   * Implementation mode, either 1 or 2.
   *
   * Mode 1 will structure its operations as a larger number of
   *   smaller dot products and additions.
   *
   * Mode 2 will batch them into fewer, larger operations. These modes will
   * have different performance profiles on different hardware and
   * for different applications.
   */
  implementation?: 1|2;
}

/**
 * Cell class for `LSTM`.
 *
 * `LSTMCell` is distinct from the `RNN` subclass `LSTM` in that its
 * `apply` method takes the input data of only a single time step and returns
 * the cell's output at the time step, while `LSTM` takes the input data
 * over a number of time steps. For example:
 *
 * ```js
 * const cell = tf.layers.lstmCell({units: 2});
 * const input = tf.input({shape: [10]});
 * const output = cell.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10]: This is the cell's output at a single time step. The 1st
 * // dimension is the unknown batch size.
 * ```
 *
 * Instance(s) of `LSTMCell` can be used to construct `RNN` layers. The
 * most typical use of this workflow is to combine a number of cells into a
 * stacked RNN cell (i.e., `StackedRNNCell` internally) and use it to create an
 * RNN. For example:
 *
 * ```js
 * const cells = [
 *   tf.layers.lstmCell({units: 4}),
 *   tf.layers.lstmCell({units: 8}),
 * ];
 * const rnn = tf.layers.rnn({cell: cells, returnSequences: true});
 *
 * // Create an input with 10 time steps and a length-20 vector at each step.
 * const input = tf.input({shape: [10, 20]});
 * const output = rnn.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the last `lstmCell`'s number of units.
 * ```
 *
 * To create an `RNN` consisting of only *one* `LSTMCell`, use the
 * `tf.layers.lstm`.
 */
export class LSTMCell extends RNNCell {
  readonly units: number;
  readonly activation: ActivationFn;
  readonly recurrentActivation: ActivationFn;
  readonly useBias: boolean;

  readonly kernelInitializer: Initializer;
  readonly recurrentInitializer: Initializer;
  readonly biasInitializer: Initializer;
  readonly unitForgetBias: boolean;

  readonly kernelConstraint: Constraint;
  readonly recurrentConstraint: Constraint;
  readonly biasConstraint: Constraint;

  readonly kernelRegularizer: Regularizer;
  readonly recurrentRegularizer: Regularizer;
  readonly biasRegularizer: Regularizer;

  readonly dropout: number;
  readonly recurrentDropout: number;

  readonly stateSize: number[];
  readonly implementation: number;

  readonly DEFAULT_ACTIVATION = 'tanh';
  readonly DEFAULT_RECURRENT_ACTIVATION = 'hardSigmoid';
  readonly DEFAULT_KERNEL_INITIALIZER = 'glorotNormal';
  readonly DEFAULT_RECURRENT_INITIALIZER = 'orthogonal';

  readonly DEFAULT_BIAS_INITIALIZER = 'zeros';

  kernel: LayerVariable;
  recurrentKernel: LayerVariable;
  bias: LayerVariable;

  constructor(config: LSTMCellLayerConfig) {
    super(config);

    this.units = config.units;
    this.activation = getActivation(
        config.activation === undefined ? this.DEFAULT_ACTIVATION :
                                          config.activation);
    this.recurrentActivation = getActivation(
        config.activation === undefined ? this.DEFAULT_RECURRENT_ACTIVATION :
                                          config.recurrentActivation);
    this.useBias = config.useBias == null ? true : config.useBias;

    this.kernelInitializer = getInitializer(
        config.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
    this.recurrentInitializer = getInitializer(
        config.recurrentInitializer || this.DEFAULT_RECURRENT_INITIALIZER);

    this.biasInitializer =
        getInitializer(config.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);
    this.unitForgetBias = config.unitForgetBias;

    this.kernelRegularizer = getRegularizer(config.kernelRegularizer);
    this.recurrentRegularizer = getRegularizer(config.recurrentRegularizer);
    this.biasRegularizer = getRegularizer(config.biasRegularizer);

    this.kernelConstraint = getConstraint(config.kernelConstraint);
    this.recurrentConstraint = getConstraint(config.recurrentConstraint);
    this.biasConstraint = getConstraint(config.biasConstraint);

    this.dropout = math_utils.min(
        [1, math_utils.max([0, config.dropout == null ? 0 : config.dropout])]);
    this.recurrentDropout = math_utils.min([
      1,
      math_utils.max(
          [0, config.recurrentDropout == null ? 0 : config.recurrentDropout])
    ]);
    this.implementation = config.implementation;
    this.stateSize = [this.units, this.units];
  }

  public build(inputShape: Shape|Shape[]): void {
    inputShape = generic_utils.getExactlyOneShape(inputShape);
    const inputDim = inputShape[inputShape.length - 1];
    this.kernel = this.addWeight(
        'kernel', [inputDim, this.units * 4], null, this.kernelInitializer,
        this.kernelRegularizer, true, this.kernelConstraint);
    this.recurrentKernel = this.addWeight(
        'recurrent_kernel', [this.units, this.units * 4], null,
        this.recurrentInitializer, this.recurrentRegularizer, true,
        this.recurrentConstraint);
    let biasInitializer: Initializer;
    if (this.useBias) {
      if (this.unitForgetBias) {
        const capturedBiasInit = this.biasInitializer;
        const capturedUnits = this.units;
        biasInitializer = new (class CustomInit extends Initializer {
          apply(shape: Shape, dtype?: DType): Tensor {
            // TODO(cais): More informative variable names?
            const bI = capturedBiasInit.apply([capturedUnits]);
            const bF = (new Ones()).apply([capturedUnits]);
            const bCAndH = capturedBiasInit.apply([capturedUnits * 2]);
            return K.concatAlongFirstAxis(
                K.concatAlongFirstAxis(bI, bF), bCAndH);
          }
          getClassName(): string {
            return 'CustomInit';
          }
        })();
      } else {
        biasInitializer = this.biasInitializer;
      }
      this.bias = this.addWeight(
          'bias', [this.units * 4], null, biasInitializer, this.biasRegularizer,
          true, this.biasConstraint);
    } else {
      this.bias = null;
    }
    // Porting Notes: Unlike the PyKeras implementation, we perform slicing
    //   of the weights and bias in the call() method, at execution time.
    this.built = true;
  }

  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    // TODO(cais): Implement dropout.
    if (this.dropout !== 0 || this.recurrentDropout !== 0) {
      throw new NotImplementedError(
          'Dropout is not implemented for LSTMCell yet');
    }

    inputs = inputs as Tensor[];
    if (inputs.length !== 3) {
      throw new ValueError(
          `LSTMCell expects 3 input Tensors (inputs, h, c), got ` +
          `${inputs.length}.`);
    }
    const hTMinus1 = inputs[1];  // Previous memory state.
    const cTMinus1 = inputs[2];  // Previous carry state.
    inputs = inputs[0];

    let i: Tensor;
    let f: Tensor;
    let c: Tensor;
    let o: Tensor;
    if (this.implementation === 1) {
      const kernelI = K.sliceAlongLastAxis(this.kernel.read(), 0, this.units);
      const kernelF =
          K.sliceAlongLastAxis(this.kernel.read(), this.units, this.units);
      const kernelC =
          K.sliceAlongLastAxis(this.kernel.read(), this.units * 2, this.units);
      const kernelO =
          K.sliceAlongLastAxis(this.kernel.read(), this.units * 3, this.units);
      const recurrentKernelI =
          K.sliceAlongLastAxis(this.recurrentKernel.read(), 0, this.units);
      const recurrentKernelF = K.sliceAlongLastAxis(
          this.recurrentKernel.read(), this.units, this.units);
      const recurrentKernelC = K.sliceAlongLastAxis(
          this.recurrentKernel.read(), this.units * 2, this.units);
      const recurrentKernelO = K.sliceAlongLastAxis(
          this.recurrentKernel.read(), this.units * 3, this.units);

      // TODO(cais): Add input dropout.
      const inputsI = inputs;
      const inputsF = inputs;
      const inputsC = inputs;
      const inputsO = inputs;

      let xI = K.dot(inputsI, kernelI);
      let xF = K.dot(inputsF, kernelF);
      let xC = K.dot(inputsC, kernelC);
      let xO = K.dot(inputsO, kernelO);
      if (this.useBias) {
        const biasI = K.sliceAlongFirstAxis(this.bias.read(), 0, this.units);
        const biasF =
            K.sliceAlongFirstAxis(this.bias.read(), this.units, this.units);
        const biasC =
            K.sliceAlongFirstAxis(this.bias.read(), this.units * 2, this.units);
        const biasO =
            K.sliceAlongFirstAxis(this.bias.read(), this.units * 3, this.units);
        xI = K.biasAdd(xI, biasI);
        xF = K.biasAdd(xF, biasF);
        xC = K.biasAdd(xC, biasC);
        xO = K.biasAdd(xO, biasO);
      }

      // TODO(cais): Add recurrent dropout.
      const hTMinus1I = hTMinus1;
      const hTMinus1F = hTMinus1;
      const hTMinus1C = hTMinus1;
      const hTMinus1O = hTMinus1;
      i = this.recurrentActivation(
          K.add(xI, K.dot(hTMinus1I, recurrentKernelI)));
      f = this.recurrentActivation(
          K.add(xF, K.dot(hTMinus1F, recurrentKernelF)));
      c = K.add(
          K.multiply(f, cTMinus1),
          K.multiply(
              i,
              this.activation(K.add(xC, K.dot(hTMinus1C, recurrentKernelC)))));
      o = this.recurrentActivation(
          K.add(xO, K.dot(hTMinus1O, recurrentKernelO)));
    } else {
      // TODO(cais): Add input dropout.
      let z = K.dot(inputs, this.kernel.read());
      // TODO(cais): Add recurrent dropout.
      z = K.add(z, K.dot(hTMinus1, this.recurrentKernel.read()));
      if (this.useBias) {
        z = K.biasAdd(z, this.bias.read());
      }

      const z0 = K.sliceAlongLastAxis(z, 0, this.units);
      const z1 = K.sliceAlongLastAxis(z, this.units, this.units);
      const z2 = K.sliceAlongLastAxis(z, this.units * 2, this.units);
      const z3 = K.sliceAlongLastAxis(z, this.units * 3, this.units);

      i = this.recurrentActivation(z0);
      f = this.recurrentActivation(z1);
      c = K.add(K.multiply(f, cTMinus1), K.multiply(i, this.activation(z2)));
      o = this.recurrentActivation(z3);
    }

    const h = K.multiply(o, this.activation(c));
    // TODO(cais): Add use_learning_phase flag properly.
    return [h, h, c];
  }

  getClassName(): string {
    return 'LSTMCell';
  }

  getConfig(): ConfigDict {
    const config: ConfigDict = {
      units: this.units,
      activation: serializeActivation(this.activation),
      useBias: this.useBias,
      kernelInitializer: serializeInitializer(this.kernelInitializer),
      recurrentInitializer: serializeInitializer(this.recurrentInitializer),
      biasInitializer: serializeInitializer(this.biasInitializer),
      unitForgetBias: this.unitForgetBias,
      kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
      recurrentRegularizer: serializeRegularizer(this.recurrentRegularizer),
      biasRegularizer: serializeRegularizer(this.biasRegularizer),
      activityRegularizer: serializeRegularizer(this.activityRegularizer),
      kernelConstraint: serializeConstraint(this.kernelConstraint),
      recurrentConstraint: serializeConstraint(this.recurrentConstraint),
      biasConstraint: serializeConstraint(this.biasConstraint),
      dropout: this.dropout,
      recurrentDropout: this.recurrentDropout,
      implementation: this.implementation,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
generic_utils.ClassNameMap.register('LSTMCell', LSTMCell);

// Porting Note: Since this is a superset of SimpleRNNLayerConfig, we inherit
//   from that interface instead of repeating the fields here.
export interface LSTMLayerConfig extends SimpleRNNLayerConfig {
  /**
   * If `true`, add 1 to the bias of the forget gate at initialization.
   * Setting it to `true` will also force `biasInitializer = 'zeros'`.
   * This is recommended in
   * [Jozefowicz et
   * al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
   */
  unitForgetBias?: boolean;

  /**
   * Implementation mode, either 1 or 2.
   *   Mode 1 will structure its operations as a larger number of
   *   smaller dot products and additions, whereas mode 2 will
   *   batch them into fewer, larger operations. These modes will
   *   have different performance profiles on different hardware and
   *   for different applications.
   */
  implementation?: 1|2;
}

/**
 * Long-Short Term Memory layer - Hochreiter 1997.
 *
 * This is an `RNN` layer consisting of one `LSTMCell`. However, unlike
 * the underlying `LSTMCell`, the `apply` method of `LSTM` operates
 * on a sequence of inputs. The shape of the input (not including the first,
 * batch dimension) needs to be at least 2-D, with the first dimension being
 * time steps. For example:
 *
 * ```js
 * const lstm = tf.layers.lstm({units: 8, returnSequences: true});
 *
 * // Create an input with 10 time steps.
 * const input = tf.input({shape: [10, 20]});
 * const output = lstm.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the `LSTMCell`'s number of units.
 */
export class LSTM extends RNN {
  constructor(config: LSTMLayerConfig) {
    if (config.implementation as number === 0) {
      console.warn(
          '`implementation=0` has been deprecated, and now defaults to ' +
          '`implementation=1`. Please update your layer call.');
    }
    config.cell = new LSTMCell(config);
    super(config as RNNLayerConfig);
    // TODO(cais): Add activityRegularizer.
  }

  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    // TODO(cais): Add dropoutMask and recurrentDropoutMask.
    const mask = kwargs == null ? null : kwargs['mask'];
    const training = kwargs == null ? null : kwargs['training'];
    const initialState: Tensor[] =
        kwargs == null ? null : kwargs['initialState'];
    return super.call(inputs, {mask, training, initialState});
  }

  get units(): number {
    return (this.cell as LSTMCell).units;
  }

  get activation(): ActivationFn {
    return (this.cell as LSTMCell).activation;
  }

  get useBias(): boolean {
    return (this.cell as LSTMCell).useBias;
  }

  get kernelInitializer(): Initializer {
    return (this.cell as LSTMCell).kernelInitializer;
  }

  get recurrentInitializer(): Initializer {
    return (this.cell as LSTMCell).recurrentInitializer;
  }

  get biasInitializer(): Initializer {
    return (this.cell as LSTMCell).biasInitializer;
  }

  get unitForgetBias(): boolean {
    return (this.cell as LSTMCell).unitForgetBias;
  }

  get kernelRegularizer(): Regularizer {
    return (this.cell as LSTMCell).kernelRegularizer;
  }

  get recurrentRegularizer(): Regularizer {
    return (this.cell as LSTMCell).recurrentRegularizer;
  }

  get biasRegularizer(): Regularizer {
    return (this.cell as LSTMCell).biasRegularizer;
  }

  get kernelConstraint(): Constraint {
    return (this.cell as LSTMCell).kernelConstraint;
  }

  get recurrentConstraint(): Constraint {
    return (this.cell as LSTMCell).recurrentConstraint;
  }

  get biasConstraint(): Constraint {
    return (this.cell as LSTMCell).biasConstraint;
  }

  get dropout(): number {
    return (this.cell as LSTMCell).dropout;
  }

  get recurrentDropout(): number {
    return (this.cell as LSTMCell).recurrentDropout;
  }

  get implementation(): number {
    return (this.cell as LSTMCell).implementation;
  }

  getClassName(): string {
    return 'LSTM';
  }

  getConfig(): ConfigDict {
    const config: ConfigDict = {
      units: this.units,
      activation: serializeActivation(this.activation),
      useBias: this.useBias,
      kernelInitializer: serializeInitializer(this.kernelInitializer),
      recurrentInitializer: serializeInitializer(this.recurrentInitializer),
      biasInitializer: serializeInitializer(this.biasInitializer),
      unitForgetBias: this.unitForgetBias,
      kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
      recurrentRegularizer: serializeRegularizer(this.recurrentRegularizer),
      biasRegularizer: serializeRegularizer(this.biasRegularizer),
      activityRegularizer: serializeRegularizer(this.activityRegularizer),
      kernelConstraint: serializeConstraint(this.kernelConstraint),
      recurrentConstraint: serializeConstraint(this.recurrentConstraint),
      biasConstraint: serializeConstraint(this.biasConstraint),
      dropout: this.dropout,
      recurrentDropout: this.recurrentDropout,
      implementation: this.implementation,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  static fromConfig<T>(cls: generic_utils.Constructor<T>, config: ConfigDict):
      T {
    if (config['implmentation'] === 0) {
      config['implementation'] = 1;
    }
    return new cls(config);
  }
}
generic_utils.ClassNameMap.register('LSTM', LSTM);

export interface StackedRNNCellsConfig extends LayerConfig {
  /**
   * A `Array` of `RNNCell` instances.
   */
  cells: RNNCell[];
}

/**
 * Wrapper allowing a stack of RNN cells to behave as a single cell.
 *
 * Used to implement efficient stacked RNNs.
 */
export class StackedRNNCells extends RNNCell {
  protected cells: RNNCell[];

  constructor(config: StackedRNNCellsConfig) {
    super(config);
    this.cells = config.cells;
  }

  get stateSize(): number[] {
    // States are a flat list in reverse order of the cell stack.
    // This allows perserving the requirement `stack.statesize[0] ===
    // outputDim`. E.g., states of a 2-layer LSTM would be `[h2, c2, h1, c1]`,
    // assuming one LSTM has states `[h, c]`.
    const stateSize: number[] = [];
    for (const cell of this.cells.slice().reverse()) {
      if (Array.isArray(cell.stateSize)) {
        stateSize.push(...cell.stateSize);
      } else {
        stateSize.push(cell.stateSize);
      }
    }
    return stateSize;
  }

  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    inputs = inputs as Tensor[];
    let states = inputs.slice(1);

    // Recover per-cell states.
    const nestedStates: Tensor[][] = [];
    for (const cell of this.cells.slice().reverse()) {
      if (Array.isArray(cell.stateSize)) {
        nestedStates.push(states.splice(0, cell.stateSize.length));
      } else {
        nestedStates.push(states.splice(0, 1));
      }
    }
    nestedStates.reverse();

    // Call the cells in order and store the returned states.
    const newNestedStates: Tensor[][] = [];
    let callInputs: Tensor[];
    for (let i = 0; i < this.cells.length; ++i) {
      const cell = this.cells[i];
      states = nestedStates[i];
      // TODO(cais): Take care of constants.
      if (i === 0) {
        callInputs = [inputs[0]].concat(states);
      } else {
        callInputs = [callInputs[0]].concat(states);
      }
      callInputs = cell.call(callInputs, kwargs) as Tensor[];
      newNestedStates.push(callInputs.slice(1));
    }

    // Format the new states as a flat list in reverse cell order.
    states = [];
    for (const cellStates of newNestedStates.slice().reverse()) {
      states.push(...cellStates);
    }
    return [callInputs[0]].concat(states);
  }

  public build(inputShape: Shape|Shape[]): void {
    if (generic_utils.isArrayOfShapes(inputShape)) {
      // TODO(cais): Take care of input constants.
      // const constantShape = inputShape.slice(1);
      inputShape = (inputShape as Shape[])[0];
    }
    inputShape = inputShape as Shape;
    let outputDim: number;
    for (const cell of this.cells) {
      // TODO(cais): Take care of input constants.
      cell.build(inputShape);
      if (Array.isArray(cell.stateSize)) {
        outputDim = cell.stateSize[0];
      } else {
        outputDim = cell.stateSize;
      }
      inputShape = [inputShape[0], outputDim];
    }
    this.built = true;
  }

  getClassName(): string {
    return 'StackedRNNCells';
  }

  getConfig(): ConfigDict {
    const cellConfigs: ConfigDict[] = [];
    for (const cell of this.cells) {
      cellConfigs.push({
        'className': this.getClassName(),
        'config': cell.getConfig(),
      });
    }
    const config: ConfigDict = {'cells': cellConfigs};
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  static fromConfig<T>(
      cls: generic_utils.Constructor<T>, config: ConfigDict,
      customObjects = {} as ConfigDict): T {
    const cells: RNNCell[] = [];
    for (const cellConfig of (config['cells'] as ConfigDict[])) {
      cells.push(deserialize(cellConfig, customObjects) as RNNCell);
    }
    return new cls({cells});
  }

  get trainableWeights(): LayerVariable[] {
    if (!this.trainable) {
      return [];
    }
    const weights: LayerVariable[] = [];
    for (const cell of this.cells) {
      weights.push(...cell.trainableWeights);
    }
    return weights;
  }

  get nonTrainableWeights(): LayerVariable[] {
    const weights: LayerVariable[] = [];
    for (const cell of this.cells) {
      weights.push(...cell.nonTrainableWeights);
    }
    if (!this.trainable) {
      const trainableWeights: LayerVariable[] = [];
      for (const cell of this.cells) {
        trainableWeights.push(...cell.trainableWeights);
      }
      return trainableWeights.concat(weights);
    }
    return weights;
  }

  /**
   * Retrieve the weights of a the model.
   *
   * @returns A flat `Array` of `Tensor`s.
   */
  getWeights(): Tensor[] {
    const weights: LayerVariable[] = [];
    for (const cell of this.cells) {
      weights.push(...cell.weights);
    }
    return K.batchGetValue(weights);
  }

  /**
   * Set the weights of the model.
   *
   * @param weights An `Array` of `Tensor`s with shapes and types matching the
   *   output of `getWeights()`.
   */
  setWeights(weights: Tensor[]): void {
    const tuples: Array<[LayerVariable, Tensor]> = [];
    for (const cell of this.cells) {
      const numParams = cell.weights.length;
      const inputWeights = weights.splice(numParams);
      for (let i = 0; i < cell.weights.length; ++i) {
        tuples.push([cell.weights[i], inputWeights[i]]);
      }
    }
    K.batchSetValue(tuples);
  }

  // TODO(cais): Maybe implemnt `losses` and `getLossesFor`.
}
generic_utils.ClassNameMap.register('StackedRNNCells', StackedRNNCells);
