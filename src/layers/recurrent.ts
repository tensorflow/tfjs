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

import * as tfc from '@tensorflow/tfjs-core';
import {DataType, serialization, Tensor, tidy, util} from '@tensorflow/tfjs-core';

import {Activation, getActivation, serializeActivation} from '../activations';
import * as K from '../backend/tfjs_backend';
import {Constraint, ConstraintIdentifier, getConstraint, serializeConstraint} from '../constraints';
import {InputSpec, SymbolicTensor} from '../engine/topology';
import {Layer, LayerArgs} from '../engine/topology';
import {AttributeError, NotImplementedError, ValueError} from '../errors';
import {getInitializer, Initializer, InitializerIdentifier, Ones, serializeInitializer} from '../initializers';
import {ActivationIdentifier} from '../keras_format/activation_config';
import {Shape} from '../keras_format/common';
import {getRegularizer, Regularizer, RegularizerIdentifier, serializeRegularizer} from '../regularizers';
import {Kwargs, RnnStepFunction} from '../types';
import {assertPositiveInteger} from '../utils/generic_utils';
import * as math_utils from '../utils/math_utils';
import {getExactlyOneShape, getExactlyOneTensor, isArrayOfShapes} from '../utils/types_utils';
import {batchGetValue, batchSetValue, LayerVariable} from '../variables';

import {deserialize} from './serialization';

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
export function standardizeArgs(
    inputs: Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[],
    initialState: Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[],
    constants: Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[],
    numConstants?: number): {
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
    if (numConstants != null) {
      constants = inputs.slice(inputs.length - numConstants, inputs.length);
      inputs = inputs.slice(0, inputs.length - numConstants);
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

/**
 * Iterates over the time dimension of a tensor.
 *
 * @param stepFunction RNN step function.
 *   Parameters:
 *     inputs: tensor with shape `[samples, ...]` (no time dimension),
 *       representing input for the batch of samples at a certain time step.
 *     states: an Array of tensors.
 *   Returns:
 *     outputs: tensor with shape `[samples, outputDim]` (no time dimension).
 *     newStates: list of tensors, same length and shapes as `states`. The first
 *       state in the list must be the output tensor at the previous timestep.
 * @param inputs Tensor of temporal data of shape `[samples, time, ...]` (at
 *   least 3D).
 * @param initialStates Tensor with shape `[samples, outputDim]` (no time
 *   dimension), containing the initial values of the states used in the step
 *   function.
 * @param goBackwards If `true`, do the iteration over the time dimension in
 *   reverse order and return the reversed sequence.
 * @param mask Binary tensor with shape `[sample, time, 1]`, with a zero for
 *   every element that is masked.
 * @param constants An Array of constant values passed at each step.
 * @param unroll Whether to unroll the RNN or to use a symbolic loop. *Not*
 *   applicable to this imperative deeplearn.js backend. Its value is ignored.
 * @param needPerStepOutputs Whether the per-step outputs are to be
 *   concatenated into a single tensor and returned (as the second return
 *   value). Default: `false`. This arg is included so that the relatively
 *   expensive concatenation of the stepwise outputs can be omitted unless
 *   the stepwise outputs need to be kept (e.g., for an LSTM layer of which
 *   `returnSequence` is `true`.)
 * @returns An Array: `[lastOutput, outputs, newStates]`.
 *   lastOutput: the lastest output of the RNN, of shape `[samples, ...]`.
 *   outputs: tensor with shape `[samples, time, ...]` where each entry
 *     `output[s, t]` is the output of the step function at time `t` for sample
 *     `s`. This return value is provided if and only if the
 *     `needPerStepOutputs` is set as `true`. If it is set as `false`, this
 *     return value will be `undefined`.
 *   newStates: Array of tensors, latest states returned by the step function,
 *      of shape `(samples, ...)`.
 * @throws ValueError If input dimension is less than 3.
 *
 * TODO(nielsene): This needs to be tidy-ed.
 */
export function rnn(
    stepFunction: RnnStepFunction, inputs: Tensor, initialStates: Tensor[],
    goBackwards = false, mask?: Tensor, constants?: Tensor[], unroll = false,
    needPerStepOutputs = false): [Tensor, Tensor, Tensor[]] {
  return tfc.tidy(() => {
    const ndim = inputs.shape.length;
    if (ndim < 3) {
      throw new ValueError(`Input should be at least 3D, but is ${ndim}D.`);
    }

    // Transpose to time-major, i.e., from [batch, time, ...] to [time, batch,
    // ...].
    const axes = [1, 0].concat(math_utils.range(2, ndim));
    inputs = tfc.transpose(inputs, axes);

    if (constants != null) {
      throw new NotImplementedError(
          'The rnn() functoin of the deeplearn.js backend does not support ' +
          'constants yet.');
    }

    // Porting Note: the unroll option is ignored by the imperative backend.
    if (unroll) {
      console.warn(
          'Backend rnn(): the unroll = true option is not applicable to the ' +
          'imperative deeplearn.js backend.');
    }

    if (mask != null) {
      mask = mask.asType('bool').asType('float32');
      if (mask.rank === ndim - 1) {
        mask = tfc.expandDims(mask, -1);
      }
      mask = tfc.transpose(mask, axes);
    }

    if (goBackwards) {
      inputs = tfc.reverse(inputs, 0);
      if (mask != null) {
        mask = tfc.reverse(mask, 0);
      }
    }

    // Porting Note: PyKeras with TensorFlow backend uses a symbolic loop
    //   (tf.while_loop). But for the imperative deeplearn.js backend, we just
    //   use the usual TypeScript control flow to iterate over the time steps in
    //   the inputs.
    // Porting Note: PyKeras patches a "_use_learning_phase" attribute to
    // outputs.
    //   This is not idiomatic in TypeScript. The info regarding whether we are
    //   in a learning (i.e., training) phase for RNN is passed in a different
    //   way.

    const perStepOutputs: Tensor[] = [];
    let lastOutput: Tensor;
    let states = initialStates;
    const timeSteps = inputs.shape[0];
    const perStepInputs = tfc.unstack(inputs);
    let perStepMasks: Tensor[];
    if (mask != null) {
      perStepMasks = tfc.unstack(mask);
    }

    for (let t = 0; t < timeSteps; ++t) {
      const currentInput = perStepInputs[t];
      const stepOutputs = tfc.tidy(() => stepFunction(currentInput, states));

      if (mask == null) {
        lastOutput = stepOutputs[0];
        states = stepOutputs[1];
      } else {
        const maskedOutputs = tfc.tidy(() => {
          const stepMask = perStepMasks[t];
          const negStepMask = tfc.onesLike(stepMask).sub(stepMask);
          // TODO(cais): Would tfc.where() be better for performance?
          const output = stepOutputs[0].mul(stepMask).addStrict(
              states[0].mul(negStepMask));
          const newStates = states.map((state, i) => {
            return stepOutputs[1][i].mul(stepMask).addStrict(
                state.mul(negStepMask));
          });
          return {output, newStates};
        });
        lastOutput = maskedOutputs.output;
        states = maskedOutputs.newStates;
      }

      if (needPerStepOutputs) {
        perStepOutputs.push(lastOutput);
      }
    }
    let outputs: Tensor;
    if (needPerStepOutputs) {
      const axis = 1;
      outputs = tfc.stack(perStepOutputs, axis);
    }
    return [lastOutput, outputs, states] as [Tensor, Tensor, Tensor[]];
  });
}

export declare interface BaseRNNLayerArgs extends LayerArgs {
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
   *
   * You can set RNN layers to be "stateful", which means that the states
   * computed for the samples in one batch will be reused as initial states
   * for the samples in the next batch. This assumes a one-to-one mapping
   * between samples in different successive batches.
   *
   * To enable "statefulness":
   *   - specify `stateful: true` in the layer constructor.
   *   - specify a fixed batch size for your model, by passing
   *     - if sequential model:
   *       `batchInputShape: [...]` to the first layer in your model.
   *     - else for functional model with 1 or more Input layers:
   *       `batchShape: [...]` to all the first layers in your model.
   *     This is the expected shape of your inputs
   *     *including the batch size*.
   *     It should be a tuple of integers, e.g., `[32, 10, 100]`.
   *   - specify `shuffle: false` when calling `LayersModel.fit()`.
   *
   * To reset the state of your model, call `resetStates()` on either the
   * specific layer or on the entire model.
   */
  stateful?: boolean;
  // TODO(cais): Explore whether we can warn users when they fail to set
  //   `shuffle: false` when training a model consisting of stateful RNNs
  //   and any stateful Layers in general.

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
 * `cell` property required. This interface is to be used with constructors
 * of concrete RNN layer subtypes.
 */
export declare interface RNNLayerArgs extends BaseRNNLayerArgs {
  cell: RNNCell|RNNCell[];
}

/**
 * Base class for recurrent layers.
 *
 * Input shape:
 *   3D tensor with shape `[batchSize, timeSteps, inputDim]`.
 *
 * Output shape:
 *   - if `returnState`, an Array of tensors (i.e., `tf.Tensor`s). The first
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
 *   To reset the states of your model, call `.resetStates()` on either
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
  /** @nocollapse */
  static className = 'RNN';
  public readonly cell: RNNCell;
  public readonly returnSequences: boolean;
  public readonly returnState: boolean;
  public readonly goBackwards: boolean;
  public readonly unroll: boolean;

  public stateSpec: InputSpec[];
  private states_: Tensor[];

  // NOTE(cais): For stateful RNNs, the old states cannot be disposed right
  // away when new states are set, because the old states may need to be used
  // later for backpropagation through time (BPTT) and other purposes. So we
  // keep them here for final disposal when the state is reset completely
  // (i.e., through no-arg call to `resetStates()`).
  private keptStates: Tensor[][];

  private numConstants: number;

  constructor(args: RNNLayerArgs) {
    super(args);
    let cell: RNNCell;
    if (args.cell == null) {
      throw new ValueError(
          'cell property is missing for the constructor of RNN.');
    } else if (Array.isArray(args.cell)) {
      cell = new StackedRNNCells({cells: args.cell});
    } else {
      cell = args.cell;
    }
    if ((cell as RNNCell).stateSize == null) {
      throw new ValueError(
          'The RNN cell should have an attribute `stateSize` (tuple of ' +
          'integers, one integer per RNN state).');
    }
    this.cell = cell;
    this.returnSequences =
        args.returnSequences == null ? false : args.returnSequences;
    this.returnState = args.returnState == null ? false : args.returnState;
    this.goBackwards = args.goBackwards == null ? false : args.goBackwards;
    this._stateful = args.stateful == null ? false : args.stateful;
    this.unroll = args.unroll == null ? false : args.unroll;

    this.supportsMasking = true;
    this.inputSpec = [new InputSpec({ndim: 3})];
    this.stateSpec = null;
    this.states_ = null;
    // TODO(cais): Add constantsSpec and numConstants.
    this.numConstants = null;
    // TODO(cais): Look into the use of initial_state in the kwargs of the
    //   constructor.

    this.keptStates = [];
  }

  // Porting Note: This is the equivalent of `RNN.states` property getter in
  //   PyKeras.
  getStates(): Tensor[] {
    if (this.states_ == null) {
      const numStates =
          Array.isArray(this.cell.stateSize) ? this.cell.stateSize.length : 1;
      return math_utils.range(0, numStates).map(x => null);
    } else {
      return this.states_;
    }
  }

  // Porting Note: This is the equivalent of the `RNN.states` property setter in
  //   PyKeras.
  setStates(states: Tensor[]): void {
    this.states_ = states;
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    if (isArrayOfShapes(inputShape)) {
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

  computeMask(inputs: Tensor|Tensor[], mask?: Tensor|Tensor[]): Tensor
      |Tensor[] {
    return tfc.tidy(() => {
      if (Array.isArray(mask)) {
        mask = mask[0];
      }
      const outputMask = this.returnSequences ? mask : null;

      if (this.returnState) {
        const stateMask = this.states.map(s => null);
        return [outputMask].concat(stateMask);
      } else {
        return outputMask;
      }
    });
  }

  /**
   * Get the current state tensors of the RNN.
   *
   * If the state hasn't been set, return an array of `null`s of the correct
   * length.
   */
  get states(): Tensor[] {
    if (this.states_ == null) {
      const numStates =
          Array.isArray(this.cell.stateSize) ? this.cell.stateSize.length : 1;
      const output: Tensor[] = [];
      for (let i = 0; i < numStates; ++i) {
        output.push(null);
      }
      return output;
    } else {
      return this.states_;
    }
  }

  set states(s: Tensor[]) {
    this.states_ = s;
  }

  public build(inputShape: Shape|Shape[]): void {
    // Note inputShape will be an Array of Shapes of initial states and
    // constants if these are passed in apply().
    const constantShape: Shape[] = null;
    if (this.numConstants != null) {
      throw new NotImplementedError(
          'Constants support is not implemented in RNN yet.');
    }

    if (isArrayOfShapes(inputShape)) {
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
      this.resetStates();
    }
  }

  /**
   * Reset the state tensors of the RNN.
   *
   * If the `states` argument is `undefined` or `null`, will set the
   * state tensor(s) of the RNN to all-zero tensors of the appropriate
   * shape(s).
   *
   * If `states` is provided, will set the state tensors of the RNN to its
   * value.
   *
   * @param states Optional externally-provided initial states.
   * @param training Whether this call is done during training. For stateful
   *   RNNs, this affects whether the old states are kept or discarded. In
   *   particular, if `training` is `true`, the old states will be kept so
   *   that subsequent backpropgataion through time (BPTT) may work properly.
   *   Else, the old states will be discarded.
   */
  resetStates(states?: Tensor|Tensor[], training = false): void {
    tidy(() => {
      if (!this.stateful) {
        throw new AttributeError(
            'Cannot call resetStates() on an RNN Layer that is not stateful.');
      }
      const batchSize = this.inputSpec[0].shape[0];
      if (batchSize == null) {
        throw new ValueError(
            'If an RNN is stateful, it needs to know its batch size. Specify ' +
            'the batch size of your input tensors: \n' +
            '- If using a Sequential model, specify the batch size by ' +
            'passing a `batchInputShape` option to your first layer.\n' +
            '- If using the functional API, specify the batch size by ' +
            'passing a `batchShape` option to your Input layer.');
      }
      // Initialize state if null.
      if (this.states_ == null) {
        if (Array.isArray(this.cell.stateSize)) {
          this.states_ =
              this.cell.stateSize.map(dim => tfc.zeros([batchSize, dim]));
        } else {
          this.states_ = [tfc.zeros([batchSize, this.cell.stateSize])];
        }
      } else if (states == null) {
        // Dispose old state tensors.
        tfc.dispose(this.states_);
        // For stateful RNNs, fully dispose kept old states.
        if (this.keptStates != null) {
          tfc.dispose(this.keptStates);
          this.keptStates = [];
        }

        if (Array.isArray(this.cell.stateSize)) {
          this.states_ =
              this.cell.stateSize.map(dim => tfc.zeros([batchSize, dim]));
        } else {
          this.states_[0] = tfc.zeros([batchSize, this.cell.stateSize]);
        }
      } else {
        if (!Array.isArray(states)) {
          states = [states];
        }
        if (states.length !== this.states_.length) {
          throw new ValueError(
              `Layer ${this.name} expects ${this.states_.length} state(s), ` +
              `but it received ${states.length} state value(s). Input ` +
              `received: ${states}`);
        }

        if (training === true) {
          // Store old state tensors for complete disposal later, i.e., during
          // the next no-arg call to this method. We do not dispose the old
          // states immediately because that BPTT (among other things) require
          // them.
          this.keptStates.push(this.states_.slice());
        } else {
          tfc.dispose(this.states_);
        }

        for (let index = 0; index < this.states_.length; ++index) {
          const value = states[index];
          const dim = Array.isArray(this.cell.stateSize) ?
              this.cell.stateSize[index] :
              this.cell.stateSize;
          const expectedShape = [batchSize, dim];
          if (!util.arraysEqual(value.shape, expectedShape)) {
            throw new ValueError(
                `State ${index} is incompatible with layer ${this.name}: ` +
                `expected shape=${expectedShape}, received shape=${
                    value.shape}`);
          }
          this.states_[index] = value;
        }
      }
      this.states_.forEach(state => tfc.keep(state));
    });
  }

  apply(
      inputs: Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[],
      kwargs?: Kwargs): Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[] {
    // TODO(cais): Figure out whether initialState is in kwargs or inputs.
    let initialState: Tensor[]|SymbolicTensor[] =
        kwargs == null ? null : kwargs['initialState'];
    let constants: Tensor[]|SymbolicTensor[] =
        kwargs == null ? null : kwargs['constants'];
    if (kwargs == null) {
      kwargs = {};
    }

    const standardized =
        standardizeArgs(inputs, initialState, constants, this.numConstants);
    inputs = standardized.inputs;
    initialState = standardized.initialState;
    constants = standardized.constants;

    // If any of `initial_state` or `constants` are specified and are
    // `tf.SymbolicTensor`s, then add them to the inputs and temporarily modify
    // the input_spec to include them.

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
  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    // Input shape: `[samples, time (padded with zeros), input_dim]`.
    // Note that the .build() method of subclasses **must** define
    // this.inputSpec and this.stateSpec owith complete input shapes.
    return tidy(() => {
      const mask = kwargs == null ? null : kwargs['mask'] as Tensor;
      const training = kwargs == null ? null : kwargs['training'];
      let initialState: Tensor[] =
          kwargs == null ? null : kwargs['initialState'];

      inputs = getExactlyOneTensor(inputs);
      if (initialState == null) {
        if (this.stateful) {
          initialState = this.states_;
        } else {
          initialState = this.getInitialState(inputs);
        }
      }

      const numStates =
          Array.isArray(this.cell.stateSize) ? this.cell.stateSize.length : 1;
      if (initialState.length !== numStates) {
        throw new ValueError(
            `RNN Layer has ${numStates} state(s) but was passed ` +
            `${initialState.length} initial state(s).`);
      }
      if (this.unroll) {
        console.warn(
            'Ignoring unroll = true for RNN layer, due to imperative backend.');
      }

      const cellCallKwargs: Kwargs = {training};

      // TODO(cais): Add support for constants.
      const step = (inputs: Tensor, states: Tensor[]) => {
        // `inputs` and `states` are concatenated to form a single `Array` of
        // `tf.Tensor`s as the input to `cell.call()`.
        const outputs =
            this.cell.call([inputs].concat(states), cellCallKwargs) as Tensor[];
        // Marshall the return value into output and new states.
        return [outputs[0], outputs.slice(1)] as [Tensor, Tensor[]];
      };

      // TODO(cais): Add support for constants.

      const rnnOutputs =
          rnn(step, inputs, initialState, this.goBackwards, mask, null,
              this.unroll, this.returnSequences);
      const lastOutput = rnnOutputs[0];
      const outputs = rnnOutputs[1];
      const states = rnnOutputs[2];

      if (this.stateful) {
        this.resetStates(states, training);
      }

      const output = this.returnSequences ? outputs : lastOutput;

      // TODO(cais): Porperty set learning phase flag.

      if (this.returnState) {
        return [output].concat(states);
      } else {
        return output;
      }
    });
  }

  getInitialState(inputs: Tensor): Tensor[] {
    return tidy(() => {
      // Build an all-zero tensor of shape [samples, outputDim].
      // [Samples, timeSteps, inputDim].
      let initialState = tfc.zeros(inputs.shape);
      // [Samples].
      initialState = tfc.sum(initialState, [1, 2]);
      initialState = K.expandDims(initialState);  // [Samples, 1].

      if (Array.isArray(this.cell.stateSize)) {
        return this.cell.stateSize.map(
            dim => dim > 1 ? K.tile(initialState, [1, dim]) : initialState);
      } else {
        return this.cell.stateSize > 1 ?
            [K.tile(initialState, [1, this.cell.stateSize])] :
            [initialState];
      }
    });
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

  setFastWeightInitDuringBuild(value: boolean) {
    super.setFastWeightInitDuringBuild(value);
    if (this.cell != null) {
      this.cell.setFastWeightInitDuringBuild(value);
    }
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      returnSequences: this.returnSequences,
      returnState: this.returnState,
      goBackwards: this.goBackwards,
      stateful: this.stateful,
      unroll: this.unroll,
    };
    if (this.numConstants != null) {
      config['numConstants'] = this.numConstants;
    }
    const cellConfig = this.cell.getConfig();
    config['cell'] = {
      'className': this.cell.getClassName(),
      'config': cellConfig,
    } as serialization.ConfigDictValue;
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(RNN);

/**
 * An RNNCell layer.
 */
// Porting Note: This is a common parent class for RNN cells. There is no
// equivalent of this in PyKeras. Having a common parent class forgoes the
//  need for `has_attr(cell, ...)` checks or its TypeScript equivalent.
/** @doc {heading: 'Layers', subheading: 'Classes'} */
export abstract class RNNCell extends Layer {
  /**
   * Size(s) of the states.
   * For RNN cells with only a single state, this is a single integer.
   */
  public stateSize: number|number[];
  public dropoutMask: Tensor|Tensor[];
  public recurrentDropoutMask: Tensor|Tensor[];
}

export declare interface SimpleRNNCellLayerArgs extends LayerArgs {
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
  /** @nocollapse */
  static className = 'SimpleRNNCell';
  readonly units: number;
  readonly activation: Activation;
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

  constructor(args: SimpleRNNCellLayerArgs) {
    super(args);
    this.units = args.units;
    assertPositiveInteger(this.units, `units`);
    this.activation = getActivation(
        args.activation == null ? this.DEFAULT_ACTIVATION : args.activation);
    this.useBias = args.useBias == null ? true : args.useBias;

    this.kernelInitializer = getInitializer(
        args.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
    this.recurrentInitializer = getInitializer(
        args.recurrentInitializer || this.DEFAULT_RECURRENT_INITIALIZER);

    this.biasInitializer =
        getInitializer(args.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);

    this.kernelRegularizer = getRegularizer(args.kernelRegularizer);
    this.recurrentRegularizer = getRegularizer(args.recurrentRegularizer);
    this.biasRegularizer = getRegularizer(args.biasRegularizer);

    this.kernelConstraint = getConstraint(args.kernelConstraint);
    this.recurrentConstraint = getConstraint(args.recurrentConstraint);
    this.biasConstraint = getConstraint(args.biasConstraint);

    this.dropout = math_utils.min(
        [1, math_utils.max([0, args.dropout == null ? 0 : args.dropout])]);
    this.recurrentDropout = math_utils.min([
      1,
      math_utils.max(
          [0, args.recurrentDropout == null ? 0 : args.recurrentDropout])
    ]);
    this.stateSize = this.units;
    this.dropoutMask = null;
    this.recurrentDropoutMask = null;
  }

  build(inputShape: Shape|Shape[]): void {
    inputShape = getExactlyOneShape(inputShape);
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
  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      inputs = inputs as Tensor[];
      if (inputs.length !== 2) {
        throw new ValueError(
            `SimpleRNNCell expects 2 input Tensors, got ${inputs.length}.`);
      }
      let prevOutput = inputs[1];
      inputs = inputs[0];
      const training = kwargs['training'] == null ? false : kwargs['training'];

      if (0 < this.dropout && this.dropout < 1 && this.dropoutMask == null) {
        this.dropoutMask = generateDropoutMask(
                               () => tfc.onesLike(inputs as Tensor),
                               this.dropout, training) as Tensor;
      }
      if (0 < this.recurrentDropout && this.recurrentDropout < 1 &&
          this.recurrentDropoutMask == null) {
        this.recurrentDropoutMask =
            generateDropoutMask(
                () => tfc.onesLike(prevOutput), this.recurrentDropout,
                training) as Tensor;
      }
      let h: Tensor;
      const dpMask: Tensor = this.dropoutMask as Tensor;
      const recDpMask: Tensor = this.recurrentDropoutMask as Tensor;
      if (dpMask != null) {
        h = K.dot(tfc.mul(inputs, dpMask), this.kernel.read());
      } else {
        h = K.dot(inputs, this.kernel.read());
      }
      if (this.bias != null) {
        h = K.biasAdd(h, this.bias.read());
      }
      if (recDpMask != null) {
        prevOutput = tfc.mul(prevOutput, recDpMask);
      }
      let output = tfc.add(h, K.dot(prevOutput, this.recurrentKernel.read()));
      if (this.activation != null) {
        output = this.activation.apply(output);
      }

      // TODO(cais): Properly set learning phase on output tensor?
      return [output, output];
    });
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
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
serialization.registerClass(SimpleRNNCell);

export declare interface SimpleRNNLayerArgs extends BaseRNNLayerArgs {
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
  /** @nocollapse */
  static className = 'SimpleRNN';
  constructor(args: SimpleRNNLayerArgs) {
    args.cell = new SimpleRNNCell(args);
    super(args as RNNLayerArgs);
    // TODO(cais): Add activityRegularizer.
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      if (this.cell.dropoutMask != null) {
        tfc.dispose(this.cell.dropoutMask);
        this.cell.dropoutMask = null;
      }
      if (this.cell.recurrentDropoutMask != null) {
        tfc.dispose(this.cell.recurrentDropoutMask);
        this.cell.recurrentDropoutMask = null;
      }
      const mask = kwargs == null ? null : kwargs['mask'];
      const training = kwargs == null ? null : kwargs['training'];
      const initialState: Tensor[] =
          kwargs == null ? null : kwargs['initialState'];
      return super.call(inputs, {mask, training, initialState});
    });
  }

  // TODO(cais): Research possibility of refactoring out the tedious all
  //   the getters that delegate to `this.cell` below.
  get units(): number {
    return (this.cell as SimpleRNNCell).units;
  }

  get activation(): Activation {
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

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
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
    delete baseConfig['cell'];
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(SimpleRNN);

// Porting Note: Since this is a superset of SimpleRNNLayerConfig, we extend
//   that interface instead of repeating the fields.
export declare interface GRUCellLayerArgs extends SimpleRNNCellLayerArgs {
  /**
   * Activation function to use for the recurrent step.
   *
   * Defaults to hard sigmoid (`hardSigmoid`).
   *
   * If `null`, no activation is applied.
   */
  recurrentActivation?: ActivationIdentifier;

  /**
   * Implementation mode, either 1 or 2.
   *
   * Mode 1 will structure its operations as a larger number of
   *   smaller dot products and additions.
   *
   * Mode 2 will batch them into fewer, larger operations. These modes will
   * have different performance profiles on different hardware and
   * for different applications.
   *
   * Note: For superior performance, TensorFlow.js always uses implementation
   * 2, regardless of the actual value of this configuration field.
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
  /** @nocollapse */
  static className = 'GRUCell';
  readonly units: number;
  readonly activation: Activation;
  readonly recurrentActivation: Activation;
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
  readonly DEFAULT_RECURRENT_ACTIVATION: ActivationIdentifier = 'hardSigmoid';

  readonly DEFAULT_KERNEL_INITIALIZER = 'glorotNormal';
  readonly DEFAULT_RECURRENT_INITIALIZER = 'orthogonal';
  readonly DEFAULT_BIAS_INITIALIZER: InitializerIdentifier = 'zeros';

  kernel: LayerVariable;
  recurrentKernel: LayerVariable;
  bias: LayerVariable;

  constructor(args: GRUCellLayerArgs) {
    super(args);

    this.units = args.units;
    assertPositiveInteger(this.units, 'units');
    this.activation = getActivation(
        args.activation === undefined ? this.DEFAULT_ACTIVATION :
                                        args.activation);
    this.recurrentActivation = getActivation(
        args.recurrentActivation === undefined ?
            this.DEFAULT_RECURRENT_ACTIVATION :
            args.recurrentActivation);
    this.useBias = args.useBias == null ? true : args.useBias;

    this.kernelInitializer = getInitializer(
        args.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
    this.recurrentInitializer = getInitializer(
        args.recurrentInitializer || this.DEFAULT_RECURRENT_INITIALIZER);

    this.biasInitializer =
        getInitializer(args.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);

    this.kernelRegularizer = getRegularizer(args.kernelRegularizer);
    this.recurrentRegularizer = getRegularizer(args.recurrentRegularizer);
    this.biasRegularizer = getRegularizer(args.biasRegularizer);

    this.kernelConstraint = getConstraint(args.kernelConstraint);
    this.recurrentConstraint = getConstraint(args.recurrentConstraint);
    this.biasConstraint = getConstraint(args.biasConstraint);

    this.dropout = math_utils.min(
        [1, math_utils.max([0, args.dropout == null ? 0 : args.dropout])]);
    this.recurrentDropout = math_utils.min([
      1,
      math_utils.max(
          [0, args.recurrentDropout == null ? 0 : args.recurrentDropout])
    ]);
    this.implementation = args.implementation;
    this.stateSize = this.units;
    this.dropoutMask = null;
    this.recurrentDropoutMask = null;
  }

  public build(inputShape: Shape|Shape[]): void {
    inputShape = getExactlyOneShape(inputShape);
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

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      inputs = inputs as Tensor[];
      if (inputs.length !== 2) {
        throw new ValueError(
            `GRUCell expects 2 input Tensors (inputs, h, c), got ` +
            `${inputs.length}.`);
      }

      const training = kwargs['training'] == null ? false : kwargs['training'];
      let hTMinus1 = inputs[1];  // Previous memory state.
      inputs = inputs[0];

      // Note: For superior performance, TensorFlow.js always uses
      // implementation 2, regardless of the actual value of
      // config.implementation.
      if (0 < this.dropout && this.dropout < 1 && this.dropoutMask == null) {
        this.dropoutMask = generateDropoutMask(
                               () => tfc.onesLike(inputs as Tensor),
                               this.dropout, training, 3) as Tensor[];
      }
      if (0 < this.recurrentDropout && this.recurrentDropout < 1 &&
          this.recurrentDropoutMask == null) {
        this.recurrentDropoutMask =
            generateDropoutMask(
                () => tfc.onesLike(hTMinus1), this.recurrentDropout, training,
                3) as Tensor[];
      }
      const dpMask = this.dropoutMask as [Tensor, Tensor, Tensor];
      const recDpMask = this.recurrentDropoutMask as [Tensor, Tensor, Tensor];
      let z: Tensor;
      let r: Tensor;
      let hh: Tensor;

      if (0 < this.dropout && this.dropout < 1) {
        inputs = tfc.mul(inputs, dpMask[0]);
      }
      let matrixX = K.dot(inputs, this.kernel.read());
      if (this.useBias) {
        matrixX = K.biasAdd(matrixX, this.bias.read());
      }
      if (0 < this.recurrentDropout && this.recurrentDropout < 1) {
        hTMinus1 = tfc.mul(hTMinus1, recDpMask[0]);
      }

      const recurrentKernelValue = this.recurrentKernel.read();
      const [rk1, rk2] = tfc.split(
          recurrentKernelValue, [2 * this.units, this.units],
          recurrentKernelValue.rank - 1);
      const matrixInner = K.dot(hTMinus1, rk1);

      const [xZ, xR, xH] = tfc.split(matrixX, 3, matrixX.rank - 1);
      const [recurrentZ, recurrentR] =
          tfc.split(matrixInner, 2, matrixInner.rank - 1);
      z = this.recurrentActivation.apply(tfc.add(xZ, recurrentZ));
      r = this.recurrentActivation.apply(tfc.add(xR, recurrentR));

      const recurrentH = K.dot(tfc.mul(r, hTMinus1), rk2);
      hh = this.activation.apply(tfc.add(xH, recurrentH));

      const h =
          tfc.add(tfc.mul(z, hTMinus1), tfc.mul(tfc.add(1, tfc.neg(z)), hh));
      // TODO(cais): Add use_learning_phase flag properly.
      return [h, h];
    });
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      units: this.units,
      activation: serializeActivation(this.activation),
      recurrentActivation: serializeActivation(this.recurrentActivation),
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
serialization.registerClass(GRUCell);

// Porting Note: Since this is a superset of SimpleRNNLayerConfig, we inherit
//   from that interface instead of repeating the fields here.
export declare interface GRULayerArgs extends SimpleRNNLayerArgs {
  /**
   * Activation function to use for the recurrent step.
   *
   * Defaults to hard sigmoid (`hardSigmoid`).
   *
   * If `null`, no activation is applied.
   */
  recurrentActivation?: ActivationIdentifier;

  /**
   * Implementation mode, either 1 or 2.
   *
   * Mode 1 will structure its operations as a larger number of
   * smaller dot products and additions.
   *
   * Mode 2 will batch them into fewer, larger operations. These modes will
   * have different performance profiles on different hardware and
   * for different applications.
   *
   * Note: For superior performance, TensorFlow.js always uses implementation
   * 2, regardless of the actual value of this configuration field.
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
  /** @nocollapse */
  static className = 'GRU';
  constructor(args: GRULayerArgs) {
    if (args.implementation === 0) {
      console.warn(
          '`implementation=0` has been deprecated, and now defaults to ' +
          '`implementation=1`. Please update your layer call.');
    }
    args.cell = new GRUCell(args);
    super(args as RNNLayerArgs);
    // TODO(cais): Add activityRegularizer.
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      if (this.cell.dropoutMask != null) {
        tfc.dispose(this.cell.dropoutMask);
        this.cell.dropoutMask = null;
      }
      if (this.cell.recurrentDropoutMask != null) {
        tfc.dispose(this.cell.recurrentDropoutMask);
        this.cell.recurrentDropoutMask = null;
      }
      const mask = kwargs == null ? null : kwargs['mask'];
      const training = kwargs == null ? null : kwargs['training'];
      const initialState: Tensor[] =
          kwargs == null ? null : kwargs['initialState'];
      return super.call(inputs, {mask, training, initialState});
    });
  }

  get units(): number {
    return (this.cell as GRUCell).units;
  }

  get activation(): Activation {
    return (this.cell as GRUCell).activation;
  }

  get recurrentActivation(): Activation {
    return (this.cell as GRUCell).recurrentActivation;
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

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      units: this.units,
      activation: serializeActivation(this.activation),
      recurrentActivation: serializeActivation(this.recurrentActivation),
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
    delete baseConfig['cell'];
    Object.assign(config, baseConfig);
    return config;
  }

  /** @nocollapse */
  static fromConfig<T extends serialization.Serializable>(
      cls: serialization.SerializableConstructor<T>,
      config: serialization.ConfigDict): T {
    if (config['implmentation'] === 0) {
      config['implementation'] = 1;
    }
    return new cls(config);
  }
}
serialization.registerClass(GRU);

// Porting Note: Since this is a superset of SimpleRNNLayerConfig, we extend
//   that interface instead of repeating the fields.
export declare interface LSTMCellLayerArgs extends SimpleRNNCellLayerArgs {
  /**
   * Activation function to use for the recurrent step.
   *
   * Defaults to hard sigmoid (`hardSigmoid`).
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
   *
   * Note: For superior performance, TensorFlow.js always uses implementation
   * 2, regardless of the actual value of this configuration field.
   */
  implementation?: number;
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
  /** @nocollapse */
  static className = 'LSTMCell';
  readonly units: number;
  readonly activation: Activation;
  readonly recurrentActivation: Activation;
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

  constructor(args: LSTMCellLayerArgs) {
    super(args);

    this.units = args.units;
    assertPositiveInteger(this.units, 'units');
    this.activation = getActivation(
        args.activation === undefined ? this.DEFAULT_ACTIVATION :
                                        args.activation);
    this.recurrentActivation = getActivation(
        args.recurrentActivation === undefined ?
            this.DEFAULT_RECURRENT_ACTIVATION :
            args.recurrentActivation);
    this.useBias = args.useBias == null ? true : args.useBias;

    this.kernelInitializer = getInitializer(
        args.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
    this.recurrentInitializer = getInitializer(
        args.recurrentInitializer || this.DEFAULT_RECURRENT_INITIALIZER);

    this.biasInitializer =
        getInitializer(args.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);
    this.unitForgetBias = args.unitForgetBias;

    this.kernelRegularizer = getRegularizer(args.kernelRegularizer);
    this.recurrentRegularizer = getRegularizer(args.recurrentRegularizer);
    this.biasRegularizer = getRegularizer(args.biasRegularizer);

    this.kernelConstraint = getConstraint(args.kernelConstraint);
    this.recurrentConstraint = getConstraint(args.recurrentConstraint);
    this.biasConstraint = getConstraint(args.biasConstraint);

    this.dropout = math_utils.min(
        [1, math_utils.max([0, args.dropout == null ? 0 : args.dropout])]);
    this.recurrentDropout = math_utils.min([
      1,
      math_utils.max(
          [0, args.recurrentDropout == null ? 0 : args.recurrentDropout])
    ]);
    this.implementation = args.implementation;
    this.stateSize = [this.units, this.units];
    this.dropoutMask = null;
    this.recurrentDropoutMask = null;
  }

  public build(inputShape: Shape|Shape[]): void {
    inputShape = getExactlyOneShape(inputShape);
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
          /** @nocollapse */
          static className = 'CustomInit';

          apply(shape: Shape, dtype?: DataType): Tensor {
            // TODO(cais): More informative variable names?
            const bI = capturedBiasInit.apply([capturedUnits]);
            const bF = (new Ones()).apply([capturedUnits]);
            const bCAndH = capturedBiasInit.apply([capturedUnits * 2]);
            return K.concatAlongFirstAxis(
                K.concatAlongFirstAxis(bI, bF), bCAndH);
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

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      const training = kwargs['training'] == null ? false : kwargs['training'];
      inputs = inputs as Tensor[];
      if (inputs.length !== 3) {
        throw new ValueError(
            `LSTMCell expects 3 input Tensors (inputs, h, c), got ` +
            `${inputs.length}.`);
      }
      let hTMinus1 = inputs[1];    // Previous memory state.
      const cTMinus1 = inputs[2];  // Previous carry state.
      inputs = inputs[0];
      if (0 < this.dropout && this.dropout < 1 && this.dropoutMask == null) {
        this.dropoutMask = generateDropoutMask(
                               () => tfc.onesLike(inputs as Tensor),
                               this.dropout, training, 4) as Tensor[];
      }
      if (0 < this.recurrentDropout && this.recurrentDropout < 1 &&
          this.recurrentDropoutMask == null) {
        this.recurrentDropoutMask =
            generateDropoutMask(
                () => tfc.onesLike(hTMinus1), this.recurrentDropout, training,
                4) as Tensor[];
      }
      const dpMask = this.dropoutMask as [Tensor, Tensor, Tensor, Tensor];
      const recDpMask =
          this.recurrentDropoutMask as [Tensor, Tensor, Tensor, Tensor];

      // Note: For superior performance, TensorFlow.js always uses
      // implementation 2 regardless of the actual value of
      // config.implementation.
      let i: Tensor;
      let f: Tensor;
      let c: Tensor;
      let o: Tensor;
      if (0 < this.dropout && this.dropout < 1) {
        inputs = tfc.mul(inputs, dpMask[0]);
      }
      let z = K.dot(inputs, this.kernel.read());
      if (0 < this.recurrentDropout && this.recurrentDropout < 1) {
        hTMinus1 = tfc.mul(hTMinus1, recDpMask[0]);
      }
      z = tfc.add(z, K.dot(hTMinus1, this.recurrentKernel.read()));
      if (this.useBias) {
        z = K.biasAdd(z, this.bias.read());
      }

      const [z0, z1, z2, z3] = tfc.split(z, 4, z.rank - 1);

      i = this.recurrentActivation.apply(z0);
      f = this.recurrentActivation.apply(z1);
      c = tfc.add(tfc.mul(f, cTMinus1), tfc.mul(i, this.activation.apply(z2)));
      o = this.recurrentActivation.apply(z3);

      const h = tfc.mul(o, this.activation.apply(c));
      // TODO(cais): Add use_learning_phase flag properly.
      return [h, h, c];
    });
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      units: this.units,
      activation: serializeActivation(this.activation),
      recurrentActivation: serializeActivation(this.recurrentActivation),
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
serialization.registerClass(LSTMCell);

// Porting Note: Since this is a superset of SimpleRNNLayerConfig, we inherit
//   from that interface instead of repeating the fields here.
export declare interface LSTMLayerArgs extends SimpleRNNLayerArgs {
  /**
   * Activation function to use for the recurrent step.
   *
   * Defaults to hard sigmoid (`hardSigmoid`).
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
   *   Mode 1 will structure its operations as a larger number of
   *   smaller dot products and additions, whereas mode 2 will
   *   batch them into fewer, larger operations. These modes will
   *   have different performance profiles on different hardware and
   *   for different applications.
   *
   * Note: For superior performance, TensorFlow.js always uses implementation
   * 2, regardless of the actual value of this config field.
   */
  implementation?: number;
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
  /** @nocollapse */
  static className = 'LSTM';
  constructor(args: LSTMLayerArgs) {
    if (args.implementation as number === 0) {
      console.warn(
          '`implementation=0` has been deprecated, and now defaults to ' +
          '`implementation=1`. Please update your layer call.');
    }
    args.cell = new LSTMCell(args);
    super(args as RNNLayerArgs);
    // TODO(cais): Add activityRegularizer.
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      if (this.cell.dropoutMask != null) {
        tfc.dispose(this.cell.dropoutMask);
        this.cell.dropoutMask = null;
      }
      if (this.cell.recurrentDropoutMask != null) {
        tfc.dispose(this.cell.recurrentDropoutMask);
        this.cell.recurrentDropoutMask = null;
      }
      const mask = kwargs == null ? null : kwargs['mask'];
      const training = kwargs == null ? null : kwargs['training'];
      const initialState: Tensor[] =
          kwargs == null ? null : kwargs['initialState'];
      return super.call(inputs, {mask, training, initialState});
    });
  }

  get units(): number {
    return (this.cell as LSTMCell).units;
  }

  get activation(): Activation {
    return (this.cell as LSTMCell).activation;
  }

  get recurrentActivation(): Activation {
    return (this.cell as LSTMCell).recurrentActivation;
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

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      units: this.units,
      activation: serializeActivation(this.activation),
      recurrentActivation: serializeActivation(this.recurrentActivation),
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
    delete baseConfig['cell'];
    Object.assign(config, baseConfig);
    return config;
  }

  /** @nocollapse */
  static fromConfig<T extends serialization.Serializable>(
      cls: serialization.SerializableConstructor<T>,
      config: serialization.ConfigDict): T {
    if (config['implmentation'] === 0) {
      config['implementation'] = 1;
    }
    return new cls(config);
  }
}
serialization.registerClass(LSTM);

export declare interface StackedRNNCellsArgs extends LayerArgs {
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
  /** @nocollapse */
  static className = 'StackedRNNCells';
  protected cells: RNNCell[];

  constructor(args: StackedRNNCellsArgs) {
    super(args);
    this.cells = args.cells;
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

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
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
    });
  }

  public build(inputShape: Shape|Shape[]): void {
    if (isArrayOfShapes(inputShape)) {
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

  getConfig(): serialization.ConfigDict {
    const cellConfigs: serialization.ConfigDict[] = [];
    for (const cell of this.cells) {
      cellConfigs.push({
        'className': this.getClassName(),
        'config': cell.getConfig(),
      });
    }
    const config: serialization.ConfigDict = {'cells': cellConfigs};
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  /** @nocollapse */
  static fromConfig<T extends serialization.Serializable>(
      cls: serialization.SerializableConstructor<T>,
      config: serialization.ConfigDict,
      customObjects = {} as serialization.ConfigDict): T {
    const cells: RNNCell[] = [];
    for (const cellConfig of (config['cells'] as serialization.ConfigDict[])) {
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
   * @returns A flat `Array` of `tf.Tensor`s.
   */
  getWeights(): Tensor[] {
    const weights: LayerVariable[] = [];
    for (const cell of this.cells) {
      weights.push(...cell.weights);
    }
    return batchGetValue(weights);
  }

  /**
   * Set the weights of the model.
   *
   * @param weights An `Array` of `tf.Tensor`s with shapes and types matching
   *     the output of `getWeights()`.
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
    batchSetValue(tuples);
  }

  // TODO(cais): Maybe implemnt `losses` and `getLossesFor`.
}
serialization.registerClass(StackedRNNCells);

function generateDropoutMask(
    ones: () => Tensor, rate: number, training: boolean = null,
    count = 1): Tensor|Tensor[] {
  function droppedInputs(): Tensor {
    return K.dropout(ones(), rate);
  }
  if (count > 1) {
    const mask: Tensor[] = [];
    for (let i = 0; i < count; i++) {
      mask.push(K.inTrainPhase(droppedInputs, ones, training));
    }
    mask.forEach(m => tfc.keep(m));
    return mask;
  } else {
    return tfc.keep(K.inTrainPhase(droppedInputs, ones, training));
  }
}
