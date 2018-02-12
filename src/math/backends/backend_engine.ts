/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {ENV} from '../../environment';
import {tidy} from '../../globals';
import * as util from '../../util';
import * as ops from '../ops';
import {DataId, Tensor, Tensor3D, Variable} from '../tensor';
import {NamedTensorMap, NamedVariableMap, TypedArray} from '../types';
import {Rank} from '../types';

import {MathBackend} from './backend';
import * as kernel_registry from './kernel_registry';
import {KernelConfigRegistry} from './kernel_registry';
import {Profiler} from './profiler';
// tslint:disable-next-line:max-line-length
import {KernelNode, Tape, TapeNode, TapeNodeInputGradientTensors} from './tape_types';
import * as tape_util from './tape_util';
import {ScopeResultImmediate} from './tape_util';

interface ScopeState {
  keep: Tensor[];
  track: Tensor[];
}

/**
 * @docalias (...inputs: Tensor[]) => {
 *   value: Tensor,
 *   gradFunc: (dy: Tensor) => Tensor[]
 * }
 */
export type CustomGradientFunc<T extends Tensor> = (...args: Tensor[]) => {
  value: T, gradFunc: (dy: T) => Tensor[];
};

export interface TensorManager {
  registerTensor(a: Tensor): void;
  registerVariable(v: Variable): void;
  disposeTensor(a: Tensor): void;
  memory(): {numDataBuffers: number; numBytes: number;};
}

export type MemoryInfo = {
  numTensors: number; numDataBuffers: number; numBytes: number; backendInfo: {};
  unreliable?: boolean;
};

export class BackendEngine implements TensorManager {
  // Public since optimizers will use it.
  registeredVariables: NamedVariableMap = {};

  private refCounter = new WeakMap<DataId, number>();
  private nextTapeNodeId = 0;
  private numBytes = 0;
  private numTensors = 0;
  private numDataBuffers = 0;

  private activeTape: Tape;
  private gradientScopeCount = 0;
  private customGradientDepth = 0;

  // Keep Tensors that parallel the tapes.
  private activeScope: ScopeState;
  private scopeStack: ScopeState[];
  private profiler: Profiler;

  constructor(
      public backend: MathBackend, private customBackend: boolean,
      public safeMode: boolean) {
    // Create a default outer scope.
    this.activeScope = {keep: [], track: []};
    this.scopeStack = [this.activeScope];
    this.profiler = new Profiler(backend);
  }

  executeKernel<R extends Rank, K extends keyof KernelConfigRegistry<R>, C
                    extends KernelConfigRegistry<R>[K]['inputAndArgs']>(
      kernelName: K, config: C, grad?: KernelConfigRegistry<R>[K]['gradient']):
      KernelConfigRegistry<R>[K]['output'] {
    let result: KernelConfigRegistry<R>[K]['output'];
    if (!ENV.get('DEBUG')) {
      // NOTE: This isn't pulled out into a separate function to so that we
      // keep a shallow stack trace.
      result = kernel_registry.executeKernel(this.backend, kernelName, config);
    } else {
      result = this.profiler.profileKernel(
          kernelName,
          () =>
              kernel_registry.executeKernel(this.backend, kernelName, config));
    }

    const recordKernel =
        this.activeTape != null && this.customGradientDepth === 0;
    if (recordKernel) {
      config = tape_util.stripUndefinedInputsFromInputConfig(config) as C;

      const evaluatedNode: KernelNode = {
        id: this.nextTapeNodeId++,
        type: 'kernel',
        name: `kernel: ${kernelName}`,
        kernel: kernelName,
        inputAndArgs: config,
        output: result,
        gradient: grad
      };
      this.activeTape.push(evaluatedNode);
    }

    return result;
  }

  // TensorManager implementation.

  registerTensor(a: Tensor|Variable): void {
    const refCount =
        this.refCounter.has(a.dataId) ? this.refCounter.get(a.dataId) : 0;
    this.numTensors++;
    if (refCount === 0) {
      this.numDataBuffers++;
      this.numBytes +=
          util.sizeFromShape(a.shape) * util.bytesPerElement(a.dtype);
      this.backend.register(a.dataId, a.shape, a.dtype);
    }
    this.refCounter.set(a.dataId, refCount + 1);
    if (!(a instanceof Variable)) {
      this.track(a);
    }
  }

  registerVariable(v: Variable) {
    if (this.registeredVariables[v.name] != null) {
      throw new Error(`Variable with name ${v.name} was already registered`);
    }
    this.registeredVariables[v.name] = v;
  }

  disposeTensor(a: Tensor): void {
    if (!this.refCounter.has(a.dataId)) {
      return;
    }
    this.numTensors--;
    const refCount = this.refCounter.get(a.dataId);
    if (refCount <= 1) {
      this.refCounter.delete(a.dataId);
      this.backend.disposeData(a.dataId);
      this.numDataBuffers--;
      this.numBytes -=
          util.sizeFromShape(a.shape) * util.bytesPerElement(a.dtype);
    } else {
      this.refCounter.set(a.dataId, refCount - 1);
    }
    // TODO(nsthorat): Construct an error and save the stack trace for
    // debugging when in debug mode. Creating a stack trace is too expensive
    // to do unconditionally.
  }

  memory(): MemoryInfo {
    const backendInfo = this.backend.memory();
    const memInfo: MemoryInfo = {
      numTensors: this.numTensors,
      numDataBuffers: this.numDataBuffers,
      numBytes: this.numBytes,
      backendInfo,
    };
    if (backendInfo.unreliable) {
      memInfo.unreliable = true;
    }
    return memInfo;
  }

  private shouldRecord(): boolean {
    return this.activeTape != null && this.customGradientDepth === 0;
  }

  private addTapeNode(
      inputs: Tensor[], result: Tensor,
      gradientsFunc: (dy: Tensor) => Tensor[]): void {
    const inputsMap: NamedTensorMap = {};
    inputs.forEach((input, idx) => {
      inputsMap[idx] = input;
    });

    const gradient = (dy: Tensor) => {
      const res = gradientsFunc(dy);
      const resMap: TapeNodeInputGradientTensors = {};
      res.forEach((r, idx) => {
        resMap[idx] = () => r;
      });
      return resMap;
    };

    const evaluatedNode: TapeNode<Tensor> = {
      id: this.nextTapeNodeId++,
      type: 'customGradient',
      name,
      inputAndArgs: {inputs: inputsMap},
      output: result,
      gradient
    };
    this.activeTape.push(evaluatedNode);
  }

  keep<T extends Tensor>(result: T): T {
    if (this.scopeStack.length === 1 && ENV.engine.safeMode) {
      throw new Error(
          'Safe mode is ON. Enclose all tensor operations inside dl.tidy(): ' +
          'dl.tidy(() => {...}) to avoid memory leaks.');
    }
    this.activeScope.keep.push(result);
    return result;
  }

  /**
   * Start a scope. Use this with endScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  startScope(gradientsMode = false) {
    if (gradientsMode && this.gradientScopeCount === 0) {
      this.activeTape = [];
    }
    if (gradientsMode) {
      this.gradientScopeCount++;
    }

    const newScopeArrays: ScopeState = {keep: [], track: []};
    this.scopeStack.push(newScopeArrays);
    this.activeScope = newScopeArrays;
  }

  /**
   * End a scope. Use this with startScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  endScope(result: ScopeResultImmediate, gradientsMode = false) {
    if (gradientsMode) {
      this.gradientScopeCount--;
      if (this.gradientScopeCount === 0) {
        this.activeTape = null;
      }
    }

    let tensorsToKeep = this.activeScope.keep;
    const tensorsToTrackInParent =
        tape_util.extractTensorsFromScopeResult(result);
    tensorsToKeep = tensorsToKeep.concat(tensorsToTrackInParent);

    // Dispose the arrays tracked in this scope.
    for (let i = 0; i < this.activeScope.track.length; i++) {
      const tensor = this.activeScope.track[i];
      if (util.isTensorInList(tensor, tensorsToKeep)) {
        continue;
      }

      if (this.activeTape != null) {
        tensorsToTrackInParent.push(tensor);
      } else {
        tensor.dispose();
      }
    }

    this.scopeStack.pop();
    this.activeScope = this.scopeStack.length === 0 ?
        {keep: [], track: []} :
        this.scopeStack[this.scopeStack.length - 1];

    // Track the current result in the parent scope.
    tensorsToTrackInParent.forEach(tensor => {
      if (!util.isTensorInList(tensor, this.activeScope.keep)) {
        this.track(tensor);
      }
    });
  }

  dispose() {
    if (this.customBackend) {
      this.backend.dispose();
    }
  }

  /**
   * Returns gradients of `f` w.r.t. each of the `xs`. The gradients returned
   * are of the same length as `xs`, but some might be null if `f` was not
   * a function of that `x`. It also takes optional dy to multiply the gradient,
   * which defaults to `1`.
   */
  gradients<T extends Tensor>(f: () => T, xs: Tensor[], dy?: T):
      {value: T, grads: Tensor[]} {
    return tidy('gradients', () => {
      const y = f();
      // Filter out the nodes that don't connect x => y.
      const filteredTape =
          tape_util.getFilteredNodesXToY(this.activeTape, xs, y);
      if (filteredTape.length === 0 && xs.length > 0) {
        throw new Error(
            `Cannot compute gradient: y is not a function of \`x\`s. ` +
            `Make sure the xs you are computing gradients with respect ` +
            `to are used inside the gradient function.`);
      }

      const accumulatedGradientMap: {[tensorId: number]: Tensor} = {};
      accumulatedGradientMap[y.id] = (dy == null) ? ops.onesLike(y) : dy;

      // Backprop gradients through the filtered nodes.
      tape_util.backpropagateGradients(accumulatedGradientMap, filteredTape);

      const grads = xs.map(x => accumulatedGradientMap[x.id]);
      return {value: y, grads};
    }, true /* gradientsMode */);
  }

  customGrad<T extends Tensor>(f: CustomGradientFunc<T>):
      (...args: Tensor[]) => T {
    this.customGradientDepth++;

    return (...inputs: Tensor[]): T => {
      let gradientsFunc: (dy: T) => Tensor[];
      const gradientsMode = true;
      const result = tidy(f.name, () => {
        const {value, gradFunc} = f(...inputs);
        gradientsFunc = gradFunc;
        return value;
      }, gradientsMode);

      this.customGradientDepth--;

      if (this.shouldRecord()) {
        this.addTapeNode(inputs, result, gradientsFunc);
      }
      return result;
    };
  }

  // Forwarding to backend.
  write(dataId: DataId, values: TypedArray): void {
    this.backend.write(dataId, values);
  }
  readSync(dataId: DataId): TypedArray {
    return this.backend.readSync(dataId);
  }
  read(dataId: DataId): Promise<TypedArray> {
    return this.backend.read(dataId);
  }
  fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): Tensor3D {
    return this.backend.fromPixels(pixels, numChannels);
  }
  time(query: () => void): Promise<number> {
    return this.backend.time(query);
  }

  /**
   * Tracks a Tensor in the current scope to be automatically cleaned up
   * when the current scope ends, and returns the value.
   *
   * @param result The Tensor to track in the current scope.
   */
  private track<T extends Tensor>(result: T): T {
    if (this.scopeStack.length === 1 && this.safeMode) {
      throw new Error(
          'Safe mode is ON. Enclose all tensor operations inside dl.tidy(): ' +
          'dl.tidy(() => {op();...}); to avoid memory leaks.');
    }
    this.activeScope.track.push(result);
    return result;
  }
}
