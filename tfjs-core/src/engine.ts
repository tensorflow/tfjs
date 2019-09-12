/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {BackendTimingInfo, DataMover, KernelBackend} from './backends/backend';
import {Environment, setEnvironmentGlobal} from './environment';
import {Profiler} from './profiler';
import {backpropagateGradients, getFilteredNodesXToY, NamedGradientMap, TapeNode} from './tape';
import {DataId, setTensorTracker, Tensor, Tensor3D, TensorTracker, Variable} from './tensor';
import {GradSaveFunc, NamedTensorMap, NamedVariableMap, TensorContainer} from './tensor_types';
import {getTensorsInContainer} from './tensor_util';
import {BackendValues, DataType, PixelData} from './types';
import * as util from './util';
import {bytesFromStringArray, makeOnesTypedArray, now, sizeFromShape} from './util';

/**
 * A function that computes an output. The save function is for saving tensors
 * computed in the forward pass, that we need in the backward pass.
 */
export type ForwardFunc<T> = (backend: KernelBackend, save?: GradSaveFunc) => T;

/**
 * @docalias (a: Tensor, b: Tensor,..., save?: Function) => {
 *   value: Tensor,
 *   gradFunc: (dy: Tensor, saved?: NamedTensorMap) => Tensor | Tensor[]
 * }
 */
export type CustomGradientFunc<T extends Tensor> =
    (...inputs: Array<Tensor|GradSaveFunc>) => {
      value: T;
      gradFunc: (dy: T, saved: Tensor[]) => Tensor | Tensor[];
    };

export type MemoryInfo = {
  numTensors: number; numDataBuffers: number; numBytes: number;
  unreliable?: boolean; reasons: string[];
};

type KernelProfile = {
  name: string; bytesAdded: number; totalBytesSnapshot: number;
  tensorsAdded: number;
  totalTensorsSnapshot: number;
  inputShapes: number[][];
  outputShape: number[] | number[][];
};

export type ProfileInfo = {
  newBytes: number; newTensors: number; peakBytes: number;
  kernels: KernelProfile[];
  result: TensorContainer;
};

export interface TimingInfo extends BackendTimingInfo {
  wallMs: number;
}

/** @docalias Function */
export type ScopeFn<T extends TensorContainer> = () => T;

export interface TensorManager {
  registerTensor(a: Tensor, backend?: KernelBackend): void;
  registerVariable(v: Variable): void;
  disposeTensor(a: Tensor): void;
  memory(): {numDataBuffers: number; numBytes: number;};
}

interface ScopeState {
  track: Tensor[];
  name: string;
  id: number;
}

class EngineState {
  // Public since optimizers will use it.
  registeredVariables: NamedVariableMap = {};

  nextTapeNodeId = 0;
  numBytes = 0;
  numTensors = 0;
  numStringTensors = 0;
  numDataBuffers = 0;

  activeTape: TapeNode[];
  // Number of nested tf.grad() statements when computing higher-order
  // gradients. E.g. `1` for first-order gradients and `2` for second-order
  // gradients. Used to track if the tape should be removed after a backprop.
  gradientDepth = 0;
  // Number of nested kernel calls. When kernel depth is greater than 1, we turn
  // off the tape.
  kernelDepth = 0;

  // Keep Tensors that parallel the tapes.
  activeScope: ScopeState;
  scopeStack: ScopeState[] = [];
  nextScopeId = 0;

  tensorInfo = new WeakMap<DataId, {
    backend: KernelBackend,
    bytes: number,
    dtype: DataType,
    shape: number[],
    refCount: number
  }>();

  profiling = false;
  activeProfile: ProfileInfo =
      {newBytes: 0, newTensors: 0, peakBytes: 0, kernels: [], result: null};

  dispose() {
    for (const variableName in this.registeredVariables) {
      this.registeredVariables[variableName].dispose();
    }
  }
}

export class Engine implements TensorManager, TensorTracker, DataMover {
  state: EngineState;
  backendName: string;
  registry: {[id: string]: KernelBackend} = {};
  registryFactory: {
    [id: string]: {
      factory: () => KernelBackend | Promise<KernelBackend>,
      priority: number
    }
  } = {};

  private profiler: Profiler;
  private backendInstance: KernelBackend;
  private pendingBackendInit: Promise<boolean>;
  private pendingBackendInitId = 0;

  constructor(public ENV: Environment) {
    this.state = new EngineState();
  }

  async ready(): Promise<void> {
    if (this.pendingBackendInit != null) {
      return this.pendingBackendInit.then(() => {});
    }
    if (this.backendInstance != null) {
      return;
    }
    const sortedBackends = this.getSortedBackends();

    for (let i = 0; i < sortedBackends.length; i++) {
      const backendName = sortedBackends[i];
      const success = await this.initializeBackend(backendName).success;
      if (success) {
        await this.setBackend(backendName);
        return;
      }
    }

    throw new Error(
        `Could not initialize any backends, all backend initializations ` +
        `failed.`);
  }

  get backend(): KernelBackend {
    if (this.pendingBackendInit != null) {
      throw new Error(
          `Backend '${this.backendName}' has not yet been initialized. Make ` +
          `sure to await tf.ready() before calling other methods`);
    }
    if (this.backendInstance == null) {
      const {name, asyncInit} = this.initializeBackendsAndReturnBest();
      if (asyncInit) {
        throw new Error(
            `The highest priority backend '${name}' has not yet been ` +
            `initialized. Make sure to await tf.ready() before calling ` +
            `other methods`);
      }
      this.setBackend(name);
    }
    return this.backendInstance;
  }

  backendNames(): string[] {
    return Object.keys(this.registryFactory);
  }

  findBackend(backendName: string): KernelBackend {
    if (!(backendName in this.registry)) {
      // If the backend hasn't been initialized but we have a registry entry for
      // it, initialize it and return it.
      if (backendName in this.registryFactory) {
        const {asyncInit} = this.initializeBackend(backendName);
        if (asyncInit) {
          // Backend is not ready yet.
          return null;
        }
      } else {
        return null;
      }
    }
    return this.registry[backendName];
  }

  findBackendFactory(backendName: string):
      () => KernelBackend | Promise<KernelBackend> {
    if (!(backendName in this.registryFactory)) {
      return null;
    }
    return this.registryFactory[backendName].factory;
  }

  registerBackend(
      backendName: string,
      factory: () => KernelBackend | Promise<KernelBackend>,
      priority = 1): boolean {
    if (backendName in this.registryFactory) {
      console.warn(
          `${backendName} backend was already registered. ` +
          `Reusing existing backend factory.`);
      return false;
    }
    this.registryFactory[backendName] = {factory, priority};
    return true;
  }

  async setBackend(backendName: string): Promise<boolean> {
    if (this.registryFactory[backendName] == null) {
      throw new Error(`Backend name '${backendName}' not found in registry`);
    }
    this.backendName = backendName;
    if (this.registry[backendName] == null) {
      this.backendInstance = null;
      const {success, asyncInit} = this.initializeBackend(backendName);
      const result = asyncInit ? await success : success;
      if (!result) {
        return false;
      }
    }
    this.backendInstance = this.registry[backendName];

    // Reset the profiler.
    this.profiler = new Profiler(this.backendInstance);

    return true;
  }

  /**
   * Initializes a backend by looking up the backend name in the factory
   * registry and calling the factory method. Returns a boolean representing
   * whether the initialization of the backend suceeded. Throws an error if
   * there is no backend in the factory registry.
   */
  private initializeBackend(backendName: string):
      {success: boolean|Promise<boolean>, asyncInit: boolean} {
    const registryFactoryEntry = this.registryFactory[backendName];
    if (registryFactoryEntry == null) {
      throw new Error(
          `Cannot initialize backend ${backendName}, no registration found.`);
    }

    try {
      const backend = registryFactoryEntry.factory();
      // Test if the factory returns a promise.
      if (Promise.resolve(backend) === backend) {
        const promiseId = ++this.pendingBackendInitId;
        const success =
            backend
                .then(backendInstance => {
                  // Outdated promise. Another backend was set in the meantime.
                  if (promiseId < this.pendingBackendInitId) {
                    return false;
                  }
                  this.registry[backendName] = backendInstance;
                  this.pendingBackendInit = null;
                  return true;
                })
                .catch(err => {
                  // Outdated promise. Another backend was set in the meantime.
                  if (promiseId < this.pendingBackendInitId) {
                    return false;
                  }
                  this.pendingBackendInit = null;
                  console.warn(
                      `Initialization of backend ${backendName} failed`);
                  console.warn(err.stack || err.message);
                  return false;
                });
        this.pendingBackendInit = success;
        return {success, asyncInit: true};
      } else {
        this.registry[backendName] = backend as KernelBackend;
        return {success: true, asyncInit: false};
      }
    } catch (err) {
      console.warn(`Initialization of backend ${backendName} failed`);
      console.warn(err.stack || err.message);
      return {success: false, asyncInit: false};
    }
  }

  removeBackend(backendName: string): void {
    if (!(backendName in this.registryFactory)) {
      throw new Error(`${backendName} backend not found in registry`);
    }
    if (this.backendName === backendName && this.pendingBackendInit != null) {
      // There is a pending promise of the backend we want to remove. Make it
      // obsolete.
      this.pendingBackendInitId++;
    }

    if (backendName in this.registry) {
      this.registry[backendName].dispose();
      delete this.registry[backendName];
    }

    delete this.registryFactory[backendName];

    // Unset the backend if it is active.
    if (this.backendName === backendName) {
      this.pendingBackendInit = null;
      this.backendName = null;
      this.backendInstance = null;
    }
  }

  private getSortedBackends(): string[] {
    if (Object.keys(this.registryFactory).length === 0) {
      throw new Error('No backend found in registry.');
    }
    return Object.keys(this.registryFactory).sort((a: string, b: string) => {
      // Highest priority comes first.
      return this.registryFactory[b].priority -
          this.registryFactory[a].priority;
    });
  }

  private initializeBackendsAndReturnBest():
      {name: string, asyncInit: boolean} {
    const sortedBackends = this.getSortedBackends();

    for (let i = 0; i < sortedBackends.length; i++) {
      const backendName = sortedBackends[i];
      const {success, asyncInit} = this.initializeBackend(backendName);
      if (asyncInit || success) {
        return {name: backendName, asyncInit};
      }
    }
    throw new Error(
        `Could not initialize any backends, all backend initializations ` +
        `failed.`);
  }

  moveData(destBackend: KernelBackend, dataId: DataId) {
    this.write(destBackend, dataId, this.readSync(dataId));
  }

  tidy<T extends TensorContainer>(nameOrFn: string|ScopeFn<T>, fn?: ScopeFn<T>):
      T {
    let name: string = null;
    if (fn == null) {
      // Called with only 1 argument.
      if (typeof nameOrFn !== 'function') {
        throw new Error('Please provide a function to tidy()');
      }
      fn = nameOrFn;
    } else {
      // Called with 2 arguments.
      if (typeof nameOrFn !== 'string' && !(nameOrFn instanceof String)) {
        throw new Error(
            'When calling with two arguments, the first argument ' +
            'to tidy() must be a string');
      }
      if (typeof fn !== 'function') {
        throw new Error(
            'When calling with two arguments, the 2nd argument ' +
            'to tidy() must be a function');
      }
      name = nameOrFn as string;
      // TODO(nsthorat,smilkov): Do operation logging and performance
      // profiling.
    }
    let result: T;
    return this.scopedRun(
        () => this.startScope(name), () => this.endScope(result), () => {
          result = fn();
          if (result instanceof Promise) {
            console.error('Cannot return a Promise inside of tidy.');
          }
          return result;
        });
  }

  private scopedRun<T>(start: () => void, end: () => void, f: () => T): T {
    start();
    try {
      const res = f();
      end();
      return res;
    } catch (ex) {
      end();
      throw ex;
    }
  }

  private static nextTensorId = 0;
  nextTensorId(): number {
    return Engine.nextTensorId++;
  }

  private static nextVariableId = 0;
  nextVariableId(): number {
    return Engine.nextVariableId++;
  }

  /**
   * This method is called instead of the public-facing tensor.clone() when
   * saving a tensor for backwards pass. It makes sure to add the clone
   * operation to the tape regardless of being called inside a kernel
   * execution.
   */
  private clone(x: Tensor): Tensor {
    const y = Tensor.make(x.shape, {dataId: x.dataId}, x.dtype);
    this.addTapeNode([x], y, dy => [dy.toFloat()]);
    return y;
  }

  runKernel<T extends Tensor|Tensor[], I extends NamedTensorMap>(
      forwardFunc: ForwardFunc<T>,
      inputs: I,
      backwardsFunc?: (dy: T, saved: Tensor[]) => {[P in keyof I]: () => I[P]},
      ): T {
    let result: T;
    let saved: Tensor[] = [];
    const isTapeOn = this.isTapeOn();
    const scopeName =
        this.state.activeScope != null ? this.state.activeScope.name : '';
    const saveFunc: GradSaveFunc = (tensors) => {
      // Do not save unless we are recording to the tape. Otherwise it would
      // cause a mem leak since we would never run backprop, which disposes
      // the kept tensors.
      if (!isTapeOn) {
        return;
      }
      saved = tensors.map(tensor => this.keep(this.clone(tensor)));
    };

    const startingBytecount = this.state.numBytes;
    const startingNumTensors = this.state.numTensors;

    // Stop recording to a tape when running a kernel.
    this.scopedRun(
        () => this.state.kernelDepth++, () => this.state.kernelDepth--, () => {
          if (!this.ENV.getBool('DEBUG')) {
            result = forwardFunc(this.backend, saveFunc);
          } else {
            result = this.profiler.profileKernel(
                scopeName, inputs, () => forwardFunc(this.backend, saveFunc));
          }
        });

    if (isTapeOn) {
      const tapeNode: TapeNode = {
        id: this.state.nextTapeNodeId++,
        name: scopeName,
        inputs,
        outputs: Array.isArray(result) ? result : [result] as Tensor[],
        saved
      };
      if (backwardsFunc != null) {
        tapeNode.gradient = (dy: T) => backwardsFunc(dy, saved);
      }
      this.state.activeTape.push(tapeNode);
    }

    if (this.state.profiling) {
      this.state.activeProfile.kernels.push({
        name: scopeName,
        bytesAdded: this.state.numBytes - startingBytecount,
        totalBytesSnapshot: this.state.numBytes,
        tensorsAdded: this.state.numTensors - startingNumTensors,
        totalTensorsSnapshot: this.state.numTensors,
        inputShapes: Object.keys(inputs).map(key => inputs[key].shape),
        outputShape: Array.isArray(result) ?
            (result as Tensor[]).map(item => item.shape) :
            (result as Tensor).shape
      });
    }

    return result;
  }

  // TensorManager implementation.

  registerTensor(a: Tensor|Variable, backend?: KernelBackend): void {
    const refCount = this.state.tensorInfo.has(a.dataId) ?
        this.state.tensorInfo.get(a.dataId).refCount :
        0;
    this.state.numTensors++;
    if (a.dtype === 'string') {
      this.state.numStringTensors++;
    }
    if (refCount === 0) {
      this.state.numDataBuffers++;

      // Bytes for complex numbers are counted by their components. Bytes for
      // string tensors are counted when writing values.
      let bytes = 0;
      if (a.dtype !== 'complex64' && a.dtype !== 'string') {
        bytes = a.size * util.bytesPerElement(a.dtype);
      }
      this.state.tensorInfo.set(a.dataId, {
        backend: backend != null ? backend : this.backend,
        dtype: a.dtype,
        shape: a.shape,
        bytes,
        refCount: 0
      });
      this.state.numBytes += bytes;
      if (backend != null) {
        backend.register(a.dataId, a.shape, a.dtype);
      } else {
        this.backend.register(a.dataId, a.shape, a.dtype);
      }
    }
    this.state.tensorInfo.get(a.dataId).refCount++;
    if (!(a instanceof Variable)) {
      this.track(a);
    }
  }

  registerVariable(v: Variable) {
    if (this.state.registeredVariables[v.name] != null) {
      throw new Error(`Variable with name ${v.name} was already registered`);
    }
    this.state.registeredVariables[v.name] = v;
  }

  disposeTensor(a: Tensor): void {
    if (!this.state.tensorInfo.has(a.dataId)) {
      return;
    }

    this.state.numTensors--;
    if (a.dtype === 'string') {
      this.state.numStringTensors--;
    }
    const info = this.state.tensorInfo.get(a.dataId);
    const refCount = info.refCount;
    if (refCount <= 1) {
      // Don't count bytes for complex numbers as they are counted by their
      // components.
      if (a.dtype !== 'complex64') {
        this.state.numBytes -= info.bytes;
      }
      this.state.numDataBuffers--;
      info.backend.disposeData(a.dataId);
      this.state.tensorInfo.delete(a.dataId);
    } else {
      this.state.tensorInfo.get(a.dataId).refCount--;
    }
    // TODO(nsthorat): Construct an error and save the stack trace for
    // debugging when in debug mode. Creating a stack trace is too expensive
    // to do unconditionally.
  }

  disposeVariables(): void {
    for (const varName in this.state.registeredVariables) {
      const v = this.state.registeredVariables[varName];
      this.disposeVariable(v);
    }
  }

  disposeVariable(v: Variable): void {
    this.disposeTensor(v);
    if (this.state.registeredVariables[v.name] != null) {
      delete this.state.registeredVariables[v.name];
    }
  }

  memory(): MemoryInfo {
    const info = this.backend.memory() as MemoryInfo;
    info.numTensors = this.state.numTensors;
    info.numDataBuffers = this.state.numDataBuffers;
    info.numBytes = this.state.numBytes;
    if (this.state.numStringTensors > 0) {
      info.unreliable = true;
      if (info.reasons == null) {
        info.reasons = [];
      }
      info.reasons.push(
          'Memory usage by string tensors is approximate ' +
          '(2 bytes per character)');
    }
    return info;
  }

  async profile(query: () => TensorContainer): Promise<ProfileInfo> {
    this.state.profiling = true;

    const startBytes = this.state.numBytes;
    const startNumTensors = this.state.numTensors;

    this.state.activeProfile.kernels = [];
    this.state.activeProfile.result = query();

    this.state.profiling = false;

    this.state.activeProfile.peakBytes = Math.max(
        ...this.state.activeProfile.kernels.map(d => d.totalBytesSnapshot));
    this.state.activeProfile.newBytes = this.state.numBytes - startBytes;
    this.state.activeProfile.newTensors =
        this.state.numTensors - startNumTensors;
    return this.state.activeProfile;
  }

  isTapeOn(): boolean {
    return this.state.gradientDepth > 0 && this.state.kernelDepth === 0;
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
      const resMap: NamedGradientMap = {};
      res.forEach((r, idx) => {
        resMap[idx] = () => r;
      });
      return resMap;
    };

    const tapeNode: TapeNode = {
      id: this.state.nextTapeNodeId++,
      name: this.state.activeScope.name,
      inputs: inputsMap,
      outputs: [result],
      gradient
    };
    this.state.activeTape.push(tapeNode);
  }

  keep<T extends Tensor>(result: T): T {
    result.kept = true;
    return result;
  }

  private startTape() {
    if (this.state.gradientDepth === 0) {
      this.state.activeTape = [];
    }
    this.state.gradientDepth++;
  }

  private endTape() {
    this.state.gradientDepth--;
  }

  /**
   * Start a scope. Use this with endScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  startScope(name?: string) {
    const scopeInfo: ScopeState = {
      track: [],
      name: 'unnamed scope',
      id: this.state.nextScopeId++
    };
    if (name) {
      scopeInfo.name = name;
    }
    this.state.scopeStack.push(scopeInfo);
    this.state.activeScope = scopeInfo;
  }

  /**
   * End a scope. Use this with startScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  endScope(result?: TensorContainer) {
    const tensorsToTrackInParent = getTensorsInContainer(result);
    const tensorsToTrackInParentSet =
        new Set(tensorsToTrackInParent.map(t => t.id));

    // Dispose the arrays tracked in this scope.
    for (let i = 0; i < this.state.activeScope.track.length; i++) {
      const tensor = this.state.activeScope.track[i];
      if (!tensor.kept && !tensorsToTrackInParentSet.has(tensor.id)) {
        tensor.dispose();
      }
    }

    const oldScope = this.state.scopeStack.pop();
    this.state.activeScope = this.state.scopeStack.length === 0 ?
        null :
        this.state.scopeStack[this.state.scopeStack.length - 1];

    // Track the current result in the parent scope.
    tensorsToTrackInParent.forEach(tensor => {
      // Only track the tensor if was allocated in the inner scope and is not
      // globally kept.
      if (!tensor.kept && tensor.scopeId === oldScope.id) {
        this.track(tensor);
      }
    });
  }

  /**
   * Returns gradients of `f` with respect to each of the `xs`. The gradients
   * returned are of the same length as `xs`, but some might be null if `f`
   * was not a function of that `x`. It also takes optional dy to multiply the
   * gradient, which defaults to `1`.
   */
  gradients<T extends Tensor>(
      f: () => T, xs: Tensor[], dy?: T,
      allowNoGradients = false): {value: T, grads: Tensor[]} {
    util.assert(
        xs.length > 0, () => 'gradients() received an empty list of xs.');
    if (dy != null && dy.dtype !== 'float32') {
      throw new Error(`dy must have 'float32' dtype, but has '${dy.dtype}'`);
    }

    const y = this.scopedRun(
        () => this.startTape(), () => this.endTape(),
        () => this.tidy('forward', f));

    util.assert(
        y instanceof Tensor,
        () => 'The result y returned by f() must be a tensor.');
    // Filter out the nodes that don't connect x => y.
    const filteredTape = getFilteredNodesXToY(this.state.activeTape, xs, y);
    if (!allowNoGradients && filteredTape.length === 0 && xs.length > 0) {
      throw new Error(
          'Cannot compute gradient of y=f(x) with respect to x. Make sure ' +
          'that the f you passed encloses all operations that lead from x ' +
          'to y.');
    }

    return this.tidy('backward', () => {
      const accumulatedGradientMap: {[tensorId: number]: Tensor} = {};
      accumulatedGradientMap[y.id] = (dy == null) ? ones(y.shape) : dy;

      // Backprop gradients through the filtered nodes.
      backpropagateGradients(
          accumulatedGradientMap, filteredTape,
          // Pass the tidy function to avoid circular dep with `tape.ts`.
          f => this.tidy(f as ScopeFn<Tensor>));
      const grads = xs.map(x => accumulatedGradientMap[x.id]);

      if (this.state.gradientDepth === 0) {
        // This means that we are not computing higher-order gradients
        // and can clean up the tape.
        this.state.activeTape.forEach(node => {
          for (const key in node.saved) {
            node.saved[key].dispose();
          }
        });
        this.state.activeTape = null;
      }
      return {value: y, grads};
    });
  }

  customGrad<T extends Tensor>(f: CustomGradientFunc<T>):
      (...args: Array<Tensor|GradSaveFunc>) => T {
    util.assert(
        util.isFunction(f),
        () => 'The f passed in customGrad(f) must be a function.');
    return (...inputs: Tensor[]): T => {
      util.assert(
          inputs.every(t => t instanceof Tensor),
          () => 'The args passed in customGrad(f)(x1, x2,...) must all be ' +
              'tensors');

      let res: {
        value: T,
        gradFunc: (dy: T, saved: Tensor[]) => Tensor | Tensor[],
      };
      const inputMap: NamedTensorMap = {};
      inputs.forEach((input, i) => {
        inputMap[i] = input;
      });
      return this.runKernel(
          (_, save) => {
            res = f(...[...inputs, save]);
            util.assert(
                res.value instanceof Tensor,
                () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.value` is a tensor');
            util.assert(
                util.isFunction(res.gradFunc),
                () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.gradFunc` is a function.');
            return res.value;
          },
          inputMap,
          (dy: T, saved: Tensor[]) => {
            const gradRes = res.gradFunc(dy, saved);
            const grads: Tensor[] =
                Array.isArray(gradRes) ? gradRes : [gradRes];
            util.assert(
                grads.length === inputs.length,
                () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.gradFunc` is a function that returns ' +
                    'the same number of tensors as inputs passed to f(...).');
            util.assert(
                grads.every(t => t instanceof Tensor),
                () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.gradFunc` is a function that returns ' +
                    'a list of only tensors.');
            const gradMap: {[key: string]: () => Tensor} = {};
            grads.forEach((grad, i) => {
              gradMap[i] = () => grad;
            });
            return gradMap;
          });
    };
  }

  // Forwarding to backend.
  write(destBackend: KernelBackend, dataId: DataId, values: BackendValues):
      void {
    const info = this.state.tensorInfo.get(dataId);

    const srcBackend = info.backend;
    destBackend = destBackend || this.backend;

    // Bytes for string tensors are counted when writing.
    if (info.dtype === 'string') {
      const newBytes = bytesFromStringArray(values as Uint8Array[]);
      this.state.numBytes += newBytes - info.bytes;
      info.bytes = newBytes;
    }

    if (destBackend !== srcBackend) {
      // Delete the tensor from the old backend and move it to the new
      // backend.
      srcBackend.disposeData(dataId);
      info.backend = destBackend;
      destBackend.register(dataId, info.shape, info.dtype);
    }
    destBackend.write(dataId, values);
  }
  readSync(dataId: DataId): BackendValues {
    // Route the read to the correct backend.
    const info = this.state.tensorInfo.get(dataId);
    return info.backend.readSync(dataId);
  }
  read(dataId: DataId): Promise<BackendValues> {
    // Route the read to the correct backend.
    const info = this.state.tensorInfo.get(dataId);
    return info.backend.read(dataId);
  }
  fromPixels(
      pixels: PixelData|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement,
      numChannels: number): Tensor3D {
    return this.backend.fromPixels(pixels, numChannels);
  }
  async time(query: () => void): Promise<TimingInfo> {
    const start = now();
    const timingInfo = await this.backend.time(query) as TimingInfo;
    timingInfo.wallMs = now() - start;
    return timingInfo;
  }

  /**
   * Tracks a Tensor in the current scope to be automatically cleaned up
   * when the current scope ends, and returns the value.
   *
   * @param result The Tensor to track in the current scope.
   */
  private track<T extends Tensor>(result: T): T {
    if (this.state.activeScope != null) {
      result.scopeId = this.state.activeScope.id;
      this.state.activeScope.track.push(result);
    }

    return result;
  }

  get registeredVariables(): NamedVariableMap {
    return this.state.registeredVariables;
  }

  /**
   * Resets the engine state. Removes all backends but does not remove
   * registered backend factories.
   */
  reset(): void {
    // Make any pending promise obsolete.
    this.pendingBackendInitId++;

    this.state.dispose();
    this.ENV.reset();
    this.state = new EngineState();

    for (const backendName in this.registry) {
      this.registry[backendName].dispose();
      delete this.registry[backendName];
    }
    this.backendName = null;
    this.backendInstance = null;
    this.pendingBackendInit = null;
  }
}

function ones(shape: number[]): Tensor {
  const values = makeOnesTypedArray(sizeFromShape(shape), 'float32');
  return Tensor.make(shape, {values});
}

let GLOBAL: {_tfengine: Engine};
function getGlobalNamespace(): {_tfengine: Engine} {
  if (GLOBAL == null) {
    // tslint:disable-next-line:no-any
    let ns: any;
    if (typeof (window) !== 'undefined') {
      ns = window;
    } else if (typeof (global) !== 'undefined') {
      ns = global;
    } else if (typeof (process) !== 'undefined') {
      ns = process;
    } else if (typeof (self) !== 'undefined') {
      ns = self;
    } else {
      throw new Error('Could not find a global object');
    }
    GLOBAL = ns;
  }
  return GLOBAL;
}

function getOrMakeEngine(): Engine {
  const ns = getGlobalNamespace();
  if (ns._tfengine == null) {
    const environment = new Environment(ns);
    ns._tfengine = new Engine(environment);
  }
  setEnvironmentGlobal(ns._tfengine.ENV);

  // Tell the current tensor interface that the global engine is responsible
  // for tracking.
  setTensorTracker(() => ns._tfengine);
  return ns._tfengine;
}

export let ENGINE = getOrMakeEngine();
