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

import * as util from '../../util';
import {NamedArrayMap} from '../../util';
import {DataType, NDArray, Rank, Scalar} from '../ndarray';
import {MathBackend} from './backend';
import * as kernel_registry from './kernel_registry';
import {KernelConfigRegistry} from './kernel_registry';
// tslint:disable-next-line:max-line-length
import {KernelNode, Tape, TapeNode, TapeNodeInputGradientArrays} from './tape_types';
import * as tape_util from './tape_util';
import {ScopeResult, ScopeResultImmediate} from './tape_util';

interface ScopeState {
  keep: NDArray[];
  track: NDArray[];
}

export class BackendEngine {
  private nextTapeNodeId = 0;

  private activeTape: Tape;
  private gradientScopeCount = 0;

  private customGradientDepth = 0;

  // Keep NDArrays that parallel the tapes.
  private activeScope: ScopeState;
  private scopeStack: ScopeState[];

  private debugMode = false;

  constructor(private backend: MathBackend, private safeMode: boolean) {
    // Create a default outer scope.
    this.activeScope = {keep: [], track: []};
    this.scopeStack = [this.activeScope];
  }

  enableDebugMode() {
    this.debugMode = true;
  }

  executeKernel<K extends keyof KernelConfigRegistry,
                          C extends KernelConfigRegistry[K]['inputAndArgs']>(
      kernelName: K, config: C, grad?: KernelConfigRegistry[K]['gradient']):
      KernelConfigRegistry[K]['output'] {
    const kernelFn = () =>
        kernel_registry.executeKernel(this.backend, kernelName, config);

    let start: number;
    if (this.debugMode) {
      start = performance.now();
    }
    const result = kernelFn();
    if (this.debugMode) {
      const vals = result.dataSync();
      const time = util.rightPad(`${performance.now() - start}ms`, 9);
      const paddedName = util.rightPad(kernelName, 25);
      const rank = result.rank;
      const size = result.size;
      const shape = util.rightPad(result.shape.toString(), 14);
      console.log(
          `%c${paddedName}\t%c${time}\t%c${rank}D ${shape}\t%c${size}`,
          'font-weight:bold', 'color:red', 'color:blue', 'color: orange');
      util.checkForNaN(vals, result.dtype, name);
    }

    if (this.activeTape != null && this.customGradientDepth === 0) {
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

  customGradient<D extends DataType, R extends Rank>(
      f: () => {
        value: NDArray<D, R>,
        gradients: (dy: NDArray<'float32', R>, y: NDArray<D, R>) =>
            TapeNodeInputGradientArrays
      },
      inputs: NamedArrayMap, name: string): NDArray<D, R> {
    this.customGradientDepth++;

    let gradientsFunc: (dy: NDArray<'float32', R>, y: NDArray<D, R>) =>
        TapeNodeInputGradientArrays;
    const gradientsMode = true;
    const result = this.scope('customGradient', () => {
      const {value, gradients} = f();
      gradientsFunc = gradients;
      return value;
    }, gradientsMode);

    this.customGradientDepth--;

    if (this.activeTape != null && this.customGradientDepth === 0) {
      const evaluatedNode: TapeNode<NDArray<D, R>> = {
        id: this.nextTapeNodeId++,
        type: 'customGradient',
        name,
        inputAndArgs: {inputs},
        output: result,
        gradient: gradientsFunc
      };

      this.activeTape.push(evaluatedNode);
    }

    return result;
  }

  gradients(f: () => Scalar, xs: NDArray[], returnValue: boolean): NDArray[]|
      {value: Scalar, gradients: NDArray[]} {
    const gradientsMode = true;
    const result = this.scope('gradients', () => {
      const y = f();
      if (y.rank !== 0) {
        throw new Error(
            `Cannot compute gradient of non-scalar y output. ` +
            `Got y with rank ${y.rank}`);
      }
      const gradients = this.gradientWrt(y, xs);
      if (returnValue) {
        return [y, ...gradients];
      } else {
        return gradients;
      }
    }, gradientsMode);

    if (returnValue) {
      return {value: result[0] as Scalar, gradients: result.slice(1)};
    } else {
      return result;
    }
  }

  vjp<T extends NDArray>(f: () => T, xs: NDArray[], dy: T): NDArray[] {
    const gradientsMode = true;
    return this.scope('vjp', () => {
      const y = f();
      if (!util.arraysEqual(y.shape, dy.shape)) {
        throw new Error(
            `Cannot compute vector jacobian product, ` +
            `y shape (${y.shape}) does not match dy shape (${dy.shape}).`);
      }
      return this.gradientWrt(y, xs, dy);
    }, gradientsMode);
  }

  private gradientWrt<T extends NDArray>(y: T, xs: NDArray[], dy?: T):
      NDArray[] {
    // Filter out the nodes that don't connect x => y.
    const filteredTape = tape_util.getFilteredNodesXToY(this.activeTape, xs, y);
    if (filteredTape.length === 0) {
      throw new Error(
          `Cannot compute gradient: y is not a function of xs.` +
          `Make sure the xs you are computing gradients with respect ` +
          `to are used inside the gradient function.`);
    }

    const arrayAccumulatedGradientMap: {[ndarrayId: number]: NDArray} = {};
    arrayAccumulatedGradientMap[y.id] = dy == null ? Scalar.new(1) : dy;

    // Backprop gradients through the filtered nodes.
    tape_util.backpropagateGradients(arrayAccumulatedGradientMap, filteredTape);

    const gradients = xs.map(x => arrayAccumulatedGradientMap[x.id]);
    gradients.forEach((grad, i) => {
      if (grad == null) {
        throw new Error(`Gradient error: y was not a function of xs[${i}]`);
      }
    });
    return gradients;
  }

  /**
   * Create a new math scope. Put chained math operations inside a scope
   * function closure so that the library automatically cleans up NDArrays
   * from intermediate math operations. You must create a scope in safe mode
   * to call math operations. If a result is returned from the scope, it will
   * also be tracked, which means there must be yet another wrapping scope.
   * @param name The name of the scope. Used for logging.
   * @param scopeFn The function to execute with chained math operations.
   */
  scope<T extends ScopeResult>(
      name: string,
      scopeFn:
          (keep:
               <D1 extends DataType, T1 extends NDArray<D1>>(ndarray: T1) => T1,
           track: <D2 extends DataType, T2 extends NDArray<D2>>(ndarray: T2) =>
               T2) => T,
      gradientsMode: boolean): T {
    this.startScope(gradientsMode);

    const keepFn = <T extends NDArray>(ndarray: T): T => this.keep(ndarray);
    // TODO(smilkov): trackFn is a no-op since we have global tracking.
    // Remove when we break backward compatibility.
    const trackFn = <T extends NDArray>(ndarray: T): T => ndarray;
    const result = scopeFn(keepFn, trackFn);

    if (result instanceof Promise) {
      result.then(r => this.endScope(r, gradientsMode));
      return result;
    } else {
      this.endScope(result as ScopeResultImmediate, gradientsMode);
      return result;
    }
  }

  /**
   * Start a scope. Use this with endScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  startScope(gradientsMode: boolean) {
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
  endScope(result: ScopeResultImmediate, gradientsMode: boolean) {
    if (gradientsMode) {
      this.gradientScopeCount--;
      if (this.gradientScopeCount === 0) {
        this.activeTape = null;
      }
    }

    let arraysToKeep = this.activeScope.keep;
    const arraysToTrackInParent =
        tape_util.extractNDArraysFromScopeResult(result);
    arraysToKeep = arraysToKeep.concat(arraysToTrackInParent);

    // Dispose the arrays tracked in this scope.
    for (let i = 0; i < this.activeScope.track.length; i++) {
      const ndarray = this.activeScope.track[i];
      if (util.isNDArrayInList(ndarray, arraysToKeep)) {
        continue;
      }

      if (this.activeTape != null) {
        arraysToTrackInParent.push(ndarray);
      } else {
        ndarray.dispose();
      }
    }

    this.scopeStack.pop();
    this.activeScope = this.scopeStack.length === 0 ?
        null :
        this.scopeStack[this.scopeStack.length - 1];

    // Track the current result in the parent scope.
    arraysToTrackInParent.forEach(ndarray => {
      if (!util.isNDArrayInList(ndarray, this.activeScope.keep)) {
        this.track(ndarray);
      }
    });
  }

  /**
   * Keeps an NDArray in the current scope from being disposed automatically.
   * @param result The NDArray to keep from being disposed.
   */
  keep<T extends NDArray>(result: T): T {
    if (this.scopeStack.length === 1) {
      if (this.safeMode) {
        throw new Error(
            'You are using math in safe mode. Enclose all ' +
            'math.method() calls inside a scope: ' +
            'math.scope(() => {math.method();...}) to avoid memory ' +
            'leaks.');
      }
    }
    this.activeScope.keep.push(result);
    return result;
  }

  /**
   * Tracks an NDArray in the current scope to be automatically cleaned up
   * when the current scope ends, and returns the value.
   *
   * @param result The NDArray to track in the current scope.
   */
  track<D extends DataType, T extends NDArray<D>>(result: T): T {
    if (this.scopeStack.length === 1) {
      if (this.safeMode) {
        throw new Error(
            'You are using math in safe mode. Enclose all ' +
            'math.method() calls inside a scope: ' +
            'math.scope(() => {math.method();...}) to avoid memory ' +
            'leaks.');
      }
    }
    this.activeScope.track.push(result);
    return result;
  }

  getBackend(): MathBackend {
    return this.backend;
  }
}
