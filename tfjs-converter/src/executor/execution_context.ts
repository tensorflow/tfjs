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
import {Tensor} from '@tensorflow/tfjs-core';

import {NamedTensorsMap, TensorArrayMap, TensorListMap} from '../data/types';

import {TensorArray} from './tensor_array';
import {TensorList} from './tensor_list';
import {FunctionExecutor} from './types';

export interface ExecutionContextInfo {
  id: number;           // the unique id of the context info
  frameName: string;    // The frame name of the loop, this comes from
                        // the TensorFlow NodeDef.
  iterationId: number;  // The iteration id of the loop
}

/**
 * ExecutionContext captures the runtime environment of the node. It keeps
 * track of the current frame and iteration for the control flow ops.
 *
 * For example, typical Dynamic RNN model may contain loops, for which
 * TensorFlow will generate graphs with Enter/Exit nodes to control the
 * current execution frame, and NextIteration Nodes for iteration id increment.
 * For model with branch logic, TensorFLow will generate Switch/Merge ops.
 */
export class ExecutionContext {
  private rootContext = {id: 0, frameName: '', iterationId: 0};
  private contexts: ExecutionContextInfo[] = [this.rootContext];
  private lastId = 0;
  private _currentContextIds: string[];

  constructor(
      readonly weightMap: NamedTensorsMap = {},
      readonly tensorArrayMap: TensorArrayMap = {},
      readonly tensorListMap: TensorListMap = {},
      readonly functionMap: {[key: string]: FunctionExecutor} = {}) {
    this.generateCurrentContextIds();
  }

  private newFrame(id: number, frameName: string) {
    return {id, frameName, iterationId: 0};
  }

  /**
   * Set the current context
   * @param contexts: ExecutionContextInfo[] the current path of execution
   * frames
   */
  set currentContext(contexts: ExecutionContextInfo[]) {
    if (this.contexts !== contexts) {
      this.contexts = contexts;
      this.generateCurrentContextIds();
    }
  }

  get currentContext(): ExecutionContextInfo[] {
    return this.contexts;
  }

  /**
   * Returns the current context in string format.
   */
  get currentContextId(): string {
    return this._currentContextIds[0];
  }

  /**
   * Returns the current context and all parent contexts in string format.
   * This allow access to the nodes in the current and parent frames.
   */
  get currentContextIds(): string[] {
    return this._currentContextIds;
  }

  private generateCurrentContextIds() {
    const names = [];
    for (let i = 0; i < this.contexts.length - 1; i++) {
      const contexts = this.contexts.slice(0, this.contexts.length - i);
      names.push(this.contextIdforContexts(contexts));
    }
    names.push('');
    this._currentContextIds = names;
  }

  private contextIdforContexts(contexts: ExecutionContextInfo[]) {
    return contexts ?
        contexts
            .map(
                context => (context.id === 0 && context.iterationId === 0) ?
                    '' :
                    `${context.frameName}-${context.iterationId}`)
            .join('/') :
        '';
  }

  /**
   * Enter a new frame, a new context is pushed on the current context list.
   * @param frameId new frame id
   */
  enterFrame(frameId: string) {
    if (this.contexts) {
      this.lastId++;
      this.contexts = this.contexts.slice();
      this.contexts.push(this.newFrame(this.lastId, frameId));
      this._currentContextIds.unshift(this.contextIdforContexts(this.contexts));
    }
  }

  /**
   * Exit the current frame, the last context is removed from the current
   * context list.
   */
  exitFrame() {
    if (this.contexts && this.contexts.length > 1) {
      this.contexts = this.contexts.slice();
      this.contexts.splice(-1);
      this.currentContextIds.shift();
    } else {
      throw new Error('Cannot exit frame, the context is empty');
    }
  }

  /**
   * Enter the next iteration of a loop, the iteration id of last context is
   * increased.
   */
  nextIteration() {
    if (this.contexts && this.contexts.length > 0) {
      this.contexts = this.contexts.slice();
      this.lastId++;
      const context =
          Object.assign({}, this.contexts[this.contexts.length - 1]);
      context.iterationId += 1;
      context.id = this.lastId;
      this.contexts.splice(-1, 1, context);
      this._currentContextIds.splice(
          0, 1, this.contextIdforContexts(this.contexts));
    } else {
      throw new Error('Cannot increase frame iteration, the context is empty');
    }
  }

  getWeight(name: string): Tensor[] {
    return this.weightMap[name];
  }

  addTensorArray(tensorArray: TensorArray) {
    this.tensorArrayMap[tensorArray.id] = tensorArray;
  }

  getTensorArray(id: number): TensorArray {
    return this.tensorArrayMap[id];
  }

  addTensorList(tensorList: TensorList) {
    this.tensorListMap[tensorList.id] = tensorList;
  }

  getTensorList(id: number): TensorList {
    return this.tensorListMap[id];
  }

  dispose(keepIds: Set<number>) {
    for (const key in this.tensorArrayMap) {
      this.tensorArrayMap[key].clearAndClose(keepIds);
    }

    for (const key in this.tensorListMap) {
      this.tensorListMap[key].clearAndClose(keepIds);
    }
  }
}
