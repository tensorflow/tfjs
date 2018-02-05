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

import {NamedTensorMap} from '../../math/types';
import {Tensor} from '../tensor';
import {Rank} from '../types';
import {KernelConfigRegistry} from './kernel_registry';

export type Tape = Array<TapeNode<TapeNodeOutput>>;
export type TapeNodeOutput = Tensor|NamedTensorMap;
export type TapeNodeType = 'kernel'|'customGradient';

export interface TapeNode<T extends TapeNodeOutput> {
  id: number;
  type: TapeNodeType;
  name: string;
  inputAndArgs: TapeNodeInputConfig;

  output: T;
  gradient: (dy: Tensor|NamedTensorMap, y: T) => TapeNodeInputGradientTensors;
}

export interface TapeNodeInputConfig { inputs: NamedTensorMap; }

export type TapeNodeInputGradientTensors = {
  [inputName: string]: () => Tensor;
};

// Kernel nodes
export interface KernelNode extends TapeNode<Tensor> {
  kernel: keyof KernelConfigRegistry<Rank>;
  inputAndArgs: KernelInputConfig;
  output: Tensor;
}

export interface KernelInputConfig extends TapeNodeInputConfig {
  inputs: NamedTensorMap;
  // tslint:disable-next-line:no-any
  args?: {[argName: string]: any};
}
