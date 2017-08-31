/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import {Node, VariableNode} from './graph';
import {NDArrayMath} from './math/math';
import {SessionRuntime} from './session';
import {TensorArrayMap, SummedTensorArrayMap} from './tensor_array_map';

export abstract class Optimizer {
  protected variableNodes: VariableNode[];
  protected specifiedVariableNodes: VariableNode[]|null;

  constructor(specifiedVariableList?: Node[]) {
    if (specifiedVariableList != null) {
      this.specifiedVariableNodes = specifiedVariableList as VariableNode[];
    }
  }

  abstract beforeBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap): void;

  abstract afterExample(
      math: NDArrayMath, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap): void;

  abstract afterBatch(
      math: NDArrayMath, batchSize: number, runtime: SessionRuntime,
      activationArrayMap: TensorArrayMap,
      gradientArrayMap: SummedTensorArrayMap): void;

  abstract dispose(): void;
}
