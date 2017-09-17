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

// tslint:disable-next-line:max-line-length
import {Graph, NDArrayMath, NDArrayMathGPU, Scalar, Session, Tensor} from '../deeplearn';

class Adder {
  inputTensorA: Tensor;
  inputTensorB: Tensor;
  sum: Tensor;
  session: Session;
  math: NDArrayMath = new NDArrayMathGPU();
  setupSession(): void {
    const graph = new Graph();

    this.inputTensorA = graph.placeholder('A', []);
    this.inputTensorB = graph.placeholder('B', []);
    this.sum = graph.add(this.inputTensorA, this.inputTensorB);
    this.session = new Session(graph, this.math);
  }

  computeSum(a: number, b: number): number {
    const feeds = [
      {tensor: this.inputTensorA, data: Scalar.new(a)},
      {tensor: this.inputTensorB, data: Scalar.new(b)}
    ];
    let result;
    this.math.scope(() => {
      result = this.session.eval(this.sum, feeds).get();
    });
    return result;
  }
}


const adder = new Adder();
adder.setupSession();
const result = adder.computeSum(1, 1);

const outputEl = document.getElementById('output');
if (!outputEl) throw new Error('output element not found');
outputEl.innerText = String(result);
