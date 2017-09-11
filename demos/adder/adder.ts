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

import {Graph, NDArrayMathGPU, Scalar, Session, Tensor} from '../deeplearnjs';

class Adder {
  inputTensorA: Tensor;
  inputTensorB: Tensor;
  sum: Tensor;
  session: Session;
  math = new NDArrayMathGPU();
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

const outputEl = document.getElementById('output');
if (!outputEl) throw new Error('output element not found');
function printOutput(out: number) {
  outputEl.innerText = String(out);
}

const inA: HTMLInputElement = document.getElementById('A') as HTMLInputElement;
if (!inA) throw new Error('input A not found');
const inB: HTMLInputElement = document.getElementById('B') as HTMLInputElement;
if (!inB) throw new Error('output B not found');

export function execute(event?: Event) {
  const a = +inA.value;
  const b = +inB.value;

  printOutput(adder.computeSum(a, b));
}

inA.addEventListener('keyup', execute);
inB.addEventListener('keyup', execute);

execute();
