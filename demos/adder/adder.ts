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

import {ENV, Scalar} from 'deeplearn';

const outputElement = document.getElementById('output');
const inA: HTMLInputElement = document.getElementById('A') as HTMLInputElement;
const inB: HTMLInputElement = document.getElementById('B') as HTMLInputElement;

const math = ENV.math;

export async function execute(event?: Event) {
  const a = Scalar.new(+inA.value);
  const b = Scalar.new(+inB.value);
  const result = await math.add(a, b).data();
  outputElement.innerText = result.toString();
}

inA.addEventListener('keyup', execute);
inB.addEventListener('keyup', execute);

execute();
