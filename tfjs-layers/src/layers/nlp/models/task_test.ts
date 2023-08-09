/**
 * @license
 * Copyright 2023 Google LLC.
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

/**
 * Unit Tests for Task Layers.
 */
import { input } from '../../../exports';
// import { Preprocessor } from './preprocessor';
import { Task } from './task';
import { SymbolicTensor } from '../../../engine/topology';
import { dense } from '../../../exports_layers';

describe('Task', () => {
  it('serialization round-trip with no set tokenizer', () => {
    const inputs = input({shape: [10]});
    const outputs = dense({units: 1}).apply(inputs) as SymbolicTensor;
    const task = new Task({inputs, outputs});
    const reserialized = Task.fromConfig(
      Task, task.getConfig());
    expect(reserialized.getConfig()).toEqual(task.getConfig());
  });
});
