/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import * as tfconverter from '@tensorflow/tfjs-converter';
import * as tfcore from '@tensorflow/tfjs-core';

import {BenchmarkModelTest} from './types';
import * as util from './util';

const COCOSSD_MODEL_PATH =
    // tslint:disable-next-line:max-line-length
    'https://storage.googleapis.com/tfjs-models/savedmodel/coco-ssd-mobilenet_v1/model.json';

export class CoCoSSDBenchmark implements BenchmarkModelTest {
  private model: tfconverter.GraphModel;

  async loadModel() {
    this.model = await tfconverter.loadGraphModel(COCOSSD_MODEL_PATH);
  }

  async run(size: number): Promise<number> {
    tfcore.setBackend('webgl');

    const zeros = tfcore.zeros([1, 224, 224, 3]);

    const benchmark = async () =>
        this.model.executeAsync(zeros) as Promise<tfcore.Tensor[]>;

    const time = await util.asyncBenchmark(benchmark);

    zeros.dispose();

    return time;
  }
}
