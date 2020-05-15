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
import {webgpu} from '@tensorflow/tfjs-backend-webgpu';
import * as tfc from '@tensorflow/tfjs-core';
import * as tfl from '@tensorflow/tfjs-layers';

console.log(webgpu.webgpu_util.tilesFitEvenlyIntoShape);

import {BenchmarkModelTest} from './types';
import * as util from './util';

const MOBILENET_MODEL_PATH =
    // tslint:disable-next-line:max-line-length
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json';

export class MobileNetV1GPUBenchmark implements BenchmarkModelTest {
  private model: tfl.LayersModel;

  async loadModel() {
    this.model = await tfl.loadLayersModel(MOBILENET_MODEL_PATH);
  }

  async run(size: number): Promise<number> {
    // tfc.setBackend('webgl');
    // console.log(webgpu);
    await tfc.ready();
    // tfc.setBackend('webgpu');

    const zeros = tfc.zeros([1, 224, 224, 3]);

    const benchmark = () => this.model.predict(zeros);

    const time = await util.benchmark(benchmark);

    zeros.dispose();

    return time;
  }
}
