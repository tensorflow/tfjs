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
import * as tf from '@tensorflow/tfjs';

import {BenchmarkTest} from './types';
import * as util from './util';

const MOBILENET_MODEL_PATH =
    // tslint:disable-next-line:max-line-length
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json';

export class MobileNetV1GPUBenchmark implements BenchmarkTest {
  async run(size: number): Promise<number> {
    tf.setBackend('webgl');

    const model = await tf.loadModel(MOBILENET_MODEL_PATH);
    const zeros = tf.zeros([1, 224, 224, 3]);

    const benchmark = () => model.predict(zeros) as tf.Tensor;

    const time = await util.warmupAndBenchmarkGPU(benchmark);

    zeros.dispose();

    return time;
  }
}
