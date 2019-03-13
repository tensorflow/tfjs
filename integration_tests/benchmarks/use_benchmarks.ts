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
import * as tf from '@tensorflow/tfjs';

import {BenchmarkTest} from './types';
import * as util from './util';

const USE_MODEL_PATH =
    // tslint:disable-next-line:max-line-length
    'https://storage.googleapis.com/tfjs-models/savedmodel/universal_sentence_encoder/model.json';

export class UniversalSentenceEncoderBenchmark implements BenchmarkTest {
  async run(size: number): Promise<number> {
    tf.setBackend('webgl');

    const model = await tf.loadGraphModel(USE_MODEL_PATH);
    const indices = tf.tensor2d(
        [
          0,  0, 0,  1, 0, 2, 0, 3, 0, 4, 1, 0, 1, 1, 1, 2, 1, 3, 1, 4, 1,
          5,  2, 0,  2, 1, 2, 2, 2, 3, 2, 4, 3, 0, 3, 1, 3, 2, 3, 3, 3, 4,
          4,  0, 4,  1, 4, 2, 4, 3, 4, 4, 4, 5, 4, 6, 4, 7, 4, 8, 4, 9, 4,
          10, 4, 11, 5, 0, 5, 1, 5, 2, 5, 3, 5, 4, 5, 5, 5, 6, 5, 7
        ],
        [41, 2], 'int32');
    const values = tf.tensor1d(
        [
          16,   60,  69,   825, 6,    819, 2704, 2901, 903, 318, 6,
          728,  446, 31,   19,  54,   379, 18,   37,   735, 54,  829,
          5459, 11,  221,  8,   373,  7,   9,    969,  7,   468, 6,
          184,  621, 7582, 949, 1803, 18,  1977, 6
        ],
        'int32');

    const benchmark = async () =>
        model.executeAsync({indices, values}) as Promise<tf.Tensor>;

    const time = await util.warmupAndAsyncBenchmarkGPU(benchmark);

    indices.dispose();
    values.dispose();

    return time;
  }
}
