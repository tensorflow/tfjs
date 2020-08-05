/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

import * as tfc from '@tensorflow/tfjs-core';

import {SMOKE} from './constants';

/**
 *  This file tests backend switching scenario.
 */
describe(`${SMOKE} backends`, () => {
  describe('switch', () => {
    beforeAll(() => {
      tfc.env().set('WEBGL_CPU_FORWARD', false);
    });

    it(`from webgl to cpu.`, async () => {
      // A backend with no refCounter.
      await tfc.setBackend('webgl');

      // numDataIds = 0
      const before = tfc.engine().backend.numDataIds();

      // input tensorData is stored in webgl backend.
      const input = tfc.tensor2d([1, 1, 1, 1], [2, 2], 'float32');

      // input tensorInfo refCount = 2;
      const inputReshaped = tfc.reshape(input, [2, 2]);

      await tfc.setBackend('cpu');

      // inputReshaped tensorData is reshaped, during which it is moved to cpu
      // backend with a refCount 1. After reshape, it has a refCount 2. The
      // tensorInfo refCount becomes 3 because inputReshaped2 is an output.
      const inputReshaped2 = tfc.reshape(inputReshaped, [2, 2]);

      input.dispose();
      inputReshaped.dispose();
      inputReshaped2.dispose();

      const after = tfc.engine().backend.numDataIds();

      expect(after).toBe(before);

      // There should be no more data in webgl, because they are all moved to
      // cpu even if the tensorInfo refCount before moving is 2.
      await tfc.setBackend('webgl');
      const webglAfter = tfc.engine().backend.numDataIds();
      expect(webglAfter).toBe(0);
    });

    it(`from cpu to webgl.`, async () => {
      await tfc.setBackend('cpu');

      // numDataIds = 0.
      const before = tfc.engine().backend.numDataIds();

      // input tensorData is stored in cpu backend.
      const input = tfc.tensor2d([1, 1, 1, 1], [2, 2], 'float32');

      // input tensorData refCount = 2, tensorInfo refCount = 1.
      const inputReshaped = tfc.reshape(input, [2, 2]);

      await tfc.setBackend('webgl');

      // inputReshaped tensorData is reshaped, during which it is moved to webgl
      // backend with a refCount 2. Then tensorInfo refCount is also 2. After
      // reshape, the tensorInfo refCount becomes 3.
      const inputReshaped2 = tfc.reshape(inputReshaped, [2, 2]);

      input.dispose();
      inputReshaped.dispose();
      inputReshaped2.dispose();

      const after = tfc.engine().backend.numDataIds();

      expect(after).toBe(before);

      // There should be no more data in cpu, because they are all moved to
      // webgl.
      await tfc.setBackend('cpu');
      const cpuAfter = tfc.engine().backend.numDataIds();
      expect(cpuAfter).toBe(0);
    });
  });
});
