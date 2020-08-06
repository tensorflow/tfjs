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

      const webglBefore = tfc.engine().backend.numDataIds();

      // input is stored in webgl backend. tensorInfo refCount = 1.
      const input = tfc.tensor2d([1, 1, 1, 1], [2, 2], 'float32');

      // input tensorInfo refCount = 2;
      const inputReshaped = tfc.reshape(input, [2, 2]);

      const webglAfter = tfc.engine().backend.numDataIds();

      expect(webglAfter).toEqual(webglBefore + 1);

      await tfc.setBackend('cpu');

      const cpuBefore = tfc.engine().backend.numDataIds();

      // input moved to cpu. tensorData refCount = 1.
      const inputReshaped2 = tfc.reshape(inputReshaped, [2, 2]);

      const cpuAfter = tfc.engine().backend.numDataIds();

      // input tensorData refCount = 3, reshape increase by 1, then output
      // tensor increase by 1.
      expect(cpuAfter).toEqual(cpuBefore + 1);

      await tfc.setBackend('webgl');

      // Because input is moved to cpu, data should be deleted from webgl.
      expect(tfc.engine().backend.numDataIds()).toEqual(webglAfter - 1);

      await tfc.setBackend('cpu');

      // After dispose, tensorData refCount = 2.
      input.dispose();

      // Input is not deleted, because refCount is 2.
      expect(tfc.engine().backend.numDataIds()).toEqual(cpuAfter);

      // After dispose, tensorData refCount = 1.
      inputReshaped.dispose();

      // Input is not deleted, because refCount is 1.
      expect(tfc.engine().backend.numDataIds()).toEqual(cpuAfter);

      // After dispose, tensorData refCount = 0, data deleted.
      inputReshaped2.dispose();

      const after = tfc.engine().backend.numDataIds();

      expect(after).toBe(cpuBefore);
    });

    it(`from cpu to webgl.`, async () => {
      await tfc.setBackend('cpu');

      const cpuBefore = tfc.engine().backend.numDataIds();

      // input is stored in cpu backend. tensorData refCount = 1, tensorInfo
      // refCount = 1.
      const input = tfc.tensor2d([1, 1, 1, 1], [2, 2], 'float32');

      // input tensorData refCount = 3, reshape increase by 1, then output
      // tensor increase by 1. tensorInfo refCount = 1.
      const inputReshaped = tfc.reshape(input, [2, 2]);

      const cpuAfter = tfc.engine().backend.numDataIds();

      expect(cpuAfter).toEqual(cpuBefore + 1);

      await tfc.setBackend('webgl');

      const webglBefore = tfc.engine().backend.numDataIds();

      // input moved to webgl. tensorInfo refCount = 3.
      const inputReshaped2 = tfc.reshape(inputReshaped, [2, 2]);

      const webglAfter = tfc.engine().backend.numDataIds();

      expect(webglAfter).toEqual(webglBefore + 1);

      await tfc.setBackend('cpu');

      // Because input is moved to webgl, data should be deleted from cpu.
      expect(tfc.engine().backend.numDataIds()).toEqual(cpuAfter - 1);

      await tfc.setBackend('webgl');

      // After dispose, tensorInfo = 2.
      input.dispose();

      // Data is not deleted, because tensorInfo is 2.
      expect(tfc.engine().backend.numDataIds()).toEqual(webglAfter);

      // After dipose, tensorInfo = 1.
      inputReshaped.dispose();

      // Data is not deleted, because tensorInfo is 1.
      expect(tfc.engine().backend.numDataIds()).toEqual(webglAfter);

      // After dipose, tensorInfo = 1, data deleted.
      inputReshaped2.dispose();

      const after = tfc.engine().backend.numDataIds();

      expect(after).toBe(webglBefore);
    });
  });
});
