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
      await tfc.setBackend('webgl');

      const webglBefore = tfc.engine().backend.numDataIds();

      const input = tfc.tensor2d([1, 1, 1, 1], [2, 2], 'float32');
      // input is stored in webgl backend.

      const inputReshaped = tfc.reshape(input, [2, 2]);

      const webglAfter = tfc.engine().backend.numDataIds();

      expect(webglAfter).toEqual(webglBefore + 1);

      await tfc.setBackend('cpu');

      const cpuBefore = tfc.engine().backend.numDataIds();

      const inputReshaped2 = tfc.reshape(inputReshaped, [2, 2]);
      // input moved to cpu.

      // Because input is moved to cpu, data should be deleted from webgl.
      expect(tfc.findBackend('webgl').numDataIds()).toEqual(webglAfter - 1);

      const cpuAfter = tfc.engine().backend.numDataIds();

      expect(cpuAfter).toEqual(cpuBefore + 1);

      input.dispose();
      expect(tfc.engine().backend.numDataIds()).toEqual(cpuAfter);

      inputReshaped.dispose();

      expect(tfc.engine().backend.numDataIds()).toEqual(cpuAfter);

      inputReshaped2.dispose();

      const after = tfc.engine().backend.numDataIds();

      expect(after).toBe(cpuBefore);
    });

    it(`from cpu to webgl.`, async () => {
      await tfc.setBackend('cpu');

      const cpuBefore = tfc.engine().backend.numDataIds();

      const input = tfc.tensor2d([1, 1, 1, 1], [2, 2], 'float32');
      // input is stored in cpu backend.

      const inputReshaped = tfc.reshape(input, [2, 2]);

      const cpuAfter = tfc.engine().backend.numDataIds();

      expect(cpuAfter).toEqual(cpuBefore + 1);

      await tfc.setBackend('webgl');

      const webglBefore = tfc.engine().backend.numDataIds();

      const inputReshaped2 = tfc.reshape(inputReshaped, [2, 2]);
      // input moved to webgl.

      // Because input is moved to webgl, data should be deleted from cpu.
      expect(tfc.findBackend('cpu').numDataIds()).toEqual(cpuAfter - 1);

      const webglAfter = tfc.engine().backend.numDataIds();

      expect(webglAfter).toEqual(webglBefore + 1);

      input.dispose();

      expect(tfc.engine().backend.numDataIds()).toEqual(webglAfter);

      inputReshaped.dispose();

      expect(tfc.engine().backend.numDataIds()).toEqual(webglAfter);

      inputReshaped2.dispose();

      const after = tfc.engine().backend.numDataIds();

      expect(after).toBe(webglBefore);
    });
  });
});
