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
import '@tensorflow/tfjs-backend-webgpu';

import * as tfc from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {SMOKE} from './constants';

/**
 *  This file tests backend switching scenario.
 */
// TODO: Support backend switching between wasm and cpu.
// https://github.com/tensorflow/tfjs/issues/7623
describeWithFlags(
    `${SMOKE} backend switching`, {
      predicate: testEnv =>
          testEnv.backendName !== 'cpu' && testEnv.backendName !== 'wasm'
    },

    (env) => {
      it(`from ${env.name} to cpu.`, async () => {
        await tfc.setBackend(env.name);

        const backendBefore = tfc.engine().backend.numDataIds();

        const input = tfc.tensor2d([1, 1, 1, 1], [2, 2], 'float32');
        // input is stored in backend.

        const inputReshaped = tfc.reshape(input, [2, 2]);

        const backendAfter = tfc.engine().backend.numDataIds();

        expect(backendAfter).toEqual(backendBefore + 1);

        await tfc.setBackend('cpu');

        const cpuBefore = tfc.engine().backend.numDataIds();

        const inputReshaped2 = tfc.reshape(inputReshaped, [2, 2]);
        // input moved to cpu.

        // Because input is moved to cpu, data should be deleted from backend.
        expect(tfc.findBackend(env.name).numDataIds())
            .toEqual(backendAfter - 1);

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

      it(`from cpu to ${env.name}.`, async () => {
        await tfc.setBackend('cpu');

        const cpuBefore = tfc.engine().backend.numDataIds();

        const input = tfc.tensor2d([1, 1, 1, 1], [2, 2], 'float32');
        // input is stored in cpu backend.

        const inputReshaped = tfc.reshape(input, [2, 2]);

        const cpuAfter = tfc.engine().backend.numDataIds();

        expect(cpuAfter).toEqual(cpuBefore + 1);

        await tfc.setBackend(env.name);

        const backendBefore = tfc.engine().backend.numDataIds();

        const inputReshaped2 = tfc.reshape(inputReshaped, [2, 2]);
        // input moved to webgl or webgpu.

        // Because input is moved to backend, data should be deleted
        // from cpu.
        expect(tfc.findBackend('cpu').numDataIds()).toEqual(cpuAfter - 1);

        const backendAfter = tfc.engine().backend.numDataIds();

        expect(backendAfter).toEqual(backendBefore + 1);

        input.dispose();

        expect(tfc.engine().backend.numDataIds()).toEqual(backendAfter);

        inputReshaped.dispose();

        expect(tfc.engine().backend.numDataIds()).toEqual(backendAfter);

        inputReshaped2.dispose();

        const after = tfc.engine().backend.numDataIds();

        expect(after).toBe(backendBefore);
      });

      it('can execute op with data from mixed backends', async () => {
        const numTensors = tfc.memory().numTensors;
        const backendNumDataIds = tfc.findBackend(env.name).numDataIds();
        const cpuNumDataIds = tfc.findBackend('cpu').numDataIds();

        await tfc.setBackend('cpu');
        // This scalar lives in cpu.
        const a = tfc.scalar(5);

        await tfc.setBackend(env.name);
        // This scalar lives in webgl or webgpu.
        const b = tfc.scalar(3);

        // Verify that ops can execute with mixed backend data.
        tfc.engine().startScope();

        await tfc.setBackend('cpu');
        const result = tfc.add(a, b);
        tfc.test_util.expectArraysClose(await result.data(), [8]);
        expect(tfc.findBackend('cpu').numDataIds()).toBe(cpuNumDataIds + 3);

        await tfc.setBackend(env.name);
        tfc.test_util.expectArraysClose(await tfc.add(a, b).data(), [8]);
        expect(tfc.findBackend(env.name).numDataIds())
            .toBe(backendNumDataIds + 3);

        tfc.engine().endScope();

        expect(tfc.memory().numTensors).toBe(numTensors + 2);
        expect(tfc.findBackend(env.name).numDataIds())
            .toBe(backendNumDataIds + 2);
        expect(tfc.findBackend('cpu').numDataIds()).toBe(cpuNumDataIds);

        tfc.dispose([a, b]);

        expect(tfc.memory().numTensors).toBe(numTensors);
        expect(tfc.findBackend(env.name).numDataIds()).toBe(backendNumDataIds);
        expect(tfc.findBackend('cpu').numDataIds()).toBe(cpuNumDataIds);
      });

      // tslint:disable-next-line: ban
      xit(`can move complex tensor from cpu to ${env.name}.`, async () => {
        await tfc.setBackend('cpu');

        const real1 = tfc.tensor1d([1]);
        const imag1 = tfc.tensor1d([2]);
        const complex1 = tfc.complex(real1, imag1);

        await tfc.setBackend(env.name);

        const real2 = tfc.tensor1d([3]);
        const imag2 = tfc.tensor1d([4]);
        const complex2 = tfc.complex(real2, imag2);

        const result = complex1.add(complex2);

        tfc.test_util.expectArraysClose(await result.data(), [4, 6]);
      });

      // tslint:disable-next-line: ban
      xit(`can move complex tensor from ${env.name} to cpu.`, async () => {
        await tfc.setBackend(env.name);

        const real1 = tfc.tensor1d([1]);
        const imag1 = tfc.tensor1d([2]);
        const complex1 = tfc.complex(real1, imag1);

        await tfc.setBackend('cpu');

        const real2 = tfc.tensor1d([3]);
        const imag2 = tfc.tensor1d([4]);
        const complex2 = tfc.complex(real2, imag2);

        const result = complex1.add(complex2);

        tfc.test_util.expectArraysClose(await result.data(), [4, 6]);
      });
    });
