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
// tslint:disable-next-line: no-imports-from-dist
import {CHROME_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {REGRESSION} from './constants';

/**
 *  This file is the test suite for CUJ: custom_module->custom_bundle->predict.
 */

function getBundleUrl(folder: string, custom: boolean, bundler: string) {
  const distFolder = custom ? 'custom' : 'full';
  return `./base/custom_bundle/${folder}/dist/${distFolder}/app_${bundler}.js`;
}

const DEBUG_WORKER_SCRIPT = false;

describeWithFlags(`${REGRESSION} blazeface`, CHROME_ENVS, () => {
  describe('webpack', () => {
    let webpackBundle: {full: string, custom: string};
    beforeAll(async () => {
      const [webpackFull, webpackCustom] = await Promise.all([
        fetch(getBundleUrl('blazeface', false /* custom */, 'webpack'))
            .then(r => r.text()),
        fetch(getBundleUrl('blazeface', true /* custom */, 'webpack'))
            .then(r => r.text()),
      ]);

      webpackBundle = {full: webpackFull, custom: webpackCustom};
    });

    it('custom webpack should be smaller', async () => {
      expect(webpackBundle.custom.length)
          .toBeLessThan(
              webpackBundle.full.length,
              'Custom bundle should be smaller than full bundle');
    });

    it('custom bundle should execute with exact kernels', async () => {
      const programUrl =
          getBundleUrl('blazeface', true /* custom */, 'webpack');
      // tslint:disable-next-line: no-any
      const result: any =
          await executeInWorker(programUrl, DEBUG_WORKER_SCRIPT);
      const kernelNames = result.kernelNames;
      expect(kernelNames).toEqual(jasmine.arrayWithExactContents([
        'Cast', 'Reshape', 'ResizeBilinear', 'Div', 'Sub', 'Multiply',
        'FusedConv2D', 'DepthwiseConv2dNative', 'Add', 'Relu', 'PadV2',
        'MaxPool', 'Slice', 'StridedSlice', 'Concat', 'Identity', 'Sigmoid',
        'NonMaxSuppressionV3'
      ]));

      expect(result.predictions.length).toEqual(1);
    });
  });
});

/**
 * Helper function for executing scripts in a webworker. We use
 * webworkers to get isolated contexts for tests for custom bundles.
 *
 * @param programUrl url to script to run in worker
 * @param debug debug mode
 */
async function executeInWorker(programUrl: string, debug = false) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(programUrl);

    worker.addEventListener('message', (evt) => {
      if (evt.data.error) {
        reject(evt.data.payload);
      }

      if (debug && evt.data.msg) {
        console.log('msg from worker: ', evt.data);
      }

      if (evt.data.result) {
        resolve(evt.data.payload);
      }
    }, false);

    worker.postMessage('start');  // Send data to our worker.
  });
}
