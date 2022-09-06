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
import {CHROME_ENVS, Constraints, describeWithFlags, HAS_WORKER} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {REGRESSION} from './constants';

const CHROME_ENVS_WITH_WORKER: Constraints =
    Object.assign({}, CHROME_ENVS, HAS_WORKER);
/**
 *  This file is the test suite for CUJ: custom_module->custom_module->predict.
 */

function getBundleUrl(folder: string, custom: boolean, bundler: string) {
  const distFolder = custom ? 'custom' : 'full';
  return `./base/custom_module/${folder}/dist/${distFolder}/app_${bundler}.js`;
}

const DEBUG_WORKER_SCRIPT = true;

describe(`${REGRESSION} blazeface`, () => {
  describeWithFlags('webpack', CHROME_ENVS_WITH_WORKER, () => {
    let webpackBundle: {full: string, custom: string};
    let originalTimeout: number;
    beforeAll(async () => {
      originalTimeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
      jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;

      const [webpackFull, webpackCustom] = await Promise.all([
        fetch(getBundleUrl('blazeface', false /* custom */, 'webpack'))
            .then(r => r.text())
            .catch(e => {
              console.error(
                  'Failed to fetch blazeface full bundle at ',
                  getBundleUrl('blazeface', false /* custom */, 'webpack'));
              throw e;
            }),
        fetch(getBundleUrl('blazeface', true /* custom */, 'webpack'))
            .then(r => r.text())
            .catch(e => {
              console.error(
                  'Failed to fetch blazeface custom bundle at ',
                  getBundleUrl('blazeface', false /* custom */, 'webpack'));
              throw e;
            }),
      ]);

      webpackBundle = {full: webpackFull, custom: webpackCustom};
    });

    afterAll(() => jasmine.DEFAULT_TIMEOUT_INTERVAL = originalTimeout);

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
          await executeInWorker(programUrl, {debug: DEBUG_WORKER_SCRIPT});
      const kernelNames = result.kernelNames;
      expect(kernelNames).toEqual(jasmine.arrayWithExactContents([
        'Cast',
        'ExpandDims',
        'Reshape',
        'ResizeBilinear',
        'RealDiv',
        'Sub',
        'Multiply',
        'FusedConv2D',
        'DepthwiseConv2dNative',
        'Add',
        'Relu',
        'Pack',
        'PadV2',
        'MaxPool',
        'Slice',
        'StridedSlice',
        'Concat',
        'Identity',
        'Sigmoid',
        'NonMaxSuppressionV3'
      ]));

      expect(result.predictions.length).toEqual(1);
    });
  });
});

describe(`${REGRESSION} dense model`, () => {
  describeWithFlags('webpack', CHROME_ENVS_WITH_WORKER, () => {
    let webpackBundle: {full: string, custom: string};
    let originalTimeout: number;

    let modelUrl: string;
    beforeAll(async () => {
      originalTimeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
      jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;

      modelUrl = `/base/custom_module/dense_model/model/model.json`;
      const [webpackFull, webpackCustom] = await Promise.all([
        fetch(getBundleUrl('dense_model', false /* custom */, 'webpack'))
            .then(r => r.text()),
        fetch(getBundleUrl('dense_model', true /* custom */, 'webpack'))
            .then(r => r.text()),
      ]);

      webpackBundle = {full: webpackFull, custom: webpackCustom};
    });

    afterAll(() => jasmine.DEFAULT_TIMEOUT_INTERVAL = originalTimeout);

    it('custom webpack should be smaller', async () => {
      expect(webpackBundle.custom.length)
          .toBeLessThan(
              webpackBundle.full.length / 2,
              'Custom bundle should be smaller than full bundle');
    });

    it('custom bundle should execute with exact kernels', async () => {
      const programUrl =
          getBundleUrl('dense_model', true /* custom */, 'webpack');

      // tslint:disable-next-line: no-any
      const result: any = await executeInWorker(
          programUrl, {debug: DEBUG_WORKER_SCRIPT, workerParams: {modelUrl}});
      const kernelNames = result.kernelNames;
      expect(kernelNames).toEqual(jasmine.arrayWithExactContents([
        'Reshape', '_FusedMatMul', 'Identity'
      ]));

      expect(Math.floor(result.predictions[0])).toEqual(38);
    });
  });

  describeWithFlags('rollup', CHROME_ENVS_WITH_WORKER, () => {
    let rollupBundle: {full: string, custom: string};
    let originalTimeout: number;

    let modelUrl: string;
    beforeAll(async () => {
      originalTimeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
      jasmine.DEFAULT_TIMEOUT_INTERVAL = 500000;

      modelUrl = `/base/custom_module/dense_model/model/model.json`;
      const [rollupFull, rollupCustom] = await Promise.all([
        fetch(getBundleUrl('dense_model', false /* custom */, 'rollup'))
            .then(r => r.text()),
        fetch(getBundleUrl('dense_model', true /* custom */, 'rollup'))
            .then(r => r.text()),
      ]);

      rollupBundle = {full: rollupFull, custom: rollupCustom};
    });

    afterAll(() => jasmine.DEFAULT_TIMEOUT_INTERVAL = originalTimeout);

    it('custom rollup should be smaller', async () => {
      expect(rollupBundle.custom.length)
          .toBeLessThan(
              rollupBundle.full.length / 2,
              'Custom bundle should be smaller than full bundle');
    });

    it('custom bundle should execute with exact kernels', async () => {
      const programUrl =
          getBundleUrl('dense_model', true /* custom */, 'webpack');

      // tslint:disable-next-line: no-any
      const result: any = await executeInWorker(
          programUrl, {debug: DEBUG_WORKER_SCRIPT, workerParams: {modelUrl}});
      const kernelNames = result.kernelNames;
      expect(kernelNames).toEqual(jasmine.arrayWithExactContents([
        'Reshape', '_FusedMatMul', 'Identity'
      ]));

      expect(Math.floor(result.predictions[0])).toEqual(38);
    });
  });
});

describe(`${REGRESSION} universal sentence encoder model`, () => {
  const expectedKernels = [
    'StridedSlice', 'Less',       'Cast',      'Reshape',       'GatherV2',
    'Max',          'Add',        'Maximum',   'SparseToDense', 'Greater',
    'Sum',          'ExpandDims', 'Concat',    'LogicalNot',    'Multiply',
    'ScatterNd',    'GatherNd',   'Cos',       'Sin',           'BatchMatMul',
    'Mean',         'Sub',        'Square',    'Rsqrt',         'Conv2D',
    'SplitV',       'Pack',       'Transpose', 'Slice',         'Softmax',
    'Prod',         'Relu',       'Range',     'RealDiv',       'Tanh'
  ];

  describeWithFlags('webpack', CHROME_ENVS_WITH_WORKER, () => {
    let webpackBundle: {full: string, custom: string};
    let originalTimeout: number;

    beforeAll(async () => {
      originalTimeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
      jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;

      const [webpackFull, webpackCustom] = await Promise.all([
        fetch(getBundleUrl(
                  'universal_sentence_encoder', false /* custom */, 'webpack'))
            .then(r => r.text()),
        fetch(getBundleUrl(
                  'universal_sentence_encoder', true /* custom */, 'webpack'))
            .then(r => r.text()),
      ]);

      webpackBundle = {full: webpackFull, custom: webpackCustom};
    });

    afterAll(() => jasmine.DEFAULT_TIMEOUT_INTERVAL = originalTimeout);

    it('custom webpack should be smaller', async () => {
      expect(webpackBundle.custom.length)
          .toBeLessThan(
              webpackBundle.full.length / 2,
              'Custom bundle should be smaller than full bundle');
    });

    it('custom bundle should execute with exact kernels', async () => {
      const programUrl = getBundleUrl(
          'universal_sentence_encoder', true /* custom */, 'webpack');

      // tslint:disable-next-line: no-any
      const result: any = await executeInWorker(
          programUrl,
          {debug: DEBUG_WORKER_SCRIPT, workerParams: {profile: false}});
      const kernelNames = result.kernelNames;
      expect(kernelNames)
          .toEqual(jasmine.arrayWithExactContents(expectedKernels));

      expect(result.predictions.shape[0]).toEqual(2);
      expect(result.predictions.shape[1]).toEqual(512);
      expect(result.predictions.shape.length).toEqual(2);
    });
  });

  describeWithFlags('rollup', CHROME_ENVS_WITH_WORKER, () => {
    let rollupBundle: {full: string, custom: string};
    let originalTimeout: number;

    beforeAll(async () => {
      originalTimeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
      jasmine.DEFAULT_TIMEOUT_INTERVAL = 500000;

      const [rollupFull, rollupCustom] = await Promise.all([
        fetch(getBundleUrl(
                  'universal_sentence_encoder', false /* custom */, 'rollup'))
            .then(r => r.text()),
        fetch(getBundleUrl(
                  'universal_sentence_encoder', true /* custom */, 'rollup'))
            .then(r => r.text()),
      ]);

      rollupBundle = {full: rollupFull, custom: rollupCustom};
    });

    afterAll(() => jasmine.DEFAULT_TIMEOUT_INTERVAL = originalTimeout);

    it('custom rollup should be smaller', async () => {
      expect(rollupBundle.custom.length)
          .toBeLessThan(
              rollupBundle.full.length / 2,
              'Custom bundle should be smaller than full bundle');
    });

    it('custom bundle should execute with exact kernels', async () => {
      const programUrl = getBundleUrl(
          'universal_sentence_encoder', true /* custom */, 'webpack');

      // tslint:disable-next-line: no-any
      const result: any = await executeInWorker(
          programUrl,
          {debug: DEBUG_WORKER_SCRIPT, workerParams: {profile: false}});
      const kernelNames = result.kernelNames;
      expect(kernelNames)
          .toEqual(jasmine.arrayWithExactContents(expectedKernels));

      expect(result.predictions.shape[0]).toEqual(2);
      expect(result.predictions.shape[1]).toEqual(512);
      expect(result.predictions.shape.length).toEqual(2);
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
async function executeInWorker(
    programUrl: string, opts: {debug?: boolean, workerParams?: {}}) {
  return new Promise((resolve, reject) => {
    const debug = opts.debug || false;
    const workerParams = opts.workerParams || {};
    const worker = new Worker(programUrl);

    worker.addEventListener('message', (evt) => {
      if (evt.data.error) {
        reject(evt.data.payload);
      }

      if (debug && evt.data.msg) {
        console.log('msg from worker: ', evt.data);
      }

      if (evt.data.result) {
        if (debug) {
          console.log('result from worker: ', evt.data.result);
        }
        resolve(evt.data.payload);
      }
    }, false);

    worker.postMessage(workerParams);  // Send data to our worker.
  });
}
