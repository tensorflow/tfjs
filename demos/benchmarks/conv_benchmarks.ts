/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import {initializeGPU} from '../../src/math/ndarray';
import {Conv2DProgram} from '../../src/math/webgl/conv_gpu';
import * as gpgpu_math from '../../src/math/webgl/gpgpu_math';
import {TextureManager} from '../../src/math/webgl/texture_manager';
// tslint:disable-next-line:max-line-length
import {Array1D, Array3D, Array4D, conv_util, ENV, GPGPUContext} from '../deeplearn';

import {BenchmarkTest} from './benchmark';

export interface ConvBenchmarkParams {
  inDepth: number;
  outDepth: number;
  filterSize: number;
  stride: number;
}

export abstract class ConvBenchmark extends BenchmarkTest {
  constructor(protected params: ConvBenchmarkParams) {
    super(params);
  }
}

export class ConvGPUBenchmark extends ConvBenchmark {
  run(size: number): Promise<number> {
    return new Promise<number>((resolve, reject) => {
      const gpgpu = new GPGPUContext();
      const texManager = new TextureManager(gpgpu);
      initializeGPU(gpgpu, texManager);

      const inDepth = this.params.inDepth;
      const inShape: [number, number, number] = [size, size, inDepth];
      const outDepth = this.params.outDepth;
      const filterSize = this.params.filterSize;
      const stride = this.params.stride;
      const hasBias = true;
      const convInfo = conv_util.computeConvInfo(
          inShape, filterSize, filterSize, outDepth, stride, stride, 'same');
      const program = new Conv2DProgram(convInfo, hasBias);
      const outputShape = program.outputShape as [number, number, number];
      const out = Array3D.zeros(outputShape);
      const x = Array3D.randUniform(inShape, -1, 1);
      const wShape =
          conv_util.computeWeightsShape4D(1, outDepth, filterSize, filterSize);
      const W = Array4D.randUniform(wShape, -1, 1);
      const b = Array1D.randUniform([outDepth], -1, 1);
      const inputs = [x, W, b];
      const binary = gpgpu_math.compileProgram(gpgpu, program, inputs, out);

      const benchmark = () => {
        gpgpu_math.runProgram(binary, inputs, out);
      };

      const immediateCleanup = () => {
        x.dispose();
        W.dispose();
        b.dispose();
        out.dispose();
        texManager.dispose();
        gpgpu.deleteProgram(binary.webGLProgram);
      };

      const delayedCleanup = () => {
        gpgpu.dispose();
      };

      if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE')) {
        gpgpu.runQuery(benchmark).then((timeElapsed: number) => {
          delayedCleanup();
          resolve(timeElapsed);
        });
        immediateCleanup();
      } else {
        const start = performance.now();

        benchmark();
        out.getValues();

        const totalTime = performance.now() - start;

        immediateCleanup();
        delayedCleanup();

        resolve(totalTime);
      }
    });
  }
}
