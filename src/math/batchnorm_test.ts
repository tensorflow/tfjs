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

import * as test_util from '../test_util';
import {MathTests} from '../test_util';

import {Array1D, Array2D, Array3D, Array4D} from './ndarray';

// math.batchNormalization4D
{
  // TODO(nsthorat): Fix the precision for byte-packed batchnorm.
  const epsilon = 1e-1;
  const tests: MathTests = it => {
    it('simple batchnorm4D, no offset or scale, 2x1x1x2', math => {
      const x = Array4D.new([2, 1, 1, 2], new Float32Array([2, 100, 4, 400]));
      const mean = Array1D.new([1, 2]);
      const variance = Array1D.new([2, 3]);
      const varianceEpsilon = .001;

      const result = math.batchNormalization4D(
          x, mean, variance, varianceEpsilon, undefined, undefined);

      test_util.expectArraysClose(
          result,
          [
            (x.get(0, 0, 0, 0) - mean.get(0)) * 1 /
                Math.sqrt(variance.get(0) + varianceEpsilon),
            (x.get(0, 0, 0, 1) - mean.get(1)) * 1 /
                Math.sqrt(variance.get(1) + varianceEpsilon),
            (x.get(1, 0, 0, 0) - mean.get(0)) * 1 /
                Math.sqrt(variance.get(0) + varianceEpsilon),
            (x.get(1, 0, 0, 1) - mean.get(1)) * 1 /
                Math.sqrt(variance.get(1) + varianceEpsilon)
          ],
          epsilon);
    });

    it('simple batchnorm4D, no offset, 2x1x1x2', math => {
      const x = Array4D.new([2, 1, 1, 2], new Float32Array([2, 100, 4, 400]));
      const mean = Array1D.new([1, 2]);
      const variance = Array1D.new([2, 3]);
      const scale = Array1D.new([4, 5]);
      const varianceEpsilon = .001;

      const result = math.batchNormalization4D(
          x, mean, variance, varianceEpsilon, scale, undefined);

      test_util.expectArraysClose(
          result,
          [
            (x.get(0, 0, 0, 0) - mean.get(0)) * scale.get(0) /
                Math.sqrt(variance.get(0) + varianceEpsilon),
            (x.get(0, 0, 0, 1) - mean.get(1)) * scale.get(1) /
                Math.sqrt(variance.get(1) + varianceEpsilon),
            (x.get(1, 0, 0, 0) - mean.get(0)) * scale.get(0) /
                Math.sqrt(variance.get(0) + varianceEpsilon),
            (x.get(1, 0, 0, 1) - mean.get(1)) * scale.get(1) /
                Math.sqrt(variance.get(1) + varianceEpsilon)
          ],
          epsilon);
    });

    it('simple batchnorm4D, no scale, 2x1x1x2', math => {
      const x = Array4D.new([2, 1, 1, 2], new Float32Array([2, 100, 4, 400]));
      const mean = Array1D.new([1, 2]);
      const variance = Array1D.new([2, 3]);
      const offset = Array1D.new([4, 5]);

      const varianceEpsilon = .001;

      const result = math.batchNormalization4D(
          x, mean, variance, varianceEpsilon, undefined, offset);

      test_util.expectArraysClose(
          result,
          [
            offset.get(0) +
                (x.get(0, 0, 0, 0) - mean.get(0)) * 1 /
                    Math.sqrt(variance.get(0) + varianceEpsilon),
            offset.get(1) +
                (x.get(0, 0, 0, 1) - mean.get(1)) * 1 /
                    Math.sqrt(variance.get(1) + varianceEpsilon),
            offset.get(0) +
                (x.get(1, 0, 0, 0) - mean.get(0)) * 1 /
                    Math.sqrt(variance.get(0) + varianceEpsilon),
            offset.get(1) +
                (x.get(1, 0, 0, 1) - mean.get(1)) * 1 /
                    Math.sqrt(variance.get(1) + varianceEpsilon)
          ],
          epsilon);
    });

    it('simple batchnorm4D, 2x1x1x2', math => {
      const x = Array4D.new([2, 1, 1, 2], new Float32Array([2, 100, 4, 400]));
      const mean = Array1D.new([1, 2]);
      const variance = Array1D.new([2, 3]);
      const offset = Array1D.new([3, 4]);
      const scale = Array1D.new([4, 5]);

      const varianceEpsilon = .001;

      const result = math.batchNormalization4D(
          x, mean, variance, varianceEpsilon, scale, offset);

      test_util.expectArraysClose(
          result,
          [
            offset.get(0) +
                (x.get(0, 0, 0, 0) - mean.get(0)) * scale.get(0) /
                    Math.sqrt(variance.get(0) + varianceEpsilon),
            offset.get(1) +
                (x.get(0, 0, 0, 1) - mean.get(1)) * scale.get(1) /
                    Math.sqrt(variance.get(1) + varianceEpsilon),
            offset.get(0) +
                (x.get(1, 0, 0, 0) - mean.get(0)) * scale.get(0) /
                    Math.sqrt(variance.get(0) + varianceEpsilon),
            offset.get(1) +
                (x.get(1, 0, 0, 1) - mean.get(1)) * scale.get(1) /
                    Math.sqrt(variance.get(1) + varianceEpsilon)
          ],
          epsilon);
    });
  };

  test_util.describeMathCPU('batchNormalization4D', [tests]);
  test_util.describeMathGPU('batchNormalization4D', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.batchNormalization3D
{
  // TODO(nsthorat): Fix the precision for byte-packed batchnorm.
  const epsilon = 1e-1;
  const tests: MathTests = it => {
    it('simple batchnorm3D, no offset or scale, 2x1x2', math => {
      const x = Array3D.new([2, 1, 2], [2, 100, 4, 400]);
      const mean = Array1D.new([1, 2]);
      const variance = Array1D.new([2, 3]);
      const varianceEpsilon = .001;

      const result = math.batchNormalization3D(
          x, mean, variance, varianceEpsilon, undefined, undefined);

      test_util.expectArraysClose(
          result, [
            (x.get(0, 0, 0) - mean.get(0)) * 1 /
                Math.sqrt(variance.get(0) + varianceEpsilon),
            (x.get(0, 0, 1) - mean.get(1)) * 1 /
                Math.sqrt(variance.get(1) + varianceEpsilon),
            (x.get(1, 0, 0) - mean.get(0)) * 1 /
                Math.sqrt(variance.get(0) + varianceEpsilon),
            (x.get(1, 0, 1) - mean.get(1)) * 1 /
                Math.sqrt(variance.get(1) + varianceEpsilon)
          ],
          epsilon);
    });

    it('simple batchnorm3D, no offset, 2x1x2', math => {
      const x = Array3D.new([2, 1, 2], [2, 100, 4, 400]);
      const mean = Array1D.new([1, 2]);
      const variance = Array1D.new([2, 3]);
      const scale = Array1D.new([4, 5]);
      const varianceEpsilon = .001;

      const result = math.batchNormalization3D(
          x, mean, variance, varianceEpsilon, scale, undefined);

      test_util.expectArraysClose(
          result, [
            (x.get(0, 0, 0) - mean.get(0)) * scale.get(0) /
                Math.sqrt(variance.get(0) + varianceEpsilon),
            (x.get(0, 0, 1) - mean.get(1)) * scale.get(1) /
                Math.sqrt(variance.get(1) + varianceEpsilon),
            (x.get(1, 0, 0) - mean.get(0)) * scale.get(0) /
                Math.sqrt(variance.get(0) + varianceEpsilon),
            (x.get(1, 0, 1) - mean.get(1)) * scale.get(1) /
                Math.sqrt(variance.get(1) + varianceEpsilon)
          ],
          epsilon);
    });

    it('simple batchnorm3D, no scale, 2x1x2', math => {
      const x = Array3D.new([2, 1, 2], [2, 100, 4, 400]);
      const mean = Array1D.new([1, 2]);
      const variance = Array1D.new([2, 3]);
      const offset = Array1D.new([4, 5]);

      const varianceEpsilon = .001;

      const result = math.batchNormalization3D(
          x, mean, variance, varianceEpsilon, undefined, offset);

      test_util.expectArraysClose(
          result, [
            offset.get(0) +
                (x.get(0, 0, 0) - mean.get(0)) * 1 /
                    Math.sqrt(variance.get(0) + varianceEpsilon),
            offset.get(1) +
                (x.get(0, 0, 1) - mean.get(1)) * 1 /
                    Math.sqrt(variance.get(1) + varianceEpsilon),
            offset.get(0) +
                (x.get(1, 0, 0) - mean.get(0)) * 1 /
                    Math.sqrt(variance.get(0) + varianceEpsilon),
            offset.get(1) +
                (x.get(1, 0, 1) - mean.get(1)) * 1 /
                    Math.sqrt(variance.get(1) + varianceEpsilon)
          ],
          epsilon);
    });

    it('simple batchnorm3D, 2x1x2', math => {
      const x = Array3D.new([2, 1, 2], [2, 100, 4, 400]);
      const mean = Array1D.new([1, 2]);
      const variance = Array1D.new([2, 3]);
      const offset = Array1D.new([3, 4]);
      const scale = Array1D.new([4, 5]);

      const varianceEpsilon = .001;

      const result = math.batchNormalization3D(
          x, mean, variance, varianceEpsilon, scale, offset);

      test_util.expectArraysClose(
          result, [
            offset.get(0) +
                (x.get(0, 0, 0) - mean.get(0)) * scale.get(0) /
                    Math.sqrt(variance.get(0) + varianceEpsilon),
            offset.get(1) +
                (x.get(0, 0, 1) - mean.get(1)) * scale.get(1) /
                    Math.sqrt(variance.get(1) + varianceEpsilon),
            offset.get(0) +
                (x.get(1, 0, 0) - mean.get(0)) * scale.get(0) /
                    Math.sqrt(variance.get(0) + varianceEpsilon),
            offset.get(1) +
                (x.get(1, 0, 1) - mean.get(1)) * scale.get(1) /
                    Math.sqrt(variance.get(1) + varianceEpsilon)
          ],
          epsilon);
    });

    it('batchnorm matches tensorflow, 2x3x3', math => {
      const x = Array3D.new(
          [2, 3, 3], [
            0.49955603, 0.04158615, -1.09440524, 2.03854165, -0.61578344,
            2.87533573, 1.18105987, 0.807462, 1.87888837, 2.26563962,
            -0.37040935, 1.35848753, -0.75347094, 0.15683117, 0.91925946,
            0.34121279, 0.92717143, 1.89683965
          ]);
      const mean = Array1D.new([0.39745062, -0.48062894, 0.4847822]);
      const variance = Array1D.new([0.32375343, 0.67117643, 1.08334653]);
      const offset = Array1D.new([0.69398749, -1.29056387, 0.9429723]);
      const scale = Array1D.new([-0.5607271, 0.9878457, 0.25181573]);
      const varianceEpsilon = .001;

      const result = math.batchNormalization3D(
          x, mean, variance, varianceEpsilon, scale, offset);

      test_util.expectArraysClose(
          result, [
            0.59352049, -0.66135202, 0.5610874, -0.92077015, -1.45341019,
            1.52106473, -0.07704776, 0.26144429, 1.28010017, -1.14422404,
            -1.15776136, 1.15425493, 1.82644104, -0.52249442, 1.04803919,
            0.74932291, 0.40568101, 1.2844412
          ]);
    });
  };

  test_util.describeMathCPU('batchNormalization3D', [tests]);
  test_util.describeMathGPU('batchNormalization3D', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.batchNormalization2D
{
  // TODO(nsthorat): Fix the precision for byte-packed batchnorm.
  const epsilon = 1e-1;
  const tests: MathTests = it => {
    it('simple batchnorm2D, no offset or scale, 2x2', math => {
      const x = Array2D.new([2, 2], [2, 100, 4, 400]);
      const mean = Array1D.new([1, 2]);
      const variance = Array1D.new([2, 3]);
      const varianceEpsilon = .001;

      const result = math.batchNormalization2D(
          x, mean, variance, varianceEpsilon, undefined, undefined);

      test_util.expectArraysClose(
          result, [
            (x.get(0, 0) - mean.get(0)) * 1 /
                Math.sqrt(variance.get(0) + varianceEpsilon),
            (x.get(0, 1) - mean.get(1)) * 1 /
                Math.sqrt(variance.get(1) + varianceEpsilon),
            (x.get(1, 0) - mean.get(0)) * 1 /
                Math.sqrt(variance.get(0) + varianceEpsilon),
            (x.get(1, 1) - mean.get(1)) * 1 /
                Math.sqrt(variance.get(1) + varianceEpsilon)
          ],
          epsilon);
    });
    it('simple batchnorm2D, no offset, 2x2', math => {
      const x = Array2D.new([2, 2], [2, 100, 4, 400]);
      const mean = Array1D.new([1, 2]);
      const variance = Array1D.new([2, 3]);
      const scale = Array1D.new([4, 5]);
      const varianceEpsilon = .001;

      const result = math.batchNormalization2D(
          x, mean, variance, varianceEpsilon, scale, undefined);

      test_util.expectArraysClose(
          result, [
            (x.get(0, 0) - mean.get(0)) * scale.get(0) /
                Math.sqrt(variance.get(0) + varianceEpsilon),
            (x.get(0, 1) - mean.get(1)) * scale.get(1) /
                Math.sqrt(variance.get(1) + varianceEpsilon),
            (x.get(1, 0) - mean.get(0)) * scale.get(0) /
                Math.sqrt(variance.get(0) + varianceEpsilon),
            (x.get(1, 1) - mean.get(1)) * scale.get(1) /
                Math.sqrt(variance.get(1) + varianceEpsilon)
          ],
          epsilon);
    });

    it('simple batchnorm2D, no scale, 2x2', math => {
      const x = Array2D.new([2, 2], [2, 100, 4, 400]);
      const mean = Array1D.new([1, 2]);
      const variance = Array1D.new([2, 3]);
      const offset = Array1D.new([4, 5]);

      const varianceEpsilon = .001;

      const result = math.batchNormalization2D(
          x, mean, variance, varianceEpsilon, undefined, offset);

      test_util.expectArraysClose(
          result, [
            offset.get(0) +
                (x.get(0, 0) - mean.get(0)) * 1 /
                    Math.sqrt(variance.get(0) + varianceEpsilon),
            offset.get(1) +
                (x.get(0, 1) - mean.get(1)) * 1 /
                    Math.sqrt(variance.get(1) + varianceEpsilon),
            offset.get(0) +
                (x.get(1, 0) - mean.get(0)) * 1 /
                    Math.sqrt(variance.get(0) + varianceEpsilon),
            offset.get(1) +
                (x.get(1, 1) - mean.get(1)) * 1 /
                    Math.sqrt(variance.get(1) + varianceEpsilon)
          ],
          epsilon);
    });

    it('simple batchnorm2D, 2x2', math => {
      const x = Array2D.new([2, 2], [2, 100, 4, 400]);
      const mean = Array1D.new([1, 2]);
      const variance = Array1D.new([2, 3]);
      const offset = Array1D.new([3, 4]);
      const scale = Array1D.new([4, 5]);

      const varianceEpsilon = .001;

      const result = math.batchNormalization2D(
          x, mean, variance, varianceEpsilon, scale, offset);

      test_util.expectArraysClose(
          result, [
            offset.get(0) +
                (x.get(0, 0) - mean.get(0)) * scale.get(0) /
                    Math.sqrt(variance.get(0) + varianceEpsilon),
            offset.get(1) +
                (x.get(0, 1) - mean.get(1)) * scale.get(1) /
                    Math.sqrt(variance.get(1) + varianceEpsilon),
            offset.get(0) +
                (x.get(1, 0) - mean.get(0)) * scale.get(0) /
                    Math.sqrt(variance.get(0) + varianceEpsilon),
            offset.get(1) +
                (x.get(1, 1) - mean.get(1)) * scale.get(1) /
                    Math.sqrt(variance.get(1) + varianceEpsilon)
          ],
          epsilon);
    });

    it('batchnorm2D matches tensorflow, 3x3', math => {
      const x =
          Array2D.new([3, 3], [
                        0.3136892, 0.92389025, 0.594782, 0.05021042, 0.67545404,
                        0.93910035, 0.13277993, 0.96474269, 0.88608916
                      ]);
      const mean = Array1D.new([0.19526312, 0.74857256, 0.45166398]);
      const variance = Array1D.new([0.22963001, 0.61521992, 0.46623685]);
      const offset = Array1D.new([0.43098484, 0.77712237, 0.47916298]);
      const scale = Array1D.new([0.62186907, 0.85673736, 0.19201061]);
      const varianceEpsilon = .001;

      const result = math.batchNormalization2D(
          x, mean, variance, varianceEpsilon, scale, offset);

      test_util.expectArraysClose(
          result, [
            0.58433646, 0.96846228, 0.51936529, 0.24315402, 0.69732157,
            0.61608542, 0.35007446, 1.01304821, 0.60119441
          ]);
    });
  };
  test_util.describeMathCPU('batchNormalization2D', [tests]);
  test_util.describeMathGPU('batchNormalization2D', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
