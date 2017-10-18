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
import {NDArrayMathGPU} from './math_gpu';

import {Array1D, Array3D, Scalar} from './ndarray';

// math.scope
{
  const gpuTests: MathTests = it => {
    it('scope returns NDArray', (math: NDArrayMathGPU) => {
      const a = Array1D.new([1, 2, 3]);
      let b = Array1D.new([0, 0, 0]);

      const numUsedTexturesBefore =
          math.getTextureManager().getNumUsedTextures();

      math.scope(() => {
        const result = math.scope(() => {
          b = math.add(a, b) as Array1D;
          b = math.add(a, b) as Array1D;
          b = math.add(a, b) as Array1D;
          return math.add(a, b);
        });

        // a, b, and result are new textures. All intermediates should be
        // disposed.
        expect(math.getTextureManager().getNumUsedTextures())
            .toEqual(numUsedTexturesBefore + 3);
        test_util.expectArraysClose(
            result.getValues(), new Float32Array([4, 8, 12]));
      });

      // a, b are new textures, result should be disposed.
      expect(math.getTextureManager().getNumUsedTextures())
          .toEqual(numUsedTexturesBefore + 2);
      a.dispose();
      b.dispose();
    });

    it('scope returns NDArray[]', (math: NDArrayMathGPU) => {
      const a = Array1D.new([1, 2, 3]);
      const b = Array1D.new([0, -1, 1]);

      const numUsedTexturesBefore =
          math.getTextureManager().getNumUsedTextures();

      math.scope(() => {
        const result = math.scope(() => {
          math.add(a, b);
          return [math.add(a, b), math.subtract(a, b)];
        });

        // a, b, and 2 results are new textures. All intermediates should be
        // disposed.
        expect(math.getTextureManager().getNumUsedTextures())
            .toEqual(numUsedTexturesBefore + 4);
        test_util.expectArraysClose(
            result[0].getValues(), new Float32Array([1, 1, 4]));
        test_util.expectArraysClose(
            result[1].getValues(), new Float32Array([1, 3, 2]));
      });

      // a, b are new textures, result should be disposed.
      expect(math.getTextureManager().getNumUsedTextures())
          .toEqual(numUsedTexturesBefore + 2);
      a.dispose();
      b.dispose();
    });

    it('basic scope usage without return', (math: NDArrayMathGPU) => {
      const a = Array1D.new([1, 2, 3]);
      let b = Array1D.new([0, 0, 0]);

      const numUsedTexturesBefore =
          math.getTextureManager().getNumUsedTextures();

      math.scope(() => {
        b = math.add(a, b) as Array1D;
        b = math.add(a, b) as Array1D;
        b = math.add(a, b) as Array1D;
        math.add(a, b);
      });

      const numUsedTexturesAfter =
          math.getTextureManager().getNumUsedTextures();

      // original a and b, all intermediates should be disposed.
      expect(numUsedTexturesAfter).toEqual(numUsedTexturesBefore + 2);
    });

    it('nested scope usage', (math: NDArrayMathGPU) => {
      const a = Array1D.new([1, 2, 3]);
      let b = Array1D.new([0, 0, 0]);

      const numUsedTexturesBefore =
          math.getTextureManager().getNumUsedTextures();

      math.scope(() => {
        const result = math.scope(() => {
          b = math.add(a, b) as Array1D;
          b = math.scope(() => {
            b = math.scope(() => {
              return math.add(a, b) as Array1D;
            });
            // a, original b, and two intermediate textures should be the only
            // textures.
            expect(math.getTextureManager().getNumUsedTextures())
                .toEqual(numUsedTexturesBefore + 4);

            math.scope(() => {
              math.add(a, b);
            });
            // All the intermediates should be cleaned up.
            expect(math.getTextureManager().getNumUsedTextures())
                .toEqual(numUsedTexturesBefore + 4);

            return math.add(a, b) as Array1D;
          });
          expect(math.getTextureManager().getNumUsedTextures())
              .toEqual(numUsedTexturesBefore + 4);

          return math.add(a, b) as Array1D;
        });

        // a, b, and result are new textures. All intermediates should be
        // disposed.
        expect(math.getTextureManager().getNumUsedTextures())
            .toEqual(numUsedTexturesBefore + 3);
        test_util.expectArraysClose(
            result.getValues(), new Float32Array([4, 8, 12]));
      });
      // a, b, are new textures, result should be disposed.
      expect(math.getTextureManager().getNumUsedTextures())
          .toEqual(numUsedTexturesBefore + 2);
    });
  };

  test_util.describeMathGPU('scope', [gpuTests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// debug mode
{
  const gpuTests: MathTests = it => {
    it('debug mode does not error when no nans', math => {
      math.enableDebugMode();
      const a = Array1D.new([2, -1, 0, 3]);

      const res = math.relu(a);

      test_util.expectArraysClose(
          res.getValues(), new Float32Array([2, 0, 0, 3]));

      a.dispose();
    });

    it('debug mode errors when there are nans', math => {
      math.enableDebugMode();
      const a = Array1D.new([2, NaN]);

      const f = () => math.relu(a);

      expect(f).toThrowError();

      a.dispose();
    });

    it('no errors where there are nans, and debug mode is disabled', math => {
      const a = Array1D.new([2, NaN]);

      const res = math.relu(a);

      test_util.expectArraysClose(res.getValues(), new Float32Array([2, NaN]));

      a.dispose();
    });
  };

  test_util.describeMathCPU('debug mode', [gpuTests]);
  test_util.describeMathGPU('debug mode', [gpuTests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// fromPixels & math
{
  const tests: MathTests = it => {
    it('debug mode does not error when no nans', math => {
      const pixels = new ImageData(2, 2);
      for (let i = 0; i < 8; i++) {
        pixels.data[i] = 100;
      }
      for (let i = 8; i < 16; i++) {
        pixels.data[i] = 250;
      }

      const a = Array3D.fromPixels(pixels, 4);
      const b = Scalar.new(20.5);

      const res = math.add(a, b);

      test_util.expectArraysClose(
          res.getValues(), new Float32Array([
            120.5, 120.5, 120.5, 120.5, 120.5, 120.5, 120.5, 120.5, 270.5,
            270.5, 270.5, 270.5, 270.5, 270.5, 270.5, 270.5
          ]));

      a.dispose();
    });
  };

  test_util.describeMathGPU('fromPixels + math', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
