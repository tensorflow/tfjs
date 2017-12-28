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
import {Array3D, Array4D} from './ndarray';

// math.depthwiseConv2D
{
  const tests: MathTests = it => {
    it('input=1x3x3x1,f=2,s=1,p=valid,chMul=1', math => {
      const fSize = 2;
      const pad = 'valid';
      const stride = 1;
      const chMul = 1;
      const inDepth = 1;

      const x = Array4D.new([1, 3, 3, inDepth], [
        0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641, 0.0741907,
        0.409265, 0.351377
      ]);
      const w = Array4D.new(
          [fSize, fSize, inDepth, chMul],
          [0.303873, 0.229223, 0.144333, 0.803373]);
      const result = math.depthwiseConv2D(x, w, stride, pad);
      expect(result.shape).toEqual([1, 2, 2, 1]);

      const expected = [1.07022, 1.03167, 0.67041, 0.778863];
      test_util.expectArraysClose(result, expected);
    });

    it('input=1x3x3x2,f=2,s=1,p=same,chMul=1', math => {
      const fSize = 2;
      const pad = 'same';
      const stride = 1;
      const chMul = 1;
      const inDepth = 2;

      const x = Array4D.new([1, 3, 3, inDepth], [
        0.111057, 0.661818, 0.701979, 0.424362, 0.992854, 0.417599, 0.423036,
        0.500499, 0.368484, 0.714135, 0.456693, 0.531058, 0.636636, 0.345024,
        0.0506303, 0.789682, 0.177473, 0.793569
      ]);
      const w = Array4D.new([fSize, fSize, inDepth, chMul], [
        0.614293, 0.0648011, 0.101113, 0.452887, 0.0582746, 0.426481, 0.872743,
        0.765767
      ]);
      const result = math.depthwiseConv2D(x, w, stride, pad);
      expect(result.shape).toEqual([1, 3, 3, 2]);

      const expected = [
        0.485445, 0.995389, 0.95166, 0.927856, 0.636516, 0.253547, 0.378414,
        1.10771, 0.430373, 1.23126, 0.290885, 0.372855, 0.3962, 0.379995,
        0.0490466, 0.410569, 0.10902, 0.0514242
      ];
      test_util.expectArraysClose(result, expected);
    });

    it('input=1x3x3x2,f=2,s=1,p=same,chMul=2', math => {
      const fSize = 2;
      const pad = 'same';
      const stride = 1;
      const chMul = 2;
      const inDepth = 2;

      const x = Array4D.new([1, 3, 3, inDepth], [
        0.675707, 0.758567, 0.413529, 0.963967, 0.217291, 0.101335, 0.804231,
        0.329673, 0.924503, 0.728742, 0.180217, 0.210459, 0.133869, 0.650827,
        0.047613, 0.554795, 0.653365, 0.442196
      ]);
      const w = Array4D.new([fSize, fSize, inDepth, chMul], [
        0.347154, 0.386692, 0.327191, 0.483784, 0.591807, 0.24263, 0.95182,
        0.174353, 0.592136, 0.623469, 0.988244, 0.660731, 0.946534, 0.0801365,
        0.864889, 0.874602
      ]);
      const result = math.depthwiseConv2D(x, w, stride, pad);
      expect(result.shape).toEqual([1, 3, 3, 4]);

      const expected = [
        1.83059,   0.937125,  2.1218,   1.39024,  0.990167, 0.803472,
        1.31405,   1.14959,   0.182147, 0.196385, 0.241141, 0.188081,
        0.950656,  0.622581,  1.92451,  1.20179,  1.07422,  0.483268,
        1.36948,   1.14256,   0.449444, 0.477042, 0.505857, 0.393989,
        0.0746509, 0.0633184, 0.74101,  0.41159,  0.403195, 0.176938,
        0.602415,  0.345499,  0.226819, 0.252651, 0.144682, 0.213927
      ];
      test_util.expectArraysClose(result, expected);
    });

    it('input=2x3x3x2,f=2,s=1,p=same,chMul=2', math => {
      const fSize = 2;
      const pad = 'same';
      const stride = 1;
      const chMul = 2;
      const inDepth = 2;

      const x = Array4D.new([2, 3, 3, inDepth], [
        0.261945, 0.0528113, 0.656698,  0.127345,  0.610039, 0.169131,
        0.458647, 0.0988288, 0.966109,  0.0421747, 0.82035,  0.274711,
        0.359377, 0.512113,  0.689682,  0.941571,  0.31961,  0.743826,
        0.858147, 0.984766,  0.926973,  0.579597,  0.444104, 0.505969,
        0.241437, 0.937999,  0.0957074, 0.773611,  0.46023,  0.469379,
        0.363789, 0.269745,  0.486136,  0.894215,  0.794299, 0.724615
      ]);
      const w = Array4D.new([fSize, fSize, inDepth, chMul], [
        0.240347, 0.906352, 0.478657, 0.825918, 0.380769, 0.184705, 0.238241,
        0.201907, 0.294087, 0.181165, 0.191303, 0.7225, 0.430064, 0.900622,
        0.670338, 0.33478
      ]);
      const result = math.depthwiseConv2D(x, w, stride, pad);
      expect(result.shape).toEqual([2, 3, 3, 4]);

      const expected = [
        0.863379, 1.3119,   0.102795, 0.154853, 1.02704,   1.62173,  0.293466,
        0.261764, 0.387876, 0.701529, 0.133508, 0.338167,  0.880395, 1.28039,
        0.786492, 0.775361, 0.884845, 1.43995,  0.764374,  1.0196,   0.291162,
        0.801428, 0.273788, 0.764303, 0.348985, 0.45311,   0.469447, 0.613073,
        0.287461, 0.684128, 0.627899, 0.927844, 0.0768174, 0.28968,  0.356037,
        0.614339, 0.67138,  1.07894,  1.30747,  1.86705,   0.617971, 1.35402,
        0.860607, 1.29693,  0.242087, 0.485892, 0.331979,  0.757015, 0.410527,
        0.740235, 1.28431,  1.42516,  0.68281,  0.975185,  1.13892,  1.62237,
        0.344208, 0.561029, 0.363292, 0.911203, 0.272541,  0.419513, 0.342154,
        0.403335, 0.419286, 0.587321, 0.600655, 0.884853,  0.190907, 0.719914,
        0.346842, 0.598472
      ];
      test_util.expectArraysClose(result, expected);
    });

    it('Array3D is allowed', math => {
      const fSize = 2;
      const pad = 'same';
      const stride = 1;
      const chMul = 3;
      const inDepth = 2;

      const x = Array3D.zeros([3, 3, inDepth]);
      const w = Array4D.zeros([fSize, fSize, inDepth, chMul]);
      const result = math.depthwiseConv2D(x, w, stride, pad);
      expect(result.shape).toEqual([3, 3, inDepth * chMul]);
    });
  };

  test_util.describeMathCPU('depthwiseConv2D', [tests]);
  test_util.describeMathGPU('depthwiseConv2D', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
