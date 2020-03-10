/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {Conv2DInfo} from '../../ops/conv_util';
import * as util from '../../util';

import {GPGPUProgram} from './gpgpu_math';

export class DepthwiseConvPacked2DProgram implements GPGPUProgram {
  variableNames = ['x', 'W'];
  packedInputs = true;
  packedOutput = true;
  outputShape: number[];
  userCode: string;

  constructor(
      convInfo: Conv2DInfo, addBias = false, activation: string = null,
      hasPreluActivation = false) {
    this.outputShape = convInfo.outShape;

    const xNumRows = convInfo.inHeight;
    const xNumCols = convInfo.inWidth;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const texelsAcross = filterWidth;

    let mainLoop = `int xR; int xC; int xCOffset;`;

    for (let r = 0; r < filterHeight; r++) {
      for (let c = 0; c < filterWidth; c++) {
        mainLoop += `
          vec4 xTexelR${r}C${c * 2} = vec4(0.);
          vec4 wR${r}C${c} = vec4(0.);
          vec4 xR${r}C${c} = vec4(0.);`;
      }
    }

    /**
     * This vectorized implementation works by gathering the values needed for
     * each output channel's dot product into vec4's and then multiplying them
     * all together (this happens in the final double for-loop below). Most of
     * the main loop consists of constructing these vec4's with the minimum
     * number of texture2D calls, which means making use of all four returned
     * values from a texture2D call at once.
     */
    for (let r = 0; r < filterHeight; r++) {
      for (let texelC = 0; texelC < texelsAcross; texelC++) {
        const c = texelC * 2;

        mainLoop += `
          xR = xRCorner + ${r * dilationHeight};
          xC = xCCorner + ${c * dilationWidth};
        `;

        if (strideWidth === 1) {
          if (c < filterWidth) {
            // If padding is odd, the outer texels have to be composed.
            if (padLeft % 2 === 1) {
              // TODO: Ensure vec4 previous does not result in redundant sample,
              // and avoid setting xTexelRC's that exceed the boundary in the
              // first place rather than resetting them to vec4(0)).

              // To compute xCOffset:
              // - If padding is odd, we must add 1 to ensure we ask for an
              // even-numbered row.
              // - We subtract 2 to access the previous texel.

              mainLoop += `
                xCOffset = xC + 1;
                if(xR >= 0 && xR < ${xNumRows} && xCOffset >= 0 && xCOffset < ${
                  xNumCols}) {
                  xTexelR${r}C${c} = getX(batch, xR, xCOffset, d1);

                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if(xCOffset + 1 >= ${xNumCols}) {
                    xTexelR${r}C${c}.zw = vec2(0.);
                  }
                } else {
                  xTexelR${r}C${c} = vec4(0.);
                }

                xCOffset = xC + 1 - 2;
                if(xR >= 0 && xR < ${xNumRows} && xCOffset >= 0 && xCOffset < ${
                  xNumCols}) {
                  vec4 previous = getX(batch, xR, xCOffset, d1);

                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if(xCOffset + 1 >= ${xNumCols}) {
                    previous.zw = vec2(0.);
                  }

                  xR${r}C${c} = vec4(previous.zw, xTexelR${r}C${c}.xy);
                } else {
                  xR${r}C${c} = vec4(0, 0, xTexelR${r}C${c}.xy);
                }
              `;
            } else {
              // Padding is even, so xRC corresponds to a single texel.
              mainLoop += `
                if(xR >= 0 && xR < ${xNumRows} && xC >= 0 && xC < ${xNumCols}) {
                  xTexelR${r}C${c} = getX(batch, xR, xC, d1);
                } else {
                  xTexelR${r}C${c} = vec4(0.);
                }

                xR${r}C${c} = xTexelR${r}C${c};
              `;
            }

            if (c + 1 < filterWidth) {
              // If dilation is even, the second entry should match the first
              // (either both are composed or both are single samples). But if
              // dilation is odd, then the second entry should be the opposite
              // of the first (if the first is composed, the second is a single
              // sample, and vice versa.)

              const nextTexelOffset = padLeft % 2 === 0 ?
                  util.nearestLargerEven(dilationWidth) :
                  dilationWidth;

              if ((dilationWidth % 2 === 0 && padLeft % 2 === 1) ||
                  (dilationWidth % 2 !== 0 && padLeft % 2 !== 1)) {
                mainLoop += `
                  xCOffset = xC + ${padLeft % 2} + ${nextTexelOffset};

                  if(xR >= 0 && xR < ${xNumRows} &&
                    xCOffset >= 0 && xCOffset < ${xNumCols}) {
                    xTexelR${r}C${c + 2} = getX(batch, xR, xCOffset, d1);
                  }
                `;

                // If dilation > 1 then the xRC's will not be able to share any
                // values, so each xRC will require two unique calls to getX.
                if (dilationWidth > 1) {
                  mainLoop += `
                    xCOffset -= 2;
                    if(xR >= 0 && xR < ${xNumRows} &&
                      xCOffset >= 0 && xCOffset < ${xNumCols}) {
                      xTexelR${r}C${c} = getX(batch, xR, xCOffset, d1);
                    } else {
                      xTexelR${r}C${c} = vec4(0.);
                    }
                  `;
                }

                mainLoop += `
                  xR${r}C${c + 1} = vec4(
                    xTexelR${r}C${c}.zw, xTexelR${r}C${c + 2}.xy);
                `;
              } else {
                mainLoop += `
                  xCOffset = xC + ${nextTexelOffset};

                  if(xR >= 0 && xR < ${xNumRows} &&
                    xCOffset >= 0 && xCOffset < ${xNumCols}) {
                    xTexelR${r}C${c + 2} = getX(batch, xR, xCOffset, d1);
                  }

                  xR${r}C${c + 1} = xTexelR${r}C${c + 2};
                `;
              }
            }
          }
        } else {  // stride > 1
          if (c < filterWidth) {
            mainLoop += `
              if(xR >= 0 && xR < ${xNumRows}) {
            `;

            // Depending on whether padLeft is even or odd, we want either the
            // xy or zw channels from X texels for xR${r}C${c}. If padLeft is
            // even, xR${r}C${c + 1} is simply the zw channels of texels we've
            // already sampled. But if padLeft is odd, xR${r}C{$c + 1}.zw will
            // need to come from the xy channels of a new texel, hence the `vec4
            // final` initialized below.
            if (padLeft % 2 === 1) {
              mainLoop += `
                xCOffset = xC + 1 - ${strideWidth};
                if(xCOffset >= 0 && xCOffset < ${xNumCols}) {
                  xTexelR${r}C${c} = getX(batch, xR, xCOffset, d1);
                } else {
                  xTexelR${r}C${c} = vec4(0.);
                }

                if(xC + 1 >= 0 && xC + 1 < ${xNumCols}) {
                  xTexelR${r}C${c + 2} = getX(batch, xR, xC + 1, d1);
                } else {
                  xTexelR${r}C${c + 2} = vec4(0.);
                }

                xR${r}C${c} = vec4(
                  xTexelR${r}C${c}.zw, xTexelR${r}C${c + 2}.zw);
              `;

              if (c + 1 < filterWidth) {
                mainLoop += `
                  vec4 final = vec4(0.);
                  xCOffset = xC + 1 + ${strideWidth};
                  if(xCOffset >= 0 && xCOffset < ${xNumCols}) {
                    final = getX(batch, xR, xCOffset, d1);
                  }
                  xR${r}C${c + 1} = vec4(xTexelR${r}C${c + 2}.xy, final.xy);
                `;
              }
            } else {
              mainLoop += `
                if(xC >= 0 && xC < ${xNumCols}) {
                  xTexelR${r}C${c} = getX(batch, xR, xC, d1);
                } else {
                  xTexelR${r}C${c} = vec4(0.);
                }

                xCOffset = xC + ${strideWidth};
                if(xCOffset >= 0 && xCOffset < ${xNumCols}) {
                  xTexelR${r}C${c + 2} = getX(batch, xR, xCOffset, d1);
                } else {
                  xTexelR${r}C${c + 2} = vec4(0.);
                }

                xR${r}C${c} = vec4(
                  xTexelR${r}C${c}.xy, xTexelR${r}C${c + 2}.xy);
              `;

              if (c + 1 < filterWidth) {
                mainLoop += `
                  xR${r}C${c + 1} = vec4(
                    xTexelR${r}C${c}.zw, xTexelR${r}C${c + 2}.zw);
                `;
              }
            }

            mainLoop += `}`;
          }
        }

        if (c < filterWidth) {
          mainLoop += `
            vec4 wTexelR${r}C${c} = getW(${r}, ${c}, d1, q);
            wR${r}C${c} = vec4(wTexelR${r}C${c}.xz, wTexelR${r}C${c}.xz);
          `;

          if (c + 1 < filterWidth) {
            mainLoop += `
              vec4 wTexelR${r}C${c + 1} = getW(${r}, ${c + 1}, d1, q);
              wR${r}C${c + 1} =
                vec4(wTexelR${r}C${c + 1}.xz, wTexelR${r}C${c + 1}.xz);`;
          }
        }
      }
    }

    for (let r = 0; r < filterHeight; r++) {
      for (let c = 0; c < filterWidth; c++) {
        mainLoop += `dotProd += xR${r}C${c} * wR${r}C${c};`;
      }
    }

    let activationSnippet = '', applyActivationSnippet = '';
    if (activation) {
      if (hasPreluActivation) {
        activationSnippet = `vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${activation}
        }`;
      } else {
        activationSnippet = `vec4 activation(vec4 x) {
          ${activation}
        }`;
      }

      applyActivationSnippet = `result = activation(result);`;
    }

    const addBiasSnippet = addBias ? 'result += getBiasAtOutCoords();' : '';
    if (addBias) {
      this.variableNames.push('bias');
    }

    if (hasPreluActivation) {
      this.variableNames.push('preluActivationWeights');
    }

    this.userCode = `
      ${activationSnippet}

      const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
      const ivec2 pads = ivec2(${padTop}, ${padLeft});

      void main() {

        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2;
        int q = 0;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        vec4 dotProd = vec4(0.);

        ${mainLoop}

        vec4 result = dotProd;
        ${addBiasSnippet}
        ${applyActivationSnippet}
        setOutput(result);
      }
    `;
  }
}
