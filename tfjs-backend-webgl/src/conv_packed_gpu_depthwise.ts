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

import {backend_util, util} from '@tensorflow/tfjs-core';

import {GPGPUProgram} from './gpgpu_math';

export class DepthwiseConvPacked2DProgram implements GPGPUProgram {
  variableNames = ['x', 'W'];
  packedInputs = true;
  packedOutput = true;
  outputShape: number[];
  userCode: string;

  constructor(
      convInfo: backend_util.Conv2DInfo, addBias = false,
      activation: string = null, hasPreluActivation = false,
      hasLeakyReluAlpha = false) {
    this.outputShape = convInfo.outShape;
    const channelMul = convInfo.outChannels / convInfo.inChannels;
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

    let mainLoop = `
      int xR; int xC; int xCOffset;
      vec4 wTexel; vec4 previous; vec4 final;`;

    for (let c = 0; c < filterWidth; c++) {
      mainLoop += `
          vec4 xTexelC${c * 2};
          int xTexelC${c * 2}Ready;
          vec4 xC${c};`;
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
      for (let c = 0; c < filterWidth; c++) {
        mainLoop += `
          xTexelC${c * 2} = vec4(0.0);
          xTexelC${c * 2}Ready = 0;
          xC${c} = vec4(0.0);`;
      }
      mainLoop += `
        xR = xRCorner + ${r * dilationHeight};
        if (xR >=0 && xR < ${xNumRows}) {
      `;

      for (let texelC = 0; texelC < (texelsAcross + 1) / 2; texelC++) {
        const colIndex = texelC * 2;
        const c = colIndex * dilationWidth;

        mainLoop += `
          xC = xCCorner + ${c};
          `;

        if (strideWidth === 1) {
          if (colIndex < filterWidth) {
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
                if (xCOffset >= 0 && xCOffset < ${xNumCols} && xTexelC${
                  c}Ready == 0) {
                  xTexelC${c} = getX(batch, xR, xCOffset, d1);

                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= ${xNumCols}) {
                    xTexelC${c}.zw = vec2(0.0);
                  }
                  xTexelC${c}Ready = 1;
                }
              `;
              // This texel has been read in previous iteration if the dilation
              // is 1.
              if (dilationWidth === 1 && c > 0) {
                mainLoop += `
                xC${colIndex} = vec4(xTexelC${c - 2}.zw, xTexelC${c}.xy);
                `;
              } else {
                mainLoop += `
                  xCOffset = xC + 1 - 2;

                  if (xCOffset >= 0 && xCOffset < ${xNumCols}) {
                    previous = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= ${xNumCols}) {
                      previous.zw = vec2(0.0);
                    }

                    xC${colIndex} = vec4(previous.zw, xTexelC${c}.xy);
                  } else {
                    xC${colIndex} = vec4(0.0, 0.0, xTexelC${c}.xy);
                  }
                  `;
              }
            } else {
              // Padding is even, so xRC corresponds to a single texel.
              mainLoop += `
                if (xC >= 0 && xC < ${xNumCols} && xTexelC${c}Ready == 0) {
                  xTexelC${c} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= ${xNumCols}) {
                    xTexelC${c}.zw = vec2(0.0);
                  }
                  xTexelC${c}Ready = 1;
                }

                xC${colIndex} = xTexelC${c};
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

                  if (xCOffset >= 0 && xCOffset < ${xNumCols} && xTexelC${
                    c + 2}Ready == 0) {
                    xTexelC${c + 2} = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= ${xNumCols}) {
                      xTexelC${c + 2}.zw = vec2(0.0);
                    }
                    xTexelC${c + 2}Ready = 1;
                  }
                  `;

                // If dilation > 1 then the xRC's will not be able to share any
                // values, so each xRC will require two unique calls to getX.
                if (dilationWidth > 1) {
                  mainLoop += `
                    xCOffset -= 2;
                    if (xCOffset >= 0 && xCOffset < ${xNumCols} && xTexelC${
                      c}Ready == 0) {
                      xTexelC${c} = getX(batch, xR, xCOffset, d1);
                      xTexelC${c}Ready = 1;
                    }
                    `;
                }

                mainLoop += `
                  xC${colIndex + 1} = vec4(xTexelC${c}.zw, xTexelC${c + 2}.xy);
                  `;
              } else {
                // If dilation is 1 and padding is odd, we have already read the
                // texel when constructing the previous x value. Here we can
                // simply skip the texture read.
                if (nextTexelOffset === 1) {
                  mainLoop += `
                    xC${colIndex + 1} = xTexelC${c};
                    `;
                } else {
                  mainLoop += `
                    xCOffset = xC + ${nextTexelOffset};

                    if (xCOffset >= 0 && xCOffset < ${xNumCols} && xTexelC${
                      c + 2}Ready == 0) {
                      xTexelC${c + 2} = getX(batch, xR, xCOffset, d1);
                      if (xCOffset + 1 >= ${xNumCols}) {
                        xTexelC${c + 2}.zw = vec2(0.0);
                      }
                      xTexelC${c + 2}Ready = 1;
                    }

                    xC${colIndex + 1} = xTexelC${c + 2};
                    `;
                }
              }
            }
          }
        } else {  // stride === 2
          if (c < filterWidth) {
            // Depending on whether padLeft is even or odd, we want either the
            // xy or zw channels from X texels for xC${colIndex}. If padLeft is
            // even, xC${colIndex +1} is simply the zw channels of texels we've
            // already sampled. But if padLeft is odd, xC{$c + 1}.zw will
            // need to come from the xy channels of a new texel, hence the `
            // vec4
            // final` initialized below.
            if (padLeft % 2 === 1) {
              mainLoop += `
                xCOffset = xC + 1 - ${strideWidth};
                if(xCOffset >= 0 && xCOffset < ${xNumCols} && xTexelC${
                  c}Ready == 0) {
                  xTexelC${c} = getX(batch, xR, xCOffset, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= ${xNumCols}) {
                    xTexelC${c}.zw = vec2(0.0);
                  }
                  xTexelC${c}Ready = 1;
                }

                if(xC + 1 >= 0 && xC + 1 < ${xNumCols} && xTexelC${
                  c + 2}Ready == 0) {
                  xTexelC${c + 2} = getX(batch, xR, xC + 1, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xC + 2 >= ${xNumCols}) {
                    xTexelC${c + 2}.zw = vec2(0.0);
                  }
                  xTexelC${c + 2}Ready = 1;
                }

                xC${colIndex} = vec4(xTexelC${c}.zw, xTexelC${c + 2}.zw);
              `;

              if (c + 1 < filterWidth) {
                mainLoop += `
                  final = vec4(0.0);
                  xCOffset = xC + 1 + ${strideWidth};
                  if(xCOffset >= 0 && xCOffset < ${xNumCols}) {
                    final = getX(batch, xR, xCOffset, d1);
                  }
                  xC${colIndex + 1} = vec4(xTexelC${c + 2}.xy, final.xy);
                `;
              }
            } else {
              mainLoop += `
                if(xC >= 0 && xC < ${xNumCols} && xTexelC${c}Ready == 0) {
                  xTexelC${c} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= ${xNumCols}) {
                    xTexelC${c}.zw = vec2(0.0);
                  }
                  xTexelC${c}Ready = 1;
                }

                xCOffset = xC + ${strideWidth};
                if(xCOffset >= 0 && xCOffset < ${xNumCols} && xTexelC${
                  c + 2}Ready == 0) {
                  xTexelC${c + 2} = getX(batch, xR, xCOffset, d1);
                  if (xCOffset + 1 >= ${xNumCols}) {
                    xTexelC${c + 2}.zw = vec2(0.);
                  }
                  xTexelC${c + 2}Ready = 1;
                }

                xC${colIndex} = vec4(
                  xTexelC${c}.xy, xTexelC${c + 2}.xy);
              `;

              if (c + 1 < filterWidth) {
                mainLoop += `
                  xC${colIndex + 1} = vec4(xTexelC${c}.zw, xTexelC${c + 2}.zw);
                `;
              }
            }
          }
        }

        // localize the dotProd accumulation within the loop, the theory is for
        // GPU with limited cache, accumulate sum across large amount of
        // veriables will cause lots of cache misses. (i.e. 5x5 filter will have
        // 50 variables)
        if (colIndex < filterWidth) {
          mainLoop += `
            wTexel = getW(${r}, ${c}, d1, q);
            dotProd += xC${colIndex} * vec4(wTexel.xz, wTexel.xz);
          `;

          if (c + 1 < filterWidth) {
            mainLoop += `
              wTexel = getW(${r}, ${c + 1}, d1, q);
              dotProd += xC${colIndex + 1} * vec4(wTexel.xz, wTexel.xz);
            `;
          }
        }
      }
      mainLoop += `
        }
      `;
    }

    let activationSnippet = '', applyActivationSnippet = '';
    if (activation) {
      if (hasPreluActivation) {
        activationSnippet = `vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${activation}
        }`;
      } else if (hasLeakyReluAlpha) {
        activationSnippet = `vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
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
    if (hasLeakyReluAlpha) {
      this.variableNames.push('leakyreluAlpha');
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
        int d1 = d2 / ${channelMul};
        int q = d2 - d1 * ${channelMul};
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
        vec4 dotProd = vec4(0.000000000000001);

        ${mainLoop}

        vec4 result = dotProd - vec4(0.000000000000001);
        ${addBiasSnippet}
        ${applyActivationSnippet}
        setOutput(result);
      }
    `;
  }
}
