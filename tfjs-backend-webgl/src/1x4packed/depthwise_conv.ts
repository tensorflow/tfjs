import {backend_util} from '@tensorflow/tfjs-core';
import {GPGPUProgram} from '../gpgpu_math';

export class DepthwiseConvColPacked2DProgram implements GPGPUProgram {
  variableNames = ['x', 'W'];
  packedInputs = true;
  packedOutput = true;
  packCol = true;
  outputShape: number[];
  userCode: string;
  packColInput = true;

  constructor(
      convInfo: backend_util.Conv2DInfo, addBias = false,
      activation: string = null, hasPreluActivation = false) {
    this.outputShape = convInfo.outShape;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;

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

      const ivec2 pads = ivec2(${padTop}, ${padLeft});
      const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.w;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        vec4 dotProd = vec4(0.);

        vec4 xTexVal;

        for(int row = 0; row < ${filterHeight}; row++){
          int xR = xRCorner + row;

          if (xR < 0 || xR >= ${convInfo.inHeight}) {
            continue;
          }

          for(int col = 0; col < ${filterWidth}; col++){
            int xC = xCCorner + col;
            xTexVal = vec4(0,0,0,0);

            if(
            (xC >= ${convInfo.inWidth}) ||
            xC < 0 ||
            (xC + ${strideWidth} >= ${convInfo.inWidth})
            ){
              continue;
            }

            xTexVal = getX(batch, xR, xC, d1);

            vec4 wVal = getW(row, col, d1);
            dotProd += xTexVal * wVal;
          }
        }

        vec4 result = dotProd;
        ${addBiasSnippet}
        ${applyActivationSnippet}
        setOutput(result);
      }
    `;
  }
}
