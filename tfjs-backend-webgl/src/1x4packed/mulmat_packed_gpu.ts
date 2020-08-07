import {GPGPUProgram} from '../gpgpu_math';

export class MatMulColPackedProgram implements GPGPUProgram {
  variableNames = ['matrixA', 'matrixB'];
  packedInputs = true;
  packedOutput = true;
  packCol = true;
  packColInput = true;
  outputShape: number[];
  userCode: string;

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      transposeA = false, transposeB = false, addBias = false,
      activation: string = null, hasPreluActivation = false) {
    this.outputShape = outputShape;

    const sharedDim = aShape[2];
    const sharedDimensionPacked = Math.ceil(sharedDim / 4);

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

      void main() {
        ivec3 rc = getOutputCoords();
        vec4 result = vec4(0);

        for(int i = 0 ; i < ${sharedDimensionPacked}; i++){
          vec4 aVal = getMatrixA(rc.x, rc.y, i);
          vec4 bVal1 = getMatrixB(rc.x, i * 4, rc.z);
          result += aVal.x * bVal1;

          vec4 bVal2 = getMatrixB(rc.x, i * 4 + 1, rc.z);
          result += aVal.y * bVal2;

          vec4 bVal3 = getMatrixB(rc.x, i * 4 + 2, rc.z);
          result += aVal.z * bVal3;

          vec4 bVal4 = getMatrixB(rc.x, i * 4 + 3, rc.z);
          result += aVal.w * bVal4;
        }

        ${addBiasSnippet}

        ${applyActivationSnippet}

        setOutput(result);
      }
    `;
  }
}
