import {GPGPUProgram} from '../gpgpu_math';

export class Pack1x4Program implements GPGPUProgram {
  variableNames = ['A'];
  outputShape: number[];
  userCode: string;
  packedInputs = false;
  packColInput = false;
  packCol = true;
  packedOutput = true;

  constructor(outputShape: [number, number, number]) {
    this.outputShape = outputShape;
    if (outputShape[0] === 1 && outputShape[1] === 1 && outputShape[2] === 1) {
      this.userCode = `
        void main() {
          setOutput(vec4(getA(0,0), 0., 0., 0.));
        }
      `;
    } else {
      let paramStr = ``;
      if (outputShape[0] === 1 && outputShape[1] === 1) {
        paramStr = `rc.x`;
      } else if (outputShape[0] === 1) {
        paramStr = `rc.x, rc.y`;
      } else {
        paramStr = `rc.x, rc.y, rc.z`;
      }
      this.userCode = `
        void main() {
          ivec3 rc = getOutputCoords();
          vec4 result;
          result = vec4(
            getA(${paramStr}),
            rc.z+1 < ${outputShape[2] - 1} ? getA(${paramStr + '+1'}) : 0.,
            rc.z+2 < ${outputShape[2] - 1} ? getA(${paramStr + '+2'}) : 0.,
            rc.z+3 < ${outputShape[2] - 1} ? getA(${paramStr + '+3'}) : 0.
          );
          setOutput(result);
        }
      `;
    }
  }
}
