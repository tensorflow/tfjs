
import {GPGPUProgram} from '../gpgpu_math';
import {getChannels, getSourceCoords} from '../packing_util';
import {getCoordsDataType} from '../shader_compiler';

export class UnpackColProgram implements GPGPUProgram {
  variableNames = ['A'];
  packedInputs = true;
  packedOutput = false;
  outputShape: number[];
  userCode: string;

  constructor(outputShape: number[]) {
    this.outputShape = outputShape;
    const rank = outputShape.length;

    const channels = getChannels('rc', rank);
    const dtype = getCoordsDataType(rank);
    const sourceCoords = getSourceCoords(rank, channels);

    this.userCode = `
      void main() {
        ${dtype} rc = getOutputCoords();
        vec4 packedInput = getA(${sourceCoords});
        int offset = imod(rc[${rank}-1], 4);
        float res = offset == 0 ? packedInput.x
        : offset == 1 ? packedInput.y
        : offset == 2 ? packedInput.z
        : packedInput.w;
        setOutput(res);
      }
    `;
  }
}
