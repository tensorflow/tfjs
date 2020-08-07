import {getGlslDifferences} from '../glsl_version';
import {GPGPUProgram} from '../gpgpu_math';
import {getChannels} from '../packing_util';
import {getCoordsDataType} from '../shader_compiler';

export class Packed2x2To1x4 implements GPGPUProgram {
  variableNames = ['A'];
  userCode: string;
  outputShape: number[];
  packedInputs = true;
  packedOutput = true;
  packColInput = false;
  packCol = true;

  constructor(outputShape: number[]) {
    const glsl = getGlslDifferences();
    this.outputShape = outputShape;
    const channelDim = outputShape.length - 1;

    const rank = outputShape.length;

    const channels = getChannels('coords', rank);
    const dtype = getCoordsDataType(rank);

    const channelsBound = channels.slice(0);
    channelsBound[channelsBound.length - 1] =
        channelsBound[channelsBound.length - 1] + '+ 2';
    this.userCode = `
      void main() {
        ${dtype} coords = getOutputCoords();
        vec4 result = vec4(
          getA(${channels.join(',')}).xy,
          0,
          0
        );
        if( ${channelsBound[channelsBound.length - 1]} < ${
        outputShape[channelDim]}){
          result.zw = getA(${channelsBound}).xy;
        }
        ${glsl.output} = result;
      }
    `;
  }
}
