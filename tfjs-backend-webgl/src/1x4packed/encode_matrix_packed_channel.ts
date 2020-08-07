import {getGlslDifferences} from '../glsl_version';
import {GPGPUProgram} from '../gpgpu_math';
import * as shader_util from '../shader_compiler_util';

export class EncodeMatrixColPackedProgram implements GPGPUProgram {
  variableNames = ['A'];
  userCode: string;
  outputShape: number[];
  packedInputs = false;
  packedOutput = true;
  packCol = true;
  packColInput = true;

  constructor(
      outputShape: [number, number, number], texShape: [number, number],
      inputIsUnsignedByte = false) {
    const glsl = getGlslDifferences();
    const [height, width] = texShape;
    this.outputShape = outputShape;

    let output = 'result';
    if (inputIsUnsignedByte) {
      output = 'floor(result * 255. + 0.5)';
    }

    let mainLoop = ``;

    for (let col = 0; col < 4; col++) {
      mainLoop += `
      if(localCoords[2] + ${col} < ${outputShape[2]}) {
      localCoords[2] += ${col};
      flatIndex = getFlatIndex(localCoords);
      offset = imod(flatIndex, 4);

      flatIndex = idiv(flatIndex, 4, 1.);

      r = flatIndex / ${width};
      c = imod(flatIndex, ${width});
      uv = (vec2(c, r) + halfCR) / vec2(${width}.0, ${height}.0);
      values = ${glsl.texture2D}(A, uv);
      if(offset == 0) {
        result[${col}] = values[0];
      } else if(offset == 1) {
        result[${col}] = values[1];
      } else if(offset == 2) {
        result[${col}] = values[2];
      } else {
        result[${col}] = values[3];
      }
    }
      `;
    }
    this.userCode = `
      ${shader_util.getFlatIndexFrom3D(outputShape)}

      void main() {
        ivec3 coords = getOutputCoords();

        vec4 result = vec4(0.);
        int flatIndex, r, c, offset;
        ivec3 localCoords;
        vec2 uv;
        vec4 values;
        localCoords = coords;
        ${mainLoop}
        ${glsl.output} = ${output};
      }
    `;
  }
}
