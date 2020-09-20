import {GPGPUProgram} from './gpgpu_math';
import {getCoordsDataType} from './shader_compiler';

export class MirrorPadProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  userCode: string;

  constructor(
      xShape: number[], paddings: Array<[number, number]>,
      mode: 'reflect'|'symmetric') {
    this.outputShape = paddings.map(
        (p, i) => p[0] /* beforePad */ + xShape[i] + p[1] /* afterPad */);
    const rank = xShape.length;
    const type = getCoordsDataType(rank);

    const start = paddings.map(p => p[0]).join(',');
    const end = paddings.map((p, i) => p[0] + xShape[i]).join(',');
    const unpackedCoords = rank === 1 ?
        'coords' :
        ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank);
    const offset = mode === 'reflect' ? 0 : 1;
    const lessThan = rank === 1 ? 'coords < start' : 'lessThan(coords, start)';
    const greatherThanEqual = rank === 1 ?
            'coords >= end' : 'greaterThanEqual(coords, end)';
    this.userCode = `
      ${type} start = ${type}(${start});
      ${type} end = ${type}(${end});
      void main() {
        ${type} coords = getOutputCoords();
        ${type} lt = ${type}(${lessThan});
        ${type} gte = ${type}(${greatherThanEqual});
        ${type} orig = 1 - (lt + gte);
        coords = orig * coords +
                 lt * (start * 2 - coords - ${offset}) +
                 gte * ((end - 1) * 2 - coords + ${offset});
        coords -= start;
        setOutput(getX(${unpackedCoords}));
      }
    `;
  }
}
