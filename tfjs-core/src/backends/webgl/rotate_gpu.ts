import {GPGPUProgram} from './gpgpu_math';

export class RotateProgram implements GPGPUProgram {
  variableNames = ['Image'];
  outputShape: number[] = [];
  userCode: string;

  constructor(
      imageShape: [number, number, number, number], radians: number,
      fillValue: number|[number, number, number]) {
    const imageHeight = imageShape[1];
    const imageWidth = imageShape[2];
    const sinFactor = Math.sin(radians);
    const cosFactor = Math.cos(radians);
    this.outputShape = imageShape;

    const halfWidth = (imageWidth / 2);
    const halfHeight = (imageHeight / 2);

    let fillSnippet = '';
    if (typeof fillValue === 'number') {
      fillSnippet = `float outputValue = ${fillValue.toFixed(2)};`;
    } else {
      fillSnippet = `
        vec3 fill = vec3(${fillValue.join(',')});
        float outputValue = fill[coords[3]];`;
    }

    this.userCode = `
        void main() {
          ivec4 coords = getOutputCoords();

          int x = coords[2];
          int y = coords[1];

          int coordX = int(float(x - ${halfWidth}) * ${cosFactor} - float(y - ${
        halfHeight}) * ${sinFactor});
          int coordY = int(float(x - ${halfWidth}) * ${sinFactor} + float(y - ${
        halfHeight}) * ${cosFactor});

          coordX = int(coordX + ${halfWidth});
          coordY = int(coordY + ${halfHeight});

          ${fillSnippet}

          if(coordX > 0 && coordX < ${imageWidth} && coordY > 0 && coordY < ${
        imageHeight}) {
            outputValue = getImage(coords[0], coordY, coordX, coords[3]);
          }
          setOutput(outputValue);
        }
    `;
  }
}
