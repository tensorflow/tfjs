import {GPGPUProgram} from './gpgpu_math';

export class RotateProgram implements GPGPUProgram {
  variableNames = ['Image'];
  outputShape: number[] = [];
  userCode: string;

  constructor(
      imageShape: [number, number, number, number], radians: number,
      fillValue: number) {
    const imageHeight = imageShape[1];
    const imageWidth = imageShape[2];
    const sinFactor = Math.sin(radians);
    const cosFactor = Math.cos(radians);
    const ratio = imageWidth / imageHeight;

    this.userCode = `
        void main() {
          ivec4 coords = getOutputCoords();

          int x = coords[2];
          int y = coords[1];

          float coordX = (x - ${imageWidth / 2}) * ${cosFactor} - (y - ${
        imageHeight / 2}) * ${sinFactor};
          float coordY = (x - ${imageWidth / 2}) * ${sinFactor} + (y - ${
        imageHeight / 2}) * ${cosFactor};

          coordX = round(coordX + ${imageWidth / 2});
          coordY = round((coordY + ${imageHeight / 2}) * ${ratio});

          float outputValue = ${fillValue};
          if(coordX > 0 && coordX < ${imageWidth} && coordY > 0 && coordY < ${
        imageHeight}) {
            outputValue = getImage(coords[0], coordY, coordX, coords[3]);
          }
          setOutput(outputValue);
        }
    `;
  }
}
