/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import { GPGPUProgram } from './gpgpu_math';

export class CropAndResizeProgram implements GPGPUProgram {
  variableNames = ['Image', 'Boxes', 'BoxInd'];
  outputShape: number[] = [];
  userCode: string;

  constructor(
    imageShape: [number, number, number, number], boxShape: [number, number],
    cropSize: [number, number], method: 'bilinear' | 'nearest',
    extrapolationValue: number) {
    const [batch, imageHeight, imageWidth, depth] = imageShape;
    const [numBoxes,] = boxShape;
    const [cropHeight, cropWidth] = cropSize;
    this.outputShape = [numBoxes, cropHeight, cropWidth, depth];
    const methodId = method === 'bilinear' ? 1 : 0;

    const [inputHeightFloat, inputWidthFloat] =
      [`${imageHeight - 1}.0`, `${imageWidth - 1}.0`];

    const [heightRatio, heightScale, inY] = cropHeight > 1 ?
      [
        `${(imageHeight-1)/(cropHeight-1)}`,
        '(y2-y1) * height_ratio',
        `y1*${inputHeightFloat} + float(y)*(height_scale)`,
      ] :
      [
        '0.0',
        '0.0',
        `0.5 * (y1+y2) * ${inputHeightFloat}`,
      ];
    const [widthRatio, widthScale, inX] = cropWidth > 1 ?
      [
        `${(imageWidth-1)/(cropWidth-1)}`,
        '(x2-x1) * width_ratio',
        `x1*${inputWidthFloat} + float(x)*(width_scale)`,
      ] :
      [
        '0.0',
        '0.0',
        `0.5 * (x1+x2) * ${inputWidthFloat}`,
      ];

    // Reference implementation
    // tslint:disable-next-line:max-line-length
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/crop_and_resize_op_gpu.cu.cc
    this.userCode = `
      const float height_ratio = float(${heightRatio});
      const float width_ratio = float(${widthRatio});
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int y = coords[1];
        int x = coords[2];
        int d = coords[3];

        // get box vals
        float y1 = getBoxes(b,0);
        float x1 = getBoxes(b,1);
        float y2 = getBoxes(b,2);
        float x2 = getBoxes(b,3);

        // get image in batch index
        int bInd = round(getBoxInd(b));
        if(bInd < 0 || bInd >= ${batch}) {
          return;
        }

        float height_scale = ${heightScale};
        float width_scale = ${widthScale};

        float in_y = ${inY};
        if( in_y < 0.0 || in_y > ${inputHeightFloat} ) {
          setOutput(float(${extrapolationValue}));
          return;
        }
        float in_x = ${inX};
        if( in_x < 0.0 || in_x > ${inputWidthFloat} ) {
          setOutput(float(${extrapolationValue}));
          return;
        }

        vec2 sourceFracIndexRC = vec2(in_y,in_x);
        if(${methodId} == 1) {
          // Compute the four integer indices.
          ivec2 sourceFloorRC = ivec2(sourceFracIndexRC);
          ivec2 sourceCeilRC = ivec2(ceil(sourceFracIndexRC));

          float topLeft = getImage(b, sourceFloorRC.x, sourceFloorRC.y, d);
          float bottomLeft = getImage(b, sourceCeilRC.x, sourceFloorRC.y, d);
          float topRight = getImage(b, sourceFloorRC.x, sourceCeilRC.y, d);
          float bottomRight = getImage(b, sourceCeilRC.x, sourceCeilRC.y, d);

          vec2 fracRC = sourceFracIndexRC - vec2(sourceFloorRC);

          float top = topLeft + (topRight - topLeft) * fracRC.y;
          float bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;
          float newValue = top + (bottom - top) * fracRC.x;
          setOutput(newValue);
        } else {
          // Compute the coordinators of nearest neighbor point.
          ivec2 sourceNearestRC = ivec2(floor(
            sourceFracIndexRC + vec2(0.5,0.5)));
          float newValue = getImage(b, sourceNearestRC.x, sourceNearestRC.y, d);
          setOutput(newValue);
        }
      }
    `;
  }
}
