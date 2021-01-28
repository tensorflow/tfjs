/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {getShapeCoords} from '../shader_preprocessor';
import {computeDispatch} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class CropAndResizeProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['Image', 'Boxes', 'BoxInd'];
  workGroupSize: [number, number, number] = [4, 4, 4];
  imageShape: [number, number, number, number];
  cropSize: [number, number];
  methodId: number;
  extrapolationValue: number;

  constructor(
      imageShape: [number, number, number, number], boxShape: [number, number],
      cropSize: [number, number], method: 'bilinear'|'nearest',
      extrapolationValue: number) {
    const [numBoxes, ] = boxShape;
    this.outputShape = [numBoxes, cropSize[0], cropSize[1], imageShape[3]];
    this.dispatchLayout = {x: [1, 2], y: [0], z: [3]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.shaderKey =
        `cropAndResize_${method}_${cropSize}_${extrapolationValue}`;
    this.imageShape = imageShape;
    this.cropSize = cropSize;
    this.methodId = method === 'bilinear' ? 1 : 0;
    this.extrapolationValue = extrapolationValue;
  }

  getUserCode(): string {
    const [batch, imageHeight, imageWidth, ] = this.imageShape;
    const [cropHeight, cropWidth] = this.cropSize;
    const [inputHeightFloat, inputWidthFloat] =
        [`${imageHeight - 1}.0`, `${imageWidth - 1}.0`];

    const [heightRatio, heightScale, inY] = cropHeight > 1 ?
        [
          `${(imageHeight - 1) / (cropHeight - 1)}`,
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
          `${(imageWidth - 1) / (cropWidth - 1)}`,
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
    const userCode = `
      const float height_ratio = float(${heightRatio});
      const float width_ratio = float(${widthRatio});
      void writeResult(ivec4 coords,float value) {
        if (coordsInBounds(coords, ${getShapeCoords(this.outputShape)})) {
          setOutput(coords[0], coords[1], coords[2], coords[3], value);
        }
      }
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
        int bInd = int(round(getBoxInd(b)));
        if(bInd < 0 || bInd >= ${batch}) {
          return;
        }
        float height_scale = ${heightScale};
        float width_scale = ${widthScale};
        float in_y = ${inY};
        if( in_y < 0.0 || in_y > ${inputHeightFloat} ) {
          writeResult(coords,float(${this.extrapolationValue}));
          return;
        }
        float in_x = ${inX};
        if( in_x < 0.0 || in_x > ${inputWidthFloat} ) {
          writeResult(coords,float(${this.extrapolationValue}));
          return;
        }
        vec2 sourceFracIndexCR = vec2(in_x,in_y);
        if(${this.methodId} == 1) {
          // Compute the four integer indices.
          ivec2 sourceFloorCR = ivec2(sourceFracIndexCR);
          ivec2 sourceCeilCR = ivec2(ceil(sourceFracIndexCR));
          float topLeft = getImage(bInd, sourceFloorCR.y, sourceFloorCR.x, d);
          float bottomLeft = getImage(bInd, sourceCeilCR.y, sourceFloorCR.x, d);
          float topRight = getImage(bInd, sourceFloorCR.y, sourceCeilCR.x, d);
          float bottomRight = getImage(bInd, sourceCeilCR.y, sourceCeilCR.x, d);
          vec2 fracCR = sourceFracIndexCR - vec2(sourceFloorCR);
          float top = topLeft + (topRight - topLeft) * fracCR.x;
          float bottom = bottomLeft + (bottomRight - bottomLeft) * fracCR.x;
          float newValue = top + (bottom - top) * fracCR.y;
          writeResult(coords,newValue);
        } else {
          // Compute the coordinators of nearest neighbor point.
          ivec2 sourceNearestCR = ivec2(floor(
            sourceFracIndexCR + vec2(0.5,0.5)));
          float newValue = getImage(
            bInd, sourceNearestCR.y, sourceNearestCR.x, d);
          writeResult(coords,newValue);
        }
      }
    `;
    return userCode;
  }
}
