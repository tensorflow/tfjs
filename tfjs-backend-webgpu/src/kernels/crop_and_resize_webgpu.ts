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

import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class CropAndResizeProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['Image', 'Boxes', 'BoxInd'];
  uniforms = 'float extrapolationValue;';
  workGroupSize: [number, number, number] = [64, 1, 1];
  methodId: number;
  cropHeightBiggerThan1: boolean;
  cropWidthBiggerThan1: boolean;

  constructor(
      channnel: number, boxShape: [number, number], cropSize: [number, number],
      method: 'bilinear'|'nearest') {
    const [numBoxes, ] = boxShape;
    this.outputShape = [numBoxes, cropSize[0], cropSize[1], channnel];
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.methodId = method === 'bilinear' ? 1 : 0;
    this.cropHeightBiggerThan1 = this.outputShape[1] > 1;
    this.cropWidthBiggerThan1 = this.outputShape[2] > 1;
    this.shaderKey = `cropAndResize_${this.methodId}_${
        this.cropHeightBiggerThan1}_${this.cropWidthBiggerThan1}`;
  }

  getUserCode(): string {
    const [inputHeightFloat, inputWidthFloat] =
        [`float(imageShape[1] - 1)`, `float(imageShape[2] - 1)`];

    const [heightRatio, heightScale, inY] = this.cropHeightBiggerThan1 ?
        [
          `(${inputHeightFloat} / float(outShape[1] - 1))`,
          '(y2-y1) * height_ratio',
          `y1*${inputHeightFloat} + float(y)*(height_scale)`,
        ] :
        [
          '0.0',
          '0.0',
          `0.5 * (y1+y2) * ${inputHeightFloat}`,
        ];
    const [widthRatio, widthScale, inX] = this.cropWidthBiggerThan1 ?
        [
          `(${inputWidthFloat} / float(outShape[2] - 1))`,
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
      void writeResult(ivec4 coords,float value) {
        if (coordsInBounds(coords, outShape)) {
          setOutput(coords[0], coords[1], coords[2], coords[3], value);
        }
      }
      void main() {
        const float height_ratio = float(${heightRatio});
        const float width_ratio = float(${widthRatio});
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
        if(bInd < 0 || bInd >= outShape[0]) {
          return;
        }
        float height_scale = ${heightScale};
        float width_scale = ${widthScale};
        float in_y = ${inY};
        if( in_y < 0.0 || in_y > ${inputHeightFloat} ) {
          writeResult(coords,extrapolationValue);
          return;
        }
        float in_x = ${inX};
        if( in_x < 0.0 || in_x > ${inputWidthFloat} ) {
          writeResult(coords,extrapolationValue);
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
