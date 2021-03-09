/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {KernelConfig, KernelFunc, NumericDataType, TensorInfo, Transform, TransformAttrs, TransformInputs, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

export function transform(args: {
  inputs: TransformInputs,
  attrs: TransformAttrs,
  backend: MathBackendCPU
}): TensorInfo {
  const {inputs, attrs, backend} = args;
  const {image, transforms} = inputs;
  const {interpolation, fillMode, fillValue, outputShape} = attrs;

  const [batch, imageHeight, imageWidth, numChannels] = image.shape;
  const [outHeight, outWidth] =
      outputShape != null ? outputShape : [imageHeight, imageWidth];
  const outShape = [batch, outHeight, outWidth, numChannels];

  const strides = util.computeStrides(image.shape);
  const batchStride = strides[0];
  const rowStride = strides[1];
  const colStride = strides[2];

  const outVals = util.getTypedArrayFromDType(
      image.dtype as NumericDataType, util.sizeFromShape(outShape));

  outVals.fill(fillValue);

  const imageVals = backend.data.get(image.dataId).values as TypedArray;
  const transformVals =
      backend.data.get(transforms.dataId).values as TypedArray;

  // Ref TF implementation:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/image/image_ops.h
  for (let b = 0; b < batch; ++b) {
    const transform = transforms.shape[0] === 1 ?
        transformVals :
        transformVals.subarray(b * 8, b * 8 + 8);

    for (let outY = 0; outY < outHeight; ++outY) {
      for (let outX = 0; outX < outWidth; ++outX) {
        for (let channel = 0; channel < numChannels; ++channel) {
          let val;

          const projection = transform[6] * outX + transform[7] * outY + 1;

          if (projection === 0) {
            // Return the fill value for infinite coordinates,
            // which are outside the input image
            continue;
          }

          const inX =
              (transform[0] * outX + transform[1] * outY + transform[2]) /
              projection;
          const inY =
              (transform[3] * outX + transform[4] * outY + transform[5]) /
              projection;

          const x = mapCoord(inX, imageWidth, fillMode);
          const y = mapCoord(inY, imageHeight, fillMode);

          switch (interpolation) {
            case 'nearest':
              val = nearestInterpolation(
                  imageVals, imageHeight, imageWidth, batchStride, rowStride,
                  colStride, b, y, x, channel, fillValue);
              break;
            case 'bilinear':
              val = bilinearInterpolation(
                  imageVals, imageHeight, imageWidth, batchStride, rowStride,
                  colStride, b, y, x, channel, fillValue);
              break;
            default:
              throw new Error(
                  `Error in Transform: Expect 'nearest' or ` +
                  `'bilinear', but got ${interpolation}`);
          }

          const ind =
              b * batchStride + outY * rowStride + outX * colStride + channel;

          outVals[ind] = val;
        }
      }
    }

    return backend.makeTensorInfo(outShape, image.dtype, outVals);
  }

  const dataId = backend.write(outVals, outShape, image.dtype);
  return {dataId, shape: image.shape, dtype: image.dtype};
}

export const transformConfig: KernelConfig = {
  kernelName: Transform,
  backendName: 'cpu',
  kernelFunc: transform as {} as KernelFunc
};

function mapCoord(
    outCoord: number, len: number,
    mode: 'constant'|'reflect'|'wrap'|'nearest') {
  switch (mode) {
    case 'reflect':
      return mapCoordReflect(outCoord, len);
    case 'wrap':
      return mapCoordWrap(outCoord, len);
    case 'nearest':
      return mapCoordNearest(outCoord, len);
    case 'constant':
    default:
      return mapCoordConstant(outCoord, len);
  }
}

function mapCoordReflect(outCoord: number, len: number): number {
  // Reflect [abcd] to [dcba|abcd|dcba].
  let inCoord = outCoord;
  if (inCoord < 0) {
    if (len <= 1) {
      inCoord = 0;
    } else {
      const sz2 = 2 * len;
      if (inCoord < sz2) {
        inCoord = sz2 * Math.trunc(-inCoord / sz2) + inCoord;
      }
      inCoord = inCoord < -len ? inCoord + sz2 : -inCoord - 1;
    }
  } else if (inCoord > len - 1) {
    if (len <= 1) {
      inCoord = 0;
    } else {
      const sz2 = 2 * len;
      inCoord -= sz2 * Math.trunc(inCoord / sz2);
      if (inCoord >= len) {
        inCoord = sz2 - inCoord - 1;
      }
    }
  }
  // clamp is necessary because when outCoord = 3.5 and len = 4,
  // inCoord = 3.5 and will be rounded to 4 in nearest interpolation.
  return util.clamp(0, inCoord, len - 1);
}

function mapCoordWrap(outCoord: number, len: number): number {
  // Wrap [abcd] to [abcd|abcd|abcd].
  let inCoord = outCoord;
  if (inCoord < 0) {
    if (len <= 1) {
      inCoord = 0;
    } else {
      const sz = len - 1;
      inCoord += len * (Math.trunc(-inCoord / sz) + 1);
    }
  } else if (inCoord > len - 1) {
    if (len <= 1) {
      inCoord = 0;
    } else {
      const sz = len - 1;
      inCoord -= len * Math.trunc(inCoord / sz);
    }
  }
  // clamp is necessary because when outCoord = -0.5 and len = 4,
  // inCoord = 3.5 and will be rounded to 4 in nearest interpolation.
  return util.clamp(0, inCoord, len - 1);
}

function mapCoordConstant(outCoord: number, len: number): number {
  return outCoord;
}

function mapCoordNearest(outCoord: number, len: number): number {
  return util.clamp(0, outCoord, len - 1);
}

function readWithFillValue(
    imageVals: TypedArray, imageHeight: number, imageWidth: number,
    batchStride: number, rowStride: number, colStride: number, batch: number,
    y: number, x: number, channel: number, fillValue: number): number {
  const ind = batch * batchStride + y * rowStride + x * colStride + channel;
  if (0 <= y && y < imageHeight && 0 <= x && x < imageWidth) {
    return imageVals[ind];
  } else {
    return fillValue;
  }
}

function nearestInterpolation(
    imageVals: TypedArray, imageHeight: number, imageWidth: number,
    batchStride: number, rowStride: number, colStride: number, batch: number,
    y: number, x: number, channel: number, fillValue: number): number {
  const $y = Math.round(y);
  const $x = Math.round(x);

  return readWithFillValue(
      imageVals, imageHeight, imageWidth, batchStride, rowStride, colStride,
      batch, $y, $x, channel, fillValue);
}

function bilinearInterpolation(
    imageVals: TypedArray, imageHeight: number, imageWidth: number,
    batchStride: number, rowStride: number, colStride: number, batch: number,
    y: number, x: number, channel: number, fillValue: number) {
  const yFloor = Math.floor(y);
  const xFloor = Math.floor(x);
  const yCeil = yFloor + 1;
  const xCeil = xFloor + 1;
  // f(x, yFloor) = (xCeil - x) / (xCeil - xFloor) * f(xFloor, yFloor)
  //               + (x - xFloor) / (xCeil - xFloor) * f(xCeil, yFloor)
  const valueYFloor =
      (xCeil - x) *
          readWithFillValue(
              imageVals, imageHeight, imageWidth, batchStride, rowStride,
              colStride, batch, yFloor, xFloor, channel, fillValue) +
      (x - xFloor) *
          readWithFillValue(
              imageVals, imageHeight, imageWidth, batchStride, rowStride,
              colStride, batch, yFloor, xCeil, channel, fillValue);
  // f(x, yCeil) = (xCeil - x) / (xCeil - xFloor) * f(xFloor, yCeil)
  //             + (x - xFloor) / (xCeil - xFloor) * f(xCeil, yCeil)
  const valueYCeil =
      (xCeil - x) *
          readWithFillValue(
              imageVals, imageHeight, imageWidth, batchStride, rowStride,
              colStride, batch, yCeil, xFloor, channel, fillValue) +
      (x - xFloor) *
          readWithFillValue(
              imageVals, imageHeight, imageWidth, batchStride, rowStride,
              colStride, batch, yCeil, xCeil, channel, fillValue);
  // f(x, y) = (yCeil - y) / (yCeil - yFloor) * f(x, yFloor)
  //         + (y - yFloor) / (yCeil - yFloor) * f(x, yCeil)
  return (yCeil - y) * valueYFloor + (y - yFloor) * valueYCeil;
}
