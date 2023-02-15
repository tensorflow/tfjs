/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {serialization,DataType,unstack,stack,tensor,Tensor,Tensor1D,Tensor2D, Tensor3D, Tensor4D, tidy, range, image} from '@tensorflow/tfjs-core';
import {getExactlyOneShape, getExactlyOneTensor} from '../../utils/types_utils';
import {LayerArgs, Layer} from '../../engine/topology';
import {Kwargs} from '../../types';
import {Shape} from '../../keras_format/common';
import * as K from '../../backend/tfjs_backend';

const {resizeBilinear, cropAndResize} = image;

export declare interface CenterCropArgs extends LayerArgs{
  height: number;
  width: number;
}

export class CenterCrop extends Layer {
  /** @nocollapse */
  static className = 'CenterCrop';
  private readonly height: number;
  private readonly width: number;
  constructor(args: CenterCropArgs) {
    super(args);
    this.height = args.height;
    this.width = args.width;
  }

  centerCrop(inputs: Tensor3D | Tensor4D, hBuffer: number, wBuffer: number,
            height: number, width: number, inputHeight: number,
            inputWidth: number, dtype: DataType): Tensor | Tensor[] {

    return tidy(() => {
      let input: Tensor4D;
      let isRank3      = false;
      const top      = hBuffer / inputHeight;
      const left     = wBuffer / inputWidth;
      const bottom   = ((height) + hBuffer) / inputHeight;
      const right    = ((width) + wBuffer) / inputWidth;
      const bound    = [top, left, bottom, right];
      const boxesArr = [];

      if(inputs.rank === 3) {
        isRank3  = true;
        input  = stack([inputs]) as Tensor4D;
      } else {
        input = inputs as Tensor4D;
      }

      for (let i = 0; i < input.shape[0]; i++) {
        boxesArr.push(bound);
      }

      const boxes: Tensor2D  = tensor(boxesArr, [boxesArr.length, 4]);
      const boxInd: Tensor1D = range(0, boxesArr.length, 1, 'int32');

      const cropSize: [number, number] = [height, width];
      const cropped = cropAndResize(input, boxes, boxInd, cropSize, 'nearest');

      if(isRank3) {
        return K.cast(getExactlyOneTensor(unstack(cropped)), dtype);
      }
      return K.cast(cropped, dtype);
   });

  }

  upsize(inputs : Tensor3D | Tensor4D, height: number,
         width: number, dtype: DataType): Tensor | Tensor[] {

    return tidy(() => {
      const outputs = resizeBilinear(inputs, [height, width]);
      return K.cast(outputs, dtype);
  });

}

  override call(inputs: Tensor3D | Tensor4D , kwargs: Kwargs):
      Tensor[] | Tensor {
    return tidy(() => {
      const rankedInputs = getExactlyOneTensor(inputs) as Tensor3D | Tensor4D;
      const dtype       = rankedInputs.dtype;
      const inputShape  = rankedInputs.shape;
      const inputHeight = inputShape[inputShape.length - 3];
      const inputWidth  =  inputShape[inputShape.length - 2];

      let hBuffer = 0;
      if (inputHeight !== this.height) {
        hBuffer =  Math.floor((inputHeight - this.height) / 2);
      }

      let wBuffer = 0;
      if (inputWidth !== this.width) {
        wBuffer = Math.floor((inputWidth - this.width) / 2);

        if (wBuffer === 0) {
          wBuffer = 1;
        }
      }

      if(hBuffer >= 0 && wBuffer >= 0) {
        return this.centerCrop(rankedInputs, hBuffer, wBuffer,
                              this.height, this.width, inputHeight,
                              inputWidth, dtype);
      } else {
        return this.upsize(inputs, this.height, this.width, dtype);
      }
   });

  }

  override getConfig(): serialization.ConfigDict{

    const config: serialization.ConfigDict = {
      'height' : this.height,
      'width' : this.width
    };

    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  override computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[] {
    inputShape = getExactlyOneShape(inputShape);
    const hAxis = inputShape.length - 3;
    const wAxis = inputShape.length - 2;
    inputShape[hAxis] = this.height;
    inputShape[wAxis] = this.width;
    return inputShape;
  }
}

serialization.registerClass(CenterCrop);
