import {serialization, Tensor, Tensor3D, Tensor4D, tensor, Tensor2D, Tensor1D, stack, unstack, DataType} from '@tensorflow/tfjs-core';
import {getExactlyOneShape, getExactlyOneTensor} from '../../utils/types_utils';
import {resizeBilinear} from '@tensorflow/tfjs-core/ops/image/resize_bilinear';
import {cropAndResize} from '@tensorflow/tfjs-core/ops/image/crop_and_resize';
import {LayerArgs, Layer} from '../../engine/topology';
import {Kwargs} from '../../types';
import {Shape} from '../../keras_format/common';
import * as K from '../../backend/tfjs_backend';

export declare interface CenterCropArgs extends LayerArgs{
  height: number;
  width: number;
}

export class CenterCrop extends Layer {
  static className = 'CenterCrop';
  private readonly height: number;
  private readonly width: number;
  constructor(args: CenterCropArgs) {
    super(args);
    this.height = args.height;
    this.width = args.width;
  }

  centerCrop(inputs: Tensor3D|Tensor4D, hBuffer: number, wBuffer: number,
            height: number, width: number, inputHeight: number,
            inputWidth:number, dtype: DataType): Tensor|Tensor[] {

    let rank3      = false;
    const top      = hBuffer / inputHeight;
    const left     = wBuffer / inputWidth;
    const bottom   = ((height) + hBuffer) / inputHeight;
    const right    = ((width) + wBuffer) / inputWidth;
    const bound    = [top, left, bottom, right];
    const boxesArr = [];

    if(inputs.rank === 3) {
      rank3  = true;
      inputs = stack([inputs]) as Tensor4D;
    }

    for (let i = 0; i < inputs.shape[0]; i++) {
      boxesArr.push(bound);
    }

    const boxes: Tensor2D  = tensor(boxesArr, [boxesArr.length, 4]);
    const boxInd: Tensor1D = tensor([...Array(boxesArr.length).keys()],
                          [boxesArr.length], 'int32');

    const cropSize: [number, number] = [height, width];
    inputs = inputs as Tensor4D;
    const cropped = cropAndResize(inputs, boxes, boxInd, cropSize, 'nearest');

    if(rank3) {
      return K.cast(getExactlyOneTensor(unstack(cropped)), dtype);
    }
      return K.cast(cropped, dtype);
  }

  upsize(inputs : Tensor3D|Tensor4D, height: number,
         width: number, dtype: DataType): Tensor|Tensor[] {

    const outputs = resizeBilinear(inputs, [height, width]);
    return K.cast(outputs, dtype);
}

  call(inputs: Tensor3D|Tensor4D , kwargs: Kwargs): Tensor[]|Tensor {
    const rankedInputs = getExactlyOneTensor(inputs) as Tensor3D|Tensor4D;
    const dtype = rankedInputs.dtype;
    const hAxis = rankedInputs.shape.length - 3;
    const wAxis = rankedInputs.shape.length - 2;
    const inputShape = rankedInputs.shape;
    const inputHeight = inputShape[hAxis];
    const inputWidth = inputShape[wAxis];

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
  }

  getConfig(): serialization.ConfigDict{

    const config: serialization.ConfigDict = {
      'height' : this.height,
      'width' : this.width
    };

    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);
    const hAxis = inputShape.length - 3;
    const wAxis = inputShape.length - 2;
    inputShape[hAxis] = this.height;
    inputShape[wAxis] = this.width;
    return inputShape;
  }
}

serialization.registerClass(CenterCrop);
