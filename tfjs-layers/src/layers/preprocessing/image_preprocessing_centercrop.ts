import {serialization, Tensor, Tensor3D, Tensor4D, tensor, Tensor2D, Tensor1D, stack, unstack} from '@tensorflow/tfjs-core';
import {getExactlyOneShape, getExactlyOneTensor} from '../../utils/types_utils';
import {resizeBilinear} from '@tensorflow/tfjs-core/ops/image/resize_bilinear';
import {cropAndResize} from '@tensorflow/tfjs-core/ops/image/crop_and_resize';
import {LayerArgs, Layer} from '../../engine/topology';
import {Kwargs} from '../../types';
import {Shape} from '../../keras_format/common';

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

  call(inputs: Tensor3D|Tensor4D , kwargs: Kwargs): Tensor[]|Tensor {
    let rankedInputs = getExactlyOneTensor(inputs) as Tensor3D|Tensor4D;
    const hAxis = rankedInputs.shape.length - 3;
    console.log(`hAxis: ${hAxis}`);
    const wAxis = rankedInputs.shape.length - 2;
    console.log(`wAxis: ${wAxis}`);
    const inputShape = rankedInputs.shape;
    const inputHeight = inputShape[hAxis];
    const inputWidth = inputShape[wAxis];

    let hBuffer : number;
    if (inputHeight !== this.height) {
      hBuffer =  Math.floor((inputHeight - this.height) / 2);
    } else {
      hBuffer = 0;
    }

    let wBuffer : number;
    if (inputWidth !== this.width) {
      wBuffer = Math.floor((inputWidth - this.width) / 2);
      if (wBuffer === 0) {
        wBuffer = 1;
      }
    } else {
      wBuffer = 0;
    }

    console.log(` hBuffer ${hBuffer}`)
    console.log(` wBuffer ${wBuffer}`)


    function centerCrop(hBuffer: number, wBuffer: number, height: number, width: number): Tensor|Tensor[] {
      console.log('TOP')
      let rank3: boolean = false
      const top = hBuffer / inputHeight;
      const left = wBuffer / inputWidth;
      const bottom = ((height) + hBuffer) / inputHeight;
      const right = ((width) + wBuffer) / inputWidth;
      const bound = [top, left, bottom, right];
      const boxesArr = [];

      if(rankedInputs.rank === 3) {
        rankedInputs = stack([rankedInputs]) as Tensor4D;
        rank3 = true
        console.log(`rankedShape ${rankedInputs.shape}`);
      }

      for (let i = 0; i < rankedInputs.shape[0]; i++) {
        boxesArr.push(bound);
      }

      const boxes = tensor(boxesArr, [boxesArr.length, 4]) as Tensor2D;
      console.log(`boxes.shape ${boxes.shape}`);
      const boxInd = tensor([...Array(boxesArr.length).keys()],
                            [boxesArr.length], 'int32') as Tensor1D;

      const cropSize: [number, number] = [height, width];


      rankedInputs = rankedInputs as Tensor4D;

      console.log(`Inputs ${rankedInputs.bufferSync().values}`);
      console.log(`Inputs shape ${rankedInputs.shape}`);
      console.log(`boxes ${boxes}`);
      console.log(`boxInd ${boxInd}`);
      console.log(`cropSize ${cropSize}`);
      const cropped = cropAndResize(rankedInputs, boxes, boxInd, cropSize, 'nearest');
      if(rank3) {
        return unstack(cropped)
      } else {
        return cropped
      }
    }

    function upsize(inputs : Tensor3D|Tensor4D, height: number, width: number): Tensor|Tensor[] {
      const outputs = resizeBilinear(rankedInputs, [height, width]);
      return outputs;
    }

    if(hBuffer >= 0 && wBuffer >= 0) {
      return centerCrop(hBuffer, wBuffer, this.height, this.width);
    } else {
      return upsize(inputs, this.height, this.width);
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
