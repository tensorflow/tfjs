import { serialization, Tensor} from '@tensorflow/tfjs-core';

import {LayerArgs, Layer} from '../../engine/topology';
import { Kwargs } from '../../types';

import {cropAndResize} from '../../tfjs-backend-cpu'
import * as K from '../../backend/tfjs_backend';

const h_axis = -3
const w_axis = -2

export declare interface CenterCropArgs extends LayerArgs{
  height: number;
  width: number;
}

export class CenterCrop extends Layer {

  private readonly height: number;
  private readonly width: number;
  constructor(args: CenterCropArgs) {
    super(args);
  }

  call(inputs: Tensor, kwargs: Kwargs): Tensor[]|Tensor{
    const inputShape = inputs.shape
    let h_diff = inputShape[h_axis] - this.height
    let w_diff = inputShape[w_axis] - this.width


    cropAndResize

    return inputs
  }



  get_config(): serialization.ConfigDict{
    const config: serialization.ConfigDict = {
      'height' : this.height,
      'width' : this.width
    };

    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config
  }
}



//  serialization.registerClass(CenterCrop)
