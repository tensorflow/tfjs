import { serialization, Tensor,  ResizeBilinear} from '@tensorflow/tfjs-core';
import { getExactlyOneTensor } from '../../utils/types_utils';
import { crop_to_bounding_box } from './crop_to_bounding_box';
import {LayerArgs, Layer} from '../../engine/topology';
import { Kwargs } from '../../types';






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
    inputs = getExactlyOneTensor(inputs)
    const inputShape = inputs.shape
    let h_diff = inputShape[h_axis] - this.height
    let w_diff = inputShape[w_axis] - this.width



    function centercrop(){
      let h_start = K.cast(h_diff/2 , "float32") //issue with tensor rank
      let w_start = K.cast(w_diff/2, 'float32')

      return crop_to_bounding_box(inputs, h_start, w_start, this.height, width)
    }





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
