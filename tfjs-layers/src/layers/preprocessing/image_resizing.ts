/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {image, Rank, serialization, Tensor, tidy} from '@tensorflow/tfjs-core';  // mul, add

import {Layer, LayerArgs} from '../../engine/topology';
import {ValueError} from '../../errors';
import {Shape} from '../../keras_format/common';
import {Kwargs} from '../../types';
import {getExactlyOneShape} from '../../utils/types_utils';  //, getExactlyOneTensor

// tf methods unimplemented in tfjs: 'bicubic', 'area', 'lanczos3', 'lanczos5',
//                                   'gaussian', 'mitchellcubic'
const validInterpolationMethods = ['bilinear', 'nearest'];

export declare interface ResizingArgs extends LayerArgs {
  height: number;
  width: number;
  interpolation?: string;       // default = 'bilinear';
  cropToAspectRatio?: boolean;  // default = false;
}

/**
 * Preprocessing Resizing Layer
 *
 * This resizes images by a scaling and offset factor
 */

export class Resizing extends Layer {
  /** @nocollapse */
  static className = 'Resizing';
  private readonly height: number;
  private readonly width: number;
  // method of interpolation to be used; default = "bilinear";
  private readonly interpolation: string;
  // toggle whether the aspect ratio should be preserved; default = false;
  private readonly cropToAspectRatio: boolean;

  constructor(args: ResizingArgs) {
    super(args);

    this.height = args.height;
    this.width = args.width;

    if (args.interpolation) {
      if (validInterpolationMethods.includes(args.interpolation)) {
        this.interpolation = args.interpolation;
      } else {
        throw new ValueError(`Invalid interpolation parameter: ${
            args.interpolation} is not implemented`);
      }
    } else {
      this.interpolation = 'bilinear';
    }
    if (args.cropToAspectRatio) {
      this.cropToAspectRatio = args.cropToAspectRatio;
    } else {
      this.cropToAspectRatio = false;
    }
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);
    const numChannels = inputShape[2];
    const outputShape = [];
    outputShape.push(this.height);
    outputShape.push(this.width);
    outputShape.push(numChannels);
    return outputShape;
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      'height': this.height,
      'width': this.width,
      'interpolation': this.interpolation,
      'cropToAspectRatio': this.cropToAspectRatio
    };

    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  call(inputs: Tensor<Rank.R3>|Tensor<Rank.R4>, kwargs: Kwargs):
      Tensor[]|Tensor {
    return tidy(() => {
      const size: [number, number] = [this.height, this.width];
      if (this.interpolation === 'bilinear') {
        return image.resizeBilinear(inputs, size, !this.cropToAspectRatio);
      } else if (this.interpolation === 'nearest') {
        return image.resizeNearestNeighbor(
            inputs, size, !this.cropToAspectRatio);
      } else {
        throw new Error('interpolation error encountered in image_resizing.ts');
      }
    });
  }
}

serialization.registerClass(Resizing);
