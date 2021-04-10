/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import { Tensor1D, Tensor3D } from '../../tensor';
import { tensor1d } from '../tensor1d';
import { TensorLike } from '../../types';
import { op } from '../operation';
import { cast } from '../cast';
import { split } from '../split';
import { bincount } from '../bincount';
import { lessEqual } from '../less_equal';
import { greater } from '../greater';
import { sum } from '../sum';
import { add } from '../add';
import { fill } from '../fill';
import { range } from '../range';
import { tensor } from '../tensor';
import * as util from '../../util';
import { convertToTensor } from '../../tensor_util_env';

/**
 * Performs image binarization with corresponding threshold
 * (depends on the method)value, which creates a binary image from a grayscale.
 * @param image 4d tensor of shape [imageHeight,imageWidth, depth],
 * where imageHeight and imageWidth must be positive.The image color
 * range should be [0, 255].
 * @param method Optional string from `'binary' | 'otsu'`
 *  which specifies the method for thresholding. Defaults to 'binary'.
 * @param inverted Optional boolean whichspecifies
 *  if colours should be inverted. Defaults to false.
 * @param threshValue Optional number which defines threshold value from 0 to 1.
 *  Defaults to 0.5.
 * @return A 3d tensor of shape [imageHeight,imageWidth, depth], which 
 * contains binarized image.
 */

function threshold_(
    image: Tensor3D | TensorLike,
    method = 'binary',
    inverted = false,
    threshValue =  0.5
): Tensor3D {
    const $image = convertToTensor(image, 'image', 'threshold');
    const redIntencityCoef = 0.2126;
    const greenIntencityCoef = 0.7152;
    const blueIntencityCoef = 0.0722;
    const totalPixelsInImage = $image.shape[0] * $image.shape[1];

    let $threshold = tensor1d([threshValue]).mul(255);
    let r, g, b, grayscale;

    util.assert(
        $image.rank === 3,
        () => 'Error in threshold: image must be rank 3,' +
            `but got rank ${$image.rank}.`);

    util.assert(
        $image.dtype === 'int32' || $image.dtype === 'float32',
        () => 'Error in dtype: image dtype must be int32 or float32,' +
            `but got dtype ${$image.dtype}.`);

    util.assert(
        method === 'otsu' || method === 'binary',
        () => `Method must be binary or otsu, but was ${method}`);

    if ($image.shape[2] === 3) {
        [r, g, b] = split($image, [1, 1, 1], -1);
        grayscale = r.mul(redIntencityCoef)
            .add(g.mul(greenIntencityCoef))
            .add(b.mul(blueIntencityCoef));
    } else {
        grayscale = image;
    }

    if (method === 'otsu') {
        const $histogram = bincount(
            cast(grayscale.round(), 'int32'),
            tensor([]),
            256
        );
        $threshold = otsu($histogram, totalPixelsInImage);
    }

    const invCondition = inverted ?
        lessEqual(grayscale, $threshold) : greater(grayscale, $threshold);

    const result = cast(invCondition.mul(255), 'int32');
    return result as Tensor3D;
}

function otsu(histogram: Tensor1D, total: number) {

    let bestThreshold = tensor([-1]);
    let bestInBetweenVariance = tensor([0]);
    let curInBetweenVariance = tensor([0]);
    let classFirst, classSecond, meanFirst,
        meanSecond, weightForeground, weightBackground;

    for (let index = 0; index < histogram.shape[0]; index++) {
        if (index !== histogram.shape[0] - 1) {

            classFirst = histogram.slice(0, index + 1);

            classSecond = histogram.slice(index + 1);

            weightForeground = sum(classFirst).div(total);

            weightBackground = sum(classSecond).div(total);

            meanFirst = sum(classFirst.mul(range(0, classFirst.shape[0])))
                .div(classFirst.sum());

            meanSecond = sum(classSecond.mul(
                add(range(0, classSecond.shape[0]),
                    fill(classSecond.shape, classFirst.shape[0]))))
                .div(classSecond.sum());

            curInBetweenVariance = (weightForeground.mul(weightBackground)
                .mul(meanFirst.sub(meanSecond)).mul(meanFirst.sub(meanSecond)))
                .reshape([1]);

            const condition = curInBetweenVariance
                .greater(bestInBetweenVariance);

            bestInBetweenVariance = curInBetweenVariance
                .where(condition, bestInBetweenVariance);

            bestThreshold = tensor1d([index]).where(condition, bestThreshold);

        }
    }
    return bestThreshold;
}

export const threshold = op({ threshold_ });
