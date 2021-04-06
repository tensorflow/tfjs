import { Tensor3D } from '../../tensor';
import { TensorLike } from '../../types';
import { op } from '../operation';
import { tensor3d } from '../tensor3d';
import * as util from '../../util';
import {convertToTensor} from '../../tensor_util_env';


/**
 * Performs threshold algorithms on Tensors
 * @param image a 2d image tensor of shape `[x , y, 3]`.
 * @param method Optional string from `'binary' | 'otsu'`,
 *     defaults to binary, which specifies type of the threshold operation
 * @param coeff Optional number which defines Threshold coefficient from 0 to 1.
 * @param inverted Optional boolian which specifies if colours should be inverted
 */

function otsu_alg(histData: number[], total: number) {

  let sum = 0;
  for (let t = 0; t < 256; t++) {
    sum += t * histData[t];
  }
  let sumB = 0;
  let wB = 0;
  let wF = 0;

  let varMax = 0;
  let thresh = 0;

  for (let t = 0; t < 256; t++) {
    wB += histData[t];               // Weight Background
    if (wB == 0) continue;

    wF = total - wB;                 // Weight Foreground
    if (wF == 0) break;

    sumB += t * histData[t];

    let mB = sumB / wB;            // Mean Background
    let mF = (sum - sumB) / wF;    // Mean Foreground

    // Calculate Between Class Variance
    let varBetween = wB * wF * (mB - mF) * (mB - mF);

    // Check if new maximum found
    if (varBetween > varMax) {
      varMax = varBetween;
      thresh = t;
    }
  }

  return thresh;
}



function threshold_(
  image: Tensor3D | TensorLike,
  method: 'binary' | 'otsu' = 'binary',
  coeff?: number,
  inverted?: boolean
): Tensor3D {
  //const $image = image;

  const $image = convertToTensor(image, 'image', 'threshold');

  const threshold = coeff * 255;
  let arrayed_image = Array.from($image.dataSync());


  util.assert(
    $image.rank === 3,
    () => 'Error in threshold: image must be rank 3,' +
      `but got rank ${$image.rank}.`);

  util.assert(
    method === 'otsu' || method === 'binary',
    () => `method must be binary or otsu, but was ${method}`);

  if (method == 'binary') {
    arrayed_image.forEach(function (item, i) {
      if (!inverted) {
        arrayed_image[i] = (item < threshold) ? 0 : 255;
      } else arrayed_image[i] = (item > threshold) ? 0 : 255;
    });

  } else if (method == 'otsu') {
    let histogram = Array(256).fill(0);

    for (let i = 0; i < arrayed_image.length; i++) {
      let gray = arrayed_image[i];
      histogram[Math.round(gray)] += 1;
    }

    let threshold = otsu_alg(histogram, arrayed_image.length);

    arrayed_image.forEach(function (item, i) {
      arrayed_image[i] = (item < threshold) ? 0 : 255;
    });

  }

  return tensor3d(arrayed_image, $image.shape, 'float32');

}

export const threshold = op({ threshold_ });
