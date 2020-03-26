import {Tensor3D} from '../tensor';
import {TensorLike} from '../types';
import {assert} from '../util';
import {op} from './operation';
import {pad} from './pad';

/**
 * Pads a `tf.Tensor3D` with a given value and paddings. See `pad` for details.
 */
function pad3d_(
    x: Tensor3D|TensorLike,
    paddings: [[number, number], [number, number], [number, number]],
    constantValue = 0): Tensor3D {
  assert(
      paddings.length === 3 && paddings[0].length === 2 &&
          paddings[1].length === 2 && paddings[2].length === 2,
      () => 'Invalid number of paddings. Must be length of 2 each.');
  return pad(x, paddings, constantValue);
}

export const pad3d = op({pad3d_});
