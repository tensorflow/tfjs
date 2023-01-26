/**
 * @license
 * Copyright 2023 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Unit Tests for random width layer.
 */

import { image, Rank, serialization, Tensor, cast, stack, tidy } from '@tensorflow/tfjs-core';
import { getExactlyOneTensor } from '../../utils/types_utils';
import * as K from '../../backend/tfjs_backend';
import { Kwargs } from 'tfjs-layers/src/types';
import { ValueError } from 'tfjs-layers/src/errors';
import { BaseRandomLayerArgs, BaseRandomLayer } from 'tfjs-layers/src/engine/base_random_layer';
import * as tf from "@tensorflow/tfjs"

// test that incorrect inputs returns error
// test for 3D unbatched tensor should return float
// test for 4D batched tensor should return float
// test for width lower < -1
// test for width upper < -1
// test for upper width < lower width
// test bilinear
// test nearest

