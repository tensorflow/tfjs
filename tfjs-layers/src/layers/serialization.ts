/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/* Original Source layers/__init__.py */
import {serialization} from '@tensorflow/tfjs-core';

import {deserializeKerasObject} from '../utils/generic_utils';

/**
 * Instantiate a layer from a config dictionary.
 * @param config dict of the form {class_name: str, config: dict}
 * @param customObjects dict mapping class names (or function names)
 *   of custom (non-Keras) objects to class/functions
 * @param fastWeightInit Optional flag to use fast weight initialization
 *   during deserialization. This is applicable to cases in which
 *   the initialization will be immediately overwritten by loaded weight
 *   values. Default: `false`.
 * @returns Layer instance (may be LayersModel, Sequential, Layer...)
 */
export function deserialize(
    config: serialization.ConfigDict,
    customObjects = {} as serialization.ConfigDict,
    fastWeightInit = false): serialization.Serializable {
  return deserializeKerasObject(
      config, serialization.SerializationMap.getMap().classNameMap,
      customObjects, 'layer', fastWeightInit);
}
