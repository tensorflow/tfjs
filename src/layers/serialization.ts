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

import {ConfigDict} from '../types';
import {ClassNameMap, deserializeKerasObject} from '../utils/generic_utils';

/**
 * Instantiate a layer from a config dictionary.
 * @param config: dict of the form {class_name: str, config: dict}
 * @param custom_objects: dict mapping class names (or function names)
 *      of custom (non-Keras) objects to class/functions
 * @returns Layer instance (may be Model, Sequential, Layer...)
 */
export function deserialize(
    // tslint:disable-next-line:no-any
    config: ConfigDict, customObjects = {} as ConfigDict): any {
  return deserializeKerasObject(
      config, ClassNameMap.getMap().pythonClassNameMap, customObjects, 'layer');
}
