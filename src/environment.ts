/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as device_util from './device_util';
import * as webgl_util from './math/webgl/webgl_util';
import * as util from './util';

export enum Type {
  NUMBER,
  BOOLEAN
}

export interface Features {
  'WEBGL_DISJOINT_QUERY_TIMER'?: boolean;
  'WEBGL_VERSION'?: number;
}

export const URL_PROPERTIES: URLProperty[] = [
  {name: 'WEBGL_DISJOINT_QUERY_TIMER', type: Type.BOOLEAN},
  {name: 'WEBGL_VERSION', type: Type.NUMBER}
];

export interface URLProperty {
  name: keyof Features;
  type: Type;
}

function evaluateFeature<K extends keyof Features>(feature: K): Features[K] {
  if (feature === 'WEBGL_DISJOINT_QUERY_TIMER') {
    return !device_util.isMobile();
  } else if (feature === 'WEBGL_VERSION') {
    if (webgl_util.isWebGL2Enabled()) {
      return 2;
    } else if (webgl_util.isWebGL1Enabled()) {
      return 1;
    }
    return 0;
  }
  throw new Error(`Unknown feature ${feature}.`);
}

export class Environment {
  private features: Features = {};

  constructor(features?: Features) {
    if (features != null) {
      this.features = features;
    }
  }

  get<K extends keyof Features>(feature: K): Features[K] {
    if (feature in this.features) {
      return this.features[feature];
    }

    this.features[feature] = evaluateFeature(feature);

    return this.features[feature];
  }
}

// Expects flags from URL in the format ?dljsflags=FLAG1:1,FLAG2:true.
const DEEPLEARNJS_FLAGS_PREFIX = 'dljsflags';
function getFeaturesFromURL(): Features {
  const features: Features = {};

  const urlParams = util.getQueryParams(window.location.search);
  if (DEEPLEARNJS_FLAGS_PREFIX in urlParams) {
    const urlFlags: {[key: string]: string} = {};

    const keyValues = urlParams[DEEPLEARNJS_FLAGS_PREFIX].split(',');
    keyValues.forEach(keyValue => {
      const [key, value] = keyValue.split(':') as [string, string];
      urlFlags[key] = value;
    });

    URL_PROPERTIES.forEach(urlProperty => {
      if (urlProperty.name in urlFlags) {
        console.log(
            `Setting feature override from URL ${urlProperty.name}: ` +
            `${urlFlags[urlProperty.name]}`);
        if (urlProperty.type === Type.NUMBER) {
          features[urlProperty.name] = +urlFlags[urlProperty.name];
        } else if (urlProperty.type === Type.BOOLEAN) {
          features[urlProperty.name] = urlFlags[urlProperty.name] === 'true';
        } else {
          console.warn(`Unknown URL param: ${urlProperty.name}.`);
        }
      }
    });
  }

  return features;
}

export let ENV = new Environment(getFeaturesFromURL());

export function setEnvironment(environment: Environment) {
  ENV = environment;
}
