/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {Optimizer} from './optimizers/optimizer';
import {ConfigDict, registerClass, SerializationMap} from './serialization';
import {NamedVariableMap} from './tensor_types';

describe('registerClass', () => {
  const randomClassName = `OptimizerForTest${Math.random()}`;
  class OptimizerForTest extends Optimizer {
    static className = randomClassName;
    constructor() {
      super();
    }
    applyGradients(variableGradients: NamedVariableMap) {}

    getConfig(): ConfigDict {
      return {};
    }
  }
  it('registerClass succeeds', () => {
    registerClass(OptimizerForTest);
    expect(SerializationMap.getMap().classNameMap[randomClassName] != null)
        .toEqual(true);
  });

  class OptimizerWithoutClassName extends Optimizer {
    constructor() {
      super();
    }
    applyGradients(variableGradients: NamedVariableMap) {}

    getConfig(): ConfigDict {
      return {};
    }
  }
  it('registerClass fails on missing className', () => {
    // tslint:disable-next-line:no-any
    expect(() => registerClass(OptimizerWithoutClassName as any))
        .toThrowError(/does not have the static className property/);
  });

  class OptimizerWithEmptyClassName extends Optimizer {
    static className = '';
    constructor() {
      super();
    }
    applyGradients(variableGradients: NamedVariableMap) {}

    getConfig(): ConfigDict {
      return {};
    }
  }
  it('registerClass fails on missing className', () => {
    expect(() => registerClass(OptimizerWithEmptyClassName))
        .toThrowError(/has an empty-string as its className/);
  });

  class OptimizerWithNonStringClassName extends Optimizer {
    static className = 42;
    constructor() {
      super();
    }
    applyGradients(variableGradients: NamedVariableMap) {}

    getConfig(): ConfigDict {
      return {};
    }
  }
  it('registerClass fails on missing className', () => {
    // tslint:disable-next-line:no-any
    expect(() => registerClass(OptimizerWithNonStringClassName as any))
        .toThrowError(/is required to be a string, but got type number/);
  });
});