/**
 * @license
 * Copyright 2023 Google LLC.
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

/**
 * Unit Tests for StartEndPacker Layer.
 */

import { StartEndPacker } from './start_end_packer';

describe('StartEndPacker', () => {
  it('correct getConfig', () => {
    const startEndPacker = new StartEndPacker({
      sequenceLength: 512,
      startValue: 10,
      endValue: 20,
      padValue: 100,
    });
    const config = startEndPacker.getConfig();

    expect(config.sequenceLength).toEqual(512);
    expect(config.startValue).toEqual(10);
    expect(config.endValue).toEqual(20);
    expect(config.padValue).toEqual(100);
    expect(config.name).toEqual(undefined);
  });
});
