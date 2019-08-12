/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import {monitorPromisesProgress} from './progress';

describe('util.monitorPromisesProgress', () => {
  it('Default progress from 0 to 1', (done) => {
    const expectFractions: number[] = [0.25, 0.50, 0.75, 1.00];
    const fractionList: number[] = [];
    const tasks = Array(4).fill(0).map(() => {
      return Promise.resolve();
    });
    monitorPromisesProgress(tasks, (progress: number) => {
      fractionList.push(parseFloat(progress.toFixed(2)));
    }).then(() => {
      expect(fractionList).toEqual(expectFractions);
      done();
    });
  });

  it('Progress with pre-defined range', (done) => {
    const startFraction = 0.2;
    const endFraction = 0.8;
    const expectFractions: number[] = [0.35, 0.50, 0.65, 0.80];
    const fractionList: number[] = [];
    const tasks = Array(4).fill(0).map(() => {
      return Promise.resolve();
    });
    monitorPromisesProgress(tasks, (progress: number) => {
      fractionList.push(parseFloat(progress.toFixed(2)));
    }, startFraction, endFraction).then(() => {
      expect(fractionList).toEqual(expectFractions);
      done();
    });
  });

  it('throws error when progress fraction is out of range', () => {
    expect(() => {
      const startFraction = -1;
      const endFraction = 1;
      const tasks = Array(4).fill(0).map(() => {
        return Promise.resolve();
      });
      monitorPromisesProgress(
          tasks, (progress: number) => {}, startFraction, endFraction);
    }).toThrowError();
  });

  it('throws error when startFraction more than endFraction', () => {
    expect(() => {
      const startFraction = 0.8;
      const endFraction = 0.2;
      const tasks = Array(4).fill(0).map(() => {
        return Promise.resolve();
      });
      monitorPromisesProgress(
          tasks, (progress: number) => {}, startFraction, endFraction);
    }).toThrowError();
  });

  it('throws error when promises is null', () => {
    expect(() => {
      monitorPromisesProgress(null, (progress: number) => {});
    }).toThrowError();
  });

  it('throws error when promises is empty array', () => {
    expect(() => {
      monitorPromisesProgress([], (progress: number) => {});
    }).toThrowError();
  });
});
