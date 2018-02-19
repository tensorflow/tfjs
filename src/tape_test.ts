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

import * as dl from './index';
import * as tape_util from './tape';
import {TapeNode} from './tape';
import {Scalar, Tensor} from './tensor';
import {CPU_ENVS, describeWithFlags, expectArraysClose} from './test_util';

describeWithFlags('getFilteredNodesXToY', CPU_ENVS, () => {
  it('getFilteredNodesXToY no paths from x to y', () => {
    const x = dl.scalar(1);
    const intermediate1 = dl.scalar(0);

    const intermediate2 = dl.scalar(0);
    const y = dl.scalar(2);

    const tape: TapeNode[] = [
      {
        id: 0,
        name: 'node0',
        inputs: {x},
        output: intermediate1,
        gradient: null
      },
      {
        id: 1,
        name: 'node1',
        inputs: {intermediate2},
        output: y,
        gradient: null
      }
    ];

    const filteredTapeNodes = tape_util.getFilteredNodesXToY(tape, [x], y);

    expect(filteredTapeNodes.length).toBe(0);
    expect(filteredTapeNodes).toEqual([]);
  });

  it('getFilteredNodesXToY one operation x => y', () => {
    const x = dl.scalar(1);
    const y = dl.scalar(2);

    const tape: TapeNode[] =
        [{id: 0, name: 'node0', inputs: {x}, output: y, gradient: null}];

    const filteredTapeNodes = tape_util.getFilteredNodesXToY(tape, [x], y);

    expect(filteredTapeNodes.length).toBe(1);
    expect(filteredTapeNodes).toEqual(tape);
  });

  it('getFilteredNodesXToY 1 operation [x0, x1] => y, all input paths', () => {
    const x0 = dl.scalar(0);
    const x1 = dl.scalar(1);
    const y = dl.scalar(2);

    const tape: TapeNode[] =
        [{id: 0, name: 'node0', inputs: {x0, x1}, output: y, gradient: null}];

    const filteredTapeNodes = tape_util.getFilteredNodesXToY(tape, [x0, x1], y);

    expect(filteredTapeNodes.length).toBe(1);
    expect(filteredTapeNodes).toEqual(tape);
  });

  it('getFilteredNodesXToY one operation [x0, x1] => y, one input paths',
     () => {
       const x0 = dl.scalar(0);
       const x1 = dl.scalar(1);
       const y = dl.scalar(2);

       const tape: TapeNode[] = [
         {id: 0, name: 'node0', inputs: {x0, x1}, output: y, gradient: null}
       ];

       const filteredTapeNodes = tape_util.getFilteredNodesXToY(tape, [x0], y);

       expect(filteredTapeNodes.length).toBe(1);
       // x1 input should be pruned, we don't ask for the gradient of x1.
       expect(filteredTapeNodes[0])
           .toEqual(
               {id: 0, name: 'node0', inputs: {x0}, output: y, gradient: null});
     });

  it('getFilteredNodesXToY two operations x => intermediate => y', () => {
    const x = dl.scalar(1);
    const intermediate = dl.scalar(0);
    const y = dl.scalar(2);

    const tape: TapeNode[] = [
      {id: 0, name: 'node0', inputs: {x}, output: intermediate, gradient: null},
      {
        id: 1,
        name: 'node1',
        inputs: {intermediate},
        output: y,
        gradient: null
      }
    ];

    const filteredTapeNodes = tape_util.getFilteredNodesXToY(tape, [x], y);

    expect(filteredTapeNodes.length).toBe(2);
    expect(filteredTapeNodes).toEqual(tape);
  });

  it('getFilteredNodesXToY two operations [x0, x1], [x2] => ' +
         'intermediate => y',
     () => {
       const x0 = dl.scalar(1);
       const x1 = dl.scalar(2);
       const x2 = dl.scalar(3);
       const intermediate = dl.scalar(4);
       const y = dl.scalar(2);

       const tape: TapeNode[] = [
         {
           id: 0,
           name: 'node0',
           inputs: {x0, x1},
           output: intermediate,
           gradient: null
         },
         {
           id: 1,
           name: 'node1',
           inputs: {x2, intermediate},
           output: y,
           gradient: null
         }
       ];

       const filteredTapeNodes =
           tape_util.getFilteredNodesXToY(tape, [x0, x1, x2], y);

       expect(filteredTapeNodes.length).toBe(2);
       expect(filteredTapeNodes).toEqual(tape);
     });

  it('getFilteredNodesXToY x => y and x => orphan', () => {
    const x = dl.scalar(1);
    const orphan = dl.scalar(0);
    const y = dl.scalar(2);

    const tape: TapeNode[] = [
      {id: 0, name: 'node0', inputs: {x}, output: orphan, gradient: null},
      {id: 1, name: 'node1', inputs: {x}, output: y, gradient: null}
    ];

    const filteredTapeNodes = tape_util.getFilteredNodesXToY(tape, [x], y);

    expect(filteredTapeNodes.length).toBe(1);
    // The orphan should be removed.
    expect(filteredTapeNodes[0]).toEqual(tape[1]);
  });

  it('getFilteredNodesXToY x => y and orphan => y', () => {
    const x = dl.scalar(1);
    const orphan = dl.scalar(0);
    const y = dl.scalar(2);

    const tape: TapeNode[] = [
      {id: 0, name: 'node0', inputs: {x, orphan}, output: y, gradient: null}
    ];

    const filteredTapeNodes = tape_util.getFilteredNodesXToY(tape, [x], y);

    expect(filteredTapeNodes.length).toBe(1);
    // The orphan should be pruned from the node's input.
    expect(filteredTapeNodes[0])
        .toEqual(
            {id: 0, name: 'node0', inputs: {x}, output: y, gradient: null});
  });
});

describeWithFlags('backpropagateGradients', CPU_ENVS, () => {
  it('Throws if gradient is not defined', () => {
    const x = dl.scalar(0);
    const y = dl.scalar(1);

    const dy = dl.scalar(1);

    const accumulatedGradientsMap: {[tensorId: number]: Tensor} = {};
    accumulatedGradientsMap[y.id] = dy;

    const tape: TapeNode[] =
        [{id: 0, name: 'node0', inputs: {x}, output: y, gradient: null}];

    expect(
        () => tape_util.backpropagateGradients(accumulatedGradientsMap, tape))
        .toThrowError();
  });

  it('basic backprop with 1 node', () => {
    const x = dl.scalar(0);
    const y = dl.scalar(1);

    const dy = dl.scalar(1);

    const accumulatedGradientsMap: {[tensorId: number]: Tensor} = {};
    accumulatedGradientsMap[y.id] = dy;

    const tape: TapeNode[] = [{
      id: 0,
      name: 'node0',
      inputs: {x},
      output: y,
      gradient: (dy: Scalar) => {
        return {x: () => dy.add(dl.scalar(1))};
      }
    }];

    tape_util.backpropagateGradients(accumulatedGradientsMap, tape);

    expectArraysClose(accumulatedGradientsMap[x.id], [2]);
  });

  it('basic backprop with 2 nodes', () => {
    const x = dl.scalar(0);
    const intermediate = dl.scalar(1);
    const y = dl.scalar(2);

    const dy = dl.scalar(1);

    const accumulatedGradientsMap: {[tensorId: number]: Tensor} = {};
    accumulatedGradientsMap[y.id] = dy;

    const tape: TapeNode[] = [
      {
        id: 0,
        name: 'node0',
        inputs: {x},
        output: intermediate,
        gradient: (dy: Scalar) => {
          return {x: () => dy.add(dl.scalar(1))};
        }
      },
      {
        id: 1,
        name: 'node1',
        inputs: {intermediate},
        output: y,
        gradient: (dy: Scalar) => {
          return {intermediate: () => dy.add(dl.scalar(1))};
        }
      }
    ];

    tape_util.backpropagateGradients(accumulatedGradientsMap, tape);

    // dx = dy + 1 + 1
    expectArraysClose(accumulatedGradientsMap[x.id], [3]);
  });

  it('basic backprop with a split node accumulates gradients', () => {
    const x = dl.scalar(0);
    const intermediate1 = dl.scalar(1);
    const intermediate2 = dl.scalar(2);
    const y = dl.scalar(3);

    const dy = dl.scalar(1);

    const accumulatedGradientsMap: {[tensorId: number]: Tensor} = {};
    accumulatedGradientsMap[y.id] = dy;

    const tape: TapeNode[] = [
      {
        id: 0,
        name: 'node0',
        inputs: {x},
        output: intermediate1,
        gradient: (dy: Scalar) => {
          return {x: () => dy.add(dl.scalar(1))};
        }
      },
      {
        id: 1,
        name: 'node1',
        inputs: {x},
        output: intermediate2,
        gradient: (dy: Scalar) => {
          return {x: () => dy.add(dl.scalar(1))};
        }
      },
      {
        id: 2,
        name: 'node2',
        inputs: {intermediate1, intermediate2},
        output: y,
        gradient: (dy: Scalar) => {
          return {
            intermediate1: () => dy.add(dl.scalar(1)),
            intermediate2: () => dy.add(dl.scalar(1))
          };
        }
      }
    ];

    tape_util.backpropagateGradients(accumulatedGradientsMap, tape);

    // dx = dy + 1 + 1 + 1 + 1 + 1
    expectArraysClose(accumulatedGradientsMap[x.id], [dy.dataSync()[0] + 5]);
  });
});

describeWithFlags('extractTensorsFromScopeResult', CPU_ENVS, () => {
  it('null input returns empty tensor', () => {
    const results = tape_util.extractTensorsFromScopeResult(null);

    expect(results).toEqual([]);
  });

  it('tensor input returns one element tensor', () => {
    const x = dl.scalar(1);
    const results = tape_util.extractTensorsFromScopeResult(x);

    expect(results).toEqual([x]);
  });

  it('name tensor map returns flattened tensor', () => {
    const x1 = dl.scalar(1);
    const x2 = dl.scalar(3);
    const x3 = dl.scalar(4);
    const results = tape_util.extractTensorsFromScopeResult({x1, x2, x3});

    expect(results).toEqual([x1, x2, x3]);
  });
});
