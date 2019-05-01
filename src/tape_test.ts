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

import {ScopeFn} from './engine';
import * as tf from './index';
import {ALL_ENVS, describeWithFlags} from './jasmine_util';
import {backpropagateGradients, getFilteredNodesXToY, TapeNode} from './tape';
import {expectArraysClose} from './test_util';

describeWithFlags('getFilteredNodesXToY', ALL_ENVS, () => {
  it('no paths from x to y', () => {
    const x = tf.scalar(1);
    const intermediate1 = tf.scalar(0);

    const intermediate2 = tf.scalar(0);
    const y = tf.scalar(2);

    const tape: TapeNode[] = [
      {
        id: 0,
        name: 'node0',
        inputs: {x},
        outputs: [intermediate1],
        gradient: null
      },
      {
        id: 1,
        name: 'node1',
        inputs: {intermediate2},
        outputs: [y],
        gradient: null
      }
    ];

    const filteredTapeNodes = getFilteredNodesXToY(tape, [x], y);

    expect(filteredTapeNodes.length).toBe(0);
    expect(filteredTapeNodes).toEqual([]);
  });

  it('one operation x => y', () => {
    const x = tf.scalar(1);
    const y = tf.scalar(2);

    const tape: TapeNode[] =
        [{id: 0, name: 'node0', inputs: {x}, outputs: [y], gradient: null}];

    const filteredTapeNodes = getFilteredNodesXToY(tape, [x], y);

    expect(filteredTapeNodes.length).toBe(1);
    expect(filteredTapeNodes).toEqual(tape);
  });

  it('1 operation [x0, x1] => y, all input paths', () => {
    const x0 = tf.scalar(0);
    const x1 = tf.scalar(1);
    const y = tf.scalar(2);

    const tape: TapeNode[] = [
      {id: 0, name: 'node0', inputs: {x0, x1}, outputs: [y], gradient: null}
    ];

    const filteredTapeNodes = getFilteredNodesXToY(tape, [x0, x1], y);

    expect(filteredTapeNodes.length).toBe(1);
    expect(filteredTapeNodes).toEqual(tape);
  });

  it('one operation [x0, x1] => y, one input paths', () => {
    const x0 = tf.scalar(0);
    const x1 = tf.scalar(1);
    const y = tf.scalar(2);

    const tape: TapeNode[] = [
      {id: 0, name: 'node0', inputs: {x0, x1}, outputs: [y], gradient: null}
    ];

    const filteredTapeNodes = getFilteredNodesXToY(tape, [x0], y);

    expect(filteredTapeNodes.length).toBe(1);
    // x1 input should be pruned, we don't ask for the gradient of x1.
    expect(filteredTapeNodes[0])
        .toEqual(
            {id: 0, name: 'node0', inputs: {x0}, outputs: [y], gradient: null});
  });

  it('two operations x => intermediate => y', () => {
    const x = tf.scalar(1);
    const intermediate = tf.scalar(0);
    const y = tf.scalar(2);

    const tape: TapeNode[] = [
      {
        id: 0,
        name: 'node0',
        inputs: {x},
        outputs: [intermediate],
        gradient: null
      },
      {
        id: 1,
        name: 'node1',
        inputs: {intermediate},
        outputs: [y],
        gradient: null
      }
    ];

    const filteredTapeNodes = getFilteredNodesXToY(tape, [x], y);

    expect(filteredTapeNodes.length).toBe(2);
    expect(filteredTapeNodes).toEqual(tape);
  });

  it('two operations [x0, x1], [x2] => ' +
         'intermediate => y',
     () => {
       const x0 = tf.scalar(1);
       const x1 = tf.scalar(2);
       const x2 = tf.scalar(3);
       const intermediate = tf.scalar(4);
       const y = tf.scalar(2);

       const tape: TapeNode[] = [
         {
           id: 0,
           name: 'node0',
           inputs: {x0, x1},
           outputs: [intermediate],
           gradient: null
         },
         {
           id: 1,
           name: 'node1',
           inputs: {x2, intermediate},
           outputs: [y],
           gradient: null
         }
       ];

       const filteredTapeNodes = getFilteredNodesXToY(tape, [x0, x1, x2], y);

       expect(filteredTapeNodes.length).toBe(2);
       expect(filteredTapeNodes).toEqual(tape);
     });

  it('x => y and x => orphan', () => {
    const x = tf.scalar(1);
    const orphan = tf.scalar(0);
    const y = tf.scalar(2);

    const tape: TapeNode[] = [
      {id: 0, name: 'node0', inputs: {x}, outputs: [orphan], gradient: null},
      {id: 1, name: 'node1', inputs: {x}, outputs: [y], gradient: null}
    ];

    const filteredTapeNodes = getFilteredNodesXToY(tape, [x], y);

    expect(filteredTapeNodes.length).toBe(1);
    // The orphan should be removed.
    expect(filteredTapeNodes[0]).toEqual(tape[1]);
  });

  it('x => y and orphan => y', () => {
    const x = tf.scalar(1);
    const orphan = tf.scalar(0);
    const y = tf.scalar(2);

    const tape: TapeNode[] = [
      {id: 0, name: 'node0', inputs: {x, orphan}, outputs: [y], gradient: null}
    ];

    const filteredTapeNodes = getFilteredNodesXToY(tape, [x], y);

    expect(filteredTapeNodes.length).toBe(1);
    // The orphan should be pruned from the node's input.
    expect(filteredTapeNodes[0])
        .toEqual(
            {id: 0, name: 'node0', inputs: {x}, outputs: [y], gradient: null});
  });

  it('1 op with 3 outputs x => y1, y2, y3', () => {
    const x = tf.scalar(1);
    const y1 = tf.scalar(2);
    const y2 = tf.scalar(2);
    const y3 = tf.scalar(2);

    const tape: TapeNode[] = [
      {id: 0, name: 'node0', inputs: {x}, outputs: [y1, y2, y3], gradient: null}
    ];

    const filteredNodes1 = getFilteredNodesXToY(tape, [x], y1);
    expect(filteredNodes1.length).toBe(1);
    expect(filteredNodes1).toEqual(tape);

    const filteredNodes2 = getFilteredNodesXToY(tape, [x], y2);
    expect(filteredNodes2.length).toBe(1);
    expect(filteredNodes2).toEqual(tape);

    const filteredNodes3 = getFilteredNodesXToY(tape, [x], y3);
    expect(filteredNodes3.length).toBe(1);
    expect(filteredNodes3).toEqual(tape);
  });
});

describeWithFlags('backpropagateGradients', ALL_ENVS, () => {
  it('Throws if gradient is not defined', () => {
    const x = tf.scalar(0);
    const y = tf.scalar(1);

    const dy = tf.scalar(1);

    const accumulatedGradientsMap: {[tensorId: number]: tf.Tensor} = {};
    accumulatedGradientsMap[y.id] = dy;

    const tape: TapeNode[] =
        [{id: 0, name: 'node0', inputs: {x}, outputs: [y], gradient: null}];

    expect(
        () => backpropagateGradients(
            accumulatedGradientsMap, tape,
            f => tf.tidy(f as ScopeFn<tf.Tensor>)))
        .toThrowError();
  });

  it('basic backprop with 1 node', async () => {
    const x = tf.scalar(0);
    const y = tf.scalar(1);

    const dy = tf.scalar(1);

    const accumulatedGradientsMap: {[tensorId: number]: tf.Tensor} = {};
    accumulatedGradientsMap[y.id] = dy;

    const tape: TapeNode[] = [{
      id: 0,
      name: 'node0',
      inputs: {x},
      outputs: [y],
      gradient: (dy: tf.Scalar) => {
        return {x: () => dy.add(tf.scalar(1))};
      }
    }];

    backpropagateGradients(
        accumulatedGradientsMap, tape, f => tf.tidy(f as ScopeFn<tf.Tensor>));

    expectArraysClose(await accumulatedGradientsMap[x.id].data(), [2]);
  });

  it('basic backprop with 2 nodes', async () => {
    const x = tf.scalar(0);
    const intermediate = tf.scalar(1);
    const y = tf.scalar(2);

    const dy = tf.scalar(1);

    const accumulatedGradientsMap: {[tensorId: number]: tf.Tensor} = {};
    accumulatedGradientsMap[y.id] = dy;

    const tape: TapeNode[] = [
      {
        id: 0,
        name: 'node0',
        inputs: {x},
        outputs: [intermediate],
        gradient: (dy: tf.Scalar) => {
          return {x: () => dy.add(tf.scalar(1))};
        }
      },
      {
        id: 1,
        name: 'node1',
        inputs: {intermediate},
        outputs: [y],
        gradient: (dy: tf.Scalar) => {
          return {intermediate: () => dy.add(tf.scalar(1))};
        }
      }
    ];

    backpropagateGradients(
        accumulatedGradientsMap, tape, f => tf.tidy(f as ScopeFn<tf.Tensor>));

    // dx = dy + 1 + 1
    expectArraysClose(await accumulatedGradientsMap[x.id].data(), [3]);
  });

  it('basic backprop with a split node accumulates gradients', async () => {
    const x = tf.scalar(0);
    const intermediate1 = tf.scalar(1);
    const intermediate2 = tf.scalar(2);
    const y = tf.scalar(3);

    const dy = tf.scalar(1);

    const accumulatedGradientsMap: {[tensorId: number]: tf.Tensor} = {};
    accumulatedGradientsMap[y.id] = dy;

    const tape: TapeNode[] = [
      {
        id: 0,
        name: 'node0',
        inputs: {x},
        outputs: [intermediate1],
        gradient: (dy: tf.Scalar) => {
          return {x: () => dy.add(tf.scalar(1))};
        }
      },
      {
        id: 1,
        name: 'node1',
        inputs: {x},
        outputs: [intermediate2],
        gradient: (dy: tf.Scalar) => {
          return {x: () => dy.add(tf.scalar(1))};
        }
      },
      {
        id: 2,
        name: 'node2',
        inputs: {intermediate1, intermediate2},
        outputs: [y],
        gradient: (dy: tf.Scalar) => {
          return {
            intermediate1: () => dy.add(tf.scalar(1)),
            intermediate2: () => dy.add(tf.scalar(1))
          };
        }
      }
    ];

    backpropagateGradients(
        accumulatedGradientsMap, tape, f => tf.tidy(f as ScopeFn<tf.Tensor>));

    // dx = dy + 1 + 1 + 1 + 1 + 1
    expectArraysClose(
        await accumulatedGradientsMap[x.id].data(), [(await dy.data())[0] + 5]);
  });

  it('backprop over 1 node with 3 outputs, w.r.t to the 2nd output',
     async () => {
       const x = tf.tensor1d([1, 1, 1]);
       const y1 = tf.scalar(1);
       const y2 = tf.scalar(1);
       const y3 = tf.scalar(1);

       const accumulatedGradientsMap: {[tensorId: number]: tf.Tensor} = {};
       // Backproping through the 2nd output.
       const dy2 = tf.scalar(5);
       accumulatedGradientsMap[y2.id] = dy2;

       let dys: tf.Scalar[];
       const tape: TapeNode[] = [{
         id: 0,
         name: 'node0',
         inputs: {x},
         outputs: [y1, y2, y3],
         gradient: (dys_: tf.Scalar[]) => {
           dys = dys_;
           return {x: () => tf.stack(dys_)};
         }
       }];

       backpropagateGradients(
           accumulatedGradientsMap, tape,
           f => tf.tidy(f as ScopeFn<tf.Tensor>));
       expectArraysClose(await accumulatedGradientsMap[x.id].data(), [0, 5, 0]);
       expectArraysClose(await dys[0].data(), [0]);
       expectArraysClose(await dys[1].data(), [5]);
       expectArraysClose(await dys[2].data(), [0]);
     });
});
