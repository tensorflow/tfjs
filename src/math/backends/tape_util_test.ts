
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

import * as dl from '../../index';
import {NamedTensorMap} from '../../math/types';
import * as test_util from '../../test_util';
import {Scalar, Tensor} from '../tensor';
// tslint:disable-next-line:max-line-length
import {Tape, TapeNode, TapeNodeInputConfig, TapeNodeOutput} from './tape_types';
import * as tape_util from './tape_util';

// getFilteredNodesXToY
{
  const tests = () => {
    it('getFilteredNodesXToY no paths from x to y', () => {
      const x = dl.scalar(1);
      const intermediate1 = dl.scalar(0);

      const intermediate2 = dl.scalar(0);
      const y = dl.scalar(2);

      const tape: Tape = [
        {
          id: 0,
          type: 'kernel',
          name: 'node0',
          inputAndArgs: {
            inputs: {x},
          },
          output: intermediate1,
          gradient: null
        },
        {
          id: 1,
          type: 'kernel',
          name: 'node1',
          inputAndArgs: {
            inputs: {intermediate2},
          },
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

      const tape: Tape = [{
        id: 0,
        type: 'kernel',
        name: 'node0',
        inputAndArgs: {
          inputs: {x},
        },
        output: y,
        gradient: null
      }];

      const filteredTapeNodes = tape_util.getFilteredNodesXToY(tape, [x], y);

      expect(filteredTapeNodes.length).toBe(1);
      expect(filteredTapeNodes).toEqual(tape);
    });

    it('getFilteredNodesXToY 1 operation [x0, x1] => y, all input paths',
       () => {
         const x0 = dl.scalar(0);
         const x1 = dl.scalar(1);
         const y = dl.scalar(2);

         const tape: Tape = [{
           id: 0,
           type: 'kernel',
           name: 'node0',
           inputAndArgs: {
             inputs: {x0, x1},
           },
           output: y,
           gradient: null
         }];

         const filteredTapeNodes =
             tape_util.getFilteredNodesXToY(tape, [x0, x1], y);

         expect(filteredTapeNodes.length).toBe(1);
         expect(filteredTapeNodes).toEqual(tape);
       });

    it('getFilteredNodesXToY one operation [x0, x1] => y, one input paths',
       () => {
         const x0 = dl.scalar(0);
         const x1 = dl.scalar(1);
         const y = dl.scalar(2);

         const tape: Tape = [{
           id: 0,
           type: 'kernel',
           name: 'node0',
           inputAndArgs: {
             inputs: {x0, x1},
           },
           output: y,
           gradient: null
         }];

         const filteredTapeNodes =
             tape_util.getFilteredNodesXToY(tape, [x0], y);

         expect(filteredTapeNodes.length).toBe(1);
         // x1 input should be pruned, we don't ask for the gradient of x1.
         expect(filteredTapeNodes[0]).toEqual({
           id: 0,
           type: 'kernel',
           name: 'node0',
           inputAndArgs: {
             inputs: {x0},
           },
           output: y,
           gradient: null
         });
       });

    it('getFilteredNodesXToY two operations x => intermediate => y', () => {
      const x = dl.scalar(1);
      const intermediate = dl.scalar(0);
      const y = dl.scalar(2);

      const tape: Tape = [
        {
          id: 0,
          type: 'kernel',
          name: 'node0',
          inputAndArgs: {
            inputs: {x},
          },
          output: intermediate,
          gradient: null
        },
        {
          id: 1,
          type: 'kernel',
          name: 'node1',
          inputAndArgs: {
            inputs: {intermediate},
          },
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

         const tape: Tape = [
           {
             id: 0,
             type: 'kernel',
             name: 'node0',
             inputAndArgs: {
               inputs: {x0, x1},
             },
             output: intermediate,
             gradient: null
           },
           {
             id: 1,
             type: 'kernel',
             name: 'node1',
             inputAndArgs: {
               inputs: {x2, intermediate},
             },
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

      const tape: Tape = [
        {
          id: 0,
          type: 'kernel',
          name: 'node0',
          inputAndArgs: {
            inputs: {x},
          },
          output: orphan,
          gradient: null
        },
        {
          id: 1,
          type: 'kernel',
          name: 'node1',
          inputAndArgs: {
            inputs: {x},
          },
          output: y,
          gradient: null
        }
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

      const tape: Tape = [{
        id: 0,
        type: 'kernel',
        name: 'node0',
        inputAndArgs: {
          inputs: {x, orphan},
        },
        output: y,
        gradient: null
      }];

      const filteredTapeNodes = tape_util.getFilteredNodesXToY(tape, [x], y);

      expect(filteredTapeNodes.length).toBe(1);
      // The orphan should be pruned from the node's input.
      expect(filteredTapeNodes[0]).toEqual({
        id: 0,
        type: 'kernel',
        name: 'node0',
        inputAndArgs: {
          inputs: {x},
        },
        output: y,
        gradient: null
      });
    });

    it('getFilteredNodesXToY x => {intermediate, orphan1} and ' +
           '{orphan2, intermediate} => {y, orphan3}',
       () => {
         const x = dl.scalar(1);
         const intermediate = dl.scalar(5);
         const orphan1 = dl.scalar(1);
         const orphan2 = dl.scalar(2);
         const orphan3 = dl.scalar(3);
         const y = dl.scalar(2);

         const tape: Array<TapeNode<TapeNodeOutput>> = [
           {
             id: 0,
             type: 'kernel',
             name: 'node0',
             inputAndArgs: {
               inputs: {x},
             },
             output: {orphan1, intermediate},
             gradient: null
           },
           {
             id: 1,
             type: 'kernel',
             name: 'node1',
             inputAndArgs: {
               inputs: {intermediate, orphan2},
             },
             output: {y, orphan3},
             gradient: null
           }
         ];

         const filteredTapeNodes = tape_util.getFilteredNodesXToY(tape, [x], y);

         expect(filteredTapeNodes.length).toBe(2);
         // The orphans should be pruned from inputs and outputs.
         expect(filteredTapeNodes[0]).toEqual({
           id: 0,
           type: 'kernel',
           name: 'node0',
           inputAndArgs: {
             inputs: {x},
           },
           output: {intermediate},
           gradient: null
         });
         expect(filteredTapeNodes[1]).toEqual({
           id: 1,
           type: 'kernel',
           name: 'node1',
           inputAndArgs: {
             inputs: {intermediate},
           },
           output: {y},
           gradient: null
         });
       });

    it('getFilteredNodesXToY x0 => orphan0, ' +
           'x0 => intermediate0, x0 => intermediate1, ' +
           '[intermediate0, intermediate1, x1, orphan1] => {y, orphan2}',
       () => {
         const x0 = dl.scalar(1);
         const orphan0 = dl.scalar(2);

         const intermediate0 = dl.scalar(3);
         const intermediate1 = dl.scalar(4);

         const x1 = dl.scalar(5);
         const orphan1 = dl.scalar(6);
         const y = dl.scalar(7);
         const orphan2 = dl.scalar(8);

         const tape: Tape = [
           {
             id: 0,
             type: 'kernel',
             name: 'node0',
             inputAndArgs: {
               inputs: {x0},
             },
             output: intermediate0,
             gradient: null
           },
           {
             id: 1,
             type: 'kernel',
             name: 'node1',
             inputAndArgs: {
               inputs: {x0},
             },
             output: intermediate1,
             gradient: null
           },
           {
             id: 2,
             type: 'kernel',
             name: 'node2',
             inputAndArgs: {
               inputs: {x0},
             },
             output: orphan0,
             gradient: null
           },
           {
             id: 3,
             type: 'kernel',
             name: 'node3',
             inputAndArgs: {
               inputs: {intermediate0, intermediate1, x1, orphan1},
             },
             output: {y, orphan2},
             gradient: null
           }
         ];

         const filteredTapeNodes =
             tape_util.getFilteredNodesXToY(tape, [x0, x1], y);

         expect(filteredTapeNodes.length).toBe(3);
         expect(filteredTapeNodes[0]).toEqual(tape[0]);
         expect(filteredTapeNodes[1]).toEqual(tape[1]);
         // The orphans should be removed and the orphan1 should be pruned from
         // inputs.
         expect(filteredTapeNodes[2]).toEqual({
           id: 3,
           type: 'kernel',
           name: 'node3',
           inputAndArgs: {
             inputs: {intermediate0, intermediate1, x1},
           },
           output: {y},
           gradient: null
         });
       });
  };

  test_util.describeMathCPU('tape_util.getFilteredNodesXToY', [tests]);
}

// backpropagateGradients
{
  const tests = () => {
    it('Throws if gradient is not defined', () => {
      const x = dl.scalar(0);
      const y = dl.scalar(1);

      const dy = dl.scalar(1);

      const accumulatedGradientsMap: {[tensorId: number]: Tensor} = {};
      accumulatedGradientsMap[y.id] = dy;

      const tape: Tape = [{
        id: 0,
        type: 'kernel',
        name: 'node0',
        inputAndArgs: {
          inputs: {x},
        },
        output: y,
        gradient: null
      }];

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

      const tape: Tape = [{
        id: 0,
        type: 'kernel',
        name: 'node0',
        inputAndArgs: {
          inputs: {x},
        },
        output: y,
        gradient: (dy: Scalar, y: Scalar) => {
          return {x: () => dy.add(dl.scalar(1))};
        }
      }];

      tape_util.backpropagateGradients(accumulatedGradientsMap, tape);

      test_util.expectArraysClose(accumulatedGradientsMap[x.id], [2]);
    });

    it('basic backprop with 2 nodes', () => {
      const x = dl.scalar(0);
      const intermediate = dl.scalar(1);
      const y = dl.scalar(2);

      const dy = dl.scalar(1);

      const accumulatedGradientsMap: {[tensorId: number]: Tensor} = {};
      accumulatedGradientsMap[y.id] = dy;

      const tape: Tape = [
        {
          id: 0,
          type: 'kernel',
          name: 'node0',
          inputAndArgs: {
            inputs: {x},
          },
          output: intermediate,
          gradient: (dy: Scalar, y: Scalar) => {
            return {x: () => dy.add(dl.scalar(1))};
          }
        },
        {
          id: 1,
          type: 'kernel',
          name: 'node1',
          inputAndArgs: {
            inputs: {intermediate},
          },
          output: y,
          gradient: (dy: Scalar, y: Scalar) => {
            return {intermediate: () => dy.add(dl.scalar(1))};
          }
        }
      ];

      tape_util.backpropagateGradients(accumulatedGradientsMap, tape);

      // dx = dy + 1 + 1
      test_util.expectArraysClose(accumulatedGradientsMap[x.id], [3]);
    });

    it('basic backprop with a split node accumulates gradients', () => {
      const x = dl.scalar(0);
      const intermediate1 = dl.scalar(1);
      const intermediate2 = dl.scalar(2);
      const y = dl.scalar(3);

      const dy = dl.scalar(1);

      const accumulatedGradientsMap: {[tensorId: number]: Tensor} = {};
      accumulatedGradientsMap[y.id] = dy;

      const tape: Tape = [
        {
          id: 0,
          type: 'kernel',
          name: 'node0',
          inputAndArgs: {
            inputs: {x},
          },
          output: intermediate1,
          gradient: (dy: Scalar, y: Scalar) => {
            return {x: () => dy.add(dl.scalar(1))};
          }
        },
        {
          id: 1,
          type: 'kernel',
          name: 'node1',
          inputAndArgs: {
            inputs: {x},
          },
          output: intermediate2,
          gradient: (dy: Scalar, y: Scalar) => {
            return {x: () => dy.add(dl.scalar(1))};
          }
        },
        {
          id: 2,
          type: 'kernel',
          name: 'node2',
          inputAndArgs: {
            inputs: {intermediate1, intermediate2},
          },
          output: y,
          gradient: (dy: Scalar, y: Scalar) => {
            return {
              intermediate1: () => dy.add(dl.scalar(1)),
              intermediate2: () => dy.add(dl.scalar(1))
            };
          }
        }
      ];

      tape_util.backpropagateGradients(accumulatedGradientsMap, tape);

      // dx = dy + 1 + 1 + 1 + 1 + 1
      test_util.expectArraysClose(
          accumulatedGradientsMap[x.id], [dy.dataSync()[0] + 5]);
    });

    it('basic backprop with a multi-output split node accumulates gradients',
       () => {
         const x = dl.scalar(0);
         const intermediate1 = dl.scalar(1);
         const intermediate2 = dl.scalar(2);
         const y = dl.scalar(3);

         const dy = dl.scalar(1);

         const accumulatedGradientsMap: {[tensorId: number]: Tensor} = {};
         accumulatedGradientsMap[y.id] = dy;

         const tape: Array<TapeNode<TapeNodeOutput>> = [
           {
             id: 0,
             type: 'kernel',
             name: 'node0',
             inputAndArgs: {
               inputs: {x},
             },
             output: {intermediate1, intermediate2},
             gradient: (dy: NamedTensorMap, y: NamedTensorMap) => {
               return {x: () => dy['intermediate1'].mul(dy['intermediate2'])};
             }
           },
           {
             id: 1,
             type: 'kernel',
             name: 'node1',
             inputAndArgs: {
               inputs: {intermediate1, intermediate2},
             },
             output: y,
             gradient: (dy: Scalar, y: Scalar) => {
               return {
                 intermediate1: () => dy.add(dl.scalar(2)),
                 intermediate2: () => dy.add(dl.scalar(3))
               };
             }
           }
         ];

         tape_util.backpropagateGradients(accumulatedGradientsMap, tape);

         test_util.expectArraysClose(
             accumulatedGradientsMap[x.id], [(dy.get() + 2) * (dy.get() + 3)]);
       });
  };

  test_util.describeMathCPU('tape_util.backpropagateGradients', [tests]);
}

// extractTensorsFromScopeResult
{
  const tests = () => {
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
  };

  test_util.describeMathCPU('tape_util.extractTensorsFromScopeResult', [tests]);
}

{
  const tests = () => {
    it('pass through when all inputs are defined', () => {
      const x1 = dl.scalar(1);
      const x2 = dl.scalar(2);
      const config: TapeNodeInputConfig = {
        inputs: {x1, x2},
      };
      expect(tape_util.stripUndefinedInputsFromInputConfig(config)).toEqual({
        inputs: {x1, x2}
      });
    });

    it('strips undefined inputs', () => {
      const x1 = dl.scalar(1);
      const x4 = dl.scalar(2);
      const config: TapeNodeInputConfig = {
        inputs: {x1, x2: undefined, x3: undefined, x4},
      };
      expect(tape_util.stripUndefinedInputsFromInputConfig(config)).toEqual({
        inputs: {x1, x4}
      });
    });
    test_util.describeMathCPU(
        'tape_util.extractTensorsFromScopeResult', [tests]);
  };
}
