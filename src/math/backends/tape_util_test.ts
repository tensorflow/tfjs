
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

import * as test_util from '../../test_util';
import {MathTests} from '../../test_util';
import {NamedArrayMap} from '../../util';
import {NDArray, Scalar} from '../ndarray';

// tslint:disable-next-line:max-line-length
import {Tape, TapeNode, TapeNodeInputConfig, TapeNodeOutput} from './tape_types';
import * as tape_util from './tape_util';

// getFilteredNodesXToY
{
  const tests: MathTests = it => {
    it('getFilteredNodesXToY no paths from x to y', math => {
      const x = Scalar.new(1);
      const intermediate1 = Scalar.new(0);

      const intermediate2 = Scalar.new(0);
      const y = Scalar.new(2);

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

    it('getFilteredNodesXToY one operation x => y', math => {
      const x = Scalar.new(1);
      const y = Scalar.new(2);

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
       math => {
         const x0 = Scalar.new(0);
         const x1 = Scalar.new(1);
         const y = Scalar.new(2);

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
       math => {
         const x0 = Scalar.new(0);
         const x1 = Scalar.new(1);
         const y = Scalar.new(2);

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

    it('getFilteredNodesXToY two operations x => intermediate => y', math => {
      const x = Scalar.new(1);
      const intermediate = Scalar.new(0);
      const y = Scalar.new(2);

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
       math => {
         const x0 = Scalar.new(1);
         const x1 = Scalar.new(2);
         const x2 = Scalar.new(3);
         const intermediate = Scalar.new(4);
         const y = Scalar.new(2);

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

    it('getFilteredNodesXToY x => y and x => orphan', math => {
      const x = Scalar.new(1);
      const orphan = Scalar.new(0);
      const y = Scalar.new(2);

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

    it('getFilteredNodesXToY x => y and orphan => y', math => {
      const x = Scalar.new(1);
      const orphan = Scalar.new(0);
      const y = Scalar.new(2);

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
       math => {
         const x = Scalar.new(1);
         const intermediate = Scalar.new(5);
         const orphan1 = Scalar.new(1);
         const orphan2 = Scalar.new(2);
         const orphan3 = Scalar.new(3);
         const y = Scalar.new(2);

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
       math => {
         const x0 = Scalar.new(1);
         const orphan0 = Scalar.new(2);

         const intermediate0 = Scalar.new(3);
         const intermediate1 = Scalar.new(4);

         const x1 = Scalar.new(5);
         const orphan1 = Scalar.new(6);
         const y = Scalar.new(7);
         const orphan2 = Scalar.new(8);

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
  const tests: MathTests = it => {
    it('Throws if gradient is not defined', math => {
      const x = Scalar.new(0);
      const y = Scalar.new(1);

      const dy = Scalar.new(1);

      const accumulatedGradientsMap: {[ndarrayId: number]: NDArray} = {};
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

    it('basic backprop with 1 node', math => {
      const x = Scalar.new(0);
      const y = Scalar.new(1);

      const dy = Scalar.new(1);

      const accumulatedGradientsMap: {[ndarrayId: number]: NDArray} = {};
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
          return {x: () => math.add(dy, Scalar.new(1))};
        }
      }];

      tape_util.backpropagateGradients(accumulatedGradientsMap, tape);

      test_util.expectArraysClose(accumulatedGradientsMap[x.id], [2]);
    });

    it('basic backprop with 2 nodes', math => {
      const x = Scalar.new(0);
      const intermediate = Scalar.new(1);
      const y = Scalar.new(2);

      const dy = Scalar.new(1);

      const accumulatedGradientsMap: {[ndarrayId: number]: NDArray} = {};
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
            return {x: () => math.add(dy, Scalar.new(1))};
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
            return {intermediate: () => math.add(dy, Scalar.new(1))};
          }
        }
      ];

      tape_util.backpropagateGradients(accumulatedGradientsMap, tape);

      // dx = dy + 1 + 1
      test_util.expectArraysClose(accumulatedGradientsMap[x.id], [3]);
    });

    it('basic backprop with a split node accumulates gradients', math => {
      const x = Scalar.new(0);
      const intermediate1 = Scalar.new(1);
      const intermediate2 = Scalar.new(2);
      const y = Scalar.new(3);

      const dy = Scalar.new(1);

      const accumulatedGradientsMap: {[ndarrayId: number]: NDArray} = {};
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
            return {x: () => math.add(dy, Scalar.new(1))};
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
            return {x: () => math.add(dy, Scalar.new(1))};
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
              intermediate1: () => math.add(dy, Scalar.new(1)),
              intermediate2: () => math.add(dy, Scalar.new(1))
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
       math => {
         const x = Scalar.new(0);
         const intermediate1 = Scalar.new(1);
         const intermediate2 = Scalar.new(2);
         const y = Scalar.new(3);

         const dy = Scalar.new(1);

         const accumulatedGradientsMap: {[ndarrayId: number]: NDArray} = {};
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
             gradient: (dy: NamedArrayMap, y: NamedArrayMap) => {
               return {
                 x: () =>
                     math.multiply(dy['intermediate1'], dy['intermediate2'])
               };
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
                 intermediate1: () => math.add(dy, Scalar.new(2)),
                 intermediate2: () => math.add(dy, Scalar.new(3))
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

// computeInputs
{
  const tests: MathTests = it => {
    it('no inputs', math => {
      const y = Scalar.new(2);

      const tape: Tape = [{
        id: 0,
        type: 'kernel',
        name: 'node0',
        inputAndArgs: {
          inputs: {},
        },
        output: y,
        gradient: null
      }];

      const inputs = tape_util.computeInputs(tape);

      expect(inputs).toEqual({});
    });

    it('basic', math => {
      const x = Scalar.new(1);
      const y = Scalar.new(2);

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

      const inputs = tape_util.computeInputs(tape);

      expect(inputs).toEqual({'0': x});
    });

    it('multiple inputs from multiple ops', math => {
      const x1 = Scalar.new(1);
      const intermediate1 = Scalar.new(0);

      const x2 = Scalar.new(0);
      const y = Scalar.new(2);

      const tape: Tape = [
        {
          id: 0,
          type: 'kernel',
          name: 'node0',
          inputAndArgs: {
            inputs: {x1},
          },
          output: intermediate1,
          gradient: null
        },
        {
          id: 1,
          type: 'kernel',
          name: 'node1',
          inputAndArgs: {
            inputs: {intermediate1, x2},
          },
          output: y,
          gradient: null
        }
      ];

      const inputs = tape_util.computeInputs(tape);

      expect(inputs).toEqual({'0': x1, '1': x2});
    });
  };
  test_util.describeMathCPU('tape_util.computeInputs', [tests]);
}

// extractNDArraysFromScopeResult
{
  const tests: MathTests = it => {
    it('null input returns empty array', math => {
      const results = tape_util.extractNDArraysFromScopeResult(null);

      expect(results).toEqual([]);
    });

    it('ndarray input returns one element array', math => {
      const x = Scalar.new(1);
      const results = tape_util.extractNDArraysFromScopeResult(x);

      expect(results).toEqual([x]);
    });

    it('name array map returns flattened array', math => {
      const x1 = Scalar.new(1);
      const x2 = Scalar.new(3);
      const x3 = Scalar.new(4);
      const results = tape_util.extractNDArraysFromScopeResult({x1, x2, x3});

      expect(results).toEqual([x1, x2, x3]);
    });
  };

  test_util.describeMathCPU(
      'tape_util.extractNDArraysFromScopeResult', [tests]);
}

{
  const tests: MathTests = it => {
    it('pass through when all inputs are defined', () => {
      const x1 = Scalar.new(1);
      const x2 = Scalar.new(2);
      const config: TapeNodeInputConfig = {
        inputs: {x1, x2},
      };
      expect(tape_util.stripUndefinedInputsFromInputConfig(config)).toEqual({
        inputs: {x1, x2}
      });
    });

    it('strips undefined inputs', () => {
      const x1 = Scalar.new(1);
      const x4 = Scalar.new(2);
      const config: TapeNodeInputConfig = {
        inputs: {x1, x2: undefined, x3: undefined, x4},
      };
      expect(tape_util.stripUndefinedInputsFromInputConfig(config)).toEqual({
        inputs: {x1, x4}
      });
    });
    test_util.describeMathCPU(
        'tape_util.extractNDArraysFromScopeResult', [tests]);
  };
}
