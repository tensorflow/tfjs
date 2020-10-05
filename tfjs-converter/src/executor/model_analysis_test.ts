/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {NamedTensorMap, scalar} from '@tensorflow/tfjs-core';

import {IGraphDef} from '../data/compiled_api';
import {OperationMapper} from '../operations/operation_mapper';

import {getExecutionSubgraph} from './model_analysis';

describe('getExecutionInfo', () => {
  it('2 disconnected subgraphs, no dynamic ops', () => {
    const weightMap = {};
    const graphDef: IGraphDef = {
      node: [
        {name: 'input', op: 'Placeholder'},
        {name: 'intermediate', op: 'Add', input: ['input', 'input']},
        {name: 'output', op: 'Square', input: ['intermediate']},
        {name: 'input2', op: 'Const'},                    // Unrelated to input.
        {name: 'output2', op: 'Sqrt', input: ['input2']}  // Related to input2.
      ],
      versions: {producer: 1.0, minConsumer: 3}
    };
    const graph = OperationMapper.Instance.transformGraph(graphDef);

    // input --> output
    let inputs: NamedTensorMap = {'input': scalar(0)};
    let outputs = [graph.nodes['output']];
    let executionInfo = getExecutionSubgraph(inputs, outputs, weightMap);
    expect(executionInfo.inputs).toBe(inputs);
    expect(executionInfo.outputs).toBe(outputs);
    expect(executionInfo.dynamicNode).toBeFalsy();
    expect(executionInfo.missingInputs).toEqual([]);
    expect(executionInfo.syncInputs).toBeFalsy();
    expect(executionInfo.usedNodes).toContain('input');
    expect(executionInfo.usedNodes).toContain('intermediate');
    expect(executionInfo.usedNodes).toContain('output');
    expect(executionInfo.usedNodes.size).toBe(3);

    // input --> intermediate
    inputs = {'input': scalar(0)};
    outputs = [graph.nodes['intermediate']];
    executionInfo = getExecutionSubgraph(inputs, outputs, weightMap);
    expect(executionInfo.inputs).toBe(inputs);
    expect(executionInfo.outputs).toBe(outputs);
    expect(executionInfo.dynamicNode).toBeFalsy();
    expect(executionInfo.missingInputs).toEqual([]);
    expect(executionInfo.syncInputs).toBeFalsy();
    expect(executionInfo.usedNodes).toContain('input');
    expect(executionInfo.usedNodes).toContain('intermediate');
    expect(executionInfo.usedNodes.size).toBe(2);

    // input2 --> output2
    inputs = {'input2': scalar(0)};
    outputs = [graph.nodes['output2']];
    executionInfo = getExecutionSubgraph(inputs, outputs, weightMap);
    expect(executionInfo.inputs).toBe(inputs);
    expect(executionInfo.outputs).toBe(outputs);
    expect(executionInfo.dynamicNode).toBeFalsy();
    expect(executionInfo.missingInputs).toEqual([]);
    expect(executionInfo.syncInputs).toBeFalsy();
    expect(executionInfo.usedNodes).toContain('input2');
    expect(executionInfo.usedNodes).toContain('output2');
    expect(executionInfo.usedNodes.size).toBe(2);

    // input --> output2 is disconnected.
    inputs = {'input': scalar(0)};
    outputs = [graph.nodes['output2']];
    executionInfo = getExecutionSubgraph(inputs, outputs, weightMap);
    expect(executionInfo.inputs).toBe(inputs);
    expect(executionInfo.outputs).toBe(outputs);
    expect(executionInfo.dynamicNode).toBeFalsy();
    expect(executionInfo.missingInputs).toEqual(['input2']);
    expect(executionInfo.syncInputs).toBeFalsy();
    expect(executionInfo.usedNodes).toContain('output2');
    expect(executionInfo.usedNodes).toContain('input2');
    expect(executionInfo.usedNodes.size).toBe(2);

    // input2 --> output is disconnected.
    inputs = {'input2': scalar(0)};
    outputs = [graph.nodes['output']];
    executionInfo = getExecutionSubgraph(inputs, outputs, weightMap);
    expect(executionInfo.inputs).toBe(inputs);
    expect(executionInfo.outputs).toBe(outputs);
    expect(executionInfo.dynamicNode).toBeFalsy();
    expect(executionInfo.missingInputs).toEqual(['input']);
    expect(executionInfo.syncInputs).toBeFalsy();
    expect(executionInfo.usedNodes).toContain('input');
    expect(executionInfo.usedNodes).toContain('intermediate');
    expect(executionInfo.usedNodes).toContain('output');
    expect(executionInfo.usedNodes.size).toBe(3);
  });

  it('Async graph', () => {
    const weightMap = {};
    const graphDef: IGraphDef = {
      node: [
        {name: 'input', op: 'Placeholder'},
        {name: 'intermediate', op: 'Enter', input: ['input']},
        {name: 'intermediate2', op: 'Const', input: ['intermediate']},
        {name: 'output', op: 'Square', input: ['intermediate2']},
      ],
      versions: {producer: 1.0, minConsumer: 3}
    };
    const graph = OperationMapper.Instance.transformGraph(graphDef);

    // input --> output
    const inputs: NamedTensorMap = {'input': scalar(0)};
    const outputs = [graph.nodes['output']];
    const executionInfo = getExecutionSubgraph(inputs, outputs, weightMap);
    expect(executionInfo.inputs).toBe(inputs);
    expect(executionInfo.outputs).toBe(outputs);
    expect(executionInfo.dynamicNode).toBe(graph.nodes['intermediate']);
    expect(executionInfo.missingInputs).toEqual([]);
    expect(executionInfo.syncInputs).toEqual(['intermediate2']);
    expect(executionInfo.usedNodes).toContain('input');
    expect(executionInfo.usedNodes).toContain('intermediate');
    expect(executionInfo.usedNodes).toContain('intermediate2');
    expect(executionInfo.usedNodes).toContain('output');
    expect(executionInfo.usedNodes.size).toBe(4);
  });
});
