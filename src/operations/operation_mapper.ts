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
import {tensorflow} from '../data/index';

import {ParamValue} from './index';
import * as data from './op_list.json';
import {Graph, Node, OpMapper} from './types';

export class OperationMapper {
  private static _instance: OperationMapper;

  private opMappers: {[key: string]: OpMapper};

  public static get Instance() {
    return this._instance || (this._instance = new this());
  }

  private constructor() {
    this.opMappers =
        ((data as any) as OpMapper[])
            .reduce<{[key: string]: OpMapper}>((map, mapper: OpMapper) => {
              map[mapper.tfOpName] = mapper;
              return map;
            }, {});
  }

  transformGraph(graph: tensorflow.IGraphDef): Graph {
    const tfNodes = graph.node;
    const nodes = tfNodes.reduce<{[key: string]: Node}>((map, node) => {
      map[node.name] = this.mapNode(node);
      return map;
    }, {});

    const inputs: Node[] = [];
    const outputs: Node[] = [];
    Object.keys(nodes).forEach(key => {
      const node = nodes[key];
      node.inputNames.forEach(name => {
        node.inputs.push(nodes[name]);
        nodes[name].children.push(node);
      });
      if (node.inputs.length === 0) inputs.push(node);
    });

    Object.keys(nodes).forEach(key => {
      const node = nodes[key];
      if (node.children.length === 0) outputs.push(node);
    });
    return {nodes, inputs, outputs};
  }

  private mapNode(node: tensorflow.INodeDef): Node {
    const mapper = this.opMappers[node.op];
    if (mapper === undefined) {
      throw new Error('Tensorflow Op is not supported: ' + node.op);
    }
    const newNode: Node = {
      name: node.name,
      op: mapper.dlOpName,
      inputNames: node.input || [],
      inputs: [],
      children: [],
      params: {}
    };

    if (!!mapper.params) {
      newNode.params = mapper.params.reduce<{[key: string]:
                                                 ParamValue}>((map, param) => {
        const inputIndex = param.tfInputIndex;
        const type = param.type;
        let value = undefined;
        if (inputIndex === undefined) {
          switch (param.type) {
            case 'string':
              value = this.getStringParam(
                  node.attr, param.tfParamName, param.defaultValue as string);
              break;
            case 'number':
              value = this.getNumberParam(
                  node.attr, param.tfParamName, param.defaultValue as number);
              break;
            case 'number[]':
              value = this.getNumericArrayParam(
                  node.attr, param.tfParamName, param.defaultValue as number[]);
              break;
            case 'bool':
              value = this.getBoolParam(
                  node.attr, param.tfParamName, param.defaultValue as boolean);
              break;
            case 'shape':
              value = this.getTensorShapeParam(
                  node.attr, param.tfParamName, param.defaultValue as number[]);
              break;
            default:
              throw new Error(
                  'Unsupported param type: ' + param.type +
                  ' for op: ' + node.op);
          }
        }
        map[param.dlParamName] = {value, inputIndex, type};
        return map;
      }, {});
    }
    return newNode;
  }

  private getStringParam(
      attrs: {[key: string]: tensorflow.IAttrValue}, name: string, def: string,
      keepCase = false): string {
    const param = attrs[name];
    if (param !== undefined) {
      const value = String.fromCharCode.apply(null, param.s);
      return keepCase ? value : value.toLowerCase();
    }
    return def;
  }

  private getBoolParam(
      attrs: {[key: string]: tensorflow.IAttrValue}, name: string,
      def: boolean): boolean {
    const param = attrs[name];
    return param ? param.b : def;
  }

  private getNumberParam(
      attrs: {[key: string]: tensorflow.IAttrValue}, name: string,
      def: number): number {
    const param = attrs[name];
    return (param ? ((param.f !== undefined) ? param.f : param.i) : def) as
        number;
  }

  private getTensorShapeParam(
      attrs: {[key: string]: tensorflow.IAttrValue}, name: string,
      def?: number[]): number[]|undefined {
    const param = attrs[name];
    if (param && param.shape) {
      return param.shape.dim.map(dim => dim.size as number);
    }
    return def;
  }

  private getNumericArrayParam(
      attrs: {[key: string]: tensorflow.IAttrValue}, name: string,
      def: number[]): number[] {
    const param = attrs[name];
    if (param) {
      return (param.list.f.length ? param.list.f : param.list.i) as number[];
    }
    return def;
  }
}
