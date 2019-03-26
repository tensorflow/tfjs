/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {DataType, ENV} from '@tensorflow/tfjs-core';
import * as tensorflow from '../data/compiled_api';
import {getNodeNameAndIndex} from './executors/utils';
import * as arithmetic from './op_list/arithmetic';
import * as basicMath from './op_list/basic_math';
import * as control from './op_list/control';
import * as convolution from './op_list/convolution';
import * as creation from './op_list/creation';
import * as dynamic from './op_list/dynamic';
import * as evaluation from './op_list/evaluation';
import * as graph from './op_list/graph';
import * as image from './op_list/image';
import * as logical from './op_list/logical';
import * as matrices from './op_list/matrices';
import * as normalization from './op_list/normalization';
import * as reduction from './op_list/reduction';
import * as sliceJoin from './op_list/slice_join';
import * as spectral from './op_list/spectral';
import * as transformation from './op_list/transformation';
import {Graph, InputParamValue, Node, OpMapper, ParamValue} from './types';

const CONTROL_FLOW_OPS = ['Switch', 'Merge', 'Enter', 'Exit', 'NextIteration'];
const DYNAMIC_SHAPE_OPS =
    ['NonMaxSuppressionV2', 'NonMaxSuppressionV3', 'Where'];

export class OperationMapper {
  private static _instance: OperationMapper;

  private opMappers: {[key: string]: OpMapper};

  // Singleton instance for the mapper
  public static get Instance() {
    return this._instance || (this._instance = new this());
  }

  // Loads the op mapping from the JSON file.
  private constructor() {
    const ops = [
      arithmetic, basicMath, control, convolution, creation, dynamic,
      evaluation, logical, image, graph, matrices, normalization, reduction,
      sliceJoin, spectral, transformation
    ];
    const mappersJson: OpMapper[] = [].concat.apply([], ops.map(op => op.json));

    this.opMappers = mappersJson.reduce<{[key: string]: OpMapper}>(
        (map, mapper: OpMapper) => {
          map[mapper.tfOpName] = mapper;
          return map;
        },
        {});
  }

  private isControlFlow(node: tensorflow.INodeDef) {
    return CONTROL_FLOW_OPS.some(op => op === node.op);
  }

  private isDynamicShape(node: tensorflow.INodeDef) {
    return DYNAMIC_SHAPE_OPS.some(op => op === node.op);
  }
  // Converts the model from Tensorflow GraphDef to local representation for
  // TensorFlow.js API
  transformGraph(graph: tensorflow.IGraphDef): Graph {
    const tfNodes = graph.node;
    let withControlFlow = false;
    let withDynamicShape = false;
    const placeholders: Node[] = [];
    const weights: Node[] = [];
    const nodes = tfNodes.reduce<{[key: string]: Node}>((map, node) => {
      map[node.name] = this.mapNode(node);
      if (this.isControlFlow(node)) withControlFlow = true;
      if (this.isDynamicShape(node)) withDynamicShape = true;
      if (node.op === 'Placeholder') placeholders.push(map[node.name]);
      if (node.op === 'Const') weights.push(map[node.name]);
      return map;
    }, {});

    const inputs: Node[] = [];
    const outputs: Node[] = [];
    Object.keys(nodes).forEach(key => {
      const node = nodes[key];
      node.inputNames.forEach(name => {
        const [nodeName, ] = getNodeNameAndIndex(name);
        node.inputs.push(nodes[nodeName]);
        nodes[nodeName].children.push(node);
      });
      if (node.inputs.length === 0) inputs.push(node);
    });

    Object.keys(nodes).forEach(key => {
      const node = nodes[key];
      if (node.children.length === 0) outputs.push(node);
    });

    return {
      nodes,
      inputs,
      outputs,
      weights,
      placeholders,
      withControlFlow,
      withDynamicShape
    };
  }

  private mapNode(node: tensorflow.INodeDef): Node {
    const mapper = this.opMappers[node.op];
    if (mapper === undefined) {
      throw new Error('Tensorflow Op is not supported: ' + node.op);
    }
    const newNode: Node = {
      name: node.name,
      op: node.op,
      category: mapper.category,
      inputNames:
          (node.input ||
           []).map(input => input.startsWith('^') ? input.substr(1) : input),
      inputs: [],
      children: [],
      inputParams: {},
      attrParams: {}
    };
    if (node.attr == null) {
      node.attr = {};
    }

    if (mapper.inputs != null) {
      newNode.inputParams =
          mapper.inputs.reduce<{[key: string]: InputParamValue}>(
              (map, param) => {
                map[param.name] = {
                  type: param.type,
                  inputIndexStart: param.start,
                  inputIndexEnd: param.end
                };
                return map;
              },
              {});
    }
    if (mapper.attrs != null) {
      newNode.attrParams =
          mapper.attrs.reduce<{[key: string]: ParamValue}>((map, param) => {
            const type = param.type;
            let value = undefined;
            switch (param.type) {
              case 'string':
                value = this.getStringParam(
                    node.attr, param.tfName, param.defaultValue as string);

                if (value === undefined && !!param.tfDeprecatedName) {
                  value = this.getStringParam(
                      node.attr, param.tfDeprecatedName,
                      param.defaultValue as string);
                }
                break;
              case 'number':
                value = this.getNumberParam(
                    node.attr, param.tfName,
                    (param.defaultValue || 0) as number);
                if (value === undefined && !!param.tfDeprecatedName) {
                  value = this.getNumberParam(
                      node.attr, param.tfDeprecatedName,
                      param.defaultValue as number);
                }
                break;
              case 'number[]':
                value = this.getNumericArrayParam(
                    node.attr, param.tfName, param.defaultValue as number[]);
                if (value === undefined && !!param.tfDeprecatedName) {
                  value = this.getNumericArrayParam(
                      node.attr, param.tfDeprecatedName,
                      param.defaultValue as number[]);
                }
                break;
              case 'bool':
                value = this.getBoolParam(
                    node.attr, param.tfName, param.defaultValue as boolean);
                if (value === undefined && !!param.tfDeprecatedName) {
                  value = this.getBoolParam(
                      node.attr, param.tfDeprecatedName,
                      param.defaultValue as boolean);
                }
                break;
              case 'shape':
                value = this.getTensorShapeParam(
                    node.attr, param.tfName, param.defaultValue as number[]);
                if (value === undefined && !!param.tfDeprecatedName) {
                  value = this.getTensorShapeParam(
                      node.attr, param.tfDeprecatedName,
                      param.defaultValue as number[]);
                }
                break;
              case 'dtype':
                value = this.getDtypeParam(
                    node.attr, param.tfName, param.defaultValue as DataType);
                if (value === undefined && !!param.tfDeprecatedName) {
                  value = this.getDtypeParam(
                      node.attr, param.tfDeprecatedName,
                      param.defaultValue as DataType);
                }
                break;
              case 'tensor':
              case 'tensors':
                break;
              default:
                throw new Error(
                    `Unsupported param type: ${param.type} for op: ${node.op}`);
            }
            map[param.name] = {value, type};
            return map;
          }, {});
    }
    return newNode;
  }

  private decodeBase64(text: string): string {
    // tslint:disable-next-line:no-any
    const global = ENV.global as any;
    if (typeof global.atob !== 'undefined') {
      return global.atob(text);
    } else if (typeof Buffer !== 'undefined') {
      return new Buffer(text, 'base64').toString();
    } else {
      throw new Error(
          'Unable to decode base64 in this environment. ' +
          'Missing built-in atob() or Buffer()');
    }
  }

  private getStringParam(
      attrs: {[key: string]: tensorflow.IAttrValue}, name: string, def: string,
      keepCase = false): string {
    const param = attrs[name];
    if (param !== undefined) {
      const value = Array.isArray(param.s) ?
          String.fromCharCode.apply(null, param.s) :
          this.decodeBase64(param.s);
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
    const param = attrs[name] || {};
    const value = param['i'] ? param['i'] : (param['f'] ? param['f'] : def);
    return (typeof value === 'number') ?
        value :
        parseInt(value as string, 10) as number;
  }
  private getDtypeParam(
      attrs: {[key: string]: tensorflow.IAttrValue}, name: string,
      def: DataType): DataType {
    const param = attrs[name];
    if (param && param.type) {
      // tslint:disable-next-line:no-any
      let type: any = param.type;
      if (typeof (param.type) === 'string') {
        type = tensorflow.DataType[param.type];
      }
      switch (type) {
        case tensorflow.DataType.DT_FLOAT:
          return 'float32';
        case tensorflow.DataType.DT_INT32:
          return 'int32';
        case tensorflow.DataType.DT_BOOL:
          return 'bool';
        default:
          return def;
      }
    }
    return def;
  }
  private getTensorShapeParam(
      attrs: {[key: string]: tensorflow.IAttrValue}, name: string,
      def?: number[]): number[]|undefined {
    const param = attrs[name];
    if (param && param.shape) {
      if (param.shape.unknownRank) {
        return undefined;
      }
      if (param.shape.dim != null) {
        return param.shape.dim.map(
            dim => (typeof dim.size === 'number') ?
                dim.size :
                parseInt(dim.size as string, 10));
      }
    }
    return def;
  }

  private getNumericArrayParam(
      attrs: {[key: string]: tensorflow.IAttrValue}, name: string,
      def: number[]): number[] {
    const param = attrs[name];
    if (param) {
      return ((param.list.f && param.list.f.length ? param.list.f :
                                                     param.list.i))
                 .map(
                     v => (typeof v === 'number') ?
                         v :
                         parseInt(v as string, 10)) as number[];
    }
    return def;
  }
}
