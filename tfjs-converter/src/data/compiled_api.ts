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
 *
 * =============================================================================
 */

/* tslint:disable */

/** Properties of an Any. */
export declare interface IAny {
  /** Any typeUrl */
  typeUrl?: (string|null);

  /** Any value */
  value?: (Uint8Array|null);
}

/** DataType enum. */
export enum DataType {
  'DT_INVALID' = 0,
  'DT_FLOAT' = 1,
  'DT_DOUBLE' = 2,
  'DT_INT32' = 3,
  'DT_UINT8' = 4,
  'DT_INT16' = 5,
  'DT_INT8' = 6,
  'DT_STRING' = 7,
  'DT_COMPLEX64' = 8,
  'DT_INT64' = 9,
  'DT_BOOL' = 10,
  'DT_QINT8' = 11,
  'DT_QUINT8' = 12,
  'DT_QINT32' = 13,
  'DT_BFLOAT16' = 14,
  'DT_FLOAT_REF' = 101,
  'DT_DOUBLE_REF' = 102,
  'DT_INT32_REF' = 103,
  'DT_UINT8_REF' = 104,
  'DT_INT16_REF' = 105,
  'DT_INT8_REF' = 106,
  'DT_STRING_REF' = 107,
  'DT_COMPLEX64_REF' = 108,
  'DT_INT64_REF' = 109,
  'DT_BOOL_REF' = 110,
  'DT_QINT8_REF' = 111,
  'DT_QUINT8_REF' = 112,
  'DT_QINT32_REF' = 113,
  'DT_BFLOAT16_REF' = 114
}

/** Properties of a TensorShape. */
export declare interface ITensorShape {
  /** TensorShape dim */
  dim?: (TensorShape.IDim[]|null);

  /** TensorShape unknownRank */
  unknownRank?: (boolean|null);
}

export namespace TensorShape {
  /** Properties of a Dim. */
  export declare interface IDim {
    /** Dim size */
    size?: (number|string|null);

    /** Dim name */
    name?: (string|null);
  }
}

/** Properties of a Tensor. */
export declare interface ITensor {
  /** Tensor dtype */
  dtype?: (DataType|null);

  /** Tensor tensorShape */
  tensorShape?: (ITensorShape|null);

  /** Tensor versionNumber */
  versionNumber?: (number|null);

  /** Tensor tensorContent */
  tensorContent?: (Uint8Array|null);

  /** Tensor floatVal */
  floatVal?: (number[]|null);

  /** Tensor doubleVal */
  doubleVal?: (number[]|null);

  /** Tensor intVal */
  intVal?: (number[]|null);

  /** Tensor stringVal */
  stringVal?: (Uint8Array[]|null);

  /** Tensor scomplexVal */
  scomplexVal?: (number[]|null);

  /** Tensor int64Val */
  int64Val?: ((number | string)[]|null);

  /** Tensor boolVal */
  boolVal?: (boolean[]|null);

  /** Tensor uint32Val */
  uint32Val?: (number[]|null);

  /** Tensor uint64Val */
  uint64Val?: ((number | string)[]|null);
}

/** Properties of an AttrValue. */
export declare interface IAttrValue {
  /** AttrValue list */
  list?: (AttrValue.IListValue|null);

  /** AttrValue s */
  s?: (string|null);

  /** AttrValue i */
  i?: (number|string|null);

  /** AttrValue f */
  f?: (number|null);

  /** AttrValue b */
  b?: (boolean|null);

  /** AttrValue type */
  type?: (DataType|null);

  /** AttrValue shape */
  shape?: (ITensorShape|null);

  /** AttrValue tensor */
  tensor?: (ITensor|null);

  /** AttrValue placeholder */
  placeholder?: (string|null);

  /** AttrValue func */
  func?: (INameAttrList|null);
}

export namespace AttrValue {
  /** Properties of a ListValue. */
  export declare interface IListValue {
    /** ListValue s */
    s?: (string[]|null);

    /** ListValue i */
    i?: ((number | string)[]|null);

    /** ListValue f */
    f?: (number[]|null);

    /** ListValue b */
    b?: (boolean[]|null);

    /** ListValue type */
    type?: (DataType[]|null);

    /** ListValue shape */
    shape?: (ITensorShape[]|null);

    /** ListValue tensor */
    tensor?: (ITensor[]|null);

    /** ListValue func */
    func?: (INameAttrList[]|null);
  }
}

/** Properties of a NameAttrList. */
export declare interface INameAttrList {
  /** NameAttrList name */
  name?: (string|null);

  /** NameAttrList attr */
  attr?: ({[k: string]: IAttrValue}|null);
}

/** Properties of a NodeDef. */
export declare interface INodeDef {
  /** NodeDef name */
  name?: (string|null);

  /** NodeDef op */
  op?: (string|null);

  /** NodeDef input */
  input?: (string[]|null);

  /** NodeDef device */
  device?: (string|null);

  /** NodeDef attr */
  attr?: ({[k: string]: IAttrValue}|null);
}

/** Properties of a VersionDef. */
export declare interface IVersionDef {
  /** VersionDef producer */
  producer?: (number|null);

  /** VersionDef minConsumer */
  minConsumer?: (number|null);

  /** VersionDef badConsumers */
  badConsumers?: (number[]|null);
}

/** Properties of a GraphDef. */
export declare interface IGraphDef {
  /** GraphDef node */
  node?: (INodeDef[]|null);

  /** GraphDef versions */
  versions?: (IVersionDef|null);

  /** GraphDef library */
  library?: (IFunctionDefLibrary|null);
}

/** Properties of a CollectionDef. */
export declare interface ICollectionDef {
  /** CollectionDef nodeList */
  nodeList?: (CollectionDef.INodeList|null);

  /** CollectionDef bytesList */
  bytesList?: (CollectionDef.IBytesList|null);

  /** CollectionDef int64List */
  int64List?: (CollectionDef.IInt64List|null);

  /** CollectionDef floatList */
  floatList?: (CollectionDef.IFloatList|null);

  /** CollectionDef anyList */
  anyList?: (CollectionDef.IAnyList|null);
}

export namespace CollectionDef {
  /** Properties of a NodeList. */
  export declare interface INodeList {
    /** NodeList value */
    value?: (string[]|null);
  }

  /** Properties of a BytesList. */
  export declare interface IBytesList {
    /** BytesList value */
    value?: (Uint8Array[]|null);
  }

  /** Properties of an Int64List. */
  export declare interface IInt64List {
    /** Int64List value */
    value?: ((number | string)[]|null);
  }

  /** Properties of a FloatList. */
  export declare interface IFloatList {
    /** FloatList value */
    value?: (number[]|null);
  }

  /** Properties of an AnyList. */
  export declare interface IAnyList {
    /** AnyList value */
    value?: (IAny[]|null);
  }
}

/** Properties of a SaverDef. */
export declare interface ISaverDef {
  /** SaverDef filenameTensorName */
  filenameTensorName?: (string|null);

  /** SaverDef saveTensorName */
  saveTensorName?: (string|null);

  /** SaverDef restoreOpName */
  restoreOpName?: (string|null);

  /** SaverDef maxToKeep */
  maxToKeep?: (number|null);

  /** SaverDef sharded */
  sharded?: (boolean|null);

  /** SaverDef keepCheckpointEveryNHours */
  keepCheckpointEveryNHours?: (number|null);

  /** SaverDef version */
  version?: (SaverDef.CheckpointFormatVersion|null);
}

export namespace SaverDef {
  /** CheckpointFormatVersion enum. */
  export enum CheckpointFormatVersion {'LEGACY' = 0, 'V1' = 1, 'V2' = 2}
}

/** Properties of a TensorInfo. */
export declare interface ITensorInfo {
  /** TensorInfo name */
  name?: (string|null);

  /** TensorInfo cooSparse */
  cooSparse?: (TensorInfo.ICooSparse|null);

  /** TensorInfo dtype */
  dtype?: (DataType|null);

  /** TensorInfo tensorShape */
  tensorShape?: (ITensorShape|null);
}

export namespace TensorInfo {
  /** Properties of a CooSparse. */
  export declare interface ICooSparse {
    /** CooSparse valuesTensorName */
    valuesTensorName?: (string|null);

    /** CooSparse indicesTensorName */
    indicesTensorName?: (string|null);

    /** CooSparse denseShapeTensorName */
    denseShapeTensorName?: (string|null);
  }
}

/** Properties of a SignatureDef. */
export declare interface ISignatureDef {
  /** SignatureDef inputs */
  inputs?: ({[k: string]: ITensorInfo}|null);

  /** SignatureDef outputs */
  outputs?: ({[k: string]: ITensorInfo}|null);

  /** SignatureDef methodName */
  methodName?: (string|null);
}

/** Properties of an AssetFileDef. */
export declare interface IAssetFileDef {
  /** AssetFileDef tensorInfo */
  tensorInfo?: (ITensorInfo|null);

  /** AssetFileDef filename */
  filename?: (string|null);
}

/** Properties of an OpDef. */
export declare interface IOpDef {
  /** OpDef name */
  name?: (string|null);

  /** OpDef inputArg */
  inputArg?: (OpDef.IArgDef[]|null);

  /** OpDef outputArg */
  outputArg?: (OpDef.IArgDef[]|null);

  /** OpDef attr */
  attr?: (OpDef.IAttrDef[]|null);

  /** OpDef deprecation */
  deprecation?: (OpDef.IOpDeprecation|null);

  /** OpDef summary */
  summary?: (string|null);

  /** OpDef description */
  description?: (string|null);

  /** OpDef isCommutative */
  isCommutative?: (boolean|null);

  /** OpDef isAggregate */
  isAggregate?: (boolean|null);

  /** OpDef isStateful */
  isStateful?: (boolean|null);

  /** OpDef allowsUninitializedInput */
  allowsUninitializedInput?: (boolean|null);
}

export namespace OpDef {
  /** Properties of an ArgDef. */
  export declare interface IArgDef {
    /** ArgDef name */
    name?: (string|null);

    /** ArgDef description */
    description?: (string|null);

    /** ArgDef type */
    type?: (DataType|null);

    /** ArgDef typeAttr */
    typeAttr?: (string|null);

    /** ArgDef numberAttr */
    numberAttr?: (string|null);

    /** ArgDef typeListAttr */
    typeListAttr?: (string|null);

    /** ArgDef isRef */
    isRef?: (boolean|null);
  }

  /** Properties of an AttrDef. */
  export declare interface IAttrDef {
    /** AttrDef name */
    name?: (string|null);

    /** AttrDef type */
    type?: (string|null);

    /** AttrDef defaultValue */
    defaultValue?: (IAttrValue|null);

    /** AttrDef description */
    description?: (string|null);

    /** AttrDef hasMinimum */
    hasMinimum?: (boolean|null);

    /** AttrDef minimum */
    minimum?: (number|string|null);

    /** AttrDef allowedValues */
    allowedValues?: (IAttrValue|null);
  }

  /** Properties of an OpDeprecation. */
  export declare interface IOpDeprecation {
    /** OpDeprecation version */
    version?: (number|null);

    /** OpDeprecation explanation */
    explanation?: (string|null);
  }
}

/** Properties of an OpList. */
export declare interface IOpList {
  /** OpList op */
  op?: (IOpDef[]|null);
}

/** Properties of a MetaGraphDef. */
export declare interface IMetaGraphDef {
  /** MetaGraphDef metaInfoDef */
  metaInfoDef?: (MetaGraphDef.IMetaInfoDef|null);

  /** MetaGraphDef graphDef */
  graphDef?: (IGraphDef|null);

  /** MetaGraphDef saverDef */
  saverDef?: (ISaverDef|null);

  /** MetaGraphDef collectionDef */
  collectionDef?: ({[k: string]: ICollectionDef}|null);

  /** MetaGraphDef signatureDef */
  signatureDef?: ({[k: string]: ISignatureDef}|null);

  /** MetaGraphDef assetFileDef */
  assetFileDef?: (IAssetFileDef[]|null);
}

export namespace MetaGraphDef {
  /** Properties of a MetaInfoDef. */
  export declare interface IMetaInfoDef {
    /** MetaInfoDef metaGraphVersion */
    metaGraphVersion?: (string|null);

    /** MetaInfoDef strippedOpList */
    strippedOpList?: (IOpList|null);

    /** MetaInfoDef anyInfo */
    anyInfo?: (IAny|null);

    /** MetaInfoDef tags */
    tags?: (string[]|null);

    /** MetaInfoDef tensorflowVersion */
    tensorflowVersion?: (string|null);

    /** MetaInfoDef tensorflowGitVersion */
    tensorflowGitVersion?: (string|null);
  }
}

/** Properties of a SavedModel. */
export declare interface ISavedModel {
  /** SavedModel savedModelSchemaVersion */
  savedModelSchemaVersion?: (number|string|null);

  /** SavedModel metaGraphs */
  metaGraphs?: (IMetaGraphDef[]|null);
}

/** Properties of a FunctionDefLibrary. */
export declare interface IFunctionDefLibrary {
  /** FunctionDefLibrary function */
  'function'?: (IFunctionDef[]|null);

  /** FunctionDefLibrary gradient */
  gradient?: (IGradientDef[]|null);
}

/** Properties of a FunctionDef. */
export declare interface IFunctionDef {
  /** FunctionDef signature */
  signature?: (IOpDef|null);

  /** FunctionDef attr */
  attr?: ({[k: string]: IAttrValue}|null);

  /** FunctionDef nodeDef */
  nodeDef?: (INodeDef[]|null);

  /** FunctionDef ret */
  ret?: ({[k: string]: string}|null);
}

/** Properties of a GradientDef. */
export declare interface IGradientDef {
  /** GradientDef functionName */
  functionName?: (string|null);

  /** GradientDef gradientFunc */
  gradientFunc?: (string|null);
}
