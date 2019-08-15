import * as $protobuf from 'protobufjs';

/** Namespace tensorflow. */
export namespace tensorflow {
  /** Properties of an Any. */
  interface IAny {
    /** Any typeUrl */
    typeUrl?: (string|null);

    /** Any value */
    value?: (Uint8Array|null);
  }

  /** Represents an Any. */
  class Any implements IAny {
    /**
     * Constructs a new Any.
     * @param [p] Properties to set
     */
    constructor(p?: tensorflow.IAny);

    /** Any typeUrl. */
    public typeUrl: string;

    /** Any value. */
    public value: Uint8Array;

    /**
     * Decodes an Any message from the specified reader or buffer.
     * @param r Reader or buffer to decode from
     * @param [l] Message length if known beforehand
     * @returns Any
     * @throws {Error} If the payload is not a reader or valid buffer
     * @throws {$protobuf.util.ProtocolError} If required fields are missing
     */
    public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
        tensorflow.Any;
  }

  /** DataType enum. */
  enum DataType {
    DT_INVALID = 0,
    DT_FLOAT = 1,
    DT_DOUBLE = 2,
    DT_INT32 = 3,
    DT_UINT8 = 4,
    DT_INT16 = 5,
    DT_INT8 = 6,
    DT_STRING = 7,
    DT_COMPLEX64 = 8,
    DT_INT64 = 9,
    DT_BOOL = 10,
    DT_QINT8 = 11,
    DT_QUINT8 = 12,
    DT_QINT32 = 13,
    DT_BFLOAT16 = 14,
    DT_FLOAT_REF = 101,
    DT_DOUBLE_REF = 102,
    DT_INT32_REF = 103,
    DT_UINT8_REF = 104,
    DT_INT16_REF = 105,
    DT_INT8_REF = 106,
    DT_STRING_REF = 107,
    DT_COMPLEX64_REF = 108,
    DT_INT64_REF = 109,
    DT_BOOL_REF = 110,
    DT_QINT8_REF = 111,
    DT_QUINT8_REF = 112,
    DT_QINT32_REF = 113,
    DT_BFLOAT16_REF = 114
  }

  /** Properties of a TensorShape. */
  interface ITensorShape {
    /** TensorShape dim */
    dim?: (tensorflow.TensorShape.IDim[]|null);

    /** TensorShape unknownRank */
    unknownRank?: (boolean|null);
  }

  /** Represents a TensorShape. */
  class TensorShape implements ITensorShape {
    /**
     * Constructs a new TensorShape.
     * @param [p] Properties to set
     */
    constructor(p?: tensorflow.ITensorShape);

    /** TensorShape dim. */
    public dim: tensorflow.TensorShape.IDim[];

    /** TensorShape unknownRank. */
    public unknownRank: boolean;

    /**
     * Decodes a TensorShape message from the specified reader or buffer.
     * @param r Reader or buffer to decode from
     * @param [l] Message length if known beforehand
     * @returns TensorShape
     * @throws {Error} If the payload is not a reader or valid buffer
     * @throws {$protobuf.util.ProtocolError} If required fields are missing
     */
    public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
        tensorflow.TensorShape;
  }

  namespace TensorShape {

    /** Properties of a Dim. */
    interface IDim {
      /** Dim size */
      size?: (number|Long|null);

      /** Dim name */
      name?: (string|null);
    }

    /** Represents a Dim. */
    class Dim implements IDim {
      /**
       * Constructs a new Dim.
       * @param [p] Properties to set
       */
      constructor(p?: tensorflow.TensorShape.IDim);

      /** Dim size. */
      public size: (number|Long);

      /** Dim name. */
      public name: string;

      /**
       * Decodes a Dim message from the specified reader or buffer.
       * @param r Reader or buffer to decode from
       * @param [l] Message length if known beforehand
       * @returns Dim
       * @throws {Error} If the payload is not a reader or valid buffer
       * @throws {$protobuf.util.ProtocolError} If required fields are missing
       */
      public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
          tensorflow.TensorShape.Dim;
    }
  }

  /** Properties of a Tensor. */
  interface ITensor {
    /** Tensor dtype */
    dtype?: (tensorflow.DataType|null);

    /** Tensor tensorShape */
    tensorShape?: (tensorflow.ITensorShape|null);

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
    int64Val?: ((number | Long)[]|null);

    /** Tensor boolVal */
    boolVal?: (boolean[]|null);

    /** Tensor uint32Val */
    uint32Val?: (number[]|null);

    /** Tensor uint64Val */
    uint64Val?: ((number | Long)[]|null);
  }

  /** Represents a Tensor. */
  class Tensor implements ITensor {
    /**
     * Constructs a new Tensor.
     * @param [p] Properties to set
     */
    constructor(p?: tensorflow.ITensor);

    /** Tensor dtype. */
    public dtype: tensorflow.DataType;

    /** Tensor tensorShape. */
    public tensorShape?: (tensorflow.ITensorShape|null);

    /** Tensor versionNumber. */
    public versionNumber: number;

    /** Tensor tensorContent. */
    public tensorContent: Uint8Array;

    /** Tensor floatVal. */
    public floatVal: number[];

    /** Tensor doubleVal. */
    public doubleVal: number[];

    /** Tensor intVal. */
    public intVal: number[];

    /** Tensor stringVal. */
    public stringVal: Uint8Array[];

    /** Tensor scomplexVal. */
    public scomplexVal: number[];

    /** Tensor int64Val. */
    public int64Val: (number|Long)[];

    /** Tensor boolVal. */
    public boolVal: boolean[];

    /** Tensor uint32Val. */
    public uint32Val: number[];

    /** Tensor uint64Val. */
    public uint64Val: (number|Long)[];

    /**
     * Decodes a Tensor message from the specified reader or buffer.
     * @param r Reader or buffer to decode from
     * @param [l] Message length if known beforehand
     * @returns Tensor
     * @throws {Error} If the payload is not a reader or valid buffer
     * @throws {$protobuf.util.ProtocolError} If required fields are missing
     */
    public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
        tensorflow.Tensor;
  }

  /** Properties of an AttrValue. */
  interface IAttrValue {
    /** AttrValue list */
    list?: (tensorflow.AttrValue.IListValue|null);

    /** AttrValue s */
    s?: (Uint8Array|null);

    /** AttrValue i */
    i?: (number|Long|null);

    /** AttrValue f */
    f?: (number|null);

    /** AttrValue b */
    b?: (boolean|null);

    /** AttrValue type */
    type?: (tensorflow.DataType|null);

    /** AttrValue shape */
    shape?: (tensorflow.ITensorShape|null);

    /** AttrValue tensor */
    tensor?: (tensorflow.ITensor|null);

    /** AttrValue placeholder */
    placeholder?: (string|null);

    /** AttrValue func */
    func?: (tensorflow.INameAttrList|null);
  }

  /** Represents an AttrValue. */
  class AttrValue implements IAttrValue {
    /**
     * Constructs a new AttrValue.
     * @param [p] Properties to set
     */
    constructor(p?: tensorflow.IAttrValue);

    /** AttrValue list. */
    public list?: (tensorflow.AttrValue.IListValue|null);

    /** AttrValue s. */
    public s: Uint8Array;

    /** AttrValue i. */
    public i: (number|Long);

    /** AttrValue f. */
    public f: number;

    /** AttrValue b. */
    public b: boolean;

    /** AttrValue type. */
    public type: tensorflow.DataType;

    /** AttrValue shape. */
    public shape?: (tensorflow.ITensorShape|null);

    /** AttrValue tensor. */
    public tensor?: (tensorflow.ITensor|null);

    /** AttrValue placeholder. */
    public placeholder: string;

    /** AttrValue func. */
    public func?: (tensorflow.INameAttrList|null);

    /** AttrValue value. */
    public value?: ('list'|'s'|'i'|'f'|'b'|'type'|'shape'|'tensor'|
                    'placeholder'|'func');

    /**
     * Decodes an AttrValue message from the specified reader or buffer.
     * @param r Reader or buffer to decode from
     * @param [l] Message length if known beforehand
     * @returns AttrValue
     * @throws {Error} If the payload is not a reader or valid buffer
     * @throws {$protobuf.util.ProtocolError} If required fields are missing
     */
    public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
        tensorflow.AttrValue;
  }

  namespace AttrValue {

    /** Properties of a ListValue. */
    interface IListValue {
      /** ListValue s */
      s?: (Uint8Array[]|null);

      /** ListValue i */
      i?: ((number | Long)[]|null);

      /** ListValue f */
      f?: (number[]|null);

      /** ListValue b */
      b?: (boolean[]|null);

      /** ListValue type */
      type?: (tensorflow.DataType[]|null);

      /** ListValue shape */
      shape?: (tensorflow.ITensorShape[]|null);

      /** ListValue tensor */
      tensor?: (tensorflow.ITensor[]|null);

      /** ListValue func */
      func?: (tensorflow.INameAttrList[]|null);
    }

    /** Represents a ListValue. */
    class ListValue implements IListValue {
      /**
       * Constructs a new ListValue.
       * @param [p] Properties to set
       */
      constructor(p?: tensorflow.AttrValue.IListValue);

      /** ListValue s. */
      public s: Uint8Array[];

      /** ListValue i. */
      public i: (number|Long)[];

      /** ListValue f. */
      public f: number[];

      /** ListValue b. */
      public b: boolean[];

      /** ListValue type. */
      public type: tensorflow.DataType[];

      /** ListValue shape. */
      public shape: tensorflow.ITensorShape[];

      /** ListValue tensor. */
      public tensor: tensorflow.ITensor[];

      /** ListValue func. */
      public func: tensorflow.INameAttrList[];

      /**
       * Decodes a ListValue message from the specified reader or buffer.
       * @param r Reader or buffer to decode from
       * @param [l] Message length if known beforehand
       * @returns ListValue
       * @throws {Error} If the payload is not a reader or valid buffer
       * @throws {$protobuf.util.ProtocolError} If required fields are missing
       */
      public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
          tensorflow.AttrValue.ListValue;
    }
  }

  /** Properties of a NameAttrList. */
  interface INameAttrList {
    /** NameAttrList name */
    name?: (string|null);

    /** NameAttrList attr */
    attr?: ({[k: string]: tensorflow.IAttrValue}|null);
  }

  /** Represents a NameAttrList. */
  class NameAttrList implements INameAttrList {
    /**
     * Constructs a new NameAttrList.
     * @param [p] Properties to set
     */
    constructor(p?: tensorflow.INameAttrList);

    /** NameAttrList name. */
    public name: string;

    /** NameAttrList attr. */
    public attr: {[k: string]: tensorflow.IAttrValue};

    /**
     * Decodes a NameAttrList message from the specified reader or buffer.
     * @param r Reader or buffer to decode from
     * @param [l] Message length if known beforehand
     * @returns NameAttrList
     * @throws {Error} If the payload is not a reader or valid buffer
     * @throws {$protobuf.util.ProtocolError} If required fields are missing
     */
    public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
        tensorflow.NameAttrList;
  }

  /** Properties of a NodeDef. */
  interface INodeDef {
    /** NodeDef name */
    name?: (string|null);

    /** NodeDef op */
    op?: (string|null);

    /** NodeDef input */
    input?: (string[]|null);

    /** NodeDef device */
    device?: (string|null);

    /** NodeDef attr */
    attr?: ({[k: string]: tensorflow.IAttrValue}|null);
  }

  /** Represents a NodeDef. */
  class NodeDef implements INodeDef {
    /**
     * Constructs a new NodeDef.
     * @param [p] Properties to set
     */
    constructor(p?: tensorflow.INodeDef);

    /** NodeDef name. */
    public name: string;

    /** NodeDef op. */
    public op: string;

    /** NodeDef input. */
    public input: string[];

    /** NodeDef device. */
    public device: string;

    /** NodeDef attr. */
    public attr: {[k: string]: tensorflow.IAttrValue};

    /**
     * Decodes a NodeDef message from the specified reader or buffer.
     * @param r Reader or buffer to decode from
     * @param [l] Message length if known beforehand
     * @returns NodeDef
     * @throws {Error} If the payload is not a reader or valid buffer
     * @throws {$protobuf.util.ProtocolError} If required fields are missing
     */
    public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
        tensorflow.NodeDef;
  }

  /** Properties of a VersionDef. */
  interface IVersionDef {
    /** VersionDef producer */
    producer?: (number|null);

    /** VersionDef minConsumer */
    minConsumer?: (number|null);

    /** VersionDef badConsumers */
    badConsumers?: (number[]|null);
  }

  /** Represents a VersionDef. */
  class VersionDef implements IVersionDef {
    /**
     * Constructs a new VersionDef.
     * @param [p] Properties to set
     */
    constructor(p?: tensorflow.IVersionDef);

    /** VersionDef producer. */
    public producer: number;

    /** VersionDef minConsumer. */
    public minConsumer: number;

    /** VersionDef badConsumers. */
    public badConsumers: number[];

    /**
     * Decodes a VersionDef message from the specified reader or buffer.
     * @param r Reader or buffer to decode from
     * @param [l] Message length if known beforehand
     * @returns VersionDef
     * @throws {Error} If the payload is not a reader or valid buffer
     * @throws {$protobuf.util.ProtocolError} If required fields are missing
     */
    public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
        tensorflow.VersionDef;
  }

  /** Properties of a GraphDef. */
  interface IGraphDef {
    /** GraphDef node */
    node?: (tensorflow.INodeDef[]|null);

    /** GraphDef versions */
    versions?: (tensorflow.IVersionDef|null);

    /** GraphDef library */
    library?: (tensorflow.IFunctionDefLibrary|null);
  }

  /** Represents a GraphDef. */
  class GraphDef implements IGraphDef {
    /**
     * Constructs a new GraphDef.
     * @param [p] Properties to set
     */
    constructor(p?: tensorflow.IGraphDef);

    /** GraphDef node. */
    public node: tensorflow.INodeDef[];

    /** GraphDef versions. */
    public versions?: (tensorflow.IVersionDef|null);

    /** GraphDef library. */
    public library?: (tensorflow.IFunctionDefLibrary|null);

    /**
     * Decodes a GraphDef message from the specified reader or buffer.
     * @param r Reader or buffer to decode from
     * @param [l] Message length if known beforehand
     * @returns GraphDef
     * @throws {Error} If the payload is not a reader or valid buffer
     * @throws {$protobuf.util.ProtocolError} If required fields are missing
     */
    public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
        tensorflow.GraphDef;
  }

  /** Properties of a CollectionDef. */
  interface ICollectionDef {
    /** CollectionDef nodeList */
    nodeList?: (tensorflow.CollectionDef.INodeList|null);

    /** CollectionDef bytesList */
    bytesList?: (tensorflow.CollectionDef.IBytesList|null);

    /** CollectionDef int64List */
    int64List?: (tensorflow.CollectionDef.IInt64List|null);

    /** CollectionDef floatList */
    floatList?: (tensorflow.CollectionDef.IFloatList|null);

    /** CollectionDef anyList */
    anyList?: (tensorflow.CollectionDef.IAnyList|null);
  }

  /** Represents a CollectionDef. */
  class CollectionDef implements ICollectionDef {
    /**
     * Constructs a new CollectionDef.
     * @param [p] Properties to set
     */
    constructor(p?: tensorflow.ICollectionDef);

    /** CollectionDef nodeList. */
    public nodeList?: (tensorflow.CollectionDef.INodeList|null);

    /** CollectionDef bytesList. */
    public bytesList?: (tensorflow.CollectionDef.IBytesList|null);

    /** CollectionDef int64List. */
    public int64List?: (tensorflow.CollectionDef.IInt64List|null);

    /** CollectionDef floatList. */
    public floatList?: (tensorflow.CollectionDef.IFloatList|null);

    /** CollectionDef anyList. */
    public anyList?: (tensorflow.CollectionDef.IAnyList|null);

    /** CollectionDef kind. */
    public kind?: ('nodeList'|'bytesList'|'int64List'|'floatList'|'anyList');

    /**
     * Decodes a CollectionDef message from the specified reader or buffer.
     * @param r Reader or buffer to decode from
     * @param [l] Message length if known beforehand
     * @returns CollectionDef
     * @throws {Error} If the payload is not a reader or valid buffer
     * @throws {$protobuf.util.ProtocolError} If required fields are missing
     */
    public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
        tensorflow.CollectionDef;
  }

  namespace CollectionDef {

    /** Properties of a NodeList. */
    interface INodeList {
      /** NodeList value */
      value?: (string[]|null);
    }

    /** Represents a NodeList. */
    class NodeList implements INodeList {
      /**
       * Constructs a new NodeList.
       * @param [p] Properties to set
       */
      constructor(p?: tensorflow.CollectionDef.INodeList);

      /** NodeList value. */
      public value: string[];

      /**
       * Decodes a NodeList message from the specified reader or buffer.
       * @param r Reader or buffer to decode from
       * @param [l] Message length if known beforehand
       * @returns NodeList
       * @throws {Error} If the payload is not a reader or valid buffer
       * @throws {$protobuf.util.ProtocolError} If required fields are missing
       */
      public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
          tensorflow.CollectionDef.NodeList;
    }

    /** Properties of a BytesList. */
    interface IBytesList {
      /** BytesList value */
      value?: (Uint8Array[]|null);
    }

    /** Represents a BytesList. */
    class BytesList implements IBytesList {
      /**
       * Constructs a new BytesList.
       * @param [p] Properties to set
       */
      constructor(p?: tensorflow.CollectionDef.IBytesList);

      /** BytesList value. */
      public value: Uint8Array[];

      /**
       * Decodes a BytesList message from the specified reader or buffer.
       * @param r Reader or buffer to decode from
       * @param [l] Message length if known beforehand
       * @returns BytesList
       * @throws {Error} If the payload is not a reader or valid buffer
       * @throws {$protobuf.util.ProtocolError} If required fields are missing
       */
      public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
          tensorflow.CollectionDef.BytesList;
    }

    /** Properties of an Int64List. */
    interface IInt64List {
      /** Int64List value */
      value?: ((number | Long)[]|null);
    }

    /** Represents an Int64List. */
    class Int64List implements IInt64List {
      /**
       * Constructs a new Int64List.
       * @param [p] Properties to set
       */
      constructor(p?: tensorflow.CollectionDef.IInt64List);

      /** Int64List value. */
      public value: (number|Long)[];

      /**
       * Decodes an Int64List message from the specified reader or buffer.
       * @param r Reader or buffer to decode from
       * @param [l] Message length if known beforehand
       * @returns Int64List
       * @throws {Error} If the payload is not a reader or valid buffer
       * @throws {$protobuf.util.ProtocolError} If required fields are missing
       */
      public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
          tensorflow.CollectionDef.Int64List;
    }

    /** Properties of a FloatList. */
    interface IFloatList {
      /** FloatList value */
      value?: (number[]|null);
    }

    /** Represents a FloatList. */
    class FloatList implements IFloatList {
      /**
       * Constructs a new FloatList.
       * @param [p] Properties to set
       */
      constructor(p?: tensorflow.CollectionDef.IFloatList);

      /** FloatList value. */
      public value: number[];

      /**
       * Decodes a FloatList message from the specified reader or buffer.
       * @param r Reader or buffer to decode from
       * @param [l] Message length if known beforehand
       * @returns FloatList
       * @throws {Error} If the payload is not a reader or valid buffer
       * @throws {$protobuf.util.ProtocolError} If required fields are missing
       */
      public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
          tensorflow.CollectionDef.FloatList;
    }

    /** Properties of an AnyList. */
    interface IAnyList {
      /** AnyList value */
      value?: (tensorflow.IAny[]|null);
    }

    /** Represents an AnyList. */
    class AnyList implements IAnyList {
      /**
       * Constructs a new AnyList.
       * @param [p] Properties to set
       */
      constructor(p?: tensorflow.CollectionDef.IAnyList);

      /** AnyList value. */
      public value: tensorflow.IAny[];

      /**
       * Decodes an AnyList message from the specified reader or buffer.
       * @param r Reader or buffer to decode from
       * @param [l] Message length if known beforehand
       * @returns AnyList
       * @throws {Error} If the payload is not a reader or valid buffer
       * @throws {$protobuf.util.ProtocolError} If required fields are missing
       */
      public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
          tensorflow.CollectionDef.AnyList;
    }
  }

  /** Properties of a SaverDef. */
  interface ISaverDef {
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
    version?: (tensorflow.SaverDef.CheckpointFormatVersion|null);
  }

  /** Represents a SaverDef. */
  class SaverDef implements ISaverDef {
    /**
     * Constructs a new SaverDef.
     * @param [p] Properties to set
     */
    constructor(p?: tensorflow.ISaverDef);

    /** SaverDef filenameTensorName. */
    public filenameTensorName: string;

    /** SaverDef saveTensorName. */
    public saveTensorName: string;

    /** SaverDef restoreOpName. */
    public restoreOpName: string;

    /** SaverDef maxToKeep. */
    public maxToKeep: number;

    /** SaverDef sharded. */
    public sharded: boolean;

    /** SaverDef keepCheckpointEveryNHours. */
    public keepCheckpointEveryNHours: number;

    /** SaverDef version. */
    public version: tensorflow.SaverDef.CheckpointFormatVersion;

    /**
     * Decodes a SaverDef message from the specified reader or buffer.
     * @param r Reader or buffer to decode from
     * @param [l] Message length if known beforehand
     * @returns SaverDef
     * @throws {Error} If the payload is not a reader or valid buffer
     * @throws {$protobuf.util.ProtocolError} If required fields are missing
     */
    public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
        tensorflow.SaverDef;
  }

  namespace SaverDef {

    /** CheckpointFormatVersion enum. */
    enum CheckpointFormatVersion { LEGACY = 0, V1 = 1, V2 = 2 }
  }

  /** Properties of a TensorInfo. */
  interface ITensorInfo {
    /** TensorInfo name */
    name?: (string|null);

    /** TensorInfo cooSparse */
    cooSparse?: (tensorflow.TensorInfo.ICooSparse|null);

    /** TensorInfo dtype */
    dtype?: (tensorflow.DataType|null);

    /** TensorInfo tensorShape */
    tensorShape?: (tensorflow.ITensorShape|null);
  }

  /** Represents a TensorInfo. */
  class TensorInfo implements ITensorInfo {
    /**
     * Constructs a new TensorInfo.
     * @param [p] Properties to set
     */
    constructor(p?: tensorflow.ITensorInfo);

    /** TensorInfo name. */
    public name: string;

    /** TensorInfo cooSparse. */
    public cooSparse?: (tensorflow.TensorInfo.ICooSparse|null);

    /** TensorInfo dtype. */
    public dtype: tensorflow.DataType;

    /** TensorInfo tensorShape. */
    public tensorShape?: (tensorflow.ITensorShape|null);

    /** TensorInfo encoding. */
    public encoding?: ('name'|'cooSparse');

    /**
     * Decodes a TensorInfo message from the specified reader or buffer.
     * @param r Reader or buffer to decode from
     * @param [l] Message length if known beforehand
     * @returns TensorInfo
     * @throws {Error} If the payload is not a reader or valid buffer
     * @throws {$protobuf.util.ProtocolError} If required fields are missing
     */
    public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
        tensorflow.TensorInfo;
  }

  namespace TensorInfo {

    /** Properties of a CooSparse. */
    interface ICooSparse {
      /** CooSparse valuesTensorName */
      valuesTensorName?: (string|null);

      /** CooSparse indicesTensorName */
      indicesTensorName?: (string|null);

      /** CooSparse denseShapeTensorName */
      denseShapeTensorName?: (string|null);
    }

    /** Represents a CooSparse. */
    class CooSparse implements ICooSparse {
      /**
       * Constructs a new CooSparse.
       * @param [p] Properties to set
       */
      constructor(p?: tensorflow.TensorInfo.ICooSparse);

      /** CooSparse valuesTensorName. */
      public valuesTensorName: string;

      /** CooSparse indicesTensorName. */
      public indicesTensorName: string;

      /** CooSparse denseShapeTensorName. */
      public denseShapeTensorName: string;

      /**
       * Decodes a CooSparse message from the specified reader or buffer.
       * @param r Reader or buffer to decode from
       * @param [l] Message length if known beforehand
       * @returns CooSparse
       * @throws {Error} If the payload is not a reader or valid buffer
       * @throws {$protobuf.util.ProtocolError} If required fields are missing
       */
      public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
          tensorflow.TensorInfo.CooSparse;
    }
  }

  /** Properties of a SignatureDef. */
  interface ISignatureDef {
    /** SignatureDef inputs */
    inputs?: ({[k: string]: tensorflow.ITensorInfo}|null);

    /** SignatureDef outputs */
    outputs?: ({[k: string]: tensorflow.ITensorInfo}|null);

    /** SignatureDef methodName */
    methodName?: (string|null);
  }

  /** Represents a SignatureDef. */
  class SignatureDef implements ISignatureDef {
    /**
     * Constructs a new SignatureDef.
     * @param [p] Properties to set
     */
    constructor(p?: tensorflow.ISignatureDef);

    /** SignatureDef inputs. */
    public inputs: {[k: string]: tensorflow.ITensorInfo};

    /** SignatureDef outputs. */
    public outputs: {[k: string]: tensorflow.ITensorInfo};

    /** SignatureDef methodName. */
    public methodName: string;

    /**
     * Decodes a SignatureDef message from the specified reader or buffer.
     * @param r Reader or buffer to decode from
     * @param [l] Message length if known beforehand
     * @returns SignatureDef
     * @throws {Error} If the payload is not a reader or valid buffer
     * @throws {$protobuf.util.ProtocolError} If required fields are missing
     */
    public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
        tensorflow.SignatureDef;
  }

  /** Properties of an AssetFileDef. */
  interface IAssetFileDef {
    /** AssetFileDef tensorInfo */
    tensorInfo?: (tensorflow.ITensorInfo|null);

    /** AssetFileDef filename */
    filename?: (string|null);
  }

  /** Represents an AssetFileDef. */
  class AssetFileDef implements IAssetFileDef {
    /**
     * Constructs a new AssetFileDef.
     * @param [p] Properties to set
     */
    constructor(p?: tensorflow.IAssetFileDef);

    /** AssetFileDef tensorInfo. */
    public tensorInfo?: (tensorflow.ITensorInfo|null);

    /** AssetFileDef filename. */
    public filename: string;

    /**
     * Decodes an AssetFileDef message from the specified reader or buffer.
     * @param r Reader or buffer to decode from
     * @param [l] Message length if known beforehand
     * @returns AssetFileDef
     * @throws {Error} If the payload is not a reader or valid buffer
     * @throws {$protobuf.util.ProtocolError} If required fields are missing
     */
    public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
        tensorflow.AssetFileDef;
  }

  /** Properties of an OpDef. */
  interface IOpDef {
    /** OpDef name */
    name?: (string|null);

    /** OpDef inputArg */
    inputArg?: (tensorflow.OpDef.IArgDef[]|null);

    /** OpDef outputArg */
    outputArg?: (tensorflow.OpDef.IArgDef[]|null);

    /** OpDef attr */
    attr?: (tensorflow.OpDef.IAttrDef[]|null);

    /** OpDef deprecation */
    deprecation?: (tensorflow.OpDef.IOpDeprecation|null);

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

  /** Represents an OpDef. */
  class OpDef implements IOpDef {
    /**
     * Constructs a new OpDef.
     * @param [p] Properties to set
     */
    constructor(p?: tensorflow.IOpDef);

    /** OpDef name. */
    public name: string;

    /** OpDef inputArg. */
    public inputArg: tensorflow.OpDef.IArgDef[];

    /** OpDef outputArg. */
    public outputArg: tensorflow.OpDef.IArgDef[];

    /** OpDef attr. */
    public attr: tensorflow.OpDef.IAttrDef[];

    /** OpDef deprecation. */
    public deprecation?: (tensorflow.OpDef.IOpDeprecation|null);

    /** OpDef summary. */
    public summary: string;

    /** OpDef description. */
    public description: string;

    /** OpDef isCommutative. */
    public isCommutative: boolean;

    /** OpDef isAggregate. */
    public isAggregate: boolean;

    /** OpDef isStateful. */
    public isStateful: boolean;

    /** OpDef allowsUninitializedInput. */
    public allowsUninitializedInput: boolean;

    /**
     * Decodes an OpDef message from the specified reader or buffer.
     * @param r Reader or buffer to decode from
     * @param [l] Message length if known beforehand
     * @returns OpDef
     * @throws {Error} If the payload is not a reader or valid buffer
     * @throws {$protobuf.util.ProtocolError} If required fields are missing
     */
    public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
        tensorflow.OpDef;
  }

  namespace OpDef {

    /** Properties of an ArgDef. */
    interface IArgDef {
      /** ArgDef name */
      name?: (string|null);

      /** ArgDef description */
      description?: (string|null);

      /** ArgDef type */
      type?: (tensorflow.DataType|null);

      /** ArgDef typeAttr */
      typeAttr?: (string|null);

      /** ArgDef numberAttr */
      numberAttr?: (string|null);

      /** ArgDef typeListAttr */
      typeListAttr?: (string|null);

      /** ArgDef isRef */
      isRef?: (boolean|null);
    }

    /** Represents an ArgDef. */
    class ArgDef implements IArgDef {
      /**
       * Constructs a new ArgDef.
       * @param [p] Properties to set
       */
      constructor(p?: tensorflow.OpDef.IArgDef);

      /** ArgDef name. */
      public name: string;

      /** ArgDef description. */
      public description: string;

      /** ArgDef type. */
      public type: tensorflow.DataType;

      /** ArgDef typeAttr. */
      public typeAttr: string;

      /** ArgDef numberAttr. */
      public numberAttr: string;

      /** ArgDef typeListAttr. */
      public typeListAttr: string;

      /** ArgDef isRef. */
      public isRef: boolean;

      /**
       * Decodes an ArgDef message from the specified reader or buffer.
       * @param r Reader or buffer to decode from
       * @param [l] Message length if known beforehand
       * @returns ArgDef
       * @throws {Error} If the payload is not a reader or valid buffer
       * @throws {$protobuf.util.ProtocolError} If required fields are missing
       */
      public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
          tensorflow.OpDef.ArgDef;
    }

    /** Properties of an AttrDef. */
    interface IAttrDef {
      /** AttrDef name */
      name?: (string|null);

      /** AttrDef type */
      type?: (string|null);

      /** AttrDef defaultValue */
      defaultValue?: (tensorflow.IAttrValue|null);

      /** AttrDef description */
      description?: (string|null);

      /** AttrDef hasMinimum */
      hasMinimum?: (boolean|null);

      /** AttrDef minimum */
      minimum?: (number|Long|null);

      /** AttrDef allowedValues */
      allowedValues?: (tensorflow.IAttrValue|null);
    }

    /** Represents an AttrDef. */
    class AttrDef implements IAttrDef {
      /**
       * Constructs a new AttrDef.
       * @param [p] Properties to set
       */
      constructor(p?: tensorflow.OpDef.IAttrDef);

      /** AttrDef name. */
      public name: string;

      /** AttrDef type. */
      public type: string;

      /** AttrDef defaultValue. */
      public defaultValue?: (tensorflow.IAttrValue|null);

      /** AttrDef description. */
      public description: string;

      /** AttrDef hasMinimum. */
      public hasMinimum: boolean;

      /** AttrDef minimum. */
      public minimum: (number|Long);

      /** AttrDef allowedValues. */
      public allowedValues?: (tensorflow.IAttrValue|null);

      /**
       * Decodes an AttrDef message from the specified reader or buffer.
       * @param r Reader or buffer to decode from
       * @param [l] Message length if known beforehand
       * @returns AttrDef
       * @throws {Error} If the payload is not a reader or valid buffer
       * @throws {$protobuf.util.ProtocolError} If required fields are missing
       */
      public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
          tensorflow.OpDef.AttrDef;
    }

    /** Properties of an OpDeprecation. */
    interface IOpDeprecation {
      /** OpDeprecation version */
      version?: (number|null);

      /** OpDeprecation explanation */
      explanation?: (string|null);
    }

    /** Represents an OpDeprecation. */
    class OpDeprecation implements IOpDeprecation {
      /**
       * Constructs a new OpDeprecation.
       * @param [p] Properties to set
       */
      constructor(p?: tensorflow.OpDef.IOpDeprecation);

      /** OpDeprecation version. */
      public version: number;

      /** OpDeprecation explanation. */
      public explanation: string;

      /**
       * Decodes an OpDeprecation message from the specified reader or buffer.
       * @param r Reader or buffer to decode from
       * @param [l] Message length if known beforehand
       * @returns OpDeprecation
       * @throws {Error} If the payload is not a reader or valid buffer
       * @throws {$protobuf.util.ProtocolError} If required fields are missing
       */
      public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
          tensorflow.OpDef.OpDeprecation;
    }
  }

  /** Properties of an OpList. */
  interface IOpList {
    /** OpList op */
    op?: (tensorflow.IOpDef[]|null);
  }

  /** Represents an OpList. */
  class OpList implements IOpList {
    /**
     * Constructs a new OpList.
     * @param [p] Properties to set
     */
    constructor(p?: tensorflow.IOpList);

    /** OpList op. */
    public op: tensorflow.IOpDef[];

    /**
     * Decodes an OpList message from the specified reader or buffer.
     * @param r Reader or buffer to decode from
     * @param [l] Message length if known beforehand
     * @returns OpList
     * @throws {Error} If the payload is not a reader or valid buffer
     * @throws {$protobuf.util.ProtocolError} If required fields are missing
     */
    public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
        tensorflow.OpList;
  }

  /** Properties of a MetaGraphDef. */
  interface IMetaGraphDef {
    /** MetaGraphDef metaInfoDef */
    metaInfoDef?: (tensorflow.MetaGraphDef.IMetaInfoDef|null);

    /** MetaGraphDef graphDef */
    graphDef?: (tensorflow.IGraphDef|null);

    /** MetaGraphDef saverDef */
    saverDef?: (tensorflow.ISaverDef|null);

    /** MetaGraphDef collectionDef */
    collectionDef?: ({[k: string]: tensorflow.ICollectionDef}|null);

    /** MetaGraphDef signatureDef */
    signatureDef?: ({[k: string]: tensorflow.ISignatureDef}|null);

    /** MetaGraphDef assetFileDef */
    assetFileDef?: (tensorflow.IAssetFileDef[]|null);
  }

  /** Represents a MetaGraphDef. */
  class MetaGraphDef implements IMetaGraphDef {
    /**
     * Constructs a new MetaGraphDef.
     * @param [p] Properties to set
     */
    constructor(p?: tensorflow.IMetaGraphDef);

    /** MetaGraphDef metaInfoDef. */
    public metaInfoDef?: (tensorflow.MetaGraphDef.IMetaInfoDef|null);

    /** MetaGraphDef graphDef. */
    public graphDef?: (tensorflow.IGraphDef|null);

    /** MetaGraphDef saverDef. */
    public saverDef?: (tensorflow.ISaverDef|null);

    /** MetaGraphDef collectionDef. */
    public collectionDef: {[k: string]: tensorflow.ICollectionDef};

    /** MetaGraphDef signatureDef. */
    public signatureDef: {[k: string]: tensorflow.ISignatureDef};

    /** MetaGraphDef assetFileDef. */
    public assetFileDef: tensorflow.IAssetFileDef[];

    /**
     * Decodes a MetaGraphDef message from the specified reader or buffer.
     * @param r Reader or buffer to decode from
     * @param [l] Message length if known beforehand
     * @returns MetaGraphDef
     * @throws {Error} If the payload is not a reader or valid buffer
     * @throws {$protobuf.util.ProtocolError} If required fields are missing
     */
    public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
        tensorflow.MetaGraphDef;
  }

  namespace MetaGraphDef {

    /** Properties of a MetaInfoDef. */
    interface IMetaInfoDef {
      /** MetaInfoDef metaGraphVersion */
      metaGraphVersion?: (string|null);

      /** MetaInfoDef strippedOpList */
      strippedOpList?: (tensorflow.IOpList|null);

      /** MetaInfoDef anyInfo */
      anyInfo?: (tensorflow.IAny|null);

      /** MetaInfoDef tags */
      tags?: (string[]|null);

      /** MetaInfoDef tensorflowVersion */
      tensorflowVersion?: (string|null);

      /** MetaInfoDef tensorflowGitVersion */
      tensorflowGitVersion?: (string|null);
    }

    /** Represents a MetaInfoDef. */
    class MetaInfoDef implements IMetaInfoDef {
      /**
       * Constructs a new MetaInfoDef.
       * @param [p] Properties to set
       */
      constructor(p?: tensorflow.MetaGraphDef.IMetaInfoDef);

      /** MetaInfoDef metaGraphVersion. */
      public metaGraphVersion: string;

      /** MetaInfoDef strippedOpList. */
      public strippedOpList?: (tensorflow.IOpList|null);

      /** MetaInfoDef anyInfo. */
      public anyInfo?: (tensorflow.IAny|null);

      /** MetaInfoDef tags. */
      public tags: string[];

      /** MetaInfoDef tensorflowVersion. */
      public tensorflowVersion: string;

      /** MetaInfoDef tensorflowGitVersion. */
      public tensorflowGitVersion: string;

      /**
       * Decodes a MetaInfoDef message from the specified reader or buffer.
       * @param r Reader or buffer to decode from
       * @param [l] Message length if known beforehand
       * @returns MetaInfoDef
       * @throws {Error} If the payload is not a reader or valid buffer
       * @throws {$protobuf.util.ProtocolError} If required fields are missing
       */
      public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
          tensorflow.MetaGraphDef.MetaInfoDef;
    }
  }

  /** Properties of a SavedModel. */
  interface ISavedModel {
    /** SavedModel savedModelSchemaVersion */
    savedModelSchemaVersion?: (number|Long|null);

    /** SavedModel metaGraphs */
    metaGraphs?: (tensorflow.IMetaGraphDef[]|null);
  }

  /** Represents a SavedModel. */
  class SavedModel implements ISavedModel {
    /**
     * Constructs a new SavedModel.
     * @param [p] Properties to set
     */
    constructor(p?: tensorflow.ISavedModel);

    /** SavedModel savedModelSchemaVersion. */
    public savedModelSchemaVersion: (number|Long);

    /** SavedModel metaGraphs. */
    public metaGraphs: tensorflow.IMetaGraphDef[];

    /**
     * Decodes a SavedModel message from the specified reader or buffer.
     * @param r Reader or buffer to decode from
     * @param [l] Message length if known beforehand
     * @returns SavedModel
     * @throws {Error} If the payload is not a reader or valid buffer
     * @throws {$protobuf.util.ProtocolError} If required fields are missing
     */
    public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
        tensorflow.SavedModel;
  }

  /** Properties of a FunctionDefLibrary. */
  interface IFunctionDefLibrary {
    /** FunctionDefLibrary function */
    'function'?: (tensorflow.IFunctionDef[]|null);

    /** FunctionDefLibrary gradient */
    gradient?: (tensorflow.IGradientDef[]|null);
  }

  /** Represents a FunctionDefLibrary. */
  class FunctionDefLibrary implements IFunctionDefLibrary {
    /**
     * Constructs a new FunctionDefLibrary.
     * @param [p] Properties to set
     */
    constructor(p?: tensorflow.IFunctionDefLibrary);

    /** FunctionDefLibrary function. */
    public function: tensorflow.IFunctionDef[];

    /** FunctionDefLibrary gradient. */
    public gradient: tensorflow.IGradientDef[];

    /**
     * Decodes a FunctionDefLibrary message from the specified reader or buffer.
     * @param r Reader or buffer to decode from
     * @param [l] Message length if known beforehand
     * @returns FunctionDefLibrary
     * @throws {Error} If the payload is not a reader or valid buffer
     * @throws {$protobuf.util.ProtocolError} If required fields are missing
     */
    public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
        tensorflow.FunctionDefLibrary;
  }

  /** Properties of a FunctionDef. */
  interface IFunctionDef {
    /** FunctionDef signature */
    signature?: (tensorflow.IOpDef|null);

    /** FunctionDef attr */
    attr?: ({[k: string]: tensorflow.IAttrValue}|null);

    /** FunctionDef nodeDef */
    nodeDef?: (tensorflow.INodeDef[]|null);

    /** FunctionDef ret */
    ret?: ({[k: string]: string}|null);
  }

  /** Represents a FunctionDef. */
  class FunctionDef implements IFunctionDef {
    /**
     * Constructs a new FunctionDef.
     * @param [p] Properties to set
     */
    constructor(p?: tensorflow.IFunctionDef);

    /** FunctionDef signature. */
    public signature?: (tensorflow.IOpDef|null);

    /** FunctionDef attr. */
    public attr: {[k: string]: tensorflow.IAttrValue};

    /** FunctionDef nodeDef. */
    public nodeDef: tensorflow.INodeDef[];

    /** FunctionDef ret. */
    public ret: {[k: string]: string};

    /**
     * Decodes a FunctionDef message from the specified reader or buffer.
     * @param r Reader or buffer to decode from
     * @param [l] Message length if known beforehand
     * @returns FunctionDef
     * @throws {Error} If the payload is not a reader or valid buffer
     * @throws {$protobuf.util.ProtocolError} If required fields are missing
     */
    public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
        tensorflow.FunctionDef;
  }

  /** Properties of a GradientDef. */
  interface IGradientDef {
    /** GradientDef functionName */
    functionName?: (string|null);

    /** GradientDef gradientFunc */
    gradientFunc?: (string|null);
  }

  /** Represents a GradientDef. */
  class GradientDef implements IGradientDef {
    /**
     * Constructs a new GradientDef.
     * @param [p] Properties to set
     */
    constructor(p?: tensorflow.IGradientDef);

    /** GradientDef functionName. */
    public functionName: string;

    /** GradientDef gradientFunc. */
    public gradientFunc: string;

    /**
     * Decodes a GradientDef message from the specified reader or buffer.
     * @param r Reader or buffer to decode from
     * @param [l] Message length if known beforehand
     * @returns GradientDef
     * @throws {Error} If the payload is not a reader or valid buffer
     * @throws {$protobuf.util.ProtocolError} If required fields are missing
     */
    public static decode(r: ($protobuf.Reader|Uint8Array), l?: number):
        tensorflow.GradientDef;
  }
}
