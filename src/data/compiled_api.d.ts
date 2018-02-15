import * as $protobuf from "protobufjs";

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
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.IAny);

        /** Any typeUrl. */
        public typeUrl: string;

        /** Any value. */
        public value: Uint8Array;

        /**
         * Creates a new Any instance using the specified properties.
         * @param [properties] Properties to set
         * @returns Any instance
         */
        public static create(properties?: tensorflow.IAny): tensorflow.Any;

        /**
         * Encodes the specified Any message. Does not implicitly {@link tensorflow.Any.verify|verify} messages.
         * @param message Any message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.IAny, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified Any message, length delimited. Does not implicitly {@link tensorflow.Any.verify|verify} messages.
         * @param message Any message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.IAny, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an Any message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns Any
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.Any;

        /**
         * Decodes an Any message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns Any
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.Any;

        /**
         * Verifies an Any message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates an Any message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns Any
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.Any;

        /**
         * Creates a plain object from an Any message. Also converts values to other types if specified.
         * @param message Any
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.Any, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this Any to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
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
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.ITensorShape);

        /** TensorShape dim. */
        public dim: tensorflow.TensorShape.IDim[];

        /** TensorShape unknownRank. */
        public unknownRank: boolean;

        /**
         * Creates a new TensorShape instance using the specified properties.
         * @param [properties] Properties to set
         * @returns TensorShape instance
         */
        public static create(properties?: tensorflow.ITensorShape): tensorflow.TensorShape;

        /**
         * Encodes the specified TensorShape message. Does not implicitly {@link tensorflow.TensorShape.verify|verify} messages.
         * @param message TensorShape message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.ITensorShape, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified TensorShape message, length delimited. Does not implicitly {@link tensorflow.TensorShape.verify|verify} messages.
         * @param message TensorShape message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.ITensorShape, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a TensorShape message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns TensorShape
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.TensorShape;

        /**
         * Decodes a TensorShape message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns TensorShape
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.TensorShape;

        /**
         * Verifies a TensorShape message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a TensorShape message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns TensorShape
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.TensorShape;

        /**
         * Creates a plain object from a TensorShape message. Also converts values to other types if specified.
         * @param message TensorShape
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.TensorShape, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this TensorShape to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
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
             * @param [properties] Properties to set
             */
            constructor(properties?: tensorflow.TensorShape.IDim);

            /** Dim size. */
            public size: (number|Long);

            /** Dim name. */
            public name: string;

            /**
             * Creates a new Dim instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Dim instance
             */
            public static create(properties?: tensorflow.TensorShape.IDim): tensorflow.TensorShape.Dim;

            /**
             * Encodes the specified Dim message. Does not implicitly {@link tensorflow.TensorShape.Dim.verify|verify} messages.
             * @param message Dim message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: tensorflow.TensorShape.IDim, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified Dim message, length delimited. Does not implicitly {@link tensorflow.TensorShape.Dim.verify|verify} messages.
             * @param message Dim message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: tensorflow.TensorShape.IDim, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a Dim message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns Dim
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.TensorShape.Dim;

            /**
             * Decodes a Dim message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns Dim
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.TensorShape.Dim;

            /**
             * Verifies a Dim message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a Dim message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns Dim
             */
            public static fromObject(object: { [k: string]: any }): tensorflow.TensorShape.Dim;

            /**
             * Creates a plain object from a Dim message. Also converts values to other types if specified.
             * @param message Dim
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: tensorflow.TensorShape.Dim, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this Dim to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
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
        int64Val?: ((number|Long)[]|null);

        /** Tensor boolVal */
        boolVal?: (boolean[]|null);

        /** Tensor uint32Val */
        uint32Val?: (number[]|null);

        /** Tensor uint64Val */
        uint64Val?: ((number|Long)[]|null);
    }

    /** Represents a Tensor. */
    class Tensor implements ITensor {

        /**
         * Constructs a new Tensor.
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.ITensor);

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
         * Creates a new Tensor instance using the specified properties.
         * @param [properties] Properties to set
         * @returns Tensor instance
         */
        public static create(properties?: tensorflow.ITensor): tensorflow.Tensor;

        /**
         * Encodes the specified Tensor message. Does not implicitly {@link tensorflow.Tensor.verify|verify} messages.
         * @param message Tensor message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.ITensor, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified Tensor message, length delimited. Does not implicitly {@link tensorflow.Tensor.verify|verify} messages.
         * @param message Tensor message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.ITensor, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a Tensor message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns Tensor
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.Tensor;

        /**
         * Decodes a Tensor message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns Tensor
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.Tensor;

        /**
         * Verifies a Tensor message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a Tensor message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns Tensor
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.Tensor;

        /**
         * Creates a plain object from a Tensor message. Also converts values to other types if specified.
         * @param message Tensor
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.Tensor, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this Tensor to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
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
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.IAttrValue);

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
        public value?: ("list"|"s"|"i"|"f"|"b"|"type"|"shape"|"tensor"|"placeholder"|"func");

        /**
         * Creates a new AttrValue instance using the specified properties.
         * @param [properties] Properties to set
         * @returns AttrValue instance
         */
        public static create(properties?: tensorflow.IAttrValue): tensorflow.AttrValue;

        /**
         * Encodes the specified AttrValue message. Does not implicitly {@link tensorflow.AttrValue.verify|verify} messages.
         * @param message AttrValue message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.IAttrValue, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified AttrValue message, length delimited. Does not implicitly {@link tensorflow.AttrValue.verify|verify} messages.
         * @param message AttrValue message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.IAttrValue, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an AttrValue message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns AttrValue
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.AttrValue;

        /**
         * Decodes an AttrValue message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns AttrValue
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.AttrValue;

        /**
         * Verifies an AttrValue message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates an AttrValue message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns AttrValue
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.AttrValue;

        /**
         * Creates a plain object from an AttrValue message. Also converts values to other types if specified.
         * @param message AttrValue
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.AttrValue, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this AttrValue to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    namespace AttrValue {

        /** Properties of a ListValue. */
        interface IListValue {

            /** ListValue s */
            s?: (Uint8Array[]|null);

            /** ListValue i */
            i?: ((number|Long)[]|null);

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
             * @param [properties] Properties to set
             */
            constructor(properties?: tensorflow.AttrValue.IListValue);

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
             * Creates a new ListValue instance using the specified properties.
             * @param [properties] Properties to set
             * @returns ListValue instance
             */
            public static create(properties?: tensorflow.AttrValue.IListValue): tensorflow.AttrValue.ListValue;

            /**
             * Encodes the specified ListValue message. Does not implicitly {@link tensorflow.AttrValue.ListValue.verify|verify} messages.
             * @param message ListValue message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: tensorflow.AttrValue.IListValue, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified ListValue message, length delimited. Does not implicitly {@link tensorflow.AttrValue.ListValue.verify|verify} messages.
             * @param message ListValue message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: tensorflow.AttrValue.IListValue, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a ListValue message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns ListValue
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.AttrValue.ListValue;

            /**
             * Decodes a ListValue message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns ListValue
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.AttrValue.ListValue;

            /**
             * Verifies a ListValue message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a ListValue message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns ListValue
             */
            public static fromObject(object: { [k: string]: any }): tensorflow.AttrValue.ListValue;

            /**
             * Creates a plain object from a ListValue message. Also converts values to other types if specified.
             * @param message ListValue
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: tensorflow.AttrValue.ListValue, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this ListValue to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
        }
    }

    /** Properties of a NameAttrList. */
    interface INameAttrList {

        /** NameAttrList name */
        name?: (string|null);

        /** NameAttrList attr */
        attr?: ({ [k: string]: tensorflow.IAttrValue }|null);
    }

    /** Represents a NameAttrList. */
    class NameAttrList implements INameAttrList {

        /**
         * Constructs a new NameAttrList.
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.INameAttrList);

        /** NameAttrList name. */
        public name: string;

        /** NameAttrList attr. */
        public attr: { [k: string]: tensorflow.IAttrValue };

        /**
         * Creates a new NameAttrList instance using the specified properties.
         * @param [properties] Properties to set
         * @returns NameAttrList instance
         */
        public static create(properties?: tensorflow.INameAttrList): tensorflow.NameAttrList;

        /**
         * Encodes the specified NameAttrList message. Does not implicitly {@link tensorflow.NameAttrList.verify|verify} messages.
         * @param message NameAttrList message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.INameAttrList, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified NameAttrList message, length delimited. Does not implicitly {@link tensorflow.NameAttrList.verify|verify} messages.
         * @param message NameAttrList message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.INameAttrList, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a NameAttrList message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns NameAttrList
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.NameAttrList;

        /**
         * Decodes a NameAttrList message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns NameAttrList
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.NameAttrList;

        /**
         * Verifies a NameAttrList message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a NameAttrList message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns NameAttrList
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.NameAttrList;

        /**
         * Creates a plain object from a NameAttrList message. Also converts values to other types if specified.
         * @param message NameAttrList
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.NameAttrList, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this NameAttrList to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
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
        attr?: ({ [k: string]: tensorflow.IAttrValue }|null);
    }

    /** Represents a NodeDef. */
    class NodeDef implements INodeDef {

        /**
         * Constructs a new NodeDef.
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.INodeDef);

        /** NodeDef name. */
        public name: string;

        /** NodeDef op. */
        public op: string;

        /** NodeDef input. */
        public input: string[];

        /** NodeDef device. */
        public device: string;

        /** NodeDef attr. */
        public attr: { [k: string]: tensorflow.IAttrValue };

        /**
         * Creates a new NodeDef instance using the specified properties.
         * @param [properties] Properties to set
         * @returns NodeDef instance
         */
        public static create(properties?: tensorflow.INodeDef): tensorflow.NodeDef;

        /**
         * Encodes the specified NodeDef message. Does not implicitly {@link tensorflow.NodeDef.verify|verify} messages.
         * @param message NodeDef message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.INodeDef, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified NodeDef message, length delimited. Does not implicitly {@link tensorflow.NodeDef.verify|verify} messages.
         * @param message NodeDef message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.INodeDef, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a NodeDef message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns NodeDef
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.NodeDef;

        /**
         * Decodes a NodeDef message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns NodeDef
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.NodeDef;

        /**
         * Verifies a NodeDef message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a NodeDef message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns NodeDef
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.NodeDef;

        /**
         * Creates a plain object from a NodeDef message. Also converts values to other types if specified.
         * @param message NodeDef
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.NodeDef, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this NodeDef to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
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
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.IVersionDef);

        /** VersionDef producer. */
        public producer: number;

        /** VersionDef minConsumer. */
        public minConsumer: number;

        /** VersionDef badConsumers. */
        public badConsumers: number[];

        /**
         * Creates a new VersionDef instance using the specified properties.
         * @param [properties] Properties to set
         * @returns VersionDef instance
         */
        public static create(properties?: tensorflow.IVersionDef): tensorflow.VersionDef;

        /**
         * Encodes the specified VersionDef message. Does not implicitly {@link tensorflow.VersionDef.verify|verify} messages.
         * @param message VersionDef message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.IVersionDef, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified VersionDef message, length delimited. Does not implicitly {@link tensorflow.VersionDef.verify|verify} messages.
         * @param message VersionDef message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.IVersionDef, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a VersionDef message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns VersionDef
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.VersionDef;

        /**
         * Decodes a VersionDef message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns VersionDef
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.VersionDef;

        /**
         * Verifies a VersionDef message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a VersionDef message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns VersionDef
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.VersionDef;

        /**
         * Creates a plain object from a VersionDef message. Also converts values to other types if specified.
         * @param message VersionDef
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.VersionDef, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this VersionDef to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a GraphDef. */
    interface IGraphDef {

        /** GraphDef node */
        node?: (tensorflow.INodeDef[]|null);

        /** GraphDef versions */
        versions?: (tensorflow.IVersionDef|null);
    }

    /** Represents a GraphDef. */
    class GraphDef implements IGraphDef {

        /**
         * Constructs a new GraphDef.
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.IGraphDef);

        /** GraphDef node. */
        public node: tensorflow.INodeDef[];

        /** GraphDef versions. */
        public versions?: (tensorflow.IVersionDef|null);

        /**
         * Creates a new GraphDef instance using the specified properties.
         * @param [properties] Properties to set
         * @returns GraphDef instance
         */
        public static create(properties?: tensorflow.IGraphDef): tensorflow.GraphDef;

        /**
         * Encodes the specified GraphDef message. Does not implicitly {@link tensorflow.GraphDef.verify|verify} messages.
         * @param message GraphDef message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.IGraphDef, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified GraphDef message, length delimited. Does not implicitly {@link tensorflow.GraphDef.verify|verify} messages.
         * @param message GraphDef message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.IGraphDef, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a GraphDef message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns GraphDef
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.GraphDef;

        /**
         * Decodes a GraphDef message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns GraphDef
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.GraphDef;

        /**
         * Verifies a GraphDef message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a GraphDef message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns GraphDef
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.GraphDef;

        /**
         * Creates a plain object from a GraphDef message. Also converts values to other types if specified.
         * @param message GraphDef
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.GraphDef, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this GraphDef to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
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
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.ICollectionDef);

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
        public kind?: ("nodeList"|"bytesList"|"int64List"|"floatList"|"anyList");

        /**
         * Creates a new CollectionDef instance using the specified properties.
         * @param [properties] Properties to set
         * @returns CollectionDef instance
         */
        public static create(properties?: tensorflow.ICollectionDef): tensorflow.CollectionDef;

        /**
         * Encodes the specified CollectionDef message. Does not implicitly {@link tensorflow.CollectionDef.verify|verify} messages.
         * @param message CollectionDef message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.ICollectionDef, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified CollectionDef message, length delimited. Does not implicitly {@link tensorflow.CollectionDef.verify|verify} messages.
         * @param message CollectionDef message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.ICollectionDef, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a CollectionDef message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns CollectionDef
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.CollectionDef;

        /**
         * Decodes a CollectionDef message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns CollectionDef
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.CollectionDef;

        /**
         * Verifies a CollectionDef message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a CollectionDef message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns CollectionDef
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.CollectionDef;

        /**
         * Creates a plain object from a CollectionDef message. Also converts values to other types if specified.
         * @param message CollectionDef
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.CollectionDef, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this CollectionDef to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
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
             * @param [properties] Properties to set
             */
            constructor(properties?: tensorflow.CollectionDef.INodeList);

            /** NodeList value. */
            public value: string[];

            /**
             * Creates a new NodeList instance using the specified properties.
             * @param [properties] Properties to set
             * @returns NodeList instance
             */
            public static create(properties?: tensorflow.CollectionDef.INodeList): tensorflow.CollectionDef.NodeList;

            /**
             * Encodes the specified NodeList message. Does not implicitly {@link tensorflow.CollectionDef.NodeList.verify|verify} messages.
             * @param message NodeList message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: tensorflow.CollectionDef.INodeList, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified NodeList message, length delimited. Does not implicitly {@link tensorflow.CollectionDef.NodeList.verify|verify} messages.
             * @param message NodeList message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: tensorflow.CollectionDef.INodeList, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a NodeList message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns NodeList
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.CollectionDef.NodeList;

            /**
             * Decodes a NodeList message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns NodeList
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.CollectionDef.NodeList;

            /**
             * Verifies a NodeList message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a NodeList message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns NodeList
             */
            public static fromObject(object: { [k: string]: any }): tensorflow.CollectionDef.NodeList;

            /**
             * Creates a plain object from a NodeList message. Also converts values to other types if specified.
             * @param message NodeList
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: tensorflow.CollectionDef.NodeList, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this NodeList to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
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
             * @param [properties] Properties to set
             */
            constructor(properties?: tensorflow.CollectionDef.IBytesList);

            /** BytesList value. */
            public value: Uint8Array[];

            /**
             * Creates a new BytesList instance using the specified properties.
             * @param [properties] Properties to set
             * @returns BytesList instance
             */
            public static create(properties?: tensorflow.CollectionDef.IBytesList): tensorflow.CollectionDef.BytesList;

            /**
             * Encodes the specified BytesList message. Does not implicitly {@link tensorflow.CollectionDef.BytesList.verify|verify} messages.
             * @param message BytesList message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: tensorflow.CollectionDef.IBytesList, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified BytesList message, length delimited. Does not implicitly {@link tensorflow.CollectionDef.BytesList.verify|verify} messages.
             * @param message BytesList message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: tensorflow.CollectionDef.IBytesList, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a BytesList message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns BytesList
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.CollectionDef.BytesList;

            /**
             * Decodes a BytesList message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns BytesList
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.CollectionDef.BytesList;

            /**
             * Verifies a BytesList message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a BytesList message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns BytesList
             */
            public static fromObject(object: { [k: string]: any }): tensorflow.CollectionDef.BytesList;

            /**
             * Creates a plain object from a BytesList message. Also converts values to other types if specified.
             * @param message BytesList
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: tensorflow.CollectionDef.BytesList, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this BytesList to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        /** Properties of an Int64List. */
        interface IInt64List {

            /** Int64List value */
            value?: ((number|Long)[]|null);
        }

        /** Represents an Int64List. */
        class Int64List implements IInt64List {

            /**
             * Constructs a new Int64List.
             * @param [properties] Properties to set
             */
            constructor(properties?: tensorflow.CollectionDef.IInt64List);

            /** Int64List value. */
            public value: (number|Long)[];

            /**
             * Creates a new Int64List instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Int64List instance
             */
            public static create(properties?: tensorflow.CollectionDef.IInt64List): tensorflow.CollectionDef.Int64List;

            /**
             * Encodes the specified Int64List message. Does not implicitly {@link tensorflow.CollectionDef.Int64List.verify|verify} messages.
             * @param message Int64List message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: tensorflow.CollectionDef.IInt64List, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified Int64List message, length delimited. Does not implicitly {@link tensorflow.CollectionDef.Int64List.verify|verify} messages.
             * @param message Int64List message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: tensorflow.CollectionDef.IInt64List, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an Int64List message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns Int64List
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.CollectionDef.Int64List;

            /**
             * Decodes an Int64List message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns Int64List
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.CollectionDef.Int64List;

            /**
             * Verifies an Int64List message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an Int64List message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns Int64List
             */
            public static fromObject(object: { [k: string]: any }): tensorflow.CollectionDef.Int64List;

            /**
             * Creates a plain object from an Int64List message. Also converts values to other types if specified.
             * @param message Int64List
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: tensorflow.CollectionDef.Int64List, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this Int64List to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
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
             * @param [properties] Properties to set
             */
            constructor(properties?: tensorflow.CollectionDef.IFloatList);

            /** FloatList value. */
            public value: number[];

            /**
             * Creates a new FloatList instance using the specified properties.
             * @param [properties] Properties to set
             * @returns FloatList instance
             */
            public static create(properties?: tensorflow.CollectionDef.IFloatList): tensorflow.CollectionDef.FloatList;

            /**
             * Encodes the specified FloatList message. Does not implicitly {@link tensorflow.CollectionDef.FloatList.verify|verify} messages.
             * @param message FloatList message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: tensorflow.CollectionDef.IFloatList, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified FloatList message, length delimited. Does not implicitly {@link tensorflow.CollectionDef.FloatList.verify|verify} messages.
             * @param message FloatList message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: tensorflow.CollectionDef.IFloatList, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a FloatList message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns FloatList
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.CollectionDef.FloatList;

            /**
             * Decodes a FloatList message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns FloatList
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.CollectionDef.FloatList;

            /**
             * Verifies a FloatList message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a FloatList message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns FloatList
             */
            public static fromObject(object: { [k: string]: any }): tensorflow.CollectionDef.FloatList;

            /**
             * Creates a plain object from a FloatList message. Also converts values to other types if specified.
             * @param message FloatList
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: tensorflow.CollectionDef.FloatList, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this FloatList to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
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
             * @param [properties] Properties to set
             */
            constructor(properties?: tensorflow.CollectionDef.IAnyList);

            /** AnyList value. */
            public value: tensorflow.IAny[];

            /**
             * Creates a new AnyList instance using the specified properties.
             * @param [properties] Properties to set
             * @returns AnyList instance
             */
            public static create(properties?: tensorflow.CollectionDef.IAnyList): tensorflow.CollectionDef.AnyList;

            /**
             * Encodes the specified AnyList message. Does not implicitly {@link tensorflow.CollectionDef.AnyList.verify|verify} messages.
             * @param message AnyList message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: tensorflow.CollectionDef.IAnyList, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified AnyList message, length delimited. Does not implicitly {@link tensorflow.CollectionDef.AnyList.verify|verify} messages.
             * @param message AnyList message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: tensorflow.CollectionDef.IAnyList, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an AnyList message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns AnyList
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.CollectionDef.AnyList;

            /**
             * Decodes an AnyList message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns AnyList
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.CollectionDef.AnyList;

            /**
             * Verifies an AnyList message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an AnyList message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns AnyList
             */
            public static fromObject(object: { [k: string]: any }): tensorflow.CollectionDef.AnyList;

            /**
             * Creates a plain object from an AnyList message. Also converts values to other types if specified.
             * @param message AnyList
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: tensorflow.CollectionDef.AnyList, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this AnyList to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
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
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.ISaverDef);

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
         * Creates a new SaverDef instance using the specified properties.
         * @param [properties] Properties to set
         * @returns SaverDef instance
         */
        public static create(properties?: tensorflow.ISaverDef): tensorflow.SaverDef;

        /**
         * Encodes the specified SaverDef message. Does not implicitly {@link tensorflow.SaverDef.verify|verify} messages.
         * @param message SaverDef message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.ISaverDef, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified SaverDef message, length delimited. Does not implicitly {@link tensorflow.SaverDef.verify|verify} messages.
         * @param message SaverDef message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.ISaverDef, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a SaverDef message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns SaverDef
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.SaverDef;

        /**
         * Decodes a SaverDef message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns SaverDef
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.SaverDef;

        /**
         * Verifies a SaverDef message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a SaverDef message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns SaverDef
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.SaverDef;

        /**
         * Creates a plain object from a SaverDef message. Also converts values to other types if specified.
         * @param message SaverDef
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.SaverDef, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this SaverDef to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    namespace SaverDef {

        /** CheckpointFormatVersion enum. */
        enum CheckpointFormatVersion {
            LEGACY = 0,
            V1 = 1,
            V2 = 2
        }
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
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.ITensorInfo);

        /** TensorInfo name. */
        public name: string;

        /** TensorInfo cooSparse. */
        public cooSparse?: (tensorflow.TensorInfo.ICooSparse|null);

        /** TensorInfo dtype. */
        public dtype: tensorflow.DataType;

        /** TensorInfo tensorShape. */
        public tensorShape?: (tensorflow.ITensorShape|null);

        /** TensorInfo encoding. */
        public encoding?: ("name"|"cooSparse");

        /**
         * Creates a new TensorInfo instance using the specified properties.
         * @param [properties] Properties to set
         * @returns TensorInfo instance
         */
        public static create(properties?: tensorflow.ITensorInfo): tensorflow.TensorInfo;

        /**
         * Encodes the specified TensorInfo message. Does not implicitly {@link tensorflow.TensorInfo.verify|verify} messages.
         * @param message TensorInfo message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.ITensorInfo, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified TensorInfo message, length delimited. Does not implicitly {@link tensorflow.TensorInfo.verify|verify} messages.
         * @param message TensorInfo message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.ITensorInfo, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a TensorInfo message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns TensorInfo
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.TensorInfo;

        /**
         * Decodes a TensorInfo message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns TensorInfo
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.TensorInfo;

        /**
         * Verifies a TensorInfo message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a TensorInfo message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns TensorInfo
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.TensorInfo;

        /**
         * Creates a plain object from a TensorInfo message. Also converts values to other types if specified.
         * @param message TensorInfo
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.TensorInfo, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this TensorInfo to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
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
             * @param [properties] Properties to set
             */
            constructor(properties?: tensorflow.TensorInfo.ICooSparse);

            /** CooSparse valuesTensorName. */
            public valuesTensorName: string;

            /** CooSparse indicesTensorName. */
            public indicesTensorName: string;

            /** CooSparse denseShapeTensorName. */
            public denseShapeTensorName: string;

            /**
             * Creates a new CooSparse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns CooSparse instance
             */
            public static create(properties?: tensorflow.TensorInfo.ICooSparse): tensorflow.TensorInfo.CooSparse;

            /**
             * Encodes the specified CooSparse message. Does not implicitly {@link tensorflow.TensorInfo.CooSparse.verify|verify} messages.
             * @param message CooSparse message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: tensorflow.TensorInfo.ICooSparse, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified CooSparse message, length delimited. Does not implicitly {@link tensorflow.TensorInfo.CooSparse.verify|verify} messages.
             * @param message CooSparse message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: tensorflow.TensorInfo.ICooSparse, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a CooSparse message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns CooSparse
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.TensorInfo.CooSparse;

            /**
             * Decodes a CooSparse message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns CooSparse
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.TensorInfo.CooSparse;

            /**
             * Verifies a CooSparse message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a CooSparse message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns CooSparse
             */
            public static fromObject(object: { [k: string]: any }): tensorflow.TensorInfo.CooSparse;

            /**
             * Creates a plain object from a CooSparse message. Also converts values to other types if specified.
             * @param message CooSparse
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: tensorflow.TensorInfo.CooSparse, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this CooSparse to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
        }
    }

    /** Properties of a SignatureDef. */
    interface ISignatureDef {

        /** SignatureDef inputs */
        inputs?: ({ [k: string]: tensorflow.ITensorInfo }|null);

        /** SignatureDef outputs */
        outputs?: ({ [k: string]: tensorflow.ITensorInfo }|null);

        /** SignatureDef methodName */
        methodName?: (string|null);
    }

    /** Represents a SignatureDef. */
    class SignatureDef implements ISignatureDef {

        /**
         * Constructs a new SignatureDef.
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.ISignatureDef);

        /** SignatureDef inputs. */
        public inputs: { [k: string]: tensorflow.ITensorInfo };

        /** SignatureDef outputs. */
        public outputs: { [k: string]: tensorflow.ITensorInfo };

        /** SignatureDef methodName. */
        public methodName: string;

        /**
         * Creates a new SignatureDef instance using the specified properties.
         * @param [properties] Properties to set
         * @returns SignatureDef instance
         */
        public static create(properties?: tensorflow.ISignatureDef): tensorflow.SignatureDef;

        /**
         * Encodes the specified SignatureDef message. Does not implicitly {@link tensorflow.SignatureDef.verify|verify} messages.
         * @param message SignatureDef message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.ISignatureDef, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified SignatureDef message, length delimited. Does not implicitly {@link tensorflow.SignatureDef.verify|verify} messages.
         * @param message SignatureDef message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.ISignatureDef, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a SignatureDef message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns SignatureDef
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.SignatureDef;

        /**
         * Decodes a SignatureDef message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns SignatureDef
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.SignatureDef;

        /**
         * Verifies a SignatureDef message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a SignatureDef message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns SignatureDef
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.SignatureDef;

        /**
         * Creates a plain object from a SignatureDef message. Also converts values to other types if specified.
         * @param message SignatureDef
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.SignatureDef, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this SignatureDef to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
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
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.IAssetFileDef);

        /** AssetFileDef tensorInfo. */
        public tensorInfo?: (tensorflow.ITensorInfo|null);

        /** AssetFileDef filename. */
        public filename: string;

        /**
         * Creates a new AssetFileDef instance using the specified properties.
         * @param [properties] Properties to set
         * @returns AssetFileDef instance
         */
        public static create(properties?: tensorflow.IAssetFileDef): tensorflow.AssetFileDef;

        /**
         * Encodes the specified AssetFileDef message. Does not implicitly {@link tensorflow.AssetFileDef.verify|verify} messages.
         * @param message AssetFileDef message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.IAssetFileDef, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified AssetFileDef message, length delimited. Does not implicitly {@link tensorflow.AssetFileDef.verify|verify} messages.
         * @param message AssetFileDef message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.IAssetFileDef, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an AssetFileDef message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns AssetFileDef
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.AssetFileDef;

        /**
         * Decodes an AssetFileDef message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns AssetFileDef
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.AssetFileDef;

        /**
         * Verifies an AssetFileDef message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates an AssetFileDef message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns AssetFileDef
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.AssetFileDef;

        /**
         * Creates a plain object from an AssetFileDef message. Also converts values to other types if specified.
         * @param message AssetFileDef
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.AssetFileDef, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this AssetFileDef to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
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
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.IOpDef);

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
         * Creates a new OpDef instance using the specified properties.
         * @param [properties] Properties to set
         * @returns OpDef instance
         */
        public static create(properties?: tensorflow.IOpDef): tensorflow.OpDef;

        /**
         * Encodes the specified OpDef message. Does not implicitly {@link tensorflow.OpDef.verify|verify} messages.
         * @param message OpDef message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.IOpDef, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified OpDef message, length delimited. Does not implicitly {@link tensorflow.OpDef.verify|verify} messages.
         * @param message OpDef message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.IOpDef, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an OpDef message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns OpDef
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.OpDef;

        /**
         * Decodes an OpDef message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns OpDef
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.OpDef;

        /**
         * Verifies an OpDef message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates an OpDef message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns OpDef
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.OpDef;

        /**
         * Creates a plain object from an OpDef message. Also converts values to other types if specified.
         * @param message OpDef
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.OpDef, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this OpDef to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
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
             * @param [properties] Properties to set
             */
            constructor(properties?: tensorflow.OpDef.IArgDef);

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
             * Creates a new ArgDef instance using the specified properties.
             * @param [properties] Properties to set
             * @returns ArgDef instance
             */
            public static create(properties?: tensorflow.OpDef.IArgDef): tensorflow.OpDef.ArgDef;

            /**
             * Encodes the specified ArgDef message. Does not implicitly {@link tensorflow.OpDef.ArgDef.verify|verify} messages.
             * @param message ArgDef message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: tensorflow.OpDef.IArgDef, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified ArgDef message, length delimited. Does not implicitly {@link tensorflow.OpDef.ArgDef.verify|verify} messages.
             * @param message ArgDef message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: tensorflow.OpDef.IArgDef, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an ArgDef message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns ArgDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.OpDef.ArgDef;

            /**
             * Decodes an ArgDef message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns ArgDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.OpDef.ArgDef;

            /**
             * Verifies an ArgDef message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an ArgDef message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns ArgDef
             */
            public static fromObject(object: { [k: string]: any }): tensorflow.OpDef.ArgDef;

            /**
             * Creates a plain object from an ArgDef message. Also converts values to other types if specified.
             * @param message ArgDef
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: tensorflow.OpDef.ArgDef, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this ArgDef to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
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
             * @param [properties] Properties to set
             */
            constructor(properties?: tensorflow.OpDef.IAttrDef);

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
             * Creates a new AttrDef instance using the specified properties.
             * @param [properties] Properties to set
             * @returns AttrDef instance
             */
            public static create(properties?: tensorflow.OpDef.IAttrDef): tensorflow.OpDef.AttrDef;

            /**
             * Encodes the specified AttrDef message. Does not implicitly {@link tensorflow.OpDef.AttrDef.verify|verify} messages.
             * @param message AttrDef message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: tensorflow.OpDef.IAttrDef, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified AttrDef message, length delimited. Does not implicitly {@link tensorflow.OpDef.AttrDef.verify|verify} messages.
             * @param message AttrDef message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: tensorflow.OpDef.IAttrDef, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an AttrDef message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns AttrDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.OpDef.AttrDef;

            /**
             * Decodes an AttrDef message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns AttrDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.OpDef.AttrDef;

            /**
             * Verifies an AttrDef message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an AttrDef message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns AttrDef
             */
            public static fromObject(object: { [k: string]: any }): tensorflow.OpDef.AttrDef;

            /**
             * Creates a plain object from an AttrDef message. Also converts values to other types if specified.
             * @param message AttrDef
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: tensorflow.OpDef.AttrDef, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this AttrDef to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
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
             * @param [properties] Properties to set
             */
            constructor(properties?: tensorflow.OpDef.IOpDeprecation);

            /** OpDeprecation version. */
            public version: number;

            /** OpDeprecation explanation. */
            public explanation: string;

            /**
             * Creates a new OpDeprecation instance using the specified properties.
             * @param [properties] Properties to set
             * @returns OpDeprecation instance
             */
            public static create(properties?: tensorflow.OpDef.IOpDeprecation): tensorflow.OpDef.OpDeprecation;

            /**
             * Encodes the specified OpDeprecation message. Does not implicitly {@link tensorflow.OpDef.OpDeprecation.verify|verify} messages.
             * @param message OpDeprecation message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: tensorflow.OpDef.IOpDeprecation, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified OpDeprecation message, length delimited. Does not implicitly {@link tensorflow.OpDef.OpDeprecation.verify|verify} messages.
             * @param message OpDeprecation message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: tensorflow.OpDef.IOpDeprecation, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an OpDeprecation message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns OpDeprecation
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.OpDef.OpDeprecation;

            /**
             * Decodes an OpDeprecation message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns OpDeprecation
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.OpDef.OpDeprecation;

            /**
             * Verifies an OpDeprecation message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an OpDeprecation message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns OpDeprecation
             */
            public static fromObject(object: { [k: string]: any }): tensorflow.OpDef.OpDeprecation;

            /**
             * Creates a plain object from an OpDeprecation message. Also converts values to other types if specified.
             * @param message OpDeprecation
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: tensorflow.OpDef.OpDeprecation, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this OpDeprecation to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
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
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.IOpList);

        /** OpList op. */
        public op: tensorflow.IOpDef[];

        /**
         * Creates a new OpList instance using the specified properties.
         * @param [properties] Properties to set
         * @returns OpList instance
         */
        public static create(properties?: tensorflow.IOpList): tensorflow.OpList;

        /**
         * Encodes the specified OpList message. Does not implicitly {@link tensorflow.OpList.verify|verify} messages.
         * @param message OpList message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.IOpList, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified OpList message, length delimited. Does not implicitly {@link tensorflow.OpList.verify|verify} messages.
         * @param message OpList message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.IOpList, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an OpList message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns OpList
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.OpList;

        /**
         * Decodes an OpList message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns OpList
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.OpList;

        /**
         * Verifies an OpList message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates an OpList message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns OpList
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.OpList;

        /**
         * Creates a plain object from an OpList message. Also converts values to other types if specified.
         * @param message OpList
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.OpList, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this OpList to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
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
        collectionDef?: ({ [k: string]: tensorflow.ICollectionDef }|null);

        /** MetaGraphDef signatureDef */
        signatureDef?: ({ [k: string]: tensorflow.ISignatureDef }|null);

        /** MetaGraphDef assetFileDef */
        assetFileDef?: (tensorflow.IAssetFileDef[]|null);
    }

    /** Represents a MetaGraphDef. */
    class MetaGraphDef implements IMetaGraphDef {

        /**
         * Constructs a new MetaGraphDef.
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.IMetaGraphDef);

        /** MetaGraphDef metaInfoDef. */
        public metaInfoDef?: (tensorflow.MetaGraphDef.IMetaInfoDef|null);

        /** MetaGraphDef graphDef. */
        public graphDef?: (tensorflow.IGraphDef|null);

        /** MetaGraphDef saverDef. */
        public saverDef?: (tensorflow.ISaverDef|null);

        /** MetaGraphDef collectionDef. */
        public collectionDef: { [k: string]: tensorflow.ICollectionDef };

        /** MetaGraphDef signatureDef. */
        public signatureDef: { [k: string]: tensorflow.ISignatureDef };

        /** MetaGraphDef assetFileDef. */
        public assetFileDef: tensorflow.IAssetFileDef[];

        /**
         * Creates a new MetaGraphDef instance using the specified properties.
         * @param [properties] Properties to set
         * @returns MetaGraphDef instance
         */
        public static create(properties?: tensorflow.IMetaGraphDef): tensorflow.MetaGraphDef;

        /**
         * Encodes the specified MetaGraphDef message. Does not implicitly {@link tensorflow.MetaGraphDef.verify|verify} messages.
         * @param message MetaGraphDef message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.IMetaGraphDef, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified MetaGraphDef message, length delimited. Does not implicitly {@link tensorflow.MetaGraphDef.verify|verify} messages.
         * @param message MetaGraphDef message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.IMetaGraphDef, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a MetaGraphDef message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns MetaGraphDef
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.MetaGraphDef;

        /**
         * Decodes a MetaGraphDef message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns MetaGraphDef
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.MetaGraphDef;

        /**
         * Verifies a MetaGraphDef message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a MetaGraphDef message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns MetaGraphDef
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.MetaGraphDef;

        /**
         * Creates a plain object from a MetaGraphDef message. Also converts values to other types if specified.
         * @param message MetaGraphDef
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.MetaGraphDef, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this MetaGraphDef to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
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
             * @param [properties] Properties to set
             */
            constructor(properties?: tensorflow.MetaGraphDef.IMetaInfoDef);

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
             * Creates a new MetaInfoDef instance using the specified properties.
             * @param [properties] Properties to set
             * @returns MetaInfoDef instance
             */
            public static create(properties?: tensorflow.MetaGraphDef.IMetaInfoDef): tensorflow.MetaGraphDef.MetaInfoDef;

            /**
             * Encodes the specified MetaInfoDef message. Does not implicitly {@link tensorflow.MetaGraphDef.MetaInfoDef.verify|verify} messages.
             * @param message MetaInfoDef message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: tensorflow.MetaGraphDef.IMetaInfoDef, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified MetaInfoDef message, length delimited. Does not implicitly {@link tensorflow.MetaGraphDef.MetaInfoDef.verify|verify} messages.
             * @param message MetaInfoDef message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: tensorflow.MetaGraphDef.IMetaInfoDef, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a MetaInfoDef message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns MetaInfoDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.MetaGraphDef.MetaInfoDef;

            /**
             * Decodes a MetaInfoDef message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns MetaInfoDef
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.MetaGraphDef.MetaInfoDef;

            /**
             * Verifies a MetaInfoDef message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a MetaInfoDef message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns MetaInfoDef
             */
            public static fromObject(object: { [k: string]: any }): tensorflow.MetaGraphDef.MetaInfoDef;

            /**
             * Creates a plain object from a MetaInfoDef message. Also converts values to other types if specified.
             * @param message MetaInfoDef
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: tensorflow.MetaGraphDef.MetaInfoDef, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this MetaInfoDef to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
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
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.ISavedModel);

        /** SavedModel savedModelSchemaVersion. */
        public savedModelSchemaVersion: (number|Long);

        /** SavedModel metaGraphs. */
        public metaGraphs: tensorflow.IMetaGraphDef[];

        /**
         * Creates a new SavedModel instance using the specified properties.
         * @param [properties] Properties to set
         * @returns SavedModel instance
         */
        public static create(properties?: tensorflow.ISavedModel): tensorflow.SavedModel;

        /**
         * Encodes the specified SavedModel message. Does not implicitly {@link tensorflow.SavedModel.verify|verify} messages.
         * @param message SavedModel message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.ISavedModel, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified SavedModel message, length delimited. Does not implicitly {@link tensorflow.SavedModel.verify|verify} messages.
         * @param message SavedModel message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.ISavedModel, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a SavedModel message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns SavedModel
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.SavedModel;

        /**
         * Decodes a SavedModel message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns SavedModel
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.SavedModel;

        /**
         * Verifies a SavedModel message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a SavedModel message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns SavedModel
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.SavedModel;

        /**
         * Creates a plain object from a SavedModel message. Also converts values to other types if specified.
         * @param message SavedModel
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.SavedModel, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this SavedModel to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }
}
