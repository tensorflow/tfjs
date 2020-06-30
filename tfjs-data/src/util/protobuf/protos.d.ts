import * as $protobuf from "protobufjs";
/** Namespace tensorflow. */
export namespace tensorflow {

    /** Properties of an Example. */
    interface IExample {

        /** Example features */
        features?: (tensorflow.IFeatures|null);
    }

    /** Represents an Example. */
    class Example implements IExample {

        /**
         * Constructs a new Example.
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.IExample);

        /** Example features. */
        public features?: (tensorflow.IFeatures|null);

        /**
         * Creates a new Example instance using the specified properties.
         * @param [properties] Properties to set
         * @returns Example instance
         */
        public static create(properties?: tensorflow.IExample): tensorflow.Example;

        /**
         * Encodes the specified Example message. Does not implicitly {@link tensorflow.Example.verify|verify} messages.
         * @param message Example message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.IExample, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified Example message, length delimited. Does not implicitly {@link tensorflow.Example.verify|verify} messages.
         * @param message Example message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.IExample, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an Example message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns Example
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.Example;

        /**
         * Decodes an Example message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns Example
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.Example;

        /**
         * Verifies an Example message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates an Example message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns Example
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.Example;

        /**
         * Creates a plain object from an Example message. Also converts values to other types if specified.
         * @param message Example
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.Example, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this Example to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a SequenceExample. */
    interface ISequenceExample {

        /** SequenceExample context */
        context?: (tensorflow.IFeatures|null);

        /** SequenceExample featureLists */
        featureLists?: (tensorflow.IFeatureLists|null);
    }

    /** Represents a SequenceExample. */
    class SequenceExample implements ISequenceExample {

        /**
         * Constructs a new SequenceExample.
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.ISequenceExample);

        /** SequenceExample context. */
        public context?: (tensorflow.IFeatures|null);

        /** SequenceExample featureLists. */
        public featureLists?: (tensorflow.IFeatureLists|null);

        /**
         * Creates a new SequenceExample instance using the specified properties.
         * @param [properties] Properties to set
         * @returns SequenceExample instance
         */
        public static create(properties?: tensorflow.ISequenceExample): tensorflow.SequenceExample;

        /**
         * Encodes the specified SequenceExample message. Does not implicitly {@link tensorflow.SequenceExample.verify|verify} messages.
         * @param message SequenceExample message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.ISequenceExample, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified SequenceExample message, length delimited. Does not implicitly {@link tensorflow.SequenceExample.verify|verify} messages.
         * @param message SequenceExample message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.ISequenceExample, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a SequenceExample message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns SequenceExample
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.SequenceExample;

        /**
         * Decodes a SequenceExample message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns SequenceExample
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.SequenceExample;

        /**
         * Verifies a SequenceExample message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a SequenceExample message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns SequenceExample
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.SequenceExample;

        /**
         * Creates a plain object from a SequenceExample message. Also converts values to other types if specified.
         * @param message SequenceExample
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.SequenceExample, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this SequenceExample to JSON.
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
        constructor(properties?: tensorflow.IBytesList);

        /** BytesList value. */
        public value: Uint8Array[];

        /**
         * Creates a new BytesList instance using the specified properties.
         * @param [properties] Properties to set
         * @returns BytesList instance
         */
        public static create(properties?: tensorflow.IBytesList): tensorflow.BytesList;

        /**
         * Encodes the specified BytesList message. Does not implicitly {@link tensorflow.BytesList.verify|verify} messages.
         * @param message BytesList message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.IBytesList, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified BytesList message, length delimited. Does not implicitly {@link tensorflow.BytesList.verify|verify} messages.
         * @param message BytesList message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.IBytesList, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a BytesList message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns BytesList
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.BytesList;

        /**
         * Decodes a BytesList message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns BytesList
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.BytesList;

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
        public static fromObject(object: { [k: string]: any }): tensorflow.BytesList;

        /**
         * Creates a plain object from a BytesList message. Also converts values to other types if specified.
         * @param message BytesList
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.BytesList, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this BytesList to JSON.
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
        constructor(properties?: tensorflow.IFloatList);

        /** FloatList value. */
        public value: number[];

        /**
         * Creates a new FloatList instance using the specified properties.
         * @param [properties] Properties to set
         * @returns FloatList instance
         */
        public static create(properties?: tensorflow.IFloatList): tensorflow.FloatList;

        /**
         * Encodes the specified FloatList message. Does not implicitly {@link tensorflow.FloatList.verify|verify} messages.
         * @param message FloatList message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.IFloatList, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified FloatList message, length delimited. Does not implicitly {@link tensorflow.FloatList.verify|verify} messages.
         * @param message FloatList message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.IFloatList, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a FloatList message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns FloatList
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.FloatList;

        /**
         * Decodes a FloatList message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns FloatList
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.FloatList;

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
        public static fromObject(object: { [k: string]: any }): tensorflow.FloatList;

        /**
         * Creates a plain object from a FloatList message. Also converts values to other types if specified.
         * @param message FloatList
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.FloatList, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this FloatList to JSON.
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
        constructor(properties?: tensorflow.IInt64List);

        /** Int64List value. */
        public value: (number|Long)[];

        /**
         * Creates a new Int64List instance using the specified properties.
         * @param [properties] Properties to set
         * @returns Int64List instance
         */
        public static create(properties?: tensorflow.IInt64List): tensorflow.Int64List;

        /**
         * Encodes the specified Int64List message. Does not implicitly {@link tensorflow.Int64List.verify|verify} messages.
         * @param message Int64List message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.IInt64List, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified Int64List message, length delimited. Does not implicitly {@link tensorflow.Int64List.verify|verify} messages.
         * @param message Int64List message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.IInt64List, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes an Int64List message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns Int64List
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.Int64List;

        /**
         * Decodes an Int64List message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns Int64List
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.Int64List;

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
        public static fromObject(object: { [k: string]: any }): tensorflow.Int64List;

        /**
         * Creates a plain object from an Int64List message. Also converts values to other types if specified.
         * @param message Int64List
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.Int64List, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this Int64List to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a Feature. */
    interface IFeature {

        /** Feature bytesList */
        bytesList?: (tensorflow.IBytesList|null);

        /** Feature floatList */
        floatList?: (tensorflow.IFloatList|null);

        /** Feature int64List */
        int64List?: (tensorflow.IInt64List|null);
    }

    /** Represents a Feature. */
    class Feature implements IFeature {

        /**
         * Constructs a new Feature.
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.IFeature);

        /** Feature bytesList. */
        public bytesList?: (tensorflow.IBytesList|null);

        /** Feature floatList. */
        public floatList?: (tensorflow.IFloatList|null);

        /** Feature int64List. */
        public int64List?: (tensorflow.IInt64List|null);

        /** Feature kind. */
        public kind?: ("bytesList"|"floatList"|"int64List");

        /**
         * Creates a new Feature instance using the specified properties.
         * @param [properties] Properties to set
         * @returns Feature instance
         */
        public static create(properties?: tensorflow.IFeature): tensorflow.Feature;

        /**
         * Encodes the specified Feature message. Does not implicitly {@link tensorflow.Feature.verify|verify} messages.
         * @param message Feature message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.IFeature, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified Feature message, length delimited. Does not implicitly {@link tensorflow.Feature.verify|verify} messages.
         * @param message Feature message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.IFeature, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a Feature message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns Feature
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.Feature;

        /**
         * Decodes a Feature message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns Feature
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.Feature;

        /**
         * Verifies a Feature message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a Feature message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns Feature
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.Feature;

        /**
         * Creates a plain object from a Feature message. Also converts values to other types if specified.
         * @param message Feature
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.Feature, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this Feature to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a Features. */
    interface IFeatures {

        /** Features feature */
        feature?: ({ [k: string]: tensorflow.IFeature }|null);
    }

    /** Represents a Features. */
    class Features implements IFeatures {

        /**
         * Constructs a new Features.
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.IFeatures);

        /** Features feature. */
        public feature: { [k: string]: tensorflow.IFeature };

        /**
         * Creates a new Features instance using the specified properties.
         * @param [properties] Properties to set
         * @returns Features instance
         */
        public static create(properties?: tensorflow.IFeatures): tensorflow.Features;

        /**
         * Encodes the specified Features message. Does not implicitly {@link tensorflow.Features.verify|verify} messages.
         * @param message Features message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.IFeatures, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified Features message, length delimited. Does not implicitly {@link tensorflow.Features.verify|verify} messages.
         * @param message Features message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.IFeatures, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a Features message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns Features
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.Features;

        /**
         * Decodes a Features message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns Features
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.Features;

        /**
         * Verifies a Features message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a Features message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns Features
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.Features;

        /**
         * Creates a plain object from a Features message. Also converts values to other types if specified.
         * @param message Features
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.Features, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this Features to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a FeatureList. */
    interface IFeatureList {

        /** FeatureList feature */
        feature?: (tensorflow.IFeature[]|null);
    }

    /** Represents a FeatureList. */
    class FeatureList implements IFeatureList {

        /**
         * Constructs a new FeatureList.
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.IFeatureList);

        /** FeatureList feature. */
        public feature: tensorflow.IFeature[];

        /**
         * Creates a new FeatureList instance using the specified properties.
         * @param [properties] Properties to set
         * @returns FeatureList instance
         */
        public static create(properties?: tensorflow.IFeatureList): tensorflow.FeatureList;

        /**
         * Encodes the specified FeatureList message. Does not implicitly {@link tensorflow.FeatureList.verify|verify} messages.
         * @param message FeatureList message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.IFeatureList, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified FeatureList message, length delimited. Does not implicitly {@link tensorflow.FeatureList.verify|verify} messages.
         * @param message FeatureList message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.IFeatureList, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a FeatureList message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns FeatureList
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.FeatureList;

        /**
         * Decodes a FeatureList message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns FeatureList
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.FeatureList;

        /**
         * Verifies a FeatureList message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a FeatureList message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns FeatureList
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.FeatureList;

        /**
         * Creates a plain object from a FeatureList message. Also converts values to other types if specified.
         * @param message FeatureList
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.FeatureList, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this FeatureList to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }

    /** Properties of a FeatureLists. */
    interface IFeatureLists {

        /** FeatureLists featureList */
        featureList?: ({ [k: string]: tensorflow.IFeatureList }|null);
    }

    /** Represents a FeatureLists. */
    class FeatureLists implements IFeatureLists {

        /**
         * Constructs a new FeatureLists.
         * @param [properties] Properties to set
         */
        constructor(properties?: tensorflow.IFeatureLists);

        /** FeatureLists featureList. */
        public featureList: { [k: string]: tensorflow.IFeatureList };

        /**
         * Creates a new FeatureLists instance using the specified properties.
         * @param [properties] Properties to set
         * @returns FeatureLists instance
         */
        public static create(properties?: tensorflow.IFeatureLists): tensorflow.FeatureLists;

        /**
         * Encodes the specified FeatureLists message. Does not implicitly {@link tensorflow.FeatureLists.verify|verify} messages.
         * @param message FeatureLists message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encode(message: tensorflow.IFeatureLists, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Encodes the specified FeatureLists message, length delimited. Does not implicitly {@link tensorflow.FeatureLists.verify|verify} messages.
         * @param message FeatureLists message or plain object to encode
         * @param [writer] Writer to encode to
         * @returns Writer
         */
        public static encodeDelimited(message: tensorflow.IFeatureLists, writer?: $protobuf.Writer): $protobuf.Writer;

        /**
         * Decodes a FeatureLists message from the specified reader or buffer.
         * @param reader Reader or buffer to decode from
         * @param [length] Message length if known beforehand
         * @returns FeatureLists
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): tensorflow.FeatureLists;

        /**
         * Decodes a FeatureLists message from the specified reader or buffer, length delimited.
         * @param reader Reader or buffer to decode from
         * @returns FeatureLists
         * @throws {Error} If the payload is not a reader or valid buffer
         * @throws {$protobuf.util.ProtocolError} If required fields are missing
         */
        public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): tensorflow.FeatureLists;

        /**
         * Verifies a FeatureLists message.
         * @param message Plain object to verify
         * @returns `null` if valid, otherwise the reason why it is not
         */
        public static verify(message: { [k: string]: any }): (string|null);

        /**
         * Creates a FeatureLists message from a plain object. Also converts values to their respective internal types.
         * @param object Plain object
         * @returns FeatureLists
         */
        public static fromObject(object: { [k: string]: any }): tensorflow.FeatureLists;

        /**
         * Creates a plain object from a FeatureLists message. Also converts values to other types if specified.
         * @param message FeatureLists
         * @param [options] Conversion options
         * @returns Plain object
         */
        public static toObject(message: tensorflow.FeatureLists, options?: $protobuf.IConversionOptions): { [k: string]: any };

        /**
         * Converts this FeatureLists to JSON.
         * @returns JSON object
         */
        public toJSON(): { [k: string]: any };
    }
}
