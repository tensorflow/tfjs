/*eslint-disable block-scoped-var, no-redeclare, no-control-regex, no-prototype-builtins*/
import * as $protobuf from "protobufjs/minimal";

const $Reader = $protobuf.Reader, $util = $protobuf.util;

const $root = $protobuf.roots["default"] || ($protobuf.roots["default"] = {});

export const tensorflow = $root.tensorflow = (() => {

    const tensorflow = {};

    tensorflow.Any = (function() {

        function Any(p) {
            if (p)
                for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                    if (p[ks[i]] != null)
                        this[ks[i]] = p[ks[i]];
        }

        Any.prototype.typeUrl = "";
        Any.prototype.value = $util.newBuffer([]);

        Any.decode = function decode(r, l) {
            if (!(r instanceof $Reader))
                r = $Reader.create(r);
            var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.Any();
            while (r.pos < c) {
                var t = r.uint32();
                switch (t >>> 3) {
                case 1:
                    m.typeUrl = r.string();
                    break;
                case 2:
                    m.value = r.bytes();
                    break;
                default:
                    r.skipType(t & 7);
                    break;
                }
            }
            return m;
        };

        return Any;
    })();

    tensorflow.DataType = (function() {
        const valuesById = {}, values = Object.create(valuesById);
        values[valuesById[0] = "DT_INVALID"] = 0;
        values[valuesById[1] = "DT_FLOAT"] = 1;
        values[valuesById[2] = "DT_DOUBLE"] = 2;
        values[valuesById[3] = "DT_INT32"] = 3;
        values[valuesById[4] = "DT_UINT8"] = 4;
        values[valuesById[5] = "DT_INT16"] = 5;
        values[valuesById[6] = "DT_INT8"] = 6;
        values[valuesById[7] = "DT_STRING"] = 7;
        values[valuesById[8] = "DT_COMPLEX64"] = 8;
        values[valuesById[9] = "DT_INT64"] = 9;
        values[valuesById[10] = "DT_BOOL"] = 10;
        values[valuesById[11] = "DT_QINT8"] = 11;
        values[valuesById[12] = "DT_QUINT8"] = 12;
        values[valuesById[13] = "DT_QINT32"] = 13;
        values[valuesById[14] = "DT_BFLOAT16"] = 14;
        values[valuesById[101] = "DT_FLOAT_REF"] = 101;
        values[valuesById[102] = "DT_DOUBLE_REF"] = 102;
        values[valuesById[103] = "DT_INT32_REF"] = 103;
        values[valuesById[104] = "DT_UINT8_REF"] = 104;
        values[valuesById[105] = "DT_INT16_REF"] = 105;
        values[valuesById[106] = "DT_INT8_REF"] = 106;
        values[valuesById[107] = "DT_STRING_REF"] = 107;
        values[valuesById[108] = "DT_COMPLEX64_REF"] = 108;
        values[valuesById[109] = "DT_INT64_REF"] = 109;
        values[valuesById[110] = "DT_BOOL_REF"] = 110;
        values[valuesById[111] = "DT_QINT8_REF"] = 111;
        values[valuesById[112] = "DT_QUINT8_REF"] = 112;
        values[valuesById[113] = "DT_QINT32_REF"] = 113;
        values[valuesById[114] = "DT_BFLOAT16_REF"] = 114;
        return values;
    })();

    tensorflow.TensorShape = (function() {

        function TensorShape(p) {
            this.dim = [];
            if (p)
                for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                    if (p[ks[i]] != null)
                        this[ks[i]] = p[ks[i]];
        }

        TensorShape.prototype.dim = $util.emptyArray;
        TensorShape.prototype.unknownRank = false;

        TensorShape.decode = function decode(r, l) {
            if (!(r instanceof $Reader))
                r = $Reader.create(r);
            var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.TensorShape();
            while (r.pos < c) {
                var t = r.uint32();
                switch (t >>> 3) {
                case 2:
                    if (!(m.dim && m.dim.length))
                        m.dim = [];
                    m.dim.push($root.tensorflow.TensorShape.Dim.decode(r, r.uint32()));
                    break;
                case 3:
                    m.unknownRank = r.bool();
                    break;
                default:
                    r.skipType(t & 7);
                    break;
                }
            }
            return m;
        };

        TensorShape.Dim = (function() {

            function Dim(p) {
                if (p)
                    for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                        if (p[ks[i]] != null)
                            this[ks[i]] = p[ks[i]];
            }

            Dim.prototype.size = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            Dim.prototype.name = "";

            Dim.decode = function decode(r, l) {
                if (!(r instanceof $Reader))
                    r = $Reader.create(r);
                var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.TensorShape.Dim();
                while (r.pos < c) {
                    var t = r.uint32();
                    switch (t >>> 3) {
                    case 1:
                        m.size = r.int64();
                        break;
                    case 2:
                        m.name = r.string();
                        break;
                    default:
                        r.skipType(t & 7);
                        break;
                    }
                }
                return m;
            };

            return Dim;
        })();

        return TensorShape;
    })();

    tensorflow.Tensor = (function() {

        function Tensor(p) {
            this.floatVal = [];
            this.doubleVal = [];
            this.intVal = [];
            this.stringVal = [];
            this.scomplexVal = [];
            this.int64Val = [];
            this.boolVal = [];
            this.uint32Val = [];
            this.uint64Val = [];
            if (p)
                for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                    if (p[ks[i]] != null)
                        this[ks[i]] = p[ks[i]];
        }

        Tensor.prototype.dtype = 0;
        Tensor.prototype.tensorShape = null;
        Tensor.prototype.versionNumber = 0;
        Tensor.prototype.tensorContent = $util.newBuffer([]);
        Tensor.prototype.floatVal = $util.emptyArray;
        Tensor.prototype.doubleVal = $util.emptyArray;
        Tensor.prototype.intVal = $util.emptyArray;
        Tensor.prototype.stringVal = $util.emptyArray;
        Tensor.prototype.scomplexVal = $util.emptyArray;
        Tensor.prototype.int64Val = $util.emptyArray;
        Tensor.prototype.boolVal = $util.emptyArray;
        Tensor.prototype.uint32Val = $util.emptyArray;
        Tensor.prototype.uint64Val = $util.emptyArray;

        Tensor.decode = function decode(r, l) {
            if (!(r instanceof $Reader))
                r = $Reader.create(r);
            var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.Tensor();
            while (r.pos < c) {
                var t = r.uint32();
                switch (t >>> 3) {
                case 1:
                    m.dtype = r.int32();
                    break;
                case 2:
                    m.tensorShape = $root.tensorflow.TensorShape.decode(r, r.uint32());
                    break;
                case 3:
                    m.versionNumber = r.int32();
                    break;
                case 4:
                    m.tensorContent = r.bytes();
                    break;
                case 5:
                    if (!(m.floatVal && m.floatVal.length))
                        m.floatVal = [];
                    if ((t & 7) === 2) {
                        var c2 = r.uint32() + r.pos;
                        while (r.pos < c2)
                            m.floatVal.push(r.float());
                    } else
                        m.floatVal.push(r.float());
                    break;
                case 6:
                    if (!(m.doubleVal && m.doubleVal.length))
                        m.doubleVal = [];
                    if ((t & 7) === 2) {
                        var c2 = r.uint32() + r.pos;
                        while (r.pos < c2)
                            m.doubleVal.push(r.double());
                    } else
                        m.doubleVal.push(r.double());
                    break;
                case 7:
                    if (!(m.intVal && m.intVal.length))
                        m.intVal = [];
                    if ((t & 7) === 2) {
                        var c2 = r.uint32() + r.pos;
                        while (r.pos < c2)
                            m.intVal.push(r.int32());
                    } else
                        m.intVal.push(r.int32());
                    break;
                case 8:
                    if (!(m.stringVal && m.stringVal.length))
                        m.stringVal = [];
                    m.stringVal.push(r.bytes());
                    break;
                case 9:
                    if (!(m.scomplexVal && m.scomplexVal.length))
                        m.scomplexVal = [];
                    if ((t & 7) === 2) {
                        var c2 = r.uint32() + r.pos;
                        while (r.pos < c2)
                            m.scomplexVal.push(r.float());
                    } else
                        m.scomplexVal.push(r.float());
                    break;
                case 10:
                    if (!(m.int64Val && m.int64Val.length))
                        m.int64Val = [];
                    if ((t & 7) === 2) {
                        var c2 = r.uint32() + r.pos;
                        while (r.pos < c2)
                            m.int64Val.push(r.int64());
                    } else
                        m.int64Val.push(r.int64());
                    break;
                case 11:
                    if (!(m.boolVal && m.boolVal.length))
                        m.boolVal = [];
                    if ((t & 7) === 2) {
                        var c2 = r.uint32() + r.pos;
                        while (r.pos < c2)
                            m.boolVal.push(r.bool());
                    } else
                        m.boolVal.push(r.bool());
                    break;
                case 16:
                    if (!(m.uint32Val && m.uint32Val.length))
                        m.uint32Val = [];
                    if ((t & 7) === 2) {
                        var c2 = r.uint32() + r.pos;
                        while (r.pos < c2)
                            m.uint32Val.push(r.uint32());
                    } else
                        m.uint32Val.push(r.uint32());
                    break;
                case 17:
                    if (!(m.uint64Val && m.uint64Val.length))
                        m.uint64Val = [];
                    if ((t & 7) === 2) {
                        var c2 = r.uint32() + r.pos;
                        while (r.pos < c2)
                            m.uint64Val.push(r.uint64());
                    } else
                        m.uint64Val.push(r.uint64());
                    break;
                default:
                    r.skipType(t & 7);
                    break;
                }
            }
            return m;
        };

        return Tensor;
    })();

    tensorflow.AttrValue = (function() {

        function AttrValue(p) {
            if (p)
                for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                    if (p[ks[i]] != null)
                        this[ks[i]] = p[ks[i]];
        }

        AttrValue.prototype.list = null;
        AttrValue.prototype.s = $util.newBuffer([]);
        AttrValue.prototype.i = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
        AttrValue.prototype.f = 0;
        AttrValue.prototype.b = false;
        AttrValue.prototype.type = 0;
        AttrValue.prototype.shape = null;
        AttrValue.prototype.tensor = null;
        AttrValue.prototype.placeholder = "";
        AttrValue.prototype.func = null;

        let $oneOfFields;

        Object.defineProperty(AttrValue.prototype, "value", {
            get: $util.oneOfGetter($oneOfFields = ["list", "s", "i", "f", "b", "type", "shape", "tensor", "placeholder", "func"]),
            set: $util.oneOfSetter($oneOfFields)
        });

        AttrValue.decode = function decode(r, l) {
            if (!(r instanceof $Reader))
                r = $Reader.create(r);
            var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.AttrValue();
            while (r.pos < c) {
                var t = r.uint32();
                switch (t >>> 3) {
                case 1:
                    m.list = $root.tensorflow.AttrValue.ListValue.decode(r, r.uint32());
                    break;
                case 2:
                    m.s = r.bytes();
                    break;
                case 3:
                    m.i = r.int64();
                    break;
                case 4:
                    m.f = r.float();
                    break;
                case 5:
                    m.b = r.bool();
                    break;
                case 6:
                    m.type = r.int32();
                    break;
                case 7:
                    m.shape = $root.tensorflow.TensorShape.decode(r, r.uint32());
                    break;
                case 8:
                    m.tensor = $root.tensorflow.Tensor.decode(r, r.uint32());
                    break;
                case 9:
                    m.placeholder = r.string();
                    break;
                case 10:
                    m.func = $root.tensorflow.NameAttrList.decode(r, r.uint32());
                    break;
                default:
                    r.skipType(t & 7);
                    break;
                }
            }
            return m;
        };

        AttrValue.ListValue = (function() {

            function ListValue(p) {
                this.s = [];
                this.i = [];
                this.f = [];
                this.b = [];
                this.type = [];
                this.shape = [];
                this.tensor = [];
                this.func = [];
                if (p)
                    for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                        if (p[ks[i]] != null)
                            this[ks[i]] = p[ks[i]];
            }

            ListValue.prototype.s = $util.emptyArray;
            ListValue.prototype.i = $util.emptyArray;
            ListValue.prototype.f = $util.emptyArray;
            ListValue.prototype.b = $util.emptyArray;
            ListValue.prototype.type = $util.emptyArray;
            ListValue.prototype.shape = $util.emptyArray;
            ListValue.prototype.tensor = $util.emptyArray;
            ListValue.prototype.func = $util.emptyArray;

            ListValue.decode = function decode(r, l) {
                if (!(r instanceof $Reader))
                    r = $Reader.create(r);
                var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.AttrValue.ListValue();
                while (r.pos < c) {
                    var t = r.uint32();
                    switch (t >>> 3) {
                    case 2:
                        if (!(m.s && m.s.length))
                            m.s = [];
                        m.s.push(r.bytes());
                        break;
                    case 3:
                        if (!(m.i && m.i.length))
                            m.i = [];
                        if ((t & 7) === 2) {
                            var c2 = r.uint32() + r.pos;
                            while (r.pos < c2)
                                m.i.push(r.int64());
                        } else
                            m.i.push(r.int64());
                        break;
                    case 4:
                        if (!(m.f && m.f.length))
                            m.f = [];
                        if ((t & 7) === 2) {
                            var c2 = r.uint32() + r.pos;
                            while (r.pos < c2)
                                m.f.push(r.float());
                        } else
                            m.f.push(r.float());
                        break;
                    case 5:
                        if (!(m.b && m.b.length))
                            m.b = [];
                        if ((t & 7) === 2) {
                            var c2 = r.uint32() + r.pos;
                            while (r.pos < c2)
                                m.b.push(r.bool());
                        } else
                            m.b.push(r.bool());
                        break;
                    case 6:
                        if (!(m.type && m.type.length))
                            m.type = [];
                        if ((t & 7) === 2) {
                            var c2 = r.uint32() + r.pos;
                            while (r.pos < c2)
                                m.type.push(r.int32());
                        } else
                            m.type.push(r.int32());
                        break;
                    case 7:
                        if (!(m.shape && m.shape.length))
                            m.shape = [];
                        m.shape.push($root.tensorflow.TensorShape.decode(r, r.uint32()));
                        break;
                    case 8:
                        if (!(m.tensor && m.tensor.length))
                            m.tensor = [];
                        m.tensor.push($root.tensorflow.Tensor.decode(r, r.uint32()));
                        break;
                    case 9:
                        if (!(m.func && m.func.length))
                            m.func = [];
                        m.func.push($root.tensorflow.NameAttrList.decode(r, r.uint32()));
                        break;
                    default:
                        r.skipType(t & 7);
                        break;
                    }
                }
                return m;
            };

            return ListValue;
        })();

        return AttrValue;
    })();

    tensorflow.NameAttrList = (function() {

        function NameAttrList(p) {
            this.attr = {};
            if (p)
                for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                    if (p[ks[i]] != null)
                        this[ks[i]] = p[ks[i]];
        }

        NameAttrList.prototype.name = "";
        NameAttrList.prototype.attr = $util.emptyObject;

        NameAttrList.decode = function decode(r, l) {
            if (!(r instanceof $Reader))
                r = $Reader.create(r);
            var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.NameAttrList(), k;
            while (r.pos < c) {
                var t = r.uint32();
                switch (t >>> 3) {
                case 1:
                    m.name = r.string();
                    break;
                case 2:
                    r.skip().pos++;
                    if (m.attr === $util.emptyObject)
                        m.attr = {};
                    k = r.string();
                    r.pos++;
                    m.attr[k] = $root.tensorflow.AttrValue.decode(r, r.uint32());
                    break;
                default:
                    r.skipType(t & 7);
                    break;
                }
            }
            return m;
        };

        return NameAttrList;
    })();

    tensorflow.NodeDef = (function() {

        function NodeDef(p) {
            this.input = [];
            this.attr = {};
            if (p)
                for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                    if (p[ks[i]] != null)
                        this[ks[i]] = p[ks[i]];
        }

        NodeDef.prototype.name = "";
        NodeDef.prototype.op = "";
        NodeDef.prototype.input = $util.emptyArray;
        NodeDef.prototype.device = "";
        NodeDef.prototype.attr = $util.emptyObject;

        NodeDef.decode = function decode(r, l) {
            if (!(r instanceof $Reader))
                r = $Reader.create(r);
            var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.NodeDef(), k;
            while (r.pos < c) {
                var t = r.uint32();
                switch (t >>> 3) {
                case 1:
                    m.name = r.string();
                    break;
                case 2:
                    m.op = r.string();
                    break;
                case 3:
                    if (!(m.input && m.input.length))
                        m.input = [];
                    m.input.push(r.string());
                    break;
                case 4:
                    m.device = r.string();
                    break;
                case 5:
                    r.skip().pos++;
                    if (m.attr === $util.emptyObject)
                        m.attr = {};
                    k = r.string();
                    r.pos++;
                    m.attr[k] = $root.tensorflow.AttrValue.decode(r, r.uint32());
                    break;
                default:
                    r.skipType(t & 7);
                    break;
                }
            }
            return m;
        };

        return NodeDef;
    })();

    tensorflow.VersionDef = (function() {

        function VersionDef(p) {
            this.badConsumers = [];
            if (p)
                for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                    if (p[ks[i]] != null)
                        this[ks[i]] = p[ks[i]];
        }

        VersionDef.prototype.producer = 0;
        VersionDef.prototype.minConsumer = 0;
        VersionDef.prototype.badConsumers = $util.emptyArray;

        VersionDef.decode = function decode(r, l) {
            if (!(r instanceof $Reader))
                r = $Reader.create(r);
            var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.VersionDef();
            while (r.pos < c) {
                var t = r.uint32();
                switch (t >>> 3) {
                case 1:
                    m.producer = r.int32();
                    break;
                case 2:
                    m.minConsumer = r.int32();
                    break;
                case 3:
                    if (!(m.badConsumers && m.badConsumers.length))
                        m.badConsumers = [];
                    if ((t & 7) === 2) {
                        var c2 = r.uint32() + r.pos;
                        while (r.pos < c2)
                            m.badConsumers.push(r.int32());
                    } else
                        m.badConsumers.push(r.int32());
                    break;
                default:
                    r.skipType(t & 7);
                    break;
                }
            }
            return m;
        };

        return VersionDef;
    })();

    tensorflow.GraphDef = (function() {

        function GraphDef(p) {
            this.node = [];
            if (p)
                for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                    if (p[ks[i]] != null)
                        this[ks[i]] = p[ks[i]];
        }

        GraphDef.prototype.node = $util.emptyArray;
        GraphDef.prototype.versions = null;
        GraphDef.prototype.library = null;

        GraphDef.decode = function decode(r, l) {
            if (!(r instanceof $Reader))
                r = $Reader.create(r);
            var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.GraphDef();
            while (r.pos < c) {
                var t = r.uint32();
                switch (t >>> 3) {
                case 1:
                    if (!(m.node && m.node.length))
                        m.node = [];
                    m.node.push($root.tensorflow.NodeDef.decode(r, r.uint32()));
                    break;
                case 4:
                    m.versions = $root.tensorflow.VersionDef.decode(r, r.uint32());
                    break;
                case 2:
                    m.library = $root.tensorflow.FunctionDefLibrary.decode(r, r.uint32());
                    break;
                default:
                    r.skipType(t & 7);
                    break;
                }
            }
            return m;
        };

        return GraphDef;
    })();

    tensorflow.CollectionDef = (function() {

        function CollectionDef(p) {
            if (p)
                for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                    if (p[ks[i]] != null)
                        this[ks[i]] = p[ks[i]];
        }

        CollectionDef.prototype.nodeList = null;
        CollectionDef.prototype.bytesList = null;
        CollectionDef.prototype.int64List = null;
        CollectionDef.prototype.floatList = null;
        CollectionDef.prototype.anyList = null;

        let $oneOfFields;

        Object.defineProperty(CollectionDef.prototype, "kind", {
            get: $util.oneOfGetter($oneOfFields = ["nodeList", "bytesList", "int64List", "floatList", "anyList"]),
            set: $util.oneOfSetter($oneOfFields)
        });

        CollectionDef.decode = function decode(r, l) {
            if (!(r instanceof $Reader))
                r = $Reader.create(r);
            var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.CollectionDef();
            while (r.pos < c) {
                var t = r.uint32();
                switch (t >>> 3) {
                case 1:
                    m.nodeList = $root.tensorflow.CollectionDef.NodeList.decode(r, r.uint32());
                    break;
                case 2:
                    m.bytesList = $root.tensorflow.CollectionDef.BytesList.decode(r, r.uint32());
                    break;
                case 3:
                    m.int64List = $root.tensorflow.CollectionDef.Int64List.decode(r, r.uint32());
                    break;
                case 4:
                    m.floatList = $root.tensorflow.CollectionDef.FloatList.decode(r, r.uint32());
                    break;
                case 5:
                    m.anyList = $root.tensorflow.CollectionDef.AnyList.decode(r, r.uint32());
                    break;
                default:
                    r.skipType(t & 7);
                    break;
                }
            }
            return m;
        };

        CollectionDef.NodeList = (function() {

            function NodeList(p) {
                this.value = [];
                if (p)
                    for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                        if (p[ks[i]] != null)
                            this[ks[i]] = p[ks[i]];
            }

            NodeList.prototype.value = $util.emptyArray;

            NodeList.decode = function decode(r, l) {
                if (!(r instanceof $Reader))
                    r = $Reader.create(r);
                var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.CollectionDef.NodeList();
                while (r.pos < c) {
                    var t = r.uint32();
                    switch (t >>> 3) {
                    case 1:
                        if (!(m.value && m.value.length))
                            m.value = [];
                        m.value.push(r.string());
                        break;
                    default:
                        r.skipType(t & 7);
                        break;
                    }
                }
                return m;
            };

            return NodeList;
        })();

        CollectionDef.BytesList = (function() {

            function BytesList(p) {
                this.value = [];
                if (p)
                    for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                        if (p[ks[i]] != null)
                            this[ks[i]] = p[ks[i]];
            }

            BytesList.prototype.value = $util.emptyArray;

            BytesList.decode = function decode(r, l) {
                if (!(r instanceof $Reader))
                    r = $Reader.create(r);
                var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.CollectionDef.BytesList();
                while (r.pos < c) {
                    var t = r.uint32();
                    switch (t >>> 3) {
                    case 1:
                        if (!(m.value && m.value.length))
                            m.value = [];
                        m.value.push(r.bytes());
                        break;
                    default:
                        r.skipType(t & 7);
                        break;
                    }
                }
                return m;
            };

            return BytesList;
        })();

        CollectionDef.Int64List = (function() {

            function Int64List(p) {
                this.value = [];
                if (p)
                    for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                        if (p[ks[i]] != null)
                            this[ks[i]] = p[ks[i]];
            }

            Int64List.prototype.value = $util.emptyArray;

            Int64List.decode = function decode(r, l) {
                if (!(r instanceof $Reader))
                    r = $Reader.create(r);
                var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.CollectionDef.Int64List();
                while (r.pos < c) {
                    var t = r.uint32();
                    switch (t >>> 3) {
                    case 1:
                        if (!(m.value && m.value.length))
                            m.value = [];
                        if ((t & 7) === 2) {
                            var c2 = r.uint32() + r.pos;
                            while (r.pos < c2)
                                m.value.push(r.int64());
                        } else
                            m.value.push(r.int64());
                        break;
                    default:
                        r.skipType(t & 7);
                        break;
                    }
                }
                return m;
            };

            return Int64List;
        })();

        CollectionDef.FloatList = (function() {

            function FloatList(p) {
                this.value = [];
                if (p)
                    for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                        if (p[ks[i]] != null)
                            this[ks[i]] = p[ks[i]];
            }

            FloatList.prototype.value = $util.emptyArray;

            FloatList.decode = function decode(r, l) {
                if (!(r instanceof $Reader))
                    r = $Reader.create(r);
                var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.CollectionDef.FloatList();
                while (r.pos < c) {
                    var t = r.uint32();
                    switch (t >>> 3) {
                    case 1:
                        if (!(m.value && m.value.length))
                            m.value = [];
                        if ((t & 7) === 2) {
                            var c2 = r.uint32() + r.pos;
                            while (r.pos < c2)
                                m.value.push(r.float());
                        } else
                            m.value.push(r.float());
                        break;
                    default:
                        r.skipType(t & 7);
                        break;
                    }
                }
                return m;
            };

            return FloatList;
        })();

        CollectionDef.AnyList = (function() {

            function AnyList(p) {
                this.value = [];
                if (p)
                    for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                        if (p[ks[i]] != null)
                            this[ks[i]] = p[ks[i]];
            }

            AnyList.prototype.value = $util.emptyArray;

            AnyList.decode = function decode(r, l) {
                if (!(r instanceof $Reader))
                    r = $Reader.create(r);
                var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.CollectionDef.AnyList();
                while (r.pos < c) {
                    var t = r.uint32();
                    switch (t >>> 3) {
                    case 1:
                        if (!(m.value && m.value.length))
                            m.value = [];
                        m.value.push($root.tensorflow.Any.decode(r, r.uint32()));
                        break;
                    default:
                        r.skipType(t & 7);
                        break;
                    }
                }
                return m;
            };

            return AnyList;
        })();

        return CollectionDef;
    })();

    tensorflow.SaverDef = (function() {

        function SaverDef(p) {
            if (p)
                for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                    if (p[ks[i]] != null)
                        this[ks[i]] = p[ks[i]];
        }

        SaverDef.prototype.filenameTensorName = "";
        SaverDef.prototype.saveTensorName = "";
        SaverDef.prototype.restoreOpName = "";
        SaverDef.prototype.maxToKeep = 0;
        SaverDef.prototype.sharded = false;
        SaverDef.prototype.keepCheckpointEveryNHours = 0;
        SaverDef.prototype.version = 0;

        SaverDef.decode = function decode(r, l) {
            if (!(r instanceof $Reader))
                r = $Reader.create(r);
            var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.SaverDef();
            while (r.pos < c) {
                var t = r.uint32();
                switch (t >>> 3) {
                case 1:
                    m.filenameTensorName = r.string();
                    break;
                case 2:
                    m.saveTensorName = r.string();
                    break;
                case 3:
                    m.restoreOpName = r.string();
                    break;
                case 4:
                    m.maxToKeep = r.int32();
                    break;
                case 5:
                    m.sharded = r.bool();
                    break;
                case 6:
                    m.keepCheckpointEveryNHours = r.float();
                    break;
                case 7:
                    m.version = r.int32();
                    break;
                default:
                    r.skipType(t & 7);
                    break;
                }
            }
            return m;
        };

        SaverDef.CheckpointFormatVersion = (function() {
            const valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "LEGACY"] = 0;
            values[valuesById[1] = "V1"] = 1;
            values[valuesById[2] = "V2"] = 2;
            return values;
        })();

        return SaverDef;
    })();

    tensorflow.TensorInfo = (function() {

        function TensorInfo(p) {
            if (p)
                for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                    if (p[ks[i]] != null)
                        this[ks[i]] = p[ks[i]];
        }

        TensorInfo.prototype.name = "";
        TensorInfo.prototype.cooSparse = null;
        TensorInfo.prototype.dtype = 0;
        TensorInfo.prototype.tensorShape = null;

        let $oneOfFields;

        Object.defineProperty(TensorInfo.prototype, "encoding", {
            get: $util.oneOfGetter($oneOfFields = ["name", "cooSparse"]),
            set: $util.oneOfSetter($oneOfFields)
        });

        TensorInfo.decode = function decode(r, l) {
            if (!(r instanceof $Reader))
                r = $Reader.create(r);
            var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.TensorInfo();
            while (r.pos < c) {
                var t = r.uint32();
                switch (t >>> 3) {
                case 1:
                    m.name = r.string();
                    break;
                case 4:
                    m.cooSparse = $root.tensorflow.TensorInfo.CooSparse.decode(r, r.uint32());
                    break;
                case 2:
                    m.dtype = r.int32();
                    break;
                case 3:
                    m.tensorShape = $root.tensorflow.TensorShape.decode(r, r.uint32());
                    break;
                default:
                    r.skipType(t & 7);
                    break;
                }
            }
            return m;
        };

        TensorInfo.CooSparse = (function() {

            function CooSparse(p) {
                if (p)
                    for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                        if (p[ks[i]] != null)
                            this[ks[i]] = p[ks[i]];
            }

            CooSparse.prototype.valuesTensorName = "";
            CooSparse.prototype.indicesTensorName = "";
            CooSparse.prototype.denseShapeTensorName = "";

            CooSparse.decode = function decode(r, l) {
                if (!(r instanceof $Reader))
                    r = $Reader.create(r);
                var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.TensorInfo.CooSparse();
                while (r.pos < c) {
                    var t = r.uint32();
                    switch (t >>> 3) {
                    case 1:
                        m.valuesTensorName = r.string();
                        break;
                    case 2:
                        m.indicesTensorName = r.string();
                        break;
                    case 3:
                        m.denseShapeTensorName = r.string();
                        break;
                    default:
                        r.skipType(t & 7);
                        break;
                    }
                }
                return m;
            };

            return CooSparse;
        })();

        return TensorInfo;
    })();

    tensorflow.SignatureDef = (function() {

        function SignatureDef(p) {
            this.inputs = {};
            this.outputs = {};
            if (p)
                for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                    if (p[ks[i]] != null)
                        this[ks[i]] = p[ks[i]];
        }

        SignatureDef.prototype.inputs = $util.emptyObject;
        SignatureDef.prototype.outputs = $util.emptyObject;
        SignatureDef.prototype.methodName = "";

        SignatureDef.decode = function decode(r, l) {
            if (!(r instanceof $Reader))
                r = $Reader.create(r);
            var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.SignatureDef(), k;
            while (r.pos < c) {
                var t = r.uint32();
                switch (t >>> 3) {
                case 1:
                    r.skip().pos++;
                    if (m.inputs === $util.emptyObject)
                        m.inputs = {};
                    k = r.string();
                    r.pos++;
                    m.inputs[k] = $root.tensorflow.TensorInfo.decode(r, r.uint32());
                    break;
                case 2:
                    r.skip().pos++;
                    if (m.outputs === $util.emptyObject)
                        m.outputs = {};
                    k = r.string();
                    r.pos++;
                    m.outputs[k] = $root.tensorflow.TensorInfo.decode(r, r.uint32());
                    break;
                case 3:
                    m.methodName = r.string();
                    break;
                default:
                    r.skipType(t & 7);
                    break;
                }
            }
            return m;
        };

        return SignatureDef;
    })();

    tensorflow.AssetFileDef = (function() {

        function AssetFileDef(p) {
            if (p)
                for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                    if (p[ks[i]] != null)
                        this[ks[i]] = p[ks[i]];
        }

        AssetFileDef.prototype.tensorInfo = null;
        AssetFileDef.prototype.filename = "";

        AssetFileDef.decode = function decode(r, l) {
            if (!(r instanceof $Reader))
                r = $Reader.create(r);
            var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.AssetFileDef();
            while (r.pos < c) {
                var t = r.uint32();
                switch (t >>> 3) {
                case 1:
                    m.tensorInfo = $root.tensorflow.TensorInfo.decode(r, r.uint32());
                    break;
                case 2:
                    m.filename = r.string();
                    break;
                default:
                    r.skipType(t & 7);
                    break;
                }
            }
            return m;
        };

        return AssetFileDef;
    })();

    tensorflow.OpDef = (function() {

        function OpDef(p) {
            this.inputArg = [];
            this.outputArg = [];
            this.attr = [];
            if (p)
                for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                    if (p[ks[i]] != null)
                        this[ks[i]] = p[ks[i]];
        }

        OpDef.prototype.name = "";
        OpDef.prototype.inputArg = $util.emptyArray;
        OpDef.prototype.outputArg = $util.emptyArray;
        OpDef.prototype.attr = $util.emptyArray;
        OpDef.prototype.deprecation = null;
        OpDef.prototype.summary = "";
        OpDef.prototype.description = "";
        OpDef.prototype.isCommutative = false;
        OpDef.prototype.isAggregate = false;
        OpDef.prototype.isStateful = false;
        OpDef.prototype.allowsUninitializedInput = false;

        OpDef.decode = function decode(r, l) {
            if (!(r instanceof $Reader))
                r = $Reader.create(r);
            var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.OpDef();
            while (r.pos < c) {
                var t = r.uint32();
                switch (t >>> 3) {
                case 1:
                    m.name = r.string();
                    break;
                case 2:
                    if (!(m.inputArg && m.inputArg.length))
                        m.inputArg = [];
                    m.inputArg.push($root.tensorflow.OpDef.ArgDef.decode(r, r.uint32()));
                    break;
                case 3:
                    if (!(m.outputArg && m.outputArg.length))
                        m.outputArg = [];
                    m.outputArg.push($root.tensorflow.OpDef.ArgDef.decode(r, r.uint32()));
                    break;
                case 4:
                    if (!(m.attr && m.attr.length))
                        m.attr = [];
                    m.attr.push($root.tensorflow.OpDef.AttrDef.decode(r, r.uint32()));
                    break;
                case 8:
                    m.deprecation = $root.tensorflow.OpDef.OpDeprecation.decode(r, r.uint32());
                    break;
                case 5:
                    m.summary = r.string();
                    break;
                case 6:
                    m.description = r.string();
                    break;
                case 18:
                    m.isCommutative = r.bool();
                    break;
                case 16:
                    m.isAggregate = r.bool();
                    break;
                case 17:
                    m.isStateful = r.bool();
                    break;
                case 19:
                    m.allowsUninitializedInput = r.bool();
                    break;
                default:
                    r.skipType(t & 7);
                    break;
                }
            }
            return m;
        };

        OpDef.ArgDef = (function() {

            function ArgDef(p) {
                if (p)
                    for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                        if (p[ks[i]] != null)
                            this[ks[i]] = p[ks[i]];
            }

            ArgDef.prototype.name = "";
            ArgDef.prototype.description = "";
            ArgDef.prototype.type = 0;
            ArgDef.prototype.typeAttr = "";
            ArgDef.prototype.numberAttr = "";
            ArgDef.prototype.typeListAttr = "";
            ArgDef.prototype.isRef = false;

            ArgDef.decode = function decode(r, l) {
                if (!(r instanceof $Reader))
                    r = $Reader.create(r);
                var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.OpDef.ArgDef();
                while (r.pos < c) {
                    var t = r.uint32();
                    switch (t >>> 3) {
                    case 1:
                        m.name = r.string();
                        break;
                    case 2:
                        m.description = r.string();
                        break;
                    case 3:
                        m.type = r.int32();
                        break;
                    case 4:
                        m.typeAttr = r.string();
                        break;
                    case 5:
                        m.numberAttr = r.string();
                        break;
                    case 6:
                        m.typeListAttr = r.string();
                        break;
                    case 16:
                        m.isRef = r.bool();
                        break;
                    default:
                        r.skipType(t & 7);
                        break;
                    }
                }
                return m;
            };

            return ArgDef;
        })();

        OpDef.AttrDef = (function() {

            function AttrDef(p) {
                if (p)
                    for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                        if (p[ks[i]] != null)
                            this[ks[i]] = p[ks[i]];
            }

            AttrDef.prototype.name = "";
            AttrDef.prototype.type = "";
            AttrDef.prototype.defaultValue = null;
            AttrDef.prototype.description = "";
            AttrDef.prototype.hasMinimum = false;
            AttrDef.prototype.minimum = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            AttrDef.prototype.allowedValues = null;

            AttrDef.decode = function decode(r, l) {
                if (!(r instanceof $Reader))
                    r = $Reader.create(r);
                var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.OpDef.AttrDef();
                while (r.pos < c) {
                    var t = r.uint32();
                    switch (t >>> 3) {
                    case 1:
                        m.name = r.string();
                        break;
                    case 2:
                        m.type = r.string();
                        break;
                    case 3:
                        m.defaultValue = $root.tensorflow.AttrValue.decode(r, r.uint32());
                        break;
                    case 4:
                        m.description = r.string();
                        break;
                    case 5:
                        m.hasMinimum = r.bool();
                        break;
                    case 6:
                        m.minimum = r.int64();
                        break;
                    case 7:
                        m.allowedValues = $root.tensorflow.AttrValue.decode(r, r.uint32());
                        break;
                    default:
                        r.skipType(t & 7);
                        break;
                    }
                }
                return m;
            };

            return AttrDef;
        })();

        OpDef.OpDeprecation = (function() {

            function OpDeprecation(p) {
                if (p)
                    for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                        if (p[ks[i]] != null)
                            this[ks[i]] = p[ks[i]];
            }

            OpDeprecation.prototype.version = 0;
            OpDeprecation.prototype.explanation = "";

            OpDeprecation.decode = function decode(r, l) {
                if (!(r instanceof $Reader))
                    r = $Reader.create(r);
                var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.OpDef.OpDeprecation();
                while (r.pos < c) {
                    var t = r.uint32();
                    switch (t >>> 3) {
                    case 1:
                        m.version = r.int32();
                        break;
                    case 2:
                        m.explanation = r.string();
                        break;
                    default:
                        r.skipType(t & 7);
                        break;
                    }
                }
                return m;
            };

            return OpDeprecation;
        })();

        return OpDef;
    })();

    tensorflow.OpList = (function() {

        function OpList(p) {
            this.op = [];
            if (p)
                for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                    if (p[ks[i]] != null)
                        this[ks[i]] = p[ks[i]];
        }

        OpList.prototype.op = $util.emptyArray;

        OpList.decode = function decode(r, l) {
            if (!(r instanceof $Reader))
                r = $Reader.create(r);
            var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.OpList();
            while (r.pos < c) {
                var t = r.uint32();
                switch (t >>> 3) {
                case 1:
                    if (!(m.op && m.op.length))
                        m.op = [];
                    m.op.push($root.tensorflow.OpDef.decode(r, r.uint32()));
                    break;
                default:
                    r.skipType(t & 7);
                    break;
                }
            }
            return m;
        };

        return OpList;
    })();

    tensorflow.MetaGraphDef = (function() {

        function MetaGraphDef(p) {
            this.collectionDef = {};
            this.signatureDef = {};
            this.assetFileDef = [];
            if (p)
                for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                    if (p[ks[i]] != null)
                        this[ks[i]] = p[ks[i]];
        }

        MetaGraphDef.prototype.metaInfoDef = null;
        MetaGraphDef.prototype.graphDef = null;
        MetaGraphDef.prototype.saverDef = null;
        MetaGraphDef.prototype.collectionDef = $util.emptyObject;
        MetaGraphDef.prototype.signatureDef = $util.emptyObject;
        MetaGraphDef.prototype.assetFileDef = $util.emptyArray;

        MetaGraphDef.decode = function decode(r, l) {
            if (!(r instanceof $Reader))
                r = $Reader.create(r);
            var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.MetaGraphDef(), k;
            while (r.pos < c) {
                var t = r.uint32();
                switch (t >>> 3) {
                case 1:
                    m.metaInfoDef = $root.tensorflow.MetaGraphDef.MetaInfoDef.decode(r, r.uint32());
                    break;
                case 2:
                    m.graphDef = $root.tensorflow.GraphDef.decode(r, r.uint32());
                    break;
                case 3:
                    m.saverDef = $root.tensorflow.SaverDef.decode(r, r.uint32());
                    break;
                case 4:
                    r.skip().pos++;
                    if (m.collectionDef === $util.emptyObject)
                        m.collectionDef = {};
                    k = r.string();
                    r.pos++;
                    m.collectionDef[k] = $root.tensorflow.CollectionDef.decode(r, r.uint32());
                    break;
                case 5:
                    r.skip().pos++;
                    if (m.signatureDef === $util.emptyObject)
                        m.signatureDef = {};
                    k = r.string();
                    r.pos++;
                    m.signatureDef[k] = $root.tensorflow.SignatureDef.decode(r, r.uint32());
                    break;
                case 6:
                    if (!(m.assetFileDef && m.assetFileDef.length))
                        m.assetFileDef = [];
                    m.assetFileDef.push($root.tensorflow.AssetFileDef.decode(r, r.uint32()));
                    break;
                default:
                    r.skipType(t & 7);
                    break;
                }
            }
            return m;
        };

        MetaGraphDef.MetaInfoDef = (function() {

            function MetaInfoDef(p) {
                this.tags = [];
                if (p)
                    for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                        if (p[ks[i]] != null)
                            this[ks[i]] = p[ks[i]];
            }

            MetaInfoDef.prototype.metaGraphVersion = "";
            MetaInfoDef.prototype.strippedOpList = null;
            MetaInfoDef.prototype.anyInfo = null;
            MetaInfoDef.prototype.tags = $util.emptyArray;
            MetaInfoDef.prototype.tensorflowVersion = "";
            MetaInfoDef.prototype.tensorflowGitVersion = "";

            MetaInfoDef.decode = function decode(r, l) {
                if (!(r instanceof $Reader))
                    r = $Reader.create(r);
                var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.MetaGraphDef.MetaInfoDef();
                while (r.pos < c) {
                    var t = r.uint32();
                    switch (t >>> 3) {
                    case 1:
                        m.metaGraphVersion = r.string();
                        break;
                    case 2:
                        m.strippedOpList = $root.tensorflow.OpList.decode(r, r.uint32());
                        break;
                    case 3:
                        m.anyInfo = $root.tensorflow.Any.decode(r, r.uint32());
                        break;
                    case 4:
                        if (!(m.tags && m.tags.length))
                            m.tags = [];
                        m.tags.push(r.string());
                        break;
                    case 5:
                        m.tensorflowVersion = r.string();
                        break;
                    case 6:
                        m.tensorflowGitVersion = r.string();
                        break;
                    default:
                        r.skipType(t & 7);
                        break;
                    }
                }
                return m;
            };

            return MetaInfoDef;
        })();

        return MetaGraphDef;
    })();

    tensorflow.SavedModel = (function() {

        function SavedModel(p) {
            this.metaGraphs = [];
            if (p)
                for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                    if (p[ks[i]] != null)
                        this[ks[i]] = p[ks[i]];
        }

        SavedModel.prototype.savedModelSchemaVersion = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
        SavedModel.prototype.metaGraphs = $util.emptyArray;

        SavedModel.decode = function decode(r, l) {
            if (!(r instanceof $Reader))
                r = $Reader.create(r);
            var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.SavedModel();
            while (r.pos < c) {
                var t = r.uint32();
                switch (t >>> 3) {
                case 1:
                    m.savedModelSchemaVersion = r.int64();
                    break;
                case 2:
                    if (!(m.metaGraphs && m.metaGraphs.length))
                        m.metaGraphs = [];
                    m.metaGraphs.push($root.tensorflow.MetaGraphDef.decode(r, r.uint32()));
                    break;
                default:
                    r.skipType(t & 7);
                    break;
                }
            }
            return m;
        };

        return SavedModel;
    })();

    tensorflow.FunctionDefLibrary = (function() {

        function FunctionDefLibrary(p) {
            this["function"] = [];
            this.gradient = [];
            if (p)
                for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                    if (p[ks[i]] != null)
                        this[ks[i]] = p[ks[i]];
        }

        FunctionDefLibrary.prototype["function"] = $util.emptyArray;
        FunctionDefLibrary.prototype.gradient = $util.emptyArray;

        FunctionDefLibrary.decode = function decode(r, l) {
            if (!(r instanceof $Reader))
                r = $Reader.create(r);
            var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.FunctionDefLibrary();
            while (r.pos < c) {
                var t = r.uint32();
                switch (t >>> 3) {
                case 1:
                    if (!(m["function"] && m["function"].length))
                        m["function"] = [];
                    m["function"].push($root.tensorflow.FunctionDef.decode(r, r.uint32()));
                    break;
                case 2:
                    if (!(m.gradient && m.gradient.length))
                        m.gradient = [];
                    m.gradient.push($root.tensorflow.GradientDef.decode(r, r.uint32()));
                    break;
                default:
                    r.skipType(t & 7);
                    break;
                }
            }
            return m;
        };

        return FunctionDefLibrary;
    })();

    tensorflow.FunctionDef = (function() {

        function FunctionDef(p) {
            this.attr = {};
            this.nodeDef = [];
            this.ret = {};
            if (p)
                for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                    if (p[ks[i]] != null)
                        this[ks[i]] = p[ks[i]];
        }

        FunctionDef.prototype.signature = null;
        FunctionDef.prototype.attr = $util.emptyObject;
        FunctionDef.prototype.nodeDef = $util.emptyArray;
        FunctionDef.prototype.ret = $util.emptyObject;

        FunctionDef.decode = function decode(r, l) {
            if (!(r instanceof $Reader))
                r = $Reader.create(r);
            var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.FunctionDef(), k;
            while (r.pos < c) {
                var t = r.uint32();
                switch (t >>> 3) {
                case 1:
                    m.signature = $root.tensorflow.OpDef.decode(r, r.uint32());
                    break;
                case 5:
                    r.skip().pos++;
                    if (m.attr === $util.emptyObject)
                        m.attr = {};
                    k = r.string();
                    r.pos++;
                    m.attr[k] = $root.tensorflow.AttrValue.decode(r, r.uint32());
                    break;
                case 3:
                    if (!(m.nodeDef && m.nodeDef.length))
                        m.nodeDef = [];
                    m.nodeDef.push($root.tensorflow.NodeDef.decode(r, r.uint32()));
                    break;
                case 4:
                    r.skip().pos++;
                    if (m.ret === $util.emptyObject)
                        m.ret = {};
                    k = r.string();
                    r.pos++;
                    m.ret[k] = r.string();
                    break;
                default:
                    r.skipType(t & 7);
                    break;
                }
            }
            return m;
        };

        return FunctionDef;
    })();

    tensorflow.GradientDef = (function() {

        function GradientDef(p) {
            if (p)
                for (var ks = Object.keys(p), i = 0; i < ks.length; ++i)
                    if (p[ks[i]] != null)
                        this[ks[i]] = p[ks[i]];
        }

        GradientDef.prototype.functionName = "";
        GradientDef.prototype.gradientFunc = "";

        GradientDef.decode = function decode(r, l) {
            if (!(r instanceof $Reader))
                r = $Reader.create(r);
            var c = l === undefined ? r.len : r.pos + l, m = new $root.tensorflow.GradientDef();
            while (r.pos < c) {
                var t = r.uint32();
                switch (t >>> 3) {
                case 1:
                    m.functionName = r.string();
                    break;
                case 2:
                    m.gradientFunc = r.string();
                    break;
                default:
                    r.skipType(t & 7);
                    break;
                }
            }
            return m;
        };

        return GradientDef;
    })();

    return tensorflow;
})();

export { $root as default };
