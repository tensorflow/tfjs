(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var cppn_1 = require("../nn-art/cppn");
var MAX_Z_SCALE = 400;
var DEFAULT_Z_SCALE = 1;
var NUM_NEURONS = 30;
var WEIGHTS_STDEV = 0.6;
var inferenceCanvas = document.querySelector('#inference');
var cppn = new cppn_1.CPPN(inferenceCanvas);
cppn.setActivationFunction('tanh');
cppn.setColorMode('rgb');
cppn.setNumLayers(1);
cppn.setZ1Scale(convertZScale(DEFAULT_Z_SCALE));
cppn.setZ2Scale(convertZScale(DEFAULT_Z_SCALE));
cppn.generateWeights(NUM_NEURONS, WEIGHTS_STDEV);
cppn.start();
var currentColorElement = document.querySelector('#colormode');
document.querySelector('#color-selector').addEventListener('click', function (event) {
    var colorMode = event.target.getAttribute('data-val');
    currentColorElement.value = colorMode;
    cppn.setColorMode(colorMode);
});
var currentActivationFnElement = document.querySelector('#activation-fn');
document.querySelector('#activation-selector').addEventListener('click', function (event) {
    var activationFn = event.target.getAttribute('data-val');
    currentActivationFnElement.value = activationFn;
    cppn.setActivationFunction(activationFn);
});
var layersSlider = document.querySelector('#layers-slider');
var layersCountElement = document.querySelector('#layers-count');
layersSlider.addEventListener('input', function (event) {
    var numLayers = event.target.value;
    layersCountElement.innerText = '' + numLayers;
    cppn.setNumLayers(numLayers);
});
layersCountElement.innerText = '1';
var z1Slider = document.querySelector('#z1-slider');
z1Slider.addEventListener('input', function (event) {
    var z1Scale = event.target.value;
    cppn.setZ1Scale(convertZScale(z1Scale));
});
var z2Slider = document.querySelector('#z2-slider');
z2Slider.addEventListener('input', function (event) {
    var z2Scale = event.target.value;
    cppn.setZ2Scale(convertZScale(z2Scale));
});
var randomizeButton = document.querySelector('#random');
randomizeButton.addEventListener('click', function () {
    cppn.generateWeights(NUM_NEURONS, WEIGHTS_STDEV);
});
function convertZScale(z) {
    return (103 - z);
}

},{"../nn-art/cppn":3}],2:[function(require,module,exports){
"use strict";
function __export(m) {
    for (var p in m) if (!exports.hasOwnProperty(p)) exports[p] = m[p];
}
Object.defineProperty(exports, "__esModule", { value: true });
__export(require("../src/index"));

},{"../src/index":11}],3:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var learnjs_1 = require("../learnjs");
var nn_art_util = require("./nn_art_util");
var MAX_LAYERS = 10;
var colorModeOutputDimensions = {
    'rgb': 3,
    'rgba': 4,
    'hsv': 3,
    'hsva': 4,
    'yuv': 3,
    'yuva': 4,
    'bw': 1
};
var activationFunctionMap = {
    'tanh': function (math, ndarray) { return math.tanh(ndarray); },
    'sin': function (math, ndarray) { return math.sin(ndarray); },
    'relu': function (math, ndarray) { return math.relu(ndarray); },
    'step': function (math, ndarray) { return math.step(ndarray); }
};
var NUM_IMAGE_SPACE_VARIABLES = 3;
var NUM_LATENT_VARIABLES = 2;
var CPPN = (function () {
    function CPPN(inferenceCanvas) {
        this.inferenceCanvas = inferenceCanvas;
        this.weights = [];
        this.z1Counter = 0;
        this.z2Counter = 0;
        this.colorModeNames = ['rgb', 'rgba', 'hsv', 'hsva', 'yuv', 'yuva', 'bw'];
        this.activationFunctionNames = ['tanh', 'sin', 'relu', 'step'];
        this.isInferring = false;
        this.gl = learnjs_1.gpgpu_util.createWebGLContext(this.inferenceCanvas);
        this.gpgpu = new learnjs_1.GPGPUContext(this.gl);
        this.math = new learnjs_1.NDArrayMathGPU(this.gpgpu);
        var maxTextureSize = learnjs_1.webgl_util.queryMaxTextureSize(this.gl);
        var canvasSize = Math.floor(Math.sqrt(maxTextureSize));
        this.inferenceCanvas.width = canvasSize;
        this.inferenceCanvas.height = canvasSize;
        this.renderShader = nn_art_util.getRenderShader(this.gpgpu, canvasSize);
        this.addLatentVariablesShader = nn_art_util.getAddLatentVariablesShader(this.gpgpu, NUM_IMAGE_SPACE_VARIABLES);
        this.inputAtlas = nn_art_util.createInputAtlas(canvasSize, NUM_IMAGE_SPACE_VARIABLES, NUM_LATENT_VARIABLES);
    }
    CPPN.prototype.generateWeights = function (neuronsPerLayer, weightsStdev) {
        for (var i = 0; i < this.weights.length; i++) {
            this.weights[i].dispose();
        }
        this.weights = [];
        this.weights.push(learnjs_1.Array2D.randTruncatedNormal([neuronsPerLayer, NUM_IMAGE_SPACE_VARIABLES + NUM_LATENT_VARIABLES], 0, weightsStdev));
        for (var i = 0; i < MAX_LAYERS; i++) {
            this.weights.push(learnjs_1.Array2D.randTruncatedNormal([neuronsPerLayer, neuronsPerLayer], 0, weightsStdev));
        }
        this.weights.push(learnjs_1.Array2D.randTruncatedNormal([4, neuronsPerLayer], 0, weightsStdev));
    };
    CPPN.prototype.setColorMode = function (colorMode) {
        this.selectedColorModeName = colorMode;
    };
    CPPN.prototype.setActivationFunction = function (activationFunction) {
        this.selectedActivationFunctionName = activationFunction;
    };
    CPPN.prototype.setNumLayers = function (numLayers) {
        this.numLayers = numLayers;
    };
    CPPN.prototype.setZ1Scale = function (z1Scale) {
        this.z1Scale = z1Scale;
    };
    CPPN.prototype.setZ2Scale = function (z2Scale) {
        this.z2Scale = z2Scale;
    };
    CPPN.prototype.start = function () {
        this.isInferring = true;
        this.runInferenceLoop();
    };
    CPPN.prototype.runInferenceLoop = function () {
        var _this = this;
        if (!this.isInferring) {
            return;
        }
        var colorModeIndex = this.colorModeNames.indexOf(this.selectedColorModeName);
        var outputDimensions = colorModeOutputDimensions[this.selectedColorModeName];
        this.z1Counter += 1 / this.z1Scale;
        this.z2Counter += 1 / this.z2Scale;
        var z1 = Math.sin(this.z1Counter);
        var z2 = Math.cos(this.z2Counter);
        var intermediateResults = [];
        var addLatentVariablesResultTex = this.math.getTextureManager().acquireTexture(this.inputAtlas.shape);
        nn_art_util.addLatentVariables(this.gpgpu, this.addLatentVariablesShader, this.inputAtlas.getTexture(), addLatentVariablesResultTex, this.inputAtlas.shape, z1, z2);
        var inputAtlasWithLatentVariables = learnjs_1.Array2D.make(this.inputAtlas.shape, {
            texture: addLatentVariablesResultTex,
            textureShapeRC: this.inputAtlas.shape
        });
        intermediateResults.push(inputAtlasWithLatentVariables);
        var lastOutput = inputAtlasWithLatentVariables;
        this.math.scope(function () {
            for (var i = 0; i < _this.numLayers; i++) {
                var matmulResult = _this.math.matMul(_this.weights[i], lastOutput);
                lastOutput = (i === _this.numLayers - 1) ?
                    _this.math.sigmoid(matmulResult) :
                    activationFunctionMap[_this.selectedActivationFunctionName](_this.math, matmulResult);
            }
            nn_art_util.render(_this.gpgpu, _this.renderShader, lastOutput.getTexture(), outputDimensions, colorModeIndex);
        });
        inputAtlasWithLatentVariables.dispose();
        requestAnimationFrame(function () { return _this.runInferenceLoop(); });
    };
    CPPN.prototype.stopInferenceLoop = function () {
        this.isInferring = false;
    };
    return CPPN;
}());
exports.CPPN = CPPN;

},{"../learnjs":2,"./nn_art_util":4}],4:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var learnjs_1 = require("../learnjs");
function createInputAtlas(imageSize, inputNumDimensions, numLatentVariables) {
    var coords = new Float32Array(imageSize * imageSize * (inputNumDimensions + numLatentVariables));
    var dst = 0;
    for (var d = 0; d < inputNumDimensions + numLatentVariables; d++) {
        for (var i = 0; i < imageSize * imageSize; i++) {
            var x = i % imageSize;
            var y = Math.floor(i / imageSize);
            var coord = imagePixelToNormalizedCoord(x, y, imageSize, imageSize, numLatentVariables);
            coords[dst++] = coord[d];
        }
    }
    return learnjs_1.Array2D.new([inputNumDimensions + numLatentVariables, imageSize * imageSize], coords);
}
exports.createInputAtlas = createInputAtlas;
function getAddLatentVariablesShader(gpgpu, inputNumDimensions) {
    var fragmentShaderSource = "\n    precision highp float;\n    uniform sampler2D source;\n    varying vec2 resultUV;\n\n    uniform vec2 z;\n\n    const vec2 halfCR = vec2(0.5, 0.5);\n\n    void main() {\n      vec2 outputCR = floor(gl_FragCoord.xy);\n      if (outputCR[1] == " + inputNumDimensions + ".0) {\n        gl_FragColor = vec4(z[0], 0, 0, 0);\n      } else if (outputCR[1] > " + inputNumDimensions + ".0) {\n        gl_FragColor = vec4(z[1], 0, 0, 0);\n      } else {\n        gl_FragColor = texture2D(source, resultUV);\n      }\n    }";
    return gpgpu.createProgram(fragmentShaderSource);
}
exports.getAddLatentVariablesShader = getAddLatentVariablesShader;
function addLatentVariables(gpgpu, addZShader, sourceTex, resultTex, shapeRowCol, z1, z2) {
    gpgpu.setOutputMatrixTexture(resultTex, shapeRowCol[0], shapeRowCol[1]);
    gpgpu.setProgram(addZShader);
    gpgpu.setInputMatrixTexture(sourceTex, 'source', 0);
    var zLoc = gpgpu.getUniformLocation('z');
    gpgpu.gl.uniform2f(zLoc, z1, z2);
    gpgpu.executeProgram();
}
exports.addLatentVariables = addLatentVariables;
function getRenderShader(gpgpu, imageSize) {
    var fragmentShaderSource = "\n    precision highp float;\n    uniform sampler2D source;\n    varying vec2 resultUV;\n\n    uniform int colorMode;\n    uniform float outputNumDimensions;\n\n    const float destinationSize = " + imageSize + ".0;\n\n    const mat3 yuv2rgb = mat3(\n          1,       1,     1,\n          0, -.34413, 1.772,\n      1.402, -.71414,     0);\n\n    vec3 hsv2rgb(vec3 c) {\n      vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);\n      vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);\n      return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);\n    }\n\n    void main() {\n      vec2 outputCR = floor(gl_FragCoord.xy);\n      float inputC = outputCR.y * destinationSize + outputCR.x;\n      float u = (inputC + 0.5) / " + imageSize * imageSize + ".0;\n\n      vec4 inputR = vec4(0.0, 1.0, 2.0, 3.0);\n      vec4 v = (inputR + 0.5) / outputNumDimensions;\n\n      vec4 values = vec4(\n        texture2D(source, vec2(u, v[0])).r,\n        texture2D(source, vec2(u, v[1])).r,\n        texture2D(source, vec2(u, v[2])).r,\n        texture2D(source, vec2(u, v[3])).r);\n\n      if (colorMode == 0) {\n        // RGB\n        gl_FragColor = vec4(values.rgb, 1.0);\n      } else if (colorMode == 1) {\n        // RGBA\n        gl_FragColor = values;\n      } else if (colorMode == 2) {\n        // HSV\n        vec3 rgb = hsv2rgb(values.rgb);\n        gl_FragColor = vec4(rgb, 1.0);\n      } else if (colorMode == 3) {\n        // HSVA\n        vec3 rgb = hsv2rgb(values.rgb);\n        gl_FragColor = vec4(rgb, values[3]);\n      } else if (colorMode == 4 || colorMode == 5) {\n        // YUV\n        values[0] = clamp(values[0], 0.2, 0.8);\n        values[1] = values[1] - 0.5;\n        values[2] = values[2] - 0.5;\n        vec3 rgb = yuv2rgb * values.rgb;\n        if (colorMode == 4) {\n          // YUV\n          gl_FragColor = vec4(rgb, 1.0);\n        } else if (colorMode == 5) {\n          // YUVA\n          gl_FragColor = vec4(rgb, values.a);\n        }\n      } else if (colorMode == 6) {\n        gl_FragColor = vec4(values[0], values[0], values[0], 1.0);\n      }\n    }";
    return gpgpu.createProgram(fragmentShaderSource);
}
exports.getRenderShader = getRenderShader;
function render(gpgpu, renderShader, sourceTex, outputNumDimensions, colorMode) {
    learnjs_1.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
    gpgpu.setProgram(renderShader);
    gpgpu.setInputMatrixTexture(sourceTex, 'source', 0);
    var colorModeLoc = gpgpu.getUniformLocation('colorMode');
    gpgpu.gl.uniform1i(colorModeLoc, colorMode);
    var outputNumDimensionsLoc = gpgpu.getUniformLocation('outputNumDimensions');
    gpgpu.gl.uniform1f(outputNumDimensionsLoc, outputNumDimensions);
    gpgpu.executeProgram();
}
exports.render = render;
function imagePixelToNormalizedCoord(x, y, imageWidth, imageHeight, zSize) {
    var halfWidth = imageWidth * 0.5;
    var halfHeight = imageHeight * 0.5;
    var normX = (x - halfWidth) / imageWidth;
    var normY = (y - halfHeight) / imageHeight;
    var r = Math.sqrt(normX * normX + normY * normY);
    var result = [normX, normY, r];
    for (var i = 0; i < zSize; i++) {
        result.push(0);
    }
    return result;
}
exports.imagePixelToNormalizedCoord = imagePixelToNormalizedCoord;

},{"../learnjs":2}],5:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("./math/ndarray");
var MANIFEST_FILE = 'manifest.json';
var CheckpointLoader = (function () {
    function CheckpointLoader(urlPath) {
        this.urlPath = urlPath;
        if (this.urlPath.charAt(this.urlPath.length - 1) !== '/') {
            this.urlPath += '/';
        }
    }
    CheckpointLoader.prototype.loadManifest = function () {
        var _this = this;
        return new Promise(function (resolve, reject) {
            var xhr = new XMLHttpRequest();
            xhr.open('GET', _this.urlPath + MANIFEST_FILE);
            xhr.onload = function () {
                _this.checkpointManifest = JSON.parse(xhr.responseText);
                resolve();
            };
            xhr.onerror = function (error) {
                throw new Error(MANIFEST_FILE + " not found at " + _this.urlPath + ". " + error);
            };
            xhr.send();
        });
    };
    CheckpointLoader.prototype.getCheckpointManifest = function () {
        var _this = this;
        if (this.checkpointManifest == null) {
            return new Promise(function (resolve, reject) {
                _this.loadManifest().then(function () {
                    resolve(_this.checkpointManifest);
                });
            });
        }
        return new Promise(function (resolve, reject) {
            resolve(_this.checkpointManifest);
        });
    };
    CheckpointLoader.prototype.getAllVariables = function () {
        var _this = this;
        if (this.variables != null) {
            return new Promise(function (resolve, reject) {
                resolve(_this.variables);
            });
        }
        return new Promise(function (resolve, reject) {
            _this.getCheckpointManifest().then(function (checkpointDefinition) {
                var variableNames = Object.keys(_this.checkpointManifest);
                var variablePromises = [];
                for (var i = 0; i < variableNames.length; i++) {
                    variablePromises.push(_this.getVariable(variableNames[i]));
                }
                Promise.all(variablePromises).then(function (variables) {
                    _this.variables = {};
                    for (var i = 0; i < variables.length; i++) {
                        _this.variables[variableNames[i]] = variables[i];
                    }
                    resolve(_this.variables);
                });
            });
        });
    };
    CheckpointLoader.prototype.getVariable = function (varName) {
        var _this = this;
        if (!(varName in this.checkpointManifest)) {
            throw new Error('Cannot load non-existant variable ' + varName);
        }
        var variableRequestPromiseMethod = function (resolve, reject) {
            var xhr = new XMLHttpRequest();
            xhr.responseType = 'arraybuffer';
            var fname = _this.checkpointManifest[varName].filename;
            xhr.open('GET', _this.urlPath + fname);
            xhr.onload = function () {
                var values = new Float32Array(xhr.response);
                var ndarray = ndarray_1.NDArray.make(_this.checkpointManifest[varName].shape, { values: values });
                resolve(ndarray);
            };
            xhr.onerror = function (error) {
                throw new Error('Could not fetch variable ' + varName + ': ' + error);
            };
            xhr.send();
        };
        if (this.checkpointManifest == null) {
            return new Promise(function (resolve, reject) {
                _this.loadManifest().then(function () {
                    new Promise(variableRequestPromiseMethod).then(resolve);
                });
            });
        }
        return new Promise(variableRequestPromiseMethod);
    };
    return CheckpointLoader;
}());
exports.CheckpointLoader = CheckpointLoader;

},{"./math/ndarray":22}],6:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("./math/ndarray");
var util = require("./util");
var STATS_SAMPLE_PERCENTAGE = 0.1;
var InMemoryDataset = (function () {
    function InMemoryDataset(dataShapes) {
        this.dataShapes = dataShapes;
        this.normalizationInfo = {};
    }
    InMemoryDataset.prototype.getDataShape = function (dataIndex) {
        return this.dataShapes[dataIndex];
    };
    InMemoryDataset.prototype.getData = function () {
        return this.dataset;
    };
    InMemoryDataset.prototype.getStats = function () {
        var _this = this;
        if (this.dataset == null) {
            throw new Error('Data is null.');
        }
        return this.dataset.map(function (d) { return _this.getStatsForData(d); });
    };
    InMemoryDataset.prototype.getStatsForData = function (data) {
        var inputMin = Number.POSITIVE_INFINITY;
        var inputMax = Number.NEGATIVE_INFINITY;
        var exampleIndices = data.map(function (example, i) { return i; });
        util.shuffle(exampleIndices);
        exampleIndices =
            exampleIndices.slice(exampleIndices.length * STATS_SAMPLE_PERCENTAGE);
        for (var i = 0; i < exampleIndices.length; i++) {
            var inputValues = data[exampleIndices[i]].getValues();
            for (var j = 0; j < inputValues.length; j++) {
                inputMin = Math.min(inputMin, inputValues[j]);
                inputMax = Math.max(inputMax, inputValues[j]);
            }
        }
        return {
            inputMin: inputMin,
            inputMax: inputMax,
            exampleCount: data.length,
            shape: data[0].shape,
        };
    };
    InMemoryDataset.prototype.normalizeExamplesToRange = function (examples, curLowerBounds, curUpperBounds, newLowerBounds, newUpperBounds) {
        var curBoundsIsPerDimension = (curUpperBounds instanceof Float32Array &&
            curLowerBounds instanceof Float32Array);
        var newBoundsIsPerDimension = (newLowerBounds instanceof Float32Array &&
            newUpperBounds instanceof Float32Array);
        var inputSize = util.sizeFromShape(examples[0].shape);
        var newExamples = [];
        examples.forEach(function (example) {
            var inputValues = example.getValues();
            var normalizedValues = new Float32Array(inputSize);
            for (var j = 0; j < inputSize; j++) {
                var curLowerBound = curBoundsIsPerDimension ?
                    curLowerBounds[j] :
                    curLowerBounds;
                var curUpperBound = curBoundsIsPerDimension ?
                    curUpperBounds[j] :
                    curUpperBounds;
                var curRange = curUpperBound - curLowerBound;
                var newLowerBound = newBoundsIsPerDimension ?
                    newLowerBounds[j] :
                    newLowerBounds;
                var newUpperBound = newBoundsIsPerDimension ?
                    newUpperBounds[j] :
                    newUpperBounds;
                var newRange = newUpperBound - newLowerBound;
                if (curRange === 0) {
                    normalizedValues[j] = newLowerBound;
                }
                else {
                    normalizedValues[j] = newLowerBound +
                        newRange * (inputValues[j] - curLowerBound) / curRange;
                }
            }
            newExamples.push(ndarray_1.NDArray.make(example.shape, { values: normalizedValues }));
        });
        return newExamples;
    };
    InMemoryDataset.prototype.computeBounds = function (dataIndex) {
        var _this = this;
        if (this.dataset == null) {
            throw new Error('Data is null.');
        }
        var size = util.sizeFromShape(this.dataset[dataIndex][0].shape);
        this.normalizationInfo[dataIndex] = {
            isNormalized: false,
            minValues: new Float32Array(size),
            maxValues: new Float32Array(size)
        };
        for (var i = 0; i < size; i++) {
            this.normalizationInfo[dataIndex].minValues[i] = Number.POSITIVE_INFINITY;
            this.normalizationInfo[dataIndex].maxValues[i] = Number.NEGATIVE_INFINITY;
        }
        this.dataset[dataIndex].forEach(function (example) {
            var inputValues = example.getValues();
            for (var k = 0; k < size; k++) {
                _this.normalizationInfo[dataIndex].minValues[k] = Math.min(_this.normalizationInfo[dataIndex].minValues[k], inputValues[k]);
                _this.normalizationInfo[dataIndex].maxValues[k] = Math.max(_this.normalizationInfo[dataIndex].maxValues[k], inputValues[k]);
            }
        });
    };
    InMemoryDataset.prototype.normalizeWithinBounds = function (dataIndex, lowerBound, upperBound) {
        if (this.dataset == null) {
            throw new Error('Data is null.');
        }
        if (dataIndex >= this.dataset.length) {
            throw new Error('dataIndex out of bounds.');
        }
        if (this.normalizationInfo[dataIndex] == null) {
            this.computeBounds(dataIndex);
        }
        var curLowerBounds;
        var curUpperBounds;
        if (this.normalizationInfo[dataIndex].isNormalized) {
            curLowerBounds = this.normalizationInfo[dataIndex].lowerBound;
            curUpperBounds = this.normalizationInfo[dataIndex].upperBound;
        }
        else {
            curLowerBounds = this.normalizationInfo[dataIndex].minValues;
            curUpperBounds = this.normalizationInfo[dataIndex].maxValues;
        }
        this.dataset[dataIndex] = this.normalizeExamplesToRange(this.dataset[dataIndex], curLowerBounds, curUpperBounds, lowerBound, upperBound);
        this.normalizationInfo[dataIndex].isNormalized = true;
        this.normalizationInfo[dataIndex].lowerBound = lowerBound;
        this.normalizationInfo[dataIndex].upperBound = upperBound;
    };
    InMemoryDataset.prototype.isNormalized = function (dataIndex) {
        return this.normalizationInfo != null &&
            this.normalizationInfo[dataIndex].isNormalized;
    };
    InMemoryDataset.prototype.removeNormalization = function (dataIndex) {
        if (this.dataset == null) {
            throw new Error('Training or test data is null.');
        }
        if (!this.isNormalized(dataIndex)) {
            return;
        }
        this.dataset[dataIndex] = this.normalizeExamplesToRange(this.dataset[dataIndex], this.normalizationInfo[dataIndex].lowerBound, this.normalizationInfo[dataIndex].upperBound, this.normalizationInfo[dataIndex].minValues, this.normalizationInfo[dataIndex].maxValues);
        this.normalizationInfo[dataIndex].isNormalized = false;
    };
    InMemoryDataset.prototype.unnormalizeExamples = function (examples, dataIndex) {
        if (!this.isNormalized(dataIndex)) {
            return examples;
        }
        return this.normalizeExamplesToRange(examples, this.normalizationInfo[dataIndex].lowerBound, this.normalizationInfo[dataIndex].upperBound, this.normalizationInfo[dataIndex].minValues, this.normalizationInfo[dataIndex].maxValues);
    };
    InMemoryDataset.prototype.dispose = function () {
        if (this.dataset == null) {
            return;
        }
        for (var i = 0; i < this.dataset.length; i++) {
            for (var j = 0; j < this.dataset[i].length; j++) {
                this.dataset[i][j].dispose();
            }
        }
        this.dataset = [];
    };
    return InMemoryDataset;
}());
exports.InMemoryDataset = InMemoryDataset;

},{"./math/ndarray":22,"./util":86}],7:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_layers_1 = require("./graph_layers");
var concat3d_util = require("./math/concat3d_util");
var conv_util = require("./math/conv_util");
var ndarray_1 = require("./math/ndarray");
var util = require("./util");
var Graph = (function () {
    function Graph() {
        this.nodes = [];
        this.layers = new graph_layers_1.GraphLayers(this);
    }
    Graph.prototype.variable = function (name, data) {
        return this.addNodeAndReturnOutput(new VariableNode(this, name, data));
    };
    Graph.prototype.placeholder = function (name, shape) {
        return this.addNodeAndReturnOutput(new PlaceholderNode(this, name, shape));
    };
    Graph.prototype.constant = function (value) {
        var finalValue;
        if (typeof value === 'number') {
            finalValue = ndarray_1.Scalar.new(value);
        }
        else if (value instanceof ndarray_1.NDArray) {
            finalValue = value;
        }
        else if (value instanceof Array) {
            var vals = new Float32Array(util.flatten(value));
            finalValue = ndarray_1.NDArray.make(util.inferShape(value), { values: vals });
        }
        else {
            throw new Error('unimplemented constant type.');
        }
        return this.addNodeAndReturnOutput(new ConstantNode(this, finalValue));
    };
    Graph.prototype.reshape = function (x, shape) {
        return this.addNodeAndReturnOutput(new ReshapeNode(this, 'Reshape', x, shape));
    };
    Graph.prototype.fusedLinearCombination = function (x1, x2, c1, c2) {
        return this.addNodeAndReturnOutput(new FusedLinearCombinationNode(this, x1, x2, c1, c2));
    };
    Graph.prototype.add = function (x1, x2) {
        return this.addNodeAndReturnOutput(new AddNode(this, x1, x2));
    };
    Graph.prototype.subtract = function (x1, x2) {
        return this.addNodeAndReturnOutput(new SubtractNode(this, x1, x2));
    };
    Graph.prototype.multiply = function (x1, x2) {
        return this.addNodeAndReturnOutput(new MultiplyNode(this, x1, x2));
    };
    Graph.prototype.divide = function (x1, x2) {
        return this.addNodeAndReturnOutput(new DivideNode(this, x1, x2));
    };
    Graph.prototype.reduceSum = function (x) {
        return this.addNodeAndReturnOutput(new ReduceSumNode(this, x));
    };
    Graph.prototype.concat3d = function (x1, x2, axis) {
        return this.addNodeAndReturnOutput(new Concat3DNode(this, x1, x2, axis));
    };
    Graph.prototype.matmul = function (x1, x2) {
        return this.addNodeAndReturnOutput(new MatMulNode(this, x1, x2));
    };
    Graph.prototype.conv2d = function (x, w, b, fieldSize, outputDepth, stride, zeroPad) {
        if (stride === void 0) { stride = 1; }
        return this.addNodeAndReturnOutput(new Convolution2DNode(this, x, w, b, fieldSize, outputDepth, stride, zeroPad));
    };
    Graph.prototype.maxPool = function (x, fieldSize, stride, zeroPad) {
        if (stride === void 0) { stride = 1; }
        return this.addNodeAndReturnOutput(new MaxPoolNode(this, x, fieldSize, stride, zeroPad));
    };
    Graph.prototype.exp = function (x) {
        return this.addNodeAndReturnOutput(new ExpNode(this, x));
    };
    Graph.prototype.log = function (x) {
        return this.addNodeAndReturnOutput(new LogNode(this, x));
    };
    Graph.prototype.relu = function (x) {
        return this.addNodeAndReturnOutput(new ReLUNode(this, x));
    };
    Graph.prototype.tanh = function (x) {
        return this.addNodeAndReturnOutput(new TanHNode(this, x));
    };
    Graph.prototype.sigmoid = function (x) {
        return this.addNodeAndReturnOutput(new SigmoidNode(this, x));
    };
    Graph.prototype.square = function (x) {
        return this.addNodeAndReturnOutput(new SquareNode(this, x));
    };
    Graph.prototype.softmax = function (x) {
        return this.addNodeAndReturnOutput(new SoftmaxNode(this, x));
    };
    Graph.prototype.softmaxCrossEntropyCost = function (x, target) {
        return this.addNodeAndReturnOutput(new SoftmaxCrossEntropyCostNode(this, x, target));
    };
    Graph.prototype.meanSquaredCost = function (label, prediction) {
        return this.addNodeAndReturnOutput(new MeanSquaredCostNode(this, label, prediction));
    };
    Graph.prototype.argmax = function (x) {
        return this.addNodeAndReturnOutput(new ArgMaxNode(this, x));
    };
    Graph.prototype.argmaxEquals = function (x1, x2) {
        return this.addNodeAndReturnOutput(new ArgMaxEqualsNode(this, x1, x2));
    };
    Graph.prototype.addNodeAndReturnOutput = function (node) {
        this.nodes.push(node);
        node.validate();
        return node.output;
    };
    Graph.prototype.getNodes = function () {
        return this.nodes;
    };
    return Graph;
}());
exports.Graph = Graph;
var Tensor = (function () {
    function Tensor(shape) {
        this.shape = shape;
        this.id = Tensor.nextID++;
    }
    return Tensor;
}());
Tensor.nextID = 0;
exports.Tensor = Tensor;
var Node = (function () {
    function Node(graph, name, inputs, output) {
        this.graph = graph;
        this.name = name;
        this.inputs = inputs;
        this.output = output;
        this.id = Node.nextID++;
        output.node = this;
    }
    return Node;
}());
Node.nextID = 0;
exports.Node = Node;
var VariableNode = (function (_super) {
    __extends(VariableNode, _super);
    function VariableNode(graph, name, data) {
        var _this = _super.call(this, graph, name, {}, new Tensor(data.shape)) || this;
        _this.data = data;
        return _this;
    }
    VariableNode.prototype.validate = function () {
        util.assert(this.data != null, 'Error adding variable op: Data for variable \'' + this.name +
            '\' is null or undefined');
    };
    return VariableNode;
}(Node));
exports.VariableNode = VariableNode;
var PlaceholderNode = (function (_super) {
    __extends(PlaceholderNode, _super);
    function PlaceholderNode(graph, name, shape) {
        return _super.call(this, graph, name, {}, new Tensor(shape)) || this;
    }
    PlaceholderNode.prototype.validate = function () { };
    return PlaceholderNode;
}(Node));
exports.PlaceholderNode = PlaceholderNode;
var ConstantNode = (function (_super) {
    __extends(ConstantNode, _super);
    function ConstantNode(graph, data) {
        var _this = _super.call(this, graph, 'Constant', {}, new Tensor(data.shape)) || this;
        _this.data = data;
        return _this;
    }
    ConstantNode.prototype.validate = function () {
        util.assert(this.data != null, 'Error adding constant: data for placeholder \'' + this.name +
            '\' is null or undefined');
    };
    return ConstantNode;
}(Node));
exports.ConstantNode = ConstantNode;
var ReshapeNode = (function (_super) {
    __extends(ReshapeNode, _super);
    function ReshapeNode(graph, name, x, shape) {
        var _this = _super.call(this, graph, name, { x: x }, new Tensor(shape)) || this;
        _this.name = name;
        _this.x = x;
        _this.shape = shape;
        return _this;
    }
    ReshapeNode.prototype.validate = function () {
        var xSize = util.sizeFromShape(this.x.shape);
        var shapeSize = util.sizeFromShape(this.shape);
        util.assert(xSize === shapeSize, 'Error making reshape operation: input Tensor to reshape \'' +
            this.name + '\' of shape (' + this.x.shape +
            ') does not match size of requested shape ' + this.shape + '.');
    };
    return ReshapeNode;
}(Node));
ReshapeNode.X = 'x';
exports.ReshapeNode = ReshapeNode;
var FusedLinearCombinationNode = (function (_super) {
    __extends(FusedLinearCombinationNode, _super);
    function FusedLinearCombinationNode(graph, t1, t2, c1, c2) {
        var _this = _super.call(this, graph, 'Linear Combination', { t1: t1, t2: t2, c1: c1, c2: c2 }, new Tensor(t1.shape)) || this;
        _this.t1 = t1;
        _this.t2 = t2;
        _this.c1 = c1;
        _this.c2 = c2;
        return _this;
    }
    FusedLinearCombinationNode.prototype.validate = function () {
        util.assertShapesMatch(this.t1.shape, this.t2.shape);
        if (!util.isScalarShape(this.c1.shape)) {
            throw new Error('Error adding fusedLinearCombination: c1 is not a scalar, got ' +
                'shape: ' + this.c1.shape);
        }
        if (!util.isScalarShape(this.c2.shape)) {
            throw new Error('Error adding fusedLinearCombination: c2 is not a scalar, got ' +
                'shape: ' + this.c2.shape);
        }
    };
    return FusedLinearCombinationNode;
}(Node));
FusedLinearCombinationNode.T1 = 't1';
FusedLinearCombinationNode.T2 = 't2';
FusedLinearCombinationNode.C1 = 'c1';
FusedLinearCombinationNode.C2 = 'c2';
exports.FusedLinearCombinationNode = FusedLinearCombinationNode;
var AddNode = (function (_super) {
    __extends(AddNode, _super);
    function AddNode(graph, t1, t2) {
        var _this = _super.call(this, graph, 'Add', { t1: t1, t2: t2 }, new Tensor(util.sizeFromShape(t1.shape) === 1 ? t2.shape : t1.shape)) || this;
        _this.t1 = t1;
        _this.t2 = t2;
        return _this;
    }
    AddNode.prototype.validate = function () {
        util.assert(util.sizeFromShape(this.t1.shape) === 1 ||
            util.sizeFromShape(this.t2.shape) === 1 ||
            util.arraysEqual(this.t1.shape, this.t2.shape), 'Error adding add operation op: one of inputs must be scalar or the ' +
            'shapes ' + this.t1.shape + ' and ' + this.t2.shape +
            ' must match.');
    };
    return AddNode;
}(Node));
AddNode.T1 = 't1';
AddNode.T2 = 't2';
exports.AddNode = AddNode;
var SubtractNode = (function (_super) {
    __extends(SubtractNode, _super);
    function SubtractNode(graph, t1, t2) {
        var _this = _super.call(this, graph, 'Subtract', { t1: t1, t2: t2 }, new Tensor(util.sizeFromShape(t1.shape) === 1 ? t2.shape : t1.shape)) || this;
        _this.t1 = t1;
        _this.t2 = t2;
        return _this;
    }
    SubtractNode.prototype.validate = function () {
        util.assert(util.sizeFromShape(this.t1.shape) === 1 ||
            util.sizeFromShape(this.t2.shape) === 1 ||
            util.arraysEqual(this.t1.shape, this.t2.shape), 'Error adding subtract op: one of inputs must be scalar or the ' +
            'shapes ' + this.t1.shape + ' and ' + this.t2.shape +
            ' must match.');
    };
    return SubtractNode;
}(Node));
SubtractNode.T1 = 't1';
SubtractNode.T2 = 't2';
exports.SubtractNode = SubtractNode;
var MultiplyNode = (function (_super) {
    __extends(MultiplyNode, _super);
    function MultiplyNode(graph, t1, t2) {
        var _this = _super.call(this, graph, 'Multiply', { t1: t1, t2: t2 }, new Tensor(util.sizeFromShape(t1.shape) === 1 ? t2.shape : t1.shape)) || this;
        _this.t1 = t1;
        _this.t2 = t2;
        return _this;
    }
    MultiplyNode.prototype.validate = function () {
        util.assert(util.sizeFromShape(this.t1.shape) === 1 ||
            util.sizeFromShape(this.t2.shape) === 1 ||
            util.arraysEqual(this.t1.shape, this.t2.shape), 'Error adding multiply op: one of inputs must be scalar or the ' +
            'shapes ' + this.t1.shape + ' and ' + this.t2.shape +
            ' must match.');
    };
    return MultiplyNode;
}(Node));
MultiplyNode.T1 = 't1';
MultiplyNode.T2 = 't2';
exports.MultiplyNode = MultiplyNode;
var DivideNode = (function (_super) {
    __extends(DivideNode, _super);
    function DivideNode(graph, t1, t2) {
        var _this = _super.call(this, graph, 'Divide', { t1: t1, t2: t2 }, new Tensor(util.sizeFromShape(t1.shape) === 1 ? t2.shape : t1.shape)) || this;
        _this.t1 = t1;
        _this.t2 = t2;
        return _this;
    }
    DivideNode.prototype.validate = function () {
        util.assert(util.sizeFromShape(this.t1.shape) === 1 ||
            util.sizeFromShape(this.t2.shape) === 1 ||
            util.arraysEqual(this.t1.shape, this.t2.shape), 'Error adding divide op: one of inputs must be scalar or the ' +
            'shapes ' + this.t1.shape + ' and ' + this.t2.shape +
            ' must match.');
    };
    return DivideNode;
}(Node));
DivideNode.T1 = 't1';
DivideNode.T2 = 't2';
exports.DivideNode = DivideNode;
var ReduceSumNode = (function (_super) {
    __extends(ReduceSumNode, _super);
    function ReduceSumNode(graph, x) {
        return _super.call(this, graph, 'ReduceSum', { x: x }, new Tensor([])) || this;
    }
    ReduceSumNode.prototype.validate = function () { };
    return ReduceSumNode;
}(Node));
ReduceSumNode.X = 'x';
exports.ReduceSumNode = ReduceSumNode;
var Concat3DNode = (function (_super) {
    __extends(Concat3DNode, _super);
    function Concat3DNode(graph, x1, x2, axis) {
        var _this = _super.call(this, graph, 'Concat3D', { x1: x1, x2: x2 }, new Tensor(concat3d_util.computeConcat3DOutputShape(x1.shape, x2.shape, axis))) || this;
        _this.x1 = x1;
        _this.x2 = x2;
        _this.axis = axis;
        return _this;
    }
    Concat3DNode.prototype.validate = function () {
        concat3d_util.assertConcat3DShapesMatch(this.x1.shape, this.x2.shape, this.axis);
    };
    return Concat3DNode;
}(Node));
Concat3DNode.X1 = 'x1';
Concat3DNode.X2 = 'x2';
Concat3DNode.AXIS = 'axis';
exports.Concat3DNode = Concat3DNode;
function getMatMulOutputShape(x1Shape, x2Shape) {
    if (x1Shape.length === 1 && x2Shape.length === 1) {
        return [1];
    }
    else if (x1Shape.length === 1 && x2Shape.length === 2) {
        return [x2Shape[1]];
    }
    else if (x1Shape.length === 2 && x2Shape.length === 1) {
        return [x1Shape[0]];
    }
    return [x1Shape[0], x2Shape[1]];
}
var MatMulNode = (function (_super) {
    __extends(MatMulNode, _super);
    function MatMulNode(graph, x1, x2) {
        var _this = _super.call(this, graph, 'MatMul', { x1: x1, x2: x2 }, new Tensor(getMatMulOutputShape(x1.shape, x2.shape))) || this;
        _this.x1 = x1;
        _this.x2 = x2;
        return _this;
    }
    MatMulNode.prototype.validate = function () {
        if (this.x1.shape.length === 2 && this.x2.shape.length === 2) {
            util.assert(this.x1.shape[1] === this.x2.shape[0], 'Error adding matmul op: inner shapes of matrices with shapes ' +
                this.x1.shape + ' and ' + this.x2.shape + ' must match.');
        }
        else if (this.x1.shape.length === 2 && this.x2.shape.length === 1) {
            util.assert(this.x1.shape[1] === this.x2.shape[0], 'Error adding matmul op: second dimension of matrix with shape ' +
                this.x1.shape + ' must match size of vector with shape ' +
                this.x2.shape + '.');
        }
        else if (this.x1.shape.length === 1 && this.x2.shape.length === 2) {
            util.assert(this.x1.shape[0] === this.x2.shape[0], 'Error adding matmul op: size of vector with shape ' + this.x1.shape +
                ' must match first dimension of matrix with ' +
                'shape ' + this.x2.shape + '.');
        }
        else {
            throw new Error('Error adding matmul op: inputs must be vectors or matrices.');
        }
    };
    return MatMulNode;
}(Node));
MatMulNode.X1 = 'x1';
MatMulNode.X2 = 'x2';
exports.MatMulNode = MatMulNode;
var Convolution2DNode = (function (_super) {
    __extends(Convolution2DNode, _super);
    function Convolution2DNode(graph, x, w, b, fieldSize, outputDepth, stride, zeroPad) {
        if (stride === void 0) { stride = 1; }
        var _this = _super.call(this, graph, 'Convolution 2D', { x: x, w: w, b: b }, new Tensor(conv_util.computeOutputShape3D(x.shape, fieldSize, outputDepth, stride, zeroPad))) || this;
        _this.x = x;
        _this.w = w;
        _this.b = b;
        _this.fieldSize = fieldSize;
        _this.outputDepth = outputDepth;
        _this.stride = stride;
        _this.zeroPad = zeroPad;
        return _this;
    }
    Convolution2DNode.prototype.validate = function () {
        util.assert(this.x.shape.length === 3, 'Error adding conv2d op: input must be of rank 3, but got shape: ' +
            this.x.shape + '.');
        util.assert(this.w.shape.length === 4, 'Error adding conv2d op: weights must be of rank 4, but got shape: ' +
            this.w.shape + '.');
        util.assert(this.b.shape.length === 1, 'Error adding conv2d op: biases must be of rank 1, but got shape: ' +
            this.b.shape + '.');
        util.assert(this.x.shape[2] === this.w.shape[2], 'Error adding conv2d op: depth of input (' + this.x.shape[2] +
            ') must match input depth for weights (' + this.w.shape[2] + ').');
    };
    return Convolution2DNode;
}(Node));
Convolution2DNode.X = 'x';
Convolution2DNode.W = 'w';
Convolution2DNode.B = 'b';
exports.Convolution2DNode = Convolution2DNode;
var MaxPoolNode = (function (_super) {
    __extends(MaxPoolNode, _super);
    function MaxPoolNode(graph, x, fieldSize, stride, zeroPad) {
        if (stride === void 0) { stride = 1; }
        var _this = _super.call(this, graph, 'Max pool', { x: x }, new Tensor(conv_util.computeOutputShape3D(x.shape, fieldSize, x.shape[2], stride, zeroPad))) || this;
        _this.x = x;
        _this.fieldSize = fieldSize;
        _this.stride = stride;
        _this.zeroPad = zeroPad;
        return _this;
    }
    MaxPoolNode.prototype.validate = function () {
        util.assert(this.x.shape.length === 3, 'Error adding maxPool op: input must be of rank 3, but got shape: ' +
            this.x.shape + '.');
    };
    return MaxPoolNode;
}(Node));
MaxPoolNode.X = 'x';
exports.MaxPoolNode = MaxPoolNode;
var ReLUNode = (function (_super) {
    __extends(ReLUNode, _super);
    function ReLUNode(graph, x) {
        return _super.call(this, graph, 'ReLU', { x: x }, new Tensor(x.shape)) || this;
    }
    ReLUNode.prototype.validate = function () { };
    return ReLUNode;
}(Node));
ReLUNode.X = 'x';
exports.ReLUNode = ReLUNode;
var ExpNode = (function (_super) {
    __extends(ExpNode, _super);
    function ExpNode(graph, x) {
        return _super.call(this, graph, 'Exp', { x: x }, new Tensor(x.shape)) || this;
    }
    ExpNode.prototype.validate = function () { };
    return ExpNode;
}(Node));
ExpNode.X = 'x';
exports.ExpNode = ExpNode;
var LogNode = (function (_super) {
    __extends(LogNode, _super);
    function LogNode(graph, x) {
        return _super.call(this, graph, 'Log', { x: x }, new Tensor(x.shape)) || this;
    }
    LogNode.prototype.validate = function () { };
    return LogNode;
}(Node));
LogNode.X = 'x';
exports.LogNode = LogNode;
var TanHNode = (function (_super) {
    __extends(TanHNode, _super);
    function TanHNode(graph, x) {
        return _super.call(this, graph, 'TanH', { x: x }, new Tensor(x.shape)) || this;
    }
    TanHNode.prototype.validate = function () { };
    return TanHNode;
}(Node));
TanHNode.X = 'x';
exports.TanHNode = TanHNode;
var SigmoidNode = (function (_super) {
    __extends(SigmoidNode, _super);
    function SigmoidNode(graph, x) {
        return _super.call(this, graph, 'Sigmoid', { x: x }, new Tensor(x.shape)) || this;
    }
    SigmoidNode.prototype.validate = function () { };
    return SigmoidNode;
}(Node));
SigmoidNode.X = 'x';
exports.SigmoidNode = SigmoidNode;
var SquareNode = (function (_super) {
    __extends(SquareNode, _super);
    function SquareNode(graph, x) {
        return _super.call(this, graph, 'Square', { x: x }, new Tensor(x.shape)) || this;
    }
    SquareNode.prototype.validate = function () { };
    return SquareNode;
}(Node));
SquareNode.X = 'x';
exports.SquareNode = SquareNode;
var SoftmaxCrossEntropyCostNode = (function (_super) {
    __extends(SoftmaxCrossEntropyCostNode, _super);
    function SoftmaxCrossEntropyCostNode(graph, x, target) {
        var _this = _super.call(this, graph, 'SoftmaxCrossEntropyCost', { x: x, target: target }, new Tensor([])) || this;
        _this.x = x;
        _this.target = target;
        return _this;
    }
    SoftmaxCrossEntropyCostNode.prototype.validate = function () {
        util.assert(util.arraysEqual(this.x.shape, this.target.shape), 'Error adding softmaxCrossEntropyCost op: x shape (' + this.x.shape +
            ') must match target shape (' + this.target.shape + ').');
    };
    return SoftmaxCrossEntropyCostNode;
}(Node));
SoftmaxCrossEntropyCostNode.X = 'x';
SoftmaxCrossEntropyCostNode.TARGET = 'target';
exports.SoftmaxCrossEntropyCostNode = SoftmaxCrossEntropyCostNode;
var SoftmaxNode = (function (_super) {
    __extends(SoftmaxNode, _super);
    function SoftmaxNode(graph, x) {
        var _this = _super.call(this, graph, 'Softmax', { x: x }, new Tensor(x.shape)) || this;
        _this.x = x;
        return _this;
    }
    SoftmaxNode.prototype.validate = function () {
        util.assert(this.x.shape.length === 1, 'The input to a softmax must be a 1-D tensor');
        util.assert(this.x.shape[0] >= 2, 'The input to a softmax must have at least 2 values');
    };
    return SoftmaxNode;
}(Node));
SoftmaxNode.X = 'x';
exports.SoftmaxNode = SoftmaxNode;
var MeanSquaredCostNode = (function (_super) {
    __extends(MeanSquaredCostNode, _super);
    function MeanSquaredCostNode(graph, label, prediction) {
        var _this = _super.call(this, graph, 'Mean Squared Cost', { label: label, prediction: prediction }, new Tensor([])) || this;
        _this.label = label;
        _this.prediction = prediction;
        return _this;
    }
    MeanSquaredCostNode.prototype.validate = function () {
        util.assert(util.arraysEqual(this.label.shape, this.prediction.shape), 'Error adding meanSquaredCost op: label shape (' + this.label.shape +
            ') must match prediction shape (' + this.prediction.shape + ').');
    };
    return MeanSquaredCostNode;
}(Node));
MeanSquaredCostNode.LABEL = 'label';
MeanSquaredCostNode.PREDICTION = 'prediction';
exports.MeanSquaredCostNode = MeanSquaredCostNode;
var ArgMaxNode = (function (_super) {
    __extends(ArgMaxNode, _super);
    function ArgMaxNode(graph, x) {
        var _this = _super.call(this, graph, 'ArgMax', { x: x }, new Tensor([1])) || this;
        _this.x = x;
        return _this;
    }
    ArgMaxNode.prototype.validate = function () {
        util.assert(util.sizeFromShape(this.x.shape) > 0, 'Error adding argmax op: input tensor must have at least one entry.');
    };
    return ArgMaxNode;
}(Node));
ArgMaxNode.X = 'x';
exports.ArgMaxNode = ArgMaxNode;
var ArgMaxEqualsNode = (function (_super) {
    __extends(ArgMaxEqualsNode, _super);
    function ArgMaxEqualsNode(graph, x1, x2) {
        var _this = _super.call(this, graph, 'ArgMaxEquals', { x1: x1, x2: x2 }, new Tensor([1])) || this;
        _this.x1 = x1;
        _this.x2 = x2;
        return _this;
    }
    ArgMaxEqualsNode.prototype.validate = function () {
        util.assert(util.arraysEqual(this.x1.shape, this.x2.shape), 'Error adding ArgMaxEquals op: x1 shape (' + this.x1.shape +
            ') must match x2 shape (' + this.x2.shape + ').');
    };
    return ArgMaxEqualsNode;
}(Node));
ArgMaxEqualsNode.X1 = 'x1';
ArgMaxEqualsNode.X2 = 'x2';
exports.ArgMaxEqualsNode = ArgMaxEqualsNode;
var SplitNode = (function (_super) {
    __extends(SplitNode, _super);
    function SplitNode(graph, x) {
        var _this = _super.call(this, graph, 'SplitNode', { x: x }, new Tensor(x.shape)) || this;
        _this.outputs = [];
        return _this;
    }
    SplitNode.prototype.getNewOutputTensor = function () {
        var output = new Tensor(this.inputs[SplitNode.X].shape);
        output.node = this;
        this.outputs.push(output);
        return output;
    };
    SplitNode.prototype.validate = function () { };
    return SplitNode;
}(Node));
SplitNode.X = 'x';
exports.SplitNode = SplitNode;

},{"./graph_layers":8,"./math/concat3d_util":15,"./math/conv_util":16,"./math/ndarray":22,"./util":86}],8:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var initializers_1 = require("./initializers");
var GraphLayers = (function () {
    function GraphLayers(g) {
        this.g = g;
    }
    GraphLayers.prototype.dense = function (name, x, units, activation, useBias, kernelInitializer, biasInitializer) {
        if (activation === void 0) { activation = null; }
        if (useBias === void 0) { useBias = true; }
        if (kernelInitializer === void 0) { kernelInitializer = new initializers_1.VarianceScalingInitializer(); }
        if (biasInitializer === void 0) { biasInitializer = new initializers_1.ZerosInitializer(); }
        var weights = this.g.variable(name + '-weights', kernelInitializer.initialize([x.shape[0], units], x.shape[0], units));
        var out = this.g.matmul(x, weights);
        if (useBias) {
            var bias = this.g.variable(name + '-bias', biasInitializer.initialize([units], x.shape[0], units));
            out = this.g.add(out, bias);
        }
        if (activation != null) {
            out = activation(out);
        }
        return out;
    };
    return GraphLayers;
}());
exports.GraphLayers = GraphLayers;

},{"./initializers":12}],9:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("./math/ndarray");
var session_1 = require("./session");
var DEFAULT_EVAL_INTERVAL_MS = 1500;
var DEFAULT_COST_INTERVAL_MS = 500;
var DEFAULT_INFERENCE_EXAMPLE_INTERVAL_MS = 3000;
var MetricReduction;
(function (MetricReduction) {
    MetricReduction[MetricReduction["SUM"] = 0] = "SUM";
    MetricReduction[MetricReduction["MEAN"] = 1] = "MEAN";
})(MetricReduction = exports.MetricReduction || (exports.MetricReduction = {}));
var GraphRunner = (function () {
    function GraphRunner(math, session, eventObserver) {
        this.math = math;
        this.session = session;
        this.eventObserver = eventObserver;
        this.lastCostTimestamp = 0;
        this.lastEvalTimestamp = 0;
        this.totalIdleTimeMs = 0;
        this.resetStatistics();
        this.zeroScalar = ndarray_1.Scalar.new(0);
    }
    GraphRunner.prototype.resetStatistics = function () {
        this.totalBatchesTrained = 0;
        this.totalIdleTimeMs = 0;
        this.lastStopTimestamp = null;
    };
    GraphRunner.prototype.train = function (costTensor, trainFeedEntries, batchSize, optimizer, numBatches, metricTensor, metricFeedEntries, metricBatchSize, metricReduction, evalIntervalMs, costIntervalMs) {
        if (metricReduction === void 0) { metricReduction = MetricReduction.MEAN; }
        if (evalIntervalMs === void 0) { evalIntervalMs = DEFAULT_EVAL_INTERVAL_MS; }
        if (costIntervalMs === void 0) { costIntervalMs = DEFAULT_COST_INTERVAL_MS; }
        this.costTensor = costTensor;
        this.trainFeedEntries = trainFeedEntries;
        this.metricTensor = metricTensor;
        this.metricFeedEntries = metricFeedEntries;
        if (metricBatchSize != null && this.metricBatchSize !== metricBatchSize) {
            if (this.metricBatchSizeScalar != null) {
                this.metricBatchSizeScalar.dispose();
            }
            this.metricBatchSizeScalar = ndarray_1.Scalar.new(metricBatchSize);
        }
        this.metricBatchSize = metricBatchSize;
        this.metricReduction = metricReduction;
        this.batchSize = batchSize;
        this.optimizer = optimizer;
        this.metricIntervalMs = evalIntervalMs;
        this.costIntervalMs = costIntervalMs;
        this.currentTrainLoopNumBatches = numBatches;
        this.batchesTrainedThisRun = 0;
        this.isTraining = true;
        this.trainStartTimestamp = performance.now();
        this.trainNetwork();
    };
    GraphRunner.prototype.stopTraining = function () {
        this.isTraining = false;
        this.lastStopTimestamp = performance.now();
    };
    GraphRunner.prototype.resumeTraining = function () {
        this.isTraining = true;
        if (this.lastStopTimestamp != null) {
            this.totalIdleTimeMs += performance.now() - this.lastStopTimestamp;
        }
        this.trainNetwork();
    };
    GraphRunner.prototype.trainNetwork = function () {
        var _this = this;
        if (this.batchesTrainedThisRun === this.currentTrainLoopNumBatches) {
            this.stopTraining();
        }
        if (!this.isTraining) {
            if (this.eventObserver.doneTrainingCallback != null) {
                this.eventObserver.doneTrainingCallback();
            }
            return;
        }
        var start = performance.now();
        var shouldComputeCost = this.eventObserver.avgCostCallback != null &&
            (start - this.lastCostTimestamp > this.costIntervalMs);
        if (shouldComputeCost) {
            this.lastCostTimestamp = start;
        }
        var costReduction = shouldComputeCost ? session_1.CostReduction.MEAN : session_1.CostReduction.NONE;
        this.math.scope(function (keep) {
            var avgCost = _this.session.train(_this.costTensor, _this.trainFeedEntries, _this.batchSize, _this.optimizer, costReduction);
            if (shouldComputeCost) {
                var trainTime = performance.now() - start;
                _this.eventObserver.avgCostCallback(avgCost);
                if (_this.eventObserver.trainExamplesPerSecCallback != null) {
                    var examplesPerSec = (_this.batchSize * 1000 / trainTime);
                    _this.eventObserver.trainExamplesPerSecCallback(examplesPerSec);
                }
            }
            if (_this.eventObserver.metricCallback != null &&
                _this.metricFeedEntries != null &&
                start - _this.lastEvalTimestamp > _this.metricIntervalMs) {
                _this.lastEvalTimestamp = start;
                if (_this.lastComputedMetric != null) {
                    _this.lastComputedMetric.dispose();
                }
                _this.lastComputedMetric = _this.computeMetric();
                _this.eventObserver.metricCallback(_this.lastComputedMetric);
            }
            if (_this.eventObserver.totalTimeCallback != null) {
                _this.eventObserver.totalTimeCallback((start - _this.trainStartTimestamp) / 1000);
            }
            _this.batchesTrainedThisRun++;
            _this.totalBatchesTrained++;
            if (_this.eventObserver.batchesTrainedCallback != null) {
                _this.eventObserver.batchesTrainedCallback(_this.totalBatchesTrained);
            }
        });
        setTimeout(function () { return _this.trainNetwork(); });
    };
    GraphRunner.prototype.infer = function (inferenceTensor, inferenceFeedEntries, inferenceExampleIntervalMs, inferenceExampleCount, numPasses) {
        var _this = this;
        if (inferenceExampleIntervalMs === void 0) { inferenceExampleIntervalMs = DEFAULT_INFERENCE_EXAMPLE_INTERVAL_MS; }
        if (inferenceExampleCount === void 0) { inferenceExampleCount = 5; }
        if (this.eventObserver.inferenceExamplesCallback == null &&
            this.eventObserver.inferenceExamplesPerSecCallback == null) {
            throw new Error('Cannot start inference loop, no inference example or ' +
                'examples/sec observer provided.');
        }
        for (var i = 0; i < inferenceFeedEntries.length; i++) {
            var feedEntry = inferenceFeedEntries[i];
            if (feedEntry.data instanceof ndarray_1.NDArray) {
                throw new Error('Cannot start inference on the model runner with feed entries of ' +
                    'type NDArray. Please use InputProviders.');
            }
        }
        this.inferenceExampleIntervalMs = inferenceExampleIntervalMs;
        this.inferenceTensor = inferenceTensor;
        this.inferenceFeedEntries = inferenceFeedEntries;
        this.inferenceExampleCount = inferenceExampleCount;
        this.currentInferenceLoopNumPasses = numPasses;
        if (!this.isInferring) {
            this.inferencePassesThisRun = 0;
            setTimeout(function () { return _this.inferNetwork(); });
        }
        this.isInferring = true;
    };
    GraphRunner.prototype.inferNetwork = function () {
        var _this = this;
        if (!this.isInferring ||
            this.inferencePassesThisRun === this.currentInferenceLoopNumPasses) {
            return;
        }
        this.math.scope(function (keep, track) {
            var feeds = [];
            var inferenceValues = [];
            var start = performance.now();
            for (var i = 0; i < _this.inferenceExampleCount; i++) {
                var ndarrayFeedEntries = [];
                for (var j = 0; j < _this.inferenceFeedEntries.length; j++) {
                    var feedEntry = _this.inferenceFeedEntries[j];
                    ndarrayFeedEntries.push({
                        tensor: feedEntry.tensor,
                        data: track(feedEntry.data.getNextCopy(_this.math))
                    });
                }
                feeds.push(ndarrayFeedEntries);
                inferenceValues.push(_this.session.eval(_this.inferenceTensor, ndarrayFeedEntries));
            }
            if (_this.eventObserver.inferenceExamplesPerSecCallback != null) {
                inferenceValues[inferenceValues.length - 1].getValues();
                var inferenceExamplesPerSecTime = performance.now() - start;
                var examplesPerSec = (_this.inferenceExampleCount * 1000 / inferenceExamplesPerSecTime);
                _this.eventObserver.inferenceExamplesPerSecCallback(examplesPerSec);
            }
            if (_this.eventObserver.inferenceExamplesCallback != null) {
                _this.eventObserver.inferenceExamplesCallback(feeds, inferenceValues);
            }
            _this.inferencePassesThisRun++;
        });
        setTimeout(function () { return _this.inferNetwork(); }, this.inferenceExampleIntervalMs);
    };
    GraphRunner.prototype.stopInferring = function () {
        this.isInferring = false;
    };
    GraphRunner.prototype.isInferenceRunning = function () {
        return this.isInferring;
    };
    GraphRunner.prototype.computeMetric = function () {
        var _this = this;
        if (this.metricFeedEntries == null) {
            throw new Error('Cannot compute metric, no metric FeedEntries provided.');
        }
        var metric = this.zeroScalar;
        return this.math.scope(function (keep) {
            for (var i = 0; i < _this.metricBatchSize; i++) {
                var metricValue = _this.session.eval(_this.metricTensor, _this.metricFeedEntries);
                metric = _this.math.add(metric, metricValue);
            }
            if (_this.metricReduction === MetricReduction.MEAN) {
                metric = _this.math.divide(metric, _this.metricBatchSizeScalar);
            }
            return metric;
        });
    };
    GraphRunner.prototype.getTotalBatchesTrained = function () {
        return this.totalBatchesTrained;
    };
    GraphRunner.prototype.getLastComputedMetric = function () {
        return this.lastComputedMetric;
    };
    GraphRunner.prototype.setMath = function (math) {
        this.math = math;
    };
    GraphRunner.prototype.setSession = function (session) {
        this.session = session;
    };
    GraphRunner.prototype.setInferenceTensor = function (inferenceTensor) {
        this.inferenceTensor = inferenceTensor;
    };
    GraphRunner.prototype.setInferenceExampleCount = function (inferenceExampleCount) {
        this.inferenceExampleCount = inferenceExampleCount;
    };
    return GraphRunner;
}());
exports.GraphRunner = GraphRunner;

},{"./math/ndarray":22,"./session":82}],10:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var graph_1 = require("./graph");
var priority_queue = require("./priority_queue");
var priority_queue_1 = require("./priority_queue");
function getUnorderedEvaluationSet(nodes, terminatingNodes) {
    var terminatingNodeMap = {};
    var seen = {};
    var set = [];
    var visit = nodes.slice();
    terminatingNodes.forEach(function (node) { return terminatingNodeMap[node.id] = node; });
    var _loop_1 = function () {
        var cur = visit.pop();
        if (seen[cur.id] == null) {
            if (terminatingNodeMap[cur.id] == null) {
                Object.keys(cur.inputs)
                    .map(function (inputName) { return cur.inputs[inputName]; })
                    .forEach(function (input) { return visit.push(input.node); });
            }
            set.push(cur);
            seen[cur.id] = cur;
        }
    };
    while (visit.length !== 0) {
        _loop_1();
    }
    return set;
}
exports.getUnorderedEvaluationSet = getUnorderedEvaluationSet;
function getOrderedEvaluationSet(unorderedEvaluationSet) {
    var set = [];
    var nodeIndices = {};
    var pendingDependencies = {};
    var nodeQueue = new priority_queue_1.PriorityQueue(function (a, b) { return priority_queue.defaultCompare(pendingDependencies[a.id], pendingDependencies[b.id]); }, function (node, newIndex) { return nodeIndices[node.id] = newIndex; });
    unorderedEvaluationSet.forEach(function (node) { return pendingDependencies[node.id] = 0; });
    unorderedEvaluationSet.forEach(function (node) { return Object.keys(node.inputs)
        .map(function (key) { return node.inputs[key]; })
        .forEach(function (input) {
        if (unorderedEvaluationSet.indexOf(input.node) !== -1) {
            pendingDependencies[input.node.id]++;
        }
    }); });
    unorderedEvaluationSet.forEach(function (node) { return nodeQueue.enqueue(node); });
    while (!nodeQueue.empty()) {
        set.unshift(nodeQueue.dequeue());
        Object.keys(set[0].inputs).map(function (key) { return set[0].inputs[key]; }).forEach(function (input) {
            if (unorderedEvaluationSet.indexOf(input.node) === -1) {
                return;
            }
            pendingDependencies[input.node.id]--;
            nodeQueue.update(input.node, nodeIndices[input.node.id]);
        });
    }
    return set;
}
exports.getOrderedEvaluationSet = getOrderedEvaluationSet;
function isInputNode(node) {
    return Object.keys(node.inputs).length === 0;
}
exports.isInputNode = isInputNode;
function shouldBackProp(t) {
    return !(t.node instanceof graph_1.ConstantNode);
}
exports.shouldBackProp = shouldBackProp;
function isPassthroughNode(node, map) {
    var keys = Object.keys(node.inputs);
    for (var i = 0; i < keys.length; i++) {
        var input = node.inputs[keys[i]];
        if (map.get(input, true) === map.get(node.output, true)) {
            return true;
        }
    }
    return false;
}
exports.isPassthroughNode = isPassthroughNode;

},{"./graph":7,"./priority_queue":81}],11:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("./math/conv_util");
exports.conv_util = conv_util;
var gpgpu_util = require("./math/webgl/gpgpu_util");
exports.gpgpu_util = gpgpu_util;
var render_ndarray_gpu_util = require("./math/webgl/render_ndarray_gpu_util");
exports.render_ndarray_gpu_util = render_ndarray_gpu_util;
var webgl_util = require("./math/webgl/webgl_util");
exports.webgl_util = webgl_util;
var util = require("./util");
exports.util = util;
var checkpoint_loader_1 = require("./checkpoint_loader");
exports.CheckpointLoader = checkpoint_loader_1.CheckpointLoader;
var dataset_1 = require("./dataset");
exports.InMemoryDataset = dataset_1.InMemoryDataset;
var graph_1 = require("./graph");
exports.Graph = graph_1.Graph;
exports.Tensor = graph_1.Tensor;
var graph_runner_1 = require("./graph_runner");
exports.GraphRunner = graph_runner_1.GraphRunner;
exports.MetricReduction = graph_runner_1.MetricReduction;
var initializers_1 = require("./initializers");
exports.ConstantInitializer = initializers_1.ConstantInitializer;
exports.NDArrayInitializer = initializers_1.NDArrayInitializer;
exports.OnesInitializer = initializers_1.OnesInitializer;
exports.RandomNormalInitializer = initializers_1.RandomNormalInitializer;
exports.RandomTruncatedNormalInitializer = initializers_1.RandomTruncatedNormalInitializer;
exports.RandomUniformInitializer = initializers_1.RandomUniformInitializer;
exports.VarianceScalingInitializer = initializers_1.VarianceScalingInitializer;
exports.ZerosInitializer = initializers_1.ZerosInitializer;
var input_provider_1 = require("./input_provider");
exports.InCPUMemoryShuffledInputProviderBuilder = input_provider_1.InCPUMemoryShuffledInputProviderBuilder;
exports.InGPUMemoryShuffledInputProviderBuilder = input_provider_1.InGPUMemoryShuffledInputProviderBuilder;
var math_1 = require("./math/math");
exports.MatrixOrientation = math_1.MatrixOrientation;
exports.NDArrayMath = math_1.NDArrayMath;
var math_cpu_1 = require("./math/math_cpu");
exports.NDArrayMathCPU = math_cpu_1.NDArrayMathCPU;
var math_gpu_1 = require("./math/math_gpu");
exports.NDArrayMathGPU = math_gpu_1.NDArrayMathGPU;
var ndarray_1 = require("./math/ndarray");
exports.Array1D = ndarray_1.Array1D;
exports.Array2D = ndarray_1.Array2D;
exports.Array3D = ndarray_1.Array3D;
exports.Array4D = ndarray_1.Array4D;
exports.NDArray = ndarray_1.NDArray;
exports.Scalar = ndarray_1.Scalar;
var gpgpu_context_1 = require("./math/webgl/gpgpu_context");
exports.GPGPUContext = gpgpu_context_1.GPGPUContext;
var optimizer_1 = require("./optimizer");
exports.Optimizer = optimizer_1.Optimizer;
var session_1 = require("./session");
exports.CostReduction = session_1.CostReduction;
exports.Session = session_1.Session;
var sgd_optimizer_1 = require("./sgd_optimizer");
exports.SGDOptimizer = sgd_optimizer_1.SGDOptimizer;

},{"./checkpoint_loader":5,"./dataset":6,"./graph":7,"./graph_runner":9,"./initializers":12,"./input_provider":13,"./math/conv_util":16,"./math/math":19,"./math/math_cpu":20,"./math/math_gpu":21,"./math/ndarray":22,"./math/webgl/gpgpu_context":35,"./math/webgl/gpgpu_util":36,"./math/webgl/render_ndarray_gpu_util":48,"./math/webgl/webgl_util":58,"./optimizer":80,"./session":82,"./sgd_optimizer":84,"./util":86}],12:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("./math/ndarray");
var VarianceScalingInitializer = (function () {
    function VarianceScalingInitializer(scale, mode, distribution) {
        if (scale === void 0) { scale = 1.0; }
        if (mode === void 0) { mode = 'fan_in'; }
        if (distribution === void 0) { distribution = 'normal'; }
        this.scale = scale;
        this.mode = mode;
        this.distribution = distribution;
    }
    VarianceScalingInitializer.prototype.initialize = function (weightsShape, inputUnits, outputUnits) {
        var n = 0;
        if (this.mode === 'fan_in') {
            n = inputUnits;
        }
        else if (this.mode === 'fan_out') {
            n = outputUnits;
        }
        else if (this.mode === 'fan_avg') {
            n = (inputUnits + outputUnits) / 2;
        }
        else {
            throw new Error('Unexpected mode for variance scaling initializer: ' + this.mode);
        }
        if (this.distribution === 'normal') {
            return ndarray_1.NDArray.randTruncatedNormal(weightsShape, 0.0, Math.sqrt(this.scale / n));
        }
        else if (this.distribution === 'uniform') {
            return ndarray_1.NDArray.randUniform(weightsShape, 0.0, Math.sqrt(3 * this.scale / n));
        }
        else {
            throw new Error('Unexpected distribution for variance scaling initializer: ' +
                this.distribution);
        }
    };
    return VarianceScalingInitializer;
}());
exports.VarianceScalingInitializer = VarianceScalingInitializer;
var ZerosInitializer = (function () {
    function ZerosInitializer() {
    }
    ZerosInitializer.prototype.initialize = function (weightsShape, inputUnits, outputUnits) {
        return ndarray_1.NDArray.zeros(weightsShape);
    };
    return ZerosInitializer;
}());
exports.ZerosInitializer = ZerosInitializer;
var OnesInitializer = (function () {
    function OnesInitializer() {
    }
    OnesInitializer.prototype.initialize = function (weightsShape, inputUnits, outputUnits) {
        var values = ndarray_1.NDArray.zeros(weightsShape);
        values.fill(1);
        return values;
    };
    return OnesInitializer;
}());
exports.OnesInitializer = OnesInitializer;
var ConstantInitializer = (function () {
    function ConstantInitializer(value) {
        if (value === void 0) { value = 0; }
        this.value = value;
    }
    ConstantInitializer.prototype.initialize = function (weightsShape, inputUnits, outputUnits) {
        var values = ndarray_1.NDArray.zeros(weightsShape);
        values.fill(this.value);
        return values;
    };
    return ConstantInitializer;
}());
exports.ConstantInitializer = ConstantInitializer;
var NDArrayInitializer = (function () {
    function NDArrayInitializer(ndarray) {
        this.ndarray = ndarray;
    }
    NDArrayInitializer.prototype.initialize = function (weightsShape, inputUnits, outputUnits) {
        return this.ndarray;
    };
    return NDArrayInitializer;
}());
exports.NDArrayInitializer = NDArrayInitializer;
var RandomNormalInitializer = (function () {
    function RandomNormalInitializer(mean, stdev) {
        if (mean === void 0) { mean = 0; }
        if (stdev === void 0) { stdev = .05; }
        this.mean = mean;
        this.stdev = stdev;
    }
    RandomNormalInitializer.prototype.initialize = function (weightsShape, inputUnits, outputUnits) {
        return ndarray_1.NDArray.randNormal(weightsShape, this.mean, this.stdev);
    };
    return RandomNormalInitializer;
}());
exports.RandomNormalInitializer = RandomNormalInitializer;
var RandomTruncatedNormalInitializer = (function () {
    function RandomTruncatedNormalInitializer(mean, stdev) {
        if (mean === void 0) { mean = 0; }
        if (stdev === void 0) { stdev = .05; }
        this.mean = mean;
        this.stdev = stdev;
    }
    RandomTruncatedNormalInitializer.prototype.initialize = function (weightsShape, inputUnits, outputUnits) {
        return ndarray_1.NDArray.randTruncatedNormal(weightsShape, this.mean, this.stdev);
    };
    return RandomTruncatedNormalInitializer;
}());
exports.RandomTruncatedNormalInitializer = RandomTruncatedNormalInitializer;
var RandomUniformInitializer = (function () {
    function RandomUniformInitializer(minval, maxval) {
        if (minval === void 0) { minval = -.05; }
        if (maxval === void 0) { maxval = .05; }
        this.minval = minval;
        this.maxval = maxval;
    }
    RandomUniformInitializer.prototype.initialize = function (weightsShape, inputUnits, outputUnits) {
        return ndarray_1.NDArray.randUniform(weightsShape, this.minval, this.maxval);
    };
    return RandomUniformInitializer;
}());
exports.RandomUniformInitializer = RandomUniformInitializer;

},{"./math/ndarray":22}],13:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("./math/ndarray");
var util = require("./util");
var InMemoryShuffledInputProviderBuilder = (function () {
    function InMemoryShuffledInputProviderBuilder(inputs) {
        this.inputs = inputs;
        this.idx = 0;
        this.inputCounter = 0;
        this.epoch = 0;
        this.shuffledIndices = util.createShuffledIndices(inputs[0].length);
        this.numInputs = inputs.length;
        var numExamples = this.inputs[0].length;
        for (var i = 0; i < this.numInputs; i++) {
            util.assert(this.inputs[i].length === numExamples, 'Number of examples must match across different inputs.');
        }
        for (var i = 0; i < this.numInputs; i++) {
            var inputShape = this.inputs[i][0].shape;
            for (var j = 0; j < this.inputs[i].length; j++) {
                util.assertShapesMatch(inputShape, this.inputs[i][j].shape);
            }
        }
    }
    InMemoryShuffledInputProviderBuilder.prototype.getCurrentExampleIndex = function () {
        var returnIdx = this.idx;
        this.inputCounter++;
        if (this.inputCounter >= this.numInputs) {
            this.idx++;
            this.inputCounter = 0;
            if (this.idx >= this.inputs[0].length) {
                this.idx = 0;
                this.epoch++;
            }
        }
        return returnIdx;
    };
    InMemoryShuffledInputProviderBuilder.prototype.getNextInput = function (inputId) {
        var currentExampleIndex = this.getCurrentExampleIndex();
        return this.inputs[inputId][this.shuffledIndices[currentExampleIndex]];
    };
    InMemoryShuffledInputProviderBuilder.prototype.getEpoch = function () {
        return this.epoch;
    };
    InMemoryShuffledInputProviderBuilder.prototype.getInputProviders = function () {
        var inputProviders = [];
        for (var i = 0; i < this.numInputs; i++) {
            inputProviders.push(this.getInputProvider(i));
        }
        return inputProviders;
    };
    return InMemoryShuffledInputProviderBuilder;
}());
exports.InMemoryShuffledInputProviderBuilder = InMemoryShuffledInputProviderBuilder;
var InCPUMemoryShuffledInputProviderBuilder = (function (_super) {
    __extends(InCPUMemoryShuffledInputProviderBuilder, _super);
    function InCPUMemoryShuffledInputProviderBuilder() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    InCPUMemoryShuffledInputProviderBuilder.prototype.getInputProvider = function (inputId) {
        var shuffledInputProvider = this;
        return {
            getNextCopy: function (math) {
                return ndarray_1.NDArray.like(shuffledInputProvider.getNextInput(inputId));
            },
            disposeCopy: function (math, copy) {
                copy.dispose();
            }
        };
    };
    return InCPUMemoryShuffledInputProviderBuilder;
}(InMemoryShuffledInputProviderBuilder));
exports.InCPUMemoryShuffledInputProviderBuilder = InCPUMemoryShuffledInputProviderBuilder;
var InGPUMemoryShuffledInputProviderBuilder = (function (_super) {
    __extends(InGPUMemoryShuffledInputProviderBuilder, _super);
    function InGPUMemoryShuffledInputProviderBuilder() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    InGPUMemoryShuffledInputProviderBuilder.prototype.getInputProvider = function (inputId) {
        var shuffledInputProvider = this;
        return {
            getNextCopy: function (math) {
                return math.clone(shuffledInputProvider.getNextInput(inputId));
            },
            disposeCopy: function (math, copy) {
                copy.dispose();
            }
        };
    };
    return InGPUMemoryShuffledInputProviderBuilder;
}(InMemoryShuffledInputProviderBuilder));
exports.InGPUMemoryShuffledInputProviderBuilder = InGPUMemoryShuffledInputProviderBuilder;

},{"./math/ndarray":22,"./util":86}],14:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("./ndarray");
var TanHFunc = (function () {
    function TanHFunc() {
    }
    TanHFunc.prototype.output = function (math, x) {
        return math.scope(function () {
            return math.tanh(x);
        });
    };
    TanHFunc.prototype.der = function (math, x, y) {
        return math.scope(function () {
            var ySquared = math.elementWiseMul(y, y);
            return math.scalarMinusArray(ndarray_1.Scalar.ONE, ySquared);
        });
    };
    return TanHFunc;
}());
exports.TanHFunc = TanHFunc;
var ReLUFunc = (function () {
    function ReLUFunc() {
    }
    ReLUFunc.prototype.output = function (math, x) {
        return math.scope(function () {
            return math.relu(x);
        });
    };
    ReLUFunc.prototype.der = function (math, x, y) {
        return math.scope(function () {
            return math.step(x);
        });
    };
    return ReLUFunc;
}());
exports.ReLUFunc = ReLUFunc;
var SigmoidFunc = (function () {
    function SigmoidFunc() {
    }
    SigmoidFunc.prototype.output = function (math, x) {
        return math.scope(function () {
            return math.sigmoid(x);
        });
    };
    SigmoidFunc.prototype.der = function (math, x, y) {
        return math.scope(function () {
            var ySquared = math.elementWiseMul(y, y);
            return math.sub(y, ySquared);
        });
    };
    return SigmoidFunc;
}());
exports.SigmoidFunc = SigmoidFunc;
var SquareFunc = (function () {
    function SquareFunc() {
    }
    SquareFunc.prototype.output = function (math, x) {
        return math.scope(function () {
            return math.elementWiseMul(x, x);
        });
    };
    SquareFunc.prototype.der = function (math, x, y) {
        return math.scope(function () {
            return math.scalarTimesArray(ndarray_1.Scalar.TWO, x);
        });
    };
    return SquareFunc;
}());
exports.SquareFunc = SquareFunc;

},{"./ndarray":22}],15:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../util");
function assertConcat3DShapesMatch(x1Shape, x2Shape, axis, errorMessagePrefix) {
    if (errorMessagePrefix === void 0) { errorMessagePrefix = ''; }
    util.assert(x1Shape.length === 3, errorMessagePrefix + 'Concat3D x1 shape should be of rank 3.');
    util.assert(x2Shape.length === 3, errorMessagePrefix + 'Concat3D x2 shape should be of rank 3.');
    util.assert(axis >= 0 && axis < 3, 'Axis for concat3D must be between 0 and 2.');
    for (var i = 0; i < 3; i++) {
        util.assert((i === axis) || (x1Shape[i] === x2Shape[i]), errorMessagePrefix +
            ("Shape (" + x1Shape + ") does not match (" + x2Shape + ") along ") +
            "non-concatenated axis.");
    }
}
exports.assertConcat3DShapesMatch = assertConcat3DShapesMatch;
function computeConcat3DOutputShape(x1Shape, x2Shape, axis) {
    util.assert(x1Shape.length === 3, 'Concat3D x1 shape should be of rank 3.');
    util.assert(x2Shape.length === 3, 'Concat3D x2shape should be of rank 3.');
    var outputShape = x1Shape.slice();
    outputShape[axis] += x2Shape[axis];
    return outputShape;
}
exports.computeConcat3DOutputShape = computeConcat3DOutputShape;

},{"../util":86}],16:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../util");
function computeOutputShape3D(inputShapeRowColDepth, fieldSize, depth, stride, zeroPad) {
    if (zeroPad == null) {
        zeroPad = computeDefaultPad(inputShapeRowColDepth, fieldSize, stride);
    }
    var inputRows = inputShapeRowColDepth[0];
    var inputCols = inputShapeRowColDepth[1];
    var outputRows = (inputRows - fieldSize + 2 * zeroPad) / stride + 1;
    util.assert(util.isInt(outputRows), "The output # of rows (" + outputRows + ") must be an integer. Change the " +
        "stride and/or zero pad parameters");
    var outputCols = (inputCols - fieldSize + 2 * zeroPad) / stride + 1;
    util.assert(util.isInt(outputCols), "The output # of columns (" + outputCols + ") must be an integer. Change " +
        "the stride and/or zero pad parameters");
    return [outputRows, outputCols, depth];
}
exports.computeOutputShape3D = computeOutputShape3D;
function computeDefaultPad(inputShape, fieldSize, stride) {
    return Math.floor((inputShape[0] * (stride - 1) - stride + fieldSize) / 2);
}
exports.computeDefaultPad = computeDefaultPad;
function computeTexShapeFrom3D(shapeRowColDepth) {
    return [shapeRowColDepth[0], shapeRowColDepth[1] * shapeRowColDepth[2]];
}
exports.computeTexShapeFrom3D = computeTexShapeFrom3D;
function computeWeightsShape4D(inputDepth, outputDepth, fSize) {
    return [fSize, fSize, inputDepth, outputDepth];
}
exports.computeWeightsShape4D = computeWeightsShape4D;
function computeWeightsTexShape(inputDepth, outputDepth, fieldSize) {
    return [fieldSize * fieldSize * inputDepth, outputDepth];
}
exports.computeWeightsTexShape = computeWeightsTexShape;
function computeBiasesTexShape(outputDepth) {
    return [1, outputDepth];
}
exports.computeBiasesTexShape = computeBiasesTexShape;
function computeDilatedRC(rc, origStride) {
    var rowsDilated = (rc[0] - 1) * origStride + 1;
    var colsDilated = (rc[1] - 1) * origStride + 1;
    return [rowsDilated, colsDilated];
}
exports.computeDilatedRC = computeDilatedRC;

},{"../util":86}],17:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function validateShapes(sourceSize, destSize) {
    var srcArea = sourceSize[0] * sourceSize[1];
    var dstArea = destSize[0] * destSize[1];
    if (srcArea !== dstArea) {
        var srcStr = '[' + sourceSize[0] + ', ' + sourceSize[1] + ']';
        var dstStr = '[' + destSize[0] + ', ' + destSize[1] + ']';
        throw new Error('copy2D shapes have different areas:\n  sourceSize ' + srcStr +
            ', area ' + srcArea + '\n  destSize ' + dstStr + ', area ' + dstArea);
    }
}
exports.validateShapes = validateShapes;

},{}],18:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("./ndarray");
var SquareCostFunc = (function () {
    function SquareCostFunc() {
        this.halfOne = ndarray_1.Scalar.new(0.5);
    }
    SquareCostFunc.prototype.cost = function (math, x1, x2) {
        var diff = math.sub(x1, x2);
        var diffSquared = math.elementWiseMul(diff, diff);
        var result = math.scalarTimesArray(this.halfOne, diffSquared);
        diff.dispose();
        diffSquared.dispose();
        return result;
    };
    SquareCostFunc.prototype.der = function (math, x1, x2) {
        return math.sub(x1, x2);
    };
    SquareCostFunc.prototype.dispose = function () {
        this.halfOne.dispose();
    };
    return SquareCostFunc;
}());
exports.SquareCostFunc = SquareCostFunc;

},{"./ndarray":22}],19:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../util");
var concat3d_util = require("./concat3d_util");
var copy2d_util = require("./copy2d_util");
var ndarray_1 = require("./ndarray");
var NDArrayMath = (function () {
    function NDArrayMath(safeMode) {
        this.safeMode = safeMode;
        this.ndarrayScopes = [];
        this.ndarraysToKeep = [];
        this.activeScopeNDArraysToKeep = [];
    }
    NDArrayMath.prototype.scope = function (scopeFn) {
        var _this = this;
        this.startScope();
        var keepFn = function (ndarray) { return _this.keep(ndarray); };
        var trackFn = function (ndarray) { return _this.track(ndarray); };
        var result = scopeFn(keepFn, trackFn);
        this.endScope(result);
        return result;
    };
    NDArrayMath.prototype.startScope = function () {
        var newScope = [];
        this.ndarrayScopes.push(newScope);
        this.activeScope = newScope;
        var newNDArraysToKeep = [];
        this.ndarraysToKeep.push(newNDArraysToKeep);
        this.activeScopeNDArraysToKeep = newNDArraysToKeep;
    };
    NDArrayMath.prototype.endScope = function (result) {
        var _this = this;
        for (var i = 0; i < this.activeScope.length; i++) {
            var ndarray = this.activeScope[i];
            if (this.isNDArrayDataInList(ndarray, this.activeScopeNDArraysToKeep) ||
                (result != null && result instanceof ndarray_1.NDArray &&
                    ndarray.getData() === result.getData())) {
                continue;
            }
            ndarray.dispose();
        }
        this.ndarrayScopes.pop();
        this.activeScope = this.ndarrayScopes.length === 0 ?
            null :
            this.ndarrayScopes[this.ndarrayScopes.length - 1];
        if (result instanceof ndarray_1.NDArray &&
            !this.isNDArrayDataInList(result, this.activeScopeNDArraysToKeep)) {
            this.track(result);
        }
        else if (Array.isArray(result)) {
            result.forEach(function (r) {
                if (r instanceof ndarray_1.NDArray &&
                    !_this.isNDArrayDataInList(r, _this.activeScopeNDArraysToKeep)) {
                    _this.track(r);
                }
            });
        }
        this.ndarraysToKeep.pop();
        this.activeScopeNDArraysToKeep = this.ndarraysToKeep.length === 0 ?
            null :
            this.ndarraysToKeep[this.ndarraysToKeep.length - 1];
    };
    NDArrayMath.prototype.isNDArrayDataInList = function (ndarray, ndarrayList) {
        for (var i = 0; i < ndarrayList.length; i++) {
            if (ndarrayList[i].getData() === ndarray.getData()) {
                return true;
            }
        }
        return false;
    };
    NDArrayMath.prototype.keep = function (result) {
        if (this.activeScope == null) {
            if (this.safeMode) {
                throw new Error('You are using math in safe mode. Enclose all ' +
                    'math.method() calls inside a scope: ' +
                    'math.scope(() => {math.method();...}) to avoid memory ' +
                    'leaks.');
            }
            return result;
        }
        this.activeScopeNDArraysToKeep.push(result);
        return result;
    };
    NDArrayMath.prototype.track = function (result) {
        if (this.activeScope == null) {
            if (this.safeMode) {
                throw new Error('You are using math in safe mode. Enclose all ' +
                    'math.method() calls inside a scope: ' +
                    'math.scope(() => {math.method();...}) to avoid memory ' +
                    'leaks.');
            }
            return result;
        }
        this.activeScope.push(result);
        return result;
    };
    NDArrayMath.prototype.matMul = function (a, b, aOrientation, bOrientation) {
        if (aOrientation === void 0) { aOrientation = MatrixOrientation.REGULAR; }
        if (bOrientation === void 0) { bOrientation = MatrixOrientation.REGULAR; }
        var innerShapeA = (aOrientation === MatrixOrientation.REGULAR) ? a.shape[1] : a.shape[0];
        var innerShapeB = (bOrientation === MatrixOrientation.REGULAR) ? b.shape[0] : b.shape[1];
        util.assert(a.rank === 2 && b.rank === 2, "Error in matMul: inputs must be rank 2, got ranks " + a.rank +
            ("and " + b.rank + "."));
        util.assert(innerShapeA === innerShapeB, "Error in matMul: inner shapes (" + innerShapeA + ") and (" +
            (innerShapeB + ") of NDArrays with shapes " + a.shape + " and ") +
            (b.shape + " and orientations " + MatrixOrientation[aOrientation]) +
            (" and " + MatrixOrientation[bOrientation] + " must match."));
        return this.track(this.matMulInternal(a, b, aOrientation, bOrientation));
    };
    NDArrayMath.prototype.vectorTimesMatrix = function (v, matrix) {
        util.assert(v.rank === 1, "Error in vectorTimesMatrix: first input must be rank 1, but got " +
            ("rank " + v.rank + "."));
        util.assert(matrix.rank === 2, "Error in vectorTimesMatrix: second input must be rank 2, but got " +
            ("rank " + matrix.rank + "."));
        util.assert(v.size === matrix.shape[0], "Error in vectorTimesMatrix: size of first rank 1 input (" + v.size + ") " +
            "must match inner dimension of second rank 2 input, but got " +
            ("rank " + matrix.rank + "."));
        return this.matMul(v.as2D(1, v.size), matrix).as1D();
    };
    NDArrayMath.prototype.matrixTimesVector = function (matrix, v) {
        util.assert(v.rank === 1, "Error in vectorTimesMatrix: second input must rank 1, but got " +
            ("rank " + v.rank + "."));
        util.assert(matrix.rank === 2, "Error in vectorTimesMatrix: first input must be a rank 2, but got " +
            ("rank " + matrix.rank + "."));
        util.assert(v.size === matrix.shape[1], "Error in vectorTimesMatrix: size of first rank 1 input " + v.size + " " +
            "must match inner dimension of second rank 2 input, but got " +
            ("shape " + matrix.shape + "."));
        return this.matMul(matrix, v.as2D(v.size, 1)).as1D();
    };
    NDArrayMath.prototype.dotProduct = function (v1, v2) {
        util.assert(v1.rank === 1 && v2.rank === 1, "Error in dotProduct: inputs must be rank 1, but got ranks " +
            (v1.rank + " and " + v2.rank + "."));
        util.assert(v1.size === v2.size, "Error in dotProduct: size of inputs (" + v1.size + ") and (" +
            (v2.size + ") must match."));
        return this.matMul(v1.as2D(1, v1.size), v2.as2D(v2.size, 1)).asScalar();
    };
    NDArrayMath.prototype.outerProduct = function (v1, v2) {
        util.assert(v1.rank === 1 && v2.rank === 1, "Error in outerProduct: inputs must be rank 1, but got ranks " +
            (v1.rank + " and " + v2.rank + "."));
        return this.matMul(v1.as2D(v1.size, 1), v2.as2D(1, v2.size));
    };
    NDArrayMath.prototype.clone = function (ndarray) {
        return this.track(this.cloneInternal(ndarray));
    };
    NDArrayMath.prototype.reshape = function (ndarray, newShape) {
        util.assert(ndarray.size === util.sizeFromShape(newShape), "Error in reshape: old size " + ndarray.size + " must match new size " +
            (util.sizeFromShape(newShape) + "."));
        return this.track(this.reshapeInternal(ndarray, newShape));
    };
    NDArrayMath.prototype.slice2D = function (input, begin, size) {
        util.assert(begin[0] + size[0] <= input.shape[0] &&
            begin[1] + size[1] <= input.shape[1], "Error in slice2D: requested start position " + begin + " and size " +
            (size + " would overflow input of shape " + input.shape + "."));
        return this.track(this.slice2DInternal(input, begin, size));
    };
    NDArrayMath.prototype.copy2D = function (source, sourceBegin, sourceSize, dest, destBegin, destSize) {
        util.assert(sourceBegin[0] + sourceSize[0] <= source.shape[0] &&
            sourceBegin[1] + sourceSize[1] <= source.shape[1], "Error in copy2D: requested source start position " + sourceBegin + " " +
            ("and source size " + sourceSize + " would overflow source NDArray") +
            ("of shape " + source.shape + "."));
        util.assert(destBegin[0] + destSize[0] <= dest.shape[0] &&
            destBegin[1] + destSize[1] <= dest.shape[1], "Error in copy2D: requested dest start position " + destBegin + " " +
            ("and source size " + destSize + " would overflow dest NDArray of") +
            ("shape " + dest.shape + "."));
        copy2d_util.validateShapes(sourceSize, destSize);
        return this.copy2DInternal(source, sourceBegin, sourceSize, dest, destBegin, destSize);
    };
    NDArrayMath.prototype.concat3D = function (ndarray1, ndarray2, axis) {
        concat3d_util.assertConcat3DShapesMatch(ndarray1.shape, ndarray2.shape, axis, 'Error in concat3d: ');
        return this.track(this.concat3DInternal(ndarray1, ndarray2, axis));
    };
    NDArrayMath.prototype.logSumExp = function (ndarray) {
        return this.track(this.logSumExpInternal(ndarray));
    };
    NDArrayMath.prototype.sum = function (ndarray) {
        return this.track(this.sumInternal(ndarray));
    };
    NDArrayMath.prototype.argMin = function (ndarray) {
        return this.track(this.argMinInternal(ndarray));
    };
    NDArrayMath.prototype.argMax = function (ndarray) {
        return this.track(this.argMaxInternal(ndarray));
    };
    NDArrayMath.prototype.argMaxEquals = function (x1, x2) {
        util.assertShapesMatch(x1.shape, x2.shape, 'Error in argMaxEquals: ');
        return this.track(this.argMaxEqualsInternal(x1, x2));
    };
    NDArrayMath.prototype.topK = function (ndarray, k) {
        util.assert(k <= ndarray.size, "Error in topK: k value (" + k + ") must be less than size of input " +
            ("ndarray, got shape " + ndarray.shape + "."));
        var result = this.topKInternal(ndarray, k);
        this.track(result.values);
        this.track(result.indices);
        return result;
    };
    NDArrayMath.prototype.min = function (ndarray) {
        return this.track(this.minInternal(ndarray));
    };
    NDArrayMath.prototype.max = function (ndarray) {
        return this.track(this.maxInternal(ndarray));
    };
    NDArrayMath.prototype.softmax = function (x) {
        var _this = this;
        return this.scope(function () {
            var lse = _this.logSumExp(x);
            var logResult = _this.arrayMinusScalar(x, lse);
            return _this.exp(logResult);
        });
    };
    NDArrayMath.prototype.switchDim = function (a, newDim) {
        util.assert(a.rank === newDim.length, "Error in switchDim: length of input shape " + a.shape + " " +
            ("must match size of newDim array " + newDim + "."));
        return this.track(this.switchDimInternal(a, newDim));
    };
    NDArrayMath.prototype.scalarPlusArray = function (c, a) {
        util.assert(c.size === 1, "Error in scalarPlusArray: first argument must be rank 0, but got " +
            ("rank " + c.rank + "."));
        return this.track(this.scalarPlusArrayInternal(c, a));
    };
    NDArrayMath.prototype.scalarMinusArray = function (c, a) {
        util.assert(c.size === 1, "Error in scalarMinusArray: first argument must be rank 0, but got " +
            ("rank " + c.rank + "."));
        return this.track(this.scalarMinusArrayInternal(c, a));
    };
    NDArrayMath.prototype.arrayMinusScalar = function (a, c) {
        util.assert(c.size === 1, "Error in arrayMinusScalar: second argument must be rank 0, but " +
            ("got rank " + c.rank + "."));
        return this.track(this.arrayMinusScalarInternal(a, c));
    };
    NDArrayMath.prototype.neg = function (a) {
        return this.track(this.negInternal(a));
    };
    NDArrayMath.prototype.add = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in add: ');
        return this.track(this.addInternal(a, b));
    };
    NDArrayMath.prototype.sub = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in sub: ');
        return this.track(this.subInternal(a, b));
    };
    NDArrayMath.prototype.elementWiseMul = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in elementWiseMul: ');
        return this.track(this.elementWiseMulInternal(a, b));
    };
    NDArrayMath.prototype.divide = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in divide: ');
        return this.track(this.divideInternal(a, b));
    };
    NDArrayMath.prototype.scalarDividedByArray = function (c, a) {
        util.assert(c.size === 1, "Error in scalarDividedByArray: first argument must be rank 0, but " +
            ("got NDArray of rank " + c.rank + "."));
        return this.track(this.scalarDividedByArrayInternal(c, a));
    };
    NDArrayMath.prototype.arrayDividedByScalar = function (a, c) {
        util.assert(c.size === 1, "Error in arrayDividedByScalar: second argument must be rank 0, " +
            ("but got NDArray of rank " + c.rank + "."));
        return this.track(this.arrayDividedByScalarInternal(a, c));
    };
    NDArrayMath.prototype.exp = function (ndarray) {
        return this.track(this.expInternal(ndarray));
    };
    NDArrayMath.prototype.log = function (ndarray) {
        return this.track(this.logInternal(ndarray));
    };
    NDArrayMath.prototype.relu = function (ndarray) {
        return this.track(this.reluInternal(ndarray));
    };
    NDArrayMath.prototype.sigmoid = function (ndarray) {
        return this.track(this.sigmoidInternal(ndarray));
    };
    NDArrayMath.prototype.tanh = function (ndarray) {
        return this.track(this.tanhInternal(ndarray));
    };
    NDArrayMath.prototype.sin = function (ndarray) {
        return this.track(this.sinInternal(ndarray));
    };
    NDArrayMath.prototype.step = function (ndarray) {
        return this.track(this.stepInternal(ndarray));
    };
    NDArrayMath.prototype.scaledArrayAdd = function (c1, a, c2, b) {
        util.assert(c1.size === 1, "Error in scaledArrayAdd: first argument must rank 0, but got " +
            (" rank " + c1.rank + "."));
        util.assert(c2.size === 1, "Error in scaledArrayAdd: third argument must be rank 0, but got " +
            ("NDArray of rank " + c2.rank + "."));
        util.assertShapesMatch(a.shape, b.shape, 'Error in scaledArrayAdd: ');
        return this.track(this.scaledArrayAddInternal(c1, a, c2, b));
    };
    NDArrayMath.prototype.scalarTimesArray = function (c, a) {
        util.assert(c.size === 1, "Error in arrayDividedByScalar: first argument must be rank 0, but " +
            ("got rank " + c.rank + "."));
        return this.track(this.scalarTimesArrayInternal(c, a));
    };
    NDArrayMath.prototype.elementWiseMulBroadcast = function (a, b) {
        util.assert(a.rank === 2, "Error in elementWiseMulBroadcast: first argument must be " +
            ("rank 2, but got rank " + a.rank + "."));
        util.assert(b.rank === 2, "Error in elementWiseMulBroadcast: second argument must be " +
            ("rank 2, but got rank " + b.rank + "."));
        return this.track(this.elementWiseMulBroadcastInternal(a, b));
    };
    NDArrayMath.prototype.conv2d = function (x, weights, biases, stride, zeroPad) {
        util.assert(x.rank === 3, "Error in conv2d: x must be rank 3, but got rank " + x.rank + ".");
        util.assert(weights.rank === 4, "Error in conv2d: weights must be rank 4, but got rank " +
            (weights.rank + "."));
        if (biases != null) {
            util.assert(biases.rank === 1, "Error in conv2d: biases must be rank 1, but got rank " +
                (biases.rank + "."));
        }
        util.assert(x.shape[2] === weights.shape[2], "Error in conv2d: depth of input (" + x.shape[2] + ") must match  " +
            ("input depth for weights " + weights.shape[2] + "."));
        return this.track(this.conv2dInternal(x, weights, biases, stride, zeroPad));
    };
    NDArrayMath.prototype.conv2dBackProp = function (x, dy, weights, stride, pad) {
        util.assert(x.rank === 3, "Error in conv2dBackProp: x must be rank 3, but got shape " +
            (x.shape + "."));
        util.assert(dy.rank === 3, "Error in conv2dBackProp: dy must be rank 3, but got shape " +
            (dy.shape + "."));
        util.assert(weights.rank === 4, "Error in conv2dBackProp: weights must be rank 4, but got shape " +
            (weights.shape + "."));
        util.assert(x.shape[2] === weights.shape[2], "Error in conv2dBackProp: depth of x " + x.shape[2] + ") must " +
            ("match input depth for weights (" + weights.shape[2] + "."));
        util.assert(dy.shape[2] === weights.shape[3], "Error in conv2dBackProp: depth of dy (" + dy.shape[2] + ") must " +
            ("match output depth for weights (" + weights.shape[3] + ")."));
        var backpropResult = this.conv2dBackPropInternal(x, dy, weights, stride, pad);
        this.track(backpropResult.db);
        this.track(backpropResult.dw);
        this.track(backpropResult.dx);
        return backpropResult;
    };
    NDArrayMath.prototype.conv2dTranspose = function (x, weights, biases, stride, pad) {
        util.assert(x.rank === 3, "Error in conv2dTranspose: x must be rank 3, but got rank " +
            (x.rank + "."));
        util.assert(weights.rank === 4, "Error in conv2dTranspose: weights must be rank 4, but got " +
            ("rank " + weights.rank));
        if (biases != null) {
            util.assert(biases.rank === 1, "Error in conv2dTranspose: biases must be rank 1, but got ' +\n              'rank " + biases.rank + ".");
        }
        util.assert(x.shape[2] === weights.shape[3], "Error in conv2dTranspose: depth of input (" + x.shape[2] + ") must " +
            ("match input depth for weights " + weights.shape[3] + "."));
        return this.track(this.conv2dTransposeInternal(x, weights, biases, stride, pad));
    };
    NDArrayMath.prototype.maxPool = function (x, fSize, stride, pad) {
        util.assert(x.rank === 3, 'Error in maxPool: x must be rank 3 but got rank ' + x.rank + '.');
        return this.track(this.maxPoolInternal(x, fSize, stride, pad));
    };
    NDArrayMath.prototype.maxPoolBackprop = function (dy, x, fSize, stride, pad) {
        util.assert(dy.rank === 3, "Error in maxPoolBackprop: dy must be rank 3 but got rank " +
            (dy.rank + "."));
        util.assert(x.rank === 3, "Error in maxPoolBackprop: x must be rank 3 but got rank " +
            (x.rank + "."));
        return this.track(this.maxPoolBackpropInternal(dy, x, fSize, stride, pad));
    };
    NDArrayMath.prototype.minPool = function (x, fSize, stride, pad) {
        util.assert(x.rank === 3, "Error in minPool: x must be rank 3 but got rank " + x.rank + ".");
        return this.track(this.minPoolInternal(x, fSize, stride, pad));
    };
    NDArrayMath.prototype.avgPool = function (x, fSize, stride, pad) {
        util.assert(x.rank === 3, "Error in avgPool: x must be rank 3 but got rank " + x.rank + ".");
        return this.track(this.avgPoolInternal(x, fSize, stride, pad));
    };
    NDArrayMath.prototype.resizeBilinear3D = function (x, newShape2D, alignCorners) {
        if (alignCorners === void 0) { alignCorners = false; }
        util.assert(x.rank === 3, "Error in resizeBilinear3D: x must be rank 3 but got rank " + x.rank + ".");
        util.assert(newShape2D.length === 2, "Error in resizeBilinear3D: new shape must 2D, but got shape " +
            (newShape2D + "."));
        return this.track(this.resizeBilinear3DInternal(x, newShape2D, alignCorners));
    };
    NDArrayMath.prototype.batchNormalization3D = function (x, mean, variance, varianceEpsilon, scale, offset) {
        if (varianceEpsilon === void 0) { varianceEpsilon = .001; }
        util.assert(x.rank === 3, "Error in batchNormalization3D: x must be rank 3 but got rank " +
            (x.rank + "."));
        util.assert(mean.rank === 3 || mean.rank === 1, "Error in batchNormalization3D: mean must be rank 3 or rank 1 but " +
            ("got rank " + mean.rank + "."));
        util.assert(variance.rank === 3 || variance.rank === 1, "Error in batchNormalization3D: variance must be rank 3 or rank 1 " +
            ("but got rank " + variance.rank + "."));
        if (scale != null) {
            util.assert(scale.rank === 3 || scale.rank === 1, "Error in batchNormalization3D: scale must be rank 3 or rank 1 " +
                ("but got rank " + scale.rank + "."));
        }
        if (offset != null) {
            util.assert(offset.rank === 3 || offset.rank === 1, "Error in batchNormalization3D: offset must be rank 3 or rank 1 " +
                ("but got rank " + offset.rank + "."));
        }
        return this.track(this.batchNormalization3DInternal(x, mean, variance, varianceEpsilon, scale, offset));
    };
    return NDArrayMath;
}());
exports.NDArrayMath = NDArrayMath;
var MatrixOrientation;
(function (MatrixOrientation) {
    MatrixOrientation[MatrixOrientation["REGULAR"] = 0] = "REGULAR";
    MatrixOrientation[MatrixOrientation["TRANSPOSED"] = 1] = "TRANSPOSED";
})(MatrixOrientation = exports.MatrixOrientation || (exports.MatrixOrientation = {}));

},{"../util":86,"./concat3d_util":15,"./copy2d_util":17,"./ndarray":22}],20:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../math/conv_util");
var util = require("../util");
var concat3d_util = require("./concat3d_util");
var copy2D_util = require("./copy2d_util");
var math_1 = require("./math");
var ndarray_1 = require("./ndarray");
var NDArrayMathCPU = (function (_super) {
    __extends(NDArrayMathCPU, _super);
    function NDArrayMathCPU(safeMode) {
        if (safeMode === void 0) { safeMode = false; }
        return _super.call(this, safeMode) || this;
    }
    NDArrayMathCPU.prototype.cloneInternal = function (ndarray) {
        return ndarray_1.NDArray.make(ndarray.shape, { values: new Float32Array(ndarray.getValues()) });
    };
    NDArrayMathCPU.prototype.reshapeInternal = function (ndarray, newShape) {
        return this.cloneInternal(ndarray).reshape(newShape);
    };
    NDArrayMathCPU.prototype.slice2DInternal = function (input, beginRowCol, sizeRowCol) {
        var result = ndarray_1.Array2D.zeros(sizeRowCol);
        this.copy2DInternal(input, beginRowCol, sizeRowCol, result, [0, 0], sizeRowCol);
        return result;
    };
    NDArrayMathCPU.prototype.copy2DInternal = function (source, sourceBeginRowCol, sourceSizeRowCol, dest, destBeginRowCol, destSizeRowCol) {
        copy2D_util.validateShapes(sourceSizeRowCol, destSizeRowCol);
        var srcValues = source.getValues();
        var dstValues = dest.getValues();
        var n = sourceSizeRowCol[0] * sourceSizeRowCol[1];
        for (var i = 0; i < n; ++i) {
            var srcRow = sourceBeginRowCol[0] + Math.floor(i / sourceSizeRowCol[1]);
            var srcCol = sourceBeginRowCol[1] + (i % sourceSizeRowCol[1]);
            var srcOff = srcRow * source.shape[1] + srcCol;
            var dstRow = destBeginRowCol[0] + Math.floor(i / destSizeRowCol[1]);
            var dstCol = destBeginRowCol[1] + (i % destSizeRowCol[1]);
            var dstOff = dstRow * dest.shape[1] + dstCol;
            dstValues[dstOff] = srcValues[srcOff];
        }
    };
    NDArrayMathCPU.prototype.concat3DInternal = function (x1, x2, axis) {
        var outputShape = concat3d_util.computeConcat3DOutputShape(x1.shape, x2.shape, axis);
        var values = ndarray_1.NDArray.zeros(outputShape);
        for (var i = 0; i < outputShape[0]; i++) {
            for (var j = 0; j < outputShape[1]; j++) {
                for (var k = 0; k < outputShape[2]; k++) {
                    var index = [i, j, k];
                    var value = void 0;
                    if (index[axis] < x1.shape[axis]) {
                        value = x1.get(i, j, k);
                    }
                    else {
                        index[axis] -= x1.shape[axis];
                        var i2 = index[0], j2 = index[1], k2 = index[2];
                        value = x2.get(i2, j2, k2);
                    }
                    values.set(value, i, j, k);
                }
            }
        }
        return values;
    };
    NDArrayMathCPU.prototype.scalarPlusArrayInternal = function (c, a) {
        var resultValues = new Float32Array(a.size);
        var aValues = a.getValues();
        var cVal = c.get();
        for (var i = 0; i < resultValues.length; ++i) {
            resultValues[i] = cVal + aValues[i];
        }
        return ndarray_1.NDArray.make(a.shape, { values: resultValues });
    };
    NDArrayMathCPU.prototype.scaledArrayAddInternal = function (c1, a, c2, b) {
        var cValues = new Float32Array(a.size);
        var aValues = a.getValues();
        var bValues = b.getValues();
        var c1Val = c1.get();
        var c2Val = c2.get();
        for (var i = 0; i < cValues.length; ++i) {
            cValues[i] = c1Val * aValues[i] + c2Val * bValues[i];
        }
        return ndarray_1.NDArray.make(a.shape, { values: cValues });
    };
    NDArrayMathCPU.prototype.scalarTimesArrayInternal = function (c, a) {
        var newValues = new Float32Array(a.size);
        var aValues = a.getValues();
        var cVal = c.get();
        for (var i = 0; i < aValues.length; ++i) {
            newValues[i] = cVal * aValues[i];
        }
        return ndarray_1.NDArray.make(a.shape, { values: newValues });
    };
    NDArrayMathCPU.prototype.scalarMinusArrayInternal = function (c, a) {
        var negA = this.negInternal(a);
        var result = this.scalarPlusArrayInternal(c, negA);
        negA.dispose();
        return result;
    };
    NDArrayMathCPU.prototype.arrayMinusScalarInternal = function (a, c) {
        var negC = this.negInternal(c);
        var result = this.scalarPlusArrayInternal(negC, a);
        negC.dispose();
        return result;
    };
    NDArrayMathCPU.prototype.negInternal = function (a) {
        return this.scalarTimesArrayInternal(ndarray_1.Scalar.NEG_ONE, a);
    };
    NDArrayMathCPU.prototype.addInternal = function (a, b) {
        return this.scaledArrayAddInternal(ndarray_1.Scalar.ONE, a, ndarray_1.Scalar.ONE, b);
    };
    NDArrayMathCPU.prototype.subInternal = function (a, b) {
        return this.scaledArrayAddInternal(ndarray_1.Scalar.ONE, a, ndarray_1.Scalar.NEG_ONE, b);
    };
    NDArrayMathCPU.prototype.matMulInternal = function (a, b, aOrientation, bOrientation) {
        if (aOrientation === void 0) { aOrientation = math_1.MatrixOrientation.REGULAR; }
        if (bOrientation === void 0) { bOrientation = math_1.MatrixOrientation.REGULAR; }
        var sharedDim = (aOrientation === math_1.MatrixOrientation.REGULAR) ? a.shape[1] : a.shape[0];
        var leftDim = (aOrientation === math_1.MatrixOrientation.REGULAR) ? a.shape[0] : a.shape[1];
        var rightDim = (bOrientation === math_1.MatrixOrientation.REGULAR) ? b.shape[1] : b.shape[0];
        var normalGetter = function (matrix, i, j) {
            return matrix.get(i, j);
        };
        var transposedGetter = function (matrix, i, j) {
            return matrix.get(j, i);
        };
        var aGetter = (aOrientation === math_1.MatrixOrientation.REGULAR) ?
            normalGetter :
            transposedGetter;
        var bGetter = (bOrientation === math_1.MatrixOrientation.REGULAR) ?
            normalGetter :
            transposedGetter;
        var values = new Float32Array(leftDim * rightDim);
        var index = 0;
        for (var i = 0; i < leftDim; ++i) {
            for (var j = 0; j < rightDim; ++j) {
                var sum = 0;
                for (var k = 0; k < sharedDim; ++k) {
                    sum += aGetter(a, i, k) * bGetter(b, k, j);
                }
                values[index++] = sum;
            }
        }
        return ndarray_1.Array2D.new([leftDim, rightDim], values);
    };
    NDArrayMathCPU.prototype.elementWiseMulInternal = function (a, b) {
        var newValues = new Float32Array(a.size);
        var aValues = a.getValues();
        var bValues = b.getValues();
        for (var i = 0; i < aValues.length; ++i) {
            newValues[i] = aValues[i] * bValues[i];
        }
        return ndarray_1.NDArray.make(a.shape, { values: newValues });
    };
    NDArrayMathCPU.prototype.elementWiseMulBroadcastInternal = function (a, b) {
        var maxRow = Math.max(a.shape[0], b.shape[0]);
        var maxCol = Math.max(a.shape[1], b.shape[1]);
        var values = new Float32Array(maxRow * maxCol);
        var index = 0;
        for (var row = 0; row < maxRow; row++) {
            for (var col = 0; col < maxCol; col++) {
                values[index++] = a.get(row % a.shape[0], col % a.shape[1]) *
                    b.get(row % b.shape[0], col % b.shape[1]);
            }
        }
        return ndarray_1.Array2D.new([maxRow, maxCol], values);
    };
    NDArrayMathCPU.prototype.divideInternal = function (a, b) {
        var newValues = new Float32Array(a.size);
        var aValues = a.getValues();
        var bValues = b.getValues();
        for (var i = 0; i < aValues.length; ++i) {
            newValues[i] = aValues[i] / bValues[i];
        }
        return ndarray_1.NDArray.make(a.shape, { values: newValues });
    };
    NDArrayMathCPU.prototype.scalarDividedByArrayInternal = function (c, a) {
        var newValues = new Float32Array(a.size);
        var aValues = a.getValues();
        var cValue = c.get();
        for (var i = 0; i < aValues.length; ++i) {
            newValues[i] = cValue / aValues[i];
        }
        return ndarray_1.NDArray.make(a.shape, { values: newValues });
    };
    NDArrayMathCPU.prototype.arrayDividedByScalarInternal = function (a, c) {
        var newValues = new Float32Array(a.size);
        var aValues = a.getValues();
        var cValue = c.get();
        for (var i = 0; i < aValues.length; ++i) {
            newValues[i] = aValues[i] / cValue;
        }
        return ndarray_1.NDArray.make(a.shape, { values: newValues });
    };
    NDArrayMathCPU.prototype.sumInternal = function (ndarray) {
        var sum = 0;
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            sum += values[i];
        }
        return ndarray_1.Scalar.new(sum);
    };
    NDArrayMathCPU.prototype.argMinInternal = function (ndarray) {
        var min = Number.MAX_VALUE;
        var minIndex = -1;
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            var value = values[i];
            if (isNaN(value)) {
                return ndarray_1.Scalar.new(NaN);
            }
            if (value < min) {
                min = value;
                minIndex = i;
            }
        }
        return ndarray_1.Scalar.new(minIndex);
    };
    NDArrayMathCPU.prototype.argMaxInternal = function (ndarray) {
        var max = Number.NEGATIVE_INFINITY;
        var maxIndex = -1;
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            var value = values[i];
            if (isNaN(value)) {
                return ndarray_1.Scalar.new(NaN);
            }
            if (value > max) {
                max = value;
                maxIndex = i;
            }
        }
        return ndarray_1.Scalar.new(maxIndex);
    };
    NDArrayMathCPU.prototype.argMaxEqualsInternal = function (x1, x2) {
        var argMax1 = this.argMaxInternal(x1).get();
        var argMax2 = this.argMaxInternal(x2).get();
        if (isNaN(argMax1) || isNaN(argMax2)) {
            return ndarray_1.Scalar.new(NaN);
        }
        return ndarray_1.Scalar.new(+(argMax1 === argMax2));
    };
    NDArrayMathCPU.prototype.topKInternal = function (ndarray, k) {
        var values = ndarray.getValues();
        var valuesAndIndices = [];
        for (var i = 0; i < values.length; i++) {
            valuesAndIndices.push({ value: values[i], index: i });
        }
        valuesAndIndices.sort(function (a, b) {
            return b.value - a.value;
        });
        var topkValues = new Float32Array(k);
        var topkIndices = new Float32Array(k);
        for (var i = 0; i < k; i++) {
            topkValues[i] = valuesAndIndices[i].value;
            topkIndices[i] = valuesAndIndices[i].index;
        }
        return { values: ndarray_1.Array1D.new(topkValues), indices: ndarray_1.Array1D.new(topkIndices) };
    };
    NDArrayMathCPU.prototype.minInternal = function (ndarray) {
        var values = ndarray.getValues();
        var min = values[0];
        for (var i = 1; i < values.length; ++i) {
            var value = values[i];
            if (isNaN(value)) {
                return ndarray_1.Scalar.new(NaN);
            }
            if (value < min) {
                min = value;
            }
        }
        return ndarray_1.Scalar.new(min);
    };
    NDArrayMathCPU.prototype.maxInternal = function (ndarray) {
        var values = ndarray.getValues();
        var max = values[0];
        for (var i = 1; i < values.length; ++i) {
            var value = values[i];
            if (isNaN(value)) {
                return ndarray_1.Scalar.new(NaN);
            }
            if (value > max) {
                max = value;
            }
        }
        return ndarray_1.Scalar.new(max);
    };
    NDArrayMathCPU.prototype.expInternal = function (ndarray) {
        var values = ndarray.getValues();
        var newValues = new Float32Array(values.length);
        for (var i = 0; i < values.length; ++i) {
            newValues[i] = Math.exp(values[i]);
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: newValues });
    };
    NDArrayMathCPU.prototype.logInternal = function (ndarray) {
        var values = ndarray.getValues();
        var newValues = new Float32Array(values.length);
        for (var i = 0; i < values.length; ++i) {
            var value = values[i];
            newValues[i] = Math.log(value);
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: newValues });
    };
    NDArrayMathCPU.prototype.logSumExpInternal = function (ndarray) {
        var xMax = this.max(ndarray);
        var a = this.arrayMinusScalar(ndarray, xMax);
        var b = this.exp(a);
        var c = this.sum(b);
        var d = this.log(c);
        var result = this.add(xMax, d);
        xMax.dispose();
        a.dispose();
        b.dispose();
        c.dispose();
        d.dispose();
        return result;
    };
    NDArrayMathCPU.prototype.reluInternal = function (ndarray) {
        var resultValues = new Float32Array(ndarray.size);
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = Math.max(0, values[i]);
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: resultValues });
    };
    NDArrayMathCPU.prototype.sigmoidInternal = function (ndarray) {
        var resultValues = new Float32Array(ndarray.size);
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = 1 / (1 + Math.exp(-values[i]));
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: resultValues });
    };
    NDArrayMathCPU.prototype.tanhInternal = function (ndarray) {
        var resultValues = new Float32Array(ndarray.size);
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = util.tanh(values[i]);
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: resultValues });
    };
    NDArrayMathCPU.prototype.sinInternal = function (ndarray) {
        var resultValues = new Float32Array(ndarray.size);
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = Math.sin(values[i]);
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: resultValues });
    };
    NDArrayMathCPU.prototype.stepInternal = function (ndarray) {
        var resultValues = new Float32Array(ndarray.size);
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            var value = values[i];
            resultValues[i] = value > 0 ? 1 : (value < 0 ? 0 : value);
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: resultValues });
    };
    NDArrayMathCPU.prototype.conv2dInternal = function (x, weights, biases, stride, pad) {
        var _a = x.shape, xRows = _a[0], xCols = _a[1], inputDepth = _a[2];
        var fieldSize = weights.shape[0];
        var outputDepth = weights.shape[3];
        var outputShape = conv_util.computeOutputShape3D([xRows, xCols, inputDepth], fieldSize, outputDepth, stride, pad);
        var y = ndarray_1.Array3D.zeros(outputShape);
        for (var d2 = 0; d2 < outputDepth; ++d2) {
            for (var yR = 0; yR < y.shape[0]; ++yR) {
                var xRCorner = yR * stride - pad;
                var xRMin = Math.max(0, xRCorner);
                var xRMax = Math.min(xRows, fieldSize + xRCorner);
                for (var yC = 0; yC < y.shape[1]; ++yC) {
                    var xCCorner = yC * stride - pad;
                    var xCMin = Math.max(0, xCCorner);
                    var xCMax = Math.min(xCols, fieldSize + xCCorner);
                    var dotProd = 0;
                    for (var xR = xRMin; xR < xRMax; ++xR) {
                        var wR = xR - xRCorner;
                        for (var xC = xCMin; xC < xCMax; ++xC) {
                            var wC = xC - xCCorner;
                            for (var d1 = 0; d1 < inputDepth; ++d1) {
                                var pixel = x.get(xR, xC, d1);
                                var weight = weights.get(wR, wC, d1, d2);
                                dotProd += pixel * weight;
                            }
                        }
                    }
                    var bias = (biases != null) ? biases.get(d2) : 0;
                    y.set(dotProd + bias, yR, yC, d2);
                }
            }
        }
        return y;
    };
    NDArrayMathCPU.prototype.conv2dBackPropInternal = function (x, dy, weights, stride, pad) {
        var fSize = weights.shape[0];
        var dw = this.conv2dDerWeights(x, dy, fSize, stride, pad);
        var db = this.conv2dDerBias(dy);
        var dx = this.conv2dTransposeInternal(dy, weights, null, stride, pad);
        return { dx: dx, db: db, dw: dw };
    };
    NDArrayMathCPU.prototype.conv2dTransposeInternal = function (x, weights, biases, origStride, origPad) {
        var fSize = weights.shape[0];
        var pad = fSize - 1 - origPad;
        var origInputDepth = weights.shape[2];
        var origOutputDepth = weights.shape[3];
        var _a = x.shape, xRows = _a[0], xCols = _a[1], xDepth = _a[2];
        var xRowsDilated = (xRows - 1) * origStride + 1;
        var xColsDilated = (xCols - 1) * origStride + 1;
        var outputShape = conv_util.computeOutputShape3D([xRowsDilated, xColsDilated, origOutputDepth], fSize, origInputDepth, 1, pad);
        var y = ndarray_1.Array3D.zeros(outputShape);
        for (var d2 = 0; d2 < origInputDepth; ++d2) {
            for (var yR = 0; yR < y.shape[0]; ++yR) {
                var xRCorner = yR - pad;
                var xRMin = Math.max(0, Math.ceil(xRCorner / origStride));
                var xRMax = Math.min(xRows, (fSize + xRCorner) / origStride);
                for (var yC = 0; yC < y.shape[1]; ++yC) {
                    var xCCorner = yC - pad;
                    var xCMin = Math.max(0, Math.ceil(xCCorner / origStride));
                    var xCMax = Math.min(xCols, (fSize + xCCorner) / origStride);
                    var dotProd = 0;
                    for (var xR = xRMin; xR < xRMax; ++xR) {
                        var wR = xR * origStride - xRCorner;
                        for (var xC = xCMin; xC < xCMax; ++xC) {
                            var wC = xC * origStride - xCCorner;
                            for (var d1 = 0; d1 < origOutputDepth; ++d1) {
                                var pixel = x.get(xR, xC, d1);
                                var weight = weights.get(fSize - 1 - wR, fSize - 1 - wC, d2, d1);
                                dotProd += pixel * weight;
                            }
                        }
                    }
                    var bias = biases != null ? biases.get(d2) : 0;
                    y.set(dotProd + bias, yR, yC, d2);
                }
            }
        }
        return y;
    };
    NDArrayMathCPU.prototype.conv2dTransposeShaderLike = function (x, origWeights, origStride, origPad) {
        var fSize = origWeights.shape[0];
        var pad = fSize - 1 - origPad;
        var origInputDepth = origWeights.shape[2];
        var origOutputDepth = origWeights.shape[3];
        var _a = x.shape, xRows = _a[0], xCols = _a[1], xDepth = _a[2];
        var xRowsDilated = (xRows - 1) * origStride + 1;
        var xColsDilated = (xCols - 1) * origStride + 1;
        var outputShape = conv_util.computeOutputShape3D([xRowsDilated, xColsDilated, origOutputDepth], fSize, origInputDepth, 1, pad);
        var y = ndarray_1.Array3D.zeros(outputShape);
        for (var d2 = 0; d2 < origInputDepth; ++d2) {
            for (var yR = 0; yR < y.shape[0]; ++yR) {
                for (var yC = 0; yC < y.shape[1]; ++yC) {
                    var xRCorner = yR - pad;
                    var xCCorner = yC - pad;
                    var dotProd = 0;
                    for (var wR = 0; wR < fSize; ++wR) {
                        var xR = (xRCorner + wR) / origStride;
                        if (xR < 0 || xR >= xRows || Math.floor(xR) !== xR) {
                            continue;
                        }
                        for (var wC = 0; wC < fSize; ++wC) {
                            var xC = (xCCorner + wC) / origStride;
                            if (xC < 0 || xC >= xCols || Math.floor(xC) !== xC) {
                                continue;
                            }
                            for (var d1 = 0; d1 < origOutputDepth; ++d1) {
                                var pixel = x.get(xR, xC, d1);
                                var weight = origWeights.get(fSize - 1 - wR, fSize - 1 - wC, d2, d1);
                                dotProd += pixel * weight;
                            }
                        }
                    }
                    y.set(dotProd, yR, yC, d2);
                }
            }
        }
        return y;
    };
    NDArrayMathCPU.prototype.conv2dDerWeights = function (x, dY, fSize, stride, zeroPad) {
        var inputDepth = x.shape[2];
        var outputDepth = dY.shape[2];
        var weightsShape = conv_util.computeWeightsShape4D(inputDepth, outputDepth, fSize);
        var dW = ndarray_1.Array4D.zeros(weightsShape);
        var yNumRows = dY.shape[0];
        var yNumCols = dY.shape[1];
        var xNumRows = x.shape[0];
        var xNumCols = x.shape[1];
        for (var wR = 0; wR < fSize; ++wR) {
            var yRMin = Math.max(0, Math.ceil((zeroPad - wR) / stride));
            var yRMax = Math.min(yNumRows, (xNumRows + zeroPad - wR) / stride);
            for (var wC = 0; wC < fSize; ++wC) {
                var yCMin = Math.max(0, Math.ceil((zeroPad - wC) / stride));
                var yCMax = Math.min(yNumCols, (xNumCols + zeroPad - wC) / stride);
                for (var d1 = 0; d1 < inputDepth; ++d1) {
                    for (var d2 = 0; d2 < outputDepth; ++d2) {
                        var dotProd = 0;
                        for (var yR = yRMin; yR < yRMax; ++yR) {
                            var xR = wR + yR * stride - zeroPad;
                            for (var yC = yCMin; yC < yCMax; ++yC) {
                                var xC = wC + yC * stride - zeroPad;
                                dotProd += x.get(xR, xC, d1) * dY.get(yR, yC, d2);
                            }
                        }
                        dW.set(dotProd, wR, wC, d1, d2);
                    }
                }
            }
        }
        return dW;
    };
    NDArrayMathCPU.prototype.conv2dDerBias = function (dY) {
        var outputDepth = dY.shape[2];
        var numRows = dY.shape[0];
        var numCols = dY.shape[1];
        var values = new Float32Array(outputDepth);
        for (var d2 = 0; d2 < outputDepth; ++d2) {
            var sum = 0;
            for (var r = 0; r < numRows; ++r) {
                for (var c = 0; c < numCols; ++c) {
                    sum += dY.get(r, c, d2);
                }
            }
            values[d2] = sum;
        }
        return ndarray_1.Array1D.new(values);
    };
    NDArrayMathCPU.prototype.switchDimInternal = function (t, newDim) {
        var newShape = new Array(t.rank);
        for (var i = 0; i < newShape.length; i++) {
            newShape[i] = t.shape[newDim[i]];
        }
        var resultValues = new Float32Array(t.size);
        var values = t.getValues();
        var result = ndarray_1.NDArray.make(newShape, { values: resultValues });
        for (var i = 0; i < t.size; ++i) {
            var loc = t.indexToLoc(i);
            var newLoc = new Array(loc.length);
            for (var i_1 = 0; i_1 < newLoc.length; i_1++) {
                newLoc[i_1] = loc[newDim[i_1]];
            }
            var newIndex = result.locToIndex(newLoc);
            resultValues[newIndex] = values[i];
        }
        return result;
    };
    NDArrayMathCPU.prototype.pool = function (x, fSize, stride, pad, poolType) {
        var _a = x.shape, xRows = _a[0], xCols = _a[1], depth = _a[2];
        var outputShape = conv_util.computeOutputShape3D([xRows, xCols, depth], fSize, depth, stride, pad);
        var y = ndarray_1.Array3D.zeros(outputShape);
        for (var d = 0; d < depth; ++d) {
            for (var yR = 0; yR < y.shape[0]; ++yR) {
                var xRCorner = yR * stride - pad;
                var xRMin = Math.max(0, xRCorner);
                var xRMax = Math.min(xRows, fSize + xRCorner);
                for (var yC = 0; yC < y.shape[1]; ++yC) {
                    var xCCorner = yC * stride - pad;
                    var xCMin = Math.max(0, xCCorner);
                    var xCMax = Math.min(xCols, fSize + xCCorner);
                    var minMaxValue = (poolType === 'max' ? Number.NEGATIVE_INFINITY :
                        Number.POSITIVE_INFINITY);
                    var avgValue = 0;
                    for (var xR = xRMin; xR < xRMax; ++xR) {
                        var wR = xR - xRCorner;
                        for (var xC = xCMin; xC < xCMax; ++xC) {
                            var wC = xC - xCCorner;
                            var pixel = x.get(xR, xC, d);
                            if (isNaN(pixel)) {
                                minMaxValue = NaN;
                                avgValue = NaN;
                                break;
                            }
                            if ((poolType === 'max' && pixel > minMaxValue) ||
                                (poolType === 'min' && pixel < minMaxValue)) {
                                minMaxValue = pixel;
                            }
                            else if (poolType === 'avg') {
                                avgValue += pixel / (fSize * fSize);
                            }
                        }
                        if (isNaN(minMaxValue)) {
                            break;
                        }
                    }
                    y.set(poolType === 'avg' ? avgValue : minMaxValue, yR, yC, d);
                }
            }
        }
        return y;
    };
    NDArrayMathCPU.prototype.maxPoolInternal = function (x, fSize, stride, pad) {
        return this.pool(x, fSize, stride, pad, 'max');
    };
    NDArrayMathCPU.prototype.maxPoolPositions = function (x, fSize, stride, pad) {
        var _a = x.shape, xRows = _a[0], xCols = _a[1], depth = _a[2];
        var outputShape = conv_util.computeOutputShape3D(x.shape, fSize, depth, stride, pad);
        var maxPositions = ndarray_1.Array3D.zeros(outputShape);
        for (var d = 0; d < depth; ++d) {
            for (var yR = 0; yR < outputShape[0]; ++yR) {
                var xRCorner = yR * stride - pad;
                var xRMin = Math.max(0, xRCorner);
                var xRMax = Math.min(xRows, fSize + xRCorner);
                for (var yC = 0; yC < outputShape[1]; ++yC) {
                    var xCCorner = yC * stride - pad;
                    var xCMin = Math.max(0, xCCorner);
                    var xCMax = Math.min(xCols, fSize + xCCorner);
                    var maxValue = Number.NEGATIVE_INFINITY;
                    var maxPosition = -1;
                    for (var xR = xRMin; xR < xRMax; ++xR) {
                        var wR = xR - xRCorner;
                        for (var xC = xCMin; xC < xCMax; ++xC) {
                            var wC = xC - xCCorner;
                            var pixel = x.get(xR, xC, d);
                            if (pixel > maxValue) {
                                maxValue = pixel;
                                maxPosition = wR * fSize + wC;
                            }
                        }
                    }
                    maxPositions.set(maxPosition, yR, yC, d);
                }
            }
        }
        return maxPositions;
    };
    NDArrayMathCPU.prototype.maxPoolBackpropInternal = function (dy, x, fSize, origStride, origPad) {
        var maxPositions = this.maxPoolPositions(x, fSize, origStride, origPad);
        var pad = fSize - 1 - origPad;
        var _a = dy.shape, dyRows = _a[0], dyCols = _a[1], depth = _a[2];
        var dyRowsDilated = (dyRows - 1) * origStride + 1;
        var dxColsDilated = (dyCols - 1) * origStride + 1;
        var outputShape = conv_util.computeOutputShape3D([dyRowsDilated, dxColsDilated, depth], fSize, depth, 1, pad);
        var dx = ndarray_1.Array3D.zeros(outputShape);
        for (var d = 0; d < depth; ++d) {
            for (var dxR = 0; dxR < dx.shape[0]; ++dxR) {
                for (var dxC = 0; dxC < dx.shape[1]; ++dxC) {
                    var dyRCorner = dxR - pad;
                    var dyCCorner = dxC - pad;
                    var dotProd = 0;
                    for (var wR = 0; wR < fSize; ++wR) {
                        var dyR = (dyRCorner + wR) / origStride;
                        if (dyR < 0 || dyR >= dyRows || Math.floor(dyR) !== dyR) {
                            continue;
                        }
                        for (var wC = 0; wC < fSize; ++wC) {
                            var dyC = (dyCCorner + wC) / origStride;
                            if (dyC < 0 || dyC >= dyCols || Math.floor(dyC) !== dyC) {
                                continue;
                            }
                            var maxPos = fSize * fSize - 1 - maxPositions.get(dyR, dyC, d);
                            var curPos = wR * fSize + wC;
                            var mask = maxPos === curPos ? 1 : 0;
                            if (mask === 0) {
                                continue;
                            }
                            var pixel = dy.get(dyR, dyC, d);
                            dotProd += pixel * mask;
                        }
                    }
                    dx.set(dotProd, dxR, dxC, d);
                }
            }
        }
        return dx;
    };
    NDArrayMathCPU.prototype.minPoolInternal = function (x, fSize, stride, pad) {
        return this.pool(x, fSize, stride, pad, 'min');
    };
    NDArrayMathCPU.prototype.avgPoolInternal = function (x, fSize, stride, pad) {
        return this.pool(x, fSize, stride, pad, 'avg');
    };
    NDArrayMathCPU.prototype.resizeBilinear3DInternal = function (x, newShape2D, alignCorners) {
        var output = ndarray_1.Array3D.zeros([newShape2D[0], newShape2D[1], x.shape[2]]);
        var effectiveInputSize = alignCorners ? [x.shape[0] - 1, x.shape[1] - 1, x.shape[2]] : x.shape;
        var effectiveOutputSize = alignCorners ?
            [output.shape[0] - 1, output.shape[1] - 1, output.shape[2]] :
            output.shape;
        for (var r = 0; r < output.shape[0]; r++) {
            for (var c = 0; c < output.shape[1]; c++) {
                for (var d = 0; d < output.shape[2]; d++) {
                    var sourceFracRow = (effectiveInputSize[0]) * r / (effectiveOutputSize[0]);
                    var sourceFracCol = (effectiveInputSize[1]) * c / (effectiveOutputSize[1]);
                    var sourceRowFloor = Math.floor(sourceFracRow);
                    var sourceRowCeil = Math.min(x.shape[0] - 1, Math.ceil(sourceFracRow));
                    var sourceColFloor = Math.floor(sourceFracCol);
                    var sourceColCeil = Math.min(x.shape[1] - 1, Math.ceil(sourceFracCol));
                    var topLeft = x.get(sourceRowFloor, sourceColFloor, d);
                    var bottomLeft = x.get(sourceRowCeil, sourceColFloor, d);
                    var topRight = x.get(sourceRowFloor, sourceColCeil, d);
                    var bottomRight = x.get(sourceRowCeil, sourceColCeil, d);
                    var rowFrac = sourceFracRow - sourceRowFloor;
                    var colFrac = sourceFracCol - sourceColFloor;
                    var top_1 = topLeft + (topRight - topLeft) * colFrac;
                    var bottom = bottomLeft + (bottomRight - bottomLeft) * colFrac;
                    var newValue = top_1 + (bottom - top_1) * rowFrac;
                    output.set(newValue, r, c, d);
                }
            }
        }
        return output;
    };
    NDArrayMathCPU.prototype.batchNormalization3DInternal = function (x, mean, variance, varianceEpsilon, scale, offset) {
        if (varianceEpsilon === void 0) { varianceEpsilon = .001; }
        var xValues = x.getValues();
        var meanValues = mean.getValues();
        var varianceValues = variance.getValues();
        var scaleValues = scale ? scale.getValues() : new Float32Array([1]);
        var offsetValues = offset ? offset.getValues() : new Float32Array([0]);
        var outValues = new Float32Array(xValues.length);
        for (var i = 0; i < xValues.length; i++) {
            outValues[i] = offsetValues[i % offsetValues.length] +
                (xValues[i] - meanValues[i % meanValues.length]) *
                    scaleValues[i % scaleValues.length] /
                    Math.sqrt(varianceValues[i % varianceValues.length] + varianceEpsilon);
        }
        return ndarray_1.NDArray.make(x.shape, { values: outValues });
    };
    return NDArrayMathCPU;
}(math_1.NDArrayMath));
exports.NDArrayMathCPU = NDArrayMathCPU;

},{"../math/conv_util":16,"../util":86,"./concat3d_util":15,"./copy2d_util":17,"./math":19,"./ndarray":22}],21:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../util");
var concat3d_util = require("./concat3d_util");
var conv_util = require("./conv_util");
var math_1 = require("./math");
var ndarray = require("./ndarray");
var ndarray_1 = require("./ndarray");
var addscaledmat_gpu = require("./webgl/addscaledmat_gpu");
var addsubmuldiv_gpu = require("./webgl/addsubmuldiv_gpu");
var addsubmuldiv_gpu_1 = require("./webgl/addsubmuldiv_gpu");
var argmaxequals_gpu = require("./webgl/argmaxequals_gpu");
var argminmax_gpu = require("./webgl/argminmax_gpu");
var avg_pool_gpu = require("./webgl/avg_pool_gpu");
var batchnorm_gpu = require("./webgl/batchnorm_gpu");
var concat3d_gpu = require("./webgl/concat3d_gpu");
var conv_backprop_gpu = require("./webgl/conv_backprop_gpu");
var conv_gpu = require("./webgl/conv_gpu");
var copy_gpu = require("./webgl/copy_gpu");
var exp_gpu = require("./webgl/exp_gpu");
var gpgpu_context_1 = require("./webgl/gpgpu_context");
var gpgpu_util = require("./webgl/gpgpu_util");
var log_gpu = require("./webgl/log_gpu");
var logsumexp_gpu = require("./webgl/logsumexp_gpu");
var max_pool_backprop_gpu = require("./webgl/max_pool_backprop_gpu");
var max_pool_gpu = require("./webgl/max_pool_gpu");
var min_pool_gpu = require("./webgl/min_pool_gpu");
var minmax_gpu = require("./webgl/minmax_gpu");
var mulmat_gpu = require("./webgl/mulmat_gpu");
var neg_gpu = require("./webgl/neg_gpu");
var pool_gpu = require("./webgl/pool_gpu");
var reducesum_gpu = require("./webgl/reducesum_gpu");
var relu_gpu = require("./webgl/relu_gpu");
var reshape_gpu = require("./webgl/reshape_gpu");
var resize_bilinear_gpu = require("./webgl/resize_bilinear_gpu");
var shader_compiler = require("./webgl/shader_compiler");
var sigmoid_gpu = require("./webgl/sigmoid_gpu");
var step_gpu = require("./webgl/step_gpu");
var texture_manager_1 = require("./webgl/texture_manager");
var trig_gpu = require("./webgl/trig_gpu");
var webgl_util = require("./webgl/webgl_util");
var ARGMAX_PROG = 'argmax';
var ARGMAX_EQUALS_PROG = 'argmaxequals';
var ARGMIN_PROG = 'argmin';
var BATCHNORM_PROG = 'batchnorm';
var COPY_PROG = 'copy';
var CONCAT_PROG = 'concat';
var ADD_SCALED_MAT_PROG = 'addscaledmat';
var MATMUL_PROG = 'matmul';
var RELU_PROG = 'relu';
var TANH_PROG = 'tanh';
var SIN_PROG = 'sin';
var SIGMOID_PROG = 'sigmoid';
var MAX_PROG = 'max';
var MIN_PROG = 'min';
var NEG_PROG = 'neg';
var EXP_PROG = 'exp';
var LOG_PROG = 'log';
var SUM_PROG = 'sum';
var STEP_PROG = 'step';
var LOGSUMEXP_PROG = 'logsumexp';
var RESHAPE_PROG = 'reshape';
var ADD_SUM_MUL_DIV_PROG = 'addsummuldiv';
var CONV2D_PROG = 'conv';
var CONV2D_TRANSPOSE_PROG = 'conv_transpose';
var CONV2D_DERW_PROG = 'conv_derw';
var CONV2D_DERB_PROG = 'conv_derb';
var MAX_POOL_PROG = 'maxpool';
var MAX_POOL_POSITIONS_PROG = 'maxpool_posn';
var MAX_POOL_BACKPROP_PROG = 'maxpool_backprop';
var MIN_POOL_PROG = 'minpool';
var AVG_POOL_PROG = 'avgpool';
var RESIZE_BILINEAR_PROG = 'resizebilin';
function makeCopyProgramName(sourceShapeRowCol, sourceSizeRowCol, destSizeRowCol) {
    var shapeName = sourceShapeRowCol[0] + "_" + sourceShapeRowCol[1];
    var srcSizeName = sourceSizeRowCol[0] + "_" + sourceSizeRowCol[1];
    var dstSizeName = destSizeRowCol[0] + "_" + destSizeRowCol[1];
    return COPY_PROG + "_" + shapeName + "_" + srcSizeName + "_" + dstSizeName;
}
var NDArrayMathGPU = (function (_super) {
    __extends(NDArrayMathGPU, _super);
    function NDArrayMathGPU(gpgpu, safeMode) {
        if (safeMode === void 0) { safeMode = true; }
        var _this = _super.call(this, safeMode) || this;
        _this.programCache = {};
        if (gpgpu == null) {
            var gl = gpgpu_util.createWebGLContext();
            _this.gpgpu = new gpgpu_context_1.GPGPUContext(gl);
            _this.gpgpuCreatedLocally = true;
        }
        else {
            _this.gpgpu = gpgpu;
            _this.gpgpuCreatedLocally = false;
        }
        _this.textureManager = new texture_manager_1.TextureManager(_this.gpgpu);
        ndarray.initializeGPU(_this.gpgpu, _this.textureManager);
        return _this;
    }
    NDArrayMathGPU.prototype.getGPGPUContext = function () {
        return this.gpgpu;
    };
    NDArrayMathGPU.prototype.cloneInternal = function (ndarray) {
        var textureShapeRC = ndarray.getTextureShapeRC();
        var program = this.getAndSaveProgram(makeCopyProgramName(textureShapeRC, textureShapeRC, textureShapeRC), function () { return copy_gpu.getFragmentShaderSource(textureShapeRC, textureShapeRC, textureShapeRC); });
        var resultTexture = this.textureManager.acquireTexture(textureShapeRC);
        copy_gpu.copy(this.gpgpu, program, ndarray.getTexture(), textureShapeRC, [0, 0], textureShapeRC, resultTexture, textureShapeRC, [0, 0], textureShapeRC);
        return ndarray_1.NDArray.make(ndarray.shape, { texture: resultTexture, textureShapeRC: textureShapeRC });
    };
    NDArrayMathGPU.prototype.reshapeInternal = function (ndarray, newShape) {
        var newTexShape;
        switch (newShape.length) {
            case 0:
                newTexShape = [1, 1];
                break;
            case 1:
                newTexShape = [newShape[0], 1];
                break;
            case 2:
                newTexShape = [newShape[0], newShape[1]];
                break;
            case 3:
                newTexShape = [newShape[0], newShape[1] * newShape[2]];
                break;
            default:
                throw Error("Reshapes into " + newShape.length + "-dim ndarray is not yet " +
                    "supported on GPU");
        }
        var actualTexShape = ndarray.getTextureShapeRC(newTexShape);
        var clonedArray;
        if (!util.arraysEqual(actualTexShape, newTexShape)) {
            clonedArray = this.reshapeTexture(ndarray, newTexShape);
        }
        else {
            clonedArray = this.cloneInternal(ndarray);
        }
        return clonedArray.reshape(newShape);
    };
    NDArrayMathGPU.prototype.slice2DInternal = function (input, beginRowCol, sizeRowCol) {
        var result = ndarray_1.NDArray.make(sizeRowCol, {
            texture: this.textureManager.acquireTexture(sizeRowCol),
            textureShapeRC: sizeRowCol
        });
        this.copy2DInternal(input, beginRowCol, sizeRowCol, result, [0, 0], sizeRowCol);
        return result;
    };
    NDArrayMathGPU.prototype.copy2DInternal = function (source, sourceBeginRowCol, sourceSizeRowCol, dest, destBeginRowCol, destSizeRowCol) {
        var sourceShapeRC = source.getTextureShapeRC();
        var destShapeRC = dest.getTextureShapeRC();
        var program = this.getAndSaveProgram(makeCopyProgramName(sourceShapeRC, sourceSizeRowCol, destSizeRowCol), function () { return copy_gpu.getFragmentShaderSource(sourceShapeRC, sourceSizeRowCol, destSizeRowCol); });
        copy_gpu.copy(this.gpgpu, program, source.getTexture(), sourceShapeRC, sourceBeginRowCol, sourceSizeRowCol, dest.getTexture(), destShapeRC, destBeginRowCol, destSizeRowCol);
    };
    NDArrayMathGPU.prototype.concat3DInternal = function (x1, x2, axis) {
        var x1TexShapeRC = conv_util.computeTexShapeFrom3D(x1.shape);
        var x2TexShapeRC = conv_util.computeTexShapeFrom3D(x2.shape);
        var actualX1TexShape = x1.getTextureShapeRC(x1TexShapeRC);
        var cleanupX1 = false;
        if (!util.arraysEqual(actualX1TexShape, x1TexShapeRC)) {
            x1 = this.reshapeTexture(x1, x1TexShapeRC);
            cleanupX1 = true;
        }
        var actualX2TexShape = x2.getTextureShapeRC(x2TexShapeRC);
        var cleanupX2 = false;
        if (!util.arraysEqual(actualX2TexShape, x2TexShapeRC)) {
            x2 = this.reshapeTexture(x2, x2TexShapeRC);
            cleanupX2 = true;
        }
        var resultShapeRCD = concat3d_util.computeConcat3DOutputShape(x1.shape, x2.shape, axis);
        var program = this.getAndSaveProgram(CONCAT_PROG + "_" + x1.shape + "_" + x2.shape + "_" + axis, function () { return concat3d_gpu.getFragmentShaderSource(x1.shape, x2.shape, resultShapeRCD, axis); });
        var resultTexShape = conv_util.computeTexShapeFrom3D(resultShapeRCD);
        var resultTex = this.textureManager.acquireTexture(resultTexShape);
        concat3d_gpu.concat3D(this.gpgpu, program, x1.getTexture(), x2.getTexture(), resultTex, resultTexShape);
        if (cleanupX1) {
            x1.dispose();
        }
        if (cleanupX2) {
            x2.dispose();
        }
        return ndarray_1.NDArray.make(resultShapeRCD, { texture: resultTex, textureShapeRC: resultTexShape });
    };
    NDArrayMathGPU.prototype.scalarPlusArrayInternal = function (c, a) {
        return this.addSubMulDiv(c, a, a.shape, addsubmuldiv_gpu_1.OperandType.SCALAR, '+', addsubmuldiv_gpu_1.OperandType.MATRIX);
    };
    NDArrayMathGPU.prototype.arrayMinusScalarInternal = function (a, c) {
        return this.addSubMulDiv(a, c, a.shape, addsubmuldiv_gpu_1.OperandType.MATRIX, '-', addsubmuldiv_gpu_1.OperandType.SCALAR);
    };
    NDArrayMathGPU.prototype.scalarMinusArrayInternal = function (c, a) {
        return this.addSubMulDiv(c, a, a.shape, addsubmuldiv_gpu_1.OperandType.SCALAR, '-', addsubmuldiv_gpu_1.OperandType.MATRIX);
    };
    NDArrayMathGPU.prototype.scaledArrayAddInternal = function (c1, a, c2, b) {
        var cleanupB = false;
        if (!this.doGPUShapesMatch(a, b)) {
            b = this.reshapeTexture(b, a.getTextureShapeRC());
            cleanupB = true;
        }
        var program = this.getAndSaveProgram(ADD_SCALED_MAT_PROG, function () { return addscaledmat_gpu.getFragmentShaderSource(); });
        var textureShapeRC = a.getTextureShapeRC();
        var resultTexture = this.textureManager.acquireTexture(textureShapeRC);
        addscaledmat_gpu.addScaledMatrices(this.gpgpu, program, a.getTexture(), b.getTexture(), textureShapeRC[0], textureShapeRC[1], c1.getTexture(), c2.getTexture(), resultTexture);
        if (cleanupB) {
            b.dispose();
        }
        return ndarray_1.NDArray.make(a.shape, { texture: resultTexture, textureShapeRC: textureShapeRC });
    };
    NDArrayMathGPU.prototype.scalarTimesArrayInternal = function (c, a) {
        return this.addSubMulDiv(c, a, a.shape, addsubmuldiv_gpu_1.OperandType.SCALAR, '*', addsubmuldiv_gpu_1.OperandType.MATRIX);
    };
    NDArrayMathGPU.prototype.negInternal = function (a) {
        var program = this.getAndSaveProgram(NEG_PROG, function () { return neg_gpu.getFragmentShaderSource(); });
        var textureShapeRC = a.getTextureShapeRC();
        var resultTexture = this.textureManager.acquireTexture(textureShapeRC);
        neg_gpu.neg(this.gpgpu, program, a.getTexture(), textureShapeRC[0], textureShapeRC[1], resultTexture);
        return ndarray_1.NDArray.make(a.shape, { texture: resultTexture, textureShapeRC: textureShapeRC });
    };
    NDArrayMathGPU.prototype.reshapeTexture = function (a, newTextureShape) {
        var aTexShape = a.getTextureShapeRC();
        var program = this.getAndSaveProgram(RESHAPE_PROG, function () { return reshape_gpu.getFragmentShaderSource(); });
        var resultTexture = this.textureManager.acquireTexture(newTextureShape);
        reshape_gpu.reshape(this.gpgpu, program, a.getTexture(), aTexShape[0], aTexShape[1], resultTexture, newTextureShape[0], newTextureShape[1]);
        return ndarray_1.NDArray.make(a.shape, { texture: resultTexture, textureShapeRC: newTextureShape });
    };
    NDArrayMathGPU.prototype.matMulInternal = function (a, b, aOrientation, bOrientation) {
        var sharedDim = (aOrientation === math_1.MatrixOrientation.REGULAR) ? a.shape[1] : a.shape[0];
        var outerShapeA = (aOrientation === math_1.MatrixOrientation.REGULAR) ? a.shape[0] : a.shape[1];
        var outerShapeB = (bOrientation === math_1.MatrixOrientation.REGULAR) ? b.shape[1] : b.shape[0];
        var outShape = [outerShapeA, outerShapeB];
        var outTexShape = webgl_util.getTextureShapeFromLogicalShape(this.gpgpu.gl, outShape);
        var outTexture = this.textureManager.acquireTexture(outTexShape);
        var out = new ndarray_1.Array2D(outShape, { texture: outTexture, textureShapeRC: outTexShape });
        var key = shader_compiler.makeShaderKey([a, b], out);
        var program = this.getAndSaveProgram(MATMUL_PROG + "_" + key + "_" + aOrientation + "_" + bOrientation, function () { return mulmat_gpu.getFragmentShader(a, b, out, aOrientation, bOrientation); });
        mulmat_gpu.multiplyMatrix(this.gpgpu, program, a.getTexture(), b.getTexture(), outTexture, outTexShape);
        return out;
    };
    NDArrayMathGPU.prototype.elementWiseMulInternal = function (a, b) {
        return this.addSubMulDiv(a, b, a.shape, addsubmuldiv_gpu_1.OperandType.MATRIX, '*', addsubmuldiv_gpu_1.OperandType.MATRIX);
    };
    NDArrayMathGPU.prototype.elementWiseMulBroadcastInternal = function (a, b) {
        throw new Error('Not yet implemented!');
    };
    NDArrayMathGPU.prototype.batchNormalization3DInternal = function (x, mean, variance, varianceEpsilon, scale, offset) {
        var xTexShape = x.getTextureShapeRC();
        var cleanupMean = false;
        var preferredMeanTexShape = mean.rank === 1 ? [1, mean.size] : xTexShape;
        var meanTexShape = mean.getTextureShapeRC(preferredMeanTexShape);
        if (!util.arraysEqual(meanTexShape, preferredMeanTexShape)) {
            mean = this.reshapeTexture(mean, preferredMeanTexShape);
            meanTexShape = preferredMeanTexShape;
            cleanupMean = true;
        }
        var cleanupVariance = false;
        var preferredVarianceTexShape = variance.rank === 1 ? [1, variance.size] : xTexShape;
        var varianceTexShape = variance.getTextureShapeRC(preferredMeanTexShape);
        if (!util.arraysEqual(varianceTexShape, preferredVarianceTexShape)) {
            variance = this.reshapeTexture(variance, preferredVarianceTexShape);
            varianceTexShape = preferredVarianceTexShape;
            cleanupVariance = true;
        }
        var scaleTexShape = null;
        var cleanupScale = false;
        if (scale != null) {
            var preferredScaleTexShape = scale.rank === 1 ? [1, scale.size] : xTexShape;
            scaleTexShape = scale.getTextureShapeRC(preferredScaleTexShape);
            if (!util.arraysEqual(scaleTexShape, preferredScaleTexShape)) {
                scale = this.reshapeTexture(scale, preferredScaleTexShape);
                scaleTexShape = preferredScaleTexShape;
                cleanupScale = true;
            }
        }
        var offsetTexShape = null;
        var cleanupOffset = false;
        if (offset != null) {
            var preferredOffsetTexShape = offset.rank === 1 ? [1, offset.size] : xTexShape;
            offsetTexShape = offset.getTextureShapeRC(preferredOffsetTexShape);
            if (!util.arraysEqual(offsetTexShape, preferredOffsetTexShape)) {
                offset = this.reshapeTexture(offset, preferredOffsetTexShape);
                offsetTexShape = preferredOffsetTexShape;
                cleanupOffset = true;
            }
        }
        var resultTexShape = x.getTextureShapeRC();
        var program = this.getAndSaveProgram(BATCHNORM_PROG + "_" + xTexShape + "_" + meanTexShape + "_" + varianceTexShape + "_" +
            (scaleTexShape + "_" + offsetTexShape + "_" + varianceEpsilon), function () { return batchnorm_gpu.getFragmentShaderSource(xTexShape, meanTexShape, varianceTexShape, offsetTexShape, scaleTexShape, varianceEpsilon); });
        var resultTexture = this.textureManager.acquireTexture(resultTexShape);
        batchnorm_gpu.batchNormalization(this.gpgpu, program, x.getTexture(), xTexShape, mean.getTexture(), meanTexShape, variance.getTexture(), varianceTexShape, offset != null ? offset.getTexture() : null, offset != null ? offsetTexShape : null, scale != null ? scale.getTexture() : null, scale != null ? scaleTexShape : null, resultTexture, resultTexShape);
        if (cleanupMean) {
            mean.dispose();
        }
        if (cleanupVariance) {
            variance.dispose();
        }
        if (cleanupScale) {
            scale.dispose();
        }
        if (cleanupOffset) {
            offset.dispose();
        }
        return ndarray_1.NDArray.make(x.shape, { texture: resultTexture, textureShapeRC: resultTexShape });
    };
    NDArrayMathGPU.prototype.switchDimInternal = function (a, newDim) {
        throw new Error('Not yet implemented!');
    };
    NDArrayMathGPU.prototype.sumInternal = function (ndarray) {
        var textureShapeRC = ndarray.getTextureShapeRC();
        var numRows = textureShapeRC[0], numColumns = textureShapeRC[1];
        var program = this.getAndSaveProgram(SUM_PROG + "_" + numRows + "_" + numColumns, function () { return reducesum_gpu.getFragmentShaderSource(numRows, numColumns); });
        var resultTexture = this.textureManager.acquireTexture([1, 1]);
        reducesum_gpu.reduceSum(this.gpgpu, program, ndarray.getTexture(), numRows, numColumns, resultTexture);
        return new ndarray_1.Scalar({ texture: resultTexture });
    };
    NDArrayMathGPU.prototype.argMinInternal = function (ndarray) {
        var textureShapeRC = ndarray.getTextureShapeRC();
        var numRows = textureShapeRC[0], numColumns = textureShapeRC[1];
        var program = this.getAndSaveProgram(ARGMIN_PROG + "_" + numRows + "_" + numColumns, function () { return argminmax_gpu.getArgMinFragmentShaderSource(numRows, numColumns); });
        var resultTexture = this.textureManager.acquireTexture([1, 1]);
        argminmax_gpu.argMinMax(this.gpgpu, program, ndarray.getTexture(), numRows, numColumns, resultTexture);
        return new ndarray_1.Scalar({ texture: resultTexture });
    };
    NDArrayMathGPU.prototype.argMaxInternal = function (ndarray) {
        var textureShapeRC = ndarray.getTextureShapeRC();
        var numRows = textureShapeRC[0], numColumns = textureShapeRC[1];
        var program = this.getAndSaveProgram(ARGMAX_PROG + "_" + numRows + "_" + numColumns, function () { return argminmax_gpu.getArgMaxFragmentShaderSource(numRows, numColumns); });
        var resultTexture = this.textureManager.acquireTexture([1, 1]);
        argminmax_gpu.argMinMax(this.gpgpu, program, ndarray.getTexture(), numRows, numColumns, resultTexture);
        return new ndarray_1.Scalar({ texture: resultTexture });
    };
    NDArrayMathGPU.prototype.argMaxEqualsInternal = function (x1, x2) {
        var actualX1TexShape = x1.getTextureShapeRC();
        var actualX2TexShape = x2.getTextureShapeRC();
        var cleanupX2 = false;
        if (!util.arraysEqual(actualX1TexShape, actualX2TexShape)) {
            x2 = this.reshapeTexture(x2, actualX1TexShape);
            cleanupX2 = true;
        }
        var textureShapeRC = x1.getTextureShapeRC();
        var numRows = textureShapeRC[0], numColumns = textureShapeRC[1];
        var program = this.getAndSaveProgram(ARGMAX_EQUALS_PROG + "_" + numRows + "_" + numColumns, function () { return argmaxequals_gpu.getArgMaxEqualsFragmentShaderSource(numRows, numColumns); });
        var resultTexture = this.textureManager.acquireTexture([1, 1]);
        argmaxequals_gpu.argMaxEquals(this.gpgpu, program, x1.getTexture(), x2.getTexture(), numRows, numColumns, resultTexture);
        if (cleanupX2) {
            x2.dispose();
        }
        return new ndarray_1.Scalar({ texture: resultTexture });
    };
    NDArrayMathGPU.prototype.topKInternal = function (ndarray, k) {
        throw new Error('topK GPU not yet implemented!');
    };
    NDArrayMathGPU.prototype.minInternal = function (ndarray) {
        var textureShapeRC = ndarray.getTextureShapeRC();
        var numRows = textureShapeRC[0], numColumns = textureShapeRC[1];
        var program = this.getAndSaveProgram(MIN_PROG + "_" + numRows + "_" + numColumns, function () { return minmax_gpu.getMinFragmentShaderSource(numRows, numColumns); });
        var resultTexture = this.textureManager.acquireTexture([1, 1]);
        minmax_gpu.minMax(this.gpgpu, program, ndarray.getTexture(), numRows, numColumns, resultTexture);
        return new ndarray_1.Scalar({ texture: resultTexture });
    };
    NDArrayMathGPU.prototype.maxInternal = function (ndarray) {
        var textureShapeRC = ndarray.getTextureShapeRC();
        var numRows = textureShapeRC[0], numColumns = textureShapeRC[1];
        var program = this.getAndSaveProgram(MAX_PROG + "_" + numRows + "_" + numColumns, function () { return minmax_gpu.getMaxFragmentShaderSource(numRows, numColumns); });
        var resultTexture = this.textureManager.acquireTexture([1, 1]);
        minmax_gpu.minMax(this.gpgpu, program, ndarray.getTexture(), numRows, numColumns, resultTexture);
        return new ndarray_1.Scalar({ texture: resultTexture });
    };
    NDArrayMathGPU.prototype.divideInternal = function (a, b) {
        return this.addSubMulDiv(a, b, a.shape, addsubmuldiv_gpu_1.OperandType.MATRIX, '/', addsubmuldiv_gpu_1.OperandType.MATRIX);
    };
    NDArrayMathGPU.prototype.scalarDividedByArrayInternal = function (c, a) {
        return this.addSubMulDiv(c, a, a.shape, addsubmuldiv_gpu_1.OperandType.SCALAR, '/', addsubmuldiv_gpu_1.OperandType.MATRIX);
    };
    NDArrayMathGPU.prototype.arrayDividedByScalarInternal = function (a, c) {
        return this.addSubMulDiv(a, c, a.shape, addsubmuldiv_gpu_1.OperandType.MATRIX, '/', addsubmuldiv_gpu_1.OperandType.SCALAR);
    };
    NDArrayMathGPU.prototype.addInternal = function (a, b) {
        return this.addSubMulDiv(a, b, a.shape, addsubmuldiv_gpu_1.OperandType.MATRIX, '+', addsubmuldiv_gpu_1.OperandType.MATRIX);
    };
    NDArrayMathGPU.prototype.subInternal = function (a, b) {
        return this.addSubMulDiv(a, b, a.shape, addsubmuldiv_gpu_1.OperandType.MATRIX, '-', addsubmuldiv_gpu_1.OperandType.MATRIX);
    };
    NDArrayMathGPU.prototype.logSumExpInternal = function (ndarray) {
        var _a = ndarray.getTextureShapeRC(), numRows = _a[0], numColumns = _a[1];
        var program = this.getAndSaveProgram(LOGSUMEXP_PROG + "_" + numRows + "_" + numColumns, function () { return logsumexp_gpu.getFragmentShaderSource(numRows, numColumns); });
        var result = new ndarray_1.Scalar({ texture: this.textureManager.acquireTexture([1, 1]) });
        reducesum_gpu.reduceSum(this.gpgpu, program, ndarray.getTexture(), numRows, numColumns, result.getTexture());
        return result;
    };
    NDArrayMathGPU.prototype.expInternal = function (ndarray) {
        var program = this.getAndSaveProgram(EXP_PROG, function () { return exp_gpu.getFragmentShaderSource(); });
        var textureShapeRC = ndarray.getTextureShapeRC();
        var resultTexture = this.textureManager.acquireTexture(textureShapeRC);
        exp_gpu.exp(this.gpgpu, program, ndarray.getTexture(), textureShapeRC[0], textureShapeRC[1], resultTexture);
        return ndarray_1.NDArray.make(ndarray.shape, { texture: resultTexture, textureShapeRC: textureShapeRC });
    };
    NDArrayMathGPU.prototype.logInternal = function (ndarray) {
        var program = this.getAndSaveProgram(LOG_PROG, function () { return log_gpu.getFragmentShaderSource(); });
        var textureShapeRC = ndarray.getTextureShapeRC();
        var resultTexture = this.textureManager.acquireTexture(textureShapeRC);
        log_gpu.log(this.gpgpu, program, ndarray.getTexture(), textureShapeRC[0], textureShapeRC[1], resultTexture);
        return ndarray_1.NDArray.make(ndarray.shape, { texture: resultTexture, textureShapeRC: textureShapeRC });
    };
    NDArrayMathGPU.prototype.reluInternal = function (ndarray) {
        var program = this.getAndSaveProgram(RELU_PROG, function () { return relu_gpu.getFragmentShaderSource(); });
        var textureShapeRC = ndarray.getTextureShapeRC();
        var resultTexture = this.textureManager.acquireTexture(textureShapeRC);
        relu_gpu.relu(this.gpgpu, program, ndarray.getTexture(), textureShapeRC[0], textureShapeRC[1], resultTexture);
        return ndarray_1.NDArray.make(ndarray.shape, { texture: resultTexture, textureShapeRC: textureShapeRC });
    };
    NDArrayMathGPU.prototype.sigmoidInternal = function (ndarray) {
        var program = this.getAndSaveProgram(SIGMOID_PROG, function () { return sigmoid_gpu.getSigmoidFragmentShaderSource(); });
        var textureShapeRC = ndarray.getTextureShapeRC();
        var resultTexture = this.textureManager.acquireTexture(textureShapeRC);
        sigmoid_gpu.sigmoid(this.gpgpu, program, ndarray.getTexture(), textureShapeRC[0], textureShapeRC[1], resultTexture);
        return ndarray_1.NDArray.make(ndarray.shape, { texture: resultTexture, textureShapeRC: textureShapeRC });
    };
    NDArrayMathGPU.prototype.tanhInternal = function (ndarray) {
        var program = this.getAndSaveProgram(TANH_PROG, function () { return trig_gpu.getTanhFragmentShaderSource(); });
        var textureShapeRC = ndarray.getTextureShapeRC();
        var resultTexture = this.textureManager.acquireTexture(textureShapeRC);
        trig_gpu.tanh(this.gpgpu, program, ndarray.getTexture(), textureShapeRC[0], textureShapeRC[1], resultTexture);
        return ndarray_1.NDArray.make(ndarray.shape, { texture: resultTexture, textureShapeRC: textureShapeRC });
    };
    NDArrayMathGPU.prototype.sinInternal = function (ndarray) {
        var program = this.getAndSaveProgram(SIN_PROG, function () { return trig_gpu.getSinFragmentShaderSource(); });
        var textureShapeRC = ndarray.getTextureShapeRC();
        var resultTexture = this.textureManager.acquireTexture(textureShapeRC);
        trig_gpu.sin(this.gpgpu, program, ndarray.getTexture(), textureShapeRC[0], textureShapeRC[1], resultTexture);
        return ndarray_1.NDArray.make(ndarray.shape, { texture: resultTexture, textureShapeRC: textureShapeRC });
    };
    NDArrayMathGPU.prototype.stepInternal = function (ndarray) {
        var program = this.getAndSaveProgram(STEP_PROG, function () { return step_gpu.getFragmentShaderSource(); });
        var textureShapeRC = ndarray.getTextureShapeRC();
        var resultTexture = this.textureManager.acquireTexture(textureShapeRC);
        step_gpu.step(this.gpgpu, program, ndarray.getTexture(), textureShapeRC[0], textureShapeRC[1], resultTexture);
        return ndarray_1.NDArray.make(ndarray.shape, { texture: resultTexture, textureShapeRC: textureShapeRC });
    };
    NDArrayMathGPU.prototype.conv2dInternal = function (x, weights, biases, stride, zeroPad) {
        var fieldSize = weights.shape[0];
        var inputDepth = weights.shape[2];
        var outputDepth = weights.shape[3];
        var progKey = [
            CONV2D_PROG, x.shape, outputDepth, fieldSize, stride, biases != null
        ].join('_');
        var program = this.getAndSaveProgram(progKey, function () {
            return conv_gpu.getFragmentShaderSource(x.shape, outputDepth, fieldSize, stride, zeroPad, biases != null);
        });
        var xTexShape = conv_util.computeTexShapeFrom3D(x.shape);
        var wTexShape = conv_util.computeWeightsTexShape(inputDepth, outputDepth, fieldSize);
        var biasTexShape = conv_util.computeBiasesTexShape(outputDepth);
        var actualXTexShape = x.getTextureShapeRC(xTexShape);
        var cleanupX = false;
        if (!util.arraysEqual(actualXTexShape, xTexShape)) {
            x = this.reshapeTexture(x, xTexShape);
            cleanupX = true;
        }
        var cleanupW = false;
        var actualWTexShape = weights.getTextureShapeRC(wTexShape);
        if (!util.arraysEqual(actualWTexShape, wTexShape)) {
            weights = this.reshapeTexture(weights, wTexShape);
            cleanupW = true;
        }
        var cleanupB = false;
        if (biases != null) {
            var actualBTexShape = biases.getTextureShapeRC(biasTexShape);
            if (!util.arraysEqual(actualBTexShape, biasTexShape)) {
                biases = this.reshapeTexture(biases, biasTexShape);
                cleanupB = true;
            }
        }
        var resultShape = conv_util.computeOutputShape3D(x.shape, fieldSize, outputDepth, stride, zeroPad);
        var resultTexShape = conv_util.computeTexShapeFrom3D(resultShape);
        var resultTex = this.textureManager.acquireTexture(resultTexShape);
        conv_gpu.convolve(this.gpgpu, program, x.getTexture(), weights.getTexture(), biases != null ? biases.getTexture() : null, resultTex, resultTexShape);
        if (cleanupX) {
            x.dispose();
        }
        if (cleanupW) {
            weights.dispose();
        }
        if (cleanupB && biases != null) {
            biases.dispose();
        }
        return ndarray_1.NDArray.make(resultShape, { texture: resultTex, textureShapeRC: resultTexShape });
    };
    NDArrayMathGPU.prototype.conv2dBackPropInternal = function (x, dy, weights, stride, pad) {
        var fSize = weights.shape[0];
        var inputDepth = weights.shape[2];
        var outputDepth = weights.shape[3];
        var xTexShape = conv_util.computeTexShapeFrom3D(x.shape);
        var wTexShape = conv_util.computeWeightsTexShape(inputDepth, outputDepth, fSize);
        var yTexShape = conv_util.computeTexShapeFrom3D(dy.shape);
        var cleanupX = false;
        var actualXTexShape = x.getTextureShapeRC(xTexShape);
        if (!util.arraysEqual(actualXTexShape, xTexShape)) {
            x = this.reshapeTexture(x, xTexShape);
            cleanupX = true;
        }
        var cleanupW = false;
        var actualWTexShape = weights.getTextureShapeRC(wTexShape);
        if (!util.arraysEqual(actualWTexShape, wTexShape)) {
            weights = this.reshapeTexture(weights, wTexShape);
            cleanupW = true;
        }
        var cleanupY = false;
        var actualYTexShape = dy.getTextureShapeRC(yTexShape);
        if (!util.arraysEqual(actualYTexShape, yTexShape)) {
            dy = this.reshapeTexture(dy, yTexShape);
            cleanupY = true;
        }
        var dw = this.conv2dDerWeights(x, dy, fSize, stride, pad);
        var db = this.conv2dDerBias(dy);
        var dx = this.conv2dTransposeInternal(dy, weights, null, stride, pad);
        if (cleanupX) {
            x.dispose();
        }
        if (cleanupW) {
            weights.dispose();
        }
        if (cleanupY) {
            dy.dispose();
        }
        return { dx: dx, db: db, dw: dw };
    };
    NDArrayMathGPU.prototype.conv2dTransposeInternal = function (x, weights, biases, origStride, origPad) {
        var origInputDepth = weights.shape[2];
        var origOutputDepth = weights.shape[3];
        var fieldSize = weights.shape[0];
        var progKey = [
            CONV2D_TRANSPOSE_PROG, x.shape, fieldSize, origInputDepth, origStride,
            origPad, biases != null
        ].join('_');
        var program = this.getAndSaveProgram(progKey, function () {
            return conv_backprop_gpu.getFragmentShaderConvTransposeSource(x.shape, fieldSize, origInputDepth, origStride, origPad, biases != null);
        });
        var xTexShape = conv_util.computeTexShapeFrom3D(x.shape);
        var wTexShape = conv_util.computeWeightsTexShape(origInputDepth, origOutputDepth, fieldSize);
        var biasTexShape = conv_util.computeBiasesTexShape(origInputDepth);
        var actualXTexShape = x.getTextureShapeRC(xTexShape);
        var cleanupX = false;
        if (!util.arraysEqual(actualXTexShape, xTexShape)) {
            x = this.reshapeTexture(x, xTexShape);
            cleanupX = true;
        }
        var cleanupW = false;
        var actualWTexShape = weights.getTextureShapeRC(wTexShape);
        if (!util.arraysEqual(actualWTexShape, wTexShape)) {
            weights = this.reshapeTexture(weights, wTexShape);
            cleanupW = true;
        }
        var cleanupB = false;
        if (biases != null) {
            var actualBiasTexShape = biases.getTextureShapeRC(biasTexShape);
            if (!util.arraysEqual(actualBiasTexShape, biasTexShape)) {
                biases = this.reshapeTexture(biases, biasTexShape);
                cleanupB = true;
            }
        }
        var dilatedRC = conv_util.computeDilatedRC([x.shape[0], x.shape[1]], origStride);
        var pad = fieldSize - 1 - origPad;
        var resultShape = conv_util.computeOutputShape3D([dilatedRC[0], dilatedRC[1], origOutputDepth], fieldSize, origInputDepth, 1, pad);
        var resultTexShape = conv_util.computeTexShapeFrom3D(resultShape);
        var resultTex = this.textureManager.acquireTexture(resultTexShape);
        conv_backprop_gpu.convTranspose(this.gpgpu, program, x.getTexture(), weights.getTexture(), biases != null ? biases.getTexture() : null, resultTex, resultTexShape);
        if (cleanupX) {
            x.dispose();
        }
        if (cleanupW) {
            weights.dispose();
        }
        if (cleanupB) {
            biases.dispose();
        }
        return ndarray_1.NDArray.make(resultShape, { texture: resultTex, textureShapeRC: resultTexShape });
    };
    NDArrayMathGPU.prototype.conv2dDerWeights = function (x, dY, fSize, stride, zeroPad) {
        var inputDepth = x.shape[2];
        var outputDepth = dY.shape[2];
        var progKey = [
            CONV2D_DERW_PROG, x.shape, fSize, outputDepth, stride, zeroPad
        ].join('_');
        var program = this.getAndSaveProgram(progKey, function () {
            return conv_backprop_gpu.getFragmentShaderDerWeightsSource(x.shape, fSize, outputDepth, stride, zeroPad);
        });
        var xTexShape = conv_util.computeTexShapeFrom3D(x.shape);
        var yShape = conv_util.computeOutputShape3D(x.shape, fSize, outputDepth, stride, zeroPad);
        var yTexShape = conv_util.computeTexShapeFrom3D(yShape);
        var actualXTexShape = x.getTextureShapeRC(xTexShape);
        var cleanupX = false;
        if (!util.arraysEqual(actualXTexShape, xTexShape)) {
            x = this.reshapeTexture(x, xTexShape);
            cleanupX = true;
        }
        var cleanupY = false;
        var actualYTexShape = dY.getTextureShapeRC(yTexShape);
        if (!util.arraysEqual(actualYTexShape, yTexShape)) {
            dY = this.reshapeTexture(dY, yTexShape);
            cleanupY = true;
        }
        var resultTexShape = conv_util.computeWeightsTexShape(inputDepth, outputDepth, fSize);
        var resultTex = this.textureManager.acquireTexture(resultTexShape);
        conv_backprop_gpu.derWeights(this.gpgpu, program, x.getTexture(), dY.getTexture(), resultTex, resultTexShape);
        if (cleanupX) {
            x.dispose();
        }
        if (cleanupY) {
            dY.dispose();
        }
        var weightsShape = conv_util.computeWeightsShape4D(inputDepth, outputDepth, fSize);
        return ndarray_1.NDArray.make(weightsShape, { texture: resultTex, textureShapeRC: resultTexShape });
    };
    NDArrayMathGPU.prototype.conv2dDerBias = function (dY) {
        var outputDepth = dY.shape[2];
        var progKey = [CONV2D_DERB_PROG, dY.shape].join('_');
        var program = this.getAndSaveProgram(progKey, function () {
            return conv_backprop_gpu.getFragmentShaderDerBiasSource(dY.shape);
        });
        var yTexShape = conv_util.computeTexShapeFrom3D(dY.shape);
        var cleanupY = false;
        var actualYTexShape = dY.getTextureShapeRC(yTexShape);
        if (!util.arraysEqual(actualYTexShape, yTexShape)) {
            dY = this.reshapeTexture(dY, yTexShape);
            cleanupY = true;
        }
        var resultTexShape = conv_util.computeBiasesTexShape(outputDepth);
        var resultTex = this.textureManager.acquireTexture(resultTexShape);
        conv_backprop_gpu.derBias(this.gpgpu, program, dY.getTexture(), resultTex, resultTexShape);
        if (cleanupY) {
            dY.dispose();
        }
        return ndarray_1.NDArray.make([outputDepth], { texture: resultTex, textureShapeRC: resultTexShape });
    };
    NDArrayMathGPU.prototype.pool = function (program, x, fSize, stride, pad) {
        var xTexShape = conv_util.computeTexShapeFrom3D(x.shape);
        var actualXTexShape = x.getTextureShapeRC(xTexShape);
        var cleanupX = false;
        if (!util.arraysEqual(actualXTexShape, xTexShape)) {
            x = this.reshapeTexture(x, xTexShape);
            cleanupX = true;
        }
        var resultShape = conv_util.computeOutputShape3D(x.shape, fSize, x.shape[2], stride, pad);
        var resultTexShape = conv_util.computeTexShapeFrom3D(resultShape);
        var poolResultTex = this.textureManager.acquireTexture(resultTexShape);
        pool_gpu.poolCommon(this.gpgpu, program, x.getTexture(), poolResultTex, resultTexShape);
        if (cleanupX) {
            x.dispose();
        }
        return ndarray_1.NDArray.make(resultShape, { texture: poolResultTex, textureShapeRC: resultTexShape });
    };
    NDArrayMathGPU.prototype.maxPoolInternal = function (x, fSize, stride, pad) {
        var maxPoolProgKey = [MAX_POOL_PROG, x.shape, fSize, stride, pad].join('_');
        var maxPoolProgram = this.getAndSaveProgram(maxPoolProgKey, function () {
            return max_pool_gpu.getFragmentShaderMaxPoolSource(x.shape, fSize, stride, pad);
        });
        return this.pool(maxPoolProgram, x, fSize, stride, pad);
    };
    NDArrayMathGPU.prototype.minPoolInternal = function (x, fSize, stride, pad) {
        var minPoolProgKey = [MIN_POOL_PROG, x.shape, fSize, stride, pad].join('_');
        var minPoolProgram = this.getAndSaveProgram(minPoolProgKey, function () {
            return min_pool_gpu.getFragmentShaderMinPoolSource(x.shape, fSize, stride, pad);
        });
        return this.pool(minPoolProgram, x, fSize, stride, pad);
    };
    NDArrayMathGPU.prototype.avgPoolInternal = function (x, fSize, stride, pad) {
        var avgPoolProgKey = [AVG_POOL_PROG, x.shape, fSize, stride, pad].join('_');
        var avgPoolProgram = this.getAndSaveProgram(avgPoolProgKey, function () {
            return avg_pool_gpu.getFragmentShaderAvgPoolSource(x.shape, fSize, stride, pad);
        });
        return this.pool(avgPoolProgram, x, fSize, stride, pad);
    };
    NDArrayMathGPU.prototype.maxPoolBackpropInternal = function (dy, x, fSize, origStride, origPad) {
        var maxPoolPositionsProgKey = [
            MAX_POOL_POSITIONS_PROG, x.shape, fSize, origStride, origPad
        ].join('_');
        var maxPoolPositionsProgram = this.getAndSaveProgram(maxPoolPositionsProgKey, function () {
            return max_pool_gpu.getFragmentShaderMaxPoolPositionsSource(x.shape, fSize, origStride, origPad);
        });
        var maxPoolResultShape = conv_util.computeOutputShape3D(x.shape, fSize, x.shape[2], origStride, origPad);
        var maxPoolResultTexShape = conv_util.computeTexShapeFrom3D(maxPoolResultShape);
        var maxPoolPositionsResultTex = this.textureManager.acquireTexture(maxPoolResultTexShape);
        var xTexShape = conv_util.computeTexShapeFrom3D(x.shape);
        var actualXTexShape = x.getTextureShapeRC(xTexShape);
        var cleanupX = false;
        if (!util.arraysEqual(actualXTexShape, xTexShape)) {
            x = this.reshapeTexture(x, xTexShape);
            cleanupX = true;
        }
        max_pool_gpu.maxPoolCommon(this.gpgpu, maxPoolPositionsProgram, x.getTexture(), maxPoolPositionsResultTex, maxPoolResultTexShape);
        var maxPoolBackpropProgKey = [
            MAX_POOL_BACKPROP_PROG, dy.shape, fSize, origStride, origPad
        ].join('_');
        var program = this.getAndSaveProgram(maxPoolBackpropProgKey, function () {
            return max_pool_backprop_gpu.getFragmentShaderMaxPoolBackprop(dy.shape, fSize, origStride, origPad);
        });
        var dyTexShape = conv_util.computeTexShapeFrom3D(dy.shape);
        var actualDyTexShape = dy.getTextureShapeRC(dyTexShape);
        var cleanupDy = false;
        if (!util.arraysEqual(actualDyTexShape, dyTexShape)) {
            dy = this.reshapeTexture(dy, dyTexShape);
            cleanupDy = true;
        }
        var dilatedDyRC = conv_util.computeDilatedRC([dy.shape[0], dy.shape[1]], origStride);
        var pad = fSize - 1 - origPad;
        var resultShapeRCD = conv_util.computeOutputShape3D([dilatedDyRC[0], dilatedDyRC[1], dy.shape[2]], fSize, dy.shape[2], 1, pad);
        var resultTexShape = conv_util.computeTexShapeFrom3D(resultShapeRCD);
        var resultTex = this.textureManager.acquireTexture(resultTexShape);
        max_pool_backprop_gpu.maxPoolBackprop(this.gpgpu, program, dy.getTexture(), maxPoolPositionsResultTex, resultTex, resultTexShape);
        if (cleanupDy) {
            dy.dispose();
        }
        if (cleanupX) {
            x.dispose();
        }
        this.textureManager.releaseTexture(maxPoolPositionsResultTex, maxPoolResultTexShape);
        return ndarray_1.NDArray.make(resultShapeRCD, { texture: resultTex, textureShapeRC: resultTexShape });
    };
    NDArrayMathGPU.prototype.resizeBilinear3DInternal = function (x, newShape2D, alignCorners) {
        var programKey = [RESIZE_BILINEAR_PROG, x.shape, newShape2D, alignCorners].join('_');
        var newShapeRCD = [newShape2D[0], newShape2D[1], x.shape[2]];
        var resultTexShape = conv_util.computeTexShapeFrom3D(newShapeRCD);
        var program = this.getAndSaveProgram(programKey, function () { return resize_bilinear_gpu.getFragmentShaderSource(x.shape, newShape2D, alignCorners); });
        var resultTexture = this.textureManager.acquireTexture(resultTexShape);
        resize_bilinear_gpu.resizeBilinear(this.gpgpu, program, x.getTexture(), resultTexture, resultTexShape);
        return ndarray_1.NDArray.make(newShapeRCD, { texture: resultTexture, textureShapeRC: resultTexShape });
    };
    NDArrayMathGPU.prototype.getAndSaveProgram = function (programKey, getShaderSource) {
        if (!(programKey in this.programCache)) {
            this.programCache[programKey] =
                this.gpgpu.createProgram(getShaderSource());
        }
        return this.programCache[programKey];
    };
    NDArrayMathGPU.prototype.addSubMulDiv = function (a, b, resultShape, operandA, opType, operandB) {
        var cleanupB = false;
        var aOrientation = math_1.MatrixOrientation.REGULAR;
        var bOrientation = math_1.MatrixOrientation.REGULAR;
        var logicalBTexShape;
        if (operandA === addsubmuldiv_gpu_1.OperandType.MATRIX && operandB === addsubmuldiv_gpu_1.OperandType.MATRIX) {
            util.assertShapesMatch(a.shape, b.shape);
            if (a.inGPU()) {
                b.getTextureShapeRC(a.getTextureShapeRC());
            }
            else if (b.inGPU()) {
                a.getTextureShapeRC(b.getTextureShapeRC());
            }
            var aTexShape_1 = a.getTextureShapeRC();
            var bTexShape_1 = b.getTextureShapeRC();
            logicalBTexShape = bTexShape_1;
            if (a.rank === 1) {
                if (!util.arraysEqual(bTexShape_1, aTexShape_1)) {
                    bOrientation = math_1.MatrixOrientation.TRANSPOSED;
                    logicalBTexShape = [bTexShape_1[1], bTexShape_1[0]];
                }
            }
            if (!util.arraysEqual(aTexShape_1, logicalBTexShape)) {
                b = this.reshapeTexture(b, aTexShape_1);
                bOrientation = math_1.MatrixOrientation.REGULAR;
                logicalBTexShape = b.getTextureShapeRC();
                cleanupB = true;
            }
        }
        else {
            logicalBTexShape = b.getTextureShapeRC();
        }
        var aTexShape = a.getTextureShapeRC();
        var bTexShape = b.getTextureShapeRC();
        var programKey = [
            ADD_SUM_MUL_DIV_PROG, operandA, aOrientation, opType, operandB,
            bOrientation
        ].join('_');
        var program = this.getAndSaveProgram(programKey, function () { return addsubmuldiv_gpu.getFragmentShaderSource(operandA, aOrientation, opType, operandB, bOrientation); });
        var resultTextureShape = [
            Math.max(aTexShape[0], logicalBTexShape[0]),
            Math.max(aTexShape[1], logicalBTexShape[1])
        ];
        var resultTexture = this.textureManager.acquireTexture(resultTextureShape);
        addsubmuldiv_gpu.addSubMulDiv(this.gpgpu, program, a.getTexture(), aTexShape, b.getTexture(), bTexShape, resultTexture, resultTextureShape);
        if (cleanupB) {
            b.dispose();
        }
        return ndarray_1.NDArray.make(resultShape, { texture: resultTexture, textureShapeRC: resultTextureShape });
    };
    NDArrayMathGPU.prototype.doGPUShapesMatch = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape);
        if (a.inGPU()) {
            b.getTextureShapeRC(a.getTextureShapeRC());
        }
        else if (b.inGPU()) {
            a.getTextureShapeRC(b.getTextureShapeRC());
        }
        return util.arraysEqual(a.getTextureShapeRC(), b.getTextureShapeRC());
    };
    NDArrayMathGPU.prototype.getTextureManager = function () {
        return this.textureManager;
    };
    NDArrayMathGPU.prototype.dispose = function () {
        for (var programKey in this.programCache) {
            if (this.programCache.hasOwnProperty(programKey)) {
                this.gpgpu.deleteProgram(this.programCache[programKey]);
            }
        }
        this.textureManager.dispose();
        if (this.gpgpuCreatedLocally) {
            this.gpgpu.dispose();
        }
    };
    return NDArrayMathGPU;
}(math_1.NDArrayMath));
exports.NDArrayMathGPU = NDArrayMathGPU;

},{"../util":86,"./concat3d_util":15,"./conv_util":16,"./math":19,"./ndarray":22,"./webgl/addscaledmat_gpu":23,"./webgl/addsubmuldiv_gpu":24,"./webgl/argmaxequals_gpu":25,"./webgl/argminmax_gpu":26,"./webgl/avg_pool_gpu":27,"./webgl/batchnorm_gpu":28,"./webgl/concat3d_gpu":30,"./webgl/conv_backprop_gpu":31,"./webgl/conv_gpu":32,"./webgl/copy_gpu":33,"./webgl/exp_gpu":34,"./webgl/gpgpu_context":35,"./webgl/gpgpu_util":36,"./webgl/log_gpu":37,"./webgl/logsumexp_gpu":38,"./webgl/max_pool_backprop_gpu":39,"./webgl/max_pool_gpu":40,"./webgl/min_pool_gpu":41,"./webgl/minmax_gpu":42,"./webgl/mulmat_gpu":43,"./webgl/neg_gpu":44,"./webgl/pool_gpu":45,"./webgl/reducesum_gpu":46,"./webgl/relu_gpu":47,"./webgl/reshape_gpu":49,"./webgl/resize_bilinear_gpu":50,"./webgl/shader_compiler":51,"./webgl/sigmoid_gpu":52,"./webgl/step_gpu":53,"./webgl/texture_manager":55,"./webgl/trig_gpu":56,"./webgl/webgl_util":58}],22:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../util");
var webgl_util = require("./webgl/webgl_util");
exports.GPGPU = null;
exports.TEXTURE_MANAGER = null;
function initializeGPU(gpgpu, textureManager) {
    exports.GPGPU = gpgpu;
    exports.TEXTURE_MANAGER = textureManager;
}
exports.initializeGPU = initializeGPU;
function throwIfGPUNotInitialized() {
    if (exports.GPGPU == null || exports.TEXTURE_MANAGER == null) {
        throw new Error('GPU not intialized.');
    }
}
var NDArray = (function () {
    function NDArray(shape, data) {
        util.assert(data.values != null || data.texture != null, 'Either `values` or `texture` must be defined');
        util.assert(data.texture == null || (data.textureShapeRC != null), '`textureShape` must be defined when `texture` is defined');
        this.size = util.sizeFromShape(shape);
        if (data.values != null) {
            util.assert(this.size === data.values.length, 'Constructing ndarray of shape (' + this.size + ') should match the' +
                ' length of values (' + data.values.length + ')');
        }
        this.shape = shape;
        this.data = data;
        var dim = this.shape.length;
        if (dim < 2) {
            this.strides = [];
        }
        else {
            this.strides = new Array(dim - 1);
            this.strides[dim - 2] = this.shape[dim - 1];
            for (var i = dim - 3; i >= 0; --i) {
                this.strides[i] = this.strides[i + 1] * this.shape[i + 1];
            }
        }
    }
    NDArray.zeros = function (shape) {
        var values = new Float32Array(util.sizeFromShape(shape));
        return NDArray.make(shape, { values: values });
    };
    NDArray.zerosLike = function (another) {
        return NDArray.zeros(another.shape);
    };
    NDArray.like = function (another) {
        var values = another.getValues();
        return NDArray.make(another.shape, { values: new Float32Array(values) });
    };
    NDArray.make = function (shape, data) {
        switch (shape.length) {
            case 0:
                return new Scalar(data);
            case 1:
                return new Array1D(data);
            case 2:
                return new Array2D(shape, data);
            case 3:
                return new Array3D(shape, data);
            case 4:
                return new Array4D(shape, data);
            default:
                return new NDArray(shape, data);
        }
    };
    NDArray.prototype.reshape = function (newShape) {
        if (util.arraysEqual(this.shape, newShape)) {
            return this;
        }
        util.assert(this.size === util.sizeFromShape(newShape), 'new shape and old shape must have the same number of elements.');
        return NDArray.make(newShape, this.data);
    };
    NDArray.prototype.asScalar = function () {
        util.assert(this.size === 1, 'The array must have only 1 element.');
        return this.reshape([]);
    };
    NDArray.prototype.as1D = function () {
        return this.reshape([this.size]);
    };
    NDArray.prototype.as2D = function (rows, columns) {
        return this.reshape([rows, columns]);
    };
    NDArray.prototype.as3D = function (rows, columns, depth) {
        return this.reshape([rows, columns, depth]);
    };
    NDArray.prototype.as4D = function (rows, columns, depth, depth2) {
        return this.reshape([rows, columns, depth, depth2]);
    };
    Object.defineProperty(NDArray.prototype, "rank", {
        get: function () {
            return this.shape.length;
        },
        enumerable: true,
        configurable: true
    });
    NDArray.prototype.get = function () {
        var locs = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            locs[_i] = arguments[_i];
        }
        var index = locs[locs.length - 1];
        for (var i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        return this.getValues()[index];
    };
    NDArray.prototype.add = function (value) {
        var locs = [];
        for (var _i = 1; _i < arguments.length; _i++) {
            locs[_i - 1] = arguments[_i];
        }
        this.set.apply(this, [this.get.apply(this, locs) + value].concat(locs));
    };
    NDArray.prototype.set = function (value) {
        var locs = [];
        for (var _i = 1; _i < arguments.length; _i++) {
            locs[_i - 1] = arguments[_i];
        }
        var index = locs[locs.length - 1];
        for (var i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        this.getValues()[index] = value;
    };
    NDArray.prototype.locToIndex = function (locs) {
        var index = locs[locs.length - 1];
        for (var i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        return index;
    };
    NDArray.prototype.indexToLoc = function (index) {
        var locs = new Array(this.shape.length);
        for (var i = 0; i < locs.length - 1; ++i) {
            locs[i] = Math.floor(index / this.strides[i]);
            index -= locs[i] * this.strides[i];
        }
        locs[locs.length - 1] = index;
        return locs;
    };
    NDArray.prototype.fill = function (value) {
        this.getValues().fill(value);
    };
    NDArray.prototype.getData = function () {
        return this.data;
    };
    NDArray.prototype.getValues = function () {
        if (this.data.values == null) {
            throwIfGPUNotInitialized();
            this.data.values = exports.GPGPU.downloadMatrixFromTexture(this.data.texture, this.data.textureShapeRC[0], this.data.textureShapeRC[1]);
            this.disposeTexture();
        }
        return this.data.values;
    };
    NDArray.prototype.uploadToGPU = function (preferredTexShape) {
        throwIfGPUNotInitialized();
        this.data.textureShapeRC = webgl_util.getTextureShapeFromLogicalShape(exports.GPGPU.gl, this.shape, preferredTexShape);
        this.data.texture =
            exports.TEXTURE_MANAGER.acquireTexture(this.data.textureShapeRC);
        exports.GPGPU.uploadMatrixToTexture(this.data.texture, this.data.textureShapeRC[0], this.data.textureShapeRC[1], this.data.values);
        this.data.values = null;
    };
    NDArray.prototype.getTexture = function (preferredShapeRC) {
        if (this.data.texture == null) {
            this.uploadToGPU(preferredShapeRC);
        }
        return this.data.texture;
    };
    NDArray.prototype.getTextureShapeRC = function (preferredShapeRC) {
        if (this.data.textureShapeRC == null) {
            this.uploadToGPU(preferredShapeRC);
        }
        return this.data.textureShapeRC;
    };
    NDArray.prototype.dispose = function () {
        this.data.values = null;
        this.shape = null;
        if (this.data.texture != null) {
            this.disposeTexture();
        }
    };
    NDArray.prototype.disposeTexture = function () {
        throwIfGPUNotInitialized();
        exports.TEXTURE_MANAGER.releaseTexture(this.data.texture, this.data.textureShapeRC);
        this.data.texture = null;
        this.data.textureShapeRC = null;
    };
    NDArray.prototype.inGPU = function () {
        return this.data.texture != null;
    };
    NDArray.prototype.equals = function (t) {
        return util.arraysEqual(this.shape, t.shape) &&
            util.arraysEqual(this.getValues(), t.getValues());
    };
    NDArray.rand = function (shape, randFunction) {
        var size = util.sizeFromShape(shape);
        var values = new Float32Array(size);
        for (var i = 0; i < size; i++) {
            values[i] = randFunction();
        }
        return NDArray.make(shape, { values: values });
    };
    NDArray.randNormal = function (shape, mean, stdDev) {
        if (mean === void 0) { mean = 0; }
        if (stdDev === void 0) { stdDev = 1; }
        return NDArray.rand(shape, function () { return util.randGauss(mean, stdDev); });
    };
    NDArray.randTruncatedNormal = function (shape, mean, stdDev) {
        if (mean === void 0) { mean = 0; }
        if (stdDev === void 0) { stdDev = 1; }
        return NDArray.rand(shape, function () { return util.randGauss(mean, stdDev, true); });
    };
    NDArray.randUniform = function (shape, a, b) {
        return NDArray.rand(shape, function () { return util.randUniform(a, b); });
    };
    return NDArray;
}());
exports.NDArray = NDArray;
var Scalar = (function (_super) {
    __extends(Scalar, _super);
    function Scalar(data) {
        var _this = this;
        if (data.texture != null) {
            data.textureShapeRC = [1, 1];
        }
        _this = _super.call(this, [], data) || this;
        return _this;
    }
    Scalar.new = function (value) {
        return new Scalar({ values: new Float32Array([value]) });
    };
    Scalar.prototype.get = function () {
        return this.getValues()[0];
    };
    Scalar.prototype.set = function (value) {
        this.getValues()[0] = value;
    };
    Scalar.prototype.add = function (value) {
        this.getValues()[0] += value;
    };
    return Scalar;
}(NDArray));
Scalar.ZERO = Scalar.new(0);
Scalar.ONE = Scalar.new(1);
Scalar.TWO = Scalar.new(2);
Scalar.NEG_ONE = Scalar.new(-1);
exports.Scalar = Scalar;
var Array1D = (function (_super) {
    __extends(Array1D, _super);
    function Array1D(data) {
        var _this = this;
        var shape = (data.values != null) ?
            [data.values.length] :
            [util.sizeFromShape(data.textureShapeRC)];
        _this = _super.call(this, shape, data) || this;
        return _this;
    }
    Array1D.new = function (values) {
        if (!(values instanceof Float32Array)) {
            var inferredShape = util.inferShape(values);
            util.assert(inferredShape.length === 1, "Error constructing Array1D. Shape of values " + inferredShape + " is " +
                "not 1 dimensional.");
        }
        return new Array1D({ values: toTypedArray(values) });
    };
    Array1D.prototype.get = function (i) {
        return this.getValues()[i];
    };
    Array1D.prototype.set = function (value, i) {
        this.getValues()[i] = value;
    };
    Array1D.prototype.add = function (value, i) {
        this.getValues()[i] += value;
    };
    Array1D.prototype.locToIndex = function (loc) {
        return loc[0];
    };
    Array1D.prototype.indexToLoc = function (index) {
        return [index];
    };
    Array1D.zeros = function (shape) {
        return NDArray.zeros(shape);
    };
    return Array1D;
}(NDArray));
exports.Array1D = Array1D;
var Array2D = (function (_super) {
    __extends(Array2D, _super);
    function Array2D(shape, data) {
        var _this = this;
        util.assert(shape.length === 2, 'Shape should be of length 2');
        _this = _super.call(this, shape, data) || this;
        _this.stride0 = _this.strides[0];
        return _this;
    }
    Array2D.new = function (shape, values) {
        if (!(values instanceof Float32Array)) {
            var inferredShape = util.inferShape(values);
            if (inferredShape.length > 1) {
                util.assertShapesMatch(shape, inferredShape, "Error when constructing Array2D. Shape of values " +
                    (inferredShape + " does not match the provided shape ") +
                    (shape + ". "));
            }
        }
        return new Array2D(shape, { values: toTypedArray(values) });
    };
    Array2D.prototype.get = function (i, j) {
        return this.getValues()[this.stride0 * i + j];
    };
    Array2D.prototype.set = function (value, i, j) {
        this.getValues()[this.stride0 * i + j] = value;
    };
    Array2D.prototype.add = function (value, i, j) {
        this.getValues()[this.stride0 * i + j] += value;
    };
    Array2D.prototype.locToIndex = function (locs) {
        return this.stride0 * locs[0] + locs[1];
    };
    Array2D.prototype.indexToLoc = function (index) {
        return [Math.floor(index / this.stride0), index % this.stride0];
    };
    Array2D.zeros = function (shape) {
        return NDArray.zeros(shape);
    };
    return Array2D;
}(NDArray));
exports.Array2D = Array2D;
var Array3D = (function (_super) {
    __extends(Array3D, _super);
    function Array3D(shape, data) {
        var _this = this;
        util.assert(shape.length === 3, 'Shape should be of length 3');
        _this = _super.call(this, shape, data) || this;
        _this.stride0 = _this.strides[0];
        _this.stride1 = _this.strides[1];
        return _this;
    }
    Array3D.new = function (shape, values) {
        if (!(values instanceof Float32Array)) {
            var inferredShape = util.inferShape(values);
            if (inferredShape.length > 1) {
                util.assertShapesMatch(shape, inferredShape, "Error when constructing Array3D. Shape of values " +
                    (inferredShape + " does not match the provided shape ") +
                    (shape + ". "));
            }
        }
        return new Array3D(shape, { values: toTypedArray(values) });
    };
    Array3D.prototype.get = function (i, j, k) {
        return this.getValues()[this.stride0 * i + this.stride1 * j + k];
    };
    Array3D.prototype.set = function (value, i, j, k) {
        this.getValues()[this.stride0 * i + this.stride1 * j + k] = value;
    };
    Array3D.prototype.add = function (value, i, j, k) {
        this.getValues()[this.stride0 * i + this.stride1 * j + k] += value;
    };
    Array3D.prototype.locToIndex = function (locs) {
        return this.stride0 * locs[0] + this.stride1 * locs[1] + locs[2];
    };
    Array3D.prototype.indexToLoc = function (index) {
        var i = Math.floor(index / this.stride0);
        index -= i * this.stride0;
        return [i, Math.floor(index / this.stride1), index % this.stride1];
    };
    Array3D.zeros = function (shape) {
        return NDArray.zeros(shape);
    };
    return Array3D;
}(NDArray));
exports.Array3D = Array3D;
var Array4D = (function (_super) {
    __extends(Array4D, _super);
    function Array4D(shape, data) {
        var _this = this;
        util.assert(shape.length === 4, 'Shape should be of length 4');
        _this = _super.call(this, shape, data) || this;
        _this.stride0 = _this.strides[0];
        _this.stride1 = _this.strides[1];
        _this.stride2 = _this.strides[2];
        return _this;
    }
    Array4D.new = function (shape, values) {
        if (!(values instanceof Float32Array)) {
            var inferredShape = util.inferShape(values);
            if (inferredShape.length > 1) {
                util.assertShapesMatch(shape, inferredShape, "Error when constructing Array4D. Shape of values " +
                    (inferredShape + " does not match the provided shape ") +
                    (shape + ". "));
            }
        }
        return new Array4D(shape, { values: toTypedArray(values) });
    };
    Array4D.prototype.get = function (i, j, k, l) {
        return this.getValues()[this.stride0 * i + this.stride1 * j + this.stride2 * k + l];
    };
    Array4D.prototype.set = function (value, i, j, k, l) {
        this.getValues()[this.stride0 * i + this.stride1 * j + this.stride2 * k + l] = value;
    };
    Array4D.prototype.add = function (value, i, j, k, l) {
        this.getValues()[this.stride0 * i + this.stride1 * j + this.stride2 * k + l] += value;
    };
    Array4D.prototype.locToIndex = function (locs) {
        return this.stride0 * locs[0] + this.stride1 * locs[1] +
            this.stride2 * locs[2] + locs[3];
    };
    Array4D.prototype.indexToLoc = function (index) {
        var i = Math.floor(index / this.stride0);
        index -= i * this.stride0;
        var j = Math.floor(index / this.stride1);
        index -= j * this.stride1;
        return [i, j, Math.floor(index / this.stride2), index % this.stride2];
    };
    Array4D.zeros = function (shape) {
        return NDArray.zeros(shape);
    };
    return Array4D;
}(NDArray));
exports.Array4D = Array4D;
function toTypedArray(a) {
    return (a instanceof Float32Array) ? a : new Float32Array(util.flatten(a));
}

},{"../util":86,"./webgl/webgl_util":58}],23:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var gpgpu_context_1 = require("./gpgpu_context");
function getFragmentShaderSource() {
    return "\n    precision highp float;\n    uniform sampler2D matrixA;\n    uniform sampler2D matrixB;\n    uniform sampler2D matrixAScalar;\n    uniform sampler2D matrixBScalar;\n    varying vec2 resultUV;\n\n    const vec2 halfTexel = vec2(0.5, 0.5);\n\n    void main() {\n      float a = texture2D(matrixA, resultUV).r;\n      float b = texture2D(matrixB, resultUV).r;\n      float aScalar = texture2D(matrixAScalar, halfTexel).r;\n      float bScalar = texture2D(matrixBScalar, halfTexel).r;\n      vec2 abScaled = vec2(a, b) * vec2(aScalar, bScalar);\n      gl_FragColor = vec4(abScaled.x + abScaled.y, 0, 0, 0);\n    }";
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function addScaledMatrices(gpgpu, addScaledMatricesProgram, a, b, rows, columns, aScalar, bScalar, result) {
    gpgpu.setOutputMatrixTexture(result, rows, columns);
    gpgpu.setProgram(addScaledMatricesProgram);
    gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
    gpgpu.setInputMatrixTexture(b, 'matrixB', 1);
    gpgpu.setInputMatrixTexture(aScalar, 'matrixAScalar', 2);
    gpgpu.setInputMatrixTexture(bScalar, 'matrixBScalar', 3);
    gpgpu.executeProgram();
}
exports.addScaledMatrices = addScaledMatrices;
function uploadAddScaledMatricesDownload(a, b, rows, columns, aScalar, bScalar) {
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var program = gpgpu.createProgram(getFragmentShaderSource());
    var aTex = gpgpu.createMatrixTexture(rows, columns);
    var bTex = gpgpu.createMatrixTexture(rows, columns);
    var aScalarTex = gpgpu.createMatrixTexture(1, 1);
    var bScalarTex = gpgpu.createMatrixTexture(1, 1);
    var resultTex = gpgpu.createMatrixTexture(rows, columns);
    gpgpu.uploadMatrixToTexture(aTex, rows, columns, a);
    gpgpu.uploadMatrixToTexture(bTex, rows, columns, b);
    gpgpu.uploadMatrixToTexture(aScalarTex, 1, 1, new Float32Array([aScalar]));
    gpgpu.uploadMatrixToTexture(bScalarTex, 1, 1, new Float32Array([bScalar]));
    addScaledMatrices(gpgpu, program, aTex, bTex, rows, columns, aScalarTex, bScalarTex, resultTex);
    var result = gpgpu.downloadMatrixFromTexture(resultTex, rows, columns);
    gpgpu.deleteMatrixTexture(aTex);
    gpgpu.deleteMatrixTexture(bTex);
    gpgpu.deleteMatrixTexture(resultTex);
    gpgpu.deleteMatrixTexture(aScalarTex);
    gpgpu.deleteMatrixTexture(bScalarTex);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return result;
}
exports.uploadAddScaledMatricesDownload = uploadAddScaledMatricesDownload;

},{"./gpgpu_context":35}],24:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var math_1 = require("../math");
var binaryop_gpu = require("./binaryop_gpu");
var OperandType;
(function (OperandType) {
    OperandType[OperandType["MATRIX"] = 0] = "MATRIX";
    OperandType[OperandType["SCALAR"] = 1] = "SCALAR";
})(OperandType = exports.OperandType || (exports.OperandType = {}));
function getFragmentShaderSource(aType, aOrientation, op, bType, bOrientation) {
    var aUV = operandToShaderSnippet(aType, aOrientation);
    var bUV = operandToShaderSnippet(bType, bOrientation);
    var resultOp = "gl_FragColor = vec4(a " + op + " b, 0, 0, 0);";
    return binaryop_gpu.getFragmentShaderSource(aUV, bUV, resultOp);
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function operandToShaderSnippet(operand, orientation) {
    switch (operand) {
        case OperandType.MATRIX:
            return 'resultUV' +
                (orientation === math_1.MatrixOrientation.REGULAR ? '.st' : '.ts');
        case OperandType.SCALAR:
            return 'vec2(0.5, 0.5)';
        default:
            throw new Error('Unknown operand type');
    }
}
function addSubMulDiv(gpgpu, program, a, aShapeRowCol, b, bShapeRowCol, result, resultShapeRowCol) {
    return binaryop_gpu.binaryOp(gpgpu, program, a, aShapeRowCol, b, bShapeRowCol, result, resultShapeRowCol);
}
exports.addSubMulDiv = addSubMulDiv;
function uploadScalarPlusMatrixDownload(a, b, bShape, bOrientation) {
    if (bOrientation === void 0) { bOrientation = math_1.MatrixOrientation.REGULAR; }
    var src = getFragmentShaderSource(OperandType.SCALAR, math_1.MatrixOrientation.REGULAR, '+', OperandType.MATRIX, bOrientation);
    return binaryop_gpu.uploadBinaryOpDownload(new Float32Array([a]), [1, 1], b, bShape, src);
}
exports.uploadScalarPlusMatrixDownload = uploadScalarPlusMatrixDownload;
function uploadMatrixMinusScalarDownload(a, aShape, b, aOrientation) {
    if (aOrientation === void 0) { aOrientation = math_1.MatrixOrientation.REGULAR; }
    var src = getFragmentShaderSource(OperandType.MATRIX, aOrientation, '-', OperandType.SCALAR, math_1.MatrixOrientation.REGULAR);
    return binaryop_gpu.uploadBinaryOpDownload(a, aShape, new Float32Array([b]), [1, 1], src);
}
exports.uploadMatrixMinusScalarDownload = uploadMatrixMinusScalarDownload;
function uploadScalarMinusMatrixDownload(a, b, bShape, bOrientation) {
    if (bOrientation === void 0) { bOrientation = math_1.MatrixOrientation.REGULAR; }
    var src = getFragmentShaderSource(OperandType.SCALAR, math_1.MatrixOrientation.REGULAR, '-', OperandType.MATRIX, bOrientation);
    return binaryop_gpu.uploadBinaryOpDownload(new Float32Array([a]), [1, 1], b, bShape, src);
}
exports.uploadScalarMinusMatrixDownload = uploadScalarMinusMatrixDownload;
function uploadScalarTimesMatrixDownload(a, b, bShape, bOrientation) {
    if (bOrientation === void 0) { bOrientation = math_1.MatrixOrientation.REGULAR; }
    var src = getFragmentShaderSource(OperandType.SCALAR, math_1.MatrixOrientation.REGULAR, '*', OperandType.MATRIX, bOrientation);
    return binaryop_gpu.uploadBinaryOpDownload(new Float32Array([a]), [1, 1], b, bShape, src);
}
exports.uploadScalarTimesMatrixDownload = uploadScalarTimesMatrixDownload;
function uploadMatrixTimesMatrixDownload(a, b, shape, aOrientation, bOrientation) {
    if (aOrientation === void 0) { aOrientation = math_1.MatrixOrientation.REGULAR; }
    if (bOrientation === void 0) { bOrientation = math_1.MatrixOrientation.REGULAR; }
    var src = getFragmentShaderSource(OperandType.MATRIX, aOrientation, '*', OperandType.MATRIX, bOrientation);
    return binaryop_gpu.uploadBinaryOpDownload(a, shape, b, shape, src);
}
exports.uploadMatrixTimesMatrixDownload = uploadMatrixTimesMatrixDownload;
function uploadMatrixPlusMatrixDownload(a, b, shape, aOrientation, bOrientation) {
    if (aOrientation === void 0) { aOrientation = math_1.MatrixOrientation.REGULAR; }
    if (bOrientation === void 0) { bOrientation = math_1.MatrixOrientation.REGULAR; }
    var src = getFragmentShaderSource(OperandType.MATRIX, aOrientation, '+', OperandType.MATRIX, bOrientation);
    return binaryop_gpu.uploadBinaryOpDownload(a, shape, b, shape, src);
}
exports.uploadMatrixPlusMatrixDownload = uploadMatrixPlusMatrixDownload;

},{"../math":19,"./binaryop_gpu":29}],25:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var argminmax_gpu = require("./argminmax_gpu");
function getFragmentShaderPrologueSource() {
    return "\n    precision highp float;\n    uniform sampler2D matrixA;\n    uniform sampler2D matrixB;\n    varying vec2 resultUV;";
}
function getFragmentShaderMainSource() {
    return "\n    void main() {\n      float argMaxA = getArgMinMax(matrixA);\n      float argMaxB = getArgMinMax(matrixB);\n      float value;\n      if (isNaN(argMaxA)) {\n        value = argMaxA;\n      } else if (isNaN(argMaxB)) {\n        value = argMaxB;\n      } else {\n        value = float(argMaxA == argMaxB);\n      }\n      gl_FragColor = vec4(value, 0, 0, 0);\n    }";
}
function getArgMaxEqualsFragmentShaderSource(rows, columns) {
    return [
        getFragmentShaderPrologueSource(),
        argminmax_gpu.getFragmentShaderGetArgMinMaxSource('>', rows, columns),
        getFragmentShaderMainSource()
    ].join('\n');
}
exports.getArgMaxEqualsFragmentShaderSource = getArgMaxEqualsFragmentShaderSource;
function argMaxEquals(gpgpu, maxEqualsProgram, a, b, numRows, numCols, result) {
    gpgpu.setOutputMatrixTexture(result, 1, 1);
    gpgpu.setProgram(maxEqualsProgram);
    gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
    gpgpu.setInputMatrixTexture(b, 'matrixB', 1);
    gpgpu.executeProgram();
}
exports.argMaxEquals = argMaxEquals;

},{"./argminmax_gpu":26}],26:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var webgl_util_1 = require("./webgl_util");
function getFragmentShaderPrologueSource() {
    return "\n    precision highp float;\n    uniform sampler2D matrixA;\n    varying vec2 resultUV;";
}
exports.getFragmentShaderPrologueSource = getFragmentShaderPrologueSource;
function getFragmentShaderMainSource() {
    return "\n    void main() {\n      gl_FragColor = vec4(getArgMinMax(matrixA), 0, 0, 0);\n    }";
}
function getArgMinMaxFragmentShaderSource(rows, columns, compOp) {
    return [
        getFragmentShaderPrologueSource(),
        getFragmentShaderGetArgMinMaxSource(compOp, rows, columns),
        getFragmentShaderMainSource()
    ].join('\n');
}
function getArgMinFragmentShaderSource(rows, columns) {
    return getArgMinMaxFragmentShaderSource(rows, columns, '<');
}
exports.getArgMinFragmentShaderSource = getArgMinFragmentShaderSource;
function getArgMaxFragmentShaderSource(rows, columns) {
    return getArgMinMaxFragmentShaderSource(rows, columns, '>');
}
exports.getArgMaxFragmentShaderSource = getArgMaxFragmentShaderSource;
function getFragmentShaderGetArgMinMaxSource(compOp, rows, columns) {
    return "\n    const vec2 dimCR = vec2(" + columns + ".0, " + rows + ".0);\n    const vec2 halfCR = vec2(0.5, 0.5);\n\n    " + webgl_util_1.IS_NAN_SHADER_FUNC + "\n\n    float getArgMinMax(in sampler2D matrix) {\n      vec2 bestCR = vec2(0, 0);\n      float bestValue = texture2D(matrix, bestCR).r;\n\n      for (float c = 0.0; c < dimCR.x; c += 1.0) {\n        for (float r = 0.0; r < dimCR.y; r += 1.0) {\n          vec2 cr = vec2(c, r);\n          vec2 uv = (cr + halfCR) / dimCR;\n          float value = texture2D(matrix, uv).r;\n          if (isNaN(value)) {\n            return value;\n          }\n          if (value " + compOp + " bestValue) {\n            bestValue = value;\n            bestCR = cr;\n          }\n        }\n      }\n      return bestCR.x + (bestCR.y * dimCR.x);\n    }\n  ";
}
exports.getFragmentShaderGetArgMinMaxSource = getFragmentShaderGetArgMinMaxSource;
function argMinMax(gpgpu, minMaxProgram, a, aNumRows, aNumCols, result) {
    gpgpu.setOutputMatrixTexture(result, 1, 1);
    gpgpu.setProgram(minMaxProgram);
    gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
    gpgpu.executeProgram();
}
exports.argMinMax = argMinMax;

},{"./webgl_util":58}],27:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var pool_gpu = require("./pool_gpu");
function getFragmentShaderAvgPoolSource(xShapeRCD, fSize, stride, pad) {
    return pool_gpu.getFragmentShaderPoolCommonSource(xShapeRCD, fSize, stride, pad, 'avg', false);
}
exports.getFragmentShaderAvgPoolSource = getFragmentShaderAvgPoolSource;
function avgPool(gpgpu, program, x, result, resultShapeRowCol) {
    pool_gpu.poolCommon(gpgpu, program, x, result, resultShapeRowCol);
}
exports.avgPool = avgPool;

},{"./pool_gpu":45}],28:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function getFragmentShaderSource(xTexShapeRC, meanTexShapeRC, varianceTexShapeRC, offsetTexShapeRC, scaleTexShapeRC, varianceEpsilon) {
    if (varianceEpsilon === void 0) { varianceEpsilon = 0.001; }
    var offsetSamplerSnippet = '';
    var offsetShapeInitializationSnippet = '';
    var offsetCoordsSnippet = '';
    var offsetUVSnippet = '';
    var offsetValueSnippet = '';
    var offsetOperationSnippet = '0.0';
    var scaleSamplerSnippet = '';
    var scaleShapeInitializationSnippet = '';
    var scaleCoordsSnippet = '';
    var scaleUVSnippet = '';
    var scaleValueSnippet = '';
    var scaleOperationSnippet = '';
    if (offsetTexShapeRC != null) {
        offsetSamplerSnippet = 'uniform sampler2D offset;';
        offsetShapeInitializationSnippet = "const vec2 offsetShapeCR = vec2(\n            " + offsetTexShapeRC[1] + ", " + offsetTexShapeRC[0] + ");";
        offsetCoordsSnippet = 'vec2 offsetCoordsCR = mod(yTexCR, offsetShapeCR);';
        offsetUVSnippet =
            'vec2 offsetUV = (offsetCoordsCR + halfCR) / offsetShapeCR;';
        offsetValueSnippet = 'float offsetValue = texture2D(offset, offsetUV).r;';
        offsetOperationSnippet = 'offsetValue';
    }
    if (scaleTexShapeRC != null) {
        scaleSamplerSnippet = 'uniform sampler2D scale;';
        scaleShapeInitializationSnippet = "const vec2 scaleShapeCR = vec2(\n            " + scaleTexShapeRC[1] + ", " + scaleTexShapeRC[0] + ");";
        scaleCoordsSnippet = 'vec2 scaleCoordsCR = mod(yTexCR, scaleShapeCR);';
        scaleUVSnippet = 'vec2 scaleUV = (scaleCoordsCR + halfCR) / scaleShapeCR;';
        scaleValueSnippet = 'float scaleValue = texture2D(scale, scaleUV).r;';
        scaleOperationSnippet = 'inv *= scaleValue;';
    }
    return "\n    precision highp float;\n    uniform sampler2D x;\n    uniform sampler2D mean;\n    uniform sampler2D variance;\n    " + offsetSamplerSnippet + "\n    " + scaleSamplerSnippet + "\n\n    varying vec2 resultUV;\n\n    const vec2 xShapeCR = vec2(" + xTexShapeRC[1] + ", " + xTexShapeRC[0] + ");\n    const vec2 meanShapeCR = vec2(" + meanTexShapeRC[1] + ", " + meanTexShapeRC[0] + ");\n    const vec2 varianceShapeCR = vec2(\n        " + varianceTexShapeRC[1] + ", " + varianceTexShapeRC[0] + ");\n\n    " + offsetShapeInitializationSnippet + "\n    " + scaleShapeInitializationSnippet + "\n\n    const vec2 halfCR = vec2(0.5, 0.5);\n    const float varianceEpsilon = " + varianceEpsilon + ";\n\n    void main() {\n      vec2 yTexCR = floor(gl_FragCoord.xy);\n\n      vec2 meanCoordsCR = mod(yTexCR, meanShapeCR);\n      vec2 varianceCoordsCR = mod(yTexCR, varianceShapeCR);\n      " + offsetCoordsSnippet + "\n      " + scaleCoordsSnippet + "\n\n      vec2 meanUV = (meanCoordsCR + halfCR) / meanShapeCR;\n      vec2 varianceUV = (varianceCoordsCR + halfCR) / varianceShapeCR;\n      " + offsetUVSnippet + "\n      " + scaleUVSnippet + "\n\n      float xValue = texture2D(x, resultUV).r;\n      float meanValue = texture2D(mean, meanUV).r;\n      float varianceValue = texture2D(variance, varianceUV).r;\n      " + offsetValueSnippet + "\n      " + scaleValueSnippet + "\n\n      float inv = 1.0 / sqrt(varianceValue + varianceEpsilon);\n      " + scaleOperationSnippet + "\n      float xTimesInv = xValue * inv;\n      float meanTimesInvWithOffset = " + offsetOperationSnippet + "\n          - meanValue * inv;\n\n      gl_FragColor = vec4(xTimesInv + meanTimesInvWithOffset, 0, 0, 0);\n    }";
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function batchNormalization(gpgpu, program, x, xShapeRowCol, mean, meanShapeRowCol, variance, varianceShapeRowCol, offset, offsetShapeRowCol, scale, scaleShapeRowCol, result, resultShapeRowCol) {
    gpgpu.setOutputMatrixTexture(result, resultShapeRowCol[0], resultShapeRowCol[1]);
    gpgpu.setProgram(program);
    gpgpu.setInputMatrixTexture(x, 'x', 0);
    gpgpu.setInputMatrixTexture(mean, 'mean', 1);
    gpgpu.setInputMatrixTexture(variance, 'variance', 2);
    var nextIndex = 3;
    if (offset != null) {
        gpgpu.setInputMatrixTexture(offset, 'offset', nextIndex);
        nextIndex++;
    }
    if (scale != null) {
        gpgpu.setInputMatrixTexture(scale, 'scale', nextIndex);
    }
    gpgpu.executeProgram();
}
exports.batchNormalization = batchNormalization;

},{}],29:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var gpgpu_context_1 = require("./gpgpu_context");
function getFragmentShaderSource(aResultUV, bResultUV, op) {
    return "\n    precision highp float;\n    uniform sampler2D matrixA;\n    uniform sampler2D matrixB;\n    varying vec2 resultUV;\n\n    void main() {\n      float a = texture2D(matrixA, " + aResultUV + ").r;\n      float b = texture2D(matrixB, " + bResultUV + ").r;\n      " + op + "\n    }";
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function binaryOp(gpgpu, program, a, aShapeRowCol, b, bShapeRowCol, result, resultShapeRowCol) {
    gpgpu.setOutputMatrixTexture(result, resultShapeRowCol[0], resultShapeRowCol[1]);
    gpgpu.setProgram(program);
    gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
    gpgpu.setInputMatrixTexture(b, 'matrixB', 1);
    gpgpu.executeProgram();
}
exports.binaryOp = binaryOp;
function uploadBinaryOpDownload(a, aShape, b, bShape, fragmentShaderSource) {
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var program = gpgpu.createProgram(fragmentShaderSource);
    var aTexture = gpgpu.createMatrixTexture(aShape[0], aShape[1]);
    var bTexture = gpgpu.createMatrixTexture(bShape[0], bShape[1]);
    var resultShape = [Math.max(aShape[0], bShape[0]), Math.max(aShape[1], bShape[1])];
    var resultTexture = gpgpu.createMatrixTexture(resultShape[0], resultShape[1]);
    gpgpu.uploadMatrixToTexture(aTexture, aShape[0], aShape[1], a);
    gpgpu.uploadMatrixToTexture(bTexture, bShape[0], bShape[1], b);
    binaryOp(gpgpu, program, aTexture, aShape, bTexture, bShape, resultTexture, resultShape);
    var result = gpgpu.downloadMatrixFromTexture(resultTexture, resultShape[0], resultShape[1]);
    gpgpu.deleteMatrixTexture(aTexture);
    gpgpu.deleteMatrixTexture(bTexture);
    gpgpu.deleteMatrixTexture(resultTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return result;
}
exports.uploadBinaryOpDownload = uploadBinaryOpDownload;

},{"./gpgpu_context":35}],30:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../conv_util");
function getFragmentShaderSource(x1ShapeRCD, x2ShapeRCD, resultShapeRCD, axis) {
    var x1TexShapeRC = conv_util.computeTexShapeFrom3D(x1ShapeRCD);
    var x2TexShapeRC = conv_util.computeTexShapeFrom3D(x2ShapeRCD);
    var yAxes = ['yR', 'yC', 'yD'];
    var concatAxis = yAxes[axis];
    return "\n    precision highp float;\n    uniform sampler2D x1;\n    uniform sampler2D x2;\n\n    const vec2 x1ShapeCR = vec2(" + x1TexShapeRC[1] + ", " + x1TexShapeRC[0] + ");\n    const vec2 x2ShapeCR = vec2(" + x2TexShapeRC[1] + ".0, " + x2TexShapeRC[0] + ".0);\n\n    const vec2 halfCR = vec2(0.5, 0.5);\n\n    void main() {\n      vec2 yTexCR = floor(gl_FragCoord.xy);\n\n      // Map from 2D (yTexR, yTexC) to 3D (yR, yC, yD).\n      float yR = yTexCR.y;\n      float yC = floor(yTexCR.x / " + resultShapeRCD[2] + ".0);\n      float yD = mod(yTexCR.x, " + resultShapeRCD[2] + ".0);\n\n      float value = 0.0;\n\n      if (" + concatAxis + " < " + x1ShapeRCD[axis] + ".0) {\n        // Map yR, yC, yD back to x1 coordinates.\n        vec2 x1CR = vec2(yC * " + x1ShapeRCD[2] + ".0 + yD, yR);\n        vec2 x1UV = (x1CR + halfCR) / x1ShapeCR;\n        value = texture2D(x1, x1UV).r;\n      } else {\n        " + concatAxis + " = " + concatAxis + " - " + x1ShapeRCD[axis] + ".0;\n\n        // Map yR, yC, yD back to x2 coordinates.\n        vec2 x2CR = vec2(yC * " + x2ShapeRCD[2] + ".0 + yD, yR);\n        vec2 x2UV = (x2CR + halfCR) / x2ShapeCR;\n        value = texture2D(x2, x2UV).r;\n      }\n\n      gl_FragColor = vec4(value, 0.0, 0.0, 0.0);\n    }";
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function concat3D(gpgpu, program, x1, x2, result, resultShapeRC) {
    gpgpu.setOutputMatrixTexture(result, resultShapeRC[0], resultShapeRC[1]);
    gpgpu.setProgram(program);
    gpgpu.setInputMatrixTexture(x1, 'x1', 0);
    gpgpu.setInputMatrixTexture(x2, 'x2', 1);
    gpgpu.executeProgram();
}
exports.concat3D = concat3D;

},{"../conv_util":16}],31:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../conv_util");
var conv_gpu = require("./conv_gpu");
function getFragmentShaderDerWeightsSource(xShapeRowColDepth, fSize, outputDepth, stride, zeroPad) {
    var getMatrixValueOrZeroPad = conv_gpu.getFragmentShaderGetMatrixValueOrZeroPadSource();
    var inputDepth = xShapeRowColDepth[2];
    var xTexShapeRC = conv_util.computeTexShapeFrom3D(xShapeRowColDepth);
    var yShape = conv_util.computeOutputShape3D(xShapeRowColDepth, fSize, outputDepth, stride, zeroPad);
    var yNumRows = yShape[0];
    var yNumCols = yShape[1];
    var yTexShapeRC = conv_util.computeTexShapeFrom3D(yShape);
    var fSizeTimesInputDepth = fSize * inputDepth;
    var prologue = "\n    precision highp float;\n    uniform sampler2D x;\n    uniform sampler2D dy;\n  ";
    return prologue + '\n' + getMatrixValueOrZeroPad + '\n' +
        ("\n    const vec2 halfCR = vec2(0.5, 0.5);\n    const vec2 xShapeCR = vec2(" + xTexShapeRC[1] + ", " + xTexShapeRC[0] + ");\n    const vec2 dyShapeCR = vec2(" + yTexShapeRC[1] + ", " + yTexShapeRC[0] + ");\n\n    void main() {\n      vec2 wTexCR = floor(gl_FragCoord.xy);\n\n      // Map from 2D (wTexR, wTexC) to 4D (wR, wC, d1, d2).\n      float wR = floor(wTexCR.y / " + fSizeTimesInputDepth + ".0);\n      float wTexRLeftover = wTexCR.y - wR * " + fSizeTimesInputDepth + ".0;\n      float wC = floor(wTexRLeftover / " + inputDepth + ".0);\n      float d1 = mod(wTexRLeftover, " + inputDepth + ".0);\n      float d2 = wTexCR.x;\n\n      // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).\n      // ? = to be determined. : = across all values in that axis.\n      float dotProd = 0.0;\n      for (float yR = 0.0; yR < " + yNumRows + ".0; yR += 1.0) {\n        float xR = wR + yR * " + stride + ".0 - " + zeroPad + ".0;\n        float xTexR = xR;\n        float yTexR = yR;\n        for (float yC = 0.0; yC < " + yNumCols + ".0; yC += 1.0) {\n          float xC = wC + yC * " + stride + ".0 - " + zeroPad + ".0;\n\n          // Map from 3D (xR, xC, d1) to 2D (xTexR, xTexC).\n          // Map from 3D (yR, yC, d2) to 2D (yTexR, yTexC).\n          vec2 xyTexC = vec2(xC, yC) * vec2(" + inputDepth + ".0, " + outputDepth + ".0) +\n                        vec2(d1, d2);\n          float xTexC = xyTexC.x;\n          float yTexC = xyTexC.y;\n\n          // Read dy(yR, yC, d2).\n          vec2 dyUV = (vec2(yTexC, yTexR) + halfCR) / dyShapeCR;\n          float dyValue = texture2D(dy, dyUV).r;\n\n          // Read x(xR, xC, d1) (potentially zero-padded).\n          float xValue =\n            getMatrixValueOrZeroPad(x, xShapeCR, vec2(xTexC, xTexR));\n\n          dotProd += (xValue * dyValue);\n        }\n      }\n      gl_FragColor = vec4(dotProd, 0, 0, 0);\n    }");
}
exports.getFragmentShaderDerWeightsSource = getFragmentShaderDerWeightsSource;
function getFragmentShaderConvTransposeSource(xShapeRCD, fSize, origInputDepth, origStride, origPad, hasBias) {
    var pad = fSize - 1 - origPad;
    var xRows = xShapeRCD[0], xCols = xShapeRCD[1], origOutputDepth = xShapeRCD[2];
    var xTexShapeRC = conv_util.computeTexShapeFrom3D(xShapeRCD);
    var wTexShapeRC = conv_util.computeWeightsTexShape(origInputDepth, origOutputDepth, fSize);
    var getBiasValue = hasBias ?
        conv_gpu.getFragmentShaderGetBiasValueSource(origInputDepth) :
        '';
    var biasPrologue = hasBias ? 'uniform sampler2D biases;' : '';
    var biasOperation = hasBias ? 'dotProd += getBiasValue(biases, d2);' : '';
    var prologue = "\n    precision highp float;\n    uniform sampler2D x;\n    uniform sampler2D weights;\n    " + biasPrologue + "\n    ";
    return prologue + '\n' + getBiasValue + '\n' +
        ("\n    const vec2 halfCR = vec2(0.5, 0.5);\n    const vec2 xShapeCR = vec2(" + xTexShapeRC[1] + ", " + xTexShapeRC[0] + ");\n    const vec2 wShapeCR = vec2(" + wTexShapeRC[1] + ", " + wTexShapeRC[0] + ");\n\n    void main() {\n      vec2 yTexCR = floor(gl_FragCoord.xy);\n\n      // Map from 2D (yTexR, yTexC) to 3D (yR, yC, d2).\n      float yR = yTexCR.y;\n      float yC = floor(yTexCR.x / " + origInputDepth + ".0);\n      float d2 = mod(yTexCR.x, " + origInputDepth + ".0);\n\n      vec2 xRCCorner = vec2(yR, yC) - vec2(" + pad + ".0, " + pad + ".0);\n      float xRCorner = xRCCorner.x;\n      float xCCorner = xRCCorner.y;\n\n      // Convolve x(?, ?, d1) with w(:, :, d2, d1) to get y(yR, yC, d2).\n      // ? = to be determined. : = across all values in that axis.\n      float dotProd = 0.0;\n      for (float wR = 0.0; wR < " + fSize + ".0; wR += 1.0) {\n\n        float xR = (xRCorner + wR) / " + origStride + ".0;\n        // TODO(smilkov): Splice this with another version where you call\n        // getMatrixValueOrZeroPad(). Here and below.\n        if (xR < 0.0 || xR >= " + xRows + ".0 || fract(xR) > 0.0) {\n          continue;\n        }\n\n        float wRPerm = " + fSize + ".0 - 1.0 - wR;\n        float xTexR = xR;\n\n        for (float wC = 0.0; wC < " + fSize + ".0; wC += 1.0) {\n\n          float xC = (xCCorner + wC) / " + origStride + ".0;\n          if (xC < 0.0 || xC >= " + xCols + ".0 || fract(xC) > 0.0) {\n            continue;\n          }\n\n          float wCPerm = " + fSize + ".0 - 1.0 - wC;\n          float wTexR = wRPerm * " + fSize + ".0 * " + origInputDepth + ".0 +\n                        wCPerm * " + origInputDepth + ".0 + d2;\n\n          for (float d1 = 0.0; d1 < " + origOutputDepth + ".0; d1 += 1.0) {\n            float xTexC = xC * " + origOutputDepth + ".0 + d1;\n            float wTexC = d1;\n\n            // Read x(xR, xC, d1).\n            vec2 xUV = (vec2(xTexC, xTexR) + halfCR) / xShapeCR;\n            float xValue = texture2D(x, xUV).r;\n\n            // Read w(wRPerm, wCPerm, d2, d1).\n            vec2 wUV = (vec2(wTexC, wTexR) + halfCR) / wShapeCR;\n            float wValue = texture2D(weights, wUV).r;\n\n            dotProd += xValue * wValue;\n          }\n        }\n      }\n      " + biasOperation + "\n      gl_FragColor = vec4(dotProd, 0, 0, 0);\n    }");
}
exports.getFragmentShaderConvTransposeSource = getFragmentShaderConvTransposeSource;
function getFragmentShaderDerBiasSource(dyShapeRCD) {
    var dyTexShapeRC = conv_util.computeTexShapeFrom3D(dyShapeRCD);
    var yNumRows = dyShapeRCD[0], yNumCols = dyShapeRCD[1], outputDepth = dyShapeRCD[2];
    return "\n    precision highp float;\n    uniform sampler2D dy;\n\n    const vec2 halfCR = vec2(0.5, 0.5);\n    const vec2 dyShapeCR = vec2(" + dyTexShapeRC[1] + ", " + dyTexShapeRC[0] + ");\n\n    void main() {\n      vec2 biasTexCR = floor(gl_FragCoord.xy);\n\n      // The bias texture RC shape is [1, d2].\n      float d2 = biasTexCR.x;\n\n      float derBias = 0.0;\n      for (float yR = 0.0; yR < " + yNumRows + ".0; yR += 1.0) {\n        float yTexR = yR;\n\n        for (float yC = 0.0; yC < " + yNumCols + ".0; yC += 1.0) {\n          // Map from 3D (yR, yC, d2) to 2D (yTexR, yTexC).\n          float yTexC = yC * " + outputDepth + ".0 + d2;\n\n          // Read dy(yR, yC, d2).\n          vec2 dyUV = (vec2(yTexC, yTexR) + halfCR) / dyShapeCR;\n          float dyValue = texture2D(dy, dyUV).r;\n\n          derBias += dyValue;\n        }\n      }\n      gl_FragColor = vec4(derBias, 0, 0, 0);\n    }";
}
exports.getFragmentShaderDerBiasSource = getFragmentShaderDerBiasSource;
function derBias(gpgpu, program, dyTex, result, resultTexShapeRC) {
    gpgpu.setOutputMatrixTexture(result, resultTexShapeRC[0], resultTexShapeRC[1]);
    gpgpu.setProgram(program);
    gpgpu.setInputMatrixTexture(dyTex, 'dy', 0);
    gpgpu.executeProgram();
}
exports.derBias = derBias;
function derWeights(gpgpu, program, xTex, dyTex, result, resultTexShapeRC) {
    gpgpu.setOutputMatrixTexture(result, resultTexShapeRC[0], resultTexShapeRC[1]);
    gpgpu.setProgram(program);
    gpgpu.setInputMatrixTexture(xTex, 'x', 0);
    gpgpu.setInputMatrixTexture(dyTex, 'dy', 1);
    gpgpu.executeProgram();
}
exports.derWeights = derWeights;
function convTranspose(gpgpu, program, xTex, weightsTex, biasesTex, resultTex, resultTexShapeRC) {
    gpgpu.setOutputMatrixTexture(resultTex, resultTexShapeRC[0], resultTexShapeRC[1]);
    gpgpu.setProgram(program);
    gpgpu.setInputMatrixTexture(xTex, 'x', 0);
    gpgpu.setInputMatrixTexture(weightsTex, 'weights', 1);
    if (biasesTex != null) {
        gpgpu.setInputMatrixTexture(biasesTex, 'biases', 2);
    }
    gpgpu.executeProgram();
}
exports.convTranspose = convTranspose;

},{"../conv_util":16,"./conv_gpu":32}],32:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../conv_util");
function getFragmentShaderPrologueSource() {
    return "\n    precision highp float;\n    uniform sampler2D x;\n    uniform sampler2D weights;\n    uniform sampler2D biases;\n    varying vec2 resultUV;";
}
exports.getFragmentShaderPrologueSource = getFragmentShaderPrologueSource;
function getFragmentShaderGetMatrixValueOrZeroPadSource() {
    return "\n    float getMatrixValueOrZeroPad(in sampler2D matrix, vec2 matrixShapeCR,\n        vec2 requestedCR) {\n      vec2 uv = (requestedCR + vec2(0.5, 0.5)) / matrixShapeCR;\n      float value = texture2D(matrix, uv).r;\n      bool lessThanZero = any(lessThan(uv, vec2(0, 0)));\n      bool greaterThanOne = any(greaterThan(uv, vec2(1, 1)));\n      bool outside = lessThanZero || greaterThanOne;\n      return mix(value, 0.0, float(outside));\n    }";
}
exports.getFragmentShaderGetMatrixValueOrZeroPadSource = getFragmentShaderGetMatrixValueOrZeroPadSource;
function getFragmentShaderConvolveSource(xShapeRCD, fSize, outputDepth, stride, pad, hasBias) {
    var xRows = xShapeRCD[0], xCols = xShapeRCD[1], inputDepth = xShapeRCD[2];
    var xTexShapeRC = conv_util.computeTexShapeFrom3D(xShapeRCD);
    var wTexShapeRC = conv_util.computeWeightsTexShape(inputDepth, outputDepth, fSize);
    return "\n    const vec2 halfCR = vec2(0.5, 0.5);\n    const vec2 xShapeCR = vec2(" + xTexShapeRC[1] + ", " + xTexShapeRC[0] + ");\n    const vec2 wShapeCR = vec2(" + wTexShapeRC[1] + ", " + wTexShapeRC[0] + ");\n\n    void main() {\n      vec2 yTexCR = floor(gl_FragCoord.xy);\n\n      // Map from 2D (yTexR, yTexC) to 3D (yR, yC, d2).\n      float yR = yTexCR.y;\n      float yC = floor(yTexCR.x / " + outputDepth + ".0);\n      float d2 = mod(yTexCR.x, " + outputDepth + ".0);\n      float wTexC = d2;\n\n      vec2 xRCCorner = vec2(yR, yC) * vec2(" + stride + ", " + stride + ") -\n          vec2(" + pad + ".0, " + pad + ".0);\n      float xRCorner = xRCCorner.x;\n      float xCCorner = xRCCorner.y;\n\n      // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).\n      // ? = to be determined. : = across all values in that axis.\n      float dotProd = 0.0;\n      for (float wR = 0.0; wR < " + fSize + ".0; wR += 1.0) {\n        float xR = xRCorner + wR;\n        float xTexR = xR;\n\n        for (float wC = 0.0; wC < " + fSize + ".0; wC += 1.0) {\n          float xC = xCCorner + wC;\n\n          for (float d1 = 0.0; d1 < " + inputDepth + ".0; d1 += 1.0) {\n            float xTexC = xC * " + inputDepth + ".0 + d1;\n            float wTexR = wR * " + fSize * inputDepth + ".0 +\n                wC * " + inputDepth + ".0 + d1;\n\n            float xValue =\n                getMatrixValueOrZeroPad(x, xShapeCR, vec2(xTexC, xTexR));\n\n            // Read w(wR, wC, d1, d2).\n            vec2 wUV = (vec2(wTexC, wTexR) + halfCR) / wShapeCR;\n            float wValue = texture2D(weights, wUV).r;\n\n            dotProd += xValue * wValue;\n          }\n        }\n      }\n      if (" + hasBias + ") {\n        dotProd += getBiasValue(biases, d2);\n      }\n      gl_FragColor = vec4(dotProd, 0, 0, 0);\n    }";
}
exports.getFragmentShaderConvolveSource = getFragmentShaderConvolveSource;
function getFragmentShaderGetBiasValueSource(outputDepth) {
    return "\n    float getBiasValue(in sampler2D bias, float biasC) {\n      const vec2 biasShapeCR = vec2(" + outputDepth + ", 1);\n      vec2 biasCR = vec2(mod(biasC, " + outputDepth + ".0), 0);\n      vec2 biasUV = (biasCR + vec2(0.5, 0.5)) / biasShapeCR;\n      return texture2D(bias, biasUV).r;\n    }";
}
exports.getFragmentShaderGetBiasValueSource = getFragmentShaderGetBiasValueSource;
function getFragmentShaderSource(aShapeRowColDepth, resultDepth, fieldSize, stride, zeroPad, hasBias) {
    var aShapeRC = conv_util.computeTexShapeFrom3D(aShapeRowColDepth);
    var weightShapeRC = conv_util.computeWeightsTexShape(aShapeRowColDepth[2], resultDepth, fieldSize);
    var prologue = getFragmentShaderPrologueSource();
    var getMatrixValueOrZeroPad = getFragmentShaderGetMatrixValueOrZeroPadSource();
    var convolve = getFragmentShaderConvolveSource(aShapeRowColDepth, fieldSize, resultDepth, stride, zeroPad, hasBias);
    var getBiasValue = getFragmentShaderGetBiasValueSource(resultDepth);
    return [
        prologue,
        getMatrixValueOrZeroPad,
        getBiasValue,
        convolve,
    ].join('\n');
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function convolve(gpgpu, program, a, weights, biases, result, resultShapeRowCol) {
    gpgpu.setOutputMatrixTexture(result, resultShapeRowCol[0], resultShapeRowCol[1]);
    gpgpu.setProgram(program);
    gpgpu.setInputMatrixTexture(a, 'x', 0);
    gpgpu.setInputMatrixTexture(weights, 'weights', 1);
    if (biases != null) {
        gpgpu.setInputMatrixTexture(biases, 'biases', 2);
    }
    gpgpu.executeProgram();
}
exports.convolve = convolve;

},{"../conv_util":16}],33:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function getFragmentShaderSource(sourceShapeRowCol, sourceSizeRowCol, destSizeRowCol) {
    return "\n    precision highp float;\n    uniform sampler2D source;\n    uniform vec2 sourceStartCR;\n    uniform vec2 destStartCR;\n\n    const vec2 sourceShapeCR =\n      vec2(" + sourceShapeRowCol[1] + ", " + sourceShapeRowCol[0] + ");\n    const vec2 sourceSizeCR =\n      vec2(" + sourceSizeRowCol[1] + ", " + sourceSizeRowCol[0] + ");\n    const vec2 destSizeCR =\n      vec2(" + destSizeRowCol[1] + ", " + destSizeRowCol[0] + ");\n\n    void main() {\n      vec2 destOffsetCR = floor(gl_FragCoord.xy) - destStartCR;\n      float destOffsetFlat = (destOffsetCR.y * destSizeCR.x) + destOffsetCR.x;\n      vec2 sourceOffsetCR = vec2(mod(destOffsetFlat, sourceSizeCR.x),\n        floor(destOffsetFlat / sourceSizeCR.x));\n      vec2 sourceCR = sourceStartCR + sourceOffsetCR;\n      vec2 sourceUV = (sourceCR + vec2(0.5, 0.5)) / sourceShapeCR;\n      gl_FragColor = texture2D(source, sourceUV);\n    }";
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function copy(gpgpu, program, source, sourceShapeRowCol, sourceStartRowCol, sourceSizeRowCol, dest, destShapeRowCol, destStartRowCol, destSizeRowCol) {
    gpgpu.setOutputMatrixTexture(dest, destShapeRowCol[0], destShapeRowCol[1]);
    gpgpu.setOutputMatrixWriteRegion(destStartRowCol[0], destSizeRowCol[0], destStartRowCol[1], destSizeRowCol[1]);
    gpgpu.setProgram(program);
    gpgpu.setInputMatrixTexture(source, 'source', 0);
    var sourceStartCRLoc = gpgpu.getUniformLocation('sourceStartCR');
    gpgpu.gl.uniform2f(sourceStartCRLoc, sourceStartRowCol[1], sourceStartRowCol[0]);
    var destStartCRLoc = gpgpu.getUniformLocation('destStartCR');
    gpgpu.gl.uniform2f(destStartCRLoc, destStartRowCol[1], destStartRowCol[0]);
    gpgpu.executeProgram();
}
exports.copy = copy;

},{}],34:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var unaryop_gpu = require("./unaryop_gpu");
function getExpUnaryOp() {
    return 'gl_FragColor = vec4(exp(value), 0, 0, 0);';
}
function getFragmentShaderSource() {
    return unaryop_gpu.getFragmentShaderSource(getExpUnaryOp());
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function exp(gpgpu, expProgram, a, rows, columns, result) {
    unaryop_gpu.unaryOp(gpgpu, expProgram, a, rows, columns, result);
}
exports.exp = exp;
function uploadExpDownload(a, rows, columns) {
    return unaryop_gpu.uploadUnaryOpDownload(a, rows, columns, getExpUnaryOp());
}
exports.uploadExpDownload = uploadExpDownload;

},{"./unaryop_gpu":57}],35:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var gpgpu_util = require("./gpgpu_util");
var tex_util = require("./tex_util");
var webgl_util = require("./webgl_util");
var GPGPUContext = (function () {
    function GPGPUContext(gl) {
        this.outputTexture = null;
        this.program = null;
        this.disposed = false;
        this.autoDebugValidate = false;
        if (gl != null) {
            this.gl = gl;
        }
        else {
            this.gl = gpgpu_util.createWebGLContext();
        }
        if (!webgl_util.isWebGL2Enabled()) {
            this.textureFloatExtension =
                webgl_util.getExtensionOrThrow(this.gl, 'OES_texture_float');
        }
        else {
            this.colorBufferFloatExtension =
                webgl_util.getExtensionOrThrow(this.gl, 'EXT_color_buffer_float');
        }
        this.loseContextExtension =
            webgl_util.getExtensionOrThrow(this.gl, 'WEBGL_lose_context');
        this.vertexBuffer = gpgpu_util.createVertexBuffer(this.gl);
        this.indexBuffer = gpgpu_util.createIndexBuffer(this.gl);
        this.framebuffer = webgl_util.createFramebuffer(this.gl);
    }
    GPGPUContext.prototype.dispose = function () {
        var _this = this;
        this.throwIfDisposed();
        if (this.program != null) {
            console.warn('Disposing a GPGPUContext that still has a bound WebGLProgram.' +
                ' This is probably a resource leak, delete the program with ' +
                'GPGPUContext.deleteProgram before disposing.');
        }
        if (this.outputTexture != null) {
            console.warn('Disposing a GPGPUContext that still has a bound output matrix ' +
                'texture.  This is probably a resource leak, delete the output ' +
                'matrix texture with GPGPUContext.deleteMatrixTexture before ' +
                'disposing.');
        }
        var gl = this.gl;
        webgl_util.callAndCheck(gl, function () { return gl.finish(); });
        webgl_util.callAndCheck(gl, function () { return gl.bindFramebuffer(gl.FRAMEBUFFER, null); });
        webgl_util.callAndCheck(gl, function () { return gl.deleteFramebuffer(_this.framebuffer); });
        webgl_util.callAndCheck(gl, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, null); });
        webgl_util.callAndCheck(gl, function () { return gl.deleteBuffer(_this.vertexBuffer); });
        webgl_util.callAndCheck(gl, function () { return gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null); });
        webgl_util.callAndCheck(gl, function () { return gl.deleteBuffer(_this.indexBuffer); });
        this.loseContextExtension.loseContext();
        this.disposed = true;
    };
    GPGPUContext.prototype.enableAutomaticDebugValidation = function (enabled) {
        this.autoDebugValidate = enabled;
        webgl_util.enableDebugWebGLErrorChecking(enabled);
    };
    GPGPUContext.prototype.createMatrixTexture = function (rows, columns) {
        this.throwIfDisposed();
        return gpgpu_util.createMatrixTexture(this.gl, rows, columns);
    };
    GPGPUContext.prototype.uploadPixelDataToTexture = function (texture, pixels) {
        this.throwIfDisposed();
        gpgpu_util.uploadPixelDataToTexture(this.gl, texture, pixels);
    };
    GPGPUContext.prototype.createPackedMatrixTexture = function (rows, columns) {
        this.throwIfDisposed();
        return gpgpu_util.createPackedMatrixTexture(this.gl, rows, columns);
    };
    GPGPUContext.prototype.deleteMatrixTexture = function (texture) {
        var _this = this;
        this.throwIfDisposed();
        if (this.outputTexture === texture) {
            webgl_util.unbindColorTextureFromFramebuffer(this.gl, this.framebuffer);
            this.outputTexture = null;
        }
        webgl_util.callAndCheck(this.gl, function () { return _this.gl.deleteTexture(texture); });
    };
    GPGPUContext.prototype.uploadMatrixToTexture = function (texture, rows, columns, matrix) {
        this.throwIfDisposed();
        var numChannels = 1;
        return gpgpu_util.uploadMatrixToTexture(this.gl, texture, rows, columns, matrix, numChannels);
    };
    GPGPUContext.prototype.uploadMatrixToPackedTexture = function (texture, rows, columns, matrix) {
        this.throwIfDisposed();
        return gpgpu_util.uploadMatrixToPackedTexture(this.gl, texture, rows, columns, matrix);
    };
    GPGPUContext.prototype.downloadMatrixFromTexture = function (texture, rows, columns) {
        var _this = this;
        return this.downloadMatrixDriver(texture, function () {
            return gpgpu_util.downloadMatrixFromOutputTexture(_this.gl, rows, columns);
        });
    };
    GPGPUContext.prototype.downloadMatrixFromPackedTexture = function (texture, rows, columns) {
        var _this = this;
        return this.downloadMatrixDriver(texture, function () { return gpgpu_util.downloadMatrixFromPackedOutputTexture(_this.gl, rows, columns); });
    };
    GPGPUContext.prototype.createProgram = function (fragmentShaderSource) {
        this.throwIfDisposed();
        var gl = this.gl;
        var fragmentShader = webgl_util.createFragmentShader(gl, fragmentShaderSource);
        var vertexShader = gpgpu_util.createVertexShader(gl);
        var program = webgl_util.createProgram(gl);
        webgl_util.callAndCheck(gl, function () { return gl.attachShader(program, vertexShader); });
        webgl_util.callAndCheck(gl, function () { return gl.attachShader(program, fragmentShader); });
        webgl_util.linkProgram(gl, program);
        if (this.autoDebugValidate) {
            webgl_util.validateProgram(gl, program);
        }
        webgl_util.callAndCheck(gl, function () { return gl.detachShader(program, vertexShader); });
        webgl_util.callAndCheck(gl, function () { return gl.deleteShader(vertexShader); });
        webgl_util.callAndCheck(gl, function () { return gl.detachShader(program, fragmentShader); });
        webgl_util.callAndCheck(gl, function () { return gl.deleteShader(fragmentShader); });
        return program;
    };
    GPGPUContext.prototype.deleteProgram = function (program) {
        var _this = this;
        this.throwIfDisposed();
        if (program === this.program) {
            this.program = null;
        }
        if (program != null) {
            webgl_util.callAndCheck(this.gl, function () { return _this.gl.deleteProgram(program); });
        }
    };
    GPGPUContext.prototype.setProgram = function (program) {
        var _this = this;
        this.throwIfDisposed();
        this.program = program;
        if ((this.program != null) && this.autoDebugValidate) {
            webgl_util.validateProgram(this.gl, this.program);
        }
        webgl_util.callAndCheck(this.gl, function () { return _this.gl.useProgram(program); });
    };
    GPGPUContext.prototype.getUniformLocation = function (uniformName) {
        this.throwIfDisposed();
        this.throwIfNoProgram();
        return webgl_util.getProgramUniformLocationOrThrow(this.gl, this.program, uniformName);
    };
    GPGPUContext.prototype.setInputMatrixTexture = function (inputMatrixTexture, uniformName, textureUnit) {
        this.throwIfDisposed();
        this.throwIfNoProgram();
        webgl_util.bindTextureToProgramUniformSampler(this.gl, this.program, inputMatrixTexture, uniformName, textureUnit);
    };
    GPGPUContext.prototype.setOutputMatrixTexture = function (outputMatrixTexture, rows, columns) {
        this.setOutputMatrixTextureDriver(outputMatrixTexture, columns, rows);
    };
    GPGPUContext.prototype.setOutputPackedMatrixTexture = function (outputPackedMatrixTexture, rows, columns) {
        this.throwIfDisposed();
        var _a = tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
        this.setOutputMatrixTextureDriver(outputPackedMatrixTexture, width, height);
    };
    GPGPUContext.prototype.setOutputMatrixWriteRegion = function (startRow, numRows, startColumn, numColumns) {
        this.setOutputMatrixWriteRegionDriver(startColumn, startRow, numColumns, numRows);
    };
    GPGPUContext.prototype.setOutputPackedMatrixWriteRegion = function (startRow, numRows, startColumn, numColumns) {
        throw new Error('setOutputPackedMatrixWriteRegion not implemented.');
    };
    GPGPUContext.prototype.debugValidate = function () {
        if (this.program != null) {
            webgl_util.validateProgram(this.gl, this.program);
        }
        webgl_util.validateFramebuffer(this.gl);
    };
    GPGPUContext.prototype.executeProgram = function () {
        this.throwIfDisposed();
        this.throwIfNoProgram();
        var gl = this.gl;
        gpgpu_util.bindVertexProgramAttributeStreams(gl, this.program, this.vertexBuffer);
        if (this.autoDebugValidate) {
            this.debugValidate();
        }
        webgl_util.callAndCheck(gl, function () { return gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0); });
    };
    GPGPUContext.prototype.blockUntilAllProgramsCompleted = function () {
        var _this = this;
        this.throwIfDisposed();
        webgl_util.callAndCheck(this.gl, function () { return _this.gl.finish(); });
    };
    GPGPUContext.prototype.downloadMatrixDriver = function (texture, downloadAndDecode) {
        this.throwIfDisposed();
        webgl_util.bindColorTextureToFramebuffer(this.gl, texture, this.framebuffer);
        var result = downloadAndDecode();
        if (this.outputTexture != null) {
            webgl_util.bindColorTextureToFramebuffer(this.gl, this.outputTexture, this.framebuffer);
            if (this.autoDebugValidate) {
                webgl_util.validateFramebuffer(this.gl);
            }
        }
        else {
            webgl_util.unbindColorTextureFromFramebuffer(this.gl, this.framebuffer);
        }
        return result;
    };
    GPGPUContext.prototype.setOutputMatrixTextureDriver = function (outputMatrixTextureMaybePacked, width, height) {
        this.throwIfDisposed();
        var gl = this.gl;
        webgl_util.bindColorTextureToFramebuffer(gl, outputMatrixTextureMaybePacked, this.framebuffer);
        if (this.autoDebugValidate) {
            webgl_util.validateFramebuffer(gl);
        }
        this.outputTexture = outputMatrixTextureMaybePacked;
        webgl_util.callAndCheck(gl, function () { return gl.viewport(0, 0, width, height); });
        webgl_util.callAndCheck(gl, function () { return gl.scissor(0, 0, width, height); });
    };
    GPGPUContext.prototype.setOutputMatrixWriteRegionDriver = function (x, y, width, height) {
        var _this = this;
        this.throwIfDisposed();
        webgl_util.callAndCheck(this.gl, function () { return _this.gl.scissor(x, y, width, height); });
    };
    GPGPUContext.prototype.throwIfDisposed = function () {
        if (this.disposed) {
            throw new Error('Attempted to use disposed GPGPUContext.');
        }
    };
    GPGPUContext.prototype.throwIfNoProgram = function () {
        if (this.program == null) {
            throw new Error('No GPU program is currently set.');
        }
    };
    return GPGPUContext;
}());
exports.GPGPUContext = GPGPUContext;

},{"./gpgpu_util":36,"./tex_util":54,"./webgl_util":58}],36:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tex_util = require("./tex_util");
var webgl_util = require("./webgl_util");
function getWebGLContextAttributes() {
    return {
        alpha: false,
        antialias: false,
        premultipliedAlpha: false,
        preserveDrawingBuffer: false,
        depth: false,
        stencil: false,
        failIfMajorPerformanceCaveat: true
    };
}
exports.getWebGLContextAttributes = getWebGLContextAttributes;
function createWebGLContext(canvas) {
    var attributes = getWebGLContextAttributes();
    var gl;
    if (canvas != null) {
        gl = webgl_util.createWebGLRenderingContextFromCanvas(canvas, attributes);
    }
    else {
        gl = webgl_util.createWebGLRenderingContext(attributes);
    }
    webgl_util.callAndCheck(gl, function () { return gl.disable(gl.DEPTH_TEST); });
    webgl_util.callAndCheck(gl, function () { return gl.disable(gl.STENCIL_TEST); });
    webgl_util.callAndCheck(gl, function () { return gl.disable(gl.BLEND); });
    webgl_util.callAndCheck(gl, function () { return gl.disable(gl.DITHER); });
    webgl_util.callAndCheck(gl, function () { return gl.disable(gl.POLYGON_OFFSET_FILL); });
    webgl_util.callAndCheck(gl, function () { return gl.disable(gl.SAMPLE_COVERAGE); });
    webgl_util.callAndCheck(gl, function () { return gl.enable(gl.SCISSOR_TEST); });
    webgl_util.callAndCheck(gl, function () { return gl.enable(gl.CULL_FACE); });
    webgl_util.callAndCheck(gl, function () { return gl.cullFace(gl.BACK); });
    return gl;
}
exports.createWebGLContext = createWebGLContext;
function createVertexShader(gl) {
    var vertexShaderSource = "\n    precision highp float;\n    attribute vec3 clipSpacePos;\n    attribute vec2 uv;\n    varying vec2 resultUV;\n\n    void main() {\n      gl_Position = vec4(clipSpacePos, 1);\n      resultUV = uv;\n    }";
    return webgl_util.createVertexShader(gl, vertexShaderSource);
}
exports.createVertexShader = createVertexShader;
function createVertexBuffer(gl) {
    var vertexArray = new Float32Array([-1, 1, 0, 0, 1, -1, -1, 0, 0, 0, 1, 1, 0, 1, 1, 1, -1, 0, 1, 0]);
    return webgl_util.createStaticVertexBuffer(gl, vertexArray);
}
exports.createVertexBuffer = createVertexBuffer;
function createIndexBuffer(gl) {
    var triangleVertexIndices = new Uint16Array([0, 1, 2, 2, 1, 3]);
    return webgl_util.createStaticIndexBuffer(gl, triangleVertexIndices);
}
exports.createIndexBuffer = createIndexBuffer;
function getTextureInternalFormat(gl, numChannels) {
    if (webgl_util.isWebGL2Enabled()) {
        if (numChannels === 4) {
            return gl.RGBA32F;
        }
        return gl.R32F;
    }
    return gl.RGBA;
}
function getTextureFormat(gl, numChannels) {
    if (webgl_util.isWebGL2Enabled() && numChannels === 1) {
        return gl.RED;
    }
    return gl.RGBA;
}
function createAndConfigureTexture(gl, width, height, numChannels) {
    webgl_util.validateTextureSize(gl, width, height);
    var texture = webgl_util.createTexture(gl);
    var tex2d = gl.TEXTURE_2D;
    var internalFormat = getTextureInternalFormat(gl, numChannels);
    var format = getTextureFormat(gl, numChannels);
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(tex2d, texture); });
    webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); });
    webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); });
    webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MIN_FILTER, gl.NEAREST); });
    webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MAG_FILTER, gl.NEAREST); });
    webgl_util.callAndCheck(gl, function () { return gl.texImage2D(tex2d, 0, internalFormat, width, height, 0, format, gl.FLOAT, null); });
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
    return texture;
}
function createMatrixTexture(gl, rows, columns) {
    var _a = tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
    var numChannels = 1;
    return createAndConfigureTexture(gl, width, height, numChannels);
}
exports.createMatrixTexture = createMatrixTexture;
function createColorMatrixTexture(gl, rows, columns) {
    var _a = tex_util.getColorMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
    var numChannels = 4;
    return createAndConfigureTexture(gl, width, height, numChannels);
}
exports.createColorMatrixTexture = createColorMatrixTexture;
function createPackedMatrixTexture(gl, rows, columns) {
    var _a = tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
    var numChannels = 4;
    return createAndConfigureTexture(gl, width, height, numChannels);
}
exports.createPackedMatrixTexture = createPackedMatrixTexture;
function bindVertexProgramAttributeStreams(gl, program, vertexBuffer) {
    var posOffset = 0;
    var uvOffset = 3 * 4;
    var stride = (3 * 4) + (2 * 4);
    webgl_util.callAndCheck(gl, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer); });
    webgl_util.bindVertexBufferToProgramAttribute(gl, program, 'clipSpacePos', vertexBuffer, 3, stride, posOffset);
    try {
        webgl_util.bindVertexBufferToProgramAttribute(gl, program, 'uv', vertexBuffer, 2, stride, uvOffset);
    }
    catch (e) {
        if (!e.hasOwnProperty('namedVertexAttributeNotFound')) {
            throw e;
        }
    }
}
exports.bindVertexProgramAttributeStreams = bindVertexProgramAttributeStreams;
function uploadPixelDataToTexture(gl, texture, pixels) {
    var numChannels = 4;
    var internalFormat = getTextureInternalFormat(gl, numChannels);
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, texture); });
    webgl_util.callAndCheck(gl, function () { return gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, gl.RGBA, gl.FLOAT, pixels); });
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
}
exports.uploadPixelDataToTexture = uploadPixelDataToTexture;
function uploadDataToTexture(gl, texture, width, height, data, numChannels) {
    var textureFormat = getTextureFormat(gl, numChannels);
    webgl_util.validateTextureSize(gl, width, height);
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, texture); });
    webgl_util.callAndCheck(gl, function () { return gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, textureFormat, gl.FLOAT, data); });
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
}
function uploadMatrixToTexture(gl, texture, rows, columns, matrix, numChannels) {
    var _a = tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
    var channelsPerTexture = numChannels === 1 ? webgl_util.getChannelsPerTexture() : numChannels;
    var unpackedArray = new Float32Array(tex_util.getUnpackedArraySizeFromMatrixSize(matrix.length, channelsPerTexture));
    tex_util.encodeMatrixToUnpackedArray(matrix, unpackedArray, channelsPerTexture);
    uploadDataToTexture(gl, texture, w, h, unpackedArray, numChannels);
}
exports.uploadMatrixToTexture = uploadMatrixToTexture;
function uploadMatrixToPackedTexture(gl, texture, rows, columns, matrix) {
    var _a = tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
    var packedRGBA = new Float32Array(tex_util.getPackedRGBAArraySizeFromMatrixShape(rows, columns));
    tex_util.encodeMatrixToPackedRGBA(matrix, rows, columns, packedRGBA);
    var numChannels = 4;
    uploadDataToTexture(gl, texture, w, h, packedRGBA, numChannels);
}
exports.uploadMatrixToPackedTexture = uploadMatrixToPackedTexture;
function downloadMatrixFromOutputTexture(gl, rows, columns) {
    var _a = tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
    var channelsPerTexture = 4;
    var unpackedArray = new Float32Array(tex_util.getUnpackedArraySizeFromMatrixSize(rows * columns, channelsPerTexture));
    var textureFormat = getTextureFormat(gl, channelsPerTexture);
    webgl_util.callAndCheck(gl, function () { return gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, unpackedArray); });
    var matrix = new Float32Array(rows * columns);
    tex_util.decodeMatrixFromUnpackedArray(unpackedArray, matrix, channelsPerTexture);
    return matrix;
}
exports.downloadMatrixFromOutputTexture = downloadMatrixFromOutputTexture;
function downloadMatrixFromPackedOutputTexture(gl, rows, columns) {
    var _a = tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
    var packedRGBA = new Float32Array(tex_util.getPackedRGBAArraySizeFromMatrixShape(rows, columns));
    webgl_util.callAndCheck(gl, function () { return gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, packedRGBA); });
    var matrix = new Float32Array(rows * columns);
    return tex_util.decodeMatrixFromPackedRGBA(packedRGBA, rows, columns, matrix);
}
exports.downloadMatrixFromPackedOutputTexture = downloadMatrixFromPackedOutputTexture;

},{"./tex_util":54,"./webgl_util":58}],37:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var unaryop_gpu = require("./unaryop_gpu");
function getLogUnaryOp() {
    return 'gl_FragColor = vec4(log(value), 0, 0, 0);';
}
function getFragmentShaderSource() {
    return unaryop_gpu.getFragmentShaderSource(getLogUnaryOp());
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function log(gpgpu, logProgram, a, rows, columns, result) {
    unaryop_gpu.unaryOp(gpgpu, logProgram, a, rows, columns, result);
}
exports.log = log;
function uploadLogDownload(a, rows, columns) {
    return unaryop_gpu.uploadUnaryOpDownload(a, rows, columns, getLogUnaryOp());
}
exports.uploadLogDownload = uploadLogDownload;

},{"./unaryop_gpu":57}],38:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var gpgpu_context_1 = require("./gpgpu_context");
function getFragmentShaderSource(rows, columns) {
    return "\n    precision highp float;\n    uniform sampler2D matrixA;\n    varying vec2 resultUV;\n\n    const vec2 aDimCR = vec2(" + columns + ".0, " + rows + ".0);\n    const vec2 halfCR = vec2(0.5, 0.5);\n\n    void main() {\n      float aMax = texture2D(matrixA, halfCR / aDimCR).r;\n      for (float r = 0.0; r < aDimCR.y; r += 1.0) {\n        for (float c = 0.0; c < aDimCR.x; c += 1.0) {\n          vec2 uv = (vec2(c, r) + halfCR) / aDimCR;\n          float aCur = texture2D(matrixA, uv).r;\n          aMax = max(aMax, aCur);\n        }\n      }\n\n      float expSum = 0.0;\n      for (float r = 0.0; r < aDimCR.y; r += 1.0) {\n        for (float c = 0.0; c < aDimCR.x; c += 1.0) {\n          vec2 uv = (vec2(c, r) + halfCR) / aDimCR;\n          float aCur = texture2D(matrixA, uv).r;\n          expSum += exp(aCur - aMax);\n        }\n      }\n\n      gl_FragColor = vec4(aMax + log(expSum), 0, 0, 0);\n    }";
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function logSumExp(gpgpu, logSumExpProgram, a, rows, columns, result) {
    gpgpu.setOutputMatrixTexture(result, 1, 1);
    gpgpu.setProgram(logSumExpProgram);
    gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
    gpgpu.executeProgram();
}
exports.logSumExp = logSumExp;
function uploadLogSumExpDownload(a, rows, columns) {
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var program = gpgpu.createProgram(getFragmentShaderSource(rows, columns));
    var aTexture = gpgpu.createMatrixTexture(rows, columns);
    var resultTexture = gpgpu.createMatrixTexture(1, 1);
    gpgpu.uploadMatrixToTexture(aTexture, rows, columns, a);
    logSumExp(gpgpu, program, aTexture, rows, columns, resultTexture);
    var result = gpgpu.downloadMatrixFromTexture(resultTexture, 1, 1);
    gpgpu.deleteMatrixTexture(aTexture);
    gpgpu.deleteMatrixTexture(resultTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return result[0];
}
exports.uploadLogSumExpDownload = uploadLogSumExpDownload;

},{"./gpgpu_context":35}],39:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../conv_util");
function getFragmentShaderMaxPoolBackprop(dyShapeRCD, fSize, origStride, origPad) {
    var origInputDepth = dyShapeRCD[2];
    var pad = fSize - 1 - origPad;
    var dyRows = dyShapeRCD[0], dyCols = dyShapeRCD[1], depth = dyShapeRCD[2];
    var dyTexShapeRC = conv_util.computeTexShapeFrom3D(dyShapeRCD);
    return "\n    precision highp float;\n    uniform sampler2D dy;\n    uniform sampler2D maxPos;\n\n    const vec2 halfCR = vec2(0.5, 0.5);\n    const vec2 dyShapeCR = vec2(" + dyTexShapeRC[1] + ", " + dyTexShapeRC[0] + ");\n\n    void main() {\n      vec2 dxTexCR = floor(gl_FragCoord.xy);\n\n      // Map from 2D (dxTexR, dxTexC) to 3D (dxR, dxC, d).\n      float dxR = dxTexCR.y;\n      float dxC = floor(dxTexCR.x / " + origInputDepth + ".0);\n      float d = mod(dxTexCR.x, " + origInputDepth + ".0);\n\n      vec2 dyRCCorner = vec2(dxR, dxC) - vec2(" + pad + ".0, " + pad + ".0);\n      float dyRCorner = dyRCCorner.x;\n      float dyCCorner = dyRCCorner.y;\n\n      // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(yR, dxC, d).\n      // ? = to be determined. : = across all values in that axis.\n      float dotProd = 0.0;\n      for (float wR = 0.0; wR < " + fSize + ".0; wR += 1.0) {\n\n        float dyR = (dyRCorner + wR) / " + origStride + ".0;\n        // TODO(nsthorat): Splice this with another version where you call\n        // getMatrixValueOrZeroPad(). Here and below.\n        if (dyR < 0.0 || dyR >= " + dyRows + ".0 || fract(dyR) > 0.0) {\n          continue;\n        }\n\n        float dyTexR = dyR;\n\n        for (float wC = 0.0; wC < " + fSize + ".0; wC += 1.0) {\n\n          float dyC = (dyCCorner + wC) / " + origStride + ".0;\n          if (dyC < 0.0 || dyC >= " + dyCols + ".0 || fract(dyC) > 0.0) {\n            continue;\n          }\n\n          float dyTexC = dyC * " + depth + ".0 + d;\n\n          // Read dy(dyR, dyC, d).\n          vec2 dyUV = (vec2(dyTexC, dyTexR) + halfCR) / dyShapeCR;\n          float dyValue = texture2D(dy, dyUV).r;\n\n          // Read maxPos(dyR, dyC, d).\n          float maxPosValue =\n              " + (fSize * fSize - 1) + ".0 - texture2D(maxPos, dyUV).r;\n\n          // Get the current value, check it against the value from the\n          // position matrix.\n          float curPosValue = wR * " + fSize + ".0 + wC;\n          float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);\n\n          dotProd += dyValue * mask;\n        }\n      }\n      gl_FragColor = vec4(dotProd, 0, 0, 0);\n    }";
}
exports.getFragmentShaderMaxPoolBackprop = getFragmentShaderMaxPoolBackprop;
function maxPoolBackprop(gpgpu, program, dyTex, maxPositionsTex, resultTex, resultTexShapeRC) {
    gpgpu.setOutputMatrixTexture(resultTex, resultTexShapeRC[0], resultTexShapeRC[1]);
    gpgpu.setProgram(program);
    gpgpu.setInputMatrixTexture(dyTex, 'dy', 0);
    gpgpu.setInputMatrixTexture(maxPositionsTex, 'maxPos', 1);
    gpgpu.executeProgram();
}
exports.maxPoolBackprop = maxPoolBackprop;

},{"../conv_util":16}],40:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var pool_gpu = require("./pool_gpu");
function getFragmentShaderMaxPoolPositionsSource(xShapeRCD, fSize, stride, pad) {
    return getFragmentShaderMaxPoolCommonSource(xShapeRCD, fSize, stride, pad, true);
}
exports.getFragmentShaderMaxPoolPositionsSource = getFragmentShaderMaxPoolPositionsSource;
function getFragmentShaderMaxPoolSource(xShapeRCD, fSize, stride, pad) {
    return getFragmentShaderMaxPoolCommonSource(xShapeRCD, fSize, stride, pad, false);
}
exports.getFragmentShaderMaxPoolSource = getFragmentShaderMaxPoolSource;
function getFragmentShaderMaxPoolCommonSource(xShapeRCD, fSize, stride, pad, computeMaxPositions) {
    return pool_gpu.getFragmentShaderPoolCommonSource(xShapeRCD, fSize, stride, pad, 'max', computeMaxPositions);
}
function maxPoolCommon(gpgpu, program, x, result, resultShapeRowCol) {
    pool_gpu.poolCommon(gpgpu, program, x, result, resultShapeRowCol);
}
exports.maxPoolCommon = maxPoolCommon;

},{"./pool_gpu":45}],41:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var pool_gpu = require("./pool_gpu");
function getFragmentShaderMinPoolSource(xShapeRCD, fSize, stride, pad) {
    return pool_gpu.getFragmentShaderPoolCommonSource(xShapeRCD, fSize, stride, pad, 'min', false);
}
exports.getFragmentShaderMinPoolSource = getFragmentShaderMinPoolSource;
function minPool(gpgpu, program, x, result, resultShapeRowCol) {
    pool_gpu.poolCommon(gpgpu, program, x, result, resultShapeRowCol);
}
exports.minPool = minPool;

},{"./pool_gpu":45}],42:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var webgl_util_1 = require("./webgl_util");
function getFragmentShaderSource(rows, columns, compOp) {
    return "\n    precision highp float;\n    uniform sampler2D matrixA;\n    varying vec2 outputColumnRow;\n\n    const vec2 aDimCR = vec2(" + columns + ".0, " + rows + ".0);\n    const vec2 halfCR = vec2(0.5, 0.5);\n\n    " + webgl_util_1.IS_NAN_SHADER_FUNC + "\n\n    void main() {\n      float value = texture2D(matrixA, halfCR / aDimCR).r;\n      for (float r = 0.0; r < aDimCR.y; r += 1.0) {\n        for (float c = 0.0; c < aDimCR.x; c += 1.0) {\n          vec2 cr = vec2(c, r);\n          vec2 uv = (cr + halfCR) / aDimCR;\n          float candidate = texture2D(matrixA, uv).r;\n          if (isNaN(candidate)) {\n            gl_FragColor = vec4(candidate, 0, 0, 0);\n            return;\n          }\n          value = " + compOp + "(value, candidate);\n        }\n      }\n      gl_FragColor = vec4(value, 0, 0, 0);\n    }";
}
function getMinFragmentShaderSource(rows, columns) {
    return getFragmentShaderSource(rows, columns, 'min');
}
exports.getMinFragmentShaderSource = getMinFragmentShaderSource;
function getMaxFragmentShaderSource(rows, columns) {
    return getFragmentShaderSource(rows, columns, 'max');
}
exports.getMaxFragmentShaderSource = getMaxFragmentShaderSource;
function minMax(gpgpu, minMaxProgram, a, rows, columns, result) {
    gpgpu.setOutputMatrixTexture(result, 1, 1);
    gpgpu.setProgram(minMaxProgram);
    gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
    gpgpu.executeProgram();
}
exports.minMax = minMax;

},{"./webgl_util":58}],43:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var math_1 = require("../math");
var shader_compiler = require("./shader_compiler");
function getFragmentShader(a, b, out, aOrientation, bOrientation) {
    var sharedDim = (aOrientation === math_1.MatrixOrientation.REGULAR ? a.shape[1] : a.shape[0]);
    var aSnippet = (aOrientation === math_1.MatrixOrientation.REGULAR) ? 'aRow, i' : 'i, aRow';
    var bSnippet = (bOrientation === math_1.MatrixOrientation.REGULAR) ? 'i, bCol' : 'bCol, i';
    var inputs = [{ name: 'matrixA', array: a }, { name: 'matrixB', array: b }];
    var userCode = "\n    const float sharedDim = " + sharedDim + ".0;\n\n    float dotARowBCol(float aRow, float bCol) {\n      float result = 0.0;\n      for (float i = 0.0; i < sharedDim; i += 1.0) {\n        float a = getMatrixA(" + aSnippet + ");\n        float b = getMatrixB(" + bSnippet + ");\n        result += (a * b);\n      }\n      return result;\n    }\n\n    void main() {\n      vec2 resRC = getOutputCoords();\n      setOutput(dotARowBCol(resRC.x, resRC.y));\n    }\n  ";
    return shader_compiler.makeShader(inputs, out, userCode);
}
exports.getFragmentShader = getFragmentShader;
function multiplyMatrix(gpgpu, multiplyProgram, a, b, result, outTexShape) {
    gpgpu.setOutputMatrixTexture(result, outTexShape[0], outTexShape[1]);
    gpgpu.setProgram(multiplyProgram);
    gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
    gpgpu.setInputMatrixTexture(b, 'matrixB', 1);
    gpgpu.executeProgram();
}
exports.multiplyMatrix = multiplyMatrix;

},{"../math":19,"./shader_compiler":51}],44:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var unaryop_gpu = require("./unaryop_gpu");
function getNegUnaryOp() {
    return 'gl_FragColor = vec4(-value, 0, 0, 0);';
}
function getFragmentShaderSource() {
    return unaryop_gpu.getFragmentShaderSource(getNegUnaryOp());
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function neg(gpgpu, program, a, rows, columns, result) {
    unaryop_gpu.unaryOp(gpgpu, program, a, rows, columns, result);
}
exports.neg = neg;
function uploadNegDownload(a, rows, columns) {
    return unaryop_gpu.uploadUnaryOpDownload(a, rows, columns, getNegUnaryOp());
}
exports.uploadNegDownload = uploadNegDownload;

},{"./unaryop_gpu":57}],45:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../conv_util");
var webgl_util_1 = require("./webgl_util");
function getFragmentShaderPoolCommonSource(xShapeRCD, fSize, stride, pad, poolType, computePositions) {
    if (poolType === 'avg' && computePositions) {
        throw new Error('Cannot compute positions for average pool.');
    }
    var depth = xShapeRCD[2];
    var xTexShapeRC = conv_util.computeTexShapeFrom3D(xShapeRCD);
    var returnValue = 'minMaxValue';
    if (computePositions) {
        returnValue = 'minMaxPosition';
    }
    else if (poolType === 'avg') {
        returnValue = 'avgValue';
    }
    return "\n    precision highp float;\n    uniform sampler2D x;\n    varying vec2 resultUV;\n\n    const vec2 halfCR = vec2(0.5, 0.5);\n    const vec2 xShapeCR = vec2(" + xTexShapeRC[1] + ", " + xTexShapeRC[0] + ");\n\n    " + webgl_util_1.IS_NAN_SHADER_FUNC + "\n\n    void main() {\n      vec2 yTexCR = floor(gl_FragCoord.xy);\n\n      // Map from 2D (yTexR, yTexC) to 3D (yR, yC, d2).\n      float yR = yTexCR.y;\n      float yC = floor(yTexCR.x / " + depth + ".0);\n      float d = mod(yTexCR.x, " + depth + ".0);\n\n      vec2 xRCCorner = vec2(yR, yC) * vec2(" + stride + ", " + stride + ") -\n          vec2(" + pad + ".0, " + pad + ".0);\n      float xRCorner = xRCCorner.x;\n      float xCCorner = xRCCorner.y;\n\n      // max/min x(?, ?, d) to get y(yR, yC, d).\n      // ? = to be determined\n      float minMaxValue = 0.0;\n      float minMaxValueFound = 0.0;\n      float minMaxPosition = 0.0;\n      float avgValue = 0.0;\n\n      for (float wR = 0.0; wR < " + fSize + ".0; wR += 1.0) {\n        float xR = xRCorner + wR;\n        float xTexR = xR;\n\n        for (float wC = 0.0; wC < " + fSize + ".0; wC += 1.0) {\n          float xC = xCCorner + wC;\n          float xTexC = xC * " + depth + ".0 + d;\n\n          vec2 texCR = vec2(xTexC, xTexR);\n\n          // Check if the requested UV is invalid.\n          vec2 uv = (texCR + halfCR) / xShapeCR;\n          bool lessThanZero = any(lessThan(uv, vec2(0, 0)));\n          bool greaterThanOne = any(greaterThan(uv, vec2(1, 1)));\n          bool outside = lessThanZero || greaterThanOne;\n          if (outside) {\n            continue;\n          }\n\n          float value = texture2D(x, uv).r;\n          if (isNaN(value)) {\n            gl_FragColor = vec4(value, 0, 0, 0);\n            return;\n          }\n          if (" + (poolType === 'avg') + ") {\n            avgValue += value / " + fSize * fSize + ".0;\n          } else {\n            // If a min / max value has already been found, use it. If not, use\n            // the current value.\n            float currentMinMaxValue = mix(\n                value, minMaxValue, minMaxValueFound);\n            if (value " + (poolType === 'min' ? '<=' : '>=') + " currentMinMaxValue) {\n              minMaxValue = value;\n              minMaxValueFound = 1.0;\n              if (" + computePositions + ") {\n                minMaxPosition = wR * " + fSize + ".0 + wC;\n              }\n            }\n          }\n        }\n      }\n      gl_FragColor = vec4(" + returnValue + ", 0, 0, 0);\n    }";
}
exports.getFragmentShaderPoolCommonSource = getFragmentShaderPoolCommonSource;
function poolCommon(gpgpu, program, x, result, resultShapeRowCol) {
    gpgpu.setOutputMatrixTexture(result, resultShapeRowCol[0], resultShapeRowCol[1]);
    gpgpu.setProgram(program);
    gpgpu.setInputMatrixTexture(x, 'x', 0);
    gpgpu.executeProgram();
}
exports.poolCommon = poolCommon;

},{"../conv_util":16,"./webgl_util":58}],46:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var gpgpu_context_1 = require("./gpgpu_context");
function getFragmentShaderSource(rows, columns) {
    return "\n    precision highp float;\n    uniform sampler2D matrixA;\n    varying vec2 resultUV;\n\n    const vec2 aDimCR = vec2(" + columns + ".0, " + rows + ".0);\n    const vec2 halfCR = vec2(0.5, 0.5);\n\n    void main() {\n      float sum = 0.0;\n      for (float r = 0.0; r < aDimCR.y; r += 1.0) {\n        for (float c = 0.0; c < aDimCR.x; c += 1.0) {\n          vec2 uv = (vec2(c, r) + halfCR) / aDimCR;\n          sum += texture2D(matrixA, uv).r;\n        }\n      }\n      gl_FragColor = vec4(sum, 0, 0, 0);\n    }";
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function reduceSum(gpgpu, reduceSumProgram, a, aNumRows, aNumCols, result) {
    gpgpu.setOutputMatrixTexture(result, 1, 1);
    gpgpu.setProgram(reduceSumProgram);
    gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
    gpgpu.executeProgram();
}
exports.reduceSum = reduceSum;
function uploadReduceSumDownload(a, rows, columns) {
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var program = gpgpu.createProgram(getFragmentShaderSource(rows, columns));
    var aTexture = gpgpu.createMatrixTexture(rows, columns);
    var resultTexture = gpgpu.createMatrixTexture(1, 1);
    gpgpu.uploadMatrixToTexture(aTexture, rows, columns, a);
    reduceSum(gpgpu, program, aTexture, rows, columns, resultTexture);
    var result = gpgpu.downloadMatrixFromTexture(resultTexture, 1, 1)[0];
    gpgpu.deleteMatrixTexture(aTexture);
    gpgpu.deleteMatrixTexture(resultTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return result;
}
exports.uploadReduceSumDownload = uploadReduceSumDownload;

},{"./gpgpu_context":35}],47:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var unaryop_gpu = require("./unaryop_gpu");
function getReluUnaryOp() {
    return "\n    float result = (value < 0.0 ? 0.0 : value);\n    gl_FragColor = vec4(result, 0, 0, 0);\n  ";
}
function getFragmentShaderSource() {
    return unaryop_gpu.getFragmentShaderSource(getReluUnaryOp());
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function relu(gpgpu, reluProgram, a, rows, columns, result) {
    unaryop_gpu.unaryOp(gpgpu, reluProgram, a, rows, columns, result);
}
exports.relu = relu;
function uploadReluDownload(a, rows, columns) {
    return unaryop_gpu.uploadUnaryOpDownload(a, rows, columns, getReluUnaryOp());
}
exports.uploadReluDownload = uploadReluDownload;

},{"./unaryop_gpu":57}],48:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var webgl_util = require("./webgl_util");
function getRenderRGBShader(gpgpu, destinationWidth) {
    var fragmentShaderSource = "\n    precision highp float;\n    uniform sampler2D source;\n    varying vec2 resultUV;\n\n    const float destinationWidth = " + destinationWidth + ".0;\n    const float a = 1.0;\n\n    void main() {\n      float xr = floor(resultUV.s * destinationWidth) * 3.0;\n      vec3 x = xr + vec3(0, 1, 2);\n\n      float sourceWidth = destinationWidth * 3.0;\n      vec3 u = (x + 0.5) / sourceWidth;\n      float v = 1.0 - resultUV.t;\n\n      float r = texture2D(source, vec2(u[0], v)).r;\n      float g = texture2D(source, vec2(u[1], v)).r;\n      float b = texture2D(source, vec2(u[2], v)).r;\n\n      gl_FragColor = vec4(r, g, b, a);\n    }";
    return gpgpu.createProgram(fragmentShaderSource);
}
exports.getRenderRGBShader = getRenderRGBShader;
function renderToCanvas(gpgpu, renderShader, sourceTex) {
    webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
    renderToFramebuffer(gpgpu, renderShader, sourceTex);
}
exports.renderToCanvas = renderToCanvas;
function renderToFramebuffer(gpgpu, renderShader, sourceTex) {
    gpgpu.setProgram(renderShader);
    gpgpu.setInputMatrixTexture(sourceTex, 'source', 0);
    gpgpu.executeProgram();
}
exports.renderToFramebuffer = renderToFramebuffer;

},{"./webgl_util":58}],49:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../../util");
function getFragmentShaderSource() {
    return "\n    precision highp float;\n    uniform sampler2D matrixA;\n    uniform vec2 inputDimCR;\n    uniform vec2 resultDimCR;\n    varying vec2 resultUV;\n    const vec2 halfCR = vec2(0.5, 0.5);\n\n    void main() {\n      vec2 resultCR = floor(resultUV * resultDimCR);\n      // indexInFlat = row * stride + column, where stride == numOutputColumns\n      float indexInFlat = resultCR.y * resultDimCR.x + resultCR.x;\n\n      vec2 inputCR = vec2(\n        mod(indexInFlat, inputDimCR.x), // col = indexInFlat % numInputColumns\n        floor(indexInFlat / inputDimCR.x) // row = indexInFlat / numInputColumns\n      ) + halfCR;\n\n      vec2 inputUV = inputCR / inputDimCR;\n      gl_FragColor = texture2D(matrixA, inputUV);\n    }";
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function reshape(gpgpu, reshapeProgram, a, aNumRows, aNumCols, result, resultNumRows, resultNumCols) {
    var inputSize = aNumRows * aNumCols;
    var outputSize = resultNumCols * resultNumRows;
    util.assert(inputSize === outputSize, "The input size (" + inputSize + ") and output size (" + outputSize + ") " +
        "must match");
    gpgpu.setOutputMatrixTexture(result, resultNumRows, resultNumCols);
    gpgpu.setProgram(reshapeProgram);
    gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
    var inputDimCRLocation = gpgpu.getUniformLocation('inputDimCR');
    gpgpu.gl.uniform2f(inputDimCRLocation, aNumCols, aNumRows);
    var resultDimCRLocation = gpgpu.getUniformLocation('resultDimCR');
    gpgpu.gl.uniform2f(resultDimCRLocation, resultNumCols, resultNumRows);
    gpgpu.executeProgram();
}
exports.reshape = reshape;

},{"../../util":86}],50:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../conv_util");
function getFragmentShaderSource(inputShapeRCD, outputDimensionsRowCol, alignCorners) {
    var depth = inputShapeRCD[2];
    var inputTexShapeRC = conv_util.computeTexShapeFrom3D(inputShapeRCD);
    var effectiveInputShapeRCD = alignCorners ?
        [inputShapeRCD[0] - 1, inputShapeRCD[1] - 1, depth] :
        inputShapeRCD;
    var effectiveOutputShapeRCD = alignCorners ?
        [outputDimensionsRowCol[0] - 1, outputDimensionsRowCol[1] - 1, depth] :
        [outputDimensionsRowCol[0], outputDimensionsRowCol[1], depth];
    return "\n    precision highp float;\n    uniform sampler2D matrixA;\n    varying vec2 resultUV;\n    const vec2 halfCR = vec2(0.5, 0.5);\n\n    const vec2 inputShapeCR = vec2(" + inputShapeRCD[1] + ", " + inputShapeRCD[0] + ");\n    const vec2 inputShapeTexCR = vec2(\n        " + inputTexShapeRC[1] + ", " + inputTexShapeRC[0] + ");\n\n    const vec2 effectiveInputOverOutputRatioCR = vec2(\n        " + effectiveInputShapeRCD[1] / effectiveOutputShapeRCD[1] + ",\n        " + effectiveInputShapeRCD[0] / effectiveOutputShapeRCD[0] + ");\n\n    float sampleInput(float col, float row, float d) {\n      vec2 uv = (vec2(col * " + depth + ".0 + d, row) + halfCR) / inputShapeTexCR;\n      return texture2D(matrixA, uv).r;\n    }\n\n    void main() {\n      vec2 yTexCR = floor(gl_FragCoord.xy);\n\n      // Map from 2D (yTexR, yTexC) to 3D (yR, yC, d).\n      vec2 yCR = vec2(floor(yTexCR.x / " + depth + ".0), yTexCR.y);\n      float d = mod(yTexCR.x, " + depth + ".0);\n\n      // Fractional source index.\n      vec2 sourceFracIndexCR = yCR * effectiveInputOverOutputRatioCR;\n\n      // Compute the four integer indices.\n      vec2 sourceFloorCR = floor(sourceFracIndexCR);\n      vec2 sourceCeilCR = min(inputShapeCR - 1.0, ceil(sourceFracIndexCR));\n\n      float topLeft = sampleInput(sourceFloorCR[0], sourceFloorCR[1], d);\n      float bottomLeft = sampleInput(sourceFloorCR[0], sourceCeilCR[1], d);\n      float topRight = sampleInput(sourceCeilCR[0], sourceFloorCR[1], d);\n      float bottomRight = sampleInput(sourceCeilCR[0], sourceCeilCR[1], d);\n\n      vec2 fracCR = sourceFracIndexCR - sourceFloorCR;\n\n      float top = topLeft + (topRight - topLeft) * fracCR[0];\n      float bottom = bottomLeft + (bottomRight - bottomLeft) * fracCR[0];\n      float newValue = top + (bottom - top) * fracCR[1];\n\n      gl_FragColor = vec4(newValue, 0.0, 0.0, 0.0);\n    }";
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function resizeBilinear(gpgpu, resizeBilinearProgram, a, result, resultShapeRowCol) {
    gpgpu.setOutputMatrixTexture(result, resultShapeRowCol[0], resultShapeRowCol[1]);
    gpgpu.setProgram(resizeBilinearProgram);
    gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
    gpgpu.executeProgram();
}
exports.resizeBilinear = resizeBilinear;

},{"../conv_util":16}],51:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../../util");
function makeShaderKey(inputs, output) {
    var ins = inputs.map(function (x) { return x.shape + '_' + x.getTextureShapeRC(); });
    return ins.join('_') + '_' + output.shape + '_' + output.getTextureShapeRC();
}
exports.makeShaderKey = makeShaderKey;
function makeShader(inputs, output, userCode) {
    var inputPrefixSnippet = inputs.map(function (x) { return "uniform sampler2D " + x.name + ";"; }).join('\n');
    var inputSamplingSnippet = inputs.map(function (x) { return getInputSamplingSnippet(x); }).join('\n');
    var outTexShape = output.getTextureShapeRC();
    var outputSamplingSnippet = getOutputSamplingSnippet(output.shape, outTexShape);
    var source = [
        SHADER_PREFIX, inputPrefixSnippet, SAMPLE_2D_SNIPPET, inputSamplingSnippet,
        outputSamplingSnippet, userCode
    ].join('\n');
    return source;
}
exports.makeShader = makeShader;
function getInputSamplingSnippet(input) {
    var arr = input.array;
    var shape = arr.shape;
    var texShape = arr.getTextureShapeRC(shape);
    switch (shape.length) {
        case 2:
            return getSampler2D(input.name, shape, texShape);
        default:
            throw new Error(arr.rank + "-D input sampling is not yet supported");
    }
}
function getOutputSamplingSnippet(outShape, outTexShape) {
    switch (outShape.length) {
        case 2:
            return getOutput2DCoords(outShape, outTexShape);
        default:
            throw new Error(outShape.length + "-D output sampling is not yet supported");
    }
}
var SHADER_PREFIX = "\n  precision highp float;\n  varying vec2 resultUV;\n  const vec2 halfCR = vec2(0.5, 0.5);\n\n  void setOutput(float val) {\n    gl_FragColor = vec4(val, 0, 0, 0);\n  }\n";
var SAMPLE_2D_SNIPPET = "\n  float sample2D(sampler2D texture, float texNumR, float texNumC, float numC,\n      float row, float col) {\n    float index = dot(vec2(row, col), vec2(numC, 1.0));\n    float texR = floor(index / texNumC);\n    float texC = mod(index, texNumC);\n    vec2 uv = (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n    return texture2D(texture, uv).r;\n  }\n";
function getOutput2DCoords(shape, texShape) {
    if (util.arraysEqual(shape, texShape)) {
        return "\n      vec2 getOutputCoords() {\n        return floor(gl_FragCoord.yx);\n      }\n    ";
    }
    return "\n    vec2 getOutputCoords() {\n      vec2 resTexRC = floor(gl_FragCoord.yx);\n      float index = dot(resTexRC, vec2(" + texShape[1] + ".0, 1.0));\n      float r = floor(index / " + shape[1] + ".0);\n      float c = mod(index, " + shape[1] + ".0);\n      return vec2(r, c);\n    }\n  ";
}
function getSampler2D(texName, shape, texShape) {
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    var tR = texShape[0];
    var tC = texShape[1];
    if (util.arraysEqual(shape, texShape)) {
        return "\n      float " + funcName + "(float row, float col) {\n        vec2 uv = (vec2(col, row) + halfCR) / vec2(" + tC + ".0, " + tR + ".0);\n        return texture2D(" + texName + ", uv).r;\n      }\n    ";
    }
    return "\n    float " + funcName + "(float row, float col) {\n      return sample2D(" + texName + ", " + tR + ".0, " + tC + ".0, " + shape[1] + ".0, row, col);\n    }\n  ";
}

},{"../../util":86}],52:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var unaryop_gpu = require("./unaryop_gpu");
function getSigmoidUnaryOp() {
    return 'gl_FragColor = vec4(1.0 / (1.0 + exp(-1.0 * value)), 0, 0, 0);';
}
function getSigmoidFragmentShaderSource() {
    return unaryop_gpu.getFragmentShaderSource(getSigmoidUnaryOp());
}
exports.getSigmoidFragmentShaderSource = getSigmoidFragmentShaderSource;
function sigmoid(gpgpu, sigmoidProgram, a, rows, columns, result) {
    unaryop_gpu.unaryOp(gpgpu, sigmoidProgram, a, rows, columns, result);
}
exports.sigmoid = sigmoid;
function uploadSigmoidDownload(a, rows, columns) {
    return unaryop_gpu.uploadUnaryOpDownload(a, rows, columns, getSigmoidUnaryOp());
}
exports.uploadSigmoidDownload = uploadSigmoidDownload;

},{"./unaryop_gpu":57}],53:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var unaryop_gpu = require("./unaryop_gpu");
function getStepUnaryOp() {
    return "\n    float res = value == value ? (value > 0.0 ? 1.0 : 0.0) : value;\n    gl_FragColor = vec4(res, 0, 0, 0);\n  ";
}
function getFragmentShaderSource() {
    return unaryop_gpu.getFragmentShaderSource(getStepUnaryOp());
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function step(gpgpu, stepProgram, a, rows, columns, result) {
    unaryop_gpu.unaryOp(gpgpu, stepProgram, a, rows, columns, result);
}
exports.step = step;
function uploadStepDownload(a, rows, columns) {
    return unaryop_gpu.uploadUnaryOpDownload(a, rows, columns, getStepUnaryOp());
}
exports.uploadStepDownload = uploadStepDownload;

},{"./unaryop_gpu":57}],54:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function getUnpackedMatrixTextureShapeWidthHeight(rows, columns) {
    return [columns, rows];
}
exports.getUnpackedMatrixTextureShapeWidthHeight = getUnpackedMatrixTextureShapeWidthHeight;
function getUnpackedArraySizeFromMatrixSize(matrixSize, channelsPerTexture) {
    return matrixSize * channelsPerTexture;
}
exports.getUnpackedArraySizeFromMatrixSize = getUnpackedArraySizeFromMatrixSize;
function getColorMatrixTextureShapeWidthHeight(rows, columns) {
    return [columns * 4, rows];
}
exports.getColorMatrixTextureShapeWidthHeight = getColorMatrixTextureShapeWidthHeight;
function getMatrixSizeFromUnpackedArraySize(unpackedSize, channelsPerTexture) {
    if (unpackedSize % channelsPerTexture !== 0) {
        throw new Error('unpackedSize (' + unpackedSize + ') must be a multiple of ' +
            channelsPerTexture);
    }
    return unpackedSize / channelsPerTexture;
}
exports.getMatrixSizeFromUnpackedArraySize = getMatrixSizeFromUnpackedArraySize;
function encodeMatrixToUnpackedArray(matrix, unpackedArray, channelsPerTexture) {
    var requiredSize = getUnpackedArraySizeFromMatrixSize(matrix.length, channelsPerTexture);
    if (unpackedArray.length < requiredSize) {
        throw new Error('unpackedArray length (' + unpackedArray.length +
            ') must be >= ' + requiredSize);
    }
    var dst = 0;
    for (var src = 0; src < matrix.length; ++src) {
        unpackedArray[dst] = matrix[src];
        dst += channelsPerTexture;
    }
}
exports.encodeMatrixToUnpackedArray = encodeMatrixToUnpackedArray;
function decodeMatrixFromUnpackedArray(unpackedArray, matrix, channelsPerTexture) {
    var requiredSize = getMatrixSizeFromUnpackedArraySize(unpackedArray.length, channelsPerTexture);
    if (matrix.length < requiredSize) {
        throw new Error('matrix length (' + matrix.length + ') must be >= ' + requiredSize);
    }
    var dst = 0;
    for (var src = 0; src < unpackedArray.length; src += channelsPerTexture) {
        matrix[dst++] = unpackedArray[src];
    }
}
exports.decodeMatrixFromUnpackedArray = decodeMatrixFromUnpackedArray;
function getPackedMatrixTextureShapeWidthHeight(rows, columns) {
    return [Math.ceil(columns / 2), Math.ceil(rows / 2)];
}
exports.getPackedMatrixTextureShapeWidthHeight = getPackedMatrixTextureShapeWidthHeight;
function getPackedRGBAArraySizeFromMatrixShape(rows, columns) {
    var _a = getPackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
    return w * h * 4;
}
exports.getPackedRGBAArraySizeFromMatrixShape = getPackedRGBAArraySizeFromMatrixShape;
function encodeMatrixToPackedRGBA(matrix, rows, columns, packedRGBA) {
    var requiredSize = getPackedRGBAArraySizeFromMatrixShape(rows, columns);
    if (packedRGBA.length < requiredSize) {
        throw new Error('packedRGBA length (' + packedRGBA.length +
            ') must be >= ' + requiredSize);
    }
    var _a = getPackedMatrixTextureShapeWidthHeight(rows, columns), textureWidth = _a[0], textureHeight = _a[1];
    var oddWidth = (columns % 2) === 1;
    var oddHeight = (rows % 2) === 1;
    var widthInFullBlocks = Math.floor(columns / 2);
    var heightInFullBlocks = Math.floor(rows / 2);
    {
        var dstStride = (oddWidth ? 4 : 0);
        var oneRow = columns;
        var dst = 0;
        for (var blockY = 0; blockY < heightInFullBlocks; ++blockY) {
            var matrixSrcRow = (blockY * 2 * columns);
            for (var blockX = 0; blockX < widthInFullBlocks; ++blockX) {
                var matrixSrcCol = blockX * 2;
                var src = matrixSrcRow + matrixSrcCol;
                packedRGBA[dst] = matrix[src];
                packedRGBA[dst + 1] = matrix[src + 1];
                packedRGBA[dst + 2] = matrix[src + oneRow];
                packedRGBA[dst + 3] = matrix[src + oneRow + 1];
                dst += 4;
            }
            dst += dstStride;
        }
    }
    if (oddWidth) {
        var src = columns - 1;
        var dst = (textureWidth - 1) * 4;
        var srcStride = 2 * columns;
        var dstStride = textureWidth * 4;
        for (var blockY = 0; blockY < heightInFullBlocks; ++blockY) {
            packedRGBA[dst] = matrix[src];
            packedRGBA[dst + 2] = matrix[src + columns];
            src += srcStride;
            dst += dstStride;
        }
    }
    if (oddHeight) {
        var src = (rows - 1) * columns;
        var dst = (textureHeight - 1) * textureWidth * 4;
        for (var blockX = 0; blockX < widthInFullBlocks; ++blockX) {
            packedRGBA[dst++] = matrix[src++];
            packedRGBA[dst++] = matrix[src++];
            dst += 2;
        }
    }
    if (oddWidth && oddHeight) {
        packedRGBA[packedRGBA.length - 4] = matrix[matrix.length - 1];
    }
    return packedRGBA;
}
exports.encodeMatrixToPackedRGBA = encodeMatrixToPackedRGBA;
function decodeMatrixFromPackedRGBA(packedRGBA, rows, columns, matrix) {
    var requiredSize = rows * columns;
    if (requiredSize < matrix.length) {
        throw new Error('matrix length (' + matrix.length + ') must be >= ' + requiredSize);
    }
    var oddWidth = (columns % 2) === 1;
    var oddHeight = (rows % 2) === 1;
    var widthInFullBlocks = Math.floor(columns / 2);
    var heightInFullBlocks = Math.floor(rows / 2);
    var _a = getPackedMatrixTextureShapeWidthHeight(rows, columns), textureWidth = _a[0], textureHeight = _a[1];
    {
        var srcStride = oddWidth ? 4 : 0;
        var dstStride = columns + (oddWidth ? 1 : 0);
        var src = 0;
        var dstRow1 = 0;
        var dstRow2 = columns;
        for (var blockY = 0; blockY < heightInFullBlocks; ++blockY) {
            for (var blockX = 0; blockX < widthInFullBlocks; ++blockX) {
                matrix[dstRow1++] = packedRGBA[src++];
                matrix[dstRow1++] = packedRGBA[src++];
                matrix[dstRow2++] = packedRGBA[src++];
                matrix[dstRow2++] = packedRGBA[src++];
            }
            src += srcStride;
            dstRow1 += dstStride;
            dstRow2 += dstStride;
        }
    }
    if (oddWidth) {
        var src = (textureWidth - 1) * 4;
        var dst = columns - 1;
        var srcStride = textureWidth * 4;
        var dstStride = 2 * columns;
        for (var blockY = 0; blockY < heightInFullBlocks; ++blockY) {
            matrix[dst] = packedRGBA[src];
            matrix[dst + columns] = packedRGBA[src + 2];
            src += srcStride;
            dst += dstStride;
        }
    }
    if (oddHeight) {
        var src = (textureHeight - 1) * textureWidth * 4;
        var dst = (rows - 1) * columns;
        for (var blockX = 0; blockX < widthInFullBlocks; ++blockX) {
            matrix[dst++] = packedRGBA[src++];
            matrix[dst++] = packedRGBA[src++];
            src += 2;
        }
    }
    if (oddWidth && oddHeight) {
        matrix[matrix.length - 1] = packedRGBA[packedRGBA.length - 4];
    }
    return matrix;
}
exports.decodeMatrixFromPackedRGBA = decodeMatrixFromPackedRGBA;

},{}],55:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var TextureManager = (function () {
    function TextureManager(gpgpu) {
        this.gpgpu = gpgpu;
        this.numUsedTextures = 0;
        this.numFreeTextures = 0;
        this.freeTextures = {};
        this.logEnabled = false;
        this.usedTextureCount = {};
    }
    TextureManager.prototype.acquireTexture = function (shapeRC) {
        var shapeKey = getKeyFromTextureShape(shapeRC);
        if (!(shapeKey in this.freeTextures)) {
            this.freeTextures[shapeKey] = [];
        }
        if (!(shapeKey in this.usedTextureCount)) {
            this.usedTextureCount[shapeKey] = 0;
        }
        this.usedTextureCount[shapeKey]++;
        if (this.freeTextures[shapeKey].length > 0) {
            this.numFreeTextures--;
            this.numUsedTextures++;
            this.log();
            return this.freeTextures[shapeKey].shift();
        }
        this.numUsedTextures++;
        this.log();
        return this.gpgpu.createMatrixTexture(shapeRC[0], shapeRC[1]);
    };
    TextureManager.prototype.releaseTexture = function (texture, shape) {
        var shapeKey = getKeyFromTextureShape(shape);
        if (!(shapeKey in this.freeTextures)) {
            this.freeTextures[shapeKey] = [];
        }
        this.freeTextures[shapeKey].push(texture);
        this.numFreeTextures++;
        this.numUsedTextures--;
        this.usedTextureCount[shapeKey]--;
        this.log();
    };
    TextureManager.prototype.log = function () {
        if (!this.logEnabled) {
            return;
        }
        var total = this.numFreeTextures + this.numUsedTextures;
        console.log('Free/Used', this.numFreeTextures + ' / ' + this.numUsedTextures, "(" + total + ")");
    };
    TextureManager.prototype.getNumUsedTextures = function () {
        return this.numUsedTextures;
    };
    TextureManager.prototype.getNumFreeTextures = function () {
        return this.numFreeTextures;
    };
    TextureManager.prototype.dispose = function () {
        for (var shape in this.freeTextures) {
            if (this.freeTextures.hasOwnProperty(shape)) {
                for (var i = 0; i < this.freeTextures[shape].length; i++) {
                    this.gpgpu.deleteMatrixTexture(this.freeTextures[shape][i]);
                }
            }
        }
    };
    return TextureManager;
}());
exports.TextureManager = TextureManager;
function getKeyFromTextureShape(shapeRowsCol) {
    return shapeRowsCol[0] + '_' + shapeRowsCol[1];
}

},{}],56:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var unaryop_gpu = require("./unaryop_gpu");
function getSinUnaryOp() {
    return "\n    gl_FragColor = vec4(sin(value), 0, 0, 0);\n  ";
}
function getSinFragmentShaderSource() {
    return unaryop_gpu.getFragmentShaderSource(getSinUnaryOp());
}
exports.getSinFragmentShaderSource = getSinFragmentShaderSource;
function sin(gpgpu, sinProgram, a, rows, columns, result) {
    unaryop_gpu.unaryOp(gpgpu, sinProgram, a, rows, columns, result);
}
exports.sin = sin;
function uploadSinDownload(a, rows, columns) {
    return unaryop_gpu.uploadUnaryOpDownload(a, rows, columns, getSinUnaryOp());
}
exports.uploadSinDownload = uploadSinDownload;
function getTanhUnaryOp() {
    return "\n    float e2x = exp(-2.0 * value);\n    gl_FragColor = vec4((1.0 - e2x) / (1.0 + e2x), 0, 0, 0);\n  ";
}
function getTanhFragmentShaderSource() {
    return unaryop_gpu.getFragmentShaderSource(getTanhUnaryOp());
}
exports.getTanhFragmentShaderSource = getTanhFragmentShaderSource;
function tanh(gpgpu, tanhProgram, a, rows, columns, result) {
    unaryop_gpu.unaryOp(gpgpu, tanhProgram, a, rows, columns, result);
}
exports.tanh = tanh;
function uploadTanhDownload(a, rows, columns) {
    return unaryop_gpu.uploadUnaryOpDownload(a, rows, columns, getTanhUnaryOp());
}
exports.uploadTanhDownload = uploadTanhDownload;

},{"./unaryop_gpu":57}],57:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var gpgpu_context_1 = require("./gpgpu_context");
function getFragmentShaderSource(resultOp) {
    return "\n    precision highp float;\n    uniform sampler2D matrixA;\n    varying vec2 resultUV;\n\n    void main() {\n      float value = texture2D(matrixA, resultUV).r;\n      " + resultOp + "\n    }";
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function unaryOp(gpgpu, unaryOpProgram, a, rows, columns, result) {
    gpgpu.setOutputMatrixTexture(result, rows, columns);
    gpgpu.setProgram(unaryOpProgram);
    gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
    gpgpu.executeProgram();
}
exports.unaryOp = unaryOp;
function uploadUnaryOpDownload(a, rows, columns, resultOp) {
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var fragmentShaderSrc = getFragmentShaderSource(resultOp);
    var program = gpgpu.createProgram(fragmentShaderSrc);
    var aTexture = gpgpu.createMatrixTexture(rows, columns);
    var resultTexture = gpgpu.createMatrixTexture(rows, columns);
    gpgpu.uploadMatrixToTexture(aTexture, rows, columns, a);
    unaryOp(gpgpu, program, aTexture, rows, columns, resultTexture);
    var result = gpgpu.downloadMatrixFromTexture(resultTexture, rows, columns);
    gpgpu.deleteMatrixTexture(aTexture);
    gpgpu.deleteMatrixTexture(resultTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return result;
}
exports.uploadUnaryOpDownload = uploadUnaryOpDownload;

},{"./gpgpu_context":35}],58:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var USE_WEBGL2_WHEN_AVAILABLE = false;
var WEBGL2_ENABLED = null;
var MAX_TEXTURE_SIZE = null;
var util = require("../../util");
exports.IS_NAN_SHADER_FUNC = "\nbool isNaN(float val) {\n  return val == val ? false : true;\n}\n";
function createWebGLRenderingContext(attributes) {
    var canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    return createWebGLRenderingContextFromCanvas(canvas, attributes);
}
exports.createWebGLRenderingContext = createWebGLRenderingContext;
function preferWebGL1() {
    USE_WEBGL2_WHEN_AVAILABLE = false;
    WEBGL2_ENABLED = undefined;
}
exports.preferWebGL1 = preferWebGL1;
function preferWebGL2() {
    USE_WEBGL2_WHEN_AVAILABLE = true;
    WEBGL2_ENABLED = undefined;
}
exports.preferWebGL2 = preferWebGL2;
function isWebGL2Enabled() {
    if (!USE_WEBGL2_WHEN_AVAILABLE) {
        return false;
    }
    if (WEBGL2_ENABLED === undefined) {
        var tempCanvas = document.createElement('canvas');
        var gl = tempCanvas.getContext('webgl2');
        if (gl != null) {
            WEBGL2_ENABLED = true;
            var loseContextExtension = getExtensionOrThrow(gl, 'WEBGL_lose_context');
            loseContextExtension.loseContext();
        }
        else {
            WEBGL2_ENABLED = false;
        }
    }
    return WEBGL2_ENABLED;
}
exports.isWebGL2Enabled = isWebGL2Enabled;
function createWebGLRenderingContextFromCanvas(canvas, attributes) {
    var gl;
    if (isWebGL2Enabled()) {
        gl = canvas.getContext('webgl2', attributes);
    }
    else {
        gl = (canvas.getContext('webgl', attributes) ||
            canvas.getContext('experimental-webgl', attributes));
    }
    if (gl == null) {
        throw new Error('This browser does not support WebGL.');
    }
    return gl;
}
exports.createWebGLRenderingContextFromCanvas = createWebGLRenderingContextFromCanvas;
function callAndCheck(gl, func) {
    var returnValue = func();
    checkWebGLError(gl);
    return returnValue;
}
exports.callAndCheck = callAndCheck;
var webGLDebugErrorCheckingEnabled = false;
function enableDebugWebGLErrorChecking(enabled) {
    webGLDebugErrorCheckingEnabled = enabled;
}
exports.enableDebugWebGLErrorChecking = enableDebugWebGLErrorChecking;
function checkWebGLError(gl) {
    if (webGLDebugErrorCheckingEnabled) {
        var error = gl.getError();
        if (error !== gl.NO_ERROR) {
            throw new Error('WebGL Error: ' + getWebGLErrorMessage(gl, error));
        }
    }
}
exports.checkWebGLError = checkWebGLError;
function getWebGLErrorMessage(gl, status) {
    switch (status) {
        case gl.NO_ERROR:
            return 'NO_ERROR';
        case gl.INVALID_ENUM:
            return 'INVALID_ENUM';
        case gl.INVALID_VALUE:
            return 'INVALID_VALUE';
        case gl.INVALID_OPERATION:
            return 'INVALID_OPERATION';
        case gl.INVALID_FRAMEBUFFER_OPERATION:
            return 'INVALID_FRAMEBUFFER_OPERATION';
        case gl.OUT_OF_MEMORY:
            return 'OUT_OF_MEMORY';
        case gl.CONTEXT_LOST_WEBGL:
            return 'CONTEXT_LOST_WEBGL';
        default:
            return 'Unknown error code ' + status;
    }
}
exports.getWebGLErrorMessage = getWebGLErrorMessage;
function getExtensionOrThrow(gl, extensionName) {
    return throwIfNull(gl, function () { return gl.getExtension(extensionName); }, 'Extension "' + extensionName + '" not supported on this browser.');
}
exports.getExtensionOrThrow = getExtensionOrThrow;
function createVertexShader(gl, vertexShaderSource) {
    var vertexShader = throwIfNull(gl, function () { return gl.createShader(gl.VERTEX_SHADER); }, 'Unable to create vertex WebGLShader.');
    callAndCheck(gl, function () { return gl.shaderSource(vertexShader, vertexShaderSource); });
    callAndCheck(gl, function () { return gl.compileShader(vertexShader); });
    if (gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS) === false) {
        console.log(gl.getShaderInfoLog(vertexShader));
        throw new Error('Failed to compile vertex shader.');
    }
    return vertexShader;
}
exports.createVertexShader = createVertexShader;
function createFragmentShader(gl, fragmentShaderSource) {
    var fragmentShader = throwIfNull(gl, function () { return gl.createShader(gl.FRAGMENT_SHADER); }, 'Unable to create fragment WebGLShader.');
    callAndCheck(gl, function () { return gl.shaderSource(fragmentShader, fragmentShaderSource); });
    callAndCheck(gl, function () { return gl.compileShader(fragmentShader); });
    if (gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS) === false) {
        console.log(gl.getShaderInfoLog(fragmentShader));
        throw new Error('Failed to compile fragment shader.');
    }
    return fragmentShader;
}
exports.createFragmentShader = createFragmentShader;
function createProgram(gl) {
    return throwIfNull(gl, function () { return gl.createProgram(); }, 'Unable to create WebGLProgram.');
}
exports.createProgram = createProgram;
function linkProgram(gl, program) {
    callAndCheck(gl, function () { return gl.linkProgram(program); });
    if (gl.getProgramParameter(program, gl.LINK_STATUS) === false) {
        console.log(gl.getProgramInfoLog(program));
        throw new Error('Failed to link vertex and fragment shaders.');
    }
}
exports.linkProgram = linkProgram;
function validateProgram(gl, program) {
    callAndCheck(gl, function () { return gl.validateProgram(program); });
    if (gl.getProgramParameter(program, gl.VALIDATE_STATUS) === false) {
        console.log(gl.getProgramInfoLog(program));
        throw new Error('Shader program validation failed.');
    }
}
exports.validateProgram = validateProgram;
function createStaticVertexBuffer(gl, data) {
    var buffer = throwIfNull(gl, function () { return gl.createBuffer(); }, 'Unable to create WebGLBuffer');
    callAndCheck(gl, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, buffer); });
    callAndCheck(gl, function () { return gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW); });
    return buffer;
}
exports.createStaticVertexBuffer = createStaticVertexBuffer;
function createStaticIndexBuffer(gl, data) {
    var buffer = throwIfNull(gl, function () { return gl.createBuffer(); }, 'Unable to create WebGLBuffer');
    callAndCheck(gl, function () { return gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer); });
    callAndCheck(gl, function () { return gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, gl.STATIC_DRAW); });
    return buffer;
}
exports.createStaticIndexBuffer = createStaticIndexBuffer;
function queryMaxTextureSize(gl) {
    if (MAX_TEXTURE_SIZE != null) {
        return MAX_TEXTURE_SIZE;
    }
    MAX_TEXTURE_SIZE =
        callAndCheck(gl, function () { return gl.getParameter(gl.MAX_TEXTURE_SIZE); });
    return MAX_TEXTURE_SIZE;
}
exports.queryMaxTextureSize = queryMaxTextureSize;
function getChannelsPerTexture() {
    if (isWebGL2Enabled()) {
        return 1;
    }
    return 4;
}
exports.getChannelsPerTexture = getChannelsPerTexture;
function createTexture(gl) {
    return throwIfNull(gl, function () { return gl.createTexture(); }, 'Unable to create WebGLTexture.');
}
exports.createTexture = createTexture;
function validateTextureSize(gl, width, height) {
    var maxTextureSize = queryMaxTextureSize(gl);
    if ((width <= 0) || (height <= 0)) {
        var requested = '[' + width + 'x' + height + ']';
        throw new Error('Requested texture size ' + requested + ' is invalid.');
    }
    if ((width > maxTextureSize) || (height > maxTextureSize)) {
        var requested = '[' + width + 'x' + height + ']';
        var max = '[' + maxTextureSize + 'x' + maxTextureSize + ']';
        throw new Error('Requested texture size ' + requested +
            ' greater than WebGL maximum on this browser / GPU ' + max + '.');
    }
}
exports.validateTextureSize = validateTextureSize;
function createFramebuffer(gl) {
    return throwIfNull(gl, function () { return gl.createFramebuffer(); }, 'Unable to create WebGLFramebuffer.');
}
exports.createFramebuffer = createFramebuffer;
function bindVertexBufferToProgramAttribute(gl, program, attribute, buffer, arrayEntriesPerItem, itemStrideInBytes, itemOffsetInBytes) {
    var loc = gl.getAttribLocation(program, attribute);
    if (loc === -1) {
        var error = new Error('Unable to get attribute "' + attribute + '" on WebGLProgram.');
        error.namedVertexAttributeNotFound = attribute;
        throw error;
    }
    callAndCheck(gl, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, buffer); });
    callAndCheck(gl, function () { return gl.vertexAttribPointer(loc, arrayEntriesPerItem, gl.FLOAT, false, itemStrideInBytes, itemOffsetInBytes); });
    callAndCheck(gl, function () { return gl.enableVertexAttribArray(loc); });
}
exports.bindVertexBufferToProgramAttribute = bindVertexBufferToProgramAttribute;
function bindTextureUnit(gl, texture, textureUnit) {
    validateTextureUnit(gl, textureUnit);
    callAndCheck(gl, function () { return gl.activeTexture(gl.TEXTURE0 + textureUnit); });
    callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, texture); });
}
exports.bindTextureUnit = bindTextureUnit;
function unbindTextureUnit(gl, textureUnit) {
    validateTextureUnit(gl, textureUnit);
    callAndCheck(gl, function () { return gl.activeTexture(gl.TEXTURE0 + textureUnit); });
    callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
}
exports.unbindTextureUnit = unbindTextureUnit;
function getProgramUniformLocationOrThrow(gl, program, uniformName) {
    return throwIfNull(gl, function () { return gl.getUniformLocation(program, uniformName); }, 'uniform "' + uniformName + '" not present in program.');
}
exports.getProgramUniformLocationOrThrow = getProgramUniformLocationOrThrow;
function bindTextureToProgramUniformSampler(gl, program, texture, uniformSamplerName, textureUnit) {
    callAndCheck(gl, function () { return bindTextureUnit(gl, texture, textureUnit); });
    var samplerLocation = getProgramUniformLocationOrThrow(gl, program, uniformSamplerName);
    callAndCheck(gl, function () { return gl.uniform1i(samplerLocation, textureUnit); });
}
exports.bindTextureToProgramUniformSampler = bindTextureToProgramUniformSampler;
function bindCanvasToFramebuffer(gl) {
    callAndCheck(gl, function () { return gl.bindFramebuffer(gl.FRAMEBUFFER, null); });
    callAndCheck(gl, function () { return gl.viewport(0, 0, gl.canvas.width, gl.canvas.height); });
    callAndCheck(gl, function () { return gl.scissor(0, 0, gl.canvas.width, gl.canvas.height); });
}
exports.bindCanvasToFramebuffer = bindCanvasToFramebuffer;
function bindColorTextureToFramebuffer(gl, texture, framebuffer) {
    callAndCheck(gl, function () { return gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer); });
    callAndCheck(gl, function () { return gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0); });
}
exports.bindColorTextureToFramebuffer = bindColorTextureToFramebuffer;
function unbindColorTextureFromFramebuffer(gl, framebuffer) {
    callAndCheck(gl, function () { return gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer); });
    callAndCheck(gl, function () { return gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0); });
}
exports.unbindColorTextureFromFramebuffer = unbindColorTextureFromFramebuffer;
function validateFramebuffer(gl) {
    var status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error('Error binding framebuffer: ' + getFramebufferErrorMessage(gl, status));
    }
}
exports.validateFramebuffer = validateFramebuffer;
function getFramebufferErrorMessage(gl, status) {
    switch (status) {
        case gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            return 'FRAMEBUFFER_INCOMPLETE_ATTACHMENT';
        case gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            return 'FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT';
        case gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:
            return 'FRAMEBUFFER_INCOMPLETE_DIMENSIONS';
        case gl.FRAMEBUFFER_UNSUPPORTED:
            return 'FRAMEBUFFER_UNSUPPORTED';
        default:
            return 'unknown error ' + status;
    }
}
exports.getFramebufferErrorMessage = getFramebufferErrorMessage;
function throwIfNull(gl, returnTOrNull, failureMessage) {
    var tOrNull = callAndCheck(gl, function () { return returnTOrNull(); });
    if (tOrNull == null) {
        throw new Error(failureMessage);
    }
    return tOrNull;
}
function validateTextureUnit(gl, textureUnit) {
    var maxTextureUnit = gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS - 1;
    var glTextureUnit = textureUnit + gl.TEXTURE0;
    if (glTextureUnit < gl.TEXTURE0 || glTextureUnit > maxTextureUnit) {
        var textureUnitRange = '[gl.TEXTURE0, gl.TEXTURE' + maxTextureUnit + ']';
        throw new Error('textureUnit must be in ' + textureUnitRange + '.');
    }
}
function getTextureShapeFromLogicalShape(gl, logicalShape, preferredTexShape) {
    var maxTexSize = queryMaxTextureSize(gl);
    var size = util.sizeFromShape(logicalShape);
    if (preferredTexShape != null) {
        var sizePreferred = util.sizeFromShape(preferredTexShape);
        util.assert(size === sizePreferred, "Size of shape (" + size + ") must match size of " +
            ("preferredShape (" + sizePreferred + ")"));
        if (preferredTexShape[0] <= maxTexSize &&
            preferredTexShape[1] <= maxTexSize) {
            return preferredTexShape;
        }
    }
    if (logicalShape.length <= 1 && size <= maxTexSize) {
        return [size, 1];
    }
    else if (logicalShape.length === 2 && logicalShape[0] <= maxTexSize &&
        logicalShape[1] <= maxTexSize) {
        return logicalShape;
    }
    else if (logicalShape.length === 3 && logicalShape[0] <= maxTexSize &&
        logicalShape[1] * logicalShape[2] <= maxTexSize) {
        return [logicalShape[0], logicalShape[1] * logicalShape[2]];
    }
    else {
        return util.sizeToSquarishShape(size);
    }
}
exports.getTextureShapeFromLogicalShape = getTextureShapeFromLogicalShape;

},{"../../util":86}],59:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var graph_1 = require("./graph");
var graph_util = require("./graph_util");
var add_1 = require("./ops/add");
var argmax_1 = require("./ops/argmax");
var argmaxequals_1 = require("./ops/argmaxequals");
var concat3d_1 = require("./ops/concat3d");
var convolution_1 = require("./ops/convolution");
var divide_1 = require("./ops/divide");
var element_wise_activation_1 = require("./ops/element_wise_activation");
var element_wise_cost_1 = require("./ops/element_wise_cost");
var exp_1 = require("./ops/exp");
var linear_combination_1 = require("./ops/linear_combination");
var log_1 = require("./ops/log");
var matmul_1 = require("./ops/matmul");
var max_pool_1 = require("./ops/max_pool");
var multiply_1 = require("./ops/multiply");
var reduce_sum_1 = require("./ops/reduce_sum");
var reshape_1 = require("./ops/reshape");
var softmax_1 = require("./ops/softmax");
var split_1 = require("./ops/split");
var subtract_1 = require("./ops/subtract");
function emitFromGraphNodes(nodes) {
    var ops = [];
    nodes.forEach(function (node) { return Array.prototype.push.apply(ops, emitOpFromNode(node)); });
    return ops;
}
exports.emitFromGraphNodes = emitFromGraphNodes;
function emitOpFromNode(node) {
    if (node instanceof graph_1.ReshapeNode) {
        return [new reshape_1.Reshape(node.inputs[graph_1.ReshapeNode.X], node.output)];
    }
    else if (node instanceof graph_1.MatMulNode) {
        var x1 = node.inputs[graph_1.MatMulNode.X1];
        var x2 = node.inputs[graph_1.MatMulNode.X2];
        return [new matmul_1.MatMul(x1, x2, node.output)];
    }
    else if (node instanceof graph_1.Convolution2DNode) {
        var w = node.inputs[graph_1.Convolution2DNode.W];
        var x = node.inputs[graph_1.Convolution2DNode.X];
        var b = node.inputs[graph_1.Convolution2DNode.B];
        return [new convolution_1.Convolution2D(w, x, b, node.output, node.fieldSize, node.outputDepth, node.stride, node.zeroPad)];
    }
    else if (node instanceof graph_1.MaxPoolNode) {
        var x = node.inputs[graph_1.MaxPoolNode.X];
        return [new max_pool_1.MaxPool(x, node.output, node.fieldSize, node.stride, node.zeroPad)];
    }
    else if (node instanceof graph_1.ExpNode) {
        return [new exp_1.Exp(node.inputs[graph_1.ExpNode.X], node.output)];
    }
    else if (node instanceof graph_1.LogNode) {
        return [new log_1.Log(node.inputs[graph_1.LogNode.X], node.output)];
    }
    else if (node instanceof graph_1.ReLUNode) {
        return [new element_wise_activation_1.ReLU(node.inputs[graph_1.ReLUNode.X], node.output)];
    }
    else if (node instanceof graph_1.TanHNode) {
        return [new element_wise_activation_1.TanH(node.inputs[graph_1.TanHNode.X], node.output)];
    }
    else if (node instanceof graph_1.SigmoidNode) {
        return [new element_wise_activation_1.Sigmoid(node.inputs[graph_1.SigmoidNode.X], node.output)];
    }
    else if (node instanceof graph_1.SoftmaxCrossEntropyCostNode) {
        var x = node.inputs[graph_1.SoftmaxCrossEntropyCostNode.X];
        var target = node.inputs[graph_1.SoftmaxCrossEntropyCostNode.TARGET];
        return [new softmax_1.SoftmaxCrossEntropyCost(x, target, node.output)];
    }
    else if (node instanceof graph_1.SoftmaxNode) {
        return [new softmax_1.Softmax(node.inputs[graph_1.SoftmaxNode.X], node.output)];
    }
    else if (node instanceof graph_1.MeanSquaredCostNode) {
        var label = node.inputs[graph_1.MeanSquaredCostNode.LABEL];
        var prediction = node.inputs[graph_1.MeanSquaredCostNode.PREDICTION];
        return [new element_wise_cost_1.MeanSquaredCost(label, prediction, node.output)];
    }
    else if (node instanceof graph_1.ArgMaxEqualsNode) {
        return [new argmaxequals_1.ArgMaxEquals(node.inputs[graph_1.ArgMaxEqualsNode.X1], node.inputs[graph_1.ArgMaxEqualsNode.X2], node.output)];
    }
    else if (node instanceof graph_1.ArgMaxNode) {
        return [new argmax_1.ArgMax(node.x, node.output)];
    }
    else if (node instanceof graph_1.FusedLinearCombinationNode) {
        return [new linear_combination_1.LinearCombination(node.inputs[graph_1.FusedLinearCombinationNode.T1], node.inputs[graph_1.FusedLinearCombinationNode.T2], node.inputs[graph_1.FusedLinearCombinationNode.C1], node.inputs[graph_1.FusedLinearCombinationNode.C2], node.output)];
    }
    else if (node instanceof graph_1.Concat3DNode) {
        return [new concat3d_1.Concat3D(node.inputs[graph_1.Concat3DNode.X1], node.inputs[graph_1.Concat3DNode.X2], node.axis, node.output)];
    }
    else if (node instanceof graph_1.SquareNode) {
        return [new element_wise_activation_1.Square(node.inputs[graph_1.SquareNode.X], node.output)];
    }
    else if (node instanceof graph_1.AddNode) {
        return [new add_1.Add(node.inputs[graph_1.AddNode.T1], node.inputs[graph_1.AddNode.T2], node.output)];
    }
    else if (node instanceof graph_1.SubtractNode) {
        return [new subtract_1.Subtract(node.inputs[graph_1.SubtractNode.T1], node.inputs[graph_1.SubtractNode.T2], node.output)];
    }
    else if (node instanceof graph_1.MultiplyNode) {
        return [new multiply_1.Multiply(node.inputs[graph_1.MultiplyNode.T1], node.inputs[graph_1.MultiplyNode.T2], node.output)];
    }
    else if (node instanceof graph_1.DivideNode) {
        return [new divide_1.Divide(node.inputs[graph_1.DivideNode.T1], node.inputs[graph_1.DivideNode.T2], node.output)];
    }
    else if (node instanceof graph_1.SplitNode) {
        return [new split_1.Split(node.inputs[graph_1.SplitNode.X], node.outputs)];
    }
    else if (node instanceof graph_1.ReduceSumNode) {
        return [new reduce_sum_1.ReduceSum(node.inputs[graph_1.ReduceSumNode.X], node.output)];
    }
    else if (graph_util.isInputNode(node)) {
        return [];
    }
    else {
        throw Error('Unsupported node type: ' + node.constructor.name);
    }
}

},{"./graph":7,"./graph_util":10,"./ops/add":60,"./ops/argmax":61,"./ops/argmaxequals":62,"./ops/concat3d":63,"./ops/convolution":64,"./ops/divide":65,"./ops/element_wise_activation":66,"./ops/element_wise_cost":67,"./ops/exp":68,"./ops/linear_combination":69,"./ops/log":70,"./ops/matmul":71,"./ops/max_pool":72,"./ops/multiply":73,"./ops/reduce_sum":75,"./ops/reshape":76,"./ops/softmax":77,"./ops/split":78,"./ops/subtract":79}],60:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var ndarray_1 = require("../math/ndarray");
var util = require("../util");
var op_1 = require("./op");
var Add = (function (_super) {
    __extends(Add, _super);
    function Add(x1Tensor, x2Tensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.x1Tensor = x1Tensor;
        _this.x2Tensor = x2Tensor;
        _this.yTensor = yTensor;
        util.assert(util.sizeFromShape(x1Tensor.shape) === 1 ||
            util.sizeFromShape(x2Tensor.shape) === 1 ||
            util.arraysEqual(x1Tensor.shape, x2Tensor.shape), 'One of t1 or t2 must be a scalar, or t1 and t2 must have ' +
            'the same shape');
        return _this;
    }
    Add.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        math.scope(function (keep) {
            var result;
            if (util.isScalarShape(x1.shape)) {
                result = math.scalarPlusArray(x1, x2);
            }
            else if (util.isScalarShape(x2.shape)) {
                result = math.scalarPlusArray(x2, x1);
            }
            else {
                result = math.add(x1, x2);
            }
            inferenceArrays.set(_this.yTensor, keep(result));
        });
    };
    Add.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        var dy = gradientArrays.get(this.yTensor);
        math.scope(function (keep) {
            if (graph_util.shouldBackProp(_this.x1Tensor)) {
                if (util.isScalarShape(_this.x1Tensor.shape)) {
                    var sum = math.sum(dy);
                    if (_this.dySizeScalar == null) {
                        _this.dySizeScalar = ndarray_1.Scalar.new(dy.size);
                    }
                    gradientArrays.set(_this.x1Tensor, keep(math.divide(sum, _this.dySizeScalar)));
                }
                else {
                    gradientArrays.set(_this.x1Tensor, dy);
                }
            }
            if (graph_util.shouldBackProp(_this.x2Tensor)) {
                if (util.isScalarShape(_this.x2Tensor.shape)) {
                    var sum = math.sum(dy);
                    if (_this.dySizeScalar == null) {
                        _this.dySizeScalar = ndarray_1.Scalar.new(dy.size);
                    }
                    gradientArrays.set(_this.x2Tensor, keep(math.divide(sum, _this.dySizeScalar)));
                }
                else {
                    gradientArrays.set(_this.x2Tensor, dy);
                }
            }
        });
    };
    Add.prototype.dispose = function () {
        if (this.dySizeScalar != null) {
            this.dySizeScalar.dispose();
        }
    };
    return Add;
}(op_1.Operation));
exports.Add = Add;

},{"../graph_util":10,"../math/ndarray":22,"../util":86,"./op":74}],61:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var op_1 = require("./op");
var ArgMax = (function (_super) {
    __extends(ArgMax, _super);
    function ArgMax(xTensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.xTensor = xTensor;
        _this.yTensor = yTensor;
        return _this;
    }
    ArgMax.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x = inferenceArrays.get(this.xTensor);
        math.scope(function (keep) {
            inferenceArrays.set(_this.yTensor, keep(math.argMax(x)));
        });
    };
    ArgMax.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        throw new Error('ArgMax backprop unimplemented');
    };
    return ArgMax;
}(op_1.Operation));
exports.ArgMax = ArgMax;

},{"./op":74}],62:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var op_1 = require("./op");
var ArgMaxEquals = (function (_super) {
    __extends(ArgMaxEquals, _super);
    function ArgMaxEquals(x1Tensor, x2Tensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.x1Tensor = x1Tensor;
        _this.x2Tensor = x2Tensor;
        _this.yTensor = yTensor;
        return _this;
    }
    ArgMaxEquals.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        math.scope(function (keep) {
            inferenceArrays.set(_this.yTensor, keep(math.argMaxEquals(x1, x2)));
        });
    };
    ArgMaxEquals.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        throw new Error('ArgMaxEquals backprop unimplemented');
    };
    return ArgMaxEquals;
}(op_1.Operation));
exports.ArgMaxEquals = ArgMaxEquals;

},{"./op":74}],63:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var concat3d_util = require("../math/concat3d_util");
var op_1 = require("./op");
var Concat3D = (function (_super) {
    __extends(Concat3D, _super);
    function Concat3D(x1Tensor, x2Tensor, axis, yTensor) {
        var _this = _super.call(this) || this;
        _this.x1Tensor = x1Tensor;
        _this.x2Tensor = x2Tensor;
        _this.axis = axis;
        _this.yTensor = yTensor;
        concat3d_util.assertConcat3DShapesMatch(x1Tensor.shape, x2Tensor.shape, axis);
        return _this;
    }
    Concat3D.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        math.scope(function (keep) {
            var concatResult = math.concat3D(x1, x2, _this.axis);
            inferenceArrays.set(_this.yTensor, keep(concatResult));
        });
    };
    Concat3D.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        throw new Error('Concat3D backprop not implemented.');
    };
    return Concat3D;
}(op_1.Operation));
exports.Concat3D = Concat3D;

},{"../math/concat3d_util":15,"./op":74}],64:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../math/conv_util");
var util = require("../util");
var op_1 = require("./op");
var Convolution2D = (function (_super) {
    __extends(Convolution2D, _super);
    function Convolution2D(wTensor, xTensor, bTensor, yTensor, fieldSize, outputDepth, stride, zeroPad) {
        if (stride === void 0) { stride = 1; }
        var _this = _super.call(this) || this;
        _this.wTensor = wTensor;
        _this.xTensor = xTensor;
        _this.bTensor = bTensor;
        _this.yTensor = yTensor;
        _this.fieldSize = fieldSize;
        _this.outputDepth = outputDepth;
        _this.stride = stride;
        _this.assertWeightsShape(wTensor.shape);
        _this.zeroPad = zeroPad != null ?
            zeroPad :
            conv_util.computeDefaultPad(_this.xTensor.shape, _this.fieldSize, _this.stride);
        util.assert(util.isInt(_this.zeroPad), "The zero padding (" + _this.zeroPad + ") must be an integer. Change the " +
            "stride and/or zero pad parameters");
        return _this;
    }
    Convolution2D.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var weights = inferenceArrays.get(this.wTensor);
        var biases = inferenceArrays.get(this.bTensor);
        var x = inferenceArrays.get(this.xTensor);
        math.scope(function (keep) {
            inferenceArrays.set(_this.yTensor, keep(math.conv2d(x, weights, biases, _this.stride, _this.zeroPad)));
        });
    };
    Convolution2D.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var weights = inferenceArrays.get(this.wTensor);
        var x = inferenceArrays.get(this.xTensor);
        var dy = gradientArrays.get(this.yTensor);
        math.scope(function (keep) {
            var _a = math.conv2dBackProp(x, dy, weights, _this.stride, _this.zeroPad), dw = _a.dw, db = _a.db, dx = _a.dx;
            gradientArrays.set(_this.wTensor, keep(dw));
            gradientArrays.set(_this.bTensor, keep(db));
            gradientArrays.set(_this.xTensor, keep(dx));
        });
    };
    Convolution2D.prototype.assertWeightsShape = function (weightsShape) {
        util.assert(weightsShape[0] === this.fieldSize &&
            weightsShape[1] === this.fieldSize &&
            weightsShape[2] === this.xTensor.shape[2] &&
            weightsShape[3] === this.outputDepth, "weights must be of shape [" + this.fieldSize + "," + this.fieldSize + "," +
            (this.xTensor.shape[2] + "," + this.outputDepth + "] but they are of") +
            ("shape [" + weightsShape + "]"));
    };
    return Convolution2D;
}(op_1.Operation));
exports.Convolution2D = Convolution2D;

},{"../math/conv_util":16,"../util":86,"./op":74}],65:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var util = require("../util");
var op_1 = require("./op");
var Divide = (function (_super) {
    __extends(Divide, _super);
    function Divide(x1Tensor, x2Tensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.x1Tensor = x1Tensor;
        _this.x2Tensor = x2Tensor;
        _this.yTensor = yTensor;
        util.assert(util.sizeFromShape(x1Tensor.shape) === 1 ||
            util.sizeFromShape(x2Tensor.shape) === 1 ||
            util.arraysEqual(x1Tensor.shape, x2Tensor.shape), 'One of t1 or t2 must be a scalar, or t1 and t2 must have ' +
            'the same shape');
        return _this;
    }
    Divide.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var t1 = inferenceArrays.get(this.x1Tensor);
        var t2 = inferenceArrays.get(this.x2Tensor);
        math.scope(function (keep) {
            var result;
            if (util.isScalarShape(t1.shape)) {
                result = math.scalarDividedByArray(t1, t2);
            }
            else if (util.isScalarShape(t2.shape)) {
                result = math.arrayDividedByScalar(t1, t2);
            }
            else {
                result = math.divide(t1, t2);
            }
            inferenceArrays.set(_this.yTensor, keep(result));
        });
    };
    Divide.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        var dy = gradientArrays.get(this.yTensor);
        var x1IsScalar = util.isScalarShape(x1.shape);
        var x2IsScalar = util.isScalarShape(x2.shape);
        math.scope(function (keep) {
            if (graph_util.shouldBackProp(_this.x1Tensor)) {
                if (x1IsScalar) {
                    var div = math.divide(dy, x2);
                    gradientArrays.set(_this.x1Tensor, keep(math.sum(div)));
                    div.dispose();
                }
                else if (x2IsScalar) {
                    gradientArrays.set(_this.x1Tensor, keep(math.arrayDividedByScalar(dy, x2)));
                }
                else {
                    gradientArrays.set(_this.x1Tensor, keep(math.divide(dy, x2)));
                }
            }
            if (graph_util.shouldBackProp(_this.x2Tensor)) {
                var x2Squared = math.elementWiseMul(x2, x2);
                var x1OverX2Squared = void 0;
                if (x2IsScalar) {
                    x1OverX2Squared = math.arrayDividedByScalar(x1, x2Squared);
                }
                else if (x1IsScalar) {
                    x1OverX2Squared = math.scalarDividedByArray(x1, x2Squared);
                }
                else {
                    x1OverX2Squared = math.divide(x1, x2Squared);
                }
                var dx2 = math.neg(x1OverX2Squared);
                var dyTimesDerivative = math.elementWiseMul(dy, dx2);
                if (x2IsScalar) {
                    gradientArrays.set(_this.x2Tensor, keep(math.sum(dyTimesDerivative)));
                }
                else {
                    gradientArrays.set(_this.x2Tensor, keep(dyTimesDerivative));
                }
            }
        });
    };
    return Divide;
}(op_1.Operation));
exports.Divide = Divide;

},{"../graph_util":10,"../util":86,"./op":74}],66:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var activation_functions_1 = require("../math/activation_functions");
var op_1 = require("./op");
var ElementWiseActivation = (function (_super) {
    __extends(ElementWiseActivation, _super);
    function ElementWiseActivation(xTensor, yTensor, func) {
        var _this = _super.call(this) || this;
        _this.xTensor = xTensor;
        _this.yTensor = yTensor;
        _this.func = func;
        return _this;
    }
    ElementWiseActivation.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x = inferenceArrays.get(this.xTensor);
        math.scope(function (keep) {
            inferenceArrays.set(_this.yTensor, keep(_this.func.output(math, x)));
        });
    };
    ElementWiseActivation.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var x = inferenceArrays.get(this.xTensor);
        var y = inferenceArrays.get(this.yTensor);
        var dy = gradientArrays.get(this.yTensor);
        math.scope(function (keep) {
            var dydx = _this.func.der(math, x, y);
            gradientArrays.set(_this.xTensor, keep(math.elementWiseMul(dy, dydx)));
            dydx.dispose();
        });
    };
    return ElementWiseActivation;
}(op_1.Operation));
exports.ElementWiseActivation = ElementWiseActivation;
var ReLU = (function (_super) {
    __extends(ReLU, _super);
    function ReLU(xTensor, yTensor) {
        return _super.call(this, xTensor, yTensor, new activation_functions_1.ReLUFunc()) || this;
    }
    return ReLU;
}(ElementWiseActivation));
exports.ReLU = ReLU;
var TanH = (function (_super) {
    __extends(TanH, _super);
    function TanH(xTensor, yTensor) {
        return _super.call(this, xTensor, yTensor, new activation_functions_1.TanHFunc()) || this;
    }
    return TanH;
}(ElementWiseActivation));
exports.TanH = TanH;
var Sigmoid = (function (_super) {
    __extends(Sigmoid, _super);
    function Sigmoid(xTensor, yTensor) {
        return _super.call(this, xTensor, yTensor, new activation_functions_1.SigmoidFunc()) || this;
    }
    return Sigmoid;
}(ElementWiseActivation));
exports.Sigmoid = Sigmoid;
var Square = (function (_super) {
    __extends(Square, _super);
    function Square(xTensor, yTensor) {
        return _super.call(this, xTensor, yTensor, new activation_functions_1.SquareFunc()) || this;
    }
    return Square;
}(ElementWiseActivation));
exports.Square = Square;

},{"../math/activation_functions":14,"./op":74}],67:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var cost_functions_1 = require("../math/cost_functions");
var ndarray_1 = require("../math/ndarray");
var util = require("../util");
var op_1 = require("./op");
var ElementWiseCost = (function (_super) {
    __extends(ElementWiseCost, _super);
    function ElementWiseCost(x1Tensor, x2Tensor, yTensor, func) {
        var _this = _super.call(this) || this;
        _this.x1Tensor = x1Tensor;
        _this.x2Tensor = x2Tensor;
        _this.yTensor = yTensor;
        _this.func = func;
        _this.oneOverNScalar = ndarray_1.Scalar.new(1 / util.sizeFromShape(x1Tensor.shape));
        return _this;
    }
    ElementWiseCost.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        math.scope(function (keep) {
            var elementWiseCost = _this.func.cost(math, x1, x2);
            var sum = math.sum(elementWiseCost);
            var result = math.scalarTimesArray(_this.oneOverNScalar, sum);
            inferenceArrays.set(_this.yTensor, keep(result));
        });
    };
    ElementWiseCost.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        math.scope(function (keep) {
            if (graph_util.shouldBackProp(_this.x1Tensor)) {
                gradientArrays.set(_this.x1Tensor, keep(_this.func.der(math, x1, x2)));
            }
            if (graph_util.shouldBackProp(_this.x2Tensor)) {
                gradientArrays.set(_this.x2Tensor, keep(_this.func.der(math, x2, x1)));
            }
        });
    };
    ElementWiseCost.prototype.dispose = function () {
        this.func.dispose();
        this.oneOverNScalar.dispose();
    };
    return ElementWiseCost;
}(op_1.Operation));
exports.ElementWiseCost = ElementWiseCost;
var MeanSquaredCost = (function (_super) {
    __extends(MeanSquaredCost, _super);
    function MeanSquaredCost(x1Tensor, x2Tensor, yTensor) {
        return _super.call(this, x1Tensor, x2Tensor, yTensor, new cost_functions_1.SquareCostFunc()) || this;
    }
    return MeanSquaredCost;
}(ElementWiseCost));
exports.MeanSquaredCost = MeanSquaredCost;

},{"../graph_util":10,"../math/cost_functions":18,"../math/ndarray":22,"../util":86,"./op":74}],68:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var op_1 = require("./op");
var Exp = (function (_super) {
    __extends(Exp, _super);
    function Exp(xTensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.xTensor = xTensor;
        _this.yTensor = yTensor;
        return _this;
    }
    Exp.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x = inferenceArrays.get(this.xTensor);
        math.scope(function (keep) {
            inferenceArrays.set(_this.yTensor, keep(math.exp(x)));
        });
    };
    Exp.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var y = inferenceArrays.get(this.yTensor);
        var dy = gradientArrays.get(this.yTensor);
        math.scope(function (keep) {
            if (graph_util.shouldBackProp(_this.xTensor)) {
                gradientArrays.set(_this.xTensor, keep(math.elementWiseMul(y, dy)));
            }
        });
    };
    return Exp;
}(op_1.Operation));
exports.Exp = Exp;

},{"../graph_util":10,"./op":74}],69:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var op_1 = require("./op");
var LinearCombination = (function (_super) {
    __extends(LinearCombination, _super);
    function LinearCombination(x1Tensor, x2Tensor, c1Tensor, c2Tensor, outTensor) {
        var _this = _super.call(this) || this;
        _this.x1Tensor = x1Tensor;
        _this.x2Tensor = x2Tensor;
        _this.c1Tensor = c1Tensor;
        _this.c2Tensor = c2Tensor;
        _this.outTensor = outTensor;
        return _this;
    }
    LinearCombination.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        var c1 = inferenceArrays.get(this.c1Tensor).asScalar();
        var c2 = inferenceArrays.get(this.c2Tensor).asScalar();
        math.scope(function (keep) {
            inferenceArrays.set(_this.outTensor, keep(math.scaledArrayAdd(c1, x1, c2, x2)));
        });
    };
    LinearCombination.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        var c1 = inferenceArrays.get(this.c1Tensor);
        var c2 = inferenceArrays.get(this.c2Tensor);
        var dy = gradientArrays.get(this.outTensor);
        math.scope(function (keep) {
            if (graph_util.shouldBackProp(_this.x1Tensor)) {
                gradientArrays.set(_this.x1Tensor, keep(math.scalarTimesArray(c1, dy)));
            }
            if (graph_util.shouldBackProp(_this.x2Tensor)) {
                gradientArrays.set(_this.x2Tensor, keep(math.scalarTimesArray(c2, dy)));
            }
            if (graph_util.shouldBackProp(_this.c1Tensor)) {
                var dotProduct1 = math.elementWiseMul(x1, dy);
                gradientArrays.set(_this.c1Tensor, keep(math.sum(dotProduct1)));
            }
            if (graph_util.shouldBackProp(_this.c2Tensor)) {
                var dotProduct2 = math.elementWiseMul(x2, dy);
                gradientArrays.set(_this.c2Tensor, keep(math.sum(dotProduct2)));
            }
        });
    };
    return LinearCombination;
}(op_1.Operation));
exports.LinearCombination = LinearCombination;

},{"../graph_util":10,"./op":74}],70:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var op_1 = require("./op");
var Log = (function (_super) {
    __extends(Log, _super);
    function Log(xTensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.xTensor = xTensor;
        _this.yTensor = yTensor;
        return _this;
    }
    Log.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x = inferenceArrays.get(this.xTensor);
        math.scope(function (keep) {
            inferenceArrays.set(_this.yTensor, keep(math.log(x)));
        });
    };
    Log.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var x = inferenceArrays.get(this.xTensor);
        var dy = gradientArrays.get(this.yTensor);
        math.scope(function (keep) {
            if (graph_util.shouldBackProp(_this.xTensor)) {
                gradientArrays.set(_this.xTensor, keep(math.divide(dy, x)));
            }
        });
    };
    return Log;
}(op_1.Operation));
exports.Log = Log;

},{"../graph_util":10,"./op":74}],71:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var math_1 = require("../math/math");
var op_1 = require("./op");
var MatMul = (function (_super) {
    __extends(MatMul, _super);
    function MatMul(x1Tensor, x2Tensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.x1Tensor = x1Tensor;
        _this.x2Tensor = x2Tensor;
        _this.yTensor = yTensor;
        return _this;
    }
    MatMul.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        math.scope(function (keep) {
            if (x1.shape.length === 2 && x2.shape.length === 2) {
                inferenceArrays.set(_this.yTensor, keep(math.matMul(x1, x2)));
            }
            else if (x1.shape.length === 2 && x2.shape.length === 1) {
                inferenceArrays.set(_this.yTensor, keep(math.matrixTimesVector(x1, x2)));
            }
            else if (x1.shape.length === 1 && x2.shape.length === 2) {
                inferenceArrays.set(_this.yTensor, keep(math.vectorTimesMatrix(x1, x2)));
            }
        });
    };
    MatMul.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        var dy = gradientArrays.get(this.yTensor);
        if (x1.shape.length === 1) {
            x1 = x1.reshape([1, x1.size]);
            dy = dy.reshape([1, dy.size]);
        }
        if (x2.shape.length === 1) {
            x2 = x2.reshape([x2.size, 1]);
            dy = dy.reshape([dy.size, 1]);
        }
        math.scope(function (keep) {
            if (graph_util.shouldBackProp(_this.x1Tensor)) {
                var dx1 = math.matMul(dy, x2, math_1.MatrixOrientation.REGULAR, math_1.MatrixOrientation.TRANSPOSED);
                gradientArrays.set(_this.x1Tensor, keep(_this.x1Tensor.shape.length === 1 ? dx1.as1D() : dx1));
            }
            if (graph_util.shouldBackProp(_this.x2Tensor)) {
                var dx2 = math.matMul(x1, dy, math_1.MatrixOrientation.TRANSPOSED, math_1.MatrixOrientation.REGULAR);
                gradientArrays.set(_this.x2Tensor, keep(_this.x2Tensor.shape.length === 1 ? dx2.as1D() : dx2));
            }
        });
    };
    return MatMul;
}(op_1.Operation));
exports.MatMul = MatMul;

},{"../graph_util":10,"../math/math":19,"./op":74}],72:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../math/conv_util");
var util = require("../util");
var op_1 = require("./op");
var MaxPool = (function (_super) {
    __extends(MaxPool, _super);
    function MaxPool(xTensor, yTensor, fieldSize, stride, pad) {
        if (stride === void 0) { stride = 1; }
        var _this = _super.call(this) || this;
        _this.xTensor = xTensor;
        _this.yTensor = yTensor;
        _this.fieldSize = fieldSize;
        _this.stride = stride;
        if (pad != null) {
            _this.pad = pad;
        }
        else {
            _this.pad = conv_util.computeDefaultPad(xTensor.shape, _this.fieldSize, _this.stride);
        }
        util.assert(util.isInt(_this.pad), "The zero padding (" + _this.pad + ") must be an integer. Change the " +
            "stride and/or zero pad parameters");
        return _this;
    }
    MaxPool.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x = inferenceArrays.get(this.xTensor);
        math.scope(function (keep) {
            inferenceArrays.set(_this.yTensor, keep(math.maxPool(x, _this.fieldSize, _this.stride, _this.pad)));
        });
    };
    MaxPool.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var x = inferenceArrays.get(this.xTensor);
        var dy = gradientArrays.get(this.yTensor);
        math.scope(function (keep) {
            gradientArrays.set(_this.xTensor, keep(math.maxPoolBackprop(dy, x, _this.fieldSize, _this.stride, _this.pad)));
        });
    };
    return MaxPool;
}(op_1.Operation));
exports.MaxPool = MaxPool;

},{"../math/conv_util":16,"../util":86,"./op":74}],73:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var util = require("../util");
var op_1 = require("./op");
var Multiply = (function (_super) {
    __extends(Multiply, _super);
    function Multiply(x1Tensor, x2Tensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.x1Tensor = x1Tensor;
        _this.x2Tensor = x2Tensor;
        _this.yTensor = yTensor;
        util.assert(util.sizeFromShape(x1Tensor.shape) === 1 ||
            util.sizeFromShape(x2Tensor.shape) === 1 ||
            util.arraysEqual(x1Tensor.shape, x2Tensor.shape), 'One of t1 or t2 must be a scalar, or t1 and t2 must have ' +
            'the same shape');
        return _this;
    }
    Multiply.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var t1 = inferenceArrays.get(this.x1Tensor);
        var t2 = inferenceArrays.get(this.x2Tensor);
        math.scope(function (keep) {
            var result;
            if (util.isScalarShape(t1.shape)) {
                result = math.scalarTimesArray(t1, t2);
            }
            else if (util.isScalarShape(t2.shape)) {
                result = math.scalarTimesArray(t2, t1);
            }
            else {
                result = math.elementWiseMul(t1, t2);
            }
            inferenceArrays.set(_this.yTensor, keep(result));
        });
    };
    Multiply.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        var dy = gradientArrays.get(this.yTensor);
        math.scope(function (keep) {
            if (graph_util.shouldBackProp(_this.x1Tensor)) {
                if (util.isScalarShape(_this.x1Tensor.shape)) {
                    var mul = math.elementWiseMul(dy, x2);
                    gradientArrays.set(_this.x1Tensor, keep(math.sum(mul)));
                }
                else if (util.isScalarShape(x2.shape)) {
                    gradientArrays.set(_this.x1Tensor, keep(math.scalarTimesArray(x2, dy)));
                }
                else {
                    gradientArrays.set(_this.x1Tensor, keep(math.elementWiseMul(x2, dy)));
                }
            }
            if (graph_util.shouldBackProp(_this.x2Tensor)) {
                if (util.isScalarShape(_this.x2Tensor.shape)) {
                    var mul = math.elementWiseMul(dy, x1);
                    gradientArrays.set(_this.x2Tensor, keep(math.sum(mul)));
                }
                else if (util.isScalarShape(x1.shape)) {
                    gradientArrays.set(_this.x2Tensor, keep(math.scalarTimesArray(x1, dy)));
                }
                else {
                    gradientArrays.set(_this.x2Tensor, keep(math.elementWiseMul(x1, dy)));
                }
            }
        });
    };
    return Multiply;
}(op_1.Operation));
exports.Multiply = Multiply;

},{"../graph_util":10,"../util":86,"./op":74}],74:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var Operation = (function () {
    function Operation() {
    }
    Operation.prototype.disposeTransientArrays = function (inferenceArrays, gradientArrays) { };
    Operation.prototype.dispose = function () { };
    return Operation;
}());
exports.Operation = Operation;

},{}],75:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var ndarray_1 = require("../math/ndarray");
var util = require("../util");
var op_1 = require("./op");
var ReduceSum = (function (_super) {
    __extends(ReduceSum, _super);
    function ReduceSum(x, outTensor) {
        var _this = _super.call(this) || this;
        _this.x = x;
        _this.outTensor = outTensor;
        util.assertShapesMatch(outTensor.shape, []);
        return _this;
    }
    ReduceSum.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x = inferenceArrays.get(this.x);
        math.scope(function (keep) {
            inferenceArrays.set(_this.outTensor, keep(math.sum(x)));
        });
    };
    ReduceSum.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        if (!graph_util.shouldBackProp(this.x)) {
            return;
        }
        math.scope(function (keep) {
            var dy = gradientArrays.get(_this.outTensor);
            if (_this.ones == null) {
                var xArray = inferenceArrays.get(_this.x);
                _this.ones = ndarray_1.NDArray.zerosLike(xArray);
                _this.ones.fill(1);
            }
            gradientArrays.set(_this.x, keep(math.scalarTimesArray(dy, _this.ones)));
        });
    };
    return ReduceSum;
}(op_1.Operation));
exports.ReduceSum = ReduceSum;

},{"../graph_util":10,"../math/ndarray":22,"../util":86,"./op":74}],76:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../util");
var op_1 = require("./op");
var Reshape = (function (_super) {
    __extends(Reshape, _super);
    function Reshape(xTensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.xTensor = xTensor;
        _this.yTensor = yTensor;
        var xSize = util.sizeFromShape(xTensor.shape);
        var ySize = util.sizeFromShape(yTensor.shape);
        util.assert(xSize === ySize, "The input size (" + xSize + ") and output size (" + ySize + ") must match");
        return _this;
    }
    Reshape.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x = inferenceArrays.get(this.xTensor);
        math.scope(function (keep) {
            inferenceArrays.set(_this.yTensor, keep(math.reshape(x, _this.yTensor.shape)));
        });
    };
    Reshape.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var dy = gradientArrays.get(this.yTensor);
        math.scope(function (keep) {
            gradientArrays.set(_this.xTensor, keep(math.reshape(dy, _this.xTensor.shape)));
        });
    };
    return Reshape;
}(op_1.Operation));
exports.Reshape = Reshape;

},{"../util":86,"./op":74}],77:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_1 = require("../graph");
var ndarray_1 = require("../math/ndarray");
var util = require("../util");
var op_1 = require("./op");
var Softmax = (function (_super) {
    __extends(Softmax, _super);
    function Softmax(logitsTensor, output) {
        var _this = _super.call(this) || this;
        _this.logitsTensor = logitsTensor;
        _this.output = output;
        return _this;
    }
    Softmax.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var logits = inferenceArrays.get(this.logitsTensor);
        return math.scope(function (keep) {
            inferenceArrays.set(_this.output, keep(math.softmax(logits)));
        });
    };
    Softmax.prototype.backProp = function () {
        throw Error('Softmax backprop is not yet implemented');
    };
    return Softmax;
}(op_1.Operation));
exports.Softmax = Softmax;
var SoftmaxCrossEntropyCost = (function (_super) {
    __extends(SoftmaxCrossEntropyCost, _super);
    function SoftmaxCrossEntropyCost(logitsTensor, labelTensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.logitsTensor = logitsTensor;
        _this.labelTensor = labelTensor;
        _this.yTensor = yTensor;
        _this.epsilon = ndarray_1.Scalar.new(1e-5);
        _this.softmaxTensor = new graph_1.Tensor(logitsTensor.shape);
        return _this;
    }
    SoftmaxCrossEntropyCost.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var logits = inferenceArrays.get(this.logitsTensor);
        var label = inferenceArrays.get(this.labelTensor);
        math.scope(function (keep) {
            var softmaxResult = math.softmax(logits);
            inferenceArrays.set(_this.softmaxTensor, keep(softmaxResult));
            inferenceArrays.set(_this.yTensor, keep(crossEntropyCost(math, softmaxResult, label, _this.epsilon)));
        });
    };
    SoftmaxCrossEntropyCost.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var softmax = inferenceArrays.get(this.softmaxTensor);
        var label = inferenceArrays.get(this.labelTensor);
        math.scope(function (keep) {
            gradientArrays.set(_this.logitsTensor, keep(math.sub(softmax, label)));
        });
    };
    SoftmaxCrossEntropyCost.prototype.disposeTransientArrays = function (inferenceArrays, gradientArrays) {
        inferenceArrays.disposeArray(this.softmaxTensor);
    };
    SoftmaxCrossEntropyCost.prototype.dispose = function () {
        this.epsilon.dispose();
    };
    return SoftmaxCrossEntropyCost;
}(op_1.Operation));
exports.SoftmaxCrossEntropyCost = SoftmaxCrossEntropyCost;
function crossEntropyCost(math, y, target, epsilon) {
    util.assert(y.size === target.size, 'The output and target must be the same size');
    return math.scope(function () {
        var yPlusEps = math.scalarPlusArray(epsilon, y);
        var logOutput = math.log(yPlusEps);
        var tarLogOutput = math.elementWiseMul(target, logOutput);
        var costVector = math.neg(tarLogOutput);
        return math.sum(costVector);
    });
}
exports.crossEntropyCost = crossEntropyCost;

},{"../graph":7,"../math/ndarray":22,"../util":86,"./op":74}],78:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var util = require("../util");
var op_1 = require("./op");
var Split = (function (_super) {
    __extends(Split, _super);
    function Split(input, outputs) {
        var _this = _super.call(this) || this;
        _this.input = input;
        _this.outputs = outputs;
        outputs.forEach(function (output) {
            util.assertShapesMatch(input.shape, output.shape);
        });
        return _this;
    }
    Split.prototype.feedForward = function (math, inferenceArrays) {
        var inputArray = inferenceArrays.get(this.input);
        this.outputs.forEach(function (output) {
            inferenceArrays.set(output, inputArray);
        });
    };
    Split.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        if (!graph_util.shouldBackProp(this.input)) {
            return;
        }
        math.scope(function (keep) {
            var dx = math.add(gradientArrays.get(_this.outputs[0]), gradientArrays.get(_this.outputs[1]));
            _this.outputs.slice(2).forEach(function (output) {
                dx = math.add(dx, gradientArrays.get(output));
            });
            gradientArrays.set(_this.input, keep(dx));
        });
    };
    return Split;
}(op_1.Operation));
exports.Split = Split;

},{"../graph_util":10,"../util":86,"./op":74}],79:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var ndarray_1 = require("../math/ndarray");
var util = require("../util");
var op_1 = require("./op");
var Subtract = (function (_super) {
    __extends(Subtract, _super);
    function Subtract(t1, t2, outTensor) {
        var _this = _super.call(this) || this;
        _this.t1 = t1;
        _this.t2 = t2;
        _this.outTensor = outTensor;
        util.assert(util.sizeFromShape(t1.shape) === 1 ||
            util.sizeFromShape(t2.shape) === 1 ||
            util.arraysEqual(t1.shape, t2.shape), 'One of t1 or t2 must be a scalar, or t1 and t2 must have ' +
            'the same shape');
        return _this;
    }
    Subtract.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var t1 = inferenceArrays.get(this.t1);
        var t2 = inferenceArrays.get(this.t2);
        math.scope(function (keep) {
            var result;
            if (util.isScalarShape(t1.shape)) {
                result = math.scalarMinusArray(t1, t2);
            }
            else if (util.isScalarShape(t2.shape)) {
                result = math.arrayMinusScalar(t1, t2);
            }
            else {
                result = math.sub(t1, t2);
            }
            inferenceArrays.set(_this.outTensor, keep(result));
        });
    };
    Subtract.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var t1 = inferenceArrays.get(this.t1);
        var t2 = inferenceArrays.get(this.t2);
        var dy = gradientArrays.get(this.outTensor);
        math.scope(function (keep) {
            if (graph_util.shouldBackProp(_this.t1)) {
                if (util.isScalarShape(_this.t1.shape)) {
                    var sum = math.sum(dy);
                    if (_this.dySizeScalar == null) {
                        _this.dySizeScalar = ndarray_1.Scalar.new(dy.size);
                    }
                    gradientArrays.set(_this.t1, keep(math.divide(sum, _this.dySizeScalar)));
                }
                else {
                    gradientArrays.set(_this.t1, keep(dy));
                }
            }
            if (graph_util.shouldBackProp(_this.t2)) {
                if (util.isScalarShape(_this.t2.shape)) {
                    var sum = math.sum(dy);
                    var negSum = math.neg(sum);
                    if (_this.dySizeScalar == null) {
                        _this.dySizeScalar = ndarray_1.Scalar.new(dy.size);
                    }
                    gradientArrays.set(_this.t2, keep(math.divide(negSum, _this.dySizeScalar)));
                }
                else {
                    gradientArrays.set(_this.t2, keep(math.neg(dy)));
                }
            }
        });
    };
    Subtract.prototype.dispose = function () {
        if (this.dySizeScalar != null) {
            this.dySizeScalar.dispose();
        }
    };
    return Subtract;
}(op_1.Operation));
exports.Subtract = Subtract;

},{"../graph_util":10,"../math/ndarray":22,"../util":86,"./op":74}],80:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var Optimizer = (function () {
    function Optimizer(specifiedVariableList) {
        if (specifiedVariableList != null) {
            this.specifiedVariableNodes = specifiedVariableList;
        }
    }
    return Optimizer;
}());
exports.Optimizer = Optimizer;

},{}],81:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function defaultCompare(a, b) {
    if (a === b) {
        return 0;
    }
    else if (a < b) {
        return -1;
    }
    else {
        return 1;
    }
}
exports.defaultCompare = defaultCompare;
var PriorityQueue = (function () {
    function PriorityQueue(comparator, indexObserver) {
        this.comparator = comparator;
        this.indexObserver = indexObserver;
        this.heap = [];
    }
    PriorityQueue.prototype.enqueue = function (t) {
        this.heap.push(t);
        this.onIndexChanged(t, this.heap.length - 1);
        this.siftUp(this.heap.length - 1);
    };
    PriorityQueue.prototype.dequeue = function () {
        if (this.empty()) {
            throw new Error('dequeue called on empty priority queue.');
        }
        var t = this.heap[0];
        this.swap(0, this.heap.length - 1);
        this.heap.pop();
        this.siftDown(0);
        return t;
    };
    PriorityQueue.prototype.update = function (newT, index) {
        var last = (index === this.heap.length - 1);
        if (!last) {
            this.swap(index, this.heap.length - 1);
        }
        this.heap.pop();
        if (!last) {
            if (this.siftUpIndex(index) !== -1) {
                this.siftUp(index);
            }
            else if (this.siftDownIndex(index) !== -1) {
                this.siftDown(index);
            }
        }
        this.enqueue(newT);
    };
    PriorityQueue.prototype.empty = function () {
        return this.heap.length === 0;
    };
    PriorityQueue.prototype.onIndexChanged = function (t, newIndex) {
        if (this.indexObserver) {
            this.indexObserver(t, newIndex);
        }
    };
    PriorityQueue.prototype.getParentIndex = function (index) {
        if (index === 0) {
            return -1;
        }
        return Math.floor((index - 1) / 2);
    };
    PriorityQueue.prototype.getLeftChildIndex = function (index) {
        var candidate = index * 2 + 1;
        return candidate < this.heap.length ? candidate : -1;
    };
    PriorityQueue.prototype.getRightChildIndex = function (index) {
        var candidate = index * 2 + 2;
        return candidate < this.heap.length ? candidate : -1;
    };
    PriorityQueue.prototype.siftUpIndex = function (index) {
        var parentIndex = this.getParentIndex(index);
        if (parentIndex === -1) {
            return -1;
        }
        if (this.compare(parentIndex, index) > 0) {
            return parentIndex;
        }
        return -1;
    };
    PriorityQueue.prototype.siftUp = function (index) {
        var siftIndex = this.siftUpIndex(index);
        while (siftIndex !== -1) {
            this.swap(index, siftIndex);
            index = siftIndex;
            siftIndex = this.siftUpIndex(index);
        }
    };
    PriorityQueue.prototype.siftDownIndex = function (index) {
        if (index >= this.heap.length) {
            return -1;
        }
        var largestChildIndex = index;
        var leftChildIndex = this.getLeftChildIndex(index);
        if ((leftChildIndex !== -1) &&
            (this.compare(leftChildIndex, largestChildIndex) < 0)) {
            largestChildIndex = leftChildIndex;
        }
        var rightChildIndex = this.getRightChildIndex(index);
        if ((rightChildIndex !== -1) &&
            (this.compare(rightChildIndex, largestChildIndex) < 0)) {
            largestChildIndex = rightChildIndex;
        }
        return (largestChildIndex === index) ? -1 : largestChildIndex;
    };
    PriorityQueue.prototype.siftDown = function (index) {
        var siftIndex = this.siftDownIndex(index);
        while (siftIndex !== -1) {
            this.swap(index, siftIndex);
            index = siftIndex;
            siftIndex = this.siftDownIndex(index);
        }
    };
    PriorityQueue.prototype.compare = function (aIndex, bIndex) {
        return this.comparator(this.heap[aIndex], this.heap[bIndex]);
    };
    PriorityQueue.prototype.swap = function (a, b) {
        var temp = this.heap[a];
        this.heap[a] = this.heap[b];
        this.heap[b] = temp;
        this.onIndexChanged(this.heap[a], a);
        this.onIndexChanged(this.heap[b], b);
    };
    return PriorityQueue;
}());
exports.PriorityQueue = PriorityQueue;

},{}],82:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("./math/ndarray");
var operation_emitter = require("./operation_emitter");
var session_util = require("./session_util");
var tensor_array_map_1 = require("./tensor_array_map");
var util = require("./util");
var FeedDictionary = (function () {
    function FeedDictionary(feedEntries) {
        var _this = this;
        this.dict = {};
        if (feedEntries) {
            feedEntries.forEach(function (entry) { return _this.dict[entry.tensor.id] = entry; });
        }
    }
    return FeedDictionary;
}());
exports.FeedDictionary = FeedDictionary;
var CostReduction;
(function (CostReduction) {
    CostReduction[CostReduction["NONE"] = 0] = "NONE";
    CostReduction[CostReduction["SUM"] = 1] = "SUM";
    CostReduction[CostReduction["MEAN"] = 2] = "MEAN";
})(CostReduction = exports.CostReduction || (exports.CostReduction = {}));
var Session = (function () {
    function Session(graph, math) {
        this.graph = graph;
        this.math = math;
        this.activationArrayMap = new tensor_array_map_1.TensorArrayMap();
        this.gradientArrayMap = new tensor_array_map_1.TensorArrayMap();
        this.runtimeCache = {};
        this.oneScalar = ndarray_1.Scalar.new(1);
    }
    Session.prototype.dispose = function () {
        var _this = this;
        this.activationArrayMap.dispose();
        Object.keys(this.runtimeCache).forEach(function (key) {
            var runtime = _this.runtimeCache[key];
            if (runtime.operations) {
                runtime.operations.forEach(function (op) { return op.dispose(); });
            }
        });
        this.runtimeCache = {};
        if (this.batchSizeScalar != null) {
            this.batchSizeScalar.dispose();
        }
        this.oneScalar.dispose();
    };
    Session.prototype.evalAll = function (tensors, feedEntries) {
        var _this = this;
        return this.math.scope(function () {
            var feed = new FeedDictionary(feedEntries);
            var runtime = _this.getOrCreateRuntime(tensors, feed);
            var activations = _this.activationArrayMap;
            session_util.disposeAndInitializeOperationOutputs(runtime.nodes, activations);
            session_util.disposeTransientOperationArrays(runtime.operations, _this.activationArrayMap, _this.gradientArrayMap);
            session_util.addPersistentArraysToTensorArrayMap(runtime.nodes, activations);
            session_util.loadInputsFromFeedDictionaryToTensorArrayMap(feed, activations, _this.math);
            runtime.operations.forEach(function (op) { return op.feedForward(_this.math, activations); });
            var results = tensors.map(function (x) { return activations.get(x); });
            tensors.forEach(function (x) { return activations.delete(x); });
            session_util.releaseFeedDictionaryInputsFromTensorArrayMap(feed, activations, _this.math);
            return results;
        });
    };
    Session.prototype.eval = function (tensor, feedEntries) {
        return this.evalAll([tensor], feedEntries)[0];
    };
    Session.prototype.train = function (costTensor, feedEntries, batchSize, optimizer, costReduction) {
        var _this = this;
        if (costReduction === void 0) { costReduction = CostReduction.NONE; }
        util.assert(util.isScalarShape(costTensor.shape), 'Cost tensor for training must be a scalar value.');
        if (this.prevBatchSize !== batchSize) {
            this.prevBatchSize = batchSize;
            this.batchSizeScalar = ndarray_1.Scalar.new(batchSize);
        }
        var feed = new FeedDictionary(feedEntries);
        session_util.throwIfFeedDictionaryContainsNDArrays(feed);
        var runtime = this.getOrCreateRuntime([costTensor], feed);
        var inferenceOperations = runtime.operations;
        var backPropOperations = runtime.operations.slice().reverse();
        var activations = this.activationArrayMap;
        var gradients = this.gradientArrayMap;
        gradients.set(costTensor, this.oneScalar);
        session_util.addPersistentArraysToTensorArrayMap(runtime.nodes, activations);
        optimizer.beforeBatch(this.math, batchSize, runtime, activations, gradients);
        return this.math.scope(function (keep, track) {
            var cost = track(ndarray_1.Scalar.new(0));
            for (var i = 0; i < batchSize; ++i) {
                session_util.disposeAndInitializeOperationOutputs(runtime.nodes, activations);
                session_util.disposeAndInitializeOperationInputGradients(runtime.nodes, gradients);
                session_util.disposeTransientOperationArrays(runtime.operations, activations, gradients);
                session_util.loadInputsFromFeedDictionaryToTensorArrayMap(feed, activations, _this.math);
                inferenceOperations.forEach(function (op) { return op.feedForward(_this.math, activations); });
                backPropOperations.forEach(function (op) { return op.backProp(_this.math, activations, gradients); });
                optimizer.afterExample(_this.math, runtime, activations, gradients);
                session_util.releaseFeedDictionaryInputsFromTensorArrayMap(feed, activations, _this.math);
                cost = _this.updateCostForExample(cost, activations.get(costTensor), costReduction);
            }
            optimizer.afterBatch(_this.math, batchSize, runtime, activations, gradients);
            return _this.updateCostForBatch(cost, costReduction);
        });
    };
    Session.prototype.updateCostForExample = function (totalCost, currCost, costReduction) {
        if (costReduction === CostReduction.MEAN ||
            costReduction === CostReduction.SUM) {
            return this.math.add(totalCost, currCost);
        }
        return totalCost;
    };
    Session.prototype.updateCostForBatch = function (totalCost, costReduction) {
        if (costReduction === CostReduction.MEAN) {
            return this.math.divide(totalCost, this.batchSizeScalar);
        }
        return totalCost;
    };
    Session.prototype.getOrCreateRuntime = function (tensors, feed) {
        var key = this.makeRuntimeCacheKey(tensors, feed);
        var runtime = this.runtimeCache[key];
        if (runtime === undefined) {
            var nodes = session_util.getOrderedEvaluationSetFromEvalTensor(tensors, feed);
            nodes = session_util.addSplitNodes(nodes);
            session_util.removeFeedDictionaryNodesFromEvaluationSet(feed, nodes);
            session_util.throwErrorIfEvaluationSetContainsPlaceholderNodes(nodes);
            var operations = operation_emitter.emitFromGraphNodes(nodes);
            runtime = { nodes: nodes, operations: operations };
            this.runtimeCache[key] = runtime;
        }
        return runtime;
    };
    Session.prototype.makeRuntimeCacheKey = function (tensors, feed) {
        return tensors.map(function (x) { return x.id; }).sort().join('_') + '__' +
            Object.keys(feed.dict).sort().join('_');
    };
    return Session;
}());
exports.Session = Session;

},{"./math/ndarray":22,"./operation_emitter":59,"./session_util":83,"./tensor_array_map":85,"./util":86}],83:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var graph_1 = require("./graph");
var graph_util = require("./graph_util");
var ndarray_1 = require("./math/ndarray");
var util = require("./util");
function getTerminatingNodesFromFeedDictionary(feedDictionary) {
    return Object.keys(feedDictionary.dict)
        .map(function (tensorID) { return feedDictionary.dict[+tensorID].tensor.node; });
}
exports.getTerminatingNodesFromFeedDictionary = getTerminatingNodesFromFeedDictionary;
function getOrderedEvaluationSetFromEvalTensor(evalTensors, feedDictionary) {
    var terminatingNodes = getTerminatingNodesFromFeedDictionary(feedDictionary);
    var evalNodes = evalTensors.map(function (x) { return x.node; });
    var unorderedEvaluationSet = graph_util.getUnorderedEvaluationSet(evalNodes, terminatingNodes);
    var orderedEvaluationSet = graph_util.getOrderedEvaluationSet(unorderedEvaluationSet);
    return orderedEvaluationSet;
}
exports.getOrderedEvaluationSetFromEvalTensor = getOrderedEvaluationSetFromEvalTensor;
function addPersistentArraysToTensorArrayMap(evaluationSet, tensorArrayMap) {
    evaluationSet.forEach(function (node) {
        if (node instanceof graph_1.VariableNode || node instanceof graph_1.ConstantNode) {
            tensorArrayMap.set(node.output, node.data);
        }
    });
}
exports.addPersistentArraysToTensorArrayMap = addPersistentArraysToTensorArrayMap;
function getVariableNodesFromEvaluationSet(evaluationSet) {
    var nodes = [];
    evaluationSet.forEach(function (node) {
        if (node instanceof graph_1.VariableNode) {
            nodes.push(node);
        }
    });
    return nodes;
}
exports.getVariableNodesFromEvaluationSet = getVariableNodesFromEvaluationSet;
function throwIfFeedDictionaryContainsNDArrays(feedDictionary) {
    Object.keys(feedDictionary.dict).forEach(function (tensorID) {
        if (feedDictionary.dict[+tensorID].data instanceof ndarray_1.NDArray) {
            throw new Error('training requires FeedDictionary entries to be InputProviders' +
                'and not NDArrays.');
        }
    });
}
exports.throwIfFeedDictionaryContainsNDArrays = throwIfFeedDictionaryContainsNDArrays;
function loadInputsFromFeedDictionaryToTensorArrayMap(batchFeed, activations, math) {
    Object.keys(batchFeed.dict).forEach(function (tensorID) {
        var feedEntry = batchFeed.dict[+tensorID];
        var data;
        if (feedEntry.data instanceof ndarray_1.NDArray) {
            data = feedEntry.data;
        }
        else {
            var provider = feedEntry.data;
            data = provider.getNextCopy(math);
        }
        util.assert(util.arraysEqual(feedEntry.tensor.shape, data.shape), "Error loading FeedEntry: feeding NDArray of shape " + data.shape + " " +
            ("does not match Tensor (id: " + feedEntry.tensor.id + ") shape: ") +
            (feedEntry.tensor.shape + "."));
        activations.set(feedEntry.tensor, data);
    });
}
exports.loadInputsFromFeedDictionaryToTensorArrayMap = loadInputsFromFeedDictionaryToTensorArrayMap;
function releaseFeedDictionaryInputsFromTensorArrayMap(batchFeed, activations, math) {
    Object.keys(batchFeed.dict).forEach(function (tensorID) {
        var feedEntry = batchFeed.dict[+tensorID];
        if (!(feedEntry.data instanceof ndarray_1.NDArray)) {
            var provider = feedEntry.data;
            var feedEntryArray = activations.get(feedEntry.tensor);
            provider.disposeCopy(math, feedEntryArray);
        }
        activations.delete(feedEntry.tensor);
    });
}
exports.releaseFeedDictionaryInputsFromTensorArrayMap = releaseFeedDictionaryInputsFromTensorArrayMap;
function removeFeedDictionaryNodesFromEvaluationSet(feedDictionary, evaluationSet) {
    var i = 0;
    while (i < evaluationSet.length) {
        var node = evaluationSet[i];
        if (feedDictionary.dict[node.output.id] != null) {
            evaluationSet.splice(i, 1);
        }
        else {
            ++i;
        }
    }
}
exports.removeFeedDictionaryNodesFromEvaluationSet = removeFeedDictionaryNodesFromEvaluationSet;
function disposeAndInitializeOperationOutputs(evaluationSet, tensorArrayMap) {
    evaluationSet.forEach(function (node) {
        if (!graph_util.isInputNode(node)) {
            if (!graph_util.isPassthroughNode(node, tensorArrayMap)) {
                tensorArrayMap.disposeArray(node.output);
            }
            tensorArrayMap.set(node.output, null);
        }
    });
}
exports.disposeAndInitializeOperationOutputs = disposeAndInitializeOperationOutputs;
function disposeAndInitializeOperationInputGradients(evaluationSet, gradients) {
    evaluationSet.forEach(function (node) {
        Object.keys(node.inputs).forEach(function (inputName) {
            var input = node.inputs[inputName];
            if (gradients.get(input, true) !== gradients.get(node.output, true)) {
                gradients.disposeArray(input);
            }
            gradients.set(input, null);
        });
    });
}
exports.disposeAndInitializeOperationInputGradients = disposeAndInitializeOperationInputGradients;
function disposeTransientOperationArrays(operations, activations, gradients) {
    operations.forEach(function (op) { return op.disposeTransientArrays(activations, gradients); });
}
exports.disposeTransientOperationArrays = disposeTransientOperationArrays;
function throwErrorIfEvaluationSetContainsPlaceholderNodes(evaluationSet) {
    evaluationSet.forEach(function (node) {
        if (node instanceof graph_1.PlaceholderNode) {
            var shape = '[' + node.output.shape.join(', ') + ']';
            throw new Error('Placeholder node "' + node.name + '" ' + shape +
                ' not present in feed dictionary.');
        }
    });
}
exports.throwErrorIfEvaluationSetContainsPlaceholderNodes = throwErrorIfEvaluationSetContainsPlaceholderNodes;
function addSplitNodes(nodes) {
    var nodeIdToNumConsumers = [];
    var nodeIdToSplitNode = {};
    nodes.forEach(function (node) {
        var keys = Object.keys(node.inputs);
        keys.forEach(function (key) {
            var inputTensor = node.inputs[key];
            var input = inputTensor.node;
            if (nodeIdToNumConsumers[input.id] == null) {
                nodeIdToNumConsumers[input.id] = 0;
            }
            nodeIdToNumConsumers[input.id]++;
            if (nodeIdToNumConsumers[input.id] > 1 &&
                nodeIdToSplitNode[input.id] == null) {
                nodeIdToSplitNode[input.id] = new graph_1.SplitNode(input.graph, inputTensor);
            }
        });
    });
    var newNodes = [];
    nodes.forEach(function (node) {
        newNodes.push(node);
        if (node.id in nodeIdToSplitNode) {
            var splitNode = nodeIdToSplitNode[node.id];
            newNodes.push(splitNode);
        }
        var keys = Object.keys(node.inputs);
        keys.forEach(function (key) {
            var inputTensor = node.inputs[key];
            var inputId = inputTensor.node.id;
            if (inputId in nodeIdToSplitNode) {
                node.inputs[key] = nodeIdToSplitNode[inputId].getNewOutputTensor();
            }
        });
    });
    return newNodes;
}
exports.addSplitNodes = addSplitNodes;

},{"./graph":7,"./graph_util":10,"./math/ndarray":22,"./util":86}],84:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("./math/ndarray");
var optimizer_1 = require("./optimizer");
var session_util = require("./session_util");
var tensor_array_map_1 = require("./tensor_array_map");
var SGDOptimizer = (function (_super) {
    __extends(SGDOptimizer, _super);
    function SGDOptimizer(learningRate, specifiedVariableList) {
        var _this = _super.call(this, specifiedVariableList) || this;
        _this.learningRate = learningRate;
        _this.variableGradients = new tensor_array_map_1.TensorArrayMap();
        _this.one = ndarray_1.Scalar.new(1);
        return _this;
    }
    SGDOptimizer.prototype.beforeBatch = function (math, batchSize, runtime, activationArrayMap, gradientArrayMap) {
        var _this = this;
        this.variableNodes = this.specifiedVariableNodes == null ?
            session_util.getVariableNodesFromEvaluationSet(runtime.nodes) :
            this.specifiedVariableNodes;
        if (batchSize !== this.prevBatchSize) {
            this.prevBatchSize = batchSize;
            this.c = ndarray_1.Scalar.new(-this.learningRate / batchSize);
        }
        this.variableNodes.forEach(function (node) { return _this.variableGradients.set(node.output, ndarray_1.NDArray.zeros(node.output.shape)); });
    };
    SGDOptimizer.prototype.afterExample = function (math, runtime, activationArrayMap, gradientArrayMap) {
        var _this = this;
        math.scope(function (keep) {
            _this.variableNodes.forEach(function (node) {
                var gradient = gradientArrayMap.get(node.output);
                var accumulatedGradient = _this.variableGradients.get(node.output);
                _this.variableGradients.set(node.output, keep(math.add(gradient, accumulatedGradient)));
                accumulatedGradient.dispose();
            });
        });
    };
    SGDOptimizer.prototype.afterBatch = function (math, batchSize, runtime, activationArrayMap, gradientArrayMap) {
        var _this = this;
        math.scope(function (keep) {
            _this.variableNodes.forEach(function (node) {
                var oldVariable = activationArrayMap.get(node.output);
                var gradient = _this.variableGradients.get(node.output);
                var variable = math.scaledArrayAdd(_this.c, gradient, _this.one, oldVariable);
                activationArrayMap.set(node.output, keep(variable));
                node.data = variable;
                oldVariable.dispose();
            });
        });
        this.variableGradients.dispose();
        this.variableGradients = new tensor_array_map_1.TensorArrayMap();
    };
    SGDOptimizer.prototype.dispose = function () {
        if (this.c != null) {
            this.c.dispose();
        }
        this.one.dispose();
    };
    SGDOptimizer.prototype.setLearningRate = function (learningRate) {
        this.learningRate = learningRate;
    };
    return SGDOptimizer;
}(optimizer_1.Optimizer));
exports.SGDOptimizer = SGDOptimizer;

},{"./math/ndarray":22,"./optimizer":80,"./session_util":83,"./tensor_array_map":85}],85:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var TensorArrayMap = (function () {
    function TensorArrayMap() {
        this.dict = {};
    }
    TensorArrayMap.prototype.set = function (tensor, array) {
        this.dict[tensor.id] = array;
    };
    TensorArrayMap.prototype.get = function (tensor, skipChecks) {
        if (skipChecks === void 0) { skipChecks = false; }
        if (!skipChecks && this.dict[tensor.id] === undefined) {
            throw new Error('tensor ' + tensor.id + ' not in array map.');
        }
        var nda = this.dict[tensor.id];
        if (!skipChecks && nda === null) {
            throw new Error('tensor ' + tensor.id + ' has null array.');
        }
        return nda;
    };
    TensorArrayMap.prototype.delete = function (tensor) {
        delete this.dict[tensor.id];
    };
    TensorArrayMap.prototype.disposeArray = function (tensor) {
        if (this.dict[tensor.id] === undefined) {
            return;
        }
        var nda = this.dict[tensor.id];
        if (nda === null) {
            return;
        }
        nda.dispose();
        this.dict[tensor.id] = null;
    };
    TensorArrayMap.prototype.size = function () {
        return Object.keys(this.dict).length;
    };
    TensorArrayMap.prototype.dispose = function () {
        var _this = this;
        Object.keys(this.dict).forEach(function (tensorID) {
            var nda = _this.dict[+tensorID];
            if (nda) {
                nda.dispose();
            }
        });
        this.dict = {};
    };
    TensorArrayMap.prototype.hasNullArray = function (tensor) {
        if (this.dict[tensor.id] === undefined) {
            throw new Error('tensor ' + tensor.id + ' not in array map.');
        }
        return this.dict[tensor.id] === null;
    };
    return TensorArrayMap;
}());
exports.TensorArrayMap = TensorArrayMap;

},{}],86:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function shuffle(array) {
    var counter = array.length;
    var temp = 0;
    var index = 0;
    while (counter > 0) {
        index = (Math.random() * counter) | 0;
        counter--;
        temp = array[counter];
        array[counter] = array[index];
        array[index] = temp;
    }
}
exports.shuffle = shuffle;
function clamp(min, x, max) {
    return Math.max(min, Math.min(x, max));
}
exports.clamp = clamp;
function randUniform(a, b) {
    return Math.random() * (b - a) + a;
}
exports.randUniform = randUniform;
function randGauss(mean, stdDev, truncated) {
    if (mean === void 0) { mean = 0; }
    if (stdDev === void 0) { stdDev = 1; }
    if (truncated === void 0) { truncated = false; }
    var v1, v2, s;
    do {
        v1 = 2 * Math.random() - 1;
        v2 = 2 * Math.random() - 1;
        s = v1 * v1 + v2 * v2;
    } while (s > 1);
    var result = Math.sqrt(-2 * Math.log(s) / s) * v1;
    if (truncated && result > 2) {
        return randGauss(mean, stdDev, true);
    }
    return mean + stdDev * result;
}
exports.randGauss = randGauss;
function distSquared(a, b) {
    var result = 0;
    for (var i = 0; i < a.length; i++) {
        var diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}
exports.distSquared = distSquared;
function assert(expr, msg) {
    if (!expr) {
        throw new Error(msg);
    }
}
exports.assert = assert;
function assertShapesMatch(shapeA, shapeB, errorMessagePrefix) {
    if (errorMessagePrefix === void 0) { errorMessagePrefix = ''; }
    assert(arraysEqual(shapeA, shapeB), errorMessagePrefix + ("Shapes " + shapeA + " and " + shapeB + " must match"));
}
exports.assertShapesMatch = assertShapesMatch;
function flatten(arr, ret) {
    ret = (ret === undefined ? [] : ret);
    for (var i = 0; i < arr.length; ++i) {
        if (Array.isArray(arr[i])) {
            flatten(arr[i], ret);
        }
        else {
            ret.push(arr[i]);
        }
    }
    return ret;
}
exports.flatten = flatten;
function inferShape(arr) {
    var shape = [];
    while (arr instanceof Array) {
        shape.push(arr.length);
        arr = arr[0];
    }
    return shape;
}
exports.inferShape = inferShape;
function sizeFromShape(shape) {
    if (shape.length === 0) {
        return 1;
    }
    var size = shape[0];
    for (var i = 1; i < shape.length; i++) {
        size *= shape[i];
    }
    return size;
}
exports.sizeFromShape = sizeFromShape;
function isScalarShape(shape) {
    return shape.length === 0;
}
exports.isScalarShape = isScalarShape;
function arraysEqual(n1, n2) {
    if (n1.length !== n2.length) {
        return false;
    }
    for (var i = 0; i < n1.length; i++) {
        if (n1[i] !== n2[i]) {
            return false;
        }
    }
    return true;
}
exports.arraysEqual = arraysEqual;
function isInt(a) {
    return a % 1 === 0;
}
exports.isInt = isInt;
function tanh(x) {
    if (Math.tanh != null) {
        return Math.tanh(x);
    }
    if (x === Infinity) {
        return 1;
    }
    else if (x === -Infinity) {
        return -1;
    }
    else {
        var e2x = Math.exp(2 * x);
        return (e2x - 1) / (e2x + 1);
    }
}
exports.tanh = tanh;
function sizeToSquarishShape(size) {
    for (var a = Math.floor(Math.sqrt(size)); a > 1; --a) {
        if (size % a === 0) {
            return [a, size / a];
        }
    }
    return [1, size];
}
exports.sizeToSquarishShape = sizeToSquarishShape;
function createShuffledIndices(n) {
    var shuffledIndices = new Uint32Array(n);
    for (var i = 0; i < n; ++i) {
        shuffledIndices[i] = i;
    }
    shuffle(shuffledIndices);
    return shuffledIndices;
}
exports.createShuffledIndices = createShuffledIndices;

},{}]},{},[1]);
