(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var BenchmarkRun = (function () {
    function BenchmarkRun(name, benchmarkTest) {
        this.name = name;
        this.benchmarkTest = benchmarkTest;
        this.chartData = [];
    }
    return BenchmarkRun;
}());
exports.BenchmarkRun = BenchmarkRun;

},{}],2:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../../src/math/conv_util");
var conv_gpu = require("../../src/math/webgl/conv_gpu");
var gpgpu_context_1 = require("../../src/math/webgl/gpgpu_context");
var test_util = require("../../src/test_util");
var OP_RUNS = 100;
exports.BENCHMARK_TEST = function (size) {
    var inputShapeRCD = [size, size, 1];
    var outputDepth = 1;
    var fieldSize = 11;
    var stride = 1;
    var zeroPad = conv_util.computeDefaultPad(inputShapeRCD, fieldSize, stride);
    var outputShapeRCD = conv_util.computeOutputShape3D(inputShapeRCD, fieldSize, outputDepth, stride, zeroPad);
    var inputTexShapeRC = conv_util.computeTexShapeFrom3D(inputShapeRCD);
    var outputTexShapeRC = conv_util.computeTexShapeFrom3D(outputShapeRCD);
    var weightsTexShapeRC = conv_util.computeWeightsTexShape(inputShapeRCD[2], outputDepth, fieldSize);
    var biasesTexShapeRC = conv_util.computeBiasesTexShape(outputDepth);
    var hasBias = true;
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var program = gpgpu.createProgram(conv_gpu.getFragmentShaderSource(inputShapeRCD, outputDepth, fieldSize, stride, zeroPad, hasBias));
    var inputTexture = gpgpu.createMatrixTexture(inputTexShapeRC[0], inputTexShapeRC[1]);
    var weightsTexture = gpgpu.createMatrixTexture(weightsTexShapeRC[0], weightsTexShapeRC[1]);
    var biasesTexture = gpgpu.createMatrixTexture(biasesTexShapeRC[0], biasesTexShapeRC[1]);
    var outputTexture = gpgpu.createMatrixTexture(outputTexShapeRC[0], outputTexShapeRC[1]);
    var inputData = test_util.randomArrayInRange(inputTexShapeRC[0] * inputTexShapeRC[1], -1, 1);
    var weightsData = test_util.randomArrayInRange(weightsTexShapeRC[0] * weightsTexShapeRC[1], -1, 1);
    var biasesData = test_util.randomArrayInRange(biasesTexShapeRC[0] * biasesTexShapeRC[1], -1, 1);
    gpgpu.uploadMatrixToTexture(inputTexture, inputTexShapeRC[0], inputTexShapeRC[1], inputData);
    gpgpu.uploadMatrixToTexture(weightsTexture, weightsTexShapeRC[0], weightsTexShapeRC[1], weightsData);
    gpgpu.uploadMatrixToTexture(biasesTexture, biasesTexShapeRC[0], biasesTexShapeRC[1], biasesData);
    var start = performance.now();
    for (var i = 0; i < OP_RUNS; i++) {
        conv_gpu.convolve(gpgpu, program, inputTexture, weightsTexture, biasesTexture, outputTexture, outputTexShapeRC);
    }
    gpgpu.downloadMatrixFromTexture(outputTexture, outputTexShapeRC[0], outputTexShapeRC[1]);
    var end = performance.now();
    var avgTime = (end - start) / OP_RUNS;
    gpgpu.deleteMatrixTexture(inputTexture);
    gpgpu.deleteMatrixTexture(weightsTexture);
    gpgpu.deleteMatrixTexture(biasesTexture);
    gpgpu.deleteMatrixTexture(outputTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return avgTime;
};

},{"../../src/math/conv_util":17,"../../src/math/webgl/conv_gpu":23,"../../src/math/webgl/gpgpu_context":24,"../../src/test_util":35}],3:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../../src/math/conv_util");
var conv_backprop_gpu = require("../../src/math/webgl/conv_backprop_gpu");
var gpgpu_context_1 = require("../../src/math/webgl/gpgpu_context");
var test_util = require("../../src/test_util");
var OP_RUNS = 100;
exports.BENCHMARK_TEST = function (size) {
    var xShapeRCD = [size, size, 1];
    var origOutputDepth = 2;
    var fieldSize = 11;
    var origStride = 1;
    var origPad = 1;
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);
    var origInputDepth = xShapeRCD[2];
    var src = conv_backprop_gpu.getFragmentShaderConvTransposeSource(xShapeRCD, fieldSize, origInputDepth, origStride, origPad, false);
    var program = gpgpu.createProgram(src);
    var xTexShapeRC = conv_util.computeTexShapeFrom3D(xShapeRCD);
    var xTex = gpgpu.createMatrixTexture(xTexShapeRC[0], xTexShapeRC[1]);
    var xData = test_util.randomArrayInRange(xTexShapeRC[0] * xTexShapeRC[1], -1, 1);
    gpgpu.uploadMatrixToTexture(xTex, xTexShapeRC[0], xTexShapeRC[1], xData);
    var wTexShapeRC = conv_util.computeWeightsTexShape(origInputDepth, origOutputDepth, fieldSize);
    var wData = test_util.randomArrayInRange(wTexShapeRC[0] * wTexShapeRC[1], -1, 1);
    var wTex = gpgpu.createMatrixTexture(wTexShapeRC[0], wTexShapeRC[1]);
    gpgpu.uploadMatrixToTexture(wTex, wTexShapeRC[0], wTexShapeRC[1], wData);
    var dilatedRC = conv_util.computeDilatedRC([xShapeRCD[0], xShapeRCD[1]], origStride);
    var pad = fieldSize - 1 - origPad;
    var resultShapeRCD = conv_util.computeOutputShape3D([dilatedRC[0], dilatedRC[1], origOutputDepth], fieldSize, origInputDepth, 1, pad);
    var resultTexRC = conv_util.computeTexShapeFrom3D(resultShapeRCD);
    var resultTex = gpgpu.createMatrixTexture(resultTexRC[0], resultTexRC[1]);
    var start = performance.now();
    for (var i = 0; i < OP_RUNS; i++) {
        conv_backprop_gpu.convTranspose(gpgpu, program, xTex, wTex, null, resultTex, resultTexRC);
    }
    var y = gpgpu.downloadMatrixFromTexture(resultTex, resultTexRC[0], resultTexRC[1]);
    var end = performance.now();
    var avgTime = (end - start) / OP_RUNS;
    gpgpu.deleteMatrixTexture(resultTex);
    gpgpu.deleteMatrixTexture(xTex);
    gpgpu.deleteMatrixTexture(wTex);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return avgTime;
};

},{"../../src/math/conv_util":17,"../../src/math/webgl/conv_backprop_gpu":22,"../../src/math/webgl/gpgpu_context":24,"../../src/test_util":35}],4:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var math_cpu_1 = require("../../src/math/math_cpu");
var ndarray_1 = require("../../src/math/ndarray");
var OPS_PER_RUN = 10;
exports.BENCHMARK_TEST = function (size) {
    var math = new math_cpu_1.NDArrayMathCPU();
    var a = ndarray_1.NDArray.randUniform([size, size], -1, 1);
    var start = performance.now();
    for (var i = 0; i < OPS_PER_RUN; i++) {
        math.logSumExp(a);
    }
    var end = performance.now();
    return (end - start) / OPS_PER_RUN;
};

},{"../../src/math/math_cpu":20,"../../src/math/ndarray":21}],5:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var gpgpu_context_1 = require("../../src/math/webgl/gpgpu_context");
var logsumexp_gpu = require("../../src/math/webgl/logsumexp_gpu");
var test_util = require("../../src/test_util");
var OP_RUNS = 100;
exports.BENCHMARK_TEST = function (size) {
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var program = gpgpu.createProgram(logsumexp_gpu.getFragmentShaderSource(size, size));
    var aTexture = gpgpu.createMatrixTexture(size, size);
    var resultTexture = gpgpu.createMatrixTexture(size, size);
    var a = test_util.randomArrayInRange(size * size, -1, 1);
    gpgpu.uploadMatrixToTexture(aTexture, size, size, a);
    var start = performance.now();
    for (var i = 0; i < OP_RUNS; i++) {
        logsumexp_gpu.logSumExp(gpgpu, program, aTexture, size, size, resultTexture);
    }
    gpgpu.downloadMatrixFromTexture(resultTexture, size, size);
    var avgTime = (performance.now() - start) / OP_RUNS;
    gpgpu.deleteMatrixTexture(aTexture);
    gpgpu.deleteMatrixTexture(resultTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return avgTime;
};

},{"../../src/math/webgl/gpgpu_context":24,"../../src/math/webgl/logsumexp_gpu":26,"../../src/test_util":35}],6:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var benchmark_1 = require("./benchmark");
var conv_gpu_benchmark = require("./conv_gpu_benchmark");
var conv_transpose_gpu_benchmark = require("./conv_transpose_gpu_benchmark");
var logsumexp_cpu_benchmark = require("./logsumexp_cpu_benchmark");
var logsumexp_gpu_benchmark = require("./logsumexp_gpu_benchmark");
var max_pool_backprop_gpu_benchmark = require("./max_pool_backprop_gpu_benchmark");
var max_pool_gpu_benchmark = require("./max_pool_gpu_benchmark");
var mulmat_cpu_benchmark = require("./mulmat_cpu_benchmark");
var mulmat_gpu_benchmark = require("./mulmat_gpu_benchmark");
var tex_util_benchmark = require("./tex_util_benchmark");
exports.BENCHMARK_RUN_GROUPS = [
    {
        name: 'Texture encoding / decoding (unpacked vs packed)',
        min: 0,
        max: 1024,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        benchmarkRuns: [
            new benchmark_1.BenchmarkRun('encode_unpacked', tex_util_benchmark.BENCHMARK_ENCODE_UNPACKED),
            new benchmark_1.BenchmarkRun('encode_packed', tex_util_benchmark.BENCHMARK_ENCODE_PACKED),
            new benchmark_1.BenchmarkRun('decode_unpacked', tex_util_benchmark.BENCHMARK_DECODE_UNPACKED),
            new benchmark_1.BenchmarkRun('decode_packed', tex_util_benchmark.BENCHMARK_DECODE_PACKED)
        ]
    },
    {
        name: 'Matrix Multiplication (CPU vs GPU)',
        min: 0,
        max: 1024,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        benchmarkRuns: [
            new benchmark_1.BenchmarkRun('mulmat_gpu', mulmat_gpu_benchmark.BENCHMARK_TEST),
            new benchmark_1.BenchmarkRun('mulmat_packed_gpu', mulmat_gpu_benchmark.BENCHMARK_TEST_PACKED),
            new benchmark_1.BenchmarkRun('mulmat_cpu', mulmat_cpu_benchmark.BENCHMARK_TEST)
        ],
    },
    {
        name: 'LogSumExp (CPU vs GPU)',
        min: 0,
        max: 1024,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        benchmarkRuns: [
            new benchmark_1.BenchmarkRun('logsumexp_gpu', logsumexp_gpu_benchmark.BENCHMARK_TEST),
            new benchmark_1.BenchmarkRun('logsumexp_cpu', logsumexp_cpu_benchmark.BENCHMARK_TEST)
        ],
    },
    {
        name: 'Convolution (GPU)',
        min: 0,
        max: 1024,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        benchmarkRuns: [new benchmark_1.BenchmarkRun('d1=1, d2=1, f=11, s=1', conv_gpu_benchmark.BENCHMARK_TEST)],
    },
    {
        name: 'Convolution Transposed (GPU)',
        min: 0,
        max: 1024,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        benchmarkRuns: [new benchmark_1.BenchmarkRun('d1=1, d2=1, f=11, s=1', conv_transpose_gpu_benchmark.BENCHMARK_TEST)],
    },
    {
        name: 'Max pool (GPU)',
        min: 0,
        max: 1024,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        benchmarkRuns: [new benchmark_1.BenchmarkRun('d1=1, d2=1, f=11, s=1', max_pool_gpu_benchmark.MAX_POOL_BENCHMARK_TEST)],
    },
    {
        name: 'Max pool positions (GPU)',
        min: 0,
        max: 1024,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        benchmarkRuns: [new benchmark_1.BenchmarkRun('d1=1, d2=1, f=11, s=1', max_pool_gpu_benchmark.MAX_POOL_POSNS_BENCHMARK_TEST)],
    },
    {
        name: 'Max pool backprop (GPU)',
        min: 0,
        max: 1024,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        benchmarkRuns: [new benchmark_1.BenchmarkRun('d1=1, d2=1, f=11, s=1', max_pool_backprop_gpu_benchmark.BENCHMARK_TEST)],
    }
];

},{"./benchmark":1,"./conv_gpu_benchmark":2,"./conv_transpose_gpu_benchmark":3,"./logsumexp_cpu_benchmark":4,"./logsumexp_gpu_benchmark":5,"./max_pool_backprop_gpu_benchmark":8,"./max_pool_gpu_benchmark":9,"./mulmat_cpu_benchmark":10,"./mulmat_gpu_benchmark":11,"./tex_util_benchmark":12}],7:[function(require,module,exports){
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
require("../demo-header");
require("../demo-footer");
var polymer_spec_1 = require("../polymer-spec");
var math_benchmark_run_groups_1 = require("./math-benchmark-run-groups");
exports.MathBenchmarkPolymer = polymer_spec_1.PolymerElement({ is: 'math-benchmark', properties: { benchmarkRunGroupNames: Array } });
var MathBenchmark = (function (_super) {
    __extends(MathBenchmark, _super);
    function MathBenchmark() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    MathBenchmark.prototype.ready = function () {
        var _this = this;
        var benchmarkRunGroupNames = [];
        this.stopMessages = [];
        for (var i = 0; i < math_benchmark_run_groups_1.BENCHMARK_RUN_GROUPS.length; i++) {
            benchmarkRunGroupNames.push(math_benchmark_run_groups_1.BENCHMARK_RUN_GROUPS[i].name);
            this.stopMessages.push(false);
        }
        this.benchmarkRunGroupNames = benchmarkRunGroupNames;
        setTimeout(function () {
            var runButtons = _this.querySelectorAll('.run-test');
            var stopButtons = _this.querySelectorAll('.run-stop');
            var _loop_1 = function (i) {
                runButtons[i].addEventListener('click', function () {
                    _this.runBenchmarkGroup(i);
                });
                stopButtons[i].addEventListener('click', function () {
                    _this.stopMessages[i] = true;
                });
            };
            for (var i = 0; i < runButtons.length; i++) {
                _loop_1(i);
            }
        }, 0);
    };
    MathBenchmark.prototype.runBenchmarkGroup = function (benchmarkRunGroupIndex) {
        var benchmarkRunGroup = math_benchmark_run_groups_1.BENCHMARK_RUN_GROUPS[benchmarkRunGroupIndex];
        var canvas = this.querySelectorAll('.run-plot')[benchmarkRunGroupIndex];
        var context = canvas.getContext('2d');
        var datasets = [];
        for (var i = 0; i < benchmarkRunGroup.benchmarkRuns.length; i++) {
            var hue = Math.floor(360 * i / benchmarkRunGroup.benchmarkRuns.length);
            datasets.push({
                data: benchmarkRunGroup.benchmarkRuns[i].chartData,
                fill: false,
                label: benchmarkRunGroup.benchmarkRuns[i].name,
                borderColor: 'hsl(' + hue + ', 100%, 40%)',
                backgroundColor: 'hsl(' + hue + ', 100%, 70%)',
                pointRadius: 0,
                pointHitRadius: 5,
                borderWidth: 1,
                lineTension: 0
            });
        }
        var chart = new Chart(context, {
            type: 'line',
            data: { datasets: datasets },
            options: {
                animation: { duration: 0 },
                responsive: false,
                scales: {
                    xAxes: [{
                            type: 'linear',
                            position: 'bottom',
                            ticks: {
                                min: benchmarkRunGroup.min,
                                max: benchmarkRunGroup.max,
                                stepSize: benchmarkRunGroup.stepSize,
                                callback: function (label) {
                                    return benchmarkRunGroup.stepToSizeTransformation != null ?
                                        benchmarkRunGroup.stepToSizeTransformation(+label) :
                                        +label;
                                }
                            }
                        }],
                    yAxes: [{
                            ticks: {
                                callback: function (label, index, labels) {
                                    return label + 'ms';
                                }
                            },
                        }]
                },
                tooltips: { mode: 'label' },
                title: { text: benchmarkRunGroup.name }
            }
        });
        canvas.style.display = 'none';
        var runMessage = this.querySelectorAll('.run-message')[benchmarkRunGroupIndex];
        runMessage.style.display = 'block';
        var runNumbersTable = this.querySelectorAll('.run-numbers-table')[benchmarkRunGroupIndex];
        runNumbersTable.innerHTML = '';
        runNumbersTable.style.display = 'none';
        var headers = ['size'];
        for (var i = 0; i < benchmarkRunGroup.benchmarkRuns.length; i++) {
            headers.push(benchmarkRunGroup.benchmarkRuns[i].name);
        }
        runNumbersTable.appendChild(this.buildRunNumbersRow(headers));
        this.runBenchmarkSteps(chart, benchmarkRunGroup, benchmarkRunGroupIndex, benchmarkRunGroup.min);
    };
    MathBenchmark.prototype.buildRunNumbersRow = function (values) {
        var runNumberRowElement = document.createElement('div');
        runNumberRowElement.className = 'run-numbers-row math-benchmark';
        for (var i = 0; i < values.length; i++) {
            var runNumberCellElement = document.createElement('div');
            runNumberCellElement.className = 'run-numbers-cell math-benchmark';
            runNumberCellElement.innerText = values[i];
            runNumberRowElement.appendChild(runNumberCellElement);
        }
        return runNumberRowElement;
    };
    MathBenchmark.prototype.runBenchmarkSteps = function (chart, benchmarkRunGroup, benchmarkRunGroupIndex, step) {
        var _this = this;
        var runNumbersTable = this.querySelectorAll('.run-numbers-table')[benchmarkRunGroupIndex];
        if (step > benchmarkRunGroup.max ||
            this.stopMessages[benchmarkRunGroupIndex]) {
            this.stopMessages[benchmarkRunGroupIndex] = false;
            runNumbersTable.style.display = '';
            var canvas = this.querySelectorAll('.run-plot')[benchmarkRunGroupIndex];
            canvas.style.display = 'block';
            chart.update();
            var runMessage = this.querySelectorAll('.run-message')[benchmarkRunGroupIndex];
            runMessage.style.display = 'none';
            return;
        }
        var runNumberRowElement = document.createElement('div');
        runNumberRowElement.className = 'run-numbers-row math-benchmark';
        var rowValues = ['' + step];
        for (var i = 0; i < benchmarkRunGroup.benchmarkRuns.length; i++) {
            var benchmarkRun = benchmarkRunGroup.benchmarkRuns[i];
            var benchmarkTest = benchmarkRun.benchmarkTest;
            var size = benchmarkRunGroup.stepToSizeTransformation != null ?
                benchmarkRunGroup.stepToSizeTransformation(step) :
                step;
            var resultString = void 0;
            var logString = void 0;
            var time = 0;
            var success = true;
            try {
                time = benchmarkTest(size);
                resultString = time.toFixed(3) + 'ms';
                logString = resultString;
            }
            catch (e) {
                success = false;
                resultString = 'Error';
                logString = e.message;
            }
            if (time >= 0) {
                if (success) {
                    benchmarkRun.chartData.push({ x: step, y: time });
                }
                rowValues.push(resultString);
            }
            console.log(benchmarkRun.name + '[' + step + ']: ' + logString);
        }
        runNumbersTable.appendChild(this.buildRunNumbersRow(rowValues));
        step += benchmarkRunGroup.stepSize;
        setTimeout(function () { return _this.runBenchmarkSteps(chart, benchmarkRunGroup, benchmarkRunGroupIndex, step); }, 100);
    };
    return MathBenchmark;
}(exports.MathBenchmarkPolymer));
exports.MathBenchmark = MathBenchmark;
document.registerElement(MathBenchmark.prototype.is, MathBenchmark);

},{"../demo-footer":13,"../demo-header":14,"../polymer-spec":15,"./math-benchmark-run-groups":6}],8:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../../src/math/conv_util");
var gpgpu_context_1 = require("../../src/math/webgl/gpgpu_context");
var max_pool_backprop_gpu = require("../../src/math/webgl/max_pool_backprop_gpu");
var test_util = require("../../src/test_util");
var util = require("../../src/util");
var OP_RUNS = 100;
exports.BENCHMARK_TEST = function (size) {
    var dyShapeRCD = [size, size, 1];
    var outputDepth = 1;
    var fieldSize = 11;
    var stride = 1;
    var zeroPad = conv_util.computeDefaultPad(dyShapeRCD, fieldSize, stride);
    var outputShapeRCD = conv_util.computeOutputShape3D(dyShapeRCD, fieldSize, outputDepth, stride, zeroPad);
    var dyTexShapeRC = conv_util.computeTexShapeFrom3D(dyShapeRCD);
    var outputTexShapeRC = conv_util.computeTexShapeFrom3D(outputShapeRCD);
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var program = gpgpu.createProgram(max_pool_backprop_gpu.getFragmentShaderMaxPoolBackprop(dyShapeRCD, fieldSize, stride, zeroPad));
    var dyTexture = gpgpu.createMatrixTexture(dyTexShapeRC[0], dyTexShapeRC[1]);
    var maxPositionsTexture = gpgpu.createMatrixTexture(dyTexShapeRC[0], dyTexShapeRC[1]);
    var outputTexture = gpgpu.createMatrixTexture(outputTexShapeRC[0], outputTexShapeRC[1]);
    var dyData = test_util.randomArrayInRange(dyTexShapeRC[0] * dyTexShapeRC[1], -1, 1);
    var maxPositionsData = new Float32Array(util.sizeFromShape(dyShapeRCD));
    for (var i = 0; i < maxPositionsData.length; i++) {
        maxPositionsData[i] = Math.floor(Math.random() * fieldSize * fieldSize);
    }
    gpgpu.uploadMatrixToTexture(dyTexture, dyTexShapeRC[0], dyTexShapeRC[1], dyData);
    gpgpu.uploadMatrixToTexture(maxPositionsTexture, dyTexShapeRC[0], dyTexShapeRC[1], maxPositionsData);
    var start = performance.now();
    for (var i = 0; i < OP_RUNS; i++) {
        max_pool_backprop_gpu.maxPoolBackprop(gpgpu, program, dyTexture, maxPositionsTexture, outputTexture, outputTexShapeRC);
    }
    gpgpu.downloadMatrixFromTexture(outputTexture, outputTexShapeRC[0], outputTexShapeRC[1]);
    var end = performance.now();
    var avgTime = (end - start) / OP_RUNS;
    gpgpu.deleteMatrixTexture(dyTexture);
    gpgpu.deleteMatrixTexture(maxPositionsTexture);
    gpgpu.deleteMatrixTexture(outputTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return avgTime;
};

},{"../../src/math/conv_util":17,"../../src/math/webgl/gpgpu_context":24,"../../src/math/webgl/max_pool_backprop_gpu":27,"../../src/test_util":35,"../../src/util":36}],9:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../../src/math/conv_util");
var gpgpu_context_1 = require("../../src/math/webgl/gpgpu_context");
var max_pool_gpu = require("../../src/math/webgl/max_pool_gpu");
var test_util = require("../../src/test_util");
var OP_RUNS = 100;
exports.MAX_POOL_BENCHMARK_TEST = function (size) {
    var inputShapeRCD = [size, size, 1];
    var outputDepth = 1;
    var fieldSize = 11;
    var stride = 1;
    var zeroPad = conv_util.computeDefaultPad(inputShapeRCD, fieldSize, stride);
    var outputShapeRCD = conv_util.computeOutputShape3D(inputShapeRCD, fieldSize, outputDepth, stride, zeroPad);
    var inputTexShapeRC = conv_util.computeTexShapeFrom3D(inputShapeRCD);
    var outputTexShapeRC = conv_util.computeTexShapeFrom3D(outputShapeRCD);
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var program = gpgpu.createProgram(max_pool_gpu.getFragmentShaderMaxPoolSource(inputShapeRCD, fieldSize, stride, zeroPad));
    var inputTexture = gpgpu.createMatrixTexture(inputTexShapeRC[0], inputTexShapeRC[1]);
    var outputTexture = gpgpu.createMatrixTexture(outputTexShapeRC[0], outputTexShapeRC[1]);
    var inputData = test_util.randomArrayInRange(inputTexShapeRC[0] * inputTexShapeRC[1], -1, 1);
    gpgpu.uploadMatrixToTexture(inputTexture, inputTexShapeRC[0], inputTexShapeRC[1], inputData);
    var start = performance.now();
    for (var i = 0; i < OP_RUNS; i++) {
        max_pool_gpu.maxPoolCommon(gpgpu, program, inputTexture, outputTexture, outputTexShapeRC);
    }
    gpgpu.downloadMatrixFromTexture(outputTexture, outputTexShapeRC[0], outputTexShapeRC[1]);
    var end = performance.now();
    var avgTime = (end - start) / OP_RUNS;
    gpgpu.deleteMatrixTexture(inputTexture);
    gpgpu.deleteMatrixTexture(outputTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return avgTime;
};
exports.MAX_POOL_POSNS_BENCHMARK_TEST = function (size) {
    var inputShapeRCD = [size, size, 1];
    var outputDepth = 1;
    var fieldSize = 11;
    var stride = 1;
    var zeroPad = conv_util.computeDefaultPad(inputShapeRCD, fieldSize, stride);
    var outputShapeRCD = conv_util.computeOutputShape3D(inputShapeRCD, fieldSize, outputDepth, stride, zeroPad);
    var inputTexShapeRC = conv_util.computeTexShapeFrom3D(inputShapeRCD);
    var outputTexShapeRC = conv_util.computeTexShapeFrom3D(outputShapeRCD);
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var program = gpgpu.createProgram(max_pool_gpu.getFragmentShaderMaxPoolPositionsSource(inputShapeRCD, fieldSize, stride, zeroPad));
    var inputTexture = gpgpu.createMatrixTexture(inputTexShapeRC[0], inputTexShapeRC[1]);
    var outputTexture = gpgpu.createMatrixTexture(outputTexShapeRC[0], outputTexShapeRC[1]);
    var inputData = test_util.randomArrayInRange(inputTexShapeRC[0] * inputTexShapeRC[1], -1, 1);
    gpgpu.uploadMatrixToTexture(inputTexture, inputTexShapeRC[0], inputTexShapeRC[1], inputData);
    var start = performance.now();
    for (var i = 0; i < OP_RUNS; i++) {
        max_pool_gpu.maxPoolCommon(gpgpu, program, inputTexture, outputTexture, outputTexShapeRC);
    }
    gpgpu.downloadMatrixFromTexture(outputTexture, outputTexShapeRC[0], outputTexShapeRC[1]);
    var end = performance.now();
    var avgTime = (end - start) / OP_RUNS;
    gpgpu.deleteMatrixTexture(inputTexture);
    gpgpu.deleteMatrixTexture(outputTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return avgTime;
};

},{"../../src/math/conv_util":17,"../../src/math/webgl/gpgpu_context":24,"../../src/math/webgl/max_pool_gpu":28,"../../src/test_util":35}],10:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var math_cpu_1 = require("../../src/math/math_cpu");
var ndarray_1 = require("../../src/math/ndarray");
var OPS_PER_SMALL_RUN = 10;
exports.BENCHMARK_TEST = function (size) {
    if (size > 512) {
        return -1;
    }
    var math = new math_cpu_1.NDArrayMathCPU();
    var a = ndarray_1.NDArray.randUniform([size, size], -1, 1);
    var b = ndarray_1.NDArray.randUniform([size, size], -1, 1);
    var runs = (size < 192) ? OPS_PER_SMALL_RUN : 1;
    var start = performance.now();
    for (var i = 0; i < runs; i++) {
        math.matMul(a, b);
    }
    var end = performance.now();
    return (end - start) / runs;
};

},{"../../src/math/math_cpu":20,"../../src/math/ndarray":21}],11:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var math_1 = require("../../src/math/math");
var ndarray_1 = require("../../src/math/ndarray");
var gpgpu_context_1 = require("../../src/math/webgl/gpgpu_context");
var mulmat_gpu = require("../../src/math/webgl/mulmat_gpu");
var mulmat_packed_gpu = require("../../src/math/webgl/mulmat_packed_gpu");
var test_util = require("../../src/test_util");
var OP_RUNS = 100;
exports.BENCHMARK_TEST = function (size) {
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var aTexture = gpgpu.createMatrixTexture(size, size);
    var bTexture = gpgpu.createMatrixTexture(size, size);
    var resultTexture = gpgpu.createMatrixTexture(size, size);
    var aArr = new ndarray_1.Array2D([size, size], { texture: aTexture, textureShapeRC: [size, size] });
    var bArr = new ndarray_1.Array2D([size, size], { texture: bTexture, textureShapeRC: [size, size] });
    var resArr = new ndarray_1.Array2D([size, size], { texture: resultTexture, textureShapeRC: [size, size] });
    var program = gpgpu.createProgram(mulmat_gpu.getFragmentShader(aArr, bArr, resArr, math_1.MatrixOrientation.REGULAR, math_1.MatrixOrientation.REGULAR));
    var a = test_util.randomArrayInRange(size * size, -1, 1);
    var b = test_util.randomArrayInRange(size * size, -1, 1);
    gpgpu.uploadMatrixToTexture(aTexture, size, size, a);
    gpgpu.uploadMatrixToTexture(bTexture, size, size, b);
    var start = performance.now();
    for (var i = 0; i < OP_RUNS; i++) {
        mulmat_gpu.multiplyMatrix(gpgpu, program, aTexture, bTexture, resultTexture, [size, size]);
    }
    var actual = gpgpu.downloadMatrixFromTexture(resultTexture, size, size);
    var avgTime = (performance.now() - start) / OP_RUNS;
    gpgpu.deleteMatrixTexture(aTexture);
    gpgpu.deleteMatrixTexture(bTexture);
    gpgpu.deleteMatrixTexture(resultTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    var expected = test_util.cpuMultiplyMatrix(a, size, size, b, size, size);
    test_util.expectArraysClose(actual, expected, 0.001);
    return avgTime;
};
exports.BENCHMARK_TEST_PACKED = function (size) {
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var program = gpgpu.createProgram(mulmat_packed_gpu.getFragmentShaderSource(size, math_1.MatrixOrientation.REGULAR, math_1.MatrixOrientation.REGULAR));
    var aTexture = gpgpu.createPackedMatrixTexture(size, size);
    var bTexture = gpgpu.createPackedMatrixTexture(size, size);
    var resultTexture = gpgpu.createPackedMatrixTexture(size, size);
    var a = test_util.randomArrayInRange(size * size, -1, 1);
    var b = test_util.randomArrayInRange(size * size, -1, 1);
    gpgpu.uploadMatrixToPackedTexture(aTexture, size, size, a);
    gpgpu.uploadMatrixToPackedTexture(bTexture, size, size, b);
    var start = performance.now();
    for (var i = 0; i < OP_RUNS; i++) {
        mulmat_packed_gpu.multiplyMatrixPacked(gpgpu, program, aTexture, bTexture, resultTexture, [size, size]);
    }
    var actual = gpgpu.downloadMatrixFromPackedTexture(resultTexture, size, size);
    var avgTime = (performance.now() - start) / OP_RUNS;
    gpgpu.deleteMatrixTexture(aTexture);
    gpgpu.deleteMatrixTexture(bTexture);
    gpgpu.deleteMatrixTexture(resultTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    var expected = test_util.cpuMultiplyMatrix(a, size, size, b, size, size);
    test_util.expectArraysClose(actual, expected, 0.001);
    return avgTime;
};

},{"../../src/math/math":19,"../../src/math/ndarray":21,"../../src/math/webgl/gpgpu_context":24,"../../src/math/webgl/mulmat_gpu":29,"../../src/math/webgl/mulmat_packed_gpu":30,"../../src/test_util":35}],12:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tex_util = require("../../src/math/webgl/tex_util");
var webgl_util = require("../../src/math/webgl/webgl_util");
var test_util = require("../../src/test_util");
var OPS_PER_RUN = 100;
exports.BENCHMARK_ENCODE_UNPACKED = function (size) {
    var matrix = test_util.randomArrayInRange(size * size, -1, 1);
    var channelsPerTexture = webgl_util.getChannelsPerTexture();
    var unpackedArray = new Float32Array(tex_util.getUnpackedArraySizeFromMatrixSize(matrix.length, channelsPerTexture));
    var start = performance.now();
    for (var i = 0; i < OPS_PER_RUN; ++i) {
        tex_util.encodeMatrixToUnpackedArray(matrix, unpackedArray, channelsPerTexture);
    }
    var end = performance.now();
    return (end - start) / OPS_PER_RUN;
};
exports.BENCHMARK_ENCODE_PACKED = function (size) {
    var matrix = test_util.randomArrayInRange(size * size, -1, 1);
    var packedRGBA = new Float32Array(tex_util.getPackedRGBAArraySizeFromMatrixShape(size, size));
    var start = performance.now();
    for (var i = 0; i < OPS_PER_RUN; ++i) {
        tex_util.encodeMatrixToPackedRGBA(matrix, size, size, packedRGBA);
    }
    var end = performance.now();
    return (end - start) / OPS_PER_RUN;
};
exports.BENCHMARK_DECODE_UNPACKED = function (size) {
    var matrix = test_util.randomArrayInRange(size * size, -1, 1);
    var channelsPerTexture = webgl_util.getChannelsPerTexture();
    var unpackedArray = new Float32Array(tex_util.getUnpackedArraySizeFromMatrixSize(matrix.length, channelsPerTexture));
    tex_util.encodeMatrixToUnpackedArray(matrix, unpackedArray, channelsPerTexture);
    var start = performance.now();
    for (var i = 0; i < OPS_PER_RUN; ++i) {
        tex_util.decodeMatrixFromUnpackedArray(unpackedArray, matrix, channelsPerTexture);
    }
    var end = performance.now();
    return (end - start) / OPS_PER_RUN;
};
exports.BENCHMARK_DECODE_PACKED = function (size) {
    var matrix = test_util.randomArrayInRange(size * size, -1, 1);
    var packedRGBA = new Float32Array(tex_util.getPackedRGBAArraySizeFromMatrixShape(size, size));
    tex_util.encodeMatrixToPackedRGBA(matrix, size, size, packedRGBA);
    var start = performance.now();
    for (var i = 0; i < OPS_PER_RUN; ++i) {
        tex_util.decodeMatrixFromPackedRGBA(packedRGBA, size, size, matrix);
    }
    var end = performance.now();
    return (end - start) / OPS_PER_RUN;
};

},{"../../src/math/webgl/tex_util":33,"../../src/math/webgl/webgl_util":34,"../../src/test_util":35}],13:[function(require,module,exports){
Polymer({ is: 'demo-footer' });

},{}],14:[function(require,module,exports){
Polymer({ is: 'demo-header' });

},{}],15:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function PolymerElement(spec) {
    return Polymer.Class(spec);
}
exports.PolymerElement = PolymerElement;

},{}],16:[function(require,module,exports){
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

},{"../util":36}],17:[function(require,module,exports){
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

},{"../util":36}],18:[function(require,module,exports){
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

},{}],19:[function(require,module,exports){
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

},{"../util":36,"./concat3d_util":16,"./copy2d_util":18,"./ndarray":21}],20:[function(require,module,exports){
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

},{"../math/conv_util":17,"../util":36,"./concat3d_util":16,"./copy2d_util":18,"./math":19,"./ndarray":21}],21:[function(require,module,exports){
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

},{"../util":36,"./webgl/webgl_util":34}],22:[function(require,module,exports){
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

},{"../conv_util":17,"./conv_gpu":23}],23:[function(require,module,exports){
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

},{"../conv_util":17}],24:[function(require,module,exports){
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

},{"./gpgpu_util":25,"./tex_util":33,"./webgl_util":34}],25:[function(require,module,exports){
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

},{"./tex_util":33,"./webgl_util":34}],26:[function(require,module,exports){
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

},{"./gpgpu_context":24}],27:[function(require,module,exports){
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

},{"../conv_util":17}],28:[function(require,module,exports){
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

},{"./pool_gpu":31}],29:[function(require,module,exports){
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

},{"../math":19,"./shader_compiler":32}],30:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var math_1 = require("../math");
var gpgpu_context_1 = require("./gpgpu_context");
function getFragmentShaderSource(sharedDimension, aOrientation, bOrientation) {
    var sharedDimensionPacked = Math.ceil(sharedDimension / 2);
    var aSample = (aOrientation === math_1.MatrixOrientation.REGULAR) ?
        'center, resultUV.t' :
        'resultUV.t, center';
    var bSample = (bOrientation === math_1.MatrixOrientation.REGULAR) ?
        'resultUV.s, center' :
        'center, resultUV.s';
    var aSwizzle = (aOrientation === math_1.MatrixOrientation.REGULAR) ? ['a.xxzz', 'a.yyww'] :
        ['a.xxyy', 'a.zzww'];
    var bSwizzle = (bOrientation === math_1.MatrixOrientation.REGULAR) ? ['b.xyxy', 'b.zwzw'] :
        ['b.xzxz', 'b.ywyw'];
    return "\n    precision highp float;\n    uniform sampler2D matrixA;\n    uniform sampler2D matrixB;\n    varying vec2 resultUV;\n\n    const float sharedDimension = " + sharedDimensionPacked + ".0;\n\n    vec4 dot2x2ARowBCol() {\n      vec4 result = vec4(0, 0, 0, 0);\n      for (float i = 0.0; i < sharedDimension; i += 1.0) {\n        float center = (i + 0.5) / sharedDimension;\n        vec4 a = texture2D(matrixA, vec2(" + aSample + "));\n        vec4 b = texture2D(matrixB, vec2(" + bSample + "));\n        result +=\n          (" + aSwizzle[0] + " * " + bSwizzle[0] + ") + (" + aSwizzle[1] + " * " + bSwizzle[1] + ");\n      }\n      return result;\n    }\n\n    void main() {\n      gl_FragColor = dot2x2ARowBCol();\n    }";
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function multiplyMatrixPacked(gpgpu, multiplyProgram, a, b, result, resultShapeRowCol) {
    gpgpu.setOutputPackedMatrixTexture(result, resultShapeRowCol[0], resultShapeRowCol[1]);
    gpgpu.setProgram(multiplyProgram);
    gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
    gpgpu.setInputMatrixTexture(b, 'matrixB', 1);
    gpgpu.executeProgram();
}
exports.multiplyMatrixPacked = multiplyMatrixPacked;
function uploadMultiplyMatrixPackedDownload(a, aShapeRowCol, b, bShapeRowCol, aOrientation, bOrientation) {
    if (aOrientation === void 0) { aOrientation = math_1.MatrixOrientation.REGULAR; }
    if (bOrientation === void 0) { bOrientation = math_1.MatrixOrientation.REGULAR; }
    var resultNumRows = (aOrientation === math_1.MatrixOrientation.REGULAR) ?
        aShapeRowCol[0] :
        aShapeRowCol[1];
    var resultNumCols = (bOrientation === math_1.MatrixOrientation.REGULAR) ?
        bShapeRowCol[1] :
        bShapeRowCol[0];
    var sharedDimension = (aOrientation === math_1.MatrixOrientation.REGULAR) ?
        aShapeRowCol[1] :
        aShapeRowCol[0];
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var program = gpgpu.createProgram(getFragmentShaderSource(sharedDimension, aOrientation, bOrientation));
    var aTexture = gpgpu.createPackedMatrixTexture(aShapeRowCol[0], aShapeRowCol[1]);
    var bTexture = gpgpu.createPackedMatrixTexture(bShapeRowCol[0], bShapeRowCol[1]);
    var resultTexture = gpgpu.createPackedMatrixTexture(resultNumRows, resultNumCols);
    gpgpu.uploadMatrixToPackedTexture(aTexture, aShapeRowCol[0], aShapeRowCol[1], a);
    gpgpu.uploadMatrixToPackedTexture(bTexture, bShapeRowCol[0], bShapeRowCol[1], b);
    multiplyMatrixPacked(gpgpu, program, aTexture, bTexture, resultTexture, [resultNumRows, resultNumCols]);
    var result = gpgpu.downloadMatrixFromPackedTexture(resultTexture, resultNumRows, resultNumCols);
    gpgpu.deleteMatrixTexture(aTexture);
    gpgpu.deleteMatrixTexture(bTexture);
    gpgpu.deleteMatrixTexture(resultTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return result;
}
exports.uploadMultiplyMatrixPackedDownload = uploadMultiplyMatrixPackedDownload;

},{"../math":19,"./gpgpu_context":24}],31:[function(require,module,exports){
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

},{"../conv_util":17,"./webgl_util":34}],32:[function(require,module,exports){
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

},{"../../util":36}],33:[function(require,module,exports){
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

},{}],34:[function(require,module,exports){
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

},{"../../util":36}],35:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function expectArraysClose(actual, expected, epsilon) {
    if (actual.length !== expected.length) {
        throw new Error('Matrices have different lengths (' + actual.length + ' vs ' +
            expected.length + ').');
    }
    for (var i = 0; i < expected.length; ++i) {
        var a = actual[i];
        var e = expected[i];
        if (isNaN(a) && isNaN(e)) {
            continue;
        }
        if (isNaN(a) || isNaN(e) || Math.abs(a - e) > epsilon) {
            var actualStr = 'actual[' + i + '] === ' + a;
            var expectedStr = 'expected[' + i + '] === ' + e;
            throw new Error('Arrays differ: ' + actualStr + ', ' + expectedStr);
        }
    }
}
exports.expectArraysClose = expectArraysClose;
function randomArrayInRange(n, minValue, maxValue) {
    var v = new Float32Array(n);
    var range = maxValue - minValue;
    for (var i = 0; i < n; ++i) {
        v[i] = (Math.random() * range) + minValue;
    }
    return v;
}
exports.randomArrayInRange = randomArrayInRange;
function makeIdentity(n) {
    var i = new Float32Array(n * n);
    for (var j = 0; j < n; ++j) {
        i[(j * n) + j] = 1;
    }
    return i;
}
exports.makeIdentity = makeIdentity;
function setValue(m, mNumRows, mNumCols, v, row, column) {
    if (row >= mNumRows) {
        throw new Error('row (' + row + ') must be in [0 ' + mNumRows + '].');
    }
    if (column >= mNumCols) {
        throw new Error('column (' + column + ') must be in [0 ' + mNumCols + '].');
    }
    m[(row * mNumCols) + column] = v;
}
exports.setValue = setValue;
function cpuMultiplyMatrix(a, aRow, aCol, b, bRow, bCol) {
    var result = new Float32Array(aRow * bCol);
    for (var r = 0; r < aRow; ++r) {
        for (var c = 0; c < bCol; ++c) {
            var d = 0;
            for (var k = 0; k < aCol; ++k) {
                d += a[(r * aCol) + k] * b[(k * bCol) + c];
            }
            result[(r * bCol) + c] = d;
        }
    }
    return result;
}
exports.cpuMultiplyMatrix = cpuMultiplyMatrix;
function cpuDotProduct(a, b) {
    if (a.length !== b.length) {
        throw new Error('cpuDotProduct: incompatible vectors.');
    }
    var d = 0;
    for (var i = 0; i < a.length; ++i) {
        d += a[i] * b[i];
    }
    return d;
}
exports.cpuDotProduct = cpuDotProduct;

},{}],36:[function(require,module,exports){
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

},{}]},{},[7])
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIm5vZGVfbW9kdWxlcy9icm93c2VyLXBhY2svX3ByZWx1ZGUuanMiLCJkZW1vcy9iZW5jaG1hcmtzL2JlbmNobWFyay50cyIsImRlbW9zL2JlbmNobWFya3MvY29udl9ncHVfYmVuY2htYXJrLnRzIiwiZGVtb3MvYmVuY2htYXJrcy9jb252X3RyYW5zcG9zZV9ncHVfYmVuY2htYXJrLnRzIiwiZGVtb3MvYmVuY2htYXJrcy9sb2dzdW1leHBfY3B1X2JlbmNobWFyay50cyIsImRlbW9zL2JlbmNobWFya3MvbG9nc3VtZXhwX2dwdV9iZW5jaG1hcmsudHMiLCJkZW1vcy9iZW5jaG1hcmtzL21hdGgtYmVuY2htYXJrLXJ1bi1ncm91cHMudHMiLCJkZW1vcy9iZW5jaG1hcmtzL21hdGgtYmVuY2htYXJrLnRzIiwiZGVtb3MvYmVuY2htYXJrcy9tYXhfcG9vbF9iYWNrcHJvcF9ncHVfYmVuY2htYXJrLnRzIiwiZGVtb3MvYmVuY2htYXJrcy9tYXhfcG9vbF9ncHVfYmVuY2htYXJrLnRzIiwiZGVtb3MvYmVuY2htYXJrcy9tdWxtYXRfY3B1X2JlbmNobWFyay50cyIsImRlbW9zL2JlbmNobWFya3MvbXVsbWF0X2dwdV9iZW5jaG1hcmsudHMiLCJkZW1vcy9iZW5jaG1hcmtzL3RleF91dGlsX2JlbmNobWFyay50cyIsImRlbW9zL2RlbW8tZm9vdGVyLnRzIiwiZGVtb3MvZGVtby1oZWFkZXIudHMiLCJkZW1vcy9wb2x5bWVyLXNwZWMudHMiLCJzcmMvbWF0aC9jb25jYXQzZF91dGlsLnRzIiwic3JjL21hdGgvY29udl91dGlsLnRzIiwic3JjL21hdGgvY29weTJkX3V0aWwudHMiLCJzcmMvbWF0aC9tYXRoLnRzIiwic3JjL21hdGgvbWF0aF9jcHUudHMiLCJzcmMvbWF0aC9uZGFycmF5LnRzIiwic3JjL21hdGgvd2ViZ2wvY29udl9iYWNrcHJvcF9ncHUudHMiLCJzcmMvbWF0aC93ZWJnbC9jb252X2dwdS50cyIsInNyYy9tYXRoL3dlYmdsL2dwZ3B1X2NvbnRleHQudHMiLCJzcmMvbWF0aC93ZWJnbC9ncGdwdV91dGlsLnRzIiwic3JjL21hdGgvd2ViZ2wvbG9nc3VtZXhwX2dwdS50cyIsInNyYy9tYXRoL3dlYmdsL21heF9wb29sX2JhY2twcm9wX2dwdS50cyIsInNyYy9tYXRoL3dlYmdsL21heF9wb29sX2dwdS50cyIsInNyYy9tYXRoL3dlYmdsL211bG1hdF9ncHUudHMiLCJzcmMvbWF0aC93ZWJnbC9tdWxtYXRfcGFja2VkX2dwdS50cyIsInNyYy9tYXRoL3dlYmdsL3Bvb2xfZ3B1LnRzIiwic3JjL21hdGgvd2ViZ2wvc2hhZGVyX2NvbXBpbGVyLnRzIiwic3JjL21hdGgvd2ViZ2wvdGV4X3V0aWwudHMiLCJzcmMvbWF0aC93ZWJnbC93ZWJnbF91dGlsLnRzIiwic3JjL3Rlc3RfdXRpbC50cyIsInNyYy91dGlsLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7QUMyQkE7SUFLRSxzQkFBWSxJQUFZLEVBQUUsYUFBNEI7UUFDcEQsSUFBSSxDQUFDLElBQUksR0FBRyxJQUFJLENBQUM7UUFDakIsSUFBSSxDQUFDLGFBQWEsR0FBRyxhQUFhLENBQUM7UUFDbkMsSUFBSSxDQUFDLFNBQVMsR0FBRyxFQUFFLENBQUM7SUFDdEIsQ0FBQztJQUNILG1CQUFDO0FBQUQsQ0FWQSxBQVVDLElBQUE7QUFWWSxvQ0FBWTs7Ozs7QUNaekIsb0RBQXNEO0FBQ3RELHdEQUEwRDtBQUMxRCxvRUFBZ0U7QUFDaEUsK0NBQWlEO0FBSWpELElBQU0sT0FBTyxHQUFHLEdBQUcsQ0FBQztBQUVQLFFBQUEsY0FBYyxHQUFrQixVQUFDLElBQVk7SUFDeEQsSUFBTSxhQUFhLEdBQTZCLENBQUMsSUFBSSxFQUFFLElBQUksRUFBRSxDQUFDLENBQUMsQ0FBQztJQUNoRSxJQUFNLFdBQVcsR0FBRyxDQUFDLENBQUM7SUFDdEIsSUFBTSxTQUFTLEdBQUcsRUFBRSxDQUFDO0lBQ3JCLElBQU0sTUFBTSxHQUFHLENBQUMsQ0FBQztJQUNqQixJQUFNLE9BQU8sR0FBRyxTQUFTLENBQUMsaUJBQWlCLENBQUMsYUFBYSxFQUFFLFNBQVMsRUFBRSxNQUFNLENBQUMsQ0FBQztJQUM5RSxJQUFNLGNBQWMsR0FDaEIsU0FBUyxDQUFDLG9CQUFvQixDQUMxQixhQUFhLEVBQUUsU0FBUyxFQUFFLFdBQVcsRUFBRSxNQUFNLEVBQUUsT0FBTyxDQUFDLENBQUM7SUFFaEUsSUFBTSxlQUFlLEdBQUcsU0FBUyxDQUFDLHFCQUFxQixDQUFDLGFBQWEsQ0FBQyxDQUFDO0lBQ3ZFLElBQU0sZ0JBQWdCLEdBQUcsU0FBUyxDQUFDLHFCQUFxQixDQUFDLGNBQWMsQ0FBQyxDQUFDO0lBQ3pFLElBQU0saUJBQWlCLEdBQUcsU0FBUyxDQUFDLHNCQUFzQixDQUN0RCxhQUFhLENBQUMsQ0FBQyxDQUFDLEVBQUUsV0FBVyxFQUFFLFNBQVMsQ0FBQyxDQUFDO0lBQzlDLElBQU0sZ0JBQWdCLEdBQUcsU0FBUyxDQUFDLHFCQUFxQixDQUFDLFdBQVcsQ0FBQyxDQUFDO0lBRXRFLElBQU0sT0FBTyxHQUFHLElBQUksQ0FBQztJQUNyQixJQUFNLEtBQUssR0FBRyxJQUFJLDRCQUFZLEVBQUUsQ0FBQztJQUNqQyxJQUFNLE9BQU8sR0FBRyxLQUFLLENBQUMsYUFBYSxDQUFDLFFBQVEsQ0FBQyx1QkFBdUIsQ0FDaEUsYUFBYSxFQUFFLFdBQVcsRUFBRSxTQUFTLEVBQUUsTUFBTSxFQUFFLE9BQU8sRUFBRSxPQUFPLENBQUMsQ0FBQyxDQUFDO0lBRXRFLElBQU0sWUFBWSxHQUNkLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxlQUFlLENBQUMsQ0FBQyxDQUFDLEVBQUUsZUFBZSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDdEUsSUFBTSxjQUFjLEdBQ2hCLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxpQkFBaUIsQ0FBQyxDQUFDLENBQUMsRUFBRSxpQkFBaUIsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzFFLElBQU0sYUFBYSxHQUNmLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3hFLElBQU0sYUFBYSxHQUNmLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBRXhFLElBQU0sU0FBUyxHQUFHLFNBQVMsQ0FBQyxrQkFBa0IsQ0FDMUMsZUFBZSxDQUFDLENBQUMsQ0FBQyxHQUFHLGVBQWUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUNwRCxJQUFNLFdBQVcsR0FBRyxTQUFTLENBQUMsa0JBQWtCLENBQzVDLGlCQUFpQixDQUFDLENBQUMsQ0FBQyxHQUFHLGlCQUFpQixDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQ3hELElBQU0sVUFBVSxHQUFHLFNBQVMsQ0FBQyxrQkFBa0IsQ0FDM0MsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLEdBQUcsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFFdEQsS0FBSyxDQUFDLHFCQUFxQixDQUN2QixZQUFZLEVBQUUsZUFBZSxDQUFDLENBQUMsQ0FBQyxFQUFFLGVBQWUsQ0FBQyxDQUFDLENBQUMsRUFBRSxTQUFTLENBQUMsQ0FBQztJQUNyRSxLQUFLLENBQUMscUJBQXFCLENBQ3ZCLGNBQWMsRUFBRSxpQkFBaUIsQ0FBQyxDQUFDLENBQUMsRUFBRSxpQkFBaUIsQ0FBQyxDQUFDLENBQUMsRUFBRSxXQUFXLENBQUMsQ0FBQztJQUM3RSxLQUFLLENBQUMscUJBQXFCLENBQ3ZCLGFBQWEsRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsRUFBRSxVQUFVLENBQUMsQ0FBQztJQUV6RSxJQUFNLEtBQUssR0FBRyxXQUFXLENBQUMsR0FBRyxFQUFFLENBQUM7SUFDaEMsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxPQUFPLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztRQUNqQyxRQUFRLENBQUMsUUFBUSxDQUNiLEtBQUssRUFBRSxPQUFPLEVBQUUsWUFBWSxFQUFFLGNBQWMsRUFBRSxhQUFhLEVBQzNELGFBQWEsRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDO0lBQ3ZDLENBQUM7SUFFRCxLQUFLLENBQUMseUJBQXlCLENBQzNCLGFBQWEsRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzdELElBQU0sR0FBRyxHQUFHLFdBQVcsQ0FBQyxHQUFHLEVBQUUsQ0FBQztJQUU5QixJQUFNLE9BQU8sR0FBRyxDQUFDLEdBQUcsR0FBRyxLQUFLLENBQUMsR0FBRyxPQUFPLENBQUM7SUFFeEMsS0FBSyxDQUFDLG1CQUFtQixDQUFDLFlBQVksQ0FBQyxDQUFDO0lBQ3hDLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxjQUFjLENBQUMsQ0FBQztJQUMxQyxLQUFLLENBQUMsbUJBQW1CLENBQUMsYUFBYSxDQUFDLENBQUM7SUFDekMsS0FBSyxDQUFDLG1CQUFtQixDQUFDLGFBQWEsQ0FBQyxDQUFDO0lBQ3pDLEtBQUssQ0FBQyxhQUFhLENBQUMsT0FBTyxDQUFDLENBQUM7SUFDN0IsS0FBSyxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBRWhCLE1BQU0sQ0FBQyxPQUFPLENBQUM7QUFDakIsQ0FBQyxDQUFDOzs7OztBQzFFRixvREFBc0Q7QUFDdEQsMEVBQTRFO0FBQzVFLG9FQUFnRTtBQUNoRSwrQ0FBaUQ7QUFJakQsSUFBTSxPQUFPLEdBQUcsR0FBRyxDQUFDO0FBRVAsUUFBQSxjQUFjLEdBQWtCLFVBQUMsSUFBWTtJQUN4RCxJQUFNLFNBQVMsR0FBNkIsQ0FBQyxJQUFJLEVBQUUsSUFBSSxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQzVELElBQU0sZUFBZSxHQUFHLENBQUMsQ0FBQztJQUMxQixJQUFNLFNBQVMsR0FBRyxFQUFFLENBQUM7SUFDckIsSUFBTSxVQUFVLEdBQUcsQ0FBQyxDQUFDO0lBQ3JCLElBQU0sT0FBTyxHQUFHLENBQUMsQ0FBQztJQUVsQixJQUFNLEtBQUssR0FBRyxJQUFJLDRCQUFZLEVBQUUsQ0FBQztJQUNqQyxLQUFLLENBQUMsOEJBQThCLENBQUMsSUFBSSxDQUFDLENBQUM7SUFDM0MsSUFBTSxjQUFjLEdBQUcsU0FBUyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3BDLElBQU0sR0FBRyxHQUFHLGlCQUFpQixDQUFDLG9DQUFvQyxDQUM5RCxTQUFTLEVBQUUsU0FBUyxFQUFFLGNBQWMsRUFBRSxVQUFVLEVBQUUsT0FBTyxFQUFFLEtBQUssQ0FBQyxDQUFDO0lBQ3RFLElBQU0sT0FBTyxHQUFHLEtBQUssQ0FBQyxhQUFhLENBQUMsR0FBRyxDQUFDLENBQUM7SUFHekMsSUFBTSxXQUFXLEdBQUcsU0FBUyxDQUFDLHFCQUFxQixDQUFDLFNBQVMsQ0FBQyxDQUFDO0lBQy9ELElBQU0sSUFBSSxHQUFHLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxXQUFXLENBQUMsQ0FBQyxDQUFDLEVBQUUsV0FBVyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDdkUsSUFBTSxLQUFLLEdBQ1AsU0FBUyxDQUFDLGtCQUFrQixDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUMsR0FBRyxXQUFXLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDekUsS0FBSyxDQUFDLHFCQUFxQixDQUFDLElBQUksRUFBRSxXQUFXLENBQUMsQ0FBQyxDQUFDLEVBQUUsV0FBVyxDQUFDLENBQUMsQ0FBQyxFQUFFLEtBQUssQ0FBQyxDQUFDO0lBR3pFLElBQU0sV0FBVyxHQUFHLFNBQVMsQ0FBQyxzQkFBc0IsQ0FDaEQsY0FBYyxFQUFFLGVBQWUsRUFBRSxTQUFTLENBQUMsQ0FBQztJQUNoRCxJQUFNLEtBQUssR0FDUCxTQUFTLENBQUMsa0JBQWtCLENBQUMsV0FBVyxDQUFDLENBQUMsQ0FBQyxHQUFHLFdBQVcsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUN6RSxJQUFNLElBQUksR0FBRyxLQUFLLENBQUMsbUJBQW1CLENBQUMsV0FBVyxDQUFDLENBQUMsQ0FBQyxFQUFFLFdBQVcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3ZFLEtBQUssQ0FBQyxxQkFBcUIsQ0FBQyxJQUFJLEVBQUUsV0FBVyxDQUFDLENBQUMsQ0FBQyxFQUFFLFdBQVcsQ0FBQyxDQUFDLENBQUMsRUFBRSxLQUFLLENBQUMsQ0FBQztJQUd6RSxJQUFNLFNBQVMsR0FDWCxTQUFTLENBQUMsZ0JBQWdCLENBQUMsQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDLEVBQUUsU0FBUyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsVUFBVSxDQUFDLENBQUM7SUFDekUsSUFBTSxHQUFHLEdBQUcsU0FBUyxHQUFHLENBQUMsR0FBRyxPQUFPLENBQUM7SUFDcEMsSUFBTSxjQUFjLEdBQUcsU0FBUyxDQUFDLG9CQUFvQixDQUNqRCxDQUFDLFNBQVMsQ0FBQyxDQUFDLENBQUMsRUFBRSxTQUFTLENBQUMsQ0FBQyxDQUFDLEVBQUUsZUFBZSxDQUFDLEVBQUUsU0FBUyxFQUFFLGNBQWMsRUFDeEUsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxDQUFDO0lBRVosSUFBTSxXQUFXLEdBQUcsU0FBUyxDQUFDLHFCQUFxQixDQUFDLGNBQWMsQ0FBQyxDQUFDO0lBQ3BFLElBQU0sU0FBUyxHQUFHLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxXQUFXLENBQUMsQ0FBQyxDQUFDLEVBQUUsV0FBVyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFFNUUsSUFBTSxLQUFLLEdBQUcsV0FBVyxDQUFDLEdBQUcsRUFBRSxDQUFDO0lBQ2hDLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsT0FBTyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7UUFDakMsaUJBQWlCLENBQUMsYUFBYSxDQUMzQixLQUFLLEVBQUUsT0FBTyxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxXQUFXLENBQUMsQ0FBQztJQUNoRSxDQUFDO0lBRUQsSUFBTSxDQUFDLEdBQUcsS0FBSyxDQUFDLHlCQUF5QixDQUNyQyxTQUFTLEVBQUUsV0FBVyxDQUFDLENBQUMsQ0FBQyxFQUFFLFdBQVcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBRS9DLElBQU0sR0FBRyxHQUFHLFdBQVcsQ0FBQyxHQUFHLEVBQUUsQ0FBQztJQUU5QixJQUFNLE9BQU8sR0FBRyxDQUFDLEdBQUcsR0FBRyxLQUFLLENBQUMsR0FBRyxPQUFPLENBQUM7SUFFeEMsS0FBSyxDQUFDLG1CQUFtQixDQUFDLFNBQVMsQ0FBQyxDQUFDO0lBQ3JDLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxJQUFJLENBQUMsQ0FBQztJQUNoQyxLQUFLLENBQUMsbUJBQW1CLENBQUMsSUFBSSxDQUFDLENBQUM7SUFDaEMsS0FBSyxDQUFDLGFBQWEsQ0FBQyxPQUFPLENBQUMsQ0FBQztJQUM3QixLQUFLLENBQUMsT0FBTyxFQUFFLENBQUM7SUFFaEIsTUFBTSxDQUFDLE9BQU8sQ0FBQztBQUNqQixDQUFDLENBQUM7Ozs7O0FDckVGLG9EQUF1RDtBQUN2RCxrREFBd0Q7QUFJeEQsSUFBTSxXQUFXLEdBQUcsRUFBRSxDQUFDO0FBRVYsUUFBQSxjQUFjLEdBQWtCLFVBQUMsSUFBWTtJQUN4RCxJQUFNLElBQUksR0FBRyxJQUFJLHlCQUFjLEVBQUUsQ0FBQztJQUNsQyxJQUFNLENBQUMsR0FBRyxpQkFBTyxDQUFDLFdBQVcsQ0FBVSxDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUM1RCxJQUFNLEtBQUssR0FBRyxXQUFXLENBQUMsR0FBRyxFQUFFLENBQUM7SUFDaEMsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxXQUFXLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztRQUNyQyxJQUFJLENBQUMsU0FBUyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3BCLENBQUM7SUFDRCxJQUFNLEdBQUcsR0FBRyxXQUFXLENBQUMsR0FBRyxFQUFFLENBQUM7SUFDOUIsTUFBTSxDQUFDLENBQUMsR0FBRyxHQUFHLEtBQUssQ0FBQyxHQUFHLFdBQVcsQ0FBQztBQUNyQyxDQUFDLENBQUM7Ozs7O0FDaEJGLG9FQUFnRTtBQUNoRSxrRUFBb0U7QUFDcEUsK0NBQWlEO0FBSWpELElBQU0sT0FBTyxHQUFHLEdBQUcsQ0FBQztBQUVQLFFBQUEsY0FBYyxHQUFrQixVQUFDLElBQVk7SUFDeEQsSUFBTSxLQUFLLEdBQUcsSUFBSSw0QkFBWSxFQUFFLENBQUM7SUFFakMsSUFBTSxPQUFPLEdBQ1QsS0FBSyxDQUFDLGFBQWEsQ0FBQyxhQUFhLENBQUMsdUJBQXVCLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDLENBQUM7SUFFM0UsSUFBTSxRQUFRLEdBQUcsS0FBSyxDQUFDLG1CQUFtQixDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsQ0FBQztJQUN2RCxJQUFNLGFBQWEsR0FBRyxLQUFLLENBQUMsbUJBQW1CLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDO0lBRTVELElBQU0sQ0FBQyxHQUFHLFNBQVMsQ0FBQyxrQkFBa0IsQ0FBQyxJQUFJLEdBQUcsSUFBSSxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQzNELEtBQUssQ0FBQyxxQkFBcUIsQ0FBQyxRQUFRLEVBQUUsSUFBSSxFQUFFLElBQUksRUFBRSxDQUFDLENBQUMsQ0FBQztJQUVyRCxJQUFNLEtBQUssR0FBRyxXQUFXLENBQUMsR0FBRyxFQUFFLENBQUM7SUFDaEMsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxPQUFPLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztRQUNqQyxhQUFhLENBQUMsU0FBUyxDQUNuQixLQUFLLEVBQUUsT0FBTyxFQUFFLFFBQVEsRUFBRSxJQUFJLEVBQUUsSUFBSSxFQUFFLGFBQWEsQ0FBQyxDQUFDO0lBQzNELENBQUM7SUFFRCxLQUFLLENBQUMseUJBQXlCLENBQUMsYUFBYSxFQUFFLElBQUksRUFBRSxJQUFJLENBQUMsQ0FBQztJQUMzRCxJQUFNLE9BQU8sR0FBRyxDQUFDLFdBQVcsQ0FBQyxHQUFHLEVBQUUsR0FBRyxLQUFLLENBQUMsR0FBRyxPQUFPLENBQUM7SUFFdEQsS0FBSyxDQUFDLG1CQUFtQixDQUFDLFFBQVEsQ0FBQyxDQUFDO0lBQ3BDLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxhQUFhLENBQUMsQ0FBQztJQUN6QyxLQUFLLENBQUMsYUFBYSxDQUFDLE9BQU8sQ0FBQyxDQUFDO0lBQzdCLEtBQUssQ0FBQyxPQUFPLEVBQUUsQ0FBQztJQUVoQixNQUFNLENBQUMsT0FBTyxDQUFDO0FBQ2pCLENBQUMsQ0FBQzs7Ozs7QUNuQ0YseUNBQTREO0FBQzVELHlEQUEyRDtBQUMzRCw2RUFBK0U7QUFDL0UsbUVBQXFFO0FBQ3JFLG1FQUFxRTtBQUNyRSxtRkFBcUY7QUFDckYsaUVBQW1FO0FBQ25FLDZEQUErRDtBQUMvRCw2REFBK0Q7QUFDL0QseURBQTJEO0FBRTlDLFFBQUEsb0JBQW9CLEdBQXdCO0lBQ3ZEO1FBQ0UsSUFBSSxFQUFFLGtEQUFrRDtRQUN4RCxHQUFHLEVBQUUsQ0FBQztRQUNOLEdBQUcsRUFBRSxJQUFJO1FBQ1QsUUFBUSxFQUFFLEVBQUU7UUFDWix3QkFBd0IsRUFBRSxVQUFDLElBQVksSUFBSyxPQUFBLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxFQUFqQixDQUFpQjtRQUM3RCxhQUFhLEVBQUU7WUFDYixJQUFJLHdCQUFZLENBQ1osaUJBQWlCLEVBQUUsa0JBQWtCLENBQUMseUJBQXlCLENBQUM7WUFDcEUsSUFBSSx3QkFBWSxDQUNaLGVBQWUsRUFBRSxrQkFBa0IsQ0FBQyx1QkFBdUIsQ0FBQztZQUNoRSxJQUFJLHdCQUFZLENBQ1osaUJBQWlCLEVBQUUsa0JBQWtCLENBQUMseUJBQXlCLENBQUM7WUFDcEUsSUFBSSx3QkFBWSxDQUNaLGVBQWUsRUFBRSxrQkFBa0IsQ0FBQyx1QkFBdUIsQ0FBQztTQUNqRTtLQUNGO0lBQ0Q7UUFDRSxJQUFJLEVBQUUsb0NBQW9DO1FBQzFDLEdBQUcsRUFBRSxDQUFDO1FBQ04sR0FBRyxFQUFFLElBQUk7UUFDVCxRQUFRLEVBQUUsRUFBRTtRQUNaLHdCQUF3QixFQUFFLFVBQUMsSUFBWSxJQUFLLE9BQUEsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLEVBQWpCLENBQWlCO1FBQzdELGFBQWEsRUFBRTtZQUNiLElBQUksd0JBQVksQ0FBQyxZQUFZLEVBQUUsb0JBQW9CLENBQUMsY0FBYyxDQUFDO1lBQ25FLElBQUksd0JBQVksQ0FDWixtQkFBbUIsRUFBRSxvQkFBb0IsQ0FBQyxxQkFBcUIsQ0FBQztZQUNwRSxJQUFJLHdCQUFZLENBQUMsWUFBWSxFQUFFLG9CQUFvQixDQUFDLGNBQWMsQ0FBQztTQUNwRTtLQUNGO0lBQ0Q7UUFDRSxJQUFJLEVBQUUsd0JBQXdCO1FBQzlCLEdBQUcsRUFBRSxDQUFDO1FBQ04sR0FBRyxFQUFFLElBQUk7UUFDVCxRQUFRLEVBQUUsRUFBRTtRQUNaLHdCQUF3QixFQUFFLFVBQUMsSUFBWSxJQUFLLE9BQUEsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLEVBQWpCLENBQWlCO1FBQzdELGFBQWEsRUFBRTtZQUNiLElBQUksd0JBQVksQ0FBQyxlQUFlLEVBQUUsdUJBQXVCLENBQUMsY0FBYyxDQUFDO1lBQ3pFLElBQUksd0JBQVksQ0FBQyxlQUFlLEVBQUUsdUJBQXVCLENBQUMsY0FBYyxDQUFDO1NBQzFFO0tBQ0Y7SUFDRDtRQUNFLElBQUksRUFBRSxtQkFBbUI7UUFDekIsR0FBRyxFQUFFLENBQUM7UUFDTixHQUFHLEVBQUUsSUFBSTtRQUNULFFBQVEsRUFBRSxFQUFFO1FBQ1osd0JBQXdCLEVBQUUsVUFBQyxJQUFZLElBQUssT0FBQSxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsRUFBakIsQ0FBaUI7UUFDN0QsYUFBYSxFQUFFLENBQUMsSUFBSSx3QkFBWSxDQUM1Qix1QkFBdUIsRUFBRSxrQkFBa0IsQ0FBQyxjQUFjLENBQUMsQ0FBQztLQUNqRTtJQUNEO1FBQ0UsSUFBSSxFQUFFLDhCQUE4QjtRQUNwQyxHQUFHLEVBQUUsQ0FBQztRQUNOLEdBQUcsRUFBRSxJQUFJO1FBQ1QsUUFBUSxFQUFFLEVBQUU7UUFDWix3QkFBd0IsRUFBRSxVQUFDLElBQVksSUFBSyxPQUFBLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxFQUFqQixDQUFpQjtRQUM3RCxhQUFhLEVBQUUsQ0FBQyxJQUFJLHdCQUFZLENBQzVCLHVCQUF1QixFQUFFLDRCQUE0QixDQUFDLGNBQWMsQ0FBQyxDQUFDO0tBQzNFO0lBQ0Q7UUFDRSxJQUFJLEVBQUUsZ0JBQWdCO1FBQ3RCLEdBQUcsRUFBRSxDQUFDO1FBQ04sR0FBRyxFQUFFLElBQUk7UUFDVCxRQUFRLEVBQUUsRUFBRTtRQUNaLHdCQUF3QixFQUFFLFVBQUMsSUFBWSxJQUFLLE9BQUEsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLEVBQWpCLENBQWlCO1FBQzdELGFBQWEsRUFBRSxDQUFDLElBQUksd0JBQVksQ0FDNUIsdUJBQXVCLEVBQ3ZCLHNCQUFzQixDQUFDLHVCQUF1QixDQUFDLENBQUM7S0FDckQ7SUFDRDtRQUNFLElBQUksRUFBRSwwQkFBMEI7UUFDaEMsR0FBRyxFQUFFLENBQUM7UUFDTixHQUFHLEVBQUUsSUFBSTtRQUNULFFBQVEsRUFBRSxFQUFFO1FBQ1osd0JBQXdCLEVBQUUsVUFBQyxJQUFZLElBQUssT0FBQSxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsRUFBakIsQ0FBaUI7UUFDN0QsYUFBYSxFQUFFLENBQUMsSUFBSSx3QkFBWSxDQUM1Qix1QkFBdUIsRUFDdkIsc0JBQXNCLENBQUMsNkJBQTZCLENBQUMsQ0FBQztLQUMzRDtJQUNEO1FBQ0UsSUFBSSxFQUFFLHlCQUF5QjtRQUMvQixHQUFHLEVBQUUsQ0FBQztRQUNOLEdBQUcsRUFBRSxJQUFJO1FBQ1QsUUFBUSxFQUFFLEVBQUU7UUFDWix3QkFBd0IsRUFBRSxVQUFDLElBQVksSUFBSyxPQUFBLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxFQUFqQixDQUFpQjtRQUM3RCxhQUFhLEVBQUUsQ0FBQyxJQUFJLHdCQUFZLENBQzVCLHVCQUF1QixFQUN2QiwrQkFBK0IsQ0FBQyxjQUFjLENBQUMsQ0FBQztLQUNyRDtDQUNGLENBQUM7Ozs7Ozs7Ozs7Ozs7OztBQ3JHRiwwQkFBd0I7QUFDeEIsMEJBQXdCO0FBR3hCLGdEQUFtRTtBQUduRSx5RUFBaUU7QUFHdEQsUUFBQSxvQkFBb0IsR0FBRyw2QkFBYyxDQUM1QyxFQUFDLEVBQUUsRUFBRSxnQkFBZ0IsRUFBRSxVQUFVLEVBQUUsRUFBQyxzQkFBc0IsRUFBRSxLQUFLLEVBQUMsRUFBQyxDQUFDLENBQUM7QUFFekU7SUFBbUMsaUNBQW9CO0lBQXZEOztJQW1NQSxDQUFDO0lBOUxDLDZCQUFLLEdBQUw7UUFBQSxpQkF1QkM7UUFyQkMsSUFBTSxzQkFBc0IsR0FBYSxFQUFFLENBQUM7UUFDNUMsSUFBSSxDQUFDLFlBQVksR0FBRyxFQUFFLENBQUM7UUFDdkIsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxnREFBb0IsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztZQUNyRCxzQkFBc0IsQ0FBQyxJQUFJLENBQUMsZ0RBQW9CLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUM7WUFDMUQsSUFBSSxDQUFDLFlBQVksQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7UUFDaEMsQ0FBQztRQUNELElBQUksQ0FBQyxzQkFBc0IsR0FBRyxzQkFBc0IsQ0FBQztRQUdyRCxVQUFVLENBQUM7WUFDVCxJQUFNLFVBQVUsR0FBRyxLQUFJLENBQUMsZ0JBQWdCLENBQUMsV0FBVyxDQUFDLENBQUM7WUFDdEQsSUFBTSxXQUFXLEdBQUcsS0FBSSxDQUFDLGdCQUFnQixDQUFDLFdBQVcsQ0FBQyxDQUFDO29DQUM5QyxDQUFDO2dCQUNSLFVBQVUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxnQkFBZ0IsQ0FBQyxPQUFPLEVBQUU7b0JBQ3RDLEtBQUksQ0FBQyxpQkFBaUIsQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDNUIsQ0FBQyxDQUFDLENBQUM7Z0JBQ0gsV0FBVyxDQUFDLENBQUMsQ0FBQyxDQUFDLGdCQUFnQixDQUFDLE9BQU8sRUFBRTtvQkFDdkMsS0FBSSxDQUFDLFlBQVksQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUM7Z0JBQzlCLENBQUMsQ0FBQyxDQUFDO1lBQ0wsQ0FBQztZQVBELEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsVUFBVSxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUU7d0JBQWpDLENBQUM7YUFPVDtRQUNILENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUNSLENBQUM7SUFFTyx5Q0FBaUIsR0FBekIsVUFBMEIsc0JBQThCO1FBQ3RELElBQU0saUJBQWlCLEdBQUcsZ0RBQW9CLENBQUMsc0JBQXNCLENBQUMsQ0FBQztRQUV2RSxJQUFNLE1BQU0sR0FBRyxJQUFJLENBQUMsZ0JBQWdCLENBQUMsV0FBVyxDQUFDLENBQUMsc0JBQXNCLENBQ25ELENBQUM7UUFDdEIsSUFBTSxPQUFPLEdBQUcsTUFBTSxDQUFDLFVBQVUsQ0FBQyxJQUFJLENBQTZCLENBQUM7UUFFcEUsSUFBTSxRQUFRLEdBQW9CLEVBQUUsQ0FBQztRQUNyQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLGlCQUFpQixDQUFDLGFBQWEsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztZQUNoRSxJQUFNLEdBQUcsR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLEdBQUcsR0FBRyxDQUFDLEdBQUcsaUJBQWlCLENBQUMsYUFBYSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1lBQ3pFLFFBQVEsQ0FBQyxJQUFJLENBQUM7Z0JBQ1osSUFBSSxFQUFFLGlCQUFpQixDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxTQUFTO2dCQUNsRCxJQUFJLEVBQUUsS0FBSztnQkFDWCxLQUFLLEVBQUUsaUJBQWlCLENBQUMsYUFBYSxDQUFDLENBQUMsQ0FBQyxDQUFDLElBQUk7Z0JBQzlDLFdBQVcsRUFBRSxNQUFNLEdBQUcsR0FBRyxHQUFHLGNBQWM7Z0JBQzFDLGVBQWUsRUFBRSxNQUFNLEdBQUcsR0FBRyxHQUFHLGNBQWM7Z0JBQzlDLFdBQVcsRUFBRSxDQUFDO2dCQUNkLGNBQWMsRUFBRSxDQUFDO2dCQUNqQixXQUFXLEVBQUUsQ0FBQztnQkFDZCxXQUFXLEVBQUUsQ0FBQzthQUNmLENBQUMsQ0FBQztRQUNMLENBQUM7UUFFRCxJQUFNLEtBQUssR0FBRyxJQUFJLEtBQUssQ0FBQyxPQUFPLEVBQUU7WUFDL0IsSUFBSSxFQUFFLE1BQU07WUFDWixJQUFJLEVBQUUsRUFBQyxRQUFRLFVBQUEsRUFBQztZQUNoQixPQUFPLEVBQUU7Z0JBQ1AsU0FBUyxFQUFFLEVBQUMsUUFBUSxFQUFFLENBQUMsRUFBQztnQkFDeEIsVUFBVSxFQUFFLEtBQUs7Z0JBQ2pCLE1BQU0sRUFBRTtvQkFDTixLQUFLLEVBQUUsQ0FBQzs0QkFDTixJQUFJLEVBQUUsUUFBUTs0QkFDZCxRQUFRLEVBQUUsUUFBUTs0QkFDbEIsS0FBSyxFQUFFO2dDQUNMLEdBQUcsRUFBRSxpQkFBaUIsQ0FBQyxHQUFHO2dDQUMxQixHQUFHLEVBQUUsaUJBQWlCLENBQUMsR0FBRztnQ0FDMUIsUUFBUSxFQUFFLGlCQUFpQixDQUFDLFFBQVE7Z0NBQ3BDLFFBQVEsRUFBRSxVQUFDLEtBQWE7b0NBQ3RCLE1BQU0sQ0FBQyxpQkFBaUIsQ0FBQyx3QkFBd0IsSUFBSSxJQUFJO3dDQUNyRCxpQkFBaUIsQ0FBQyx3QkFBd0IsQ0FBQyxDQUFDLEtBQUssQ0FBQzt3Q0FDbEQsQ0FBQyxLQUFLLENBQUM7Z0NBQ2IsQ0FBQzs2QkFFSzt5QkFDVCxDQUFDO29CQUNGLEtBQUssRUFBRSxDQUFDOzRCQUNOLEtBQUssRUFBRTtnQ0FDTCxRQUFRLEVBQUUsVUFBQyxLQUFLLEVBQUUsS0FBSyxFQUFFLE1BQU07b0NBQzdCLE1BQU0sQ0FBQyxLQUFLLEdBQUcsSUFBSSxDQUFDO2dDQUN0QixDQUFDOzZCQUNGO3lCQUNGLENBQUM7aUJBQ0g7Z0JBQ0QsUUFBUSxFQUFFLEVBQUMsSUFBSSxFQUFFLE9BQU8sRUFBQztnQkFDekIsS0FBSyxFQUFFLEVBQUMsSUFBSSxFQUFFLGlCQUFpQixDQUFDLElBQUksRUFBQzthQUN0QztTQUNGLENBQUMsQ0FBQztRQUNILE1BQU0sQ0FBQyxLQUFLLENBQUMsT0FBTyxHQUFHLE1BQU0sQ0FBQztRQUU5QixJQUFNLFVBQVUsR0FDWixJQUFJLENBQUMsZ0JBQWdCLENBQUMsY0FBYyxDQUFDLENBQUMsc0JBQXNCLENBQ2pELENBQUM7UUFDaEIsVUFBVSxDQUFDLEtBQUssQ0FBQyxPQUFPLEdBQUcsT0FBTyxDQUFDO1FBRW5DLElBQU0sZUFBZSxHQUNqQixJQUFJLENBQUMsZ0JBQWdCLENBQUMsb0JBQW9CLENBQUMsQ0FBQyxzQkFBc0IsQ0FDdkQsQ0FBQztRQUNoQixlQUFlLENBQUMsU0FBUyxHQUFHLEVBQUUsQ0FBQztRQUMvQixlQUFlLENBQUMsS0FBSyxDQUFDLE9BQU8sR0FBRyxNQUFNLENBQUM7UUFHdkMsSUFBTSxPQUFPLEdBQUcsQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUN6QixHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLGlCQUFpQixDQUFDLGFBQWEsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztZQUNoRSxPQUFPLENBQUMsSUFBSSxDQUFDLGlCQUFpQixDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUN4RCxDQUFDO1FBQ0QsZUFBZSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsa0JBQWtCLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztRQUU5RCxJQUFJLENBQUMsaUJBQWlCLENBQ2xCLEtBQUssRUFBRSxpQkFBaUIsRUFBRSxzQkFBc0IsRUFDaEQsaUJBQWlCLENBQUMsR0FBRyxDQUFDLENBQUM7SUFDN0IsQ0FBQztJQUVPLDBDQUFrQixHQUExQixVQUEyQixNQUFnQjtRQUN6QyxJQUFNLG1CQUFtQixHQUFHLFFBQVEsQ0FBQyxhQUFhLENBQUMsS0FBSyxDQUFDLENBQUM7UUFDMUQsbUJBQW1CLENBQUMsU0FBUyxHQUFHLGdDQUFnQyxDQUFDO1FBRWpFLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsTUFBTSxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1lBQ3ZDLElBQU0sb0JBQW9CLEdBQUcsUUFBUSxDQUFDLGFBQWEsQ0FBQyxLQUFLLENBQUMsQ0FBQztZQUMzRCxvQkFBb0IsQ0FBQyxTQUFTLEdBQUcsaUNBQWlDLENBQUM7WUFDbkUsb0JBQW9CLENBQUMsU0FBUyxHQUFHLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUMzQyxtQkFBbUIsQ0FBQyxXQUFXLENBQUMsb0JBQW9CLENBQUMsQ0FBQztRQUN4RCxDQUFDO1FBQ0QsTUFBTSxDQUFDLG1CQUFtQixDQUFDO0lBQzdCLENBQUM7SUFFTyx5Q0FBaUIsR0FBekIsVUFDSSxLQUFZLEVBQUUsaUJBQW9DLEVBQ2xELHNCQUE4QixFQUFFLElBQVk7UUFGaEQsaUJBcUVDO1FBbEVDLElBQU0sZUFBZSxHQUNqQixJQUFJLENBQUMsZ0JBQWdCLENBQUMsb0JBQW9CLENBQUMsQ0FBQyxzQkFBc0IsQ0FDdkQsQ0FBQztRQUNoQixFQUFFLENBQUMsQ0FBQyxJQUFJLEdBQUcsaUJBQWlCLENBQUMsR0FBRztZQUM1QixJQUFJLENBQUMsWUFBWSxDQUFDLHNCQUFzQixDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQzlDLElBQUksQ0FBQyxZQUFZLENBQUMsc0JBQXNCLENBQUMsR0FBRyxLQUFLLENBQUM7WUFFbEQsZUFBZSxDQUFDLEtBQUssQ0FBQyxPQUFPLEdBQUcsRUFBRSxDQUFDO1lBRW5DLElBQU0sTUFBTSxHQUNSLElBQUksQ0FBQyxnQkFBZ0IsQ0FBQyxXQUFXLENBQUMsQ0FBQyxzQkFBc0IsQ0FDeEMsQ0FBQztZQUN0QixNQUFNLENBQUMsS0FBSyxDQUFDLE9BQU8sR0FBRyxPQUFPLENBQUM7WUFDL0IsS0FBSyxDQUFDLE1BQU0sRUFBRSxDQUFDO1lBRWYsSUFBTSxVQUFVLEdBQ1osSUFBSSxDQUFDLGdCQUFnQixDQUFDLGNBQWMsQ0FBQyxDQUFDLHNCQUFzQixDQUNqRCxDQUFDO1lBQ2hCLFVBQVUsQ0FBQyxLQUFLLENBQUMsT0FBTyxHQUFHLE1BQU0sQ0FBQztZQUVsQyxNQUFNLENBQUM7UUFDVCxDQUFDO1FBRUQsSUFBTSxtQkFBbUIsR0FBRyxRQUFRLENBQUMsYUFBYSxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQzFELG1CQUFtQixDQUFDLFNBQVMsR0FBRyxnQ0FBZ0MsQ0FBQztRQUVqRSxJQUFNLFNBQVMsR0FBYSxDQUFDLEVBQUUsR0FBRyxJQUFJLENBQUMsQ0FBQztRQUN4QyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLGlCQUFpQixDQUFDLGFBQWEsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztZQUNoRSxJQUFNLFlBQVksR0FBRyxpQkFBaUIsQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDeEQsSUFBTSxhQUFhLEdBQUcsWUFBWSxDQUFDLGFBQWEsQ0FBQztZQUVqRCxJQUFNLElBQUksR0FBRyxpQkFBaUIsQ0FBQyx3QkFBd0IsSUFBSSxJQUFJO2dCQUMzRCxpQkFBaUIsQ0FBQyx3QkFBd0IsQ0FBQyxJQUFJLENBQUM7Z0JBQ2hELElBQUksQ0FBQztZQUVULElBQUksWUFBWSxTQUFRLENBQUM7WUFDekIsSUFBSSxTQUFTLFNBQVEsQ0FBQztZQUN0QixJQUFJLElBQUksR0FBRyxDQUFDLENBQUM7WUFDYixJQUFJLE9BQU8sR0FBRyxJQUFJLENBQUM7WUFFbkIsSUFBSSxDQUFDO2dCQUNILElBQUksR0FBRyxhQUFhLENBQUMsSUFBSSxDQUFDLENBQUM7Z0JBQzNCLFlBQVksR0FBRyxJQUFJLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQztnQkFDdEMsU0FBUyxHQUFHLFlBQVksQ0FBQztZQUMzQixDQUFDO1lBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDWCxPQUFPLEdBQUcsS0FBSyxDQUFDO2dCQUNoQixZQUFZLEdBQUcsT0FBTyxDQUFDO2dCQUN2QixTQUFTLEdBQUcsQ0FBQyxDQUFDLE9BQU8sQ0FBQztZQUN4QixDQUFDO1lBRUQsRUFBRSxDQUFDLENBQUMsSUFBSSxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBQ2QsRUFBRSxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztvQkFDWixZQUFZLENBQUMsU0FBUyxDQUFDLElBQUksQ0FBQyxFQUFDLENBQUMsRUFBRSxJQUFJLEVBQUUsQ0FBQyxFQUFFLElBQUksRUFBQyxDQUFDLENBQUM7Z0JBQ2xELENBQUM7Z0JBQ0QsU0FBUyxDQUFDLElBQUksQ0FBQyxZQUFZLENBQUMsQ0FBQztZQUMvQixDQUFDO1lBQ0QsT0FBTyxDQUFDLEdBQUcsQ0FBQyxZQUFZLENBQUMsSUFBSSxHQUFHLEdBQUcsR0FBRyxJQUFJLEdBQUcsS0FBSyxHQUFHLFNBQVMsQ0FBQyxDQUFDO1FBQ2xFLENBQUM7UUFDRCxlQUFlLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxrQkFBa0IsQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDO1FBRWhFLElBQUksSUFBSSxpQkFBaUIsQ0FBQyxRQUFRLENBQUM7UUFFbkMsVUFBVSxDQUNOLGNBQU0sT0FBQSxLQUFJLENBQUMsaUJBQWlCLENBQ3hCLEtBQUssRUFBRSxpQkFBaUIsRUFBRSxzQkFBc0IsRUFBRSxJQUFJLENBQUMsRUFEckQsQ0FDcUQsRUFDM0QsR0FBRyxDQUFDLENBQUM7SUFDWCxDQUFDO0lBQ0gsb0JBQUM7QUFBRCxDQW5NQSxBQW1NQyxDQW5Na0MsNEJBQW9CLEdBbU10RDtBQW5NWSxzQ0FBYTtBQW9NMUIsUUFBUSxDQUFDLGVBQWUsQ0FBQyxhQUFhLENBQUMsU0FBUyxDQUFDLEVBQUUsRUFBRSxhQUFhLENBQUMsQ0FBQzs7Ozs7QUNqTnBFLG9EQUFzRDtBQUN0RCxvRUFBZ0U7QUFDaEUsa0ZBQW9GO0FBQ3BGLCtDQUFpRDtBQUNqRCxxQ0FBdUM7QUFJdkMsSUFBTSxPQUFPLEdBQUcsR0FBRyxDQUFDO0FBRVAsUUFBQSxjQUFjLEdBQWtCLFVBQUMsSUFBWTtJQUN4RCxJQUFNLFVBQVUsR0FBNkIsQ0FBQyxJQUFJLEVBQUUsSUFBSSxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQzdELElBQU0sV0FBVyxHQUFHLENBQUMsQ0FBQztJQUN0QixJQUFNLFNBQVMsR0FBRyxFQUFFLENBQUM7SUFDckIsSUFBTSxNQUFNLEdBQUcsQ0FBQyxDQUFDO0lBQ2pCLElBQU0sT0FBTyxHQUFHLFNBQVMsQ0FBQyxpQkFBaUIsQ0FBQyxVQUFVLEVBQUUsU0FBUyxFQUFFLE1BQU0sQ0FBQyxDQUFDO0lBQzNFLElBQU0sY0FBYyxHQUNoQixTQUFTLENBQUMsb0JBQW9CLENBQzFCLFVBQVUsRUFBRSxTQUFTLEVBQUUsV0FBVyxFQUFFLE1BQU0sRUFBRSxPQUFPLENBQUMsQ0FBQztJQUU3RCxJQUFNLFlBQVksR0FBRyxTQUFTLENBQUMscUJBQXFCLENBQUMsVUFBVSxDQUFDLENBQUM7SUFDakUsSUFBTSxnQkFBZ0IsR0FBRyxTQUFTLENBQUMscUJBQXFCLENBQUMsY0FBYyxDQUFDLENBQUM7SUFFekUsSUFBTSxLQUFLLEdBQUcsSUFBSSw0QkFBWSxFQUFFLENBQUM7SUFDakMsSUFBTSxPQUFPLEdBQUcsS0FBSyxDQUFDLGFBQWEsQ0FDL0IscUJBQXFCLENBQUMsZ0NBQWdDLENBQ2xELFVBQVUsRUFBRSxTQUFTLEVBQUUsTUFBTSxFQUFFLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFFakQsSUFBTSxTQUFTLEdBQUcsS0FBSyxDQUFDLG1CQUFtQixDQUFDLFlBQVksQ0FBQyxDQUFDLENBQUMsRUFBRSxZQUFZLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUM5RSxJQUFNLG1CQUFtQixHQUNyQixLQUFLLENBQUMsbUJBQW1CLENBQUMsWUFBWSxDQUFDLENBQUMsQ0FBQyxFQUFFLFlBQVksQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ2hFLElBQU0sYUFBYSxHQUNmLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBRXhFLElBQU0sTUFBTSxHQUNSLFNBQVMsQ0FBQyxrQkFBa0IsQ0FBQyxZQUFZLENBQUMsQ0FBQyxDQUFDLEdBQUcsWUFBWSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQzNFLElBQU0sZ0JBQWdCLEdBQUcsSUFBSSxZQUFZLENBQUMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxVQUFVLENBQUMsQ0FBQyxDQUFDO0lBQzFFLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsZ0JBQWdCLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7UUFDakQsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxFQUFFLEdBQUcsU0FBUyxHQUFHLFNBQVMsQ0FBQyxDQUFDO0lBQzFFLENBQUM7SUFFRCxLQUFLLENBQUMscUJBQXFCLENBQ3ZCLFNBQVMsRUFBRSxZQUFZLENBQUMsQ0FBQyxDQUFDLEVBQUUsWUFBWSxDQUFDLENBQUMsQ0FBQyxFQUFFLE1BQU0sQ0FBQyxDQUFDO0lBQ3pELEtBQUssQ0FBQyxxQkFBcUIsQ0FDdkIsbUJBQW1CLEVBQUUsWUFBWSxDQUFDLENBQUMsQ0FBQyxFQUFFLFlBQVksQ0FBQyxDQUFDLENBQUMsRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDO0lBRTdFLElBQU0sS0FBSyxHQUFHLFdBQVcsQ0FBQyxHQUFHLEVBQUUsQ0FBQztJQUNoQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE9BQU8sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1FBQ2pDLHFCQUFxQixDQUFDLGVBQWUsQ0FDakMsS0FBSyxFQUFFLE9BQU8sRUFBRSxTQUFTLEVBQUUsbUJBQW1CLEVBQUUsYUFBYSxFQUM3RCxnQkFBZ0IsQ0FBQyxDQUFDO0lBQ3hCLENBQUM7SUFFRCxLQUFLLENBQUMseUJBQXlCLENBQzNCLGFBQWEsRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzdELElBQU0sR0FBRyxHQUFHLFdBQVcsQ0FBQyxHQUFHLEVBQUUsQ0FBQztJQUU5QixJQUFNLE9BQU8sR0FBRyxDQUFDLEdBQUcsR0FBRyxLQUFLLENBQUMsR0FBRyxPQUFPLENBQUM7SUFFeEMsS0FBSyxDQUFDLG1CQUFtQixDQUFDLFNBQVMsQ0FBQyxDQUFDO0lBQ3JDLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxtQkFBbUIsQ0FBQyxDQUFDO0lBQy9DLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxhQUFhLENBQUMsQ0FBQztJQUN6QyxLQUFLLENBQUMsYUFBYSxDQUFDLE9BQU8sQ0FBQyxDQUFDO0lBQzdCLEtBQUssQ0FBQyxPQUFPLEVBQUUsQ0FBQztJQUVoQixNQUFNLENBQUMsT0FBTyxDQUFDO0FBQ2pCLENBQUMsQ0FBQzs7Ozs7QUNsRUYsb0RBQXNEO0FBQ3RELG9FQUFnRTtBQUNoRSxnRUFBa0U7QUFDbEUsK0NBQWlEO0FBSWpELElBQU0sT0FBTyxHQUFHLEdBQUcsQ0FBQztBQUVQLFFBQUEsdUJBQXVCLEdBQWtCLFVBQUMsSUFBWTtJQUNqRSxJQUFNLGFBQWEsR0FBNkIsQ0FBQyxJQUFJLEVBQUUsSUFBSSxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQ2hFLElBQU0sV0FBVyxHQUFHLENBQUMsQ0FBQztJQUN0QixJQUFNLFNBQVMsR0FBRyxFQUFFLENBQUM7SUFDckIsSUFBTSxNQUFNLEdBQUcsQ0FBQyxDQUFDO0lBQ2pCLElBQU0sT0FBTyxHQUFHLFNBQVMsQ0FBQyxpQkFBaUIsQ0FBQyxhQUFhLEVBQUUsU0FBUyxFQUFFLE1BQU0sQ0FBQyxDQUFDO0lBQzlFLElBQU0sY0FBYyxHQUNoQixTQUFTLENBQUMsb0JBQW9CLENBQzFCLGFBQWEsRUFBRSxTQUFTLEVBQUUsV0FBVyxFQUFFLE1BQU0sRUFBRSxPQUFPLENBQUMsQ0FBQztJQUVoRSxJQUFNLGVBQWUsR0FBRyxTQUFTLENBQUMscUJBQXFCLENBQUMsYUFBYSxDQUFDLENBQUM7SUFDdkUsSUFBTSxnQkFBZ0IsR0FBRyxTQUFTLENBQUMscUJBQXFCLENBQUMsY0FBYyxDQUFDLENBQUM7SUFFekUsSUFBTSxLQUFLLEdBQUcsSUFBSSw0QkFBWSxFQUFFLENBQUM7SUFDakMsSUFBTSxPQUFPLEdBQ1QsS0FBSyxDQUFDLGFBQWEsQ0FBQyxZQUFZLENBQUMsOEJBQThCLENBQzNELGFBQWEsRUFBRSxTQUFTLEVBQUUsTUFBTSxFQUFFLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFFcEQsSUFBTSxZQUFZLEdBQ2QsS0FBSyxDQUFDLG1CQUFtQixDQUFDLGVBQWUsQ0FBQyxDQUFDLENBQUMsRUFBRSxlQUFlLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUN0RSxJQUFNLGFBQWEsR0FDZixLQUFLLENBQUMsbUJBQW1CLENBQUMsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLEVBQUUsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUV4RSxJQUFNLFNBQVMsR0FBRyxTQUFTLENBQUMsa0JBQWtCLENBQzFDLGVBQWUsQ0FBQyxDQUFDLENBQUMsR0FBRyxlQUFlLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFFcEQsS0FBSyxDQUFDLHFCQUFxQixDQUN2QixZQUFZLEVBQUUsZUFBZSxDQUFDLENBQUMsQ0FBQyxFQUFFLGVBQWUsQ0FBQyxDQUFDLENBQUMsRUFBRSxTQUFTLENBQUMsQ0FBQztJQUVyRSxJQUFNLEtBQUssR0FBRyxXQUFXLENBQUMsR0FBRyxFQUFFLENBQUM7SUFDaEMsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxPQUFPLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztRQUNqQyxZQUFZLENBQUMsYUFBYSxDQUN0QixLQUFLLEVBQUUsT0FBTyxFQUFFLFlBQVksRUFBRSxhQUFhLEVBQUUsZ0JBQWdCLENBQUMsQ0FBQztJQUNyRSxDQUFDO0lBRUQsS0FBSyxDQUFDLHlCQUF5QixDQUMzQixhQUFhLEVBQUUsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLEVBQUUsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUM3RCxJQUFNLEdBQUcsR0FBRyxXQUFXLENBQUMsR0FBRyxFQUFFLENBQUM7SUFFOUIsSUFBTSxPQUFPLEdBQUcsQ0FBQyxHQUFHLEdBQUcsS0FBSyxDQUFDLEdBQUcsT0FBTyxDQUFDO0lBRXhDLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxZQUFZLENBQUMsQ0FBQztJQUN4QyxLQUFLLENBQUMsbUJBQW1CLENBQUMsYUFBYSxDQUFDLENBQUM7SUFDekMsS0FBSyxDQUFDLGFBQWEsQ0FBQyxPQUFPLENBQUMsQ0FBQztJQUM3QixLQUFLLENBQUMsT0FBTyxFQUFFLENBQUM7SUFFaEIsTUFBTSxDQUFDLE9BQU8sQ0FBQztBQUNqQixDQUFDLENBQUM7QUFFVyxRQUFBLDZCQUE2QixHQUFrQixVQUFDLElBQVk7SUFDdkUsSUFBTSxhQUFhLEdBQTZCLENBQUMsSUFBSSxFQUFFLElBQUksRUFBRSxDQUFDLENBQUMsQ0FBQztJQUNoRSxJQUFNLFdBQVcsR0FBRyxDQUFDLENBQUM7SUFDdEIsSUFBTSxTQUFTLEdBQUcsRUFBRSxDQUFDO0lBQ3JCLElBQU0sTUFBTSxHQUFHLENBQUMsQ0FBQztJQUNqQixJQUFNLE9BQU8sR0FBRyxTQUFTLENBQUMsaUJBQWlCLENBQUMsYUFBYSxFQUFFLFNBQVMsRUFBRSxNQUFNLENBQUMsQ0FBQztJQUM5RSxJQUFNLGNBQWMsR0FDaEIsU0FBUyxDQUFDLG9CQUFvQixDQUMxQixhQUFhLEVBQUUsU0FBUyxFQUFFLFdBQVcsRUFBRSxNQUFNLEVBQUUsT0FBTyxDQUFDLENBQUM7SUFFaEUsSUFBTSxlQUFlLEdBQUcsU0FBUyxDQUFDLHFCQUFxQixDQUFDLGFBQWEsQ0FBQyxDQUFDO0lBQ3ZFLElBQU0sZ0JBQWdCLEdBQUcsU0FBUyxDQUFDLHFCQUFxQixDQUFDLGNBQWMsQ0FBQyxDQUFDO0lBRXpFLElBQU0sS0FBSyxHQUFHLElBQUksNEJBQVksRUFBRSxDQUFDO0lBQ2pDLElBQU0sT0FBTyxHQUNULEtBQUssQ0FBQyxhQUFhLENBQUMsWUFBWSxDQUFDLHVDQUF1QyxDQUNwRSxhQUFhLEVBQUUsU0FBUyxFQUFFLE1BQU0sRUFBRSxPQUFPLENBQUMsQ0FBQyxDQUFDO0lBRXBELElBQU0sWUFBWSxHQUNkLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxlQUFlLENBQUMsQ0FBQyxDQUFDLEVBQUUsZUFBZSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDdEUsSUFBTSxhQUFhLEdBQ2YsS0FBSyxDQUFDLG1CQUFtQixDQUFDLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxFQUFFLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFFeEUsSUFBTSxTQUFTLEdBQUcsU0FBUyxDQUFDLGtCQUFrQixDQUMxQyxlQUFlLENBQUMsQ0FBQyxDQUFDLEdBQUcsZUFBZSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBRXBELEtBQUssQ0FBQyxxQkFBcUIsQ0FDdkIsWUFBWSxFQUFFLGVBQWUsQ0FBQyxDQUFDLENBQUMsRUFBRSxlQUFlLENBQUMsQ0FBQyxDQUFDLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFFckUsSUFBTSxLQUFLLEdBQUcsV0FBVyxDQUFDLEdBQUcsRUFBRSxDQUFDO0lBQ2hDLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsT0FBTyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7UUFDakMsWUFBWSxDQUFDLGFBQWEsQ0FDdEIsS0FBSyxFQUFFLE9BQU8sRUFBRSxZQUFZLEVBQUUsYUFBYSxFQUFFLGdCQUFnQixDQUFDLENBQUM7SUFDckUsQ0FBQztJQUVELEtBQUssQ0FBQyx5QkFBeUIsQ0FDM0IsYUFBYSxFQUFFLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxFQUFFLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDN0QsSUFBTSxHQUFHLEdBQUcsV0FBVyxDQUFDLEdBQUcsRUFBRSxDQUFDO0lBRTlCLElBQU0sT0FBTyxHQUFHLENBQUMsR0FBRyxHQUFHLEtBQUssQ0FBQyxHQUFHLE9BQU8sQ0FBQztJQUV4QyxLQUFLLENBQUMsbUJBQW1CLENBQUMsWUFBWSxDQUFDLENBQUM7SUFDeEMsS0FBSyxDQUFDLG1CQUFtQixDQUFDLGFBQWEsQ0FBQyxDQUFDO0lBQ3pDLEtBQUssQ0FBQyxhQUFhLENBQUMsT0FBTyxDQUFDLENBQUM7SUFDN0IsS0FBSyxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBRWhCLE1BQU0sQ0FBQyxPQUFPLENBQUM7QUFDakIsQ0FBQyxDQUFDOzs7OztBQ3pHRixvREFBdUQ7QUFDdkQsa0RBQXdEO0FBSXhELElBQU0saUJBQWlCLEdBQUcsRUFBRSxDQUFDO0FBRWhCLFFBQUEsY0FBYyxHQUFrQixVQUFDLElBQVk7SUFDeEQsRUFBRSxDQUFDLENBQUMsSUFBSSxHQUFHLEdBQUcsQ0FBQyxDQUFDLENBQUM7UUFDZixNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDWixDQUFDO0lBQ0QsSUFBTSxJQUFJLEdBQUcsSUFBSSx5QkFBYyxFQUFFLENBQUM7SUFDbEMsSUFBTSxDQUFDLEdBQUcsaUJBQU8sQ0FBQyxXQUFXLENBQVUsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDNUQsSUFBTSxDQUFDLEdBQUcsaUJBQU8sQ0FBQyxXQUFXLENBQVUsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDNUQsSUFBTSxJQUFJLEdBQUcsQ0FBQyxJQUFJLEdBQUcsR0FBRyxDQUFDLEdBQUcsaUJBQWlCLEdBQUcsQ0FBQyxDQUFDO0lBQ2xELElBQU0sS0FBSyxHQUFHLFdBQVcsQ0FBQyxHQUFHLEVBQUUsQ0FBQztJQUNoQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLElBQUksRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1FBQzlCLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQ3BCLENBQUM7SUFDRCxJQUFNLEdBQUcsR0FBRyxXQUFXLENBQUMsR0FBRyxFQUFFLENBQUM7SUFDOUIsTUFBTSxDQUFDLENBQUMsR0FBRyxHQUFHLEtBQUssQ0FBQyxHQUFHLElBQUksQ0FBQztBQUM5QixDQUFDLENBQUM7Ozs7O0FDckJGLDRDQUFzRDtBQUN0RCxrREFBK0M7QUFDL0Msb0VBQWdFO0FBQ2hFLDREQUE4RDtBQUM5RCwwRUFBNEU7QUFDNUUsK0NBQWlEO0FBSWpELElBQU0sT0FBTyxHQUFHLEdBQUcsQ0FBQztBQUVQLFFBQUEsY0FBYyxHQUFrQixVQUFDLElBQVk7SUFDeEQsSUFBTSxLQUFLLEdBQUcsSUFBSSw0QkFBWSxFQUFFLENBQUM7SUFDakMsSUFBTSxRQUFRLEdBQUcsS0FBSyxDQUFDLG1CQUFtQixDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsQ0FBQztJQUN2RCxJQUFNLFFBQVEsR0FBRyxLQUFLLENBQUMsbUJBQW1CLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQ3ZELElBQU0sYUFBYSxHQUFHLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFFNUQsSUFBTSxJQUFJLEdBQUcsSUFBSSxpQkFBTyxDQUNwQixDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsRUFBRSxFQUFDLE9BQU8sRUFBRSxRQUFRLEVBQUUsY0FBYyxFQUFFLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxFQUFDLENBQUMsQ0FBQztJQUNyRSxJQUFNLElBQUksR0FBRyxJQUFJLGlCQUFPLENBQ3BCLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxFQUFFLEVBQUMsT0FBTyxFQUFFLFFBQVEsRUFBRSxjQUFjLEVBQUUsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLEVBQUMsQ0FBQyxDQUFDO0lBQ3JFLElBQU0sTUFBTSxHQUFHLElBQUksaUJBQU8sQ0FDdEIsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLEVBQUUsRUFBQyxPQUFPLEVBQUUsYUFBYSxFQUFFLGNBQWMsRUFBRSxDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsRUFBQyxDQUFDLENBQUM7SUFDMUUsSUFBTSxPQUFPLEdBQUcsS0FBSyxDQUFDLGFBQWEsQ0FBQyxVQUFVLENBQUMsaUJBQWlCLENBQzVELElBQUksRUFBRSxJQUFJLEVBQUUsTUFBTSxFQUFFLHdCQUFpQixDQUFDLE9BQU8sRUFDN0Msd0JBQWlCLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUVoQyxJQUFNLENBQUMsR0FBRyxTQUFTLENBQUMsa0JBQWtCLENBQUMsSUFBSSxHQUFHLElBQUksRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUMzRCxJQUFNLENBQUMsR0FBRyxTQUFTLENBQUMsa0JBQWtCLENBQUMsSUFBSSxHQUFHLElBQUksRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUMzRCxLQUFLLENBQUMscUJBQXFCLENBQUMsUUFBUSxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDckQsS0FBSyxDQUFDLHFCQUFxQixDQUFDLFFBQVEsRUFBRSxJQUFJLEVBQUUsSUFBSSxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBRXJELElBQU0sS0FBSyxHQUFHLFdBQVcsQ0FBQyxHQUFHLEVBQUUsQ0FBQztJQUNoQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE9BQU8sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1FBQ2pDLFVBQVUsQ0FBQyxjQUFjLENBQ3JCLEtBQUssRUFBRSxPQUFPLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxhQUFhLEVBQUUsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQztJQUN2RSxDQUFDO0lBRUQsSUFBTSxNQUFNLEdBQUcsS0FBSyxDQUFDLHlCQUF5QixDQUFDLGFBQWEsRUFBRSxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDMUUsSUFBTSxPQUFPLEdBQUcsQ0FBQyxXQUFXLENBQUMsR0FBRyxFQUFFLEdBQUcsS0FBSyxDQUFDLEdBQUcsT0FBTyxDQUFDO0lBRXRELEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxRQUFRLENBQUMsQ0FBQztJQUNwQyxLQUFLLENBQUMsbUJBQW1CLENBQUMsUUFBUSxDQUFDLENBQUM7SUFDcEMsS0FBSyxDQUFDLG1CQUFtQixDQUFDLGFBQWEsQ0FBQyxDQUFDO0lBQ3pDLEtBQUssQ0FBQyxhQUFhLENBQUMsT0FBTyxDQUFDLENBQUM7SUFDN0IsS0FBSyxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBRWhCLElBQU0sUUFBUSxHQUFHLFNBQVMsQ0FBQyxpQkFBaUIsQ0FBQyxDQUFDLEVBQUUsSUFBSSxFQUFFLElBQUksRUFBRSxDQUFDLEVBQUUsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQzNFLFNBQVMsQ0FBQyxpQkFBaUIsQ0FBQyxNQUFNLEVBQUUsUUFBUSxFQUFFLEtBQUssQ0FBQyxDQUFDO0lBQ3JELE1BQU0sQ0FBQyxPQUFPLENBQUM7QUFDakIsQ0FBQyxDQUFDO0FBRVcsUUFBQSxxQkFBcUIsR0FBa0IsVUFBQyxJQUFZO0lBQy9ELElBQU0sS0FBSyxHQUFHLElBQUksNEJBQVksRUFBRSxDQUFDO0lBQ2pDLElBQU0sT0FBTyxHQUNULEtBQUssQ0FBQyxhQUFhLENBQUMsaUJBQWlCLENBQUMsdUJBQXVCLENBQ3pELElBQUksRUFBRSx3QkFBaUIsQ0FBQyxPQUFPLEVBQUUsd0JBQWlCLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUVyRSxJQUFNLFFBQVEsR0FBRyxLQUFLLENBQUMseUJBQXlCLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQzdELElBQU0sUUFBUSxHQUFHLEtBQUssQ0FBQyx5QkFBeUIsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDN0QsSUFBTSxhQUFhLEdBQUcsS0FBSyxDQUFDLHlCQUF5QixDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsQ0FBQztJQUVsRSxJQUFNLENBQUMsR0FBRyxTQUFTLENBQUMsa0JBQWtCLENBQUMsSUFBSSxHQUFHLElBQUksRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUMzRCxJQUFNLENBQUMsR0FBRyxTQUFTLENBQUMsa0JBQWtCLENBQUMsSUFBSSxHQUFHLElBQUksRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUMzRCxLQUFLLENBQUMsMkJBQTJCLENBQUMsUUFBUSxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDM0QsS0FBSyxDQUFDLDJCQUEyQixDQUFDLFFBQVEsRUFBRSxJQUFJLEVBQUUsSUFBSSxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBRTNELElBQU0sS0FBSyxHQUFHLFdBQVcsQ0FBQyxHQUFHLEVBQUUsQ0FBQztJQUNoQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE9BQU8sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1FBQ2pDLGlCQUFpQixDQUFDLG9CQUFvQixDQUNsQyxLQUFLLEVBQUUsT0FBTyxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsYUFBYSxFQUFFLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDLENBQUM7SUFDdkUsQ0FBQztJQUVELElBQU0sTUFBTSxHQUNSLEtBQUssQ0FBQywrQkFBK0IsQ0FBQyxhQUFhLEVBQUUsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQ3JFLElBQU0sT0FBTyxHQUFHLENBQUMsV0FBVyxDQUFDLEdBQUcsRUFBRSxHQUFHLEtBQUssQ0FBQyxHQUFHLE9BQU8sQ0FBQztJQUV0RCxLQUFLLENBQUMsbUJBQW1CLENBQUMsUUFBUSxDQUFDLENBQUM7SUFDcEMsS0FBSyxDQUFDLG1CQUFtQixDQUFDLFFBQVEsQ0FBQyxDQUFDO0lBQ3BDLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxhQUFhLENBQUMsQ0FBQztJQUN6QyxLQUFLLENBQUMsYUFBYSxDQUFDLE9BQU8sQ0FBQyxDQUFDO0lBQzdCLEtBQUssQ0FBQyxPQUFPLEVBQUUsQ0FBQztJQUVoQixJQUFNLFFBQVEsR0FBRyxTQUFTLENBQUMsaUJBQWlCLENBQUMsQ0FBQyxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUUsQ0FBQyxFQUFFLElBQUksRUFBRSxJQUFJLENBQUMsQ0FBQztJQUMzRSxTQUFTLENBQUMsaUJBQWlCLENBQUMsTUFBTSxFQUFFLFFBQVEsRUFBRSxLQUFLLENBQUMsQ0FBQztJQUNyRCxNQUFNLENBQUMsT0FBTyxDQUFDO0FBQ2pCLENBQUMsQ0FBQzs7Ozs7QUNyRkYsd0RBQTBEO0FBQzFELDREQUE4RDtBQUM5RCwrQ0FBaUQ7QUFJakQsSUFBTSxXQUFXLEdBQUcsR0FBRyxDQUFDO0FBRVgsUUFBQSx5QkFBeUIsR0FBa0IsVUFBQyxJQUFZO0lBQ25FLElBQU0sTUFBTSxHQUFHLFNBQVMsQ0FBQyxrQkFBa0IsQ0FBQyxJQUFJLEdBQUcsSUFBSSxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQ2hFLElBQU0sa0JBQWtCLEdBQUcsVUFBVSxDQUFDLHFCQUFxQixFQUFFLENBQUM7SUFDOUQsSUFBTSxhQUFhLEdBQ2YsSUFBSSxZQUFZLENBQUMsUUFBUSxDQUFDLGtDQUFrQyxDQUN4RCxNQUFNLENBQUMsTUFBTSxFQUFFLGtCQUFrQixDQUFDLENBQUMsQ0FBQztJQUM1QyxJQUFNLEtBQUssR0FBRyxXQUFXLENBQUMsR0FBRyxFQUFFLENBQUM7SUFDaEMsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxXQUFXLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQztRQUNyQyxRQUFRLENBQUMsMkJBQTJCLENBQ2hDLE1BQU0sRUFBRSxhQUFhLEVBQUUsa0JBQWtCLENBQUMsQ0FBQztJQUNqRCxDQUFDO0lBQ0QsSUFBTSxHQUFHLEdBQUcsV0FBVyxDQUFDLEdBQUcsRUFBRSxDQUFDO0lBQzlCLE1BQU0sQ0FBQyxDQUFDLEdBQUcsR0FBRyxLQUFLLENBQUMsR0FBRyxXQUFXLENBQUM7QUFDckMsQ0FBQyxDQUFDO0FBRVcsUUFBQSx1QkFBdUIsR0FBa0IsVUFBQyxJQUFZO0lBQ2pFLElBQU0sTUFBTSxHQUFHLFNBQVMsQ0FBQyxrQkFBa0IsQ0FBQyxJQUFJLEdBQUcsSUFBSSxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQ2hFLElBQU0sVUFBVSxHQUFHLElBQUksWUFBWSxDQUMvQixRQUFRLENBQUMscUNBQXFDLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDLENBQUM7SUFDaEUsSUFBTSxLQUFLLEdBQUcsV0FBVyxDQUFDLEdBQUcsRUFBRSxDQUFDO0lBQ2hDLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsV0FBVyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUM7UUFDckMsUUFBUSxDQUFDLHdCQUF3QixDQUFDLE1BQU0sRUFBRSxJQUFJLEVBQUUsSUFBSSxFQUFFLFVBQVUsQ0FBQyxDQUFDO0lBQ3BFLENBQUM7SUFDRCxJQUFNLEdBQUcsR0FBRyxXQUFXLENBQUMsR0FBRyxFQUFFLENBQUM7SUFDOUIsTUFBTSxDQUFDLENBQUMsR0FBRyxHQUFHLEtBQUssQ0FBQyxHQUFHLFdBQVcsQ0FBQztBQUNyQyxDQUFDLENBQUM7QUFFVyxRQUFBLHlCQUF5QixHQUFrQixVQUFDLElBQVk7SUFDbkUsSUFBTSxNQUFNLEdBQUcsU0FBUyxDQUFDLGtCQUFrQixDQUFDLElBQUksR0FBRyxJQUFJLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDaEUsSUFBTSxrQkFBa0IsR0FBRyxVQUFVLENBQUMscUJBQXFCLEVBQUUsQ0FBQztJQUM5RCxJQUFNLGFBQWEsR0FDZixJQUFJLFlBQVksQ0FBQyxRQUFRLENBQUMsa0NBQWtDLENBQ3hELE1BQU0sQ0FBQyxNQUFNLEVBQUUsa0JBQWtCLENBQUMsQ0FBQyxDQUFDO0lBQzVDLFFBQVEsQ0FBQywyQkFBMkIsQ0FDaEMsTUFBTSxFQUFFLGFBQWEsRUFBRSxrQkFBa0IsQ0FBQyxDQUFDO0lBQy9DLElBQU0sS0FBSyxHQUFHLFdBQVcsQ0FBQyxHQUFHLEVBQUUsQ0FBQztJQUNoQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLFdBQVcsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDO1FBQ3JDLFFBQVEsQ0FBQyw2QkFBNkIsQ0FDbEMsYUFBYSxFQUFFLE1BQU0sRUFBRSxrQkFBa0IsQ0FBQyxDQUFDO0lBQ2pELENBQUM7SUFDRCxJQUFNLEdBQUcsR0FBRyxXQUFXLENBQUMsR0FBRyxFQUFFLENBQUM7SUFDOUIsTUFBTSxDQUFDLENBQUMsR0FBRyxHQUFHLEtBQUssQ0FBQyxHQUFHLFdBQVcsQ0FBQztBQUNyQyxDQUFDLENBQUM7QUFFVyxRQUFBLHVCQUF1QixHQUFrQixVQUFDLElBQVk7SUFDakUsSUFBTSxNQUFNLEdBQUcsU0FBUyxDQUFDLGtCQUFrQixDQUFDLElBQUksR0FBRyxJQUFJLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDaEUsSUFBTSxVQUFVLEdBQUcsSUFBSSxZQUFZLENBQy9CLFFBQVEsQ0FBQyxxQ0FBcUMsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQztJQUNoRSxRQUFRLENBQUMsd0JBQXdCLENBQUMsTUFBTSxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUUsVUFBVSxDQUFDLENBQUM7SUFDbEUsSUFBTSxLQUFLLEdBQUcsV0FBVyxDQUFDLEdBQUcsRUFBRSxDQUFDO0lBQ2hDLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsV0FBVyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUM7UUFDckMsUUFBUSxDQUFDLDBCQUEwQixDQUFDLFVBQVUsRUFBRSxJQUFJLEVBQUUsSUFBSSxFQUFFLE1BQU0sQ0FBQyxDQUFDO0lBQ3RFLENBQUM7SUFDRCxJQUFNLEdBQUcsR0FBRyxXQUFXLENBQUMsR0FBRyxFQUFFLENBQUM7SUFDOUIsTUFBTSxDQUFDLENBQUMsR0FBRyxHQUFHLEtBQUssQ0FBQyxHQUFHLFdBQVcsQ0FBQztBQUNyQyxDQUFDLENBQUM7OztBQ2pFRixPQUFPLENBQUMsRUFBQyxFQUFFLEVBQUUsYUFBYSxFQUFDLENBQUMsQ0FBQzs7O0FDQTdCLE9BQU8sQ0FBQyxFQUFDLEVBQUUsRUFBRSxhQUFhLEVBQUMsQ0FBQyxDQUFDOzs7OztBQzRDN0Isd0JBQStCLElBQVU7SUFFdkMsTUFBTSxDQUFDLE9BQU8sQ0FBQyxLQUFLLENBQUMsSUFBVyxDQUFpQyxDQUFDO0FBQ3BFLENBQUM7QUFIRCx3Q0FHQzs7Ozs7QUM5Q0QsOEJBQWdDO0FBRWhDLG1DQUNJLE9BQWlCLEVBQUUsT0FBaUIsRUFBRSxJQUFZLEVBQ2xELGtCQUF1QjtJQUF2QixtQ0FBQSxFQUFBLHVCQUF1QjtJQUN6QixJQUFJLENBQUMsTUFBTSxDQUNQLE9BQU8sQ0FBQyxNQUFNLEtBQUssQ0FBQyxFQUNwQixrQkFBa0IsR0FBRyx3Q0FBd0MsQ0FBQyxDQUFDO0lBQ25FLElBQUksQ0FBQyxNQUFNLENBQ1AsT0FBTyxDQUFDLE1BQU0sS0FBSyxDQUFDLEVBQ3BCLGtCQUFrQixHQUFHLHdDQUF3QyxDQUFDLENBQUM7SUFFbkUsSUFBSSxDQUFDLE1BQU0sQ0FDUCxJQUFJLElBQUksQ0FBQyxJQUFJLElBQUksR0FBRyxDQUFDLEVBQUUsNENBQTRDLENBQUMsQ0FBQztJQUV6RSxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1FBQzNCLElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLEtBQUssSUFBSSxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLEtBQUssT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQzNDLGtCQUFrQjthQUNkLFlBQVUsT0FBTywwQkFBcUIsT0FBTyxhQUFVLENBQUE7WUFDdkQsd0JBQXdCLENBQUMsQ0FBQztJQUNwQyxDQUFDO0FBQ0gsQ0FBQztBQXBCRCw4REFvQkM7QUFFRCxvQ0FDSSxPQUFpQixFQUFFLE9BQWlCLEVBQ3BDLElBQVk7SUFDZCxJQUFJLENBQUMsTUFBTSxDQUFDLE9BQU8sQ0FBQyxNQUFNLEtBQUssQ0FBQyxFQUFFLHdDQUF3QyxDQUFDLENBQUM7SUFDNUUsSUFBSSxDQUFDLE1BQU0sQ0FBQyxPQUFPLENBQUMsTUFBTSxLQUFLLENBQUMsRUFBRSx1Q0FBdUMsQ0FBQyxDQUFDO0lBRTNFLElBQU0sV0FBVyxHQUFHLE9BQU8sQ0FBQyxLQUFLLEVBQUUsQ0FBQztJQUNwQyxXQUFXLENBQUMsSUFBSSxDQUFDLElBQUksT0FBTyxDQUFDLElBQUksQ0FBQyxDQUFDO0lBQ25DLE1BQU0sQ0FBQyxXQUF1QyxDQUFDO0FBQ2pELENBQUM7QUFURCxnRUFTQzs7Ozs7QUNqQ0QsOEJBQWdDO0FBRWhDLDhCQUNJLHFCQUErQyxFQUFFLFNBQWlCLEVBQ2xFLEtBQWEsRUFBRSxNQUFjLEVBQUUsT0FBZ0I7SUFDakQsRUFBRSxDQUFDLENBQUMsT0FBTyxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUM7UUFDcEIsT0FBTyxHQUFHLGlCQUFpQixDQUFDLHFCQUFxQixFQUFFLFNBQVMsRUFBRSxNQUFNLENBQUMsQ0FBQztJQUN4RSxDQUFDO0lBQ0QsSUFBTSxTQUFTLEdBQUcscUJBQXFCLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDM0MsSUFBTSxTQUFTLEdBQUcscUJBQXFCLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDM0MsSUFBTSxVQUFVLEdBQUcsQ0FBQyxTQUFTLEdBQUcsU0FBUyxHQUFHLENBQUMsR0FBRyxPQUFPLENBQUMsR0FBRyxNQUFNLEdBQUcsQ0FBQyxDQUFDO0lBQ3RFLElBQUksQ0FBQyxNQUFNLENBQ1AsSUFBSSxDQUFDLEtBQUssQ0FBQyxVQUFVLENBQUMsRUFDdEIsMkJBQXlCLFVBQVUsc0NBQW1DO1FBQ2xFLG1DQUFtQyxDQUFDLENBQUM7SUFFN0MsSUFBTSxVQUFVLEdBQUcsQ0FBQyxTQUFTLEdBQUcsU0FBUyxHQUFHLENBQUMsR0FBRyxPQUFPLENBQUMsR0FBRyxNQUFNLEdBQUcsQ0FBQyxDQUFDO0lBQ3RFLElBQUksQ0FBQyxNQUFNLENBQ1AsSUFBSSxDQUFDLEtBQUssQ0FBQyxVQUFVLENBQUMsRUFDdEIsOEJBQTRCLFVBQVUsa0NBQStCO1FBQ2pFLHVDQUF1QyxDQUFDLENBQUM7SUFFakQsTUFBTSxDQUFDLENBQUMsVUFBVSxFQUFFLFVBQVUsRUFBRSxLQUFLLENBQUMsQ0FBQztBQUN6QyxDQUFDO0FBckJELG9EQXFCQztBQUVELDJCQUNJLFVBQW9DLEVBQUUsU0FBaUIsRUFDdkQsTUFBYztJQUNoQixNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDLFVBQVUsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUMsR0FBRyxNQUFNLEdBQUcsU0FBUyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUM7QUFDN0UsQ0FBQztBQUpELDhDQUlDO0FBRUQsK0JBQ0ksZ0JBQTBDO0lBQzVDLE1BQU0sQ0FBQyxDQUFDLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxFQUFFLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxHQUFHLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7QUFDMUUsQ0FBQztBQUhELHNEQUdDO0FBRUQsK0JBQ0ksVUFBa0IsRUFBRSxXQUFtQixFQUN2QyxLQUFhO0lBQ2YsTUFBTSxDQUFDLENBQUMsS0FBSyxFQUFFLEtBQUssRUFBRSxVQUFVLEVBQUUsV0FBVyxDQUFDLENBQUM7QUFDakQsQ0FBQztBQUpELHNEQUlDO0FBRUQsZ0NBQ0ksVUFBa0IsRUFBRSxXQUFtQixFQUN2QyxTQUFpQjtJQUNuQixNQUFNLENBQUMsQ0FBQyxTQUFTLEdBQUcsU0FBUyxHQUFHLFVBQVUsRUFBRSxXQUFXLENBQUMsQ0FBQztBQUMzRCxDQUFDO0FBSkQsd0RBSUM7QUFFRCwrQkFBc0MsV0FBbUI7SUFDdkQsTUFBTSxDQUFDLENBQUMsQ0FBQyxFQUFFLFdBQVcsQ0FBQyxDQUFDO0FBQzFCLENBQUM7QUFGRCxzREFFQztBQUVELDBCQUNJLEVBQW9CLEVBQUUsVUFBa0I7SUFDMUMsSUFBTSxXQUFXLEdBQUcsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsVUFBVSxHQUFHLENBQUMsQ0FBQztJQUNqRCxJQUFNLFdBQVcsR0FBRyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsR0FBRyxVQUFVLEdBQUcsQ0FBQyxDQUFDO0lBQ2pELE1BQU0sQ0FBQyxDQUFDLFdBQVcsRUFBRSxXQUFXLENBQUMsQ0FBQztBQUNwQyxDQUFDO0FBTEQsNENBS0M7Ozs7O0FDekRELHdCQUNJLFVBQTRCLEVBQUUsUUFBMEI7SUFDMUQsSUFBTSxPQUFPLEdBQUcsVUFBVSxDQUFDLENBQUMsQ0FBQyxHQUFHLFVBQVUsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUM5QyxJQUFNLE9BQU8sR0FBRyxRQUFRLENBQUMsQ0FBQyxDQUFDLEdBQUcsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzFDLEVBQUUsQ0FBQyxDQUFDLE9BQU8sS0FBSyxPQUFPLENBQUMsQ0FBQyxDQUFDO1FBQ3hCLElBQU0sTUFBTSxHQUFHLEdBQUcsR0FBRyxVQUFVLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxHQUFHLFVBQVUsQ0FBQyxDQUFDLENBQUMsR0FBRyxHQUFHLENBQUM7UUFDaEUsSUFBTSxNQUFNLEdBQUcsR0FBRyxHQUFHLFFBQVEsQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLEdBQUcsUUFBUSxDQUFDLENBQUMsQ0FBQyxHQUFHLEdBQUcsQ0FBQztRQUM1RCxNQUFNLElBQUksS0FBSyxDQUNYLG9EQUFvRCxHQUFHLE1BQU07WUFDN0QsU0FBUyxHQUFHLE9BQU8sR0FBRyxlQUFlLEdBQUcsTUFBTSxHQUFHLFNBQVMsR0FBRyxPQUFPLENBQUMsQ0FBQztJQUM1RSxDQUFDO0FBQ0gsQ0FBQztBQVhELHdDQVdDOzs7OztBQ1hELDhCQUFnQztBQUNoQywrQ0FBaUQ7QUFDakQsMkNBQTZDO0FBRTdDLHFDQUE4RTtBQUk5RTtJQVdFLHFCQUFvQixRQUFpQjtRQUFqQixhQUFRLEdBQVIsUUFBUSxDQUFTO1FBVjdCLGtCQUFhLEdBQWdCLEVBQUUsQ0FBQztRQUdoQyxtQkFBYyxHQUFnQixFQUFFLENBQUM7UUFDakMsOEJBQXlCLEdBQWMsRUFBRSxDQUFDO0lBTVYsQ0FBQztJQVV6QywyQkFBSyxHQUFMLFVBQ0ksT0FFeUQ7UUFIN0QsaUJBYUM7UUFUQyxJQUFJLENBQUMsVUFBVSxFQUFFLENBQUM7UUFFbEIsSUFBTSxNQUFNLEdBQUcsVUFBb0IsT0FBVSxJQUFRLE9BQUEsS0FBSSxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsRUFBbEIsQ0FBa0IsQ0FBQztRQUN4RSxJQUFNLE9BQU8sR0FBRyxVQUFvQixPQUFVLElBQVEsT0FBQSxLQUFJLENBQUMsS0FBSyxDQUFDLE9BQU8sQ0FBQyxFQUFuQixDQUFtQixDQUFDO1FBQzFFLElBQU0sTUFBTSxHQUFHLE9BQU8sQ0FBQyxNQUFNLEVBQUUsT0FBTyxDQUFDLENBQUM7UUFFeEMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUV0QixNQUFNLENBQUMsTUFBTSxDQUFDO0lBQ2hCLENBQUM7SUFNRCxnQ0FBVSxHQUFWO1FBQ0UsSUFBTSxRQUFRLEdBQWMsRUFBRSxDQUFDO1FBQy9CLElBQUksQ0FBQyxhQUFhLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDO1FBQ2xDLElBQUksQ0FBQyxXQUFXLEdBQUcsUUFBUSxDQUFDO1FBRTVCLElBQU0saUJBQWlCLEdBQWMsRUFBRSxDQUFDO1FBQ3hDLElBQUksQ0FBQyxjQUFjLENBQUMsSUFBSSxDQUFDLGlCQUFpQixDQUFDLENBQUM7UUFDNUMsSUFBSSxDQUFDLHlCQUF5QixHQUFHLGlCQUFpQixDQUFDO0lBQ3JELENBQUM7SUFNRCw4QkFBUSxHQUFSLFVBQVMsTUFBbUI7UUFBNUIsaUJBb0NDO1FBbENDLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSSxDQUFDLFdBQVcsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztZQUNqRCxJQUFNLE9BQU8sR0FBRyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBRXBDLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxtQkFBbUIsQ0FBQyxPQUFPLEVBQUUsSUFBSSxDQUFDLHlCQUF5QixDQUFDO2dCQUNqRSxDQUFDLE1BQU0sSUFBSSxJQUFJLElBQUksTUFBTSxZQUFZLGlCQUFPO29CQUMzQyxPQUFPLENBQUMsT0FBTyxFQUFFLEtBQU0sTUFBa0IsQ0FBQyxPQUFPLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDMUQsUUFBUSxDQUFDO1lBQ1gsQ0FBQztZQUNELE9BQU8sQ0FBQyxPQUFPLEVBQUUsQ0FBQztRQUNwQixDQUFDO1FBR0QsSUFBSSxDQUFDLGFBQWEsQ0FBQyxHQUFHLEVBQUUsQ0FBQztRQUN6QixJQUFJLENBQUMsV0FBVyxHQUFHLElBQUksQ0FBQyxhQUFhLENBQUMsTUFBTSxLQUFLLENBQUM7WUFDOUMsSUFBSztZQUNMLElBQUksQ0FBQyxhQUFhLENBQUMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxNQUFNLEdBQUcsQ0FBQyxDQUFDLENBQUM7UUFHdEQsRUFBRSxDQUFDLENBQUMsTUFBTSxZQUFZLGlCQUFPO1lBQ3pCLENBQUMsSUFBSSxDQUFDLG1CQUFtQixDQUFDLE1BQU0sRUFBRSxJQUFJLENBQUMseUJBQXlCLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDdEUsSUFBSSxDQUFDLEtBQUssQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUNyQixDQUFDO1FBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxPQUFPLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ2pDLE1BQU0sQ0FBQyxPQUFPLENBQUMsVUFBQSxDQUFDO2dCQUNkLEVBQUUsQ0FBQyxDQUFDLENBQUMsWUFBWSxpQkFBTztvQkFDcEIsQ0FBQyxLQUFJLENBQUMsbUJBQW1CLENBQUMsQ0FBQyxFQUFFLEtBQUksQ0FBQyx5QkFBeUIsQ0FBQyxDQUFDLENBQUMsQ0FBQztvQkFDakUsS0FBSSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDaEIsQ0FBQztZQUNILENBQUMsQ0FBQyxDQUFDO1FBQ0wsQ0FBQztRQUVELElBQUksQ0FBQyxjQUFjLENBQUMsR0FBRyxFQUFFLENBQUM7UUFDMUIsSUFBSSxDQUFDLHlCQUF5QixHQUFHLElBQUksQ0FBQyxjQUFjLENBQUMsTUFBTSxLQUFLLENBQUM7WUFDN0QsSUFBSztZQUNMLElBQUksQ0FBQyxjQUFjLENBQUMsSUFBSSxDQUFDLGNBQWMsQ0FBQyxNQUFNLEdBQUcsQ0FBQyxDQUFDLENBQUM7SUFDMUQsQ0FBQztJQUVPLHlDQUFtQixHQUEzQixVQUE0QixPQUFnQixFQUFFLFdBQXNCO1FBQ2xFLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsV0FBVyxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1lBQzVDLEVBQUUsQ0FBQyxDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxPQUFPLEVBQUUsS0FBSyxPQUFPLENBQUMsT0FBTyxFQUFFLENBQUMsQ0FBQyxDQUFDO2dCQUNuRCxNQUFNLENBQUMsSUFBSSxDQUFDO1lBQ2QsQ0FBQztRQUNILENBQUM7UUFDRCxNQUFNLENBQUMsS0FBSyxDQUFDO0lBQ2YsQ0FBQztJQU1ELDBCQUFJLEdBQUosVUFBd0IsTUFBUztRQUMvQixFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsV0FBVyxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUM7WUFDN0IsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7Z0JBQ2xCLE1BQU0sSUFBSSxLQUFLLENBQ1gsK0NBQStDO29CQUMvQyxzQ0FBc0M7b0JBQ3RDLHdEQUF3RDtvQkFDeEQsUUFBUSxDQUFDLENBQUM7WUFDaEIsQ0FBQztZQUNELE1BQU0sQ0FBQyxNQUFNLENBQUM7UUFDaEIsQ0FBQztRQUNELElBQUksQ0FBQyx5QkFBeUIsQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDNUMsTUFBTSxDQUFDLE1BQU0sQ0FBQztJQUNoQixDQUFDO0lBT0QsMkJBQUssR0FBTCxVQUF5QixNQUFTO1FBQ2hDLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxXQUFXLElBQUksSUFBSSxDQUFDLENBQUMsQ0FBQztZQUM3QixFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQztnQkFDbEIsTUFBTSxJQUFJLEtBQUssQ0FDWCwrQ0FBK0M7b0JBQy9DLHNDQUFzQztvQkFDdEMsd0RBQXdEO29CQUN4RCxRQUFRLENBQUMsQ0FBQztZQUNoQixDQUFDO1lBQ0QsTUFBTSxDQUFDLE1BQU0sQ0FBQztRQUNoQixDQUFDO1FBQ0QsSUFBSSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDOUIsTUFBTSxDQUFDLE1BQU0sQ0FBQztJQUNoQixDQUFDO0lBYUQsNEJBQU0sR0FBTixVQUNJLENBQVUsRUFBRSxDQUFVLEVBQUUsWUFBd0MsRUFDaEUsWUFBd0M7UUFEaEIsNkJBQUEsRUFBQSxlQUFlLGlCQUFpQixDQUFDLE9BQU87UUFDaEUsNkJBQUEsRUFBQSxlQUFlLGlCQUFpQixDQUFDLE9BQU87UUFDMUMsSUFBTSxXQUFXLEdBQ2IsQ0FBQyxZQUFZLEtBQUssaUJBQWlCLENBQUMsT0FBTyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQzNFLElBQU0sV0FBVyxHQUNiLENBQUMsWUFBWSxLQUFLLGlCQUFpQixDQUFDLE9BQU8sQ0FBQyxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUUzRSxJQUFJLENBQUMsTUFBTSxDQUNQLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxJQUFJLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUM1Qix1REFBcUQsQ0FBQyxDQUFDLElBQU07YUFDekQsU0FBTyxDQUFDLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBRTFCLElBQUksQ0FBQyxNQUFNLENBQ1AsV0FBVyxLQUFLLFdBQVcsRUFDM0Isb0NBQWtDLFdBQVcsWUFBUzthQUMvQyxXQUFXLGtDQUE2QixDQUFDLENBQUMsS0FBSyxVQUFPLENBQUE7YUFDdEQsQ0FBQyxDQUFDLEtBQUssMEJBQXFCLGlCQUFpQixDQUFDLFlBQVksQ0FBRyxDQUFBO2FBQ2hFLFVBQVEsaUJBQWlCLENBQUMsWUFBWSxDQUFDLGlCQUFjLENBQUEsQ0FBQyxDQUFDO1FBRS9ELE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxZQUFZLEVBQUUsWUFBWSxDQUFDLENBQUMsQ0FBQztJQUMzRSxDQUFDO0lBVUQsdUNBQWlCLEdBQWpCLFVBQWtCLENBQVUsRUFBRSxNQUFlO1FBQzNDLElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1osa0VBQWtFO2FBQzlELFVBQVEsQ0FBQyxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUMzQixJQUFJLENBQUMsTUFBTSxDQUNQLE1BQU0sQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNqQixtRUFBbUU7YUFDL0QsVUFBUSxNQUFNLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQ2hDLElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUMxQiw2REFBMkQsQ0FBQyxDQUFDLElBQUksT0FBSTtZQUNqRSw2REFBNkQ7YUFDN0QsVUFBUSxNQUFNLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBRWhDLE1BQU0sQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsRUFBRSxNQUFNLENBQUMsQ0FBQyxJQUFJLEVBQUUsQ0FBQztJQUN2RCxDQUFDO0lBT0QsdUNBQWlCLEdBQWpCLFVBQWtCLE1BQWUsRUFBRSxDQUFVO1FBQzNDLElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1osZ0VBQWdFO2FBQzVELFVBQVEsQ0FBQyxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUMzQixJQUFJLENBQUMsTUFBTSxDQUNQLE1BQU0sQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNqQixvRUFBb0U7YUFDaEUsVUFBUSxNQUFNLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQ2hDLElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUMxQiw0REFBMEQsQ0FBQyxDQUFDLElBQUksTUFBRztZQUMvRCw2REFBNkQ7YUFDN0QsV0FBUyxNQUFNLENBQUMsS0FBSyxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBRWxDLE1BQU0sQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLE1BQU0sRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxJQUFJLEVBQUUsQ0FBQztJQUN2RCxDQUFDO0lBT0QsZ0NBQVUsR0FBVixVQUFXLEVBQVcsRUFBRSxFQUFXO1FBQ2pDLElBQUksQ0FBQyxNQUFNLENBQ1AsRUFBRSxDQUFDLElBQUksS0FBSyxDQUFDLElBQUksRUFBRSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQzlCLDREQUE0RDthQUNyRCxFQUFFLENBQUMsSUFBSSxhQUFRLEVBQUUsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDdEMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxFQUFFLENBQUMsSUFBSSxLQUFLLEVBQUUsQ0FBQyxJQUFJLEVBQ25CLDBDQUF3QyxFQUFFLENBQUMsSUFBSSxZQUFTO2FBQ2pELEVBQUUsQ0FBQyxJQUFJLGtCQUFlLENBQUEsQ0FBQyxDQUFDO1FBQ25DLE1BQU0sQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxJQUFJLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxJQUFJLENBQUMsRUFBRSxFQUFFLENBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxRQUFRLEVBQUUsQ0FBQztJQUMxRSxDQUFDO0lBT0Qsa0NBQVksR0FBWixVQUFhLEVBQVcsRUFBRSxFQUFXO1FBQ25DLElBQUksQ0FBQyxNQUFNLENBQ1AsRUFBRSxDQUFDLElBQUksS0FBSyxDQUFDLElBQUksRUFBRSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQzlCLDhEQUE4RDthQUN2RCxFQUFFLENBQUMsSUFBSSxhQUFRLEVBQUUsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFFdEMsTUFBTSxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsSUFBSSxFQUFFLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxJQUFJLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDO0lBQy9ELENBQUM7SUFVRCwyQkFBSyxHQUFMLFVBQXlCLE9BQVU7UUFDakMsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDO0lBQ2pELENBQUM7SUFVRCw2QkFBTyxHQUFQLFVBQ0ksT0FBVyxFQUFFLFFBQWtCO1FBQ2pDLElBQUksQ0FBQyxNQUFNLENBQ1AsT0FBTyxDQUFDLElBQUksS0FBSyxJQUFJLENBQUMsYUFBYSxDQUFDLFFBQVEsQ0FBQyxFQUM3QyxnQ0FBOEIsT0FBTyxDQUFDLElBQUksMEJBQXVCO2FBQzFELElBQUksQ0FBQyxhQUFhLENBQUMsUUFBUSxDQUFDLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDNUMsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLGVBQWUsQ0FBUyxPQUFPLEVBQUUsUUFBUSxDQUFDLENBQUMsQ0FBQztJQUNyRSxDQUFDO0lBWUQsNkJBQU8sR0FBUCxVQUFRLEtBQWMsRUFBRSxLQUF1QixFQUFFLElBQXNCO1FBRXJFLElBQUksQ0FBQyxNQUFNLENBQ1AsS0FBSyxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQztZQUNoQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQ3hDLGdEQUE4QyxLQUFLLGVBQVk7YUFDeEQsSUFBSSx1Q0FBa0MsS0FBSyxDQUFDLEtBQUssTUFBRyxDQUFBLENBQUMsQ0FBQztRQUNqRSxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsZUFBZSxDQUFDLEtBQUssRUFBRSxLQUFLLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQztJQUM5RCxDQUFDO0lBZUQsNEJBQU0sR0FBTixVQUNJLE1BQWUsRUFBRSxXQUE2QixFQUM5QyxVQUE0QixFQUFFLElBQWEsRUFBRSxTQUEyQixFQUN4RSxRQUEwQjtRQUM1QixJQUFJLENBQUMsTUFBTSxDQUNQLFdBQVcsQ0FBQyxDQUFDLENBQUMsR0FBRyxVQUFVLENBQUMsQ0FBQyxDQUFDLElBQUksTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUM7WUFDN0MsV0FBVyxDQUFDLENBQUMsQ0FBQyxHQUFHLFVBQVUsQ0FBQyxDQUFDLENBQUMsSUFBSSxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUNyRCxzREFBb0QsV0FBVyxNQUFHO2FBQzlELHFCQUFtQixVQUFVLG1DQUFnQyxDQUFBO2FBQzdELGNBQVksTUFBTSxDQUFDLEtBQUssTUFBRyxDQUFBLENBQUMsQ0FBQztRQUNyQyxJQUFJLENBQUMsTUFBTSxDQUNQLFNBQVMsQ0FBQyxDQUFDLENBQUMsR0FBRyxRQUFRLENBQUMsQ0FBQyxDQUFDLElBQUksSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUM7WUFDdkMsU0FBUyxDQUFDLENBQUMsQ0FBQyxHQUFHLFFBQVEsQ0FBQyxDQUFDLENBQUMsSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUMvQyxvREFBa0QsU0FBUyxNQUFHO2FBQzFELHFCQUFtQixRQUFRLG9DQUFpQyxDQUFBO2FBQzVELFdBQVMsSUFBSSxDQUFDLEtBQUssTUFBRyxDQUFBLENBQUMsQ0FBQztRQUNoQyxXQUFXLENBQUMsY0FBYyxDQUFDLFVBQVUsRUFBRSxRQUFRLENBQUMsQ0FBQztRQUVqRCxNQUFNLENBQUMsSUFBSSxDQUFDLGNBQWMsQ0FDdEIsTUFBTSxFQUFFLFdBQVcsRUFBRSxVQUFVLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxRQUFRLENBQUMsQ0FBQztJQUNsRSxDQUFDO0lBb0NELDhCQUFRLEdBQVIsVUFBUyxRQUFpQixFQUFFLFFBQWlCLEVBQUUsSUFBWTtRQUN6RCxhQUFhLENBQUMseUJBQXlCLENBQ25DLFFBQVEsQ0FBQyxLQUFLLEVBQUUsUUFBUSxDQUFDLEtBQUssRUFBRSxJQUFJLEVBQUUscUJBQXFCLENBQUMsQ0FBQztRQUNqRSxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsZ0JBQWdCLENBQUMsUUFBUSxFQUFFLFFBQVEsRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDO0lBQ3JFLENBQUM7SUFZRCwrQkFBUyxHQUFULFVBQVUsT0FBZ0I7UUFDeEIsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLGlCQUFpQixDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDckQsQ0FBQztJQU9ELHlCQUFHLEdBQUgsVUFBSSxPQUFnQjtRQUNsQixNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDL0MsQ0FBQztJQU9ELDRCQUFNLEdBQU4sVUFBTyxPQUFnQjtRQUNyQixNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsY0FBYyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDbEQsQ0FBQztJQU9ELDRCQUFNLEdBQU4sVUFBTyxPQUFnQjtRQUNyQixNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsY0FBYyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDbEQsQ0FBQztJQVFELGtDQUFZLEdBQVosVUFBYSxFQUFXLEVBQUUsRUFBVztRQUNuQyxJQUFJLENBQUMsaUJBQWlCLENBQUMsRUFBRSxDQUFDLEtBQUssRUFBRSxFQUFFLENBQUMsS0FBSyxFQUFFLHlCQUF5QixDQUFDLENBQUM7UUFDdEUsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLG9CQUFvQixDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQ3ZELENBQUM7SUFRRCwwQkFBSSxHQUFKLFVBQUssT0FBZ0IsRUFBRSxDQUFTO1FBQzlCLElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxJQUFJLE9BQU8sQ0FBQyxJQUFJLEVBQ2pCLDZCQUEyQixDQUFDLHVDQUFvQzthQUM1RCx3QkFBc0IsT0FBTyxDQUFDLEtBQUssTUFBRyxDQUFBLENBQUMsQ0FBQztRQUNoRCxJQUFNLE1BQU0sR0FBRyxJQUFJLENBQUMsWUFBWSxDQUFDLE9BQU8sRUFBRSxDQUFDLENBQUMsQ0FBQztRQUM3QyxJQUFJLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUMxQixJQUFJLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FBQyxPQUFPLENBQUMsQ0FBQztRQUMzQixNQUFNLENBQUMsTUFBTSxDQUFDO0lBQ2hCLENBQUM7SUFRRCx5QkFBRyxHQUFILFVBQUksT0FBZ0I7UUFDbEIsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDO0lBQy9DLENBQUM7SUFPRCx5QkFBRyxHQUFILFVBQUksT0FBZ0I7UUFDbEIsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDO0lBQy9DLENBQUM7SUFPRCw2QkFBTyxHQUFQLFVBQVEsQ0FBVTtRQUFsQixpQkFRQztRQVBDLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDO1lBR2hCLElBQU0sR0FBRyxHQUFHLEtBQUksQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDOUIsSUFBTSxTQUFTLEdBQUcsS0FBSSxDQUFDLGdCQUFnQixDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsQ0FBQztZQUNoRCxNQUFNLENBQUMsS0FBSSxDQUFDLEdBQUcsQ0FBQyxTQUFTLENBQUMsQ0FBQztRQUM3QixDQUFDLENBQUMsQ0FBQztJQUNMLENBQUM7SUFXRCwrQkFBUyxHQUFULFVBQTZCLENBQUksRUFBRSxNQUFnQjtRQUNqRCxJQUFJLENBQUMsTUFBTSxDQUNQLENBQUMsQ0FBQyxJQUFJLEtBQUssTUFBTSxDQUFDLE1BQU0sRUFDeEIsK0NBQTZDLENBQUMsQ0FBQyxLQUFLLE1BQUc7YUFDbkQscUNBQW1DLE1BQU0sTUFBRyxDQUFBLENBQUMsQ0FBQztRQUN0RCxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsaUJBQWlCLENBQUMsQ0FBQyxFQUFFLE1BQU0sQ0FBQyxDQUFDLENBQUM7SUFDdkQsQ0FBQztJQVNELHFDQUFlLEdBQWYsVUFBbUMsQ0FBUyxFQUFFLENBQUk7UUFDaEQsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWixtRUFBbUU7YUFDL0QsVUFBUSxDQUFDLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQzNCLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyx1QkFBdUIsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUN4RCxDQUFDO0lBU0Qsc0NBQWdCLEdBQWhCLFVBQW9DLENBQVMsRUFBRSxDQUFJO1FBQ2pELElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1osb0VBQW9FO2FBQ2hFLFVBQVEsQ0FBQyxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUMzQixNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsd0JBQXdCLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDekQsQ0FBQztJQVNELHNDQUFnQixHQUFoQixVQUFvQyxDQUFJLEVBQUUsQ0FBUztRQUNqRCxJQUFJLENBQUMsTUFBTSxDQUNQLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNaLGlFQUFpRTthQUM3RCxjQUFZLENBQUMsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDL0IsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLHdCQUF3QixDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3pELENBQUM7SUFRRCx5QkFBRyxHQUFILFVBQXVCLENBQUk7UUFDekIsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3pDLENBQUM7SUFRRCx5QkFBRyxHQUFILFVBQXVCLENBQUksRUFBRSxDQUFJO1FBQy9CLElBQUksQ0FBQyxpQkFBaUIsQ0FBQyxDQUFDLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQyxLQUFLLEVBQUUsZ0JBQWdCLENBQUMsQ0FBQztRQUMzRCxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzVDLENBQUM7SUFRRCx5QkFBRyxHQUFILFVBQXVCLENBQUksRUFBRSxDQUFJO1FBQy9CLElBQUksQ0FBQyxpQkFBaUIsQ0FBQyxDQUFDLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQyxLQUFLLEVBQUUsZ0JBQWdCLENBQUMsQ0FBQztRQUMzRCxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzVDLENBQUM7SUFTRCxvQ0FBYyxHQUFkLFVBQWtDLENBQUksRUFBRSxDQUFJO1FBQzFDLElBQUksQ0FBQyxpQkFBaUIsQ0FBQyxDQUFDLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQyxLQUFLLEVBQUUsMkJBQTJCLENBQUMsQ0FBQztRQUN0RSxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsc0JBQXNCLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDdkQsQ0FBQztJQVNELDRCQUFNLEdBQU4sVUFBMEIsQ0FBSSxFQUFFLENBQUk7UUFDbEMsSUFBSSxDQUFDLGlCQUFpQixDQUFDLENBQUMsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBRSxtQkFBbUIsQ0FBQyxDQUFDO1FBQzlELE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDL0MsQ0FBQztJQVNELDBDQUFvQixHQUFwQixVQUF3QyxDQUFTLEVBQUUsQ0FBSTtRQUNyRCxJQUFJLENBQUMsTUFBTSxDQUNQLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNaLG9FQUFvRTthQUNoRSx5QkFBdUIsQ0FBQyxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUMxQyxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsNEJBQTRCLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDN0QsQ0FBQztJQVVELDBDQUFvQixHQUFwQixVQUF3QyxDQUFJLEVBQUUsQ0FBUztRQUNyRCxJQUFJLENBQUMsTUFBTSxDQUNQLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNaLGlFQUFpRTthQUM3RCw2QkFBMkIsQ0FBQyxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUM5QyxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsNEJBQTRCLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDN0QsQ0FBQztJQVFELHlCQUFHLEdBQUgsVUFBdUIsT0FBVTtRQUMvQixNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDL0MsQ0FBQztJQU9ELHlCQUFHLEdBQUgsVUFBdUIsT0FBVTtRQUMvQixNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDL0MsQ0FBQztJQU9ELDBCQUFJLEdBQUosVUFBd0IsT0FBVTtRQUNoQyxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsWUFBWSxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDaEQsQ0FBQztJQU9ELDZCQUFPLEdBQVAsVUFBMkIsT0FBVTtRQUNuQyxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsZUFBZSxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDbkQsQ0FBQztJQU9ELDBCQUFJLEdBQUosVUFBd0IsT0FBVTtRQUNoQyxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsWUFBWSxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDaEQsQ0FBQztJQU9ELHlCQUFHLEdBQUgsVUFBdUIsT0FBVTtRQUMvQixNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDL0MsQ0FBQztJQVFELDBCQUFJLEdBQUosVUFBd0IsT0FBVTtRQUNoQyxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsWUFBWSxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDaEQsQ0FBQztJQVVELG9DQUFjLEdBQWQsVUFBa0MsRUFBVSxFQUFFLENBQUksRUFBRSxFQUFVLEVBQUUsQ0FBSTtRQUNsRSxJQUFJLENBQUMsTUFBTSxDQUNQLEVBQUUsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNiLCtEQUErRDthQUMzRCxXQUFTLEVBQUUsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDN0IsSUFBSSxDQUFDLE1BQU0sQ0FDUCxFQUFFLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDYixrRUFBa0U7YUFDOUQscUJBQW1CLEVBQUUsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDdkMsSUFBSSxDQUFDLGlCQUFpQixDQUFDLENBQUMsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBRSwyQkFBMkIsQ0FBQyxDQUFDO1FBRXRFLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxzQkFBc0IsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQy9ELENBQUM7SUFVRCxzQ0FBZ0IsR0FBaEIsVUFBb0MsQ0FBUyxFQUFFLENBQUk7UUFDakQsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWixvRUFBb0U7YUFDaEUsY0FBWSxDQUFDLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQy9CLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyx3QkFBd0IsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUN6RCxDQUFDO0lBV0QsNkNBQXVCLEdBQXZCLFVBQXdCLENBQVUsRUFBRSxDQUFVO1FBQzVDLElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1osMkRBQTJEO2FBQ3ZELDBCQUF3QixDQUFDLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQzNDLElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1osNERBQTREO2FBQ3hELDBCQUF3QixDQUFDLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQzNDLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQywrQkFBK0IsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNoRSxDQUFDO0lBa0JELDRCQUFNLEdBQU4sVUFDSSxDQUFVLEVBQUUsT0FBZ0IsRUFBRSxNQUFvQixFQUFFLE1BQWMsRUFDbEUsT0FBZTtRQUNqQixJQUFJLENBQUMsTUFBTSxDQUNQLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNaLHFEQUFtRCxDQUFDLENBQUMsSUFBSSxNQUFHLENBQUMsQ0FBQztRQUNsRSxJQUFJLENBQUMsTUFBTSxDQUNQLE9BQU8sQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNsQix3REFBd0Q7YUFDakQsT0FBTyxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUM1QixFQUFFLENBQUMsQ0FBQyxNQUFNLElBQUksSUFBSSxDQUFDLENBQUMsQ0FBQztZQUNuQixJQUFJLENBQUMsTUFBTSxDQUNQLE1BQU0sQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNqQix1REFBdUQ7aUJBQ2hELE1BQU0sQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDN0IsQ0FBQztRQUVELElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsS0FBSyxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUMvQixzQ0FBb0MsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsbUJBQWdCO2FBQzFELDZCQUEyQixPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBR3hELE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQyxFQUFFLE9BQU8sRUFBRSxNQUFNLEVBQUUsTUFBTSxFQUFFLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDOUUsQ0FBQztJQWNELG9DQUFjLEdBQWQsVUFDSSxDQUFVLEVBQUUsRUFBVyxFQUFFLE9BQWdCLEVBQUUsTUFBYyxFQUN6RCxHQUFXO1FBQ2IsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWiwyREFBMkQ7YUFDcEQsQ0FBQyxDQUFDLEtBQUssTUFBRyxDQUFBLENBQUMsQ0FBQztRQUN2QixJQUFJLENBQUMsTUFBTSxDQUNQLEVBQUUsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNiLDREQUE0RDthQUNyRCxFQUFFLENBQUMsS0FBSyxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQ3hCLElBQUksQ0FBQyxNQUFNLENBQ1AsT0FBTyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2xCLGlFQUFpRTthQUMxRCxPQUFPLENBQUMsS0FBSyxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQzdCLElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsS0FBSyxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUMvQix5Q0FBdUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsWUFBUzthQUN0RCxvQ0FBa0MsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsTUFBRyxDQUFBLENBQUMsQ0FBQztRQUMvRCxJQUFJLENBQUMsTUFBTSxDQUNQLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEtBQUssT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFDaEMsMkNBQXlDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLFlBQVM7YUFDekQscUNBQW1DLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLE9BQUksQ0FBQSxDQUFDLENBQUM7UUFFakUsSUFBTSxjQUFjLEdBQ2hCLElBQUksQ0FBQyxzQkFBc0IsQ0FBQyxDQUFDLEVBQUUsRUFBRSxFQUFFLE9BQU8sRUFBRSxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUM7UUFFN0QsSUFBSSxDQUFDLEtBQUssQ0FBQyxjQUFjLENBQUMsRUFBRSxDQUFDLENBQUM7UUFDOUIsSUFBSSxDQUFDLEtBQUssQ0FBQyxjQUFjLENBQUMsRUFBRSxDQUFDLENBQUM7UUFDOUIsSUFBSSxDQUFDLEtBQUssQ0FBQyxjQUFjLENBQUMsRUFBRSxDQUFDLENBQUM7UUFFOUIsTUFBTSxDQUFDLGNBQWMsQ0FBQztJQUN4QixDQUFDO0lBZ0JELHFDQUFlLEdBQWYsVUFDSSxDQUFVLEVBQUUsT0FBZ0IsRUFBRSxNQUFvQixFQUFFLE1BQWMsRUFDbEUsR0FBVztRQUNiLElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1osMkRBQTJEO2FBQ3BELENBQUMsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDdEIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxPQUFPLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDbEIsNERBQTREO2FBQ3hELFVBQVEsT0FBTyxDQUFDLElBQU0sQ0FBQSxDQUFDLENBQUM7UUFDaEMsRUFBRSxDQUFDLENBQUMsTUFBTSxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUM7WUFDbkIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxNQUFNLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDakIsdUZBQ1ksTUFBTSxDQUFDLElBQUksTUFBRyxDQUFDLENBQUM7UUFDbEMsQ0FBQztRQUNELElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsS0FBSyxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUMvQiwrQ0FBNkMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsWUFBUzthQUM1RCxtQ0FBaUMsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsTUFBRyxDQUFBLENBQUMsQ0FBQztRQUU5RCxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FDYixJQUFJLENBQUMsdUJBQXVCLENBQUMsQ0FBQyxFQUFFLE9BQU8sRUFBRSxNQUFNLEVBQUUsTUFBTSxFQUFFLEdBQUcsQ0FBQyxDQUFDLENBQUM7SUFDckUsQ0FBQztJQWFELDZCQUFPLEdBQVAsVUFBUSxDQUFVLEVBQUUsS0FBYSxFQUFFLE1BQWMsRUFBRSxHQUFXO1FBQzVELElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1osa0RBQWtELEdBQUcsQ0FBQyxDQUFDLElBQUksR0FBRyxHQUFHLENBQUMsQ0FBQztRQUN2RSxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsZUFBZSxDQUFDLENBQUMsRUFBRSxLQUFLLEVBQUUsTUFBTSxFQUFFLEdBQUcsQ0FBQyxDQUFDLENBQUM7SUFDakUsQ0FBQztJQWFELHFDQUFlLEdBQWYsVUFDSSxFQUFXLEVBQUUsQ0FBVSxFQUFFLEtBQWEsRUFBRSxNQUFjLEVBQ3RELEdBQVc7UUFDYixJQUFJLENBQUMsTUFBTSxDQUNQLEVBQUUsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNiLDJEQUEyRDthQUNwRCxFQUFFLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQ3ZCLElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1osMERBQTBEO2FBQ25ELENBQUMsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFFdEIsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLHVCQUF1QixDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsS0FBSyxFQUFFLE1BQU0sRUFBRSxHQUFHLENBQUMsQ0FBQyxDQUFDO0lBQzdFLENBQUM7SUFhRCw2QkFBTyxHQUFQLFVBQVEsQ0FBVSxFQUFFLEtBQWEsRUFBRSxNQUFjLEVBQUUsR0FBVztRQUM1RCxJQUFJLENBQUMsTUFBTSxDQUNQLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNaLHFEQUFtRCxDQUFDLENBQUMsSUFBSSxNQUFHLENBQUMsQ0FBQztRQUNsRSxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsZUFBZSxDQUFDLENBQUMsRUFBRSxLQUFLLEVBQUUsTUFBTSxFQUFFLEdBQUcsQ0FBQyxDQUFDLENBQUM7SUFDakUsQ0FBQztJQVlELDZCQUFPLEdBQVAsVUFBUSxDQUFVLEVBQUUsS0FBYSxFQUFFLE1BQWMsRUFBRSxHQUFXO1FBQzVELElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1oscURBQW1ELENBQUMsQ0FBQyxJQUFJLE1BQUcsQ0FBQyxDQUFDO1FBQ2xFLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxlQUFlLENBQUMsQ0FBQyxFQUFFLEtBQUssRUFBRSxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUMsQ0FBQztJQUNqRSxDQUFDO0lBY0Qsc0NBQWdCLEdBQWhCLFVBQ0ksQ0FBVSxFQUFFLFVBQTRCLEVBQUUsWUFBb0I7UUFBcEIsNkJBQUEsRUFBQSxvQkFBb0I7UUFDaEUsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWiw4REFBNEQsQ0FBQyxDQUFDLElBQUksTUFBRyxDQUFDLENBQUM7UUFDM0UsSUFBSSxDQUFDLE1BQU0sQ0FDUCxVQUFVLENBQUMsTUFBTSxLQUFLLENBQUMsRUFDdkIsOERBQThEO2FBQ3ZELFVBQVUsTUFBRyxDQUFBLENBQUMsQ0FBQztRQUMxQixNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FDYixJQUFJLENBQUMsd0JBQXdCLENBQUMsQ0FBQyxFQUFFLFVBQVUsRUFBRSxZQUFZLENBQUMsQ0FBQyxDQUFDO0lBQ2xFLENBQUM7SUFnQkQsMENBQW9CLEdBQXBCLFVBQ0ksQ0FBVSxFQUFFLElBQXFCLEVBQUUsUUFBeUIsRUFDNUQsZUFBc0IsRUFBRSxLQUF1QixFQUMvQyxNQUF3QjtRQUR4QixnQ0FBQSxFQUFBLHNCQUFzQjtRQUV4QixJQUFJLENBQUMsTUFBTSxDQUNQLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNaLCtEQUErRDthQUN4RCxDQUFDLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQ3RCLElBQUksQ0FBQyxNQUFNLENBQ1AsSUFBSSxDQUFDLElBQUksS0FBSyxDQUFDLElBQUksSUFBSSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2xDLG1FQUFtRTthQUMvRCxjQUFZLElBQUksQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDbEMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxRQUFRLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxRQUFRLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDMUMsbUVBQW1FO2FBQy9ELGtCQUFnQixRQUFRLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQzFDLEVBQUUsQ0FBQyxDQUFDLEtBQUssSUFBSSxJQUFJLENBQUMsQ0FBQyxDQUFDO1lBQ2xCLElBQUksQ0FBQyxNQUFNLENBQ1AsS0FBSyxDQUFDLElBQUksS0FBSyxDQUFDLElBQUksS0FBSyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ3BDLGdFQUFnRTtpQkFDNUQsa0JBQWdCLEtBQU0sQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDMUMsQ0FBQztRQUNELEVBQUUsQ0FBQyxDQUFDLE1BQU0sSUFBSSxJQUFJLENBQUMsQ0FBQyxDQUFDO1lBQ25CLElBQUksQ0FBQyxNQUFNLENBQ1AsTUFBTSxDQUFDLElBQUksS0FBSyxDQUFDLElBQUksTUFBTSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ3RDLGlFQUFpRTtpQkFDN0Qsa0JBQWdCLE1BQU8sQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDM0MsQ0FBQztRQUVELE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyw0QkFBNEIsQ0FDL0MsQ0FBQyxFQUFFLElBQUksRUFBRSxRQUFRLEVBQUUsZUFBZSxFQUFFLEtBQUssRUFBRSxNQUFNLENBQUMsQ0FBQyxDQUFDO0lBQzFELENBQUM7SUFLSCxrQkFBQztBQUFELENBNWdDQSxBQTRnQ0MsSUFBQTtBQTVnQ3FCLGtDQUFXO0FBOGdDakMsSUFBWSxpQkFHWDtBQUhELFdBQVksaUJBQWlCO0lBQzNCLCtEQUFPLENBQUE7SUFDUCxxRUFBVSxDQUFBO0FBQ1osQ0FBQyxFQUhXLGlCQUFpQixHQUFqQix5QkFBaUIsS0FBakIseUJBQWlCLFFBRzVCOzs7Ozs7Ozs7Ozs7Ozs7QUN6aENELDZDQUErQztBQUMvQyw4QkFBZ0M7QUFFaEMsK0NBQWlEO0FBQ2pELDJDQUE2QztBQUM3QywrQkFBc0Q7QUFDdEQscUNBQThFO0FBRTlFO0lBQW9DLGtDQUFXO0lBQzdDLHdCQUFZLFFBQWdCO1FBQWhCLHlCQUFBLEVBQUEsZ0JBQWdCO2VBQzFCLGtCQUFNLFFBQVEsQ0FBQztJQUNqQixDQUFDO0lBRVMsc0NBQWEsR0FBdkIsVUFBMkMsT0FBVTtRQUNuRCxNQUFNLENBQUMsaUJBQU8sQ0FBQyxJQUFJLENBQ2YsT0FBTyxDQUFDLEtBQUssRUFBRSxFQUFDLE1BQU0sRUFBRSxJQUFJLFlBQVksQ0FBQyxPQUFPLENBQUMsU0FBUyxFQUFFLENBQUMsRUFBQyxDQUFDLENBQUM7SUFDdEUsQ0FBQztJQUVTLHdDQUFlLEdBQXpCLFVBQ0ksT0FBVyxFQUFFLFFBQWtCO1FBQ2pDLE1BQU0sQ0FBQyxJQUFJLENBQUMsYUFBYSxDQUFDLE9BQU8sQ0FBQyxDQUFDLE9BQU8sQ0FBSyxRQUFRLENBQUMsQ0FBQztJQUMzRCxDQUFDO0lBRVMsd0NBQWUsR0FBekIsVUFDSSxLQUFjLEVBQUUsV0FBNkIsRUFDN0MsVUFBNEI7UUFDOUIsSUFBTSxNQUFNLEdBQUcsaUJBQU8sQ0FBQyxLQUFLLENBQUMsVUFBVSxDQUFDLENBQUM7UUFDekMsSUFBSSxDQUFDLGNBQWMsQ0FDZixLQUFLLEVBQUUsV0FBVyxFQUFFLFVBQVUsRUFBRSxNQUFNLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsVUFBVSxDQUFDLENBQUM7UUFDaEUsTUFBTSxDQUFDLE1BQU0sQ0FBQztJQUNoQixDQUFDO0lBRVMsdUNBQWMsR0FBeEIsVUFDSSxNQUFlLEVBQUUsaUJBQW1DLEVBQ3BELGdCQUFrQyxFQUFFLElBQWEsRUFDakQsZUFBaUMsRUFDakMsY0FBZ0M7UUFDbEMsV0FBVyxDQUFDLGNBQWMsQ0FBQyxnQkFBZ0IsRUFBRSxjQUFjLENBQUMsQ0FBQztRQUM3RCxJQUFNLFNBQVMsR0FBRyxNQUFNLENBQUMsU0FBUyxFQUFFLENBQUM7UUFDckMsSUFBTSxTQUFTLEdBQUcsSUFBSSxDQUFDLFNBQVMsRUFBRSxDQUFDO1FBQ25DLElBQU0sQ0FBQyxHQUFHLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxHQUFHLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3BELEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUM7WUFDM0IsSUFBTSxNQUFNLEdBQUcsaUJBQWlCLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDLEdBQUcsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUMxRSxJQUFNLE1BQU0sR0FBRyxpQkFBaUIsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsR0FBRyxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ2hFLElBQU0sTUFBTSxHQUFHLE1BQU0sR0FBRyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxHQUFHLE1BQU0sQ0FBQztZQUNqRCxJQUFNLE1BQU0sR0FBRyxlQUFlLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDLEdBQUcsY0FBYyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDdEUsSUFBTSxNQUFNLEdBQUcsZUFBZSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxHQUFHLGNBQWMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQzVELElBQU0sTUFBTSxHQUFHLE1BQU0sR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxHQUFHLE1BQU0sQ0FBQztZQUMvQyxTQUFTLENBQUMsTUFBTSxDQUFDLEdBQUcsU0FBUyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQ3hDLENBQUM7SUFDSCxDQUFDO0lBRVMseUNBQWdCLEdBQTFCLFVBQTJCLEVBQVcsRUFBRSxFQUFXLEVBQUUsSUFBWTtRQUMvRCxJQUFNLFdBQVcsR0FDYixhQUFhLENBQUMsMEJBQTBCLENBQUMsRUFBRSxDQUFDLEtBQUssRUFBRSxFQUFFLENBQUMsS0FBSyxFQUFFLElBQUksQ0FBQyxDQUFDO1FBRXZFLElBQU0sTUFBTSxHQUFHLGlCQUFPLENBQUMsS0FBSyxDQUFVLFdBQVcsQ0FBQyxDQUFDO1FBRW5ELEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsV0FBVyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7WUFDeEMsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxXQUFXLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztnQkFDeEMsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxXQUFXLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztvQkFFeEMsSUFBTSxLQUFLLEdBQTZCLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztvQkFDbEQsSUFBSSxLQUFLLFNBQVEsQ0FBQztvQkFDbEIsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxHQUFHLEVBQUUsQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO3dCQUNqQyxLQUFLLEdBQUcsRUFBRSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO29CQUMxQixDQUFDO29CQUFDLElBQUksQ0FBQyxDQUFDO3dCQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsSUFBSSxFQUFFLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxDQUFDO3dCQUN2QixJQUFBLGFBQUUsRUFBRSxhQUFFLEVBQUUsYUFBRSxDQUFVO3dCQUMzQixLQUFLLEdBQUcsRUFBRSxDQUFDLEdBQUcsQ0FBQyxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQyxDQUFDO29CQUM3QixDQUFDO29CQUVELE1BQU0sQ0FBQyxHQUFHLENBQUMsS0FBSyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7Z0JBQzdCLENBQUM7WUFDSCxDQUFDO1FBQ0gsQ0FBQztRQUVELE1BQU0sQ0FBQyxNQUFNLENBQUM7SUFDaEIsQ0FBQztJQUVTLGdEQUF1QixHQUFqQyxVQUFxRCxDQUFTLEVBQUUsQ0FBSTtRQUNsRSxJQUFNLFlBQVksR0FBRyxJQUFJLFlBQVksQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUM7UUFDOUMsSUFBTSxPQUFPLEdBQUcsQ0FBQyxDQUFDLFNBQVMsRUFBRSxDQUFDO1FBQzlCLElBQU0sSUFBSSxHQUFHLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQztRQUNyQixHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLFlBQVksQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQztZQUM3QyxZQUFZLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxHQUFHLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUN0QyxDQUFDO1FBQ0QsTUFBTSxDQUFDLGlCQUFPLENBQUMsSUFBSSxDQUFJLENBQUMsQ0FBQyxLQUFLLEVBQUUsRUFBQyxNQUFNLEVBQUUsWUFBWSxFQUFDLENBQUMsQ0FBQztJQUMxRCxDQUFDO0lBRVMsK0NBQXNCLEdBQWhDLFVBQ0ksRUFBVSxFQUFFLENBQUksRUFBRSxFQUFVLEVBQUUsQ0FBSTtRQUNwQyxJQUFNLE9BQU8sR0FBRyxJQUFJLFlBQVksQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUM7UUFDekMsSUFBTSxPQUFPLEdBQUcsQ0FBQyxDQUFDLFNBQVMsRUFBRSxDQUFDO1FBQzlCLElBQU0sT0FBTyxHQUFHLENBQUMsQ0FBQyxTQUFTLEVBQUUsQ0FBQztRQUM5QixJQUFNLEtBQUssR0FBRyxFQUFFLENBQUMsR0FBRyxFQUFFLENBQUM7UUFDdkIsSUFBTSxLQUFLLEdBQUcsRUFBRSxDQUFDLEdBQUcsRUFBRSxDQUFDO1FBQ3ZCLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsT0FBTyxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDO1lBQ3hDLE9BQU8sQ0FBQyxDQUFDLENBQUMsR0FBRyxLQUFLLEdBQUcsT0FBTyxDQUFDLENBQUMsQ0FBQyxHQUFHLEtBQUssR0FBRyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDdkQsQ0FBQztRQUNELE1BQU0sQ0FBQyxpQkFBTyxDQUFDLElBQUksQ0FBSSxDQUFDLENBQUMsS0FBSyxFQUFFLEVBQUMsTUFBTSxFQUFFLE9BQU8sRUFBQyxDQUFDLENBQUM7SUFDckQsQ0FBQztJQUVTLGlEQUF3QixHQUFsQyxVQUFzRCxDQUFTLEVBQUUsQ0FBSTtRQUNuRSxJQUFNLFNBQVMsR0FBRyxJQUFJLFlBQVksQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUM7UUFDM0MsSUFBTSxPQUFPLEdBQUcsQ0FBQyxDQUFDLFNBQVMsRUFBRSxDQUFDO1FBQzlCLElBQU0sSUFBSSxHQUFHLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQztRQUNyQixHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE9BQU8sQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQztZQUN4QyxTQUFTLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxHQUFHLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNuQyxDQUFDO1FBQ0QsTUFBTSxDQUFDLGlCQUFPLENBQUMsSUFBSSxDQUFJLENBQUMsQ0FBQyxLQUFLLEVBQUUsRUFBQyxNQUFNLEVBQUUsU0FBUyxFQUFDLENBQUMsQ0FBQztJQUN2RCxDQUFDO0lBRVMsaURBQXdCLEdBQWxDLFVBQXNELENBQVMsRUFBRSxDQUFJO1FBQ25FLElBQU0sSUFBSSxHQUFHLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDakMsSUFBTSxNQUFNLEdBQUcsSUFBSSxDQUFDLHVCQUF1QixDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsQ0FBQztRQUVyRCxJQUFJLENBQUMsT0FBTyxFQUFFLENBQUM7UUFFZixNQUFNLENBQUMsTUFBTSxDQUFDO0lBQ2hCLENBQUM7SUFFUyxpREFBd0IsR0FBbEMsVUFBc0QsQ0FBSSxFQUFFLENBQVM7UUFDbkUsSUFBTSxJQUFJLEdBQUcsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNqQyxJQUFNLE1BQU0sR0FBRyxJQUFJLENBQUMsdUJBQXVCLENBQUMsSUFBSSxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBRXJELElBQUksQ0FBQyxPQUFPLEVBQUUsQ0FBQztRQUVmLE1BQU0sQ0FBQyxNQUFNLENBQUM7SUFDaEIsQ0FBQztJQUVTLG9DQUFXLEdBQXJCLFVBQXlDLENBQUk7UUFDM0MsTUFBTSxDQUFDLElBQUksQ0FBQyx3QkFBd0IsQ0FBQyxnQkFBTSxDQUFDLE9BQU8sRUFBRSxDQUFDLENBQUMsQ0FBQztJQUMxRCxDQUFDO0lBRVMsb0NBQVcsR0FBckIsVUFBeUMsQ0FBSSxFQUFFLENBQUk7UUFDakQsTUFBTSxDQUFDLElBQUksQ0FBQyxzQkFBc0IsQ0FBSSxnQkFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLEVBQUUsZ0JBQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDdEUsQ0FBQztJQUVTLG9DQUFXLEdBQXJCLFVBQXlDLENBQUksRUFBRSxDQUFJO1FBQ2pELE1BQU0sQ0FBQyxJQUFJLENBQUMsc0JBQXNCLENBQUksZ0JBQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxFQUFFLGdCQUFNLENBQUMsT0FBTyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQzFFLENBQUM7SUFFUyx1Q0FBYyxHQUF4QixVQUNJLENBQVUsRUFBRSxDQUFVLEVBQUUsWUFBd0MsRUFDaEUsWUFBd0M7UUFEaEIsNkJBQUEsRUFBQSxlQUFlLHdCQUFpQixDQUFDLE9BQU87UUFDaEUsNkJBQUEsRUFBQSxlQUFlLHdCQUFpQixDQUFDLE9BQU87UUFDMUMsSUFBTSxTQUFTLEdBQ1gsQ0FBQyxZQUFZLEtBQUssd0JBQWlCLENBQUMsT0FBTyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBRTNFLElBQU0sT0FBTyxHQUNULENBQUMsWUFBWSxLQUFLLHdCQUFpQixDQUFDLE9BQU8sQ0FBQyxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUMzRSxJQUFNLFFBQVEsR0FDVixDQUFDLFlBQVksS0FBSyx3QkFBaUIsQ0FBQyxPQUFPLENBQUMsR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFM0UsSUFBTSxZQUFZLEdBQUcsVUFBQyxNQUFlLEVBQUUsQ0FBUyxFQUFFLENBQVM7WUFDdkQsT0FBQSxNQUFNLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUM7UUFBaEIsQ0FBZ0IsQ0FBQztRQUNyQixJQUFNLGdCQUFnQixHQUFHLFVBQUMsTUFBZSxFQUFFLENBQVMsRUFBRSxDQUFTO1lBQzNELE9BQUEsTUFBTSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDO1FBQWhCLENBQWdCLENBQUM7UUFFckIsSUFBTSxPQUFPLEdBQUcsQ0FBQyxZQUFZLEtBQUssd0JBQWlCLENBQUMsT0FBTyxDQUFDO1lBQ3hELFlBQVk7WUFDWixnQkFBZ0IsQ0FBQztRQUNyQixJQUFNLE9BQU8sR0FBRyxDQUFDLFlBQVksS0FBSyx3QkFBaUIsQ0FBQyxPQUFPLENBQUM7WUFDeEQsWUFBWTtZQUNaLGdCQUFnQixDQUFDO1FBQ3JCLElBQU0sTUFBTSxHQUFHLElBQUksWUFBWSxDQUFDLE9BQU8sR0FBRyxRQUFRLENBQUMsQ0FBQztRQUNwRCxJQUFJLEtBQUssR0FBRyxDQUFDLENBQUM7UUFFZCxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE9BQU8sRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDO1lBQ2pDLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsUUFBUSxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUM7Z0JBQ2xDLElBQUksR0FBRyxHQUFHLENBQUMsQ0FBQztnQkFDWixHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLFNBQVMsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDO29CQUVuQyxHQUFHLElBQUksT0FBTyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsT0FBTyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7Z0JBQzdDLENBQUM7Z0JBQ0QsTUFBTSxDQUFDLEtBQUssRUFBRSxDQUFDLEdBQUcsR0FBRyxDQUFDO1lBQ3hCLENBQUM7UUFDSCxDQUFDO1FBQ0QsTUFBTSxDQUFDLGlCQUFPLENBQUMsR0FBRyxDQUFDLENBQUMsT0FBTyxFQUFFLFFBQVEsQ0FBQyxFQUFFLE1BQU0sQ0FBQyxDQUFDO0lBQ2xELENBQUM7SUFFUywrQ0FBc0IsR0FBaEMsVUFBb0QsQ0FBSSxFQUFFLENBQUk7UUFDNUQsSUFBTSxTQUFTLEdBQUcsSUFBSSxZQUFZLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQzNDLElBQU0sT0FBTyxHQUFHLENBQUMsQ0FBQyxTQUFTLEVBQUUsQ0FBQztRQUM5QixJQUFNLE9BQU8sR0FBRyxDQUFDLENBQUMsU0FBUyxFQUFFLENBQUM7UUFDOUIsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxPQUFPLENBQUMsTUFBTSxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUM7WUFDeEMsU0FBUyxDQUFDLENBQUMsQ0FBQyxHQUFHLE9BQU8sQ0FBQyxDQUFDLENBQUMsR0FBRyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDekMsQ0FBQztRQUNELE1BQU0sQ0FBQyxpQkFBTyxDQUFDLElBQUksQ0FBSSxDQUFDLENBQUMsS0FBSyxFQUFFLEVBQUMsTUFBTSxFQUFFLFNBQVMsRUFBQyxDQUFDLENBQUM7SUFDdkQsQ0FBQztJQUVTLHdEQUErQixHQUF6QyxVQUEwQyxDQUFVLEVBQUUsQ0FBVTtRQUM5RCxJQUFNLE1BQU0sR0FBRyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ2hELElBQU0sTUFBTSxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFaEQsSUFBTSxNQUFNLEdBQUcsSUFBSSxZQUFZLENBQUMsTUFBTSxHQUFHLE1BQU0sQ0FBQyxDQUFDO1FBQ2pELElBQUksS0FBSyxHQUFHLENBQUMsQ0FBQztRQUNkLEdBQUcsQ0FBQyxDQUFDLElBQUksR0FBRyxHQUFHLENBQUMsRUFBRSxHQUFHLEdBQUcsTUFBTSxFQUFFLEdBQUcsRUFBRSxFQUFFLENBQUM7WUFDdEMsR0FBRyxDQUFDLENBQUMsSUFBSSxHQUFHLEdBQUcsQ0FBQyxFQUFFLEdBQUcsR0FBRyxNQUFNLEVBQUUsR0FBRyxFQUFFLEVBQUUsQ0FBQztnQkFDdEMsTUFBTSxDQUFDLEtBQUssRUFBRSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxHQUFHLEdBQUcsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxHQUFHLEdBQUcsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztvQkFDdkQsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxHQUFHLEdBQUcsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxHQUFHLEdBQUcsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ2hELENBQUM7UUFDSCxDQUFDO1FBQ0QsTUFBTSxDQUFDLGlCQUFPLENBQUMsR0FBRyxDQUFDLENBQUMsTUFBTSxFQUFFLE1BQU0sQ0FBQyxFQUFFLE1BQU0sQ0FBQyxDQUFDO0lBQy9DLENBQUM7SUFFUyx1Q0FBYyxHQUF4QixVQUE0QyxDQUFJLEVBQUUsQ0FBSTtRQUNwRCxJQUFNLFNBQVMsR0FBRyxJQUFJLFlBQVksQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUM7UUFDM0MsSUFBTSxPQUFPLEdBQUcsQ0FBQyxDQUFDLFNBQVMsRUFBRSxDQUFDO1FBQzlCLElBQU0sT0FBTyxHQUFHLENBQUMsQ0FBQyxTQUFTLEVBQUUsQ0FBQztRQUM5QixHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE9BQU8sQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQztZQUN4QyxTQUFTLENBQUMsQ0FBQyxDQUFDLEdBQUcsT0FBTyxDQUFDLENBQUMsQ0FBQyxHQUFHLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUN6QyxDQUFDO1FBQ0QsTUFBTSxDQUFDLGlCQUFPLENBQUMsSUFBSSxDQUFJLENBQUMsQ0FBQyxLQUFLLEVBQUUsRUFBQyxNQUFNLEVBQUUsU0FBUyxFQUFDLENBQUMsQ0FBQztJQUN2RCxDQUFDO0lBRVMscURBQTRCLEdBQXRDLFVBQTBELENBQVMsRUFBRSxDQUFJO1FBRXZFLElBQU0sU0FBUyxHQUFHLElBQUksWUFBWSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUMzQyxJQUFNLE9BQU8sR0FBRyxDQUFDLENBQUMsU0FBUyxFQUFFLENBQUM7UUFDOUIsSUFBTSxNQUFNLEdBQUcsQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDO1FBQ3ZCLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsT0FBTyxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDO1lBQ3hDLFNBQVMsQ0FBQyxDQUFDLENBQUMsR0FBRyxNQUFNLEdBQUcsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3JDLENBQUM7UUFDRCxNQUFNLENBQUMsaUJBQU8sQ0FBQyxJQUFJLENBQUksQ0FBQyxDQUFDLEtBQUssRUFBRSxFQUFDLE1BQU0sRUFBRSxTQUFTLEVBQUMsQ0FBQyxDQUFDO0lBQ3ZELENBQUM7SUFFUyxxREFBNEIsR0FBdEMsVUFBMEQsQ0FBSSxFQUFFLENBQVM7UUFFdkUsSUFBTSxTQUFTLEdBQUcsSUFBSSxZQUFZLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQzNDLElBQU0sT0FBTyxHQUFHLENBQUMsQ0FBQyxTQUFTLEVBQUUsQ0FBQztRQUM5QixJQUFNLE1BQU0sR0FBRyxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUM7UUFDdkIsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxPQUFPLENBQUMsTUFBTSxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUM7WUFDeEMsU0FBUyxDQUFDLENBQUMsQ0FBQyxHQUFHLE9BQU8sQ0FBQyxDQUFDLENBQUMsR0FBRyxNQUFNLENBQUM7UUFDckMsQ0FBQztRQUNELE1BQU0sQ0FBQyxpQkFBTyxDQUFDLElBQUksQ0FBSSxDQUFDLENBQUMsS0FBSyxFQUFFLEVBQUMsTUFBTSxFQUFFLFNBQVMsRUFBQyxDQUFDLENBQUM7SUFDdkQsQ0FBQztJQUVTLG9DQUFXLEdBQXJCLFVBQXNCLE9BQWdCO1FBQ3BDLElBQUksR0FBRyxHQUFHLENBQUMsQ0FBQztRQUNaLElBQU0sTUFBTSxHQUFHLE9BQU8sQ0FBQyxTQUFTLEVBQUUsQ0FBQztRQUNuQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE1BQU0sQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQztZQUN2QyxHQUFHLElBQUksTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ25CLENBQUM7UUFDRCxNQUFNLENBQUMsZ0JBQU0sQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUM7SUFDekIsQ0FBQztJQUVTLHVDQUFjLEdBQXhCLFVBQXlCLE9BQWdCO1FBQ3ZDLElBQUksR0FBRyxHQUFHLE1BQU0sQ0FBQyxTQUFTLENBQUM7UUFDM0IsSUFBSSxRQUFRLEdBQUcsQ0FBQyxDQUFDLENBQUM7UUFDbEIsSUFBTSxNQUFNLEdBQUcsT0FBTyxDQUFDLFNBQVMsRUFBRSxDQUFDO1FBQ25DLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsTUFBTSxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDO1lBQ3ZDLElBQU0sS0FBSyxHQUFHLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUN4QixFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUNqQixNQUFNLENBQUMsZ0JBQU0sQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUM7WUFDekIsQ0FBQztZQUNELEVBQUUsQ0FBQyxDQUFDLEtBQUssR0FBRyxHQUFHLENBQUMsQ0FBQyxDQUFDO2dCQUNoQixHQUFHLEdBQUcsS0FBSyxDQUFDO2dCQUNaLFFBQVEsR0FBRyxDQUFDLENBQUM7WUFDZixDQUFDO1FBQ0gsQ0FBQztRQUNELE1BQU0sQ0FBQyxnQkFBTSxDQUFDLEdBQUcsQ0FBQyxRQUFRLENBQUMsQ0FBQztJQUM5QixDQUFDO0lBRVMsdUNBQWMsR0FBeEIsVUFBeUIsT0FBZ0I7UUFDdkMsSUFBSSxHQUFHLEdBQUcsTUFBTSxDQUFDLGlCQUFpQixDQUFDO1FBQ25DLElBQUksUUFBUSxHQUFHLENBQUMsQ0FBQyxDQUFDO1FBQ2xCLElBQU0sTUFBTSxHQUFHLE9BQU8sQ0FBQyxTQUFTLEVBQUUsQ0FBQztRQUNuQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE1BQU0sQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQztZQUN2QyxJQUFNLEtBQUssR0FBRyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDeEIsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDakIsTUFBTSxDQUFDLGdCQUFNLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1lBQ3pCLENBQUM7WUFDRCxFQUFFLENBQUMsQ0FBQyxLQUFLLEdBQUcsR0FBRyxDQUFDLENBQUMsQ0FBQztnQkFDaEIsR0FBRyxHQUFHLEtBQUssQ0FBQztnQkFDWixRQUFRLEdBQUcsQ0FBQyxDQUFDO1lBQ2YsQ0FBQztRQUNILENBQUM7UUFDRCxNQUFNLENBQUMsZ0JBQU0sQ0FBQyxHQUFHLENBQUMsUUFBUSxDQUFDLENBQUM7SUFDOUIsQ0FBQztJQUVTLDZDQUFvQixHQUE5QixVQUErQixFQUFXLEVBQUUsRUFBVztRQUNyRCxJQUFNLE9BQU8sR0FBRyxJQUFJLENBQUMsY0FBYyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDO1FBQzlDLElBQU0sT0FBTyxHQUFHLElBQUksQ0FBQyxjQUFjLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUM7UUFDOUMsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLE9BQU8sQ0FBQyxJQUFJLEtBQUssQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDckMsTUFBTSxDQUFDLGdCQUFNLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBQ3pCLENBQUM7UUFDRCxNQUFNLENBQUMsZ0JBQU0sQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLE9BQU8sS0FBSyxPQUFPLENBQUMsQ0FBQyxDQUFDO0lBQzVDLENBQUM7SUFFUyxxQ0FBWSxHQUF0QixVQUF1QixPQUFnQixFQUFFLENBQVM7UUFFaEQsSUFBTSxNQUFNLEdBQUcsT0FBTyxDQUFDLFNBQVMsRUFBRSxDQUFDO1FBQ25DLElBQU0sZ0JBQWdCLEdBQTBDLEVBQUUsQ0FBQztRQUNuRSxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE1BQU0sQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztZQUN2QyxnQkFBZ0IsQ0FBQyxJQUFJLENBQUMsRUFBQyxLQUFLLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQyxFQUFFLEtBQUssRUFBRSxDQUFDLEVBQUMsQ0FBQyxDQUFDO1FBQ3RELENBQUM7UUFDRCxnQkFBZ0IsQ0FBQyxJQUFJLENBQUMsVUFBQyxDQUFDLEVBQUUsQ0FBQztZQUN6QixNQUFNLENBQUMsQ0FBQyxDQUFDLEtBQUssR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDO1FBQzNCLENBQUMsQ0FBQyxDQUFDO1FBQ0gsSUFBTSxVQUFVLEdBQUcsSUFBSSxZQUFZLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDdkMsSUFBTSxXQUFXLEdBQUcsSUFBSSxZQUFZLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDeEMsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztZQUMzQixVQUFVLENBQUMsQ0FBQyxDQUFDLEdBQUcsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDO1lBQzFDLFdBQVcsQ0FBQyxDQUFDLENBQUMsR0FBRyxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUM7UUFDN0MsQ0FBQztRQUNELE1BQU0sQ0FBQyxFQUFDLE1BQU0sRUFBRSxpQkFBTyxDQUFDLEdBQUcsQ0FBQyxVQUFVLENBQUMsRUFBRSxPQUFPLEVBQUUsaUJBQU8sQ0FBQyxHQUFHLENBQUMsV0FBVyxDQUFDLEVBQUMsQ0FBQztJQUM5RSxDQUFDO0lBRVMsb0NBQVcsR0FBckIsVUFBc0IsT0FBZ0I7UUFDcEMsSUFBTSxNQUFNLEdBQUcsT0FBTyxDQUFDLFNBQVMsRUFBRSxDQUFDO1FBQ25DLElBQUksR0FBRyxHQUFHLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNwQixHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE1BQU0sQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQztZQUN2QyxJQUFNLEtBQUssR0FBRyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDeEIsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDakIsTUFBTSxDQUFDLGdCQUFNLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1lBQ3pCLENBQUM7WUFDRCxFQUFFLENBQUMsQ0FBQyxLQUFLLEdBQUcsR0FBRyxDQUFDLENBQUMsQ0FBQztnQkFDaEIsR0FBRyxHQUFHLEtBQUssQ0FBQztZQUNkLENBQUM7UUFDSCxDQUFDO1FBQ0QsTUFBTSxDQUFDLGdCQUFNLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxDQUFDO0lBQ3pCLENBQUM7SUFFUyxvQ0FBVyxHQUFyQixVQUFzQixPQUFnQjtRQUNwQyxJQUFNLE1BQU0sR0FBRyxPQUFPLENBQUMsU0FBUyxFQUFFLENBQUM7UUFDbkMsSUFBSSxHQUFHLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3BCLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsTUFBTSxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDO1lBQ3ZDLElBQU0sS0FBSyxHQUFHLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUN4QixFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUNqQixNQUFNLENBQUMsZ0JBQU0sQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUM7WUFDekIsQ0FBQztZQUNELEVBQUUsQ0FBQyxDQUFDLEtBQUssR0FBRyxHQUFHLENBQUMsQ0FBQyxDQUFDO2dCQUNoQixHQUFHLEdBQUcsS0FBSyxDQUFDO1lBQ2QsQ0FBQztRQUNILENBQUM7UUFDRCxNQUFNLENBQUMsZ0JBQU0sQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUM7SUFDekIsQ0FBQztJQUVTLG9DQUFXLEdBQXJCLFVBQXlDLE9BQVU7UUFDakQsSUFBTSxNQUFNLEdBQUcsT0FBTyxDQUFDLFNBQVMsRUFBRSxDQUFDO1FBQ25DLElBQU0sU0FBUyxHQUFHLElBQUksWUFBWSxDQUFDLE1BQU0sQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUNsRCxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE1BQU0sQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQztZQUN2QyxTQUFTLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNyQyxDQUFDO1FBQ0QsTUFBTSxDQUFDLGlCQUFPLENBQUMsSUFBSSxDQUFJLE9BQU8sQ0FBQyxLQUFLLEVBQUUsRUFBQyxNQUFNLEVBQUUsU0FBUyxFQUFDLENBQUMsQ0FBQztJQUM3RCxDQUFDO0lBRVMsb0NBQVcsR0FBckIsVUFBeUMsT0FBVTtRQUNqRCxJQUFNLE1BQU0sR0FBRyxPQUFPLENBQUMsU0FBUyxFQUFFLENBQUM7UUFDbkMsSUFBTSxTQUFTLEdBQUcsSUFBSSxZQUFZLENBQUMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQ2xELEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsTUFBTSxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDO1lBQ3ZDLElBQU0sS0FBSyxHQUFHLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUN4QixTQUFTLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUNqQyxDQUFDO1FBQ0QsTUFBTSxDQUFDLGlCQUFPLENBQUMsSUFBSSxDQUFJLE9BQU8sQ0FBQyxLQUFLLEVBQUUsRUFBQyxNQUFNLEVBQUUsU0FBUyxFQUFDLENBQUMsQ0FBQztJQUM3RCxDQUFDO0lBRVMsMENBQWlCLEdBQTNCLFVBQTRCLE9BQWdCO1FBQzFDLElBQU0sSUFBSSxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsT0FBTyxDQUFDLENBQUM7UUFDL0IsSUFBTSxDQUFDLEdBQUcsSUFBSSxDQUFDLGdCQUFnQixDQUFDLE9BQU8sRUFBRSxJQUFJLENBQUMsQ0FBQztRQUMvQyxJQUFNLENBQUMsR0FBRyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3RCLElBQU0sQ0FBQyxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDdEIsSUFBTSxDQUFDLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUN0QixJQUFNLE1BQU0sR0FBRyxJQUFJLENBQUMsR0FBRyxDQUFDLElBQUksRUFBRSxDQUFDLENBQUMsQ0FBQztRQUVqQyxJQUFJLENBQUMsT0FBTyxFQUFFLENBQUM7UUFDZixDQUFDLENBQUMsT0FBTyxFQUFFLENBQUM7UUFDWixDQUFDLENBQUMsT0FBTyxFQUFFLENBQUM7UUFDWixDQUFDLENBQUMsT0FBTyxFQUFFLENBQUM7UUFDWixDQUFDLENBQUMsT0FBTyxFQUFFLENBQUM7UUFFWixNQUFNLENBQUMsTUFBTSxDQUFDO0lBQ2hCLENBQUM7SUFFUyxxQ0FBWSxHQUF0QixVQUEwQyxPQUFVO1FBQ2xELElBQU0sWUFBWSxHQUFHLElBQUksWUFBWSxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUNwRCxJQUFNLE1BQU0sR0FBRyxPQUFPLENBQUMsU0FBUyxFQUFFLENBQUM7UUFDbkMsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxNQUFNLENBQUMsTUFBTSxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUM7WUFDdkMsWUFBWSxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQzNDLENBQUM7UUFDRCxNQUFNLENBQUMsaUJBQU8sQ0FBQyxJQUFJLENBQUksT0FBTyxDQUFDLEtBQUssRUFBRSxFQUFDLE1BQU0sRUFBRSxZQUFZLEVBQUMsQ0FBQyxDQUFDO0lBQ2hFLENBQUM7SUFFUyx3Q0FBZSxHQUF6QixVQUE2QyxPQUFVO1FBQ3JELElBQU0sWUFBWSxHQUFHLElBQUksWUFBWSxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUNwRCxJQUFNLE1BQU0sR0FBRyxPQUFPLENBQUMsU0FBUyxFQUFFLENBQUM7UUFDbkMsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxNQUFNLENBQUMsTUFBTSxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUM7WUFDdkMsWUFBWSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNuRCxDQUFDO1FBQ0QsTUFBTSxDQUFDLGlCQUFPLENBQUMsSUFBSSxDQUFJLE9BQU8sQ0FBQyxLQUFLLEVBQUUsRUFBQyxNQUFNLEVBQUUsWUFBWSxFQUFDLENBQUMsQ0FBQztJQUNoRSxDQUFDO0lBRVMscUNBQVksR0FBdEIsVUFBMEMsT0FBVTtRQUNsRCxJQUFNLFlBQVksR0FBRyxJQUFJLFlBQVksQ0FBQyxPQUFPLENBQUMsSUFBSSxDQUFDLENBQUM7UUFDcEQsSUFBTSxNQUFNLEdBQUcsT0FBTyxDQUFDLFNBQVMsRUFBRSxDQUFDO1FBQ25DLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsTUFBTSxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDO1lBQ3ZDLFlBQVksQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3pDLENBQUM7UUFDRCxNQUFNLENBQUMsaUJBQU8sQ0FBQyxJQUFJLENBQUksT0FBTyxDQUFDLEtBQUssRUFBRSxFQUFDLE1BQU0sRUFBRSxZQUFZLEVBQUMsQ0FBQyxDQUFDO0lBQ2hFLENBQUM7SUFFUyxvQ0FBVyxHQUFyQixVQUF5QyxPQUFVO1FBQ2pELElBQU0sWUFBWSxHQUFHLElBQUksWUFBWSxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUNwRCxJQUFNLE1BQU0sR0FBRyxPQUFPLENBQUMsU0FBUyxFQUFFLENBQUM7UUFDbkMsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxNQUFNLENBQUMsTUFBTSxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUM7WUFDdkMsWUFBWSxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDeEMsQ0FBQztRQUNELE1BQU0sQ0FBQyxpQkFBTyxDQUFDLElBQUksQ0FBSSxPQUFPLENBQUMsS0FBSyxFQUFFLEVBQUMsTUFBTSxFQUFFLFlBQVksRUFBQyxDQUFDLENBQUM7SUFDaEUsQ0FBQztJQUVTLHFDQUFZLEdBQXRCLFVBQTBDLE9BQVU7UUFDbEQsSUFBTSxZQUFZLEdBQUcsSUFBSSxZQUFZLENBQUMsT0FBTyxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ3BELElBQU0sTUFBTSxHQUFHLE9BQU8sQ0FBQyxTQUFTLEVBQUUsQ0FBQztRQUNuQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE1BQU0sQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQztZQUN2QyxJQUFNLEtBQUssR0FBRyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDeEIsWUFBWSxDQUFDLENBQUMsQ0FBQyxHQUFHLEtBQUssR0FBRyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsS0FBSyxHQUFHLENBQUMsR0FBRyxDQUFDLEdBQUcsS0FBSyxDQUFDLENBQUM7UUFDNUQsQ0FBQztRQUNELE1BQU0sQ0FBQyxpQkFBTyxDQUFDLElBQUksQ0FBSSxPQUFPLENBQUMsS0FBSyxFQUFFLEVBQUMsTUFBTSxFQUFFLFlBQVksRUFBQyxDQUFDLENBQUM7SUFDaEUsQ0FBQztJQU1TLHVDQUFjLEdBQXhCLFVBQ0ksQ0FBVSxFQUFFLE9BQWdCLEVBQUUsTUFBb0IsRUFBRSxNQUFjLEVBQ2xFLEdBQVc7UUFDUCxJQUFBLFlBQW9DLEVBQW5DLGFBQUssRUFBRSxhQUFLLEVBQUUsa0JBQVUsQ0FBWTtRQUMzQyxJQUFNLFNBQVMsR0FBRyxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ25DLElBQU0sV0FBVyxHQUFHLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDckMsSUFBTSxXQUFXLEdBQUcsU0FBUyxDQUFDLG9CQUFvQixDQUM5QyxDQUFDLEtBQUssRUFBRSxLQUFLLEVBQUUsVUFBVSxDQUFDLEVBQUUsU0FBUyxFQUFFLFdBQVcsRUFBRSxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUM7UUFDckUsSUFBTSxDQUFDLEdBQUcsaUJBQU8sQ0FBQyxLQUFLLENBQUMsV0FBVyxDQUFDLENBQUM7UUFDckMsR0FBRyxDQUFDLENBQUMsSUFBSSxFQUFFLEdBQUcsQ0FBQyxFQUFFLEVBQUUsR0FBRyxXQUFXLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQztZQUN4QyxHQUFHLENBQUMsQ0FBQyxJQUFJLEVBQUUsR0FBRyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQztnQkFDdkMsSUFBTSxRQUFRLEdBQUcsRUFBRSxHQUFHLE1BQU0sR0FBRyxHQUFHLENBQUM7Z0JBQ25DLElBQU0sS0FBSyxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLFFBQVEsQ0FBQyxDQUFDO2dCQUNwQyxJQUFNLEtBQUssR0FBRyxJQUFJLENBQUMsR0FBRyxDQUFDLEtBQUssRUFBRSxTQUFTLEdBQUcsUUFBUSxDQUFDLENBQUM7Z0JBQ3BELEdBQUcsQ0FBQyxDQUFDLElBQUksRUFBRSxHQUFHLENBQUMsRUFBRSxFQUFFLEdBQUcsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDO29CQUN2QyxJQUFNLFFBQVEsR0FBRyxFQUFFLEdBQUcsTUFBTSxHQUFHLEdBQUcsQ0FBQztvQkFDbkMsSUFBTSxLQUFLLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsUUFBUSxDQUFDLENBQUM7b0JBQ3BDLElBQU0sS0FBSyxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsS0FBSyxFQUFFLFNBQVMsR0FBRyxRQUFRLENBQUMsQ0FBQztvQkFDcEQsSUFBSSxPQUFPLEdBQUcsQ0FBQyxDQUFDO29CQUNoQixHQUFHLENBQUMsQ0FBQyxJQUFJLEVBQUUsR0FBRyxLQUFLLEVBQUUsRUFBRSxHQUFHLEtBQUssRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDO3dCQUN0QyxJQUFNLEVBQUUsR0FBRyxFQUFFLEdBQUcsUUFBUSxDQUFDO3dCQUN6QixHQUFHLENBQUMsQ0FBQyxJQUFJLEVBQUUsR0FBRyxLQUFLLEVBQUUsRUFBRSxHQUFHLEtBQUssRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDOzRCQUN0QyxJQUFNLEVBQUUsR0FBRyxFQUFFLEdBQUcsUUFBUSxDQUFDOzRCQUN6QixHQUFHLENBQUMsQ0FBQyxJQUFJLEVBQUUsR0FBRyxDQUFDLEVBQUUsRUFBRSxHQUFHLFVBQVUsRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDO2dDQUN2QyxJQUFNLEtBQUssR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDLENBQUM7Z0NBQ2hDLElBQU0sTUFBTSxHQUFHLE9BQU8sQ0FBQyxHQUFHLENBQUMsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDLENBQUM7Z0NBQzNDLE9BQU8sSUFBSSxLQUFLLEdBQUcsTUFBTSxDQUFDOzRCQUM1QixDQUFDO3dCQUNILENBQUM7b0JBQ0gsQ0FBQztvQkFDRCxJQUFNLElBQUksR0FBRyxDQUFDLE1BQU0sSUFBSSxJQUFJLENBQUMsR0FBRyxNQUFNLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLENBQUMsQ0FBQztvQkFDbkQsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxPQUFPLEdBQUcsSUFBSSxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDLENBQUM7Z0JBQ3BDLENBQUM7WUFDSCxDQUFDO1FBQ0gsQ0FBQztRQUNELE1BQU0sQ0FBQyxDQUFDLENBQUM7SUFDWCxDQUFDO0lBRVMsK0NBQXNCLEdBQWhDLFVBQ0ksQ0FBVSxFQUFFLEVBQVcsRUFBRSxPQUFnQixFQUFFLE1BQWMsRUFDekQsR0FBVztRQUNiLElBQU0sS0FBSyxHQUFHLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDL0IsSUFBTSxFQUFFLEdBQUcsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUMsRUFBRSxFQUFFLEVBQUUsS0FBSyxFQUFFLE1BQU0sRUFBRSxHQUFHLENBQUMsQ0FBQztRQUM1RCxJQUFNLEVBQUUsR0FBRyxJQUFJLENBQUMsYUFBYSxDQUFDLEVBQUUsQ0FBQyxDQUFDO1FBQ2xDLElBQU0sRUFBRSxHQUFHLElBQUksQ0FBQyx1QkFBdUIsQ0FBQyxFQUFFLEVBQUUsT0FBTyxFQUFFLElBQUksRUFBRSxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUM7UUFDeEUsTUFBTSxDQUFDLEVBQUMsRUFBRSxJQUFBLEVBQUUsRUFBRSxJQUFBLEVBQUUsRUFBRSxJQUFBLEVBQUMsQ0FBQztJQUN0QixDQUFDO0lBTVMsZ0RBQXVCLEdBQWpDLFVBQ0ksQ0FBVSxFQUFFLE9BQWdCLEVBQUUsTUFBb0IsRUFBRSxVQUFrQixFQUN0RSxPQUFlO1FBQ2pCLElBQU0sS0FBSyxHQUFHLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDL0IsSUFBTSxHQUFHLEdBQUcsS0FBSyxHQUFHLENBQUMsR0FBRyxPQUFPLENBQUM7UUFDaEMsSUFBTSxjQUFjLEdBQUcsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUN4QyxJQUFNLGVBQWUsR0FBRyxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ25DLElBQUEsWUFBZ0MsRUFBL0IsYUFBSyxFQUFFLGFBQUssRUFBRSxjQUFNLENBQVk7UUFHdkMsSUFBTSxZQUFZLEdBQUcsQ0FBQyxLQUFLLEdBQUcsQ0FBQyxDQUFDLEdBQUcsVUFBVSxHQUFHLENBQUMsQ0FBQztRQUNsRCxJQUFNLFlBQVksR0FBRyxDQUFDLEtBQUssR0FBRyxDQUFDLENBQUMsR0FBRyxVQUFVLEdBQUcsQ0FBQyxDQUFDO1FBRWxELElBQU0sV0FBVyxHQUFHLFNBQVMsQ0FBQyxvQkFBb0IsQ0FDOUMsQ0FBQyxZQUFZLEVBQUUsWUFBWSxFQUFFLGVBQWUsQ0FBQyxFQUFFLEtBQUssRUFBRSxjQUFjLEVBQUUsQ0FBQyxFQUN2RSxHQUFHLENBQUMsQ0FBQztRQUNULElBQU0sQ0FBQyxHQUFHLGlCQUFPLENBQUMsS0FBSyxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBQ3JDLEdBQUcsQ0FBQyxDQUFDLElBQUksRUFBRSxHQUFHLENBQUMsRUFBRSxFQUFFLEdBQUcsY0FBYyxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUM7WUFDM0MsR0FBRyxDQUFDLENBQUMsSUFBSSxFQUFFLEdBQUcsQ0FBQyxFQUFFLEVBQUUsR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUM7Z0JBQ3ZDLElBQU0sUUFBUSxHQUFHLEVBQUUsR0FBRyxHQUFHLENBQUM7Z0JBQzFCLElBQU0sS0FBSyxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxHQUFHLFVBQVUsQ0FBQyxDQUFDLENBQUM7Z0JBQzVELElBQU0sS0FBSyxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsS0FBSyxFQUFFLENBQUMsS0FBSyxHQUFHLFFBQVEsQ0FBQyxHQUFHLFVBQVUsQ0FBQyxDQUFDO2dCQUUvRCxHQUFHLENBQUMsQ0FBQyxJQUFJLEVBQUUsR0FBRyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQztvQkFDdkMsSUFBTSxRQUFRLEdBQUcsRUFBRSxHQUFHLEdBQUcsQ0FBQztvQkFDMUIsSUFBTSxLQUFLLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLEdBQUcsVUFBVSxDQUFDLENBQUMsQ0FBQztvQkFDNUQsSUFBTSxLQUFLLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxLQUFLLEdBQUcsUUFBUSxDQUFDLEdBQUcsVUFBVSxDQUFDLENBQUM7b0JBRS9ELElBQUksT0FBTyxHQUFHLENBQUMsQ0FBQztvQkFDaEIsR0FBRyxDQUFDLENBQUMsSUFBSSxFQUFFLEdBQUcsS0FBSyxFQUFFLEVBQUUsR0FBRyxLQUFLLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQzt3QkFDdEMsSUFBTSxFQUFFLEdBQUcsRUFBRSxHQUFHLFVBQVUsR0FBRyxRQUFRLENBQUM7d0JBRXRDLEdBQUcsQ0FBQyxDQUFDLElBQUksRUFBRSxHQUFHLEtBQUssRUFBRSxFQUFFLEdBQUcsS0FBSyxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUM7NEJBQ3RDLElBQU0sRUFBRSxHQUFHLEVBQUUsR0FBRyxVQUFVLEdBQUcsUUFBUSxDQUFDOzRCQUV0QyxHQUFHLENBQUMsQ0FBQyxJQUFJLEVBQUUsR0FBRyxDQUFDLEVBQUUsRUFBRSxHQUFHLGVBQWUsRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDO2dDQUM1QyxJQUFNLEtBQUssR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDLENBQUM7Z0NBQ2hDLElBQU0sTUFBTSxHQUNSLE9BQU8sQ0FBQyxHQUFHLENBQUMsS0FBSyxHQUFHLENBQUMsR0FBRyxFQUFFLEVBQUUsS0FBSyxHQUFHLENBQUMsR0FBRyxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQyxDQUFDO2dDQUN4RCxPQUFPLElBQUksS0FBSyxHQUFHLE1BQU0sQ0FBQzs0QkFDNUIsQ0FBQzt3QkFDSCxDQUFDO29CQUNILENBQUM7b0JBQ0QsSUFBTSxJQUFJLEdBQUcsTUFBTSxJQUFJLElBQUksR0FBRyxNQUFNLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLENBQUMsQ0FBQztvQkFDakQsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxPQUFPLEdBQUcsSUFBSSxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDLENBQUM7Z0JBQ3BDLENBQUM7WUFDSCxDQUFDO1FBQ0gsQ0FBQztRQUNELE1BQU0sQ0FBQyxDQUFDLENBQUM7SUFDWCxDQUFDO0lBTVMsa0RBQXlCLEdBQW5DLFVBQ0ksQ0FBVSxFQUFFLFdBQW9CLEVBQUUsVUFBa0IsRUFDcEQsT0FBZTtRQUNqQixJQUFNLEtBQUssR0FBRyxXQUFXLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ25DLElBQU0sR0FBRyxHQUFHLEtBQUssR0FBRyxDQUFDLEdBQUcsT0FBTyxDQUFDO1FBQ2hDLElBQU0sY0FBYyxHQUFHLFdBQVcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDNUMsSUFBTSxlQUFlLEdBQUcsV0FBVyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUN2QyxJQUFBLFlBQWdDLEVBQS9CLGFBQUssRUFBRSxhQUFLLEVBQUUsY0FBTSxDQUFZO1FBR3ZDLElBQU0sWUFBWSxHQUFHLENBQUMsS0FBSyxHQUFHLENBQUMsQ0FBQyxHQUFHLFVBQVUsR0FBRyxDQUFDLENBQUM7UUFDbEQsSUFBTSxZQUFZLEdBQUcsQ0FBQyxLQUFLLEdBQUcsQ0FBQyxDQUFDLEdBQUcsVUFBVSxHQUFHLENBQUMsQ0FBQztRQUVsRCxJQUFNLFdBQVcsR0FBRyxTQUFTLENBQUMsb0JBQW9CLENBQzlDLENBQUMsWUFBWSxFQUFFLFlBQVksRUFBRSxlQUFlLENBQUMsRUFBRSxLQUFLLEVBQUUsY0FBYyxFQUFFLENBQUMsRUFDdkUsR0FBRyxDQUFDLENBQUM7UUFDVCxJQUFNLENBQUMsR0FBRyxpQkFBTyxDQUFDLEtBQUssQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUVyQyxHQUFHLENBQUMsQ0FBQyxJQUFJLEVBQUUsR0FBRyxDQUFDLEVBQUUsRUFBRSxHQUFHLGNBQWMsRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDO1lBQzNDLEdBQUcsQ0FBQyxDQUFDLElBQUksRUFBRSxHQUFHLENBQUMsRUFBRSxFQUFFLEdBQUcsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDO2dCQUN2QyxHQUFHLENBQUMsQ0FBQyxJQUFJLEVBQUUsR0FBRyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQztvQkFFdkMsSUFBTSxRQUFRLEdBQUcsRUFBRSxHQUFHLEdBQUcsQ0FBQztvQkFDMUIsSUFBTSxRQUFRLEdBQUcsRUFBRSxHQUFHLEdBQUcsQ0FBQztvQkFDMUIsSUFBSSxPQUFPLEdBQUcsQ0FBQyxDQUFDO29CQUNoQixHQUFHLENBQUMsQ0FBQyxJQUFJLEVBQUUsR0FBRyxDQUFDLEVBQUUsRUFBRSxHQUFHLEtBQUssRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDO3dCQUNsQyxJQUFNLEVBQUUsR0FBRyxDQUFDLFFBQVEsR0FBRyxFQUFFLENBQUMsR0FBRyxVQUFVLENBQUM7d0JBQ3hDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLElBQUksRUFBRSxJQUFJLEtBQUssSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFDLEVBQUUsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLENBQUM7NEJBQ25ELFFBQVEsQ0FBQzt3QkFDWCxDQUFDO3dCQUNELEdBQUcsQ0FBQyxDQUFDLElBQUksRUFBRSxHQUFHLENBQUMsRUFBRSxFQUFFLEdBQUcsS0FBSyxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUM7NEJBQ2xDLElBQU0sRUFBRSxHQUFHLENBQUMsUUFBUSxHQUFHLEVBQUUsQ0FBQyxHQUFHLFVBQVUsQ0FBQzs0QkFDeEMsRUFBRSxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsSUFBSSxFQUFFLElBQUksS0FBSyxJQUFJLElBQUksQ0FBQyxLQUFLLENBQUMsRUFBRSxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUMsQ0FBQztnQ0FDbkQsUUFBUSxDQUFDOzRCQUNYLENBQUM7NEJBQ0QsR0FBRyxDQUFDLENBQUMsSUFBSSxFQUFFLEdBQUcsQ0FBQyxFQUFFLEVBQUUsR0FBRyxlQUFlLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQztnQ0FDNUMsSUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQyxDQUFDO2dDQUNoQyxJQUFNLE1BQU0sR0FDUixXQUFXLENBQUMsR0FBRyxDQUFDLEtBQUssR0FBRyxDQUFDLEdBQUcsRUFBRSxFQUFFLEtBQUssR0FBRyxDQUFDLEdBQUcsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUMsQ0FBQztnQ0FDNUQsT0FBTyxJQUFJLEtBQUssR0FBRyxNQUFNLENBQUM7NEJBQzVCLENBQUM7d0JBQ0gsQ0FBQztvQkFDSCxDQUFDO29CQUNELENBQUMsQ0FBQyxHQUFHLENBQUMsT0FBTyxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDLENBQUM7Z0JBQzdCLENBQUM7WUFDSCxDQUFDO1FBQ0gsQ0FBQztRQUNELE1BQU0sQ0FBQyxDQUFDLENBQUM7SUFDWCxDQUFDO0lBRUQseUNBQWdCLEdBQWhCLFVBQ0ksQ0FBVSxFQUFFLEVBQVcsRUFBRSxLQUFhLEVBQUUsTUFBYyxFQUN0RCxPQUFlO1FBQ2pCLElBQU0sVUFBVSxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDOUIsSUFBTSxXQUFXLEdBQUcsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNoQyxJQUFNLFlBQVksR0FDZCxTQUFTLENBQUMscUJBQXFCLENBQUMsVUFBVSxFQUFFLFdBQVcsRUFBRSxLQUFLLENBQUMsQ0FBQztRQUNwRSxJQUFNLEVBQUUsR0FBRyxpQkFBTyxDQUFDLEtBQUssQ0FBQyxZQUFZLENBQUMsQ0FBQztRQUV2QyxJQUFNLFFBQVEsR0FBRyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQzdCLElBQU0sUUFBUSxHQUFHLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDN0IsSUFBTSxRQUFRLEdBQUcsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUM1QixJQUFNLFFBQVEsR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBRTVCLEdBQUcsQ0FBQyxDQUFDLElBQUksRUFBRSxHQUFHLENBQUMsRUFBRSxFQUFFLEdBQUcsS0FBSyxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUM7WUFDbEMsSUFBTSxLQUFLLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLE9BQU8sR0FBRyxFQUFFLENBQUMsR0FBRyxNQUFNLENBQUMsQ0FBQyxDQUFDO1lBQzlELElBQU0sS0FBSyxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsUUFBUSxFQUFFLENBQUMsUUFBUSxHQUFHLE9BQU8sR0FBRyxFQUFFLENBQUMsR0FBRyxNQUFNLENBQUMsQ0FBQztZQUVyRSxHQUFHLENBQUMsQ0FBQyxJQUFJLEVBQUUsR0FBRyxDQUFDLEVBQUUsRUFBRSxHQUFHLEtBQUssRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDO2dCQUNsQyxJQUFNLEtBQUssR0FBRyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUMsT0FBTyxHQUFHLEVBQUUsQ0FBQyxHQUFHLE1BQU0sQ0FBQyxDQUFDLENBQUM7Z0JBQzlELElBQU0sS0FBSyxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsUUFBUSxFQUFFLENBQUMsUUFBUSxHQUFHLE9BQU8sR0FBRyxFQUFFLENBQUMsR0FBRyxNQUFNLENBQUMsQ0FBQztnQkFFckUsR0FBRyxDQUFDLENBQUMsSUFBSSxFQUFFLEdBQUcsQ0FBQyxFQUFFLEVBQUUsR0FBRyxVQUFVLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQztvQkFDdkMsR0FBRyxDQUFDLENBQUMsSUFBSSxFQUFFLEdBQUcsQ0FBQyxFQUFFLEVBQUUsR0FBRyxXQUFXLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQzt3QkFFeEMsSUFBSSxPQUFPLEdBQUcsQ0FBQyxDQUFDO3dCQUNoQixHQUFHLENBQUMsQ0FBQyxJQUFJLEVBQUUsR0FBRyxLQUFLLEVBQUUsRUFBRSxHQUFHLEtBQUssRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDOzRCQUN0QyxJQUFNLEVBQUUsR0FBRyxFQUFFLEdBQUcsRUFBRSxHQUFHLE1BQU0sR0FBRyxPQUFPLENBQUM7NEJBQ3RDLEdBQUcsQ0FBQyxDQUFDLElBQUksRUFBRSxHQUFHLEtBQUssRUFBRSxFQUFFLEdBQUcsS0FBSyxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUM7Z0NBQ3RDLElBQU0sRUFBRSxHQUFHLEVBQUUsR0FBRyxFQUFFLEdBQUcsTUFBTSxHQUFHLE9BQU8sQ0FBQztnQ0FDdEMsT0FBTyxJQUFJLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUMsR0FBRyxFQUFFLENBQUMsR0FBRyxDQUFDLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDLENBQUM7NEJBQ3BELENBQUM7d0JBQ0gsQ0FBQzt3QkFDRCxFQUFFLENBQUMsR0FBRyxDQUFDLE9BQU8sRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUMsQ0FBQztvQkFDbEMsQ0FBQztnQkFDSCxDQUFDO1lBQ0gsQ0FBQztRQUNILENBQUM7UUFDRCxNQUFNLENBQUMsRUFBRSxDQUFDO0lBQ1osQ0FBQztJQUVELHNDQUFhLEdBQWIsVUFBYyxFQUFXO1FBQ3ZCLElBQU0sV0FBVyxHQUFHLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDaEMsSUFBTSxPQUFPLEdBQUcsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUM1QixJQUFNLE9BQU8sR0FBRyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQzVCLElBQU0sTUFBTSxHQUFHLElBQUksWUFBWSxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBQzdDLEdBQUcsQ0FBQyxDQUFDLElBQUksRUFBRSxHQUFHLENBQUMsRUFBRSxFQUFFLEdBQUcsV0FBVyxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUM7WUFDeEMsSUFBSSxHQUFHLEdBQUcsQ0FBQyxDQUFDO1lBQ1osR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxPQUFPLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQztnQkFDakMsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxPQUFPLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQztvQkFDakMsR0FBRyxJQUFJLEVBQUUsQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUMsQ0FBQztnQkFDMUIsQ0FBQztZQUNILENBQUM7WUFDRCxNQUFNLENBQUMsRUFBRSxDQUFDLEdBQUcsR0FBRyxDQUFDO1FBQ25CLENBQUM7UUFDRCxNQUFNLENBQUMsaUJBQU8sQ0FBQyxHQUFHLENBQUMsTUFBTSxDQUFDLENBQUM7SUFDN0IsQ0FBQztJQUVTLDBDQUFpQixHQUEzQixVQUErQyxDQUFJLEVBQUUsTUFBZ0I7UUFDbkUsSUFBTSxRQUFRLEdBQWEsSUFBSSxLQUFLLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQzdDLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsUUFBUSxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1lBQ3pDLFFBQVEsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ25DLENBQUM7UUFDRCxJQUFNLFlBQVksR0FBRyxJQUFJLFlBQVksQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUM7UUFDOUMsSUFBTSxNQUFNLEdBQUcsQ0FBQyxDQUFDLFNBQVMsRUFBRSxDQUFDO1FBQzdCLElBQU0sTUFBTSxHQUFHLGlCQUFPLENBQUMsSUFBSSxDQUFJLFFBQVEsRUFBRSxFQUFDLE1BQU0sRUFBRSxZQUFZLEVBQUMsQ0FBQyxDQUFDO1FBQ2pFLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsQ0FBQyxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDO1lBQ2hDLElBQU0sR0FBRyxHQUFHLENBQUMsQ0FBQyxVQUFVLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFHNUIsSUFBTSxNQUFNLEdBQWEsSUFBSSxLQUFLLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1lBQy9DLEdBQUcsQ0FBQyxDQUFDLElBQUksR0FBQyxHQUFHLENBQUMsRUFBRSxHQUFDLEdBQUcsTUFBTSxDQUFDLE1BQU0sRUFBRSxHQUFDLEVBQUUsRUFBRSxDQUFDO2dCQUN2QyxNQUFNLENBQUMsR0FBQyxDQUFDLEdBQUcsR0FBRyxDQUFDLE1BQU0sQ0FBQyxHQUFDLENBQUMsQ0FBQyxDQUFDO1lBQzdCLENBQUM7WUFFRCxJQUFNLFFBQVEsR0FBRyxNQUFNLENBQUMsVUFBVSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1lBQzNDLFlBQVksQ0FBQyxRQUFRLENBQUMsR0FBRyxNQUFNLENBQUMsQ0FBQyxD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LEdBQUcsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUM7UUFDM0IsZUFBZSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQztJQUN6QixDQUFDO0lBQ0QsT0FBTyxDQUFDLGVBQWUsQ0FBQyxDQUFDO0lBQ3pCLE1BQU0sQ0FBQyxlQUFlLENBQUM7QUFDekIsQ0FBQztBQVBELHNEQU9DIiwiZmlsZSI6ImdlbmVyYXRlZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzQ29udGVudCI6WyIoZnVuY3Rpb24gZSh0LG4scil7ZnVuY3Rpb24gcyhvLHUpe2lmKCFuW29dKXtpZighdFtvXSl7dmFyIGE9dHlwZW9mIHJlcXVpcmU9PVwiZnVuY3Rpb25cIiYmcmVxdWlyZTtpZighdSYmYSlyZXR1cm4gYShvLCEwKTtpZihpKXJldHVybiBpKG8sITApO3ZhciBmPW5ldyBFcnJvcihcIkNhbm5vdCBmaW5kIG1vZHVsZSAnXCIrbytcIidcIik7dGhyb3cgZi5jb2RlPVwiTU9EVUxFX05PVF9GT1VORFwiLGZ9dmFyIGw9bltvXT17ZXhwb3J0czp7fX07dFtvXVswXS5jYWxsKGwuZXhwb3J0cyxmdW5jdGlvbihlKXt2YXIgbj10W29dWzFdW2VdO3JldHVybiBzKG4/bjplKX0sbCxsLmV4cG9ydHMsZSx0LG4scil9cmV0dXJuIG5bb10uZXhwb3J0c312YXIgaT10eXBlb2YgcmVxdWlyZT09XCJmdW5jdGlvblwiJiZyZXF1aXJlO2Zvcih2YXIgbz0wO288ci5sZW5ndGg7bysrKXMocltvXSk7cmV0dXJuIHN9KSIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuZXhwb3J0IGludGVyZmFjZSBCZW5jaG1hcmtSdW5Hcm91cCB7XG4gIG5hbWU6IHN0cmluZztcbiAgLy8gTWluIGFuZCBtYXggc3RlcHMgdG8gcnVuIHRoZSBiZW5jaG1hcmsgdGVzdCBvdmVyLlxuICBtaW46IG51bWJlcjtcbiAgbWF4OiBudW1iZXI7XG4gIC8vIFRoZSBzaXplIG9mIHRoZSBzdGVwIHRvIHRha2UgYmV0d2VlbiBiZW5jaG1hcmsgcnVucy5cbiAgc3RlcFNpemU6IG51bWJlcjtcbiAgLy8gQSB0cmFuc2Zvcm1hdGlvbiBvZiBzdGVwIHRvIHRoZSBzaXplIHBhc3NlZCB0byB0aGUgYmVuY2htYXJrIHRlc3QuXG4gIHN0ZXBUb1NpemVUcmFuc2Zvcm1hdGlvbj86IChzdGVwOiBudW1iZXIpID0+IG51bWJlcjtcbiAgYmVuY2htYXJrUnVuczogQmVuY2htYXJrUnVuW107XG59XG5cbmV4cG9ydCBjbGFzcyBCZW5jaG1hcmtSdW4ge1xuICBuYW1lOiBzdHJpbmc7XG4gIGJlbmNobWFya1Rlc3Q6IEJlbmNobWFya1Rlc3Q7XG5cbiAgY2hhcnREYXRhOiBDaGFydERhdGFbXTtcbiAgY29uc3RydWN0b3IobmFtZTogc3RyaW5nLCBiZW5jaG1hcmtUZXN0OiBCZW5jaG1hcmtUZXN0KSB7XG4gICAgdGhpcy5uYW1lID0gbmFtZTtcbiAgICB0aGlzLmJlbmNobWFya1Rlc3QgPSBiZW5jaG1hcmtUZXN0O1xuICAgIHRoaXMuY2hhcnREYXRhID0gW107XG4gIH1cbn1cblxuZXhwb3J0IGludGVyZmFjZSBCZW5jaG1hcmtUZXN0IHsgKHNpemU6IG51bWJlcik6IG51bWJlcjsgfVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQgKiBhcyBjb252X3V0aWwgZnJvbSAnLi4vLi4vc3JjL21hdGgvY29udl91dGlsJztcbmltcG9ydCAqIGFzIGNvbnZfZ3B1IGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL2NvbnZfZ3B1JztcbmltcG9ydCB7R1BHUFVDb250ZXh0fSBmcm9tICcuLi8uLi9zcmMvbWF0aC93ZWJnbC9ncGdwdV9jb250ZXh0JztcbmltcG9ydCAqIGFzIHRlc3RfdXRpbCBmcm9tICcuLi8uLi9zcmMvdGVzdF91dGlsJztcblxuaW1wb3J0IHtCZW5jaG1hcmtUZXN0fSBmcm9tICcuL2JlbmNobWFyayc7XG5cbmNvbnN0IE9QX1JVTlMgPSAxMDA7XG5cbmV4cG9ydCBjb25zdCBCRU5DSE1BUktfVEVTVDogQmVuY2htYXJrVGVzdCA9IChzaXplOiBudW1iZXIpID0+IHtcbiAgY29uc3QgaW5wdXRTaGFwZVJDRDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gW3NpemUsIHNpemUsIDFdO1xuICBjb25zdCBvdXRwdXREZXB0aCA9IDE7XG4gIGNvbnN0IGZpZWxkU2l6ZSA9IDExO1xuICBjb25zdCBzdHJpZGUgPSAxO1xuICBjb25zdCB6ZXJvUGFkID0gY29udl91dGlsLmNvbXB1dGVEZWZhdWx0UGFkKGlucHV0U2hhcGVSQ0QsIGZpZWxkU2l6ZSwgc3RyaWRlKTtcbiAgY29uc3Qgb3V0cHV0U2hhcGVSQ0Q6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9XG4gICAgICBjb252X3V0aWwuY29tcHV0ZU91dHB1dFNoYXBlM0QoXG4gICAgICAgICAgaW5wdXRTaGFwZVJDRCwgZmllbGRTaXplLCBvdXRwdXREZXB0aCwgc3RyaWRlLCB6ZXJvUGFkKTtcblxuICBjb25zdCBpbnB1dFRleFNoYXBlUkMgPSBjb252X3V0aWwuY29tcHV0ZVRleFNoYXBlRnJvbTNEKGlucHV0U2hhcGVSQ0QpO1xuICBjb25zdCBvdXRwdXRUZXhTaGFwZVJDID0gY29udl91dGlsLmNvbXB1dGVUZXhTaGFwZUZyb20zRChvdXRwdXRTaGFwZVJDRCk7XG4gIGNvbnN0IHdlaWdodHNUZXhTaGFwZVJDID0gY29udl91dGlsLmNvbXB1dGVXZWlnaHRzVGV4U2hhcGUoXG4gICAgICBpbnB1dFNoYXBlUkNEWzJdLCBvdXRwdXREZXB0aCwgZmllbGRTaXplKTtcbiAgY29uc3QgYmlhc2VzVGV4U2hhcGVSQyA9IGNvbnZfdXRpbC5jb21wdXRlQmlhc2VzVGV4U2hhcGUob3V0cHV0RGVwdGgpO1xuXG4gIGNvbnN0IGhhc0JpYXMgPSB0cnVlO1xuICBjb25zdCBncGdwdSA9IG5ldyBHUEdQVUNvbnRleHQoKTtcbiAgY29uc3QgcHJvZ3JhbSA9IGdwZ3B1LmNyZWF0ZVByb2dyYW0oY29udl9ncHUuZ2V0RnJhZ21lbnRTaGFkZXJTb3VyY2UoXG4gICAgICBpbnB1dFNoYXBlUkNELCBvdXRwdXREZXB0aCwgZmllbGRTaXplLCBzdHJpZGUsIHplcm9QYWQsIGhhc0JpYXMpKTtcblxuICBjb25zdCBpbnB1dFRleHR1cmUgPVxuICAgICAgZ3BncHUuY3JlYXRlTWF0cml4VGV4dHVyZShpbnB1dFRleFNoYXBlUkNbMF0sIGlucHV0VGV4U2hhcGVSQ1sxXSk7XG4gIGNvbnN0IHdlaWdodHNUZXh0dXJlID1cbiAgICAgIGdwZ3B1LmNyZWF0ZU1hdHJpeFRleHR1cmUod2VpZ2h0c1RleFNoYXBlUkNbMF0sIHdlaWdodHNUZXhTaGFwZVJDWzFdKTtcbiAgY29uc3QgYmlhc2VzVGV4dHVyZSA9XG4gICAgICBncGdwdS5jcmVhdGVNYXRyaXhUZXh0dXJlKGJpYXNlc1RleFNoYXBlUkNbMF0sIGJpYXNlc1RleFNoYXBlUkNbMV0pO1xuICBjb25zdCBvdXRwdXRUZXh0dXJlID1cbiAgICAgIGdwZ3B1LmNyZWF0ZU1hdHJpeFRleHR1cmUob3V0cHV0VGV4U2hhcGVSQ1swXSwgb3V0cHV0VGV4U2hhcGVSQ1sxXSk7XG5cbiAgY29uc3QgaW5wdXREYXRhID0gdGVzdF91dGlsLnJhbmRvbUFycmF5SW5SYW5nZShcbiAgICAgIGlucHV0VGV4U2hhcGVSQ1swXSAqIGlucHV0VGV4U2hhcGVSQ1sxXSwgLTEsIDEpO1xuICBjb25zdCB3ZWlnaHRzRGF0YSA9IHRlc3RfdXRpbC5yYW5kb21BcnJheUluUmFuZ2UoXG4gICAgICB3ZWlnaHRzVGV4U2hhcGVSQ1swXSAqIHdlaWdodHNUZXhTaGFwZVJDWzFdLCAtMSwgMSk7XG4gIGNvbnN0IGJpYXNlc0RhdGEgPSB0ZXN0X3V0aWwucmFuZG9tQXJyYXlJblJhbmdlKFxuICAgICAgYmlhc2VzVGV4U2hhcGVSQ1swXSAqIGJpYXNlc1RleFNoYXBlUkNbMV0sIC0xLCAxKTtcblxuICBncGdwdS51cGxvYWRNYXRyaXhUb1RleHR1cmUoXG4gICAgICBpbnB1dFRleHR1cmUsIGlucHV0VGV4U2hhcGVSQ1swXSwgaW5wdXRUZXhTaGFwZVJDWzFdLCBpbnB1dERhdGEpO1xuICBncGdwdS51cGxvYWRNYXRyaXhUb1RleHR1cmUoXG4gICAgICB3ZWlnaHRzVGV4dHVyZSwgd2VpZ2h0c1RleFNoYXBlUkNbMF0sIHdlaWdodHNUZXhTaGFwZVJDWzFdLCB3ZWlnaHRzRGF0YSk7XG4gIGdwZ3B1LnVwbG9hZE1hdHJpeFRvVGV4dHVyZShcbiAgICAgIGJpYXNlc1RleHR1cmUsIGJpYXNlc1RleFNoYXBlUkNbMF0sIGJpYXNlc1RleFNoYXBlUkNbMV0sIGJpYXNlc0RhdGEpO1xuXG4gIGNvbnN0IHN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgT1BfUlVOUzsgaSsrKSB7XG4gICAgY29udl9ncHUuY29udm9sdmUoXG4gICAgICAgIGdwZ3B1LCBwcm9ncmFtLCBpbnB1dFRleHR1cmUsIHdlaWdodHNUZXh0dXJlLCBiaWFzZXNUZXh0dXJlLFxuICAgICAgICBvdXRwdXRUZXh0dXJlLCBvdXRwdXRUZXhTaGFwZVJDKTtcbiAgfVxuXG4gIGdwZ3B1LmRvd25sb2FkTWF0cml4RnJvbVRleHR1cmUoXG4gICAgICBvdXRwdXRUZXh0dXJlLCBvdXRwdXRUZXhTaGFwZVJDWzBdLCBvdXRwdXRUZXhTaGFwZVJDWzFdKTtcbiAgY29uc3QgZW5kID0gcGVyZm9ybWFuY2Uubm93KCk7XG5cbiAgY29uc3QgYXZnVGltZSA9IChlbmQgLSBzdGFydCkgLyBPUF9SVU5TO1xuXG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUoaW5wdXRUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZSh3ZWlnaHRzVGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUoYmlhc2VzVGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUob3V0cHV0VGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZVByb2dyYW0ocHJvZ3JhbSk7XG4gIGdwZ3B1LmRpc3Bvc2UoKTtcblxuICByZXR1cm4gYXZnVGltZTtcbn07XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIGNvbnZfdXRpbCBmcm9tICcuLi8uLi9zcmMvbWF0aC9jb252X3V0aWwnO1xuaW1wb3J0ICogYXMgY29udl9iYWNrcHJvcF9ncHUgZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvY29udl9iYWNrcHJvcF9ncHUnO1xuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL2dwZ3B1X2NvbnRleHQnO1xuaW1wb3J0ICogYXMgdGVzdF91dGlsIGZyb20gJy4uLy4uL3NyYy90ZXN0X3V0aWwnO1xuXG5pbXBvcnQge0JlbmNobWFya1Rlc3R9IGZyb20gJy4vYmVuY2htYXJrJztcblxuY29uc3QgT1BfUlVOUyA9IDEwMDtcblxuZXhwb3J0IGNvbnN0IEJFTkNITUFSS19URVNUOiBCZW5jaG1hcmtUZXN0ID0gKHNpemU6IG51bWJlcikgPT4ge1xuICBjb25zdCB4U2hhcGVSQ0Q6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFtzaXplLCBzaXplLCAxXTtcbiAgY29uc3Qgb3JpZ091dHB1dERlcHRoID0gMjtcbiAgY29uc3QgZmllbGRTaXplID0gMTE7XG4gIGNvbnN0IG9yaWdTdHJpZGUgPSAxO1xuICBjb25zdCBvcmlnUGFkID0gMTtcblxuICBjb25zdCBncGdwdSA9IG5ldyBHUEdQVUNvbnRleHQoKTtcbiAgZ3BncHUuZW5hYmxlQXV0b21hdGljRGVidWdWYWxpZGF0aW9uKHRydWUpO1xuICBjb25zdCBvcmlnSW5wdXREZXB0aCA9IHhTaGFwZVJDRFsyXTtcbiAgY29uc3Qgc3JjID0gY29udl9iYWNrcHJvcF9ncHUuZ2V0RnJhZ21lbnRTaGFkZXJDb252VHJhbnNwb3NlU291cmNlKFxuICAgICAgeFNoYXBlUkNELCBmaWVsZFNpemUsIG9yaWdJbnB1dERlcHRoLCBvcmlnU3RyaWRlLCBvcmlnUGFkLCBmYWxzZSk7XG4gIGNvbnN0IHByb2dyYW0gPSBncGdwdS5jcmVhdGVQcm9ncmFtKHNyYyk7XG5cbiAgLy8gVXBsb2FkIHguXG4gIGNvbnN0IHhUZXhTaGFwZVJDID0gY29udl91dGlsLmNvbXB1dGVUZXhTaGFwZUZyb20zRCh4U2hhcGVSQ0QpO1xuICBjb25zdCB4VGV4ID0gZ3BncHUuY3JlYXRlTWF0cml4VGV4dHVyZSh4VGV4U2hhcGVSQ1swXSwgeFRleFNoYXBlUkNbMV0pO1xuICBjb25zdCB4RGF0YSA9XG4gICAgICB0ZXN0X3V0aWwucmFuZG9tQXJyYXlJblJhbmdlKHhUZXhTaGFwZVJDWzBdICogeFRleFNoYXBlUkNbMV0sIC0xLCAxKTtcbiAgZ3BncHUudXBsb2FkTWF0cml4VG9UZXh0dXJlKHhUZXgsIHhUZXhTaGFwZVJDWzBdLCB4VGV4U2hhcGVSQ1sxXSwgeERhdGEpO1xuXG4gIC8vIFVwbG9hZCB3ZWlnaHRzLlxuICBjb25zdCB3VGV4U2hhcGVSQyA9IGNvbnZfdXRpbC5jb21wdXRlV2VpZ2h0c1RleFNoYXBlKFxuICAgICAgb3JpZ0lucHV0RGVwdGgsIG9yaWdPdXRwdXREZXB0aCwgZmllbGRTaXplKTtcbiAgY29uc3Qgd0RhdGEgPVxuICAgICAgdGVzdF91dGlsLnJhbmRvbUFycmF5SW5SYW5nZSh3VGV4U2hhcGVSQ1swXSAqIHdUZXhTaGFwZVJDWzFdLCAtMSwgMSk7XG4gIGNvbnN0IHdUZXggPSBncGdwdS5jcmVhdGVNYXRyaXhUZXh0dXJlKHdUZXhTaGFwZVJDWzBdLCB3VGV4U2hhcGVSQ1sxXSk7XG4gIGdwZ3B1LnVwbG9hZE1hdHJpeFRvVGV4dHVyZSh3VGV4LCB3VGV4U2hhcGVSQ1swXSwgd1RleFNoYXBlUkNbMV0sIHdEYXRhKTtcblxuICAvLyBGaWd1cmUgb3V0IHRoZSBvdXRwdXQgc2hhcGUgYnkgZGlsYXRpbmcgdGhlIGlucHV0LlxuICBjb25zdCBkaWxhdGVkUkMgPVxuICAgICAgY29udl91dGlsLmNvbXB1dGVEaWxhdGVkUkMoW3hTaGFwZVJDRFswXSwgeFNoYXBlUkNEWzFdXSwgb3JpZ1N0cmlkZSk7XG4gIGNvbnN0IHBhZCA9IGZpZWxkU2l6ZSAtIDEgLSBvcmlnUGFkO1xuICBjb25zdCByZXN1bHRTaGFwZVJDRCA9IGNvbnZfdXRpbC5jb21wdXRlT3V0cHV0U2hhcGUzRChcbiAgICAgIFtkaWxhdGVkUkNbMF0sIGRpbGF0ZWRSQ1sxXSwgb3JpZ091dHB1dERlcHRoXSwgZmllbGRTaXplLCBvcmlnSW5wdXREZXB0aCxcbiAgICAgIDEsIHBhZCk7XG5cbiAgY29uc3QgcmVzdWx0VGV4UkMgPSBjb252X3V0aWwuY29tcHV0ZVRleFNoYXBlRnJvbTNEKHJlc3VsdFNoYXBlUkNEKTtcbiAgY29uc3QgcmVzdWx0VGV4ID0gZ3BncHUuY3JlYXRlTWF0cml4VGV4dHVyZShyZXN1bHRUZXhSQ1swXSwgcmVzdWx0VGV4UkNbMV0pO1xuXG4gIGNvbnN0IHN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgT1BfUlVOUzsgaSsrKSB7XG4gICAgY29udl9iYWNrcHJvcF9ncHUuY29udlRyYW5zcG9zZShcbiAgICAgICAgZ3BncHUsIHByb2dyYW0sIHhUZXgsIHdUZXgsIG51bGwsIHJlc3VsdFRleCwgcmVzdWx0VGV4UkMpO1xuICB9XG5cbiAgY29uc3QgeSA9IGdwZ3B1LmRvd25sb2FkTWF0cml4RnJvbVRleHR1cmUoXG4gICAgICByZXN1bHRUZXgsIHJlc3VsdFRleFJDWzBdLCByZXN1bHRUZXhSQ1sxXSk7XG5cbiAgY29uc3QgZW5kID0gcGVyZm9ybWFuY2Uubm93KCk7XG5cbiAgY29uc3QgYXZnVGltZSA9IChlbmQgLSBzdGFydCkgLyBPUF9SVU5TO1xuXG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUocmVzdWx0VGV4KTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZSh4VGV4KTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZSh3VGV4KTtcbiAgZ3BncHUuZGVsZXRlUHJvZ3JhbShwcm9ncmFtKTtcbiAgZ3BncHUuZGlzcG9zZSgpO1xuXG4gIHJldHVybiBhdmdUaW1lO1xufTtcbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0IHtOREFycmF5TWF0aENQVX0gZnJvbSAnLi4vLi4vc3JjL21hdGgvbWF0aF9jcHUnO1xuaW1wb3J0IHtBcnJheTJELCBOREFycmF5fSBmcm9tICcuLi8uLi9zcmMvbWF0aC9uZGFycmF5JztcblxuaW1wb3J0IHtCZW5jaG1hcmtUZXN0fSBmcm9tICcuL2JlbmNobWFyayc7XG5cbmNvbnN0IE9QU19QRVJfUlVOID0gMTA7XG5cbmV4cG9ydCBjb25zdCBCRU5DSE1BUktfVEVTVDogQmVuY2htYXJrVGVzdCA9IChzaXplOiBudW1iZXIpID0+IHtcbiAgY29uc3QgbWF0aCA9IG5ldyBOREFycmF5TWF0aENQVSgpO1xuICBjb25zdCBhID0gTkRBcnJheS5yYW5kVW5pZm9ybTxBcnJheTJEPihbc2l6ZSwgc2l6ZV0sIC0xLCAxKTtcbiAgY29uc3Qgc3RhcnQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBPUFNfUEVSX1JVTjsgaSsrKSB7XG4gICAgbWF0aC5sb2dTdW1FeHAoYSk7XG4gIH1cbiAgY29uc3QgZW5kID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIHJldHVybiAoZW5kIC0gc3RhcnQpIC8gT1BTX1BFUl9SVU47XG59O1xuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQge0dQR1BVQ29udGV4dH0gZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvZ3BncHVfY29udGV4dCc7XG5pbXBvcnQgKiBhcyBsb2dzdW1leHBfZ3B1IGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL2xvZ3N1bWV4cF9ncHUnO1xuaW1wb3J0ICogYXMgdGVzdF91dGlsIGZyb20gJy4uLy4uL3NyYy90ZXN0X3V0aWwnO1xuXG5pbXBvcnQge0JlbmNobWFya1Rlc3R9IGZyb20gJy4vYmVuY2htYXJrJztcblxuY29uc3QgT1BfUlVOUyA9IDEwMDtcblxuZXhwb3J0IGNvbnN0IEJFTkNITUFSS19URVNUOiBCZW5jaG1hcmtUZXN0ID0gKHNpemU6IG51bWJlcikgPT4ge1xuICBjb25zdCBncGdwdSA9IG5ldyBHUEdQVUNvbnRleHQoKTtcblxuICBjb25zdCBwcm9ncmFtID1cbiAgICAgIGdwZ3B1LmNyZWF0ZVByb2dyYW0obG9nc3VtZXhwX2dwdS5nZXRGcmFnbWVudFNoYWRlclNvdXJjZShzaXplLCBzaXplKSk7XG5cbiAgY29uc3QgYVRleHR1cmUgPSBncGdwdS5jcmVhdGVNYXRyaXhUZXh0dXJlKHNpemUsIHNpemUpO1xuICBjb25zdCByZXN1bHRUZXh0dXJlID0gZ3BncHUuY3JlYXRlTWF0cml4VGV4dHVyZShzaXplLCBzaXplKTtcblxuICBjb25zdCBhID0gdGVzdF91dGlsLnJhbmRvbUFycmF5SW5SYW5nZShzaXplICogc2l6ZSwgLTEsIDEpO1xuICBncGdwdS51cGxvYWRNYXRyaXhUb1RleHR1cmUoYVRleHR1cmUsIHNpemUsIHNpemUsIGEpO1xuXG4gIGNvbnN0IHN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgT1BfUlVOUzsgaSsrKSB7XG4gICAgbG9nc3VtZXhwX2dwdS5sb2dTdW1FeHAoXG4gICAgICAgIGdwZ3B1LCBwcm9ncmFtLCBhVGV4dHVyZSwgc2l6ZSwgc2l6ZSwgcmVzdWx0VGV4dHVyZSk7XG4gIH1cblxuICBncGdwdS5kb3dubG9hZE1hdHJpeEZyb21UZXh0dXJlKHJlc3VsdFRleHR1cmUsIHNpemUsIHNpemUpO1xuICBjb25zdCBhdmdUaW1lID0gKHBlcmZvcm1hbmNlLm5vdygpIC0gc3RhcnQpIC8gT1BfUlVOUztcblxuICBncGdwdS5kZWxldGVNYXRyaXhUZXh0dXJlKGFUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZShyZXN1bHRUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlUHJvZ3JhbShwcm9ncmFtKTtcbiAgZ3BncHUuZGlzcG9zZSgpO1xuXG4gIHJldHVybiBhdmdUaW1lO1xufTtcbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0IHtCZW5jaG1hcmtSdW4sIEJlbmNobWFya1J1bkdyb3VwfSBmcm9tICcuL2JlbmNobWFyayc7XG5pbXBvcnQgKiBhcyBjb252X2dwdV9iZW5jaG1hcmsgZnJvbSAnLi9jb252X2dwdV9iZW5jaG1hcmsnO1xuaW1wb3J0ICogYXMgY29udl90cmFuc3Bvc2VfZ3B1X2JlbmNobWFyayBmcm9tICcuL2NvbnZfdHJhbnNwb3NlX2dwdV9iZW5jaG1hcmsnO1xuaW1wb3J0ICogYXMgbG9nc3VtZXhwX2NwdV9iZW5jaG1hcmsgZnJvbSAnLi9sb2dzdW1leHBfY3B1X2JlbmNobWFyayc7XG5pbXBvcnQgKiBhcyBsb2dzdW1leHBfZ3B1X2JlbmNobWFyayBmcm9tICcuL2xvZ3N1bWV4cF9ncHVfYmVuY2htYXJrJztcbmltcG9ydCAqIGFzIG1heF9wb29sX2JhY2twcm9wX2dwdV9iZW5jaG1hcmsgZnJvbSAnLi9tYXhfcG9vbF9iYWNrcHJvcF9ncHVfYmVuY2htYXJrJztcbmltcG9ydCAqIGFzIG1heF9wb29sX2dwdV9iZW5jaG1hcmsgZnJvbSAnLi9tYXhfcG9vbF9ncHVfYmVuY2htYXJrJztcbmltcG9ydCAqIGFzIG11bG1hdF9jcHVfYmVuY2htYXJrIGZyb20gJy4vbXVsbWF0X2NwdV9iZW5jaG1hcmsnO1xuaW1wb3J0ICogYXMgbXVsbWF0X2dwdV9iZW5jaG1hcmsgZnJvbSAnLi9tdWxtYXRfZ3B1X2JlbmNobWFyayc7XG5pbXBvcnQgKiBhcyB0ZXhfdXRpbF9iZW5jaG1hcmsgZnJvbSAnLi90ZXhfdXRpbF9iZW5jaG1hcmsnO1xuXG5leHBvcnQgY29uc3QgQkVOQ0hNQVJLX1JVTl9HUk9VUFM6IEJlbmNobWFya1J1bkdyb3VwW10gPSBbXG4gIHtcbiAgICBuYW1lOiAnVGV4dHVyZSBlbmNvZGluZyAvIGRlY29kaW5nICh1bnBhY2tlZCB2cyBwYWNrZWQpJyxcbiAgICBtaW46IDAsXG4gICAgbWF4OiAxMDI0LFxuICAgIHN0ZXBTaXplOiA2NCxcbiAgICBzdGVwVG9TaXplVHJhbnNmb3JtYXRpb246IChzdGVwOiBudW1iZXIpID0+IE1hdGgubWF4KDEsIHN0ZXApLFxuICAgIGJlbmNobWFya1J1bnM6IFtcbiAgICAgIG5ldyBCZW5jaG1hcmtSdW4oXG4gICAgICAgICAgJ2VuY29kZV91bnBhY2tlZCcsIHRleF91dGlsX2JlbmNobWFyay5CRU5DSE1BUktfRU5DT0RFX1VOUEFDS0VEKSxcbiAgICAgIG5ldyBCZW5jaG1hcmtSdW4oXG4gICAgICAgICAgJ2VuY29kZV9wYWNrZWQnLCB0ZXhfdXRpbF9iZW5jaG1hcmsuQkVOQ0hNQVJLX0VOQ09ERV9QQUNLRUQpLFxuICAgICAgbmV3IEJlbmNobWFya1J1bihcbiAgICAgICAgICAnZGVjb2RlX3VucGFja2VkJywgdGV4X3V0aWxfYmVuY2htYXJrLkJFTkNITUFSS19ERUNPREVfVU5QQUNLRUQpLFxuICAgICAgbmV3IEJlbmNobWFya1J1bihcbiAgICAgICAgICAnZGVjb2RlX3BhY2tlZCcsIHRleF91dGlsX2JlbmNobWFyay5CRU5DSE1BUktfREVDT0RFX1BBQ0tFRClcbiAgICBdXG4gIH0sXG4gIHtcbiAgICBuYW1lOiAnTWF0cml4IE11bHRpcGxpY2F0aW9uIChDUFUgdnMgR1BVKScsXG4gICAgbWluOiAwLFxuICAgIG1heDogMTAyNCxcbiAgICBzdGVwU2l6ZTogNjQsXG4gICAgc3RlcFRvU2l6ZVRyYW5zZm9ybWF0aW9uOiAoc3RlcDogbnVtYmVyKSA9PiBNYXRoLm1heCgxLCBzdGVwKSxcbiAgICBiZW5jaG1hcmtSdW5zOiBbXG4gICAgICBuZXcgQmVuY2htYXJrUnVuKCdtdWxtYXRfZ3B1JywgbXVsbWF0X2dwdV9iZW5jaG1hcmsuQkVOQ0hNQVJLX1RFU1QpLFxuICAgICAgbmV3IEJlbmNobWFya1J1bihcbiAgICAgICAgICAnbXVsbWF0X3BhY2tlZF9ncHUnLCBtdWxtYXRfZ3B1X2JlbmNobWFyay5CRU5DSE1BUktfVEVTVF9QQUNLRUQpLFxuICAgICAgbmV3IEJlbmNobWFya1J1bignbXVsbWF0X2NwdScsIG11bG1hdF9jcHVfYmVuY2htYXJrLkJFTkNITUFSS19URVNUKVxuICAgIF0sXG4gIH0sXG4gIHtcbiAgICBuYW1lOiAnTG9nU3VtRXhwIChDUFUgdnMgR1BVKScsXG4gICAgbWluOiAwLFxuICAgIG1heDogMTAyNCxcbiAgICBzdGVwU2l6ZTogNjQsXG4gICAgc3RlcFRvU2l6ZVRyYW5zZm9ybWF0aW9uOiAoc3RlcDogbnVtYmVyKSA9PiBNYXRoLm1heCgxLCBzdGVwKSxcbiAgICBiZW5jaG1hcmtSdW5zOiBbXG4gICAgICBuZXcgQmVuY2htYXJrUnVuKCdsb2dzdW1leHBfZ3B1JywgbG9nc3VtZXhwX2dwdV9iZW5jaG1hcmsuQkVOQ0hNQVJLX1RFU1QpLFxuICAgICAgbmV3IEJlbmNobWFya1J1bignbG9nc3VtZXhwX2NwdScsIGxvZ3N1bWV4cF9jcHVfYmVuY2htYXJrLkJFTkNITUFSS19URVNUKVxuICAgIF0sXG4gIH0sXG4gIHtcbiAgICBuYW1lOiAnQ29udm9sdXRpb24gKEdQVSknLFxuICAgIG1pbjogMCxcbiAgICBtYXg6IDEwMjQsXG4gICAgc3RlcFNpemU6IDY0LFxuICAgIHN0ZXBUb1NpemVUcmFuc2Zvcm1hdGlvbjogKHN0ZXA6IG51bWJlcikgPT4gTWF0aC5tYXgoMSwgc3RlcCksXG4gICAgYmVuY2htYXJrUnVuczogW25ldyBCZW5jaG1hcmtSdW4oXG4gICAgICAgICdkMT0xLCBkMj0xLCBmPTExLCBzPTEnLCBjb252X2dwdV9iZW5jaG1hcmsuQkVOQ0hNQVJLX1RFU1QpXSxcbiAgfSxcbiAge1xuICAgIG5hbWU6ICdDb252b2x1dGlvbiBUcmFuc3Bvc2VkIChHUFUpJyxcbiAgICBtaW46IDAsXG4gICAgbWF4OiAxMDI0LFxuICAgIHN0ZXBTaXplOiA2NCxcbiAgICBzdGVwVG9TaXplVHJhbnNmb3JtYXRpb246IChzdGVwOiBudW1iZXIpID0+IE1hdGgubWF4KDEsIHN0ZXApLFxuICAgIGJlbmNobWFya1J1bnM6IFtuZXcgQmVuY2htYXJrUnVuKFxuICAgICAgICAnZDE9MSwgZDI9MSwgZj0xMSwgcz0xJywgY29udl90cmFuc3Bvc2VfZ3B1X2JlbmNobWFyay5CRU5DSE1BUktfVEVTVCldLFxuICB9LFxuICB7XG4gICAgbmFtZTogJ01heCBwb29sIChHUFUpJyxcbiAgICBtaW46IDAsXG4gICAgbWF4OiAxMDI0LFxuICAgIHN0ZXBTaXplOiA2NCxcbiAgICBzdGVwVG9TaXplVHJhbnNmb3JtYXRpb246IChzdGVwOiBudW1iZXIpID0+IE1hdGgubWF4KDEsIHN0ZXApLFxuICAgIGJlbmNobWFya1J1bnM6IFtuZXcgQmVuY2htYXJrUnVuKFxuICAgICAgICAnZDE9MSwgZDI9MSwgZj0xMSwgcz0xJyxcbiAgICAgICAgbWF4X3Bvb2xfZ3B1X2JlbmNobWFyay5NQVhfUE9PTF9CRU5DSE1BUktfVEVTVCldLFxuICB9LFxuICB7XG4gICAgbmFtZTogJ01heCBwb29sIHBvc2l0aW9ucyAoR1BVKScsXG4gICAgbWluOiAwLFxuICAgIG1heDogMTAyNCxcbiAgICBzdGVwU2l6ZTogNjQsXG4gICAgc3RlcFRvU2l6ZVRyYW5zZm9ybWF0aW9uOiAoc3RlcDogbnVtYmVyKSA9PiBNYXRoLm1heCgxLCBzdGVwKSxcbiAgICBiZW5jaG1hcmtSdW5zOiBbbmV3IEJlbmNobWFya1J1bihcbiAgICAgICAgJ2QxPTEsIGQyPTEsIGY9MTEsIHM9MScsXG4gICAgICAgIG1heF9wb29sX2dwdV9iZW5jaG1hcmsuTUFYX1BPT0xfUE9TTlNfQkVOQ0hNQVJLX1RFU1QpXSxcbiAgfSxcbiAge1xuICAgIG5hbWU6ICdNYXggcG9vbCBiYWNrcHJvcCAoR1BVKScsXG4gICAgbWluOiAwLFxuICAgIG1heDogMTAyNCxcbiAgICBzdGVwU2l6ZTogNjQsXG4gICAgc3RlcFRvU2l6ZVRyYW5zZm9ybWF0aW9uOiAoc3RlcDogbnVtYmVyKSA9PiBNYXRoLm1heCgxLCBzdGVwKSxcbiAgICBiZW5jaG1hcmtSdW5zOiBbbmV3IEJlbmNobWFya1J1bihcbiAgICAgICAgJ2QxPTEsIGQyPTEsIGY9MTEsIHM9MScsXG4gICAgICAgIG1heF9wb29sX2JhY2twcm9wX2dwdV9iZW5jaG1hcmsuQkVOQ0hNQVJLX1RFU1QpXSxcbiAgfVxuXTtcbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICcuLi9kZW1vLWhlYWRlcic7XG5pbXBvcnQgJy4uL2RlbW8tZm9vdGVyJztcblxuLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLXVudXNlZC12YXJpYWJsZVxuaW1wb3J0IHtQb2x5bWVyRWxlbWVudCwgUG9seW1lckhUTUxFbGVtZW50fSBmcm9tICcuLi9wb2x5bWVyLXNwZWMnO1xuaW1wb3J0IHtCZW5jaG1hcmtSdW5Hcm91cH0gZnJvbSAnLi9iZW5jaG1hcmsnO1xuXG5pbXBvcnQge0JFTkNITUFSS19SVU5fR1JPVVBTfSBmcm9tICcuL21hdGgtYmVuY2htYXJrLXJ1bi1ncm91cHMnO1xuXG4vLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6dmFyaWFibGUtbmFtZVxuZXhwb3J0IGxldCBNYXRoQmVuY2htYXJrUG9seW1lciA9IFBvbHltZXJFbGVtZW50KFxuICAgIHtpczogJ21hdGgtYmVuY2htYXJrJywgcHJvcGVydGllczoge2JlbmNobWFya1J1bkdyb3VwTmFtZXM6IEFycmF5fX0pO1xuXG5leHBvcnQgY2xhc3MgTWF0aEJlbmNobWFyayBleHRlbmRzIE1hdGhCZW5jaG1hcmtQb2x5bWVyIHtcbiAgLy8gUG9seW1lciBwcm9wZXJ0aWVzLlxuICBwcml2YXRlIGJlbmNobWFya1J1bkdyb3VwTmFtZXM6IHN0cmluZ1tdO1xuICBwcml2YXRlIHN0b3BNZXNzYWdlczogYm9vbGVhbltdO1xuXG4gIHJlYWR5KCkge1xuICAgIC8vIFNldCB1cCB0aGUgYmVuY2htYXJrcyBVSS5cbiAgICBjb25zdCBiZW5jaG1hcmtSdW5Hcm91cE5hbWVzOiBzdHJpbmdbXSA9IFtdO1xuICAgIHRoaXMuc3RvcE1lc3NhZ2VzID0gW107XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBCRU5DSE1BUktfUlVOX0dST1VQUy5sZW5ndGg7IGkrKykge1xuICAgICAgYmVuY2htYXJrUnVuR3JvdXBOYW1lcy5wdXNoKEJFTkNITUFSS19SVU5fR1JPVVBTW2ldLm5hbWUpO1xuICAgICAgdGhpcy5zdG9wTWVzc2FnZXMucHVzaChmYWxzZSk7XG4gICAgfVxuICAgIHRoaXMuYmVuY2htYXJrUnVuR3JvdXBOYW1lcyA9IGJlbmNobWFya1J1bkdyb3VwTmFtZXM7XG5cbiAgICAvLyBJbiBhIHNldFRpbWVvdXQgdG8gbGV0IHRoZSBVSSB1cGRhdGUgYmVmb3JlIHdlIGFkZCBldmVudCBsaXN0ZW5lcnMuXG4gICAgc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICBjb25zdCBydW5CdXR0b25zID0gdGhpcy5xdWVyeVNlbGVjdG9yQWxsKCcucnVuLXRlc3QnKTtcbiAgICAgIGNvbnN0IHN0b3BCdXR0b25zID0gdGhpcy5xdWVyeVNlbGVjdG9yQWxsKCcucnVuLXN0b3AnKTtcbiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgcnVuQnV0dG9ucy5sZW5ndGg7IGkrKykge1xuICAgICAgICBydW5CdXR0b25zW2ldLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCkgPT4ge1xuICAgICAgICAgIHRoaXMucnVuQmVuY2htYXJrR3JvdXAoaSk7XG4gICAgICAgIH0pO1xuICAgICAgICBzdG9wQnV0dG9uc1tpXS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsICgpID0+IHtcbiAgICAgICAgICB0aGlzLnN0b3BNZXNzYWdlc1tpXSA9IHRydWU7XG4gICAgICAgIH0pO1xuICAgICAgfVxuICAgIH0sIDApO1xuICB9XG5cbiAgcHJpdmF0ZSBydW5CZW5jaG1hcmtHcm91cChiZW5jaG1hcmtSdW5Hcm91cEluZGV4OiBudW1iZXIpIHtcbiAgICBjb25zdCBiZW5jaG1hcmtSdW5Hcm91cCA9IEJFTkNITUFSS19SVU5fR1JPVVBTW2JlbmNobWFya1J1bkdyb3VwSW5kZXhdO1xuXG4gICAgY29uc3QgY2FudmFzID0gdGhpcy5xdWVyeVNlbGVjdG9yQWxsKCcucnVuLXBsb3QnKVtiZW5jaG1hcmtSdW5Hcm91cEluZGV4XSBhc1xuICAgICAgICBIVE1MQ2FudmFzRWxlbWVudDtcbiAgICBjb25zdCBjb250ZXh0ID0gY2FudmFzLmdldENvbnRleHQoJzJkJykgYXMgQ2FudmFzUmVuZGVyaW5nQ29udGV4dDJEO1xuXG4gICAgY29uc3QgZGF0YXNldHM6IENoYXJ0RGF0YVNldHNbXSA9IFtdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgYmVuY2htYXJrUnVuR3JvdXAuYmVuY2htYXJrUnVucy5sZW5ndGg7IGkrKykge1xuICAgICAgY29uc3QgaHVlID0gTWF0aC5mbG9vcigzNjAgKiBpIC8gYmVuY2htYXJrUnVuR3JvdXAuYmVuY2htYXJrUnVucy5sZW5ndGgpO1xuICAgICAgZGF0YXNldHMucHVzaCh7XG4gICAgICAgIGRhdGE6IGJlbmNobWFya1J1bkdyb3VwLmJlbmNobWFya1J1bnNbaV0uY2hhcnREYXRhLFxuICAgICAgICBmaWxsOiBmYWxzZSxcbiAgICAgICAgbGFiZWw6IGJlbmNobWFya1J1bkdyb3VwLmJlbmNobWFya1J1bnNbaV0ubmFtZSxcbiAgICAgICAgYm9yZGVyQ29sb3I6ICdoc2woJyArIGh1ZSArICcsIDEwMCUsIDQwJSknLFxuICAgICAgICBiYWNrZ3JvdW5kQ29sb3I6ICdoc2woJyArIGh1ZSArICcsIDEwMCUsIDcwJSknLFxuICAgICAgICBwb2ludFJhZGl1czogMCxcbiAgICAgICAgcG9pbnRIaXRSYWRpdXM6IDUsXG4gICAgICAgIGJvcmRlcldpZHRoOiAxLFxuICAgICAgICBsaW5lVGVuc2lvbjogMFxuICAgICAgfSk7XG4gICAgfVxuXG4gICAgY29uc3QgY2hhcnQgPSBuZXcgQ2hhcnQoY29udGV4dCwge1xuICAgICAgdHlwZTogJ2xpbmUnLFxuICAgICAgZGF0YToge2RhdGFzZXRzfSxcbiAgICAgIG9wdGlvbnM6IHtcbiAgICAgICAgYW5pbWF0aW9uOiB7ZHVyYXRpb246IDB9LFxuICAgICAgICByZXNwb25zaXZlOiBmYWxzZSxcbiAgICAgICAgc2NhbGVzOiB7XG4gICAgICAgICAgeEF4ZXM6IFt7XG4gICAgICAgICAgICB0eXBlOiAnbGluZWFyJyxcbiAgICAgICAgICAgIHBvc2l0aW9uOiAnYm90dG9tJyxcbiAgICAgICAgICAgIHRpY2tzOiB7XG4gICAgICAgICAgICAgIG1pbjogYmVuY2htYXJrUnVuR3JvdXAubWluLFxuICAgICAgICAgICAgICBtYXg6IGJlbmNobWFya1J1bkdyb3VwLm1heCxcbiAgICAgICAgICAgICAgc3RlcFNpemU6IGJlbmNobWFya1J1bkdyb3VwLnN0ZXBTaXplLFxuICAgICAgICAgICAgICBjYWxsYmFjazogKGxhYmVsOiBzdHJpbmcpID0+IHtcbiAgICAgICAgICAgICAgICByZXR1cm4gYmVuY2htYXJrUnVuR3JvdXAuc3RlcFRvU2l6ZVRyYW5zZm9ybWF0aW9uICE9IG51bGwgP1xuICAgICAgICAgICAgICAgICAgICBiZW5jaG1hcmtSdW5Hcm91cC5zdGVwVG9TaXplVHJhbnNmb3JtYXRpb24oK2xhYmVsKSA6XG4gICAgICAgICAgICAgICAgICAgICtsYWJlbDtcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgICAgICAgICB9IGFzIGFueSAgLy8gTm90ZTogdGhlIHR5cGluZ3MgZm9yIHRoaXMgYXJlIGluY29ycmVjdCwgY2FzdCBhcyBhbnkuXG4gICAgICAgICAgfV0sXG4gICAgICAgICAgeUF4ZXM6IFt7XG4gICAgICAgICAgICB0aWNrczoge1xuICAgICAgICAgICAgICBjYWxsYmFjazogKGxhYmVsLCBpbmRleCwgbGFiZWxzKSA9PiB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIGxhYmVsICsgJ21zJztcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfSxcbiAgICAgICAgICB9XVxuICAgICAgICB9LFxuICAgICAgICB0b29sdGlwczoge21vZGU6ICdsYWJlbCd9LFxuICAgICAgICB0aXRsZToge3RleHQ6IGJlbmNobWFya1J1bkdyb3VwLm5hbWV9XG4gICAgICB9XG4gICAgfSk7XG4gICAgY2FudmFzLnN0eWxlLmRpc3BsYXkgPSAnbm9uZSc7XG5cbiAgICBjb25zdCBydW5NZXNzYWdlID1cbiAgICAgICAgdGhpcy5xdWVyeVNlbGVjdG9yQWxsKCcucnVuLW1lc3NhZ2UnKVtiZW5jaG1hcmtSdW5Hcm91cEluZGV4XSBhc1xuICAgICAgICBIVE1MRWxlbWVudDtcbiAgICBydW5NZXNzYWdlLnN0eWxlLmRpc3BsYXkgPSAnYmxvY2snO1xuXG4gICAgY29uc3QgcnVuTnVtYmVyc1RhYmxlID1cbiAgICAgICAgdGhpcy5xdWVyeVNlbGVjdG9yQWxsKCcucnVuLW51bWJlcnMtdGFibGUnKVtiZW5jaG1hcmtSdW5Hcm91cEluZGV4XSBhc1xuICAgICAgICBIVE1MRWxlbWVudDtcbiAgICBydW5OdW1iZXJzVGFibGUuaW5uZXJIVE1MID0gJyc7XG4gICAgcnVuTnVtYmVyc1RhYmxlLnN0eWxlLmRpc3BsYXkgPSAnbm9uZSc7XG5cbiAgICAvLyBTZXQgdXAgdGhlIGhlYWRlciBmb3IgdGhlIHRhYmxlLlxuICAgIGNvbnN0IGhlYWRlcnMgPSBbJ3NpemUnXTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGJlbmNobWFya1J1bkdyb3VwLmJlbmNobWFya1J1bnMubGVuZ3RoOyBpKyspIHtcbiAgICAgIGhlYWRlcnMucHVzaChiZW5jaG1hcmtSdW5Hcm91cC5iZW5jaG1hcmtSdW5zW2ldLm5hbWUpO1xuICAgIH1cbiAgICBydW5OdW1iZXJzVGFibGUuYXBwZW5kQ2hpbGQodGhpcy5idWlsZFJ1bk51bWJlcnNSb3coaGVhZGVycykpO1xuXG4gICAgdGhpcy5ydW5CZW5jaG1hcmtTdGVwcyhcbiAgICAgICAgY2hhcnQsIGJlbmNobWFya1J1bkdyb3VwLCBiZW5jaG1hcmtSdW5Hcm91cEluZGV4LFxuICAgICAgICBiZW5jaG1hcmtSdW5Hcm91cC5taW4pO1xuICB9XG5cbiAgcHJpdmF0ZSBidWlsZFJ1bk51bWJlcnNSb3codmFsdWVzOiBzdHJpbmdbXSkge1xuICAgIGNvbnN0IHJ1bk51bWJlclJvd0VsZW1lbnQgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdkaXYnKTtcbiAgICBydW5OdW1iZXJSb3dFbGVtZW50LmNsYXNzTmFtZSA9ICdydW4tbnVtYmVycy1yb3cgbWF0aC1iZW5jaG1hcmsnO1xuXG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyBpKyspIHtcbiAgICAgIGNvbnN0IHJ1bk51bWJlckNlbGxFbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnZGl2Jyk7XG4gICAgICBydW5OdW1iZXJDZWxsRWxlbWVudC5jbGFzc05hbWUgPSAncnVuLW51bWJlcnMtY2VsbCBtYXRoLWJlbmNobWFyayc7XG4gICAgICBydW5OdW1iZXJDZWxsRWxlbWVudC5pbm5lclRleHQgPSB2YWx1ZXNbaV07XG4gICAgICBydW5OdW1iZXJSb3dFbGVtZW50LmFwcGVuZENoaWxkKHJ1bk51bWJlckNlbGxFbGVtZW50KTtcbiAgICB9XG4gICAgcmV0dXJuIHJ1bk51bWJlclJvd0VsZW1lbnQ7XG4gIH1cblxuICBwcml2YXRlIHJ1bkJlbmNobWFya1N0ZXBzKFxuICAgICAgY2hhcnQ6IENoYXJ0LCBiZW5jaG1hcmtSdW5Hcm91cDogQmVuY2htYXJrUnVuR3JvdXAsXG4gICAgICBiZW5jaG1hcmtSdW5Hcm91cEluZGV4OiBudW1iZXIsIHN0ZXA6IG51bWJlcikge1xuICAgIGNvbnN0IHJ1bk51bWJlcnNUYWJsZSA9XG4gICAgICAgIHRoaXMucXVlcnlTZWxlY3RvckFsbCgnLnJ1bi1udW1iZXJzLXRhYmxlJylbYmVuY2htYXJrUnVuR3JvdXBJbmRleF0gYXNcbiAgICAgICAgSFRNTEVsZW1lbnQ7XG4gICAgaWYgKHN0ZXAgPiBiZW5jaG1hcmtSdW5Hcm91cC5tYXggfHxcbiAgICAgICAgdGhpcy5zdG9wTWVzc2FnZXNbYmVuY2htYXJrUnVuR3JvdXBJbmRleF0pIHtcbiAgICAgIHRoaXMuc3RvcE1lc3NhZ2VzW2JlbmNobWFya1J1bkdyb3VwSW5kZXhdID0gZmFsc2U7XG5cbiAgICAgIHJ1bk51bWJlcnNUYWJsZS5zdHlsZS5kaXNwbGF5ID0gJyc7XG5cbiAgICAgIGNvbnN0IGNhbnZhcyA9XG4gICAgICAgICAgdGhpcy5xdWVyeVNlbGVjdG9yQWxsKCcucnVuLXBsb3QnKVtiZW5jaG1hcmtSdW5Hcm91cEluZGV4XSBhc1xuICAgICAgICAgIEhUTUxDYW52YXNFbGVtZW50O1xuICAgICAgY2FudmFzLnN0eWxlLmRpc3BsYXkgPSAnYmxvY2snO1xuICAgICAgY2hhcnQudXBkYXRlKCk7XG5cbiAgICAgIGNvbnN0IHJ1bk1lc3NhZ2UgPVxuICAgICAgICAgIHRoaXMucXVlcnlTZWxlY3RvckFsbCgnLnJ1bi1tZXNzYWdlJylbYmVuY2htYXJrUnVuR3JvdXBJbmRleF0gYXNcbiAgICAgICAgICBIVE1MRWxlbWVudDtcbiAgICAgIHJ1bk1lc3NhZ2Uuc3R5bGUuZGlzcGxheSA9ICdub25lJztcblxuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGNvbnN0IHJ1bk51bWJlclJvd0VsZW1lbnQgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdkaXYnKTtcbiAgICBydW5OdW1iZXJSb3dFbGVtZW50LmNsYXNzTmFtZSA9ICdydW4tbnVtYmVycy1yb3cgbWF0aC1iZW5jaG1hcmsnO1xuXG4gICAgY29uc3Qgcm93VmFsdWVzOiBzdHJpbmdbXSA9IFsnJyArIHN0ZXBdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgYmVuY2htYXJrUnVuR3JvdXAuYmVuY2htYXJrUnVucy5sZW5ndGg7IGkrKykge1xuICAgICAgY29uc3QgYmVuY2htYXJrUnVuID0gYmVuY2htYXJrUnVuR3JvdXAuYmVuY2htYXJrUnVuc1tpXTtcbiAgICAgIGNvbnN0IGJlbmNobWFya1Rlc3QgPSBiZW5jaG1hcmtSdW4uYmVuY2htYXJrVGVzdDtcblxuICAgICAgY29uc3Qgc2l6ZSA9IGJlbmNobWFya1J1bkdyb3VwLnN0ZXBUb1NpemVUcmFuc2Zvcm1hdGlvbiAhPSBudWxsID9cbiAgICAgICAgICBiZW5jaG1hcmtSdW5Hcm91cC5zdGVwVG9TaXplVHJhbnNmb3JtYXRpb24oc3RlcCkgOlxuICAgICAgICAgIHN0ZXA7XG5cbiAgICAgIGxldCByZXN1bHRTdHJpbmc6IHN0cmluZztcbiAgICAgIGxldCBsb2dTdHJpbmc6IHN0cmluZztcbiAgICAgIGxldCB0aW1lID0gMDtcbiAgICAgIGxldCBzdWNjZXNzID0gdHJ1ZTtcblxuICAgICAgdHJ5IHtcbiAgICAgICAgdGltZSA9IGJlbmNobWFya1Rlc3Qoc2l6ZSk7XG4gICAgICAgIHJlc3VsdFN0cmluZyA9IHRpbWUudG9GaXhlZCgzKSArICdtcyc7XG4gICAgICAgIGxvZ1N0cmluZyA9IHJlc3VsdFN0cmluZztcbiAgICAgIH0gY2F0Y2ggKGUpIHtcbiAgICAgICAgc3VjY2VzcyA9IGZhbHNlO1xuICAgICAgICByZXN1bHRTdHJpbmcgPSAnRXJyb3InO1xuICAgICAgICBsb2dTdHJpbmcgPSBlLm1lc3NhZ2U7XG4gICAgICB9XG5cbiAgICAgIGlmICh0aW1lID49IDApIHtcbiAgICAgICAgaWYgKHN1Y2Nlc3MpIHtcbiAgICAgICAgICBiZW5jaG1hcmtSdW4uY2hhcnREYXRhLnB1c2goe3g6IHN0ZXAsIHk6IHRpbWV9KTtcbiAgICAgICAgfVxuICAgICAgICByb3dWYWx1ZXMucHVzaChyZXN1bHRTdHJpbmcpO1xuICAgICAgfVxuICAgICAgY29uc29sZS5sb2coYmVuY2htYXJrUnVuLm5hbWUgKyAnWycgKyBzdGVwICsgJ106ICcgKyBsb2dTdHJpbmcpO1xuICAgIH1cbiAgICBydW5OdW1iZXJzVGFibGUuYXBwZW5kQ2hpbGQodGhpcy5idWlsZFJ1bk51bWJlcnNSb3cocm93VmFsdWVzKSk7XG5cbiAgICBzdGVwICs9IGJlbmNobWFya1J1bkdyb3VwLnN0ZXBTaXplO1xuICAgIC8vIEFsbG93IHRoZSBVSSB0byB1cGRhdGUuXG4gICAgc2V0VGltZW91dChcbiAgICAgICAgKCkgPT4gdGhpcy5ydW5CZW5jaG1hcmtTdGVwcyhcbiAgICAgICAgICAgIGNoYXJ0LCBiZW5jaG1hcmtSdW5Hcm91cCwgYmVuY2htYXJrUnVuR3JvdXBJbmRleCwgc3RlcCksXG4gICAgICAgIDEwMCk7XG4gIH1cbn1cbmRvY3VtZW50LnJlZ2lzdGVyRWxlbWVudChNYXRoQmVuY2htYXJrLnByb3RvdHlwZS5pcywgTWF0aEJlbmNobWFyayk7XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIGNvbnZfdXRpbCBmcm9tICcuLi8uLi9zcmMvbWF0aC9jb252X3V0aWwnO1xuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL2dwZ3B1X2NvbnRleHQnO1xuaW1wb3J0ICogYXMgbWF4X3Bvb2xfYmFja3Byb3BfZ3B1IGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL21heF9wb29sX2JhY2twcm9wX2dwdSc7XG5pbXBvcnQgKiBhcyB0ZXN0X3V0aWwgZnJvbSAnLi4vLi4vc3JjL3Rlc3RfdXRpbCc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uLy4uL3NyYy91dGlsJztcblxuaW1wb3J0IHtCZW5jaG1hcmtUZXN0fSBmcm9tICcuL2JlbmNobWFyayc7XG5cbmNvbnN0IE9QX1JVTlMgPSAxMDA7XG5cbmV4cG9ydCBjb25zdCBCRU5DSE1BUktfVEVTVDogQmVuY2htYXJrVGVzdCA9IChzaXplOiBudW1iZXIpID0+IHtcbiAgY29uc3QgZHlTaGFwZVJDRDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gW3NpemUsIHNpemUsIDFdO1xuICBjb25zdCBvdXRwdXREZXB0aCA9IDE7XG4gIGNvbnN0IGZpZWxkU2l6ZSA9IDExO1xuICBjb25zdCBzdHJpZGUgPSAxO1xuICBjb25zdCB6ZXJvUGFkID0gY29udl91dGlsLmNvbXB1dGVEZWZhdWx0UGFkKGR5U2hhcGVSQ0QsIGZpZWxkU2l6ZSwgc3RyaWRlKTtcbiAgY29uc3Qgb3V0cHV0U2hhcGVSQ0Q6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9XG4gICAgICBjb252X3V0aWwuY29tcHV0ZU91dHB1dFNoYXBlM0QoXG4gICAgICAgICAgZHlTaGFwZVJDRCwgZmllbGRTaXplLCBvdXRwdXREZXB0aCwgc3RyaWRlLCB6ZXJvUGFkKTtcblxuICBjb25zdCBkeVRleFNoYXBlUkMgPSBjb252X3V0aWwuY29tcHV0ZVRleFNoYXBlRnJvbTNEKGR5U2hhcGVSQ0QpO1xuICBjb25zdCBvdXRwdXRUZXhTaGFwZVJDID0gY29udl91dGlsLmNvbXB1dGVUZXhTaGFwZUZyb20zRChvdXRwdXRTaGFwZVJDRCk7XG5cbiAgY29uc3QgZ3BncHUgPSBuZXcgR1BHUFVDb250ZXh0KCk7XG4gIGNvbnN0IHByb2dyYW0gPSBncGdwdS5jcmVhdGVQcm9ncmFtKFxuICAgICAgbWF4X3Bvb2xfYmFja3Byb3BfZ3B1LmdldEZyYWdtZW50U2hhZGVyTWF4UG9vbEJhY2twcm9wKFxuICAgICAgICAgIGR5U2hhcGVSQ0QsIGZpZWxkU2l6ZSwgc3RyaWRlLCB6ZXJvUGFkKSk7XG5cbiAgY29uc3QgZHlUZXh0dXJlID0gZ3BncHUuY3JlYXRlTWF0cml4VGV4dHVyZShkeVRleFNoYXBlUkNbMF0sIGR5VGV4U2hhcGVSQ1sxXSk7XG4gIGNvbnN0IG1heFBvc2l0aW9uc1RleHR1cmUgPVxuICAgICAgZ3BncHUuY3JlYXRlTWF0cml4VGV4dHVyZShkeVRleFNoYXBlUkNbMF0sIGR5VGV4U2hhcGVSQ1sxXSk7XG4gIGNvbnN0IG91dHB1dFRleHR1cmUgPVxuICAgICAgZ3BncHUuY3JlYXRlTWF0cml4VGV4dHVyZShvdXRwdXRUZXhTaGFwZVJDWzBdLCBvdXRwdXRUZXhTaGFwZVJDWzFdKTtcblxuICBjb25zdCBkeURhdGEgPVxuICAgICAgdGVzdF91dGlsLnJhbmRvbUFycmF5SW5SYW5nZShkeVRleFNoYXBlUkNbMF0gKiBkeVRleFNoYXBlUkNbMV0sIC0xLCAxKTtcbiAgY29uc3QgbWF4UG9zaXRpb25zRGF0YSA9IG5ldyBGbG9hdDMyQXJyYXkodXRpbC5zaXplRnJvbVNoYXBlKGR5U2hhcGVSQ0QpKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBtYXhQb3NpdGlvbnNEYXRhLmxlbmd0aDsgaSsrKSB7XG4gICAgbWF4UG9zaXRpb25zRGF0YVtpXSA9IE1hdGguZmxvb3IoTWF0aC5yYW5kb20oKSAqIGZpZWxkU2l6ZSAqIGZpZWxkU2l6ZSk7XG4gIH1cblxuICBncGdwdS51cGxvYWRNYXRyaXhUb1RleHR1cmUoXG4gICAgICBkeVRleHR1cmUsIGR5VGV4U2hhcGVSQ1swXSwgZHlUZXhTaGFwZVJDWzFdLCBkeURhdGEpO1xuICBncGdwdS51cGxvYWRNYXRyaXhUb1RleHR1cmUoXG4gICAgICBtYXhQb3NpdGlvbnNUZXh0dXJlLCBkeVRleFNoYXBlUkNbMF0sIGR5VGV4U2hhcGVSQ1sxXSwgbWF4UG9zaXRpb25zRGF0YSk7XG5cbiAgY29uc3Qgc3RhcnQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBPUF9SVU5TOyBpKyspIHtcbiAgICBtYXhfcG9vbF9iYWNrcHJvcF9ncHUubWF4UG9vbEJhY2twcm9wKFxuICAgICAgICBncGdwdSwgcHJvZ3JhbSwgZHlUZXh0dXJlLCBtYXhQb3NpdGlvbnNUZXh0dXJlLCBvdXRwdXRUZXh0dXJlLFxuICAgICAgICBvdXRwdXRUZXhTaGFwZVJDKTtcbiAgfVxuXG4gIGdwZ3B1LmRvd25sb2FkTWF0cml4RnJvbVRleHR1cmUoXG4gICAgICBvdXRwdXRUZXh0dXJlLCBvdXRwdXRUZXhTaGFwZVJDWzBdLCBvdXRwdXRUZXhTaGFwZVJDWzFdKTtcbiAgY29uc3QgZW5kID0gcGVyZm9ybWFuY2Uubm93KCk7XG5cbiAgY29uc3QgYXZnVGltZSA9IChlbmQgLSBzdGFydCkgLyBPUF9SVU5TO1xuXG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUoZHlUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZShtYXhQb3NpdGlvbnNUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZShvdXRwdXRUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlUHJvZ3JhbShwcm9ncmFtKTtcbiAgZ3BncHUuZGlzcG9zZSgpO1xuXG4gIHJldHVybiBhdmdUaW1lO1xufTsiLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIGNvbnZfdXRpbCBmcm9tICcuLi8uLi9zcmMvbWF0aC9jb252X3V0aWwnO1xuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL2dwZ3B1X2NvbnRleHQnO1xuaW1wb3J0ICogYXMgbWF4X3Bvb2xfZ3B1IGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL21heF9wb29sX2dwdSc7XG5pbXBvcnQgKiBhcyB0ZXN0X3V0aWwgZnJvbSAnLi4vLi4vc3JjL3Rlc3RfdXRpbCc7XG5cbmltcG9ydCB7QmVuY2htYXJrVGVzdH0gZnJvbSAnLi9iZW5jaG1hcmsnO1xuXG5jb25zdCBPUF9SVU5TID0gMTAwO1xuXG5leHBvcnQgY29uc3QgTUFYX1BPT0xfQkVOQ0hNQVJLX1RFU1Q6IEJlbmNobWFya1Rlc3QgPSAoc2l6ZTogbnVtYmVyKSA9PiB7XG4gIGNvbnN0IGlucHV0U2hhcGVSQ0Q6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFtzaXplLCBzaXplLCAxXTtcbiAgY29uc3Qgb3V0cHV0RGVwdGggPSAxO1xuICBjb25zdCBmaWVsZFNpemUgPSAxMTtcbiAgY29uc3Qgc3RyaWRlID0gMTtcbiAgY29uc3QgemVyb1BhZCA9IGNvbnZfdXRpbC5jb21wdXRlRGVmYXVsdFBhZChpbnB1dFNoYXBlUkNELCBmaWVsZFNpemUsIHN0cmlkZSk7XG4gIGNvbnN0IG91dHB1dFNoYXBlUkNEOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPVxuICAgICAgY29udl91dGlsLmNvbXB1dGVPdXRwdXRTaGFwZTNEKFxuICAgICAgICAgIGlucHV0U2hhcGVSQ0QsIGZpZWxkU2l6ZSwgb3V0cHV0RGVwdGgsIHN0cmlkZSwgemVyb1BhZCk7XG5cbiAgY29uc3QgaW5wdXRUZXhTaGFwZVJDID0gY29udl91dGlsLmNvbXB1dGVUZXhTaGFwZUZyb20zRChpbnB1dFNoYXBlUkNEKTtcbiAgY29uc3Qgb3V0cHV0VGV4U2hhcGVSQyA9IGNvbnZfdXRpbC5jb21wdXRlVGV4U2hhcGVGcm9tM0Qob3V0cHV0U2hhcGVSQ0QpO1xuXG4gIGNvbnN0IGdwZ3B1ID0gbmV3IEdQR1BVQ29udGV4dCgpO1xuICBjb25zdCBwcm9ncmFtID1cbiAgICAgIGdwZ3B1LmNyZWF0ZVByb2dyYW0obWF4X3Bvb2xfZ3B1LmdldEZyYWdtZW50U2hhZGVyTWF4UG9vbFNvdXJjZShcbiAgICAgICAgICBpbnB1dFNoYXBlUkNELCBmaWVsZFNpemUsIHN0cmlkZSwgemVyb1BhZCkpO1xuXG4gIGNvbnN0IGlucHV0VGV4dHVyZSA9XG4gICAgICBncGdwdS5jcmVhdGVNYXRyaXhUZXh0dXJlKGlucHV0VGV4U2hhcGVSQ1swXSwgaW5wdXRUZXhTaGFwZVJDWzFdKTtcbiAgY29uc3Qgb3V0cHV0VGV4dHVyZSA9XG4gICAgICBncGdwdS5jcmVhdGVNYXRyaXhUZXh0dXJlKG91dHB1dFRleFNoYXBlUkNbMF0sIG91dHB1dFRleFNoYXBlUkNbMV0pO1xuXG4gIGNvbnN0IGlucHV0RGF0YSA9IHRlc3RfdXRpbC5yYW5kb21BcnJheUluUmFuZ2UoXG4gICAgICBpbnB1dFRleFNoYXBlUkNbMF0gKiBpbnB1dFRleFNoYXBlUkNbMV0sIC0xLCAxKTtcblxuICBncGdwdS51cGxvYWRNYXRyaXhUb1RleHR1cmUoXG4gICAgICBpbnB1dFRleHR1cmUsIGlucHV0VGV4U2hhcGVSQ1swXSwgaW5wdXRUZXhTaGFwZVJDWzFdLCBpbnB1dERhdGEpO1xuXG4gIGNvbnN0IHN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgT1BfUlVOUzsgaSsrKSB7XG4gICAgbWF4X3Bvb2xfZ3B1Lm1heFBvb2xDb21tb24oXG4gICAgICAgIGdwZ3B1LCBwcm9ncmFtLCBpbnB1dFRleHR1cmUsIG91dHB1dFRleHR1cmUsIG91dHB1dFRleFNoYXBlUkMpO1xuICB9XG5cbiAgZ3BncHUuZG93bmxvYWRNYXRyaXhGcm9tVGV4dHVyZShcbiAgICAgIG91dHB1dFRleHR1cmUsIG91dHB1dFRleFNoYXBlUkNbMF0sIG91dHB1dFRleFNoYXBlUkNbMV0pO1xuICBjb25zdCBlbmQgPSBwZXJmb3JtYW5jZS5ub3coKTtcblxuICBjb25zdCBhdmdUaW1lID0gKGVuZCAtIHN0YXJ0KSAvIE9QX1JVTlM7XG5cbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZShpbnB1dFRleHR1cmUpO1xuICBncGdwdS5kZWxldGVNYXRyaXhUZXh0dXJlKG91dHB1dFRleHR1cmUpO1xuICBncGdwdS5kZWxldGVQcm9ncmFtKHByb2dyYW0pO1xuICBncGdwdS5kaXNwb3NlKCk7XG5cbiAgcmV0dXJuIGF2Z1RpbWU7XG59O1xuXG5leHBvcnQgY29uc3QgTUFYX1BPT0xfUE9TTlNfQkVOQ0hNQVJLX1RFU1Q6IEJlbmNobWFya1Rlc3QgPSAoc2l6ZTogbnVtYmVyKSA9PiB7XG4gIGNvbnN0IGlucHV0U2hhcGVSQ0Q6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFtzaXplLCBzaXplLCAxXTtcbiAgY29uc3Qgb3V0cHV0RGVwdGggPSAxO1xuICBjb25zdCBmaWVsZFNpemUgPSAxMTtcbiAgY29uc3Qgc3RyaWRlID0gMTtcbiAgY29uc3QgemVyb1BhZCA9IGNvbnZfdXRpbC5jb21wdXRlRGVmYXVsdFBhZChpbnB1dFNoYXBlUkNELCBmaWVsZFNpemUsIHN0cmlkZSk7XG4gIGNvbnN0IG91dHB1dFNoYXBlUkNEOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPVxuICAgICAgY29udl91dGlsLmNvbXB1dGVPdXRwdXRTaGFwZTNEKFxuICAgICAgICAgIGlucHV0U2hhcGVSQ0QsIGZpZWxkU2l6ZSwgb3V0cHV0RGVwdGgsIHN0cmlkZSwgemVyb1BhZCk7XG5cbiAgY29uc3QgaW5wdXRUZXhTaGFwZVJDID0gY29udl91dGlsLmNvbXB1dGVUZXhTaGFwZUZyb20zRChpbnB1dFNoYXBlUkNEKTtcbiAgY29uc3Qgb3V0cHV0VGV4U2hhcGVSQyA9IGNvbnZfdXRpbC5jb21wdXRlVGV4U2hhcGVGcm9tM0Qob3V0cHV0U2hhcGVSQ0QpO1xuXG4gIGNvbnN0IGdwZ3B1ID0gbmV3IEdQR1BVQ29udGV4dCgpO1xuICBjb25zdCBwcm9ncmFtOiBXZWJHTFByb2dyYW0gPVxuICAgICAgZ3BncHUuY3JlYXRlUHJvZ3JhbShtYXhfcG9vbF9ncHUuZ2V0RnJhZ21lbnRTaGFkZXJNYXhQb29sUG9zaXRpb25zU291cmNlKFxuICAgICAgICAgIGlucHV0U2hhcGVSQ0QsIGZpZWxkU2l6ZSwgc3RyaWRlLCB6ZXJvUGFkKSk7XG5cbiAgY29uc3QgaW5wdXRUZXh0dXJlID1cbiAgICAgIGdwZ3B1LmNyZWF0ZU1hdHJpeFRleHR1cmUoaW5wdXRUZXhTaGFwZVJDWzBdLCBpbnB1dFRleFNoYXBlUkNbMV0pO1xuICBjb25zdCBvdXRwdXRUZXh0dXJlID1cbiAgICAgIGdwZ3B1LmNyZWF0ZU1hdHJpeFRleHR1cmUob3V0cHV0VGV4U2hhcGVSQ1swXSwgb3V0cHV0VGV4U2hhcGVSQ1sxXSk7XG5cbiAgY29uc3QgaW5wdXREYXRhID0gdGVzdF91dGlsLnJhbmRvbUFycmF5SW5SYW5nZShcbiAgICAgIGlucHV0VGV4U2hhcGVSQ1swXSAqIGlucHV0VGV4U2hhcGVSQ1sxXSwgLTEsIDEpO1xuXG4gIGdwZ3B1LnVwbG9hZE1hdHJpeFRvVGV4dHVyZShcbiAgICAgIGlucHV0VGV4dHVyZSwgaW5wdXRUZXhTaGFwZVJDWzBdLCBpbnB1dFRleFNoYXBlUkNbMV0sIGlucHV0RGF0YSk7XG5cbiAgY29uc3Qgc3RhcnQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBPUF9SVU5TOyBpKyspIHtcbiAgICBtYXhfcG9vbF9ncHUubWF4UG9vbENvbW1vbihcbiAgICAgICAgZ3BncHUsIHByb2dyYW0sIGlucHV0VGV4dHVyZSwgb3V0cHV0VGV4dHVyZSwgb3V0cHV0VGV4U2hhcGVSQyk7XG4gIH1cblxuICBncGdwdS5kb3dubG9hZE1hdHJpeEZyb21UZXh0dXJlKFxuICAgICAgb3V0cHV0VGV4dHVyZSwgb3V0cHV0VGV4U2hhcGVSQ1swXSwgb3V0cHV0VGV4U2hhcGVSQ1sxXSk7XG4gIGNvbnN0IGVuZCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuXG4gIGNvbnN0IGF2Z1RpbWUgPSAoZW5kIC0gc3RhcnQpIC8gT1BfUlVOUztcblxuICBncGdwdS5kZWxldGVNYXRyaXhUZXh0dXJlKGlucHV0VGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUob3V0cHV0VGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZVByb2dyYW0ocHJvZ3JhbSk7XG4gIGdwZ3B1LmRpc3Bvc2UoKTtcblxuICByZXR1cm4gYXZnVGltZTtcbn07IiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQge05EQXJyYXlNYXRoQ1BVfSBmcm9tICcuLi8uLi9zcmMvbWF0aC9tYXRoX2NwdSc7XG5pbXBvcnQge0FycmF5MkQsIE5EQXJyYXl9IGZyb20gJy4uLy4uL3NyYy9tYXRoL25kYXJyYXknO1xuXG5pbXBvcnQge0JlbmNobWFya1Rlc3R9IGZyb20gJy4vYmVuY2htYXJrJztcblxuY29uc3QgT1BTX1BFUl9TTUFMTF9SVU4gPSAxMDtcblxuZXhwb3J0IGNvbnN0IEJFTkNITUFSS19URVNUOiBCZW5jaG1hcmtUZXN0ID0gKHNpemU6IG51bWJlcikgPT4ge1xuICBpZiAoc2l6ZSA+IDUxMikge1xuICAgIHJldHVybiAtMTtcbiAgfVxuICBjb25zdCBtYXRoID0gbmV3IE5EQXJyYXlNYXRoQ1BVKCk7XG4gIGNvbnN0IGEgPSBOREFycmF5LnJhbmRVbmlmb3JtPEFycmF5MkQ+KFtzaXplLCBzaXplXSwgLTEsIDEpO1xuICBjb25zdCBiID0gTkRBcnJheS5yYW5kVW5pZm9ybTxBcnJheTJEPihbc2l6ZSwgc2l6ZV0sIC0xLCAxKTtcbiAgY29uc3QgcnVucyA9IChzaXplIDwgMTkyKSA/IE9QU19QRVJfU01BTExfUlVOIDogMTtcbiAgY29uc3Qgc3RhcnQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBydW5zOyBpKyspIHtcbiAgICBtYXRoLm1hdE11bChhLCBiKTtcbiAgfVxuICBjb25zdCBlbmQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgcmV0dXJuIChlbmQgLSBzdGFydCkgLyBydW5zO1xufTtcbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0IHtNYXRyaXhPcmllbnRhdGlvbn0gZnJvbSAnLi4vLi4vc3JjL21hdGgvbWF0aCc7XG5pbXBvcnQge0FycmF5MkR9IGZyb20gJy4uLy4uL3NyYy9tYXRoL25kYXJyYXknO1xuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL2dwZ3B1X2NvbnRleHQnO1xuaW1wb3J0ICogYXMgbXVsbWF0X2dwdSBmcm9tICcuLi8uLi9zcmMvbWF0aC93ZWJnbC9tdWxtYXRfZ3B1JztcbmltcG9ydCAqIGFzIG11bG1hdF9wYWNrZWRfZ3B1IGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL211bG1hdF9wYWNrZWRfZ3B1JztcbmltcG9ydCAqIGFzIHRlc3RfdXRpbCBmcm9tICcuLi8uLi9zcmMvdGVzdF91dGlsJztcblxuaW1wb3J0IHtCZW5jaG1hcmtUZXN0fSBmcm9tICcuL2JlbmNobWFyayc7XG5cbmNvbnN0IE9QX1JVTlMgPSAxMDA7XG5cbmV4cG9ydCBjb25zdCBCRU5DSE1BUktfVEVTVDogQmVuY2htYXJrVGVzdCA9IChzaXplOiBudW1iZXIpID0+IHtcbiAgY29uc3QgZ3BncHUgPSBuZXcgR1BHUFVDb250ZXh0KCk7XG4gIGNvbnN0IGFUZXh0dXJlID0gZ3BncHUuY3JlYXRlTWF0cml4VGV4dHVyZShzaXplLCBzaXplKTtcbiAgY29uc3QgYlRleHR1cmUgPSBncGdwdS5jcmVhdGVNYXRyaXhUZXh0dXJlKHNpemUsIHNpemUpO1xuICBjb25zdCByZXN1bHRUZXh0dXJlID0gZ3BncHUuY3JlYXRlTWF0cml4VGV4dHVyZShzaXplLCBzaXplKTtcblxuICBjb25zdCBhQXJyID0gbmV3IEFycmF5MkQoXG4gICAgICBbc2l6ZSwgc2l6ZV0sIHt0ZXh0dXJlOiBhVGV4dHVyZSwgdGV4dHVyZVNoYXBlUkM6IFtzaXplLCBzaXplXX0pO1xuICBjb25zdCBiQXJyID0gbmV3IEFycmF5MkQoXG4gICAgICBbc2l6ZSwgc2l6ZV0sIHt0ZXh0dXJlOiBiVGV4dHVyZSwgdGV4dHVyZVNoYXBlUkM6IFtzaXplLCBzaXplXX0pO1xuICBjb25zdCByZXNBcnIgPSBuZXcgQXJyYXkyRChcbiAgICAgIFtzaXplLCBzaXplXSwge3RleHR1cmU6IHJlc3VsdFRleHR1cmUsIHRleHR1cmVTaGFwZVJDOiBbc2l6ZSwgc2l6ZV19KTtcbiAgY29uc3QgcHJvZ3JhbSA9IGdwZ3B1LmNyZWF0ZVByb2dyYW0obXVsbWF0X2dwdS5nZXRGcmFnbWVudFNoYWRlcihcbiAgICAgIGFBcnIsIGJBcnIsIHJlc0FyciwgTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUixcbiAgICAgIE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpKTtcblxuICBjb25zdCBhID0gdGVzdF91dGlsLnJhbmRvbUFycmF5SW5SYW5nZShzaXplICogc2l6ZSwgLTEsIDEpO1xuICBjb25zdCBiID0gdGVzdF91dGlsLnJhbmRvbUFycmF5SW5SYW5nZShzaXplICogc2l6ZSwgLTEsIDEpO1xuICBncGdwdS51cGxvYWRNYXRyaXhUb1RleHR1cmUoYVRleHR1cmUsIHNpemUsIHNpemUsIGEpO1xuICBncGdwdS51cGxvYWRNYXRyaXhUb1RleHR1cmUoYlRleHR1cmUsIHNpemUsIHNpemUsIGIpO1xuXG4gIGNvbnN0IHN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgT1BfUlVOUzsgaSsrKSB7XG4gICAgbXVsbWF0X2dwdS5tdWx0aXBseU1hdHJpeChcbiAgICAgICAgZ3BncHUsIHByb2dyYW0sIGFUZXh0dXJlLCBiVGV4dHVyZSwgcmVzdWx0VGV4dHVyZSwgW3NpemUsIHNpemVdKTtcbiAgfVxuXG4gIGNvbnN0IGFjdHVhbCA9IGdwZ3B1LmRvd25sb2FkTWF0cml4RnJvbVRleHR1cmUocmVzdWx0VGV4dHVyZSwgc2l6ZSwgc2l6ZSk7XG4gIGNvbnN0IGF2Z1RpbWUgPSAocGVyZm9ybWFuY2Uubm93KCkgLSBzdGFydCkgLyBPUF9SVU5TO1xuXG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUoYVRleHR1cmUpO1xuICBncGdwdS5kZWxldGVNYXRyaXhUZXh0dXJlKGJUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZShyZXN1bHRUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlUHJvZ3JhbShwcm9ncmFtKTtcbiAgZ3BncHUuZGlzcG9zZSgpO1xuXG4gIGNvbnN0IGV4cGVjdGVkID0gdGVzdF91dGlsLmNwdU11bHRpcGx5TWF0cml4KGEsIHNpemUsIHNpemUsIGIsIHNpemUsIHNpemUpO1xuICB0ZXN0X3V0aWwuZXhwZWN0QXJyYXlzQ2xvc2UoYWN0dWFsLCBleHBlY3RlZCwgMC4wMDEpO1xuICByZXR1cm4gYXZnVGltZTtcbn07XG5cbmV4cG9ydCBjb25zdCBCRU5DSE1BUktfVEVTVF9QQUNLRUQ6IEJlbmNobWFya1Rlc3QgPSAoc2l6ZTogbnVtYmVyKSA9PiB7XG4gIGNvbnN0IGdwZ3B1ID0gbmV3IEdQR1BVQ29udGV4dCgpO1xuICBjb25zdCBwcm9ncmFtOiBXZWJHTFByb2dyYW0gPVxuICAgICAgZ3BncHUuY3JlYXRlUHJvZ3JhbShtdWxtYXRfcGFja2VkX2dwdS5nZXRGcmFnbWVudFNoYWRlclNvdXJjZShcbiAgICAgICAgICBzaXplLCBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSLCBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSk7XG5cbiAgY29uc3QgYVRleHR1cmUgPSBncGdwdS5jcmVhdGVQYWNrZWRNYXRyaXhUZXh0dXJlKHNpemUsIHNpemUpO1xuICBjb25zdCBiVGV4dHVyZSA9IGdwZ3B1LmNyZWF0ZVBhY2tlZE1hdHJpeFRleHR1cmUoc2l6ZSwgc2l6ZSk7XG4gIGNvbnN0IHJlc3VsdFRleHR1cmUgPSBncGdwdS5jcmVhdGVQYWNrZWRNYXRyaXhUZXh0dXJlKHNpemUsIHNpemUpO1xuXG4gIGNvbnN0IGEgPSB0ZXN0X3V0aWwucmFuZG9tQXJyYXlJblJhbmdlKHNpemUgKiBzaXplLCAtMSwgMSk7XG4gIGNvbnN0IGIgPSB0ZXN0X3V0aWwucmFuZG9tQXJyYXlJblJhbmdlKHNpemUgKiBzaXplLCAtMSwgMSk7XG4gIGdwZ3B1LnVwbG9hZE1hdHJpeFRvUGFja2VkVGV4dHVyZShhVGV4dHVyZSwgc2l6ZSwgc2l6ZSwgYSk7XG4gIGdwZ3B1LnVwbG9hZE1hdHJpeFRvUGFja2VkVGV4dHVyZShiVGV4dHVyZSwgc2l6ZSwgc2l6ZSwgYik7XG5cbiAgY29uc3Qgc3RhcnQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBPUF9SVU5TOyBpKyspIHtcbiAgICBtdWxtYXRfcGFja2VkX2dwdS5tdWx0aXBseU1hdHJpeFBhY2tlZChcbiAgICAgICAgZ3BncHUsIHByb2dyYW0sIGFUZXh0dXJlLCBiVGV4dHVyZSwgcmVzdWx0VGV4dHVyZSwgW3NpemUsIHNpemVdKTtcbiAgfVxuXG4gIGNvbnN0IGFjdHVhbCA9XG4gICAgICBncGdwdS5kb3dubG9hZE1hdHJpeEZyb21QYWNrZWRUZXh0dXJlKHJlc3VsdFRleHR1cmUsIHNpemUsIHNpemUpO1xuICBjb25zdCBhdmdUaW1lID0gKHBlcmZvcm1hbmNlLm5vdygpIC0gc3RhcnQpIC8gT1BfUlVOUztcblxuICBncGdwdS5kZWxldGVNYXRyaXhUZXh0dXJlKGFUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZShiVGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUocmVzdWx0VGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZVByb2dyYW0ocHJvZ3JhbSk7XG4gIGdwZ3B1LmRpc3Bvc2UoKTtcblxuICBjb25zdCBleHBlY3RlZCA9IHRlc3RfdXRpbC5jcHVNdWx0aXBseU1hdHJpeChhLCBzaXplLCBzaXplLCBiLCBzaXplLCBzaXplKTtcbiAgdGVzdF91dGlsLmV4cGVjdEFycmF5c0Nsb3NlKGFjdHVhbCwgZXhwZWN0ZWQsIDAuMDAxKTtcbiAgcmV0dXJuIGF2Z1RpbWU7XG59O1xuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQgKiBhcyBncGdwdV91dGlsIGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL2dwZ3B1X3V0aWwnO1xuaW1wb3J0ICogYXMgdGV4X3V0aWwgZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvdGV4X3V0aWwnO1xuaW1wb3J0ICogYXMgd2ViZ2xfdXRpbCBmcm9tICcuLi8uLi9zcmMvbWF0aC93ZWJnbC93ZWJnbF91dGlsJztcbmltcG9ydCAqIGFzIHRlc3RfdXRpbCBmcm9tICcuLi8uLi9zcmMvdGVzdF91dGlsJztcblxuaW1wb3J0IHtCZW5jaG1hcmtUZXN0fSBmcm9tICcuL2JlbmNobWFyayc7XG5cbmNvbnN0IE9QU19QRVJfUlVOID0gMTAwO1xuXG5leHBvcnQgY29uc3QgQkVOQ0hNQVJLX0VOQ09ERV9VTlBBQ0tFRDogQmVuY2htYXJrVGVzdCA9IChzaXplOiBudW1iZXIpID0+IHtcbiAgY29uc3QgbWF0cml4ID0gdGVzdF91dGlsLnJhbmRvbUFycmF5SW5SYW5nZShzaXplICogc2l6ZSwgLTEsIDEpO1xuICBjb25zdCBjaGFubmVsc1BlclRleHR1cmUgPSB3ZWJnbF91dGlsLmdldENoYW5uZWxzUGVyVGV4dHVyZSgpO1xuICBjb25zdCB1bnBhY2tlZEFycmF5ID1cbiAgICAgIG5ldyBGbG9hdDMyQXJyYXkodGV4X3V0aWwuZ2V0VW5wYWNrZWRBcnJheVNpemVGcm9tTWF0cml4U2l6ZShcbiAgICAgICAgICBtYXRyaXgubGVuZ3RoLCBjaGFubmVsc1BlclRleHR1cmUpKTtcbiAgY29uc3Qgc3RhcnQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBPUFNfUEVSX1JVTjsgKytpKSB7XG4gICAgdGV4X3V0aWwuZW5jb2RlTWF0cml4VG9VbnBhY2tlZEFycmF5KFxuICAgICAgICBtYXRyaXgsIHVucGFja2VkQXJyYXksIGNoYW5uZWxzUGVyVGV4dHVyZSk7XG4gIH1cbiAgY29uc3QgZW5kID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIHJldHVybiAoZW5kIC0gc3RhcnQpIC8gT1BTX1BFUl9SVU47XG59O1xuXG5leHBvcnQgY29uc3QgQkVOQ0hNQVJLX0VOQ09ERV9QQUNLRUQ6IEJlbmNobWFya1Rlc3QgPSAoc2l6ZTogbnVtYmVyKSA9PiB7XG4gIGNvbnN0IG1hdHJpeCA9IHRlc3RfdXRpbC5yYW5kb21BcnJheUluUmFuZ2Uoc2l6ZSAqIHNpemUsIC0xLCAxKTtcbiAgY29uc3QgcGFja2VkUkdCQSA9IG5ldyBGbG9hdDMyQXJyYXkoXG4gICAgICB0ZXhfdXRpbC5nZXRQYWNrZWRSR0JBQXJyYXlTaXplRnJvbU1hdHJpeFNoYXBlKHNpemUsIHNpemUpKTtcbiAgY29uc3Qgc3RhcnQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBPUFNfUEVSX1JVTjsgKytpKSB7XG4gICAgdGV4X3V0aWwuZW5jb2RlTWF0cml4VG9QYWNrZWRSR0JBKG1hdHJpeCwgc2l6ZSwgc2l6ZSwgcGFja2VkUkdCQSk7XG4gIH1cbiAgY29uc3QgZW5kID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIHJldHVybiAoZW5kIC0gc3RhcnQpIC8gT1BTX1BFUl9SVU47XG59O1xuXG5leHBvcnQgY29uc3QgQkVOQ0hNQVJLX0RFQ09ERV9VTlBBQ0tFRDogQmVuY2htYXJrVGVzdCA9IChzaXplOiBudW1iZXIpID0+IHtcbiAgY29uc3QgbWF0cml4ID0gdGVzdF91dGlsLnJhbmRvbUFycmF5SW5SYW5nZShzaXplICogc2l6ZSwgLTEsIDEpO1xuICBjb25zdCBjaGFubmVsc1BlclRleHR1cmUgPSB3ZWJnbF91dGlsLmdldENoYW5uZWxzUGVyVGV4dHVyZSgpO1xuICBjb25zdCB1bnBhY2tlZEFycmF5ID1cbiAgICAgIG5ldyBGbG9hdDMyQXJyYXkodGV4X3V0aWwuZ2V0VW5wYWNrZWRBcnJheVNpemVGcm9tTWF0cml4U2l6ZShcbiAgICAgICAgICBtYXRyaXgubGVuZ3RoLCBjaGFubmVsc1BlclRleHR1cmUpKTtcbiAgdGV4X3V0aWwuZW5jb2RlTWF0cml4VG9VbnBhY2tlZEFycmF5KFxuICAgICAgbWF0cml4LCB1bnBhY2tlZEFycmF5LCBjaGFubmVsc1BlclRleHR1cmUpO1xuICBjb25zdCBzdGFydCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IE9QU19QRVJfUlVOOyArK2kpIHtcbiAgICB0ZXhfdXRpbC5kZWNvZGVNYXRyaXhGcm9tVW5wYWNrZWRBcnJheShcbiAgICAgICAgdW5wYWNrZWRBcnJheSwgbWF0cml4LCBjaGFubmVsc1BlclRleHR1cmUpO1xuICB9XG4gIGNvbnN0IGVuZCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICByZXR1cm4gKGVuZCAtIHN0YXJ0KSAvIE9QU19QRVJfUlVOO1xufTtcblxuZXhwb3J0IGNvbnN0IEJFTkNITUFSS19ERUNPREVfUEFDS0VEOiBCZW5jaG1hcmtUZXN0ID0gKHNpemU6IG51bWJlcikgPT4ge1xuICBjb25zdCBtYXRyaXggPSB0ZXN0X3V0aWwucmFuZG9tQXJyYXlJblJhbmdlKHNpemUgKiBzaXplLCAtMSwgMSk7XG4gIGNvbnN0IHBhY2tlZFJHQkEgPSBuZXcgRmxvYXQzMkFycmF5KFxuICAgICAgdGV4X3V0aWwuZ2V0UGFja2VkUkdCQUFycmF5U2l6ZUZyb21NYXRyaXhTaGFwZShzaXplLCBzaXplKSk7XG4gIHRleF91dGlsLmVuY29kZU1hdHJpeFRvUGFja2VkUkdCQShtYXRyaXgsIHNpemUsIHNpemUsIHBhY2tlZFJHQkEpO1xuICBjb25zdCBzdGFydCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IE9QU19QRVJfUlVOOyArK2kpIHtcbiAgICB0ZXhfdXRpbC5kZWNvZGVNYXRyaXhGcm9tUGFja2VkUkdCQShwYWNrZWRSR0JBLCBzaXplLCBzaXplLCBtYXRyaXgpO1xuICB9XG4gIGNvbnN0IGVuZCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICByZXR1cm4gKGVuZCAtIHN0YXJ0KSAvIE9QU19QRVJfUlVOO1xufTtcbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblBvbHltZXIoe2lzOiAnZGVtby1mb290ZXInfSk7XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5Qb2x5bWVyKHtpczogJ2RlbW8taGVhZGVyJ30pO1xuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG4vKipcbiAqIEBmaWxlb3ZlcnZpZXdcbiAqXG4gKiBEZWZpbmVzIGFuIGludGVyZmFjZSBmb3IgY3JlYXRpbmcgUG9seW1lciBlbGVtZW50cyBpbiBUeXBlc2NyaXB0IHdpdGggdGhlXG4gKiBjb3JyZWN0IHR5cGluZ3MuIEEgUG9seW1lciBlbGVtZW50IHNob3VsZCBiZSBkZWZpbmVkIGxpa2UgdGhpczpcbiAqXG4gKiBgYGBcbiAqIGxldCBNeUVsZW1lbnRQb2x5bWVyID0gUG9seW1lckVsZW1lbnQoe1xuICogICBpczogJ215LXBvbHltZXItZWxlbWVudCcsXG4gKiAgIHByb3BlcnRpZXM6IHtcbiAqICAgICBmb286IHN0cmluZyxcbiAqICAgICBiYXI6IEFycmF5XG4gKiAgIH1cbiAqIH0pO1xuICpcbiAqIGNsYXNzIE15RWxlbWVudCBleHRlbmRzIE15RWxlbWVudFBvbHltZXIge1xuICogICBmb286IHN0cmluZztcbiAqICAgYmFyOiBudW1iZXJbXTtcbiAqXG4gKiAgIHJlYWR5KCkge1xuICogICAgIGNvbnNvbGUubG9nKCdNeUVsZW1lbnQgaW5pdGlhbGl6ZWQhJyk7XG4gKiAgIH1cbiAqIH1cbiAqXG4gKiBkb2N1bWVudC5yZWdpc3RlckVsZW1lbnQoTXlFbGVtZW50LnByb3RvdHlwZS5pcywgTXlFbGVtZW50KTtcbiAqIGBgYFxuICovXG5cbmV4cG9ydCB0eXBlIFNwZWMgPSB7XG4gIGlzOiBzdHJpbmc7IHByb3BlcnRpZXM6IHtcbiAgICBba2V5OiBzdHJpbmddOiAoRnVuY3Rpb258e1xuICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgdHlwZTogRnVuY3Rpb24sIHZhbHVlPzogYW55O1xuICAgICAgcmVmbGVjdFRvQXR0cmlidXRlPzogYm9vbGVhbjtcbiAgICAgIHJlYWRvbmx5PzogYm9vbGVhbjtcbiAgICAgIG5vdGlmeT86IGJvb2xlYW47XG4gICAgICBjb21wdXRlZD86IHN0cmluZztcbiAgICAgIG9ic2VydmVyPzogc3RyaW5nO1xuICAgIH0pXG4gIH07XG4gIG9ic2VydmVycz86IHN0cmluZ1tdO1xufTtcblxuZXhwb3J0IGZ1bmN0aW9uIFBvbHltZXJFbGVtZW50KHNwZWM6IFNwZWMpIHtcbiAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICByZXR1cm4gUG9seW1lci5DbGFzcyhzcGVjIGFzIGFueSkgYXMge25ldyAoKTogUG9seW1lckhUTUxFbGVtZW50fTtcbn1cblxuZXhwb3J0IGludGVyZmFjZSBQb2x5bWVySFRNTEVsZW1lbnQgZXh0ZW5kcyBIVE1MRWxlbWVudCwgcG9seW1lci5CYXNlIHt9XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vdXRpbCc7XG5cbmV4cG9ydCBmdW5jdGlvbiBhc3NlcnRDb25jYXQzRFNoYXBlc01hdGNoKFxuICAgIHgxU2hhcGU6IG51bWJlcltdLCB4MlNoYXBlOiBudW1iZXJbXSwgYXhpczogbnVtYmVyLFxuICAgIGVycm9yTWVzc2FnZVByZWZpeCA9ICcnKSB7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgeDFTaGFwZS5sZW5ndGggPT09IDMsXG4gICAgICBlcnJvck1lc3NhZ2VQcmVmaXggKyAnQ29uY2F0M0QgeDEgc2hhcGUgc2hvdWxkIGJlIG9mIHJhbmsgMy4nKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICB4MlNoYXBlLmxlbmd0aCA9PT0gMyxcbiAgICAgIGVycm9yTWVzc2FnZVByZWZpeCArICdDb25jYXQzRCB4MiBzaGFwZSBzaG91bGQgYmUgb2YgcmFuayAzLicpO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgYXhpcyA+PSAwICYmIGF4aXMgPCAzLCAnQXhpcyBmb3IgY29uY2F0M0QgbXVzdCBiZSBiZXR3ZWVuIDAgYW5kIDIuJyk7XG5cbiAgZm9yIChsZXQgaSA9IDA7IGkgPCAzOyBpKyspIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgKGkgPT09IGF4aXMpIHx8ICh4MVNoYXBlW2ldID09PSB4MlNoYXBlW2ldKSxcbiAgICAgICAgZXJyb3JNZXNzYWdlUHJlZml4ICtcbiAgICAgICAgICAgIGBTaGFwZSAoJHt4MVNoYXBlfSkgZG9lcyBub3QgbWF0Y2ggKCR7eDJTaGFwZX0pIGFsb25nIGAgK1xuICAgICAgICAgICAgYG5vbi1jb25jYXRlbmF0ZWQgYXhpcy5gKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gY29tcHV0ZUNvbmNhdDNET3V0cHV0U2hhcGUoXG4gICAgeDFTaGFwZTogbnVtYmVyW10sIHgyU2hhcGU6IG51bWJlcltdLFxuICAgIGF4aXM6IG51bWJlcik6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSB7XG4gIHV0aWwuYXNzZXJ0KHgxU2hhcGUubGVuZ3RoID09PSAzLCAnQ29uY2F0M0QgeDEgc2hhcGUgc2hvdWxkIGJlIG9mIHJhbmsgMy4nKTtcbiAgdXRpbC5hc3NlcnQoeDJTaGFwZS5sZW5ndGggPT09IDMsICdDb25jYXQzRCB4MnNoYXBlIHNob3VsZCBiZSBvZiByYW5rIDMuJyk7XG5cbiAgY29uc3Qgb3V0cHV0U2hhcGUgPSB4MVNoYXBlLnNsaWNlKCk7XG4gIG91dHB1dFNoYXBlW2F4aXNdICs9IHgyU2hhcGVbYXhpc107XG4gIHJldHVybiBvdXRwdXRTaGFwZSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG59IiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5leHBvcnQgZnVuY3Rpb24gY29tcHV0ZU91dHB1dFNoYXBlM0QoXG4gICAgaW5wdXRTaGFwZVJvd0NvbERlcHRoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGZpZWxkU2l6ZTogbnVtYmVyLFxuICAgIGRlcHRoOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLCB6ZXJvUGFkPzogbnVtYmVyKTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdIHtcbiAgaWYgKHplcm9QYWQgPT0gbnVsbCkge1xuICAgIHplcm9QYWQgPSBjb21wdXRlRGVmYXVsdFBhZChpbnB1dFNoYXBlUm93Q29sRGVwdGgsIGZpZWxkU2l6ZSwgc3RyaWRlKTtcbiAgfVxuICBjb25zdCBpbnB1dFJvd3MgPSBpbnB1dFNoYXBlUm93Q29sRGVwdGhbMF07XG4gIGNvbnN0IGlucHV0Q29scyA9IGlucHV0U2hhcGVSb3dDb2xEZXB0aFsxXTtcbiAgY29uc3Qgb3V0cHV0Um93cyA9IChpbnB1dFJvd3MgLSBmaWVsZFNpemUgKyAyICogemVyb1BhZCkgLyBzdHJpZGUgKyAxO1xuICB1dGlsLmFzc2VydChcbiAgICAgIHV0aWwuaXNJbnQob3V0cHV0Um93cyksXG4gICAgICBgVGhlIG91dHB1dCAjIG9mIHJvd3MgKCR7b3V0cHV0Um93c30pIG11c3QgYmUgYW4gaW50ZWdlci4gQ2hhbmdlIHRoZSBgICtcbiAgICAgICAgICBgc3RyaWRlIGFuZC9vciB6ZXJvIHBhZCBwYXJhbWV0ZXJzYCk7XG5cbiAgY29uc3Qgb3V0cHV0Q29scyA9IChpbnB1dENvbHMgLSBmaWVsZFNpemUgKyAyICogemVyb1BhZCkgLyBzdHJpZGUgKyAxO1xuICB1dGlsLmFzc2VydChcbiAgICAgIHV0aWwuaXNJbnQob3V0cHV0Q29scyksXG4gICAgICBgVGhlIG91dHB1dCAjIG9mIGNvbHVtbnMgKCR7b3V0cHV0Q29sc30pIG11c3QgYmUgYW4gaW50ZWdlci4gQ2hhbmdlIGAgK1xuICAgICAgICAgIGB0aGUgc3RyaWRlIGFuZC9vciB6ZXJvIHBhZCBwYXJhbWV0ZXJzYCk7XG5cbiAgcmV0dXJuIFtvdXRwdXRSb3dzLCBvdXRwdXRDb2xzLCBkZXB0aF07XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjb21wdXRlRGVmYXVsdFBhZChcbiAgICBpbnB1dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGZpZWxkU2l6ZTogbnVtYmVyLFxuICAgIHN0cmlkZTogbnVtYmVyKTogbnVtYmVyIHtcbiAgcmV0dXJuIE1hdGguZmxvb3IoKGlucHV0U2hhcGVbMF0gKiAoc3RyaWRlIC0gMSkgLSBzdHJpZGUgKyBmaWVsZFNpemUpIC8gMik7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjb21wdXRlVGV4U2hhcGVGcm9tM0QoXG4gICAgc2hhcGVSb3dDb2xEZXB0aDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdKTogW251bWJlciwgbnVtYmVyXSB7XG4gIHJldHVybiBbc2hhcGVSb3dDb2xEZXB0aFswXSwgc2hhcGVSb3dDb2xEZXB0aFsxXSAqIHNoYXBlUm93Q29sRGVwdGhbMl1dO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY29tcHV0ZVdlaWdodHNTaGFwZTREKFxuICAgIGlucHV0RGVwdGg6IG51bWJlciwgb3V0cHV0RGVwdGg6IG51bWJlcixcbiAgICBmU2l6ZTogbnVtYmVyKTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0ge1xuICByZXR1cm4gW2ZTaXplLCBmU2l6ZSwgaW5wdXREZXB0aCwgb3V0cHV0RGVwdGhdO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY29tcHV0ZVdlaWdodHNUZXhTaGFwZShcbiAgICBpbnB1dERlcHRoOiBudW1iZXIsIG91dHB1dERlcHRoOiBudW1iZXIsXG4gICAgZmllbGRTaXplOiBudW1iZXIpOiBbbnVtYmVyLCBudW1iZXJdIHtcbiAgcmV0dXJuIFtmaWVsZFNpemUgKiBmaWVsZFNpemUgKiBpbnB1dERlcHRoLCBvdXRwdXREZXB0aF07XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjb21wdXRlQmlhc2VzVGV4U2hhcGUob3V0cHV0RGVwdGg6IG51bWJlcik6IFtudW1iZXIsIG51bWJlcl0ge1xuICByZXR1cm4gWzEsIG91dHB1dERlcHRoXTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNvbXB1dGVEaWxhdGVkUkMoXG4gICAgcmM6IFtudW1iZXIsIG51bWJlcl0sIG9yaWdTdHJpZGU6IG51bWJlcik6IFtudW1iZXIsIG51bWJlcl0ge1xuICBjb25zdCByb3dzRGlsYXRlZCA9IChyY1swXSAtIDEpICogb3JpZ1N0cmlkZSArIDE7XG4gIGNvbnN0IGNvbHNEaWxhdGVkID0gKHJjWzFdIC0gMSkgKiBvcmlnU3RyaWRlICsgMTtcbiAgcmV0dXJuIFtyb3dzRGlsYXRlZCwgY29sc0RpbGF0ZWRdO1xufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5leHBvcnQgZnVuY3Rpb24gdmFsaWRhdGVTaGFwZXMoXG4gICAgc291cmNlU2l6ZTogW251bWJlciwgbnVtYmVyXSwgZGVzdFNpemU6IFtudW1iZXIsIG51bWJlcl0pIHtcbiAgY29uc3Qgc3JjQXJlYSA9IHNvdXJjZVNpemVbMF0gKiBzb3VyY2VTaXplWzFdO1xuICBjb25zdCBkc3RBcmVhID0gZGVzdFNpemVbMF0gKiBkZXN0U2l6ZVsxXTtcbiAgaWYgKHNyY0FyZWEgIT09IGRzdEFyZWEpIHtcbiAgICBjb25zdCBzcmNTdHIgPSAnWycgKyBzb3VyY2VTaXplWzBdICsgJywgJyArIHNvdXJjZVNpemVbMV0gKyAnXSc7XG4gICAgY29uc3QgZHN0U3RyID0gJ1snICsgZGVzdFNpemVbMF0gKyAnLCAnICsgZGVzdFNpemVbMV0gKyAnXSc7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAnY29weTJEIHNoYXBlcyBoYXZlIGRpZmZlcmVudCBhcmVhczpcXG4gIHNvdXJjZVNpemUgJyArIHNyY1N0ciArXG4gICAgICAgICcsIGFyZWEgJyArIHNyY0FyZWEgKyAnXFxuICBkZXN0U2l6ZSAnICsgZHN0U3RyICsgJywgYXJlYSAnICsgZHN0QXJlYSk7XG4gIH1cbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi91dGlsJztcbmltcG9ydCAqIGFzIGNvbmNhdDNkX3V0aWwgZnJvbSAnLi9jb25jYXQzZF91dGlsJztcbmltcG9ydCAqIGFzIGNvcHkyZF91dGlsIGZyb20gJy4vY29weTJkX3V0aWwnO1xuXG5pbXBvcnQge0FycmF5MUQsIEFycmF5MkQsIEFycmF5M0QsIEFycmF5NEQsIE5EQXJyYXksIFNjYWxhcn0gZnJvbSAnLi9uZGFycmF5JztcblxuZXhwb3J0IHR5cGUgU2NvcGVSZXN1bHQgPSBOREFycmF5W118TkRBcnJheXx2b2lkO1xuXG5leHBvcnQgYWJzdHJhY3QgY2xhc3MgTkRBcnJheU1hdGgge1xuICBwcml2YXRlIG5kYXJyYXlTY29wZXM6IE5EQXJyYXlbXVtdID0gW107XG4gIHByaXZhdGUgYWN0aXZlU2NvcGU6IE5EQXJyYXlbXTtcblxuICBwcml2YXRlIG5kYXJyYXlzVG9LZWVwOiBOREFycmF5W11bXSA9IFtdO1xuICBwcml2YXRlIGFjdGl2ZVNjb3BlTkRBcnJheXNUb0tlZXA6IE5EQXJyYXlbXSA9IFtdO1xuXG4gIC8qKlxuICAgKiBAcGFyYW0gc2FmZU1vZGUgSW4gc2FmZSBtb2RlLCB5b3UgbXVzdCB1c2UgbWF0aCBvcGVyYXRpb25zIGluc2lkZVxuICAgKiBhIG1hdGguc2NvcGUoKSB3aGljaCB3aWxsIGF1dG9tYXRpY2FsbHkgY2xlYW4gdXAgaW50ZXJtZWRpYXRlIE5EQXJyYXlzLlxuICAgKi9cbiAgY29uc3RydWN0b3IocHJpdmF0ZSBzYWZlTW9kZTogYm9vbGVhbikge31cblxuICAvKipcbiAgICogQ3JlYXRlIGEgbmV3IG1hdGggc2NvcGUuIFB1dCBjaGFpbmVkIG1hdGggb3BlcmF0aW9ucyBpbnNpZGUgYSBzY29wZVxuICAgKiBmdW5jdGlvbiBjbG9zdXJlIHNvIHRoYXQgdGhlIGxpYnJhcnkgYXV0b21hdGljYWxseSBjbGVhbnMgdXAgTkRBcnJheXNcbiAgICogZnJvbSBpbnRlcm1lZGlhdGUgbWF0aCBvcGVyYXRpb25zLiBZb3UgbXVzdCBjcmVhdGUgYSBzY29wZSBpbiBzYWZlIG1vZGVcbiAgICogdG8gY2FsbCBtYXRoIG9wZXJhdGlvbnMuIElmIGEgcmVzdWx0IGlzIHJldHVybmVkIGZyb20gdGhlIHNjb3BlLCBpdCB3aWxsXG4gICAqIGFsc28gYmUgdHJhY2tlZCwgd2hpY2ggbWVhbnMgdGhlcmUgbXVzdCBiZSB5ZXQgYW5vdGhlciB3cmFwcGluZyBzY29wZS5cbiAgICogQHBhcmFtIHNjb3BlRm4gVGhlIGZ1bmN0aW9uIHRvIGV4ZWN1dGUgd2l0aCBjaGFpbmVkIG1hdGggb3BlcmF0aW9ucy5cbiAgICovXG4gIHNjb3BlPFQgZXh0ZW5kcyBTY29wZVJlc3VsdD4oXG4gICAgICBzY29wZUZuOlxuICAgICAgICAgIChrZWVwOiA8VDEgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUMSkgPT4gVDEsXG4gICAgICAgICAgIHRyYWNrOiA8VDIgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUMikgPT4gVDIpID0+IFQpIHtcbiAgICB0aGlzLnN0YXJ0U2NvcGUoKTtcblxuICAgIGNvbnN0IGtlZXBGbiA9IDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQgPT4gdGhpcy5rZWVwKG5kYXJyYXkpO1xuICAgIGNvbnN0IHRyYWNrRm4gPSA8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUID0+IHRoaXMudHJhY2sobmRhcnJheSk7XG4gICAgY29uc3QgcmVzdWx0ID0gc2NvcGVGbihrZWVwRm4sIHRyYWNrRm4pO1xuXG4gICAgdGhpcy5lbmRTY29wZShyZXN1bHQpO1xuXG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG4gIC8qKlxuICAgKiBTdGFydCBhIHNjb3BlLiBVc2UgdGhpcyB3aXRoIGVuZFNjb3BlKCkgdG8gYWNoaWV2ZSB0aGUgc2FtZSBmdW5jdGlvbmFsaXR5XG4gICAqIGFzIHNjb3BlKCkgd2l0aG91dCB0aGUgbmVlZCBmb3IgYSBmdW5jdGlvbiBjbG9zdXJlLlxuICAgKi9cbiAgc3RhcnRTY29wZSgpIHtcbiAgICBjb25zdCBuZXdTY29wZTogTkRBcnJheVtdID0gW107XG4gICAgdGhpcy5uZGFycmF5U2NvcGVzLnB1c2gobmV3U2NvcGUpO1xuICAgIHRoaXMuYWN0aXZlU2NvcGUgPSBuZXdTY29wZTtcblxuICAgIGNvbnN0IG5ld05EQXJyYXlzVG9LZWVwOiBOREFycmF5W10gPSBbXTtcbiAgICB0aGlzLm5kYXJyYXlzVG9LZWVwLnB1c2gobmV3TkRBcnJheXNUb0tlZXApO1xuICAgIHRoaXMuYWN0aXZlU2NvcGVOREFycmF5c1RvS2VlcCA9IG5ld05EQXJyYXlzVG9LZWVwO1xuICB9XG5cbiAgLyoqXG4gICAqIEVuZCBhIHNjb3BlLiBVc2UgdGhpcyB3aXRoIHN0YXJ0U2NvcGUoKSB0byBhY2hpZXZlIHRoZSBzYW1lIGZ1bmN0aW9uYWxpdHlcbiAgICogYXMgc2NvcGUoKSB3aXRob3V0IHRoZSBuZWVkIGZvciBhIGZ1bmN0aW9uIGNsb3N1cmUuXG4gICAqL1xuICBlbmRTY29wZShyZXN1bHQ6IFNjb3BlUmVzdWx0KSB7XG4gICAgLy8gRGlzcG9zZSB0aGUgY3VycmVudCBzY29wZS5cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHRoaXMuYWN0aXZlU2NvcGUubGVuZ3RoOyBpKyspIHtcbiAgICAgIGNvbnN0IG5kYXJyYXkgPSB0aGlzLmFjdGl2ZVNjb3BlW2ldO1xuXG4gICAgICBpZiAodGhpcy5pc05EQXJyYXlEYXRhSW5MaXN0KG5kYXJyYXksIHRoaXMuYWN0aXZlU2NvcGVOREFycmF5c1RvS2VlcCkgfHxcbiAgICAgICAgICAocmVzdWx0ICE9IG51bGwgJiYgcmVzdWx0IGluc3RhbmNlb2YgTkRBcnJheSAmJlxuICAgICAgICAgICBuZGFycmF5LmdldERhdGEoKSA9PT0gKHJlc3VsdCBhcyBOREFycmF5KS5nZXREYXRhKCkpKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgbmRhcnJheS5kaXNwb3NlKCk7XG4gICAgfVxuXG4gICAgLy8gUG9wIHRoZSBjdXJyZW50IHNjb3BlLlxuICAgIHRoaXMubmRhcnJheVNjb3Blcy5wb3AoKTtcbiAgICB0aGlzLmFjdGl2ZVNjb3BlID0gdGhpcy5uZGFycmF5U2NvcGVzLmxlbmd0aCA9PT0gMCA/XG4gICAgICAgIG51bGwhIDpcbiAgICAgICAgdGhpcy5uZGFycmF5U2NvcGVzW3RoaXMubmRhcnJheVNjb3Blcy5sZW5ndGggLSAxXTtcblxuICAgIC8vIFRyYWNrIHRoZSBjdXJyZW50IHJlc3VsdCBpbiB0aGUgcGFyZW50IHNjb3BlLlxuICAgIGlmIChyZXN1bHQgaW5zdGFuY2VvZiBOREFycmF5ICYmXG4gICAgICAgICF0aGlzLmlzTkRBcnJheURhdGFJbkxpc3QocmVzdWx0LCB0aGlzLmFjdGl2ZVNjb3BlTkRBcnJheXNUb0tlZXApKSB7XG4gICAgICB0aGlzLnRyYWNrKHJlc3VsdCk7XG4gICAgfSBlbHNlIGlmIChBcnJheS5pc0FycmF5KHJlc3VsdCkpIHtcbiAgICAgIHJlc3VsdC5mb3JFYWNoKHIgPT4ge1xuICAgICAgICBpZiAociBpbnN0YW5jZW9mIE5EQXJyYXkgJiZcbiAgICAgICAgICAgICF0aGlzLmlzTkRBcnJheURhdGFJbkxpc3QociwgdGhpcy5hY3RpdmVTY29wZU5EQXJyYXlzVG9LZWVwKSkge1xuICAgICAgICAgIHRoaXMudHJhY2socik7XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgIH1cblxuICAgIHRoaXMubmRhcnJheXNUb0tlZXAucG9wKCk7XG4gICAgdGhpcy5hY3RpdmVTY29wZU5EQXJyYXlzVG9LZWVwID0gdGhpcy5uZGFycmF5c1RvS2VlcC5sZW5ndGggPT09IDAgP1xuICAgICAgICBudWxsISA6XG4gICAgICAgIHRoaXMubmRhcnJheXNUb0tlZXBbdGhpcy5uZGFycmF5c1RvS2VlcC5sZW5ndGggLSAxXTtcbiAgfVxuXG4gIHByaXZhdGUgaXNOREFycmF5RGF0YUluTGlzdChuZGFycmF5OiBOREFycmF5LCBuZGFycmF5TGlzdDogTkRBcnJheVtdKSB7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBuZGFycmF5TGlzdC5sZW5ndGg7IGkrKykge1xuICAgICAgaWYgKG5kYXJyYXlMaXN0W2ldLmdldERhdGEoKSA9PT0gbmRhcnJheS5nZXREYXRhKCkpIHtcbiAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBmYWxzZTtcbiAgfVxuXG4gIC8qKlxuICAgKiBLZWVwcyBhbiBOREFycmF5IGluIHRoZSBjdXJyZW50IHNjb3BlIGZyb20gYmVpbmcgZGlzcG9zZWQgYXV0b21hdGljYWxseS5cbiAgICogQHBhcmFtIHJlc3VsdCBUaGUgTkRBcnJheSB0byBrZWVwIGZyb20gYmVpbmcgZGlzcG9zZWQuXG4gICAqL1xuICBrZWVwPFQgZXh0ZW5kcyBOREFycmF5PihyZXN1bHQ6IFQpOiBUIHtcbiAgICBpZiAodGhpcy5hY3RpdmVTY29wZSA9PSBudWxsKSB7XG4gICAgICBpZiAodGhpcy5zYWZlTW9kZSkge1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgICAnWW91IGFyZSB1c2luZyBtYXRoIGluIHNhZmUgbW9kZS4gRW5jbG9zZSBhbGwgJyArXG4gICAgICAgICAgICAnbWF0aC5tZXRob2QoKSBjYWxscyBpbnNpZGUgYSBzY29wZTogJyArXG4gICAgICAgICAgICAnbWF0aC5zY29wZSgoKSA9PiB7bWF0aC5tZXRob2QoKTsuLi59KSB0byBhdm9pZCBtZW1vcnkgJyArXG4gICAgICAgICAgICAnbGVha3MuJyk7XG4gICAgICB9XG4gICAgICByZXR1cm4gcmVzdWx0O1xuICAgIH1cbiAgICB0aGlzLmFjdGl2ZVNjb3BlTkRBcnJheXNUb0tlZXAucHVzaChyZXN1bHQpO1xuICAgIHJldHVybiByZXN1bHQ7XG4gIH1cblxuICAvKipcbiAgICogVHJhY2tzIGFuIE5EQXJyYXkgaW4gdGhlIGN1cnJlbnQgc2NvcGUgdG8gYmUgYXV0b21hdGljYWxseSBjbGVhbmVkIHVwIHdoZW5cbiAgICogdGhlIGN1cnJlbnQgc2NvcGUgZW5kcywgYW5kIHJldHVybnMgdGhlIHZhbHVlLlxuICAgKiBAcGFyYW0gcmVzdWx0IFRoZSBOREFycmF5IHRvIHRyYWNrIGluIHRoZSBjdXJyZW50IHNjb3BlLlxuICAgKi9cbiAgdHJhY2s8VCBleHRlbmRzIE5EQXJyYXk+KHJlc3VsdDogVCk6IFQge1xuICAgIGlmICh0aGlzLmFjdGl2ZVNjb3BlID09IG51bGwpIHtcbiAgICAgIGlmICh0aGlzLnNhZmVNb2RlKSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgICdZb3UgYXJlIHVzaW5nIG1hdGggaW4gc2FmZSBtb2RlLiBFbmNsb3NlIGFsbCAnICtcbiAgICAgICAgICAgICdtYXRoLm1ldGhvZCgpIGNhbGxzIGluc2lkZSBhIHNjb3BlOiAnICtcbiAgICAgICAgICAgICdtYXRoLnNjb3BlKCgpID0+IHttYXRoLm1ldGhvZCgpOy4uLn0pIHRvIGF2b2lkIG1lbW9yeSAnICtcbiAgICAgICAgICAgICdsZWFrcy4nKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiByZXN1bHQ7XG4gICAgfVxuICAgIHRoaXMuYWN0aXZlU2NvcGUucHVzaChyZXN1bHQpO1xuICAgIHJldHVybiByZXN1bHQ7XG4gIH1cblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIGRvdCBwcm9kdWN0IG9mIHR3byBtYXRyaWNlcywgQSAqIEIuIFRoZXNlIG11c3QgYmUgbWF0cmljZXMsXG4gICAqIHVzZSBtYXRyaXhUaW1lc1ZlY3RvciBhbmQgdmVjdG9yVGltZXNNYXRyaXgsIGRvdFByb2R1Y3QsIGFuZCBvdXRlclByb2R1Y3RcbiAgICogaW4gb3RoZXIgY2FzZXMuXG4gICAqIEBwYXJhbSBhIEZpcnN0IG1hdHJpeCBpbiBkb3QgcHJvZHVjdCBvcGVyYXRpb24uXG4gICAqIEBwYXJhbSBiIFNlY29uZCBtYXRyaXggaW4gZG90IHByb2R1Y3Qgb3BlcmF0aW9uLlxuICAgKiBAcGFyYW0gYU9yaWVudGF0aW9uIFRoZSBNYXRyaXhPcmllbnRhdGlvbiBvZiBBLiBJZiB1c2luZyBUUkFOU1BPU0VELCB3aWxsXG4gICAqIGNvbXB1dGUgQV5UICogQi5cbiAgICogQHBhcmFtIGJPcmllbnRhdGlvbiBUaGUgTWF0cml4T3JpZW50YXRpb24gb2YgQi4gSWYgdXNpbmcgVFJBTlNQT1NFRCwgd2lsbFxuICAgKiBjb21wdXRlIEEgKiBCXlQuXG4gICAqL1xuICBtYXRNdWwoXG4gICAgICBhOiBBcnJheTJELCBiOiBBcnJheTJELCBhT3JpZW50YXRpb24gPSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSLFxuICAgICAgYk9yaWVudGF0aW9uID0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUik6IEFycmF5MkQge1xuICAgIGNvbnN0IGlubmVyU2hhcGVBID1cbiAgICAgICAgKGFPcmllbnRhdGlvbiA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUikgPyBhLnNoYXBlWzFdIDogYS5zaGFwZVswXTtcbiAgICBjb25zdCBpbm5lclNoYXBlQiA9XG4gICAgICAgIChiT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID8gYi5zaGFwZVswXSA6IGIuc2hhcGVbMV07XG5cbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgYS5yYW5rID09PSAyICYmIGIucmFuayA9PT0gMixcbiAgICAgICAgYEVycm9yIGluIG1hdE11bDogaW5wdXRzIG11c3QgYmUgcmFuayAyLCBnb3QgcmFua3MgJHthLnJhbmt9YCArXG4gICAgICAgICAgICBgYW5kICR7Yi5yYW5rfS5gKTtcblxuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBpbm5lclNoYXBlQSA9PT0gaW5uZXJTaGFwZUIsXG4gICAgICAgIGBFcnJvciBpbiBtYXRNdWw6IGlubmVyIHNoYXBlcyAoJHtpbm5lclNoYXBlQX0pIGFuZCAoYCArXG4gICAgICAgICAgICBgJHtpbm5lclNoYXBlQn0pIG9mIE5EQXJyYXlzIHdpdGggc2hhcGVzICR7YS5zaGFwZX0gYW5kIGAgK1xuICAgICAgICAgICAgYCR7Yi5zaGFwZX0gYW5kIG9yaWVudGF0aW9ucyAke01hdHJpeE9yaWVudGF0aW9uW2FPcmllbnRhdGlvbl19YCArXG4gICAgICAgICAgICBgIGFuZCAke01hdHJpeE9yaWVudGF0aW9uW2JPcmllbnRhdGlvbl19IG11c3QgbWF0Y2guYCk7XG5cbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLm1hdE11bEludGVybmFsKGEsIGIsIGFPcmllbnRhdGlvbiwgYk9yaWVudGF0aW9uKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IG1hdE11bEludGVybmFsKFxuICAgICAgYTogQXJyYXkyRCwgYjogQXJyYXkyRCwgYU9yaWVudGF0aW9uOiBNYXRyaXhPcmllbnRhdGlvbixcbiAgICAgIGJPcmllbnRhdGlvbjogTWF0cml4T3JpZW50YXRpb24pOiBBcnJheTJEO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgZG90IHByb2R1Y3Qgb2YgYSB2ZWN0b3IgYW5kIGEgbWF0cml4LCB2ICogQi5cbiAgICogQHBhcmFtIHYgVGhlIHZlY3RvciBpbiBkb3QgcHJvZHVjdCBvcGVyYXRpb24uXG4gICAqIEBwYXJhbSBtYXRyaXggVGhlIG1hdHJpeCBpbiBkb3QgcHJvZHVjdCBvcGVyYXRpb24uXG4gICAqL1xuICB2ZWN0b3JUaW1lc01hdHJpeCh2OiBBcnJheTFELCBtYXRyaXg6IEFycmF5MkQpOiBBcnJheTFEIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgdi5yYW5rID09PSAxLFxuICAgICAgICBgRXJyb3IgaW4gdmVjdG9yVGltZXNNYXRyaXg6IGZpcnN0IGlucHV0IG11c3QgYmUgcmFuayAxLCBidXQgZ290IGAgK1xuICAgICAgICAgICAgYHJhbmsgJHt2LnJhbmt9LmApO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBtYXRyaXgucmFuayA9PT0gMixcbiAgICAgICAgYEVycm9yIGluIHZlY3RvclRpbWVzTWF0cml4OiBzZWNvbmQgaW5wdXQgbXVzdCBiZSByYW5rIDIsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgICBgcmFuayAke21hdHJpeC5yYW5rfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgdi5zaXplID09PSBtYXRyaXguc2hhcGVbMF0sXG4gICAgICAgIGBFcnJvciBpbiB2ZWN0b3JUaW1lc01hdHJpeDogc2l6ZSBvZiBmaXJzdCByYW5rIDEgaW5wdXQgKCR7di5zaXplfSkgYCArXG4gICAgICAgICAgICBgbXVzdCBtYXRjaCBpbm5lciBkaW1lbnNpb24gb2Ygc2Vjb25kIHJhbmsgMiBpbnB1dCwgYnV0IGdvdCBgICtcbiAgICAgICAgICAgIGByYW5rICR7bWF0cml4LnJhbmt9LmApO1xuXG4gICAgcmV0dXJuIHRoaXMubWF0TXVsKHYuYXMyRCgxLCB2LnNpemUpLCBtYXRyaXgpLmFzMUQoKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgZG90IHByb2R1Y3Qgb2YgYSBtYXRyaXggYW5kIHZlY3RvciwgQSAqIHYuXG4gICAqIEBwYXJhbSBtYXRyaXggVGhlIG1hdHJpeCBpbiBkb3QgcHJvZHVjdCBvcGVyYXRpb24uXG4gICAqIEBwYXJhbSB2IFRoZSB2ZWN0b3IgaW4gZG90IHByb2R1Y3Qgb3BlcmF0aW9uLlxuICAgKi9cbiAgbWF0cml4VGltZXNWZWN0b3IobWF0cml4OiBBcnJheTJELCB2OiBBcnJheTFEKTogQXJyYXkxRCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHYucmFuayA9PT0gMSxcbiAgICAgICAgYEVycm9yIGluIHZlY3RvclRpbWVzTWF0cml4OiBzZWNvbmQgaW5wdXQgbXVzdCByYW5rIDEsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgICBgcmFuayAke3YucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIG1hdHJpeC5yYW5rID09PSAyLFxuICAgICAgICBgRXJyb3IgaW4gdmVjdG9yVGltZXNNYXRyaXg6IGZpcnN0IGlucHV0IG11c3QgYmUgYSByYW5rIDIsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgICBgcmFuayAke21hdHJpeC5yYW5rfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgdi5zaXplID09PSBtYXRyaXguc2hhcGVbMV0sXG4gICAgICAgIGBFcnJvciBpbiB2ZWN0b3JUaW1lc01hdHJpeDogc2l6ZSBvZiBmaXJzdCByYW5rIDEgaW5wdXQgJHt2LnNpemV9IGAgK1xuICAgICAgICAgICAgYG11c3QgbWF0Y2ggaW5uZXIgZGltZW5zaW9uIG9mIHNlY29uZCByYW5rIDIgaW5wdXQsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgICBgc2hhcGUgJHttYXRyaXguc2hhcGV9LmApO1xuXG4gICAgcmV0dXJuIHRoaXMubWF0TXVsKG1hdHJpeCwgdi5hczJEKHYuc2l6ZSwgMSkpLmFzMUQoKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgZG90IHByb2R1Y3Qgb2YgdHdvIHZlY3RvcnMsIHYxICogdjIuXG4gICAqIEBwYXJhbSB2MSBUaGUgZmlyc3QgdmVjdG9yIGluIHRoZSBkb3QgcHJvZHVjdCBvcGVyYXRpb24uXG4gICAqIEBwYXJhbSB2MiBUaGUgc2Vjb25kIHZlY3RvciBpbiB0aGUgZG90IHByb2R1Y3Qgb3BlcmF0aW9uLlxuICAgKi9cbiAgZG90UHJvZHVjdCh2MTogQXJyYXkxRCwgdjI6IEFycmF5MUQpOiBTY2FsYXIge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB2MS5yYW5rID09PSAxICYmIHYyLnJhbmsgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBkb3RQcm9kdWN0OiBpbnB1dHMgbXVzdCBiZSByYW5rIDEsIGJ1dCBnb3QgcmFua3MgYCArXG4gICAgICAgICAgICBgJHt2MS5yYW5rfSBhbmQgJHt2Mi5yYW5rfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgdjEuc2l6ZSA9PT0gdjIuc2l6ZSxcbiAgICAgICAgYEVycm9yIGluIGRvdFByb2R1Y3Q6IHNpemUgb2YgaW5wdXRzICgke3YxLnNpemV9KSBhbmQgKGAgK1xuICAgICAgICAgICAgYCR7djIuc2l6ZX0pIG11c3QgbWF0Y2guYCk7XG4gICAgcmV0dXJuIHRoaXMubWF0TXVsKHYxLmFzMkQoMSwgdjEuc2l6ZSksIHYyLmFzMkQodjIuc2l6ZSwgMSkpLmFzU2NhbGFyKCk7XG4gIH1cblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIG91dGVyIHByb2R1Y3Qgb2YgdHdvIHZlY3RvcnMsIHYxIGFuZCB2Mi5cbiAgICogQHBhcmFtIHYxIFRoZSBmaXJzdCB2ZWN0b3IgaW4gdGhlIG91dGVyIHByb2R1Y3Qgb3BlcmF0aW9uLlxuICAgKiBAcGFyYW0gdjIgVGhlIHNlY29uZCB2ZWN0b3IgaW4gdGhlIGRvdCBwcm9kdWN0IG9wZXJhdGlvbi5cbiAgICovXG4gIG91dGVyUHJvZHVjdCh2MTogQXJyYXkxRCwgdjI6IEFycmF5MUQpOiBBcnJheTJEIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgdjEucmFuayA9PT0gMSAmJiB2Mi5yYW5rID09PSAxLFxuICAgICAgICBgRXJyb3IgaW4gb3V0ZXJQcm9kdWN0OiBpbnB1dHMgbXVzdCBiZSByYW5rIDEsIGJ1dCBnb3QgcmFua3MgYCArXG4gICAgICAgICAgICBgJHt2MS5yYW5rfSBhbmQgJHt2Mi5yYW5rfS5gKTtcblxuICAgIHJldHVybiB0aGlzLm1hdE11bCh2MS5hczJEKHYxLnNpemUsIDEpLCB2Mi5hczJEKDEsIHYyLnNpemUpKTtcbiAgfVxuXG4gIC8vLy8vLy8vLy8vLy8vL1xuICAvLyBTaGFwZSBvcHMgLy9cbiAgLy8vLy8vLy8vLy8vLy8vXG5cbiAgLyoqXG4gICAqIENsb25lcyBhbiBOREFycmF5IG9mIGFueSBzaGFwZS5cbiAgICogQHBhcmFtIG5kYXJyYXkgVGhlIE5EQXJyYXkgdG8gY2xvbmUuXG4gICAqL1xuICBjbG9uZTxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQge1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMuY2xvbmVJbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGNsb25lSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUO1xuXG4gIC8qKlxuICAgKiBSZXNoYXBlcyBhbiBOREFycmF5IHRvIGEgbmV3IHNoYXBlLiBUaGUgc2l6ZSBvZiB0aGUgaW5wdXQgTkRBcnJheSBtdXN0XG4gICAqIG1hdGNoIHRoZSBzaXplIG9mIHRoZSByZXF1ZXN0ZWQgc2hhcGUuXG4gICAqIEBwYXJhbSBuZGFycmF5IFRoZSBpbnB1dCBOREFycmF5LlxuICAgKiBAcGFyYW0gbmV3U2hhcGUgVGhlIG5ldyBzaGFwZSB0byByZXNoYXBlIHRoZSBOREFycmF5IHRvLiBNdXN0IGJlIHRoZSBzYW1lXG4gICAqIHNpemUgYXMgdGhlIE5EQXJyYXkuXG4gICAqL1xuICByZXNoYXBlPFQxIGV4dGVuZHMgTkRBcnJheSwgVDIgZXh0ZW5kcyBOREFycmF5PihcbiAgICAgIG5kYXJyYXk6IFQxLCBuZXdTaGFwZTogbnVtYmVyW10pOiBUMiB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIG5kYXJyYXkuc2l6ZSA9PT0gdXRpbC5zaXplRnJvbVNoYXBlKG5ld1NoYXBlKSxcbiAgICAgICAgYEVycm9yIGluIHJlc2hhcGU6IG9sZCBzaXplICR7bmRhcnJheS5zaXplfSBtdXN0IG1hdGNoIG5ldyBzaXplIGAgK1xuICAgICAgICAgICAgYCR7dXRpbC5zaXplRnJvbVNoYXBlKG5ld1NoYXBlKX0uYCk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5yZXNoYXBlSW50ZXJuYWw8VDEsIFQyPihuZGFycmF5LCBuZXdTaGFwZSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCByZXNoYXBlSW50ZXJuYWw8VDEgZXh0ZW5kcyBOREFycmF5LCBUMiBleHRlbmRzIE5EQXJyYXk+KFxuICAgICAgbmRhcnJheTogVDEsIG5ld1NoYXBlOiBudW1iZXJbXSk6IFQyO1xuXG4gIC8qKlxuICAgKiBFeHRyYWN0cyBhIHNsaWNlIGZyb20gYSBtYXRyaXguIFRoZSBvcGVyYXRpb24gZXh0cmFjZXMgYSBzbGljZSBmcm9tIGlucHV0XG4gICAqIHRoYXQgc3RhcnRzIGF0IGNvb3JkaW5hdGVzIGBiZWdpbmAgYW5kIGlzIG9mIHNpemUgYHNpemVgLlxuICAgKiBAcGFyYW0gaW5wdXQgVGhlIGlucHV0IG1hdHJpeCB0byBzbGljZSBmcm9tLlxuICAgKiBAcGFyYW0gYmVnaW4gVGhlIDJEIGNvb3JkaW5hdGVzIGluIHRoZSBpbnB1dCBtYXRyaXggdG8gc3RhcnQgdGhlIHNsaWNlXG4gICAqIGZyb20uXG4gICAqIEBwYXJhbSBzaXplIFRoZSBzaWNlIG9mIHRoZSAyRCB3aW5kb3cgdG8gc2xpY2UuXG4gICAqL1xuICBzbGljZTJEKGlucHV0OiBBcnJheTJELCBiZWdpbjogW251bWJlciwgbnVtYmVyXSwgc2l6ZTogW251bWJlciwgbnVtYmVyXSk6XG4gICAgICBBcnJheTJEIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgYmVnaW5bMF0gKyBzaXplWzBdIDw9IGlucHV0LnNoYXBlWzBdICYmXG4gICAgICAgICAgICBiZWdpblsxXSArIHNpemVbMV0gPD0gaW5wdXQuc2hhcGVbMV0sXG4gICAgICAgIGBFcnJvciBpbiBzbGljZTJEOiByZXF1ZXN0ZWQgc3RhcnQgcG9zaXRpb24gJHtiZWdpbn0gYW5kIHNpemUgYCArXG4gICAgICAgICAgICBgJHtzaXplfSB3b3VsZCBvdmVyZmxvdyBpbnB1dCBvZiBzaGFwZSAke2lucHV0LnNoYXBlfS5gKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLnNsaWNlMkRJbnRlcm5hbChpbnB1dCwgYmVnaW4sIHNpemUpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3Qgc2xpY2UyREludGVybmFsKFxuICAgICAgaW5wdXQ6IEFycmF5MkQsIGJlZ2luOiBbbnVtYmVyLCBudW1iZXJdLCBzaXplOiBbbnVtYmVyLCBudW1iZXJdKTogQXJyYXkyRDtcblxuICAvKipcbiAgICogQ29waWVzIGEgd2luZG93IGZyb20gdGhlIGBzb3VyY2VgIG1hdHJpeCBzdGFydGluZyBhdCBgc291cmNlQmVnaW5gIGFuZCBpc1xuICAgKiBvZiBzaXplIGBzb3VyY2VTaXplYCB0byBhIHdpbmRvdyBpbiB0aGUgYGRlc3RgIG1hdHJpeCBzdGFydGluZyBhdFxuICAgKiBgZGVzdEJlZ2luYCBhbmQgaXMgb2Ygc2l6ZSBgZGVzdFNpemVgL1xuICAgKiBAcGFyYW0gc291cmNlIFRoZSBzb3VyY2UgbWF0cml4IHRvIGNvcHkgZnJvbS5cbiAgICogQHBhcmFtIHNvdXJjZUJlZ2luIFRoZSBjb29yZGluYXRlcyB0byBzdGFydCB0aGUgY29weSBmcm9tLlxuICAgKiBAcGFyYW0gc291cmNlU2l6ZSBUaGUgc2l6ZSBvZiB0aGUgY29weSB3aW5kb3cuXG4gICAqIEBwYXJhbSBkZXN0IFRoZSBkZXN0aW5hdGlvbiBtYXRyaXggdG8gY29weSB0by5cbiAgICogQHBhcmFtIGRlc3RCZWdpbiBUaGUgY29vcmRpbmF0ZXMgaW4gYGRlc3RgIHRvIGNvcHkgdG8uXG4gICAqIEBwYXJhbSBkZXN0U2l6ZSBUaGUgc2l6ZSBvZiB0aGUgZGVzdGluYXRpb24gd2luZG93LlxuICAgKi9cbiAgY29weTJEKFxuICAgICAgc291cmNlOiBBcnJheTJELCBzb3VyY2VCZWdpbjogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIHNvdXJjZVNpemU6IFtudW1iZXIsIG51bWJlcl0sIGRlc3Q6IEFycmF5MkQsIGRlc3RCZWdpbjogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIGRlc3RTaXplOiBbbnVtYmVyLCBudW1iZXJdKSB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHNvdXJjZUJlZ2luWzBdICsgc291cmNlU2l6ZVswXSA8PSBzb3VyY2Uuc2hhcGVbMF0gJiZcbiAgICAgICAgICAgIHNvdXJjZUJlZ2luWzFdICsgc291cmNlU2l6ZVsxXSA8PSBzb3VyY2Uuc2hhcGVbMV0sXG4gICAgICAgIGBFcnJvciBpbiBjb3B5MkQ6IHJlcXVlc3RlZCBzb3VyY2Ugc3RhcnQgcG9zaXRpb24gJHtzb3VyY2VCZWdpbn0gYCArXG4gICAgICAgICAgICBgYW5kIHNvdXJjZSBzaXplICR7c291cmNlU2l6ZX0gd291bGQgb3ZlcmZsb3cgc291cmNlIE5EQXJyYXlgICtcbiAgICAgICAgICAgIGBvZiBzaGFwZSAke3NvdXJjZS5zaGFwZX0uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGRlc3RCZWdpblswXSArIGRlc3RTaXplWzBdIDw9IGRlc3Quc2hhcGVbMF0gJiZcbiAgICAgICAgICAgIGRlc3RCZWdpblsxXSArIGRlc3RTaXplWzFdIDw9IGRlc3Quc2hhcGVbMV0sXG4gICAgICAgIGBFcnJvciBpbiBjb3B5MkQ6IHJlcXVlc3RlZCBkZXN0IHN0YXJ0IHBvc2l0aW9uICR7ZGVzdEJlZ2lufSBgICtcbiAgICAgICAgICAgIGBhbmQgc291cmNlIHNpemUgJHtkZXN0U2l6ZX0gd291bGQgb3ZlcmZsb3cgZGVzdCBOREFycmF5IG9mYCArXG4gICAgICAgICAgICBgc2hhcGUgJHtkZXN0LnNoYXBlfS5gKTtcbiAgICBjb3B5MmRfdXRpbC52YWxpZGF0ZVNoYXBlcyhzb3VyY2VTaXplLCBkZXN0U2l6ZSk7XG5cbiAgICByZXR1cm4gdGhpcy5jb3B5MkRJbnRlcm5hbChcbiAgICAgICAgc291cmNlLCBzb3VyY2VCZWdpbiwgc291cmNlU2l6ZSwgZGVzdCwgZGVzdEJlZ2luLCBkZXN0U2l6ZSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGNvcHkyREludGVybmFsKFxuICAgICAgc291cmNlOiBBcnJheTJELCBzb3VyY2VCZWdpbjogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIHNvdXJjZVNpemU6IFtudW1iZXIsIG51bWJlcl0sIGRlc3Q6IEFycmF5MkQsIGRlc3RCZWdpbjogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIGRlc3RTaXplOiBbbnVtYmVyLCBudW1iZXJdKTogdm9pZDtcblxuICAvKipcbiAgICogQ29uY2F0ZW5hdGVzIHR3byAzRCBuZGFycmF5cyBhbG9uZyBhIGdpdmVuIGF4aXMuXG4gICAqXG4gICAqIEZvciBleGFtcGxlLCBpZjpcbiAgICogQTogc2hhcGUoMiwgMSwgMykgPSB8IHIxLCBnMSwgYjEgfFxuICAgKiAgICAgICAgICAgICAgICAgICAgIHwgcjIsIGcyLCBiMiB8XG4gICAqXG4gICAqIEI6IHNoYXBlKDIsIDEsIDMpID0gfCByMywgZzMsIGIzIHxcbiAgICogICAgICAgICAgICAgICAgICAgICB8IHI0LCBnNCwgYjQgfFxuICAgKlxuICAgKiBDID0gY29uY2F0M0QoQSwgQiwgYXhpcylcbiAgICpcbiAgICogaWYgYXhpcyA9IDA6XG4gICAqIEM6IHNoYXBlKDQsIDEsIDMpID0gfCByMSwgZzEsIGIxIHxcbiAgICogICAgICAgICAgICAgICAgICAgICB8IHIyLCBnMiwgYjIgfFxuICAgKiAgICAgICAgICAgICAgICAgICAgIHwgcjMsIGczLCBiMyB8XG4gICAqICAgICAgICAgICAgICAgICAgICAgfCByNCwgZzQsIGI0IHxcbiAgICpcbiAgICogaWYgYXhpcyA9IDE6XG4gICAqIEM6IHNoYXBlKDIsIDIsIDMpID0gfCByMSwgZzEsIGIxLCByMywgZzMsIGIzIHxcbiAgICogICAgICAgICAgICAgICAgICAgICB8IHIyLCBnMiwgYjIsIHI0LCBnNCwgYjQgfFxuICAgKlxuICAgKiBpZiBheGlzID0gMjpcbiAgICogQyA9IHNoYXBlKDIsIDEsIDYpID0gfCByMSwgZzEsIGIxLCByMywgZzMsIGIzIHxcbiAgICogICAgICAgICAgICAgICAgICAgICAgfCByMiwgZzIsIGIyLCByNCwgZzQsIGI0IHxcbiAgICpcbiAgICogQHBhcmFtIG5kYXJyYXkxIFRoZSBmaXJzdCBhcnJheSB0byBjb25jYXQuXG4gICAqIEBwYXJhbSBuZGFycmF5MiBUaGUgc2Vjb25kIGFycmF5IHRvIGNvbmF0LlxuICAgKiBAcGFyYW0gYXhpcyBUaGUgYXhpcyB0byBjb25jYXRlIGFsb25nLlxuICAgKi9cbiAgY29uY2F0M0QobmRhcnJheTE6IEFycmF5M0QsIG5kYXJyYXkyOiBBcnJheTNELCBheGlzOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICBjb25jYXQzZF91dGlsLmFzc2VydENvbmNhdDNEU2hhcGVzTWF0Y2goXG4gICAgICAgIG5kYXJyYXkxLnNoYXBlLCBuZGFycmF5Mi5zaGFwZSwgYXhpcywgJ0Vycm9yIGluIGNvbmNhdDNkOiAnKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmNvbmNhdDNESW50ZXJuYWwobmRhcnJheTEsIG5kYXJyYXkyLCBheGlzKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGNvbmNhdDNESW50ZXJuYWwoXG4gICAgICBuZGFycmF5MTogQXJyYXkzRCwgbmRhcnJheTI6IEFycmF5M0QsIGF4aXM6IG51bWJlcik6IEFycmF5M0Q7XG5cbiAgLy8vLy8vLy8vLy8vLy8vLy8vL1xuICAvLyBSZWR1Y3Rpb24gb3BzIC8vXG4gIC8vLy8vLy8vLy8vLy8vLy8vLy9cblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIHRoZSBsb2coc3VtKGUgXiB4KSkgZm9yIGVhY2ggeCBpbiB0aGUgaW5wdXQgbmRhcnJheS5cbiAgICogQHBhcmFtIG5kYXJyYXkgVGhlIGlucHV0IE5EQXJyYXkgdG8gY29tcHV0ZSB0aGUgbG9nU3VtRXhwIG92ZXIuXG4gICAqL1xuICBsb2dTdW1FeHAobmRhcnJheTogTkRBcnJheSk6IFNjYWxhciB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5sb2dTdW1FeHBJbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGxvZ1N1bUV4cEludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXI7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSBzdW0gb2YgYWxsIHRoZSBlbnRyaWVzIGluIHRoZSBpbnB1dCBOREFycmF5LlxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheSB0byBjb21wdXRlIHRoZSBzdW0gb3Zlci5cbiAgICovXG4gIHN1bShuZGFycmF5OiBOREFycmF5KTogU2NhbGFyIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLnN1bUludGVybmFsKG5kYXJyYXkpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3Qgc3VtSW50ZXJuYWwobmRhcnJheTogTkRBcnJheSk6IFNjYWxhcjtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIGZsYXR0ZW5lZCBpbmRleCBvZiB0aGUgbWluaW11bSBlbGVtZW50IGluIHRoZSBuZGFycmF5LlxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheS5cbiAgICovXG4gIGFyZ01pbihuZGFycmF5OiBOREFycmF5KTogU2NhbGFyIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmFyZ01pbkludGVybmFsKG5kYXJyYXkpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgYXJnTWluSW50ZXJuYWwobmRhcnJheTogTkRBcnJheSk6IFNjYWxhcjtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIGZsYXR0ZW5lZCBpbmRleCBvZiB0aGUgbWF4aW11bSBlbGVtZW50IGluIHRoZSBuZGFycmF5LlxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheS5cbiAgICovXG4gIGFyZ01heChuZGFycmF5OiBOREFycmF5KTogU2NhbGFyIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmFyZ01heEludGVybmFsKG5kYXJyYXkpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgYXJnTWF4SW50ZXJuYWwobmRhcnJheTogTkRBcnJheSk6IFNjYWxhcjtcblxuICAvKipcbiAgICogUmV0dXJucyBhIDEgaWYgdGhlIGFyZ01heCBvZiB4MSBhbmQgeDIgYXJlIHRoZSBzYW1lLCBvdGhlcndpc2UgMC5cbiAgICogQHBhcmFtIHgxIFRoZSBmaXJzdCBpbnB1dCBOREFycmF5LlxuICAgKiBAcGFyYW0geDIgVGhlIHNlY29uZCBpbnB1dCBOREFycmF5LlxuICAgKi9cbiAgYXJnTWF4RXF1YWxzKHgxOiBOREFycmF5LCB4MjogTkRBcnJheSk6IFNjYWxhciB7XG4gICAgdXRpbC5hc3NlcnRTaGFwZXNNYXRjaCh4MS5zaGFwZSwgeDIuc2hhcGUsICdFcnJvciBpbiBhcmdNYXhFcXVhbHM6ICcpO1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMuYXJnTWF4RXF1YWxzSW50ZXJuYWwoeDEsIHgyKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGFyZ01heEVxdWFsc0ludGVybmFsKHgxOiBOREFycmF5LCB4MjogTkRBcnJheSk6IFNjYWxhcjtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIHRvcCBLIHZhbHVlcyBhbmQgZmxhdHRlbmVkIGluZGljZXMuXG4gICAqIEBwYXJhbSBuZGFycmF5IFRoZSBpbnB1dCBOREFycmF5LlxuICAgKiBAcGFyYW0gayBIb3cgbWFueSB0b3AgdmFsdWVzIHRvIGNvbXB1dGUuXG4gICAqL1xuICB0b3BLKG5kYXJyYXk6IE5EQXJyYXksIGs6IG51bWJlcik6IHt2YWx1ZXM6IEFycmF5MUQsIGluZGljZXM6IEFycmF5MUR9IHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgayA8PSBuZGFycmF5LnNpemUsXG4gICAgICAgIGBFcnJvciBpbiB0b3BLOiBrIHZhbHVlICgke2t9KSBtdXN0IGJlIGxlc3MgdGhhbiBzaXplIG9mIGlucHV0IGAgK1xuICAgICAgICAgICAgYG5kYXJyYXksIGdvdCBzaGFwZSAke25kYXJyYXkuc2hhcGV9LmApO1xuICAgIGNvbnN0IHJlc3VsdCA9IHRoaXMudG9wS0ludGVybmFsKG5kYXJyYXksIGspO1xuICAgIHRoaXMudHJhY2socmVzdWx0LnZhbHVlcyk7XG4gICAgdGhpcy50cmFjayhyZXN1bHQuaW5kaWNlcyk7XG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgdG9wS0ludGVybmFsKG5kYXJyYXk6IE5EQXJyYXksIGs6IG51bWJlcik6XG4gICAgICB7dmFsdWVzOiBBcnJheTFELCBpbmRpY2VzOiBBcnJheTFEfTtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIG1pbmltdW0gdmFsdWUgZnJvbSB0aGUgaW5wdXQuXG4gICAqIEBwYXJhbSBuZGFycmF5IFRoZSBpbnB1dCBOREFycmF5LlxuICAgKi9cbiAgbWluKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXIge1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMubWluSW50ZXJuYWwobmRhcnJheSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBtaW5JbnRlcm5hbChuZGFycmF5OiBOREFycmF5KTogU2NhbGFyO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgbWF4aW11bSB2YWx1ZSBmcm9tIHRoZSBpbnB1dC5cbiAgICogQHBhcmFtIG5kYXJyYXkgVGhlIGlucHV0IE5EQXJyYXkuXG4gICAqL1xuICBtYXgobmRhcnJheTogTkRBcnJheSk6IFNjYWxhciB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5tYXhJbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IG1heEludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXI7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSBzb2Z0bWF4IG5vcm1hbGl6ZWQgdmVjdG9yIGZyb20gdGhlIGlucHV0IHZlY3Rvci5cbiAgICogQHBhcmFtIHggVGhlIGlucHV0IHZlY3Rvci5cbiAgICovXG4gIHNvZnRtYXgoeDogQXJyYXkxRCk6IEFycmF5MUQge1xuICAgIHJldHVybiB0aGlzLnNjb3BlKCgpID0+IHtcbiAgICAgIC8vIERvIGl0IGluIGxvZyBzcGFjZSBmb3IgbnVtZXJpY2FsIHN0YWJpbGl0eS5cbiAgICAgIC8vIGV4cChYIC0gbG9nU3VtRXhwKFgpKVxuICAgICAgY29uc3QgbHNlID0gdGhpcy5sb2dTdW1FeHAoeCk7XG4gICAgICBjb25zdCBsb2dSZXN1bHQgPSB0aGlzLmFycmF5TWludXNTY2FsYXIoeCwgbHNlKTtcbiAgICAgIHJldHVybiB0aGlzLmV4cChsb2dSZXN1bHQpO1xuICAgIH0pO1xuICB9XG5cbiAgLy8vLy8vLy8vLy8vLy8vLy8vLy8vL1xuICAvLyBFbGVtZW50LXdpc2Ugb3BzIC8vXG4gIC8vLy8vLy8vLy8vLy8vLy8vLy8vLy9cblxuICAvKipcbiAgICogU3dpdGNoZXMgZGltZW5zaW9ucyBvZiB0aGUgaW5wdXQgTkRBcnJheS5cbiAgICogQHBhcmFtIGEgVGhlIGlucHV0IE5EQXJyYXkuXG4gICAqIEBwYXJhbSBuZXdEaW0gVGhlIG5ldyBpbmRpY2VzIHRoYXQgZGVmaW5lIHdoaWNoIHNoYXBlcyB2YWx1ZXMgdG8gc3dpdGNoLlxuICAgKi9cbiAgc3dpdGNoRGltPFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBuZXdEaW06IG51bWJlcltdKTogVCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGEucmFuayA9PT0gbmV3RGltLmxlbmd0aCxcbiAgICAgICAgYEVycm9yIGluIHN3aXRjaERpbTogbGVuZ3RoIG9mIGlucHV0IHNoYXBlICR7YS5zaGFwZX0gYCArXG4gICAgICAgICAgICBgbXVzdCBtYXRjaCBzaXplIG9mIG5ld0RpbSBhcnJheSAke25ld0RpbX0uYCk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5zd2l0Y2hEaW1JbnRlcm5hbChhLCBuZXdEaW0pKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3Qgc3dpdGNoRGltSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KFxuICAgICAgYTogVCwgbmV3RGltOiBudW1iZXJbXSk6IFQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIGEgc2NhbGFyIHBsdXMgTkRBcnJheSwgYyArIEEuXG4gICAqIEBwYXJhbSBjIFRoZSBzY2FsYXIgYyBpbiBjICsgQS5cbiAgICogQHBhcmFtIGEgVGhlIE5EQXJyYXkgQSBpbiBjICsgQS5cbiAgICovXG4gIHNjYWxhclBsdXNBcnJheTxUIGV4dGVuZHMgTkRBcnJheT4oYzogU2NhbGFyLCBhOiBUKTogVCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGMuc2l6ZSA9PT0gMSxcbiAgICAgICAgYEVycm9yIGluIHNjYWxhclBsdXNBcnJheTogZmlyc3QgYXJndW1lbnQgbXVzdCBiZSByYW5rIDAsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgICBgcmFuayAke2MucmFua30uYCk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5zY2FsYXJQbHVzQXJyYXlJbnRlcm5hbChjLCBhKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IHNjYWxhclBsdXNBcnJheUludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihcbiAgICAgIGM6IFNjYWxhciwgYTogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIGEgc2NhbGFyIG1pbnVzIE5EQXJyYXksIGMgLSBBLlxuICAgKiBAcGFyYW0gYyBUaGUgc2NhbGFyIGMgaW4gYyAtIEEuXG4gICAqIEBwYXJhbSBhIFRoZSBOREFycmF5IEEgaW4gYyAtIEEuXG4gICAqL1xuICBzY2FsYXJNaW51c0FycmF5PFQgZXh0ZW5kcyBOREFycmF5PihjOiBTY2FsYXIsIGE6IFQpOiBUIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgYy5zaXplID09PSAxLFxuICAgICAgICBgRXJyb3IgaW4gc2NhbGFyTWludXNBcnJheTogZmlyc3QgYXJndW1lbnQgbXVzdCBiZSByYW5rIDAsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgICBgcmFuayAke2MucmFua30uYCk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5zY2FsYXJNaW51c0FycmF5SW50ZXJuYWwoYywgYSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBzY2FsYXJNaW51c0FycmF5SW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KFxuICAgICAgYzogU2NhbGFyLCBhOiBUKTogVDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgYSBzY2FsYXIgbWludXMgTkRBcnJheSwgQSAtIGMuXG4gICAqIEBwYXJhbSBhIFRoZSBOREFycmF5IEEgaW4gQSAtIGMuXG4gICAqIEBwYXJhbSBjIFRoZSBzY2FsYXIgYyBpbiBBIC0gYy5cbiAgICovXG4gIGFycmF5TWludXNTY2FsYXI8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGM6IFNjYWxhcik6IFQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBjLnNpemUgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBhcnJheU1pbnVzU2NhbGFyOiBzZWNvbmQgYXJndW1lbnQgbXVzdCBiZSByYW5rIDAsIGJ1dCBgICtcbiAgICAgICAgICAgIGBnb3QgcmFuayAke2MucmFua30uYCk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5hcnJheU1pbnVzU2NhbGFySW50ZXJuYWwoYSwgYykpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBhcnJheU1pbnVzU2NhbGFySW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KFxuICAgICAgYTogVCwgYzogU2NhbGFyKTogVDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgLTEgKiBBIGVsZW1lbnQtd2lzZS5cbiAgICogQHBhcmFtIGEgVGhlIGlucHV0IGFycmF5LlxuICAgKi9cbiAgbmVnPFQgZXh0ZW5kcyBOREFycmF5PihhOiBUKTogVCB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5uZWdJbnRlcm5hbChhKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IG5lZ0ludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihhOiBUKTogVDtcblxuICAvKipcbiAgICogQWRkcyB0d28gTkRBcnJheXMgZWxlbWVudC13aXNlLCBBICsgQi4gSW5wdXRzIG11c3QgYmUgdGhlIHNhbWUgc2hhcGUuXG4gICAqIEBwYXJhbSBhIFRoZSBmaXJzdCBOREFycmF5IHRvIGFkZCBlbGVtZW50LXdpc2UuXG4gICAqIEBwYXJhbSBiIFRoZSBzZWNvbmQgTkRBcnJheSB0byBhZGQgZWxlbWVudC13aXNlLlxuICAgKi9cbiAgYWRkPFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBiOiBUKTogVCB7XG4gICAgdXRpbC5hc3NlcnRTaGFwZXNNYXRjaChhLnNoYXBlLCBiLnNoYXBlLCAnRXJyb3IgaW4gYWRkOiAnKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmFkZEludGVybmFsKGEsIGIpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgYWRkSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGI6IFQpOiBUO1xuXG4gIC8qKlxuICAgKiBTdWJ0cmFjdHMgdHdvIE5EQXJyYXlzIGVsZW1lbnQtd2lzZSwgQSAtIEIuIElucHV0cyBtdXN0IGJlIHRoZSBzYW1lIHNoYXBlLlxuICAgKiBAcGFyYW0gYSBUaGUgZmlyc3QgTkRBcnJheSB0byBzdWJ0cmFjdCBlbGVtZW50LXdpc2UuXG4gICAqIEBwYXJhbSBiIFRoZSBzZWNvbmQgTkRBcnJheSB0byBzdWJ0cmFjdCBlbGVtZW50LXdpc2UuXG4gICAqL1xuICBzdWI8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGI6IFQpOiBUIHtcbiAgICB1dGlsLmFzc2VydFNoYXBlc01hdGNoKGEuc2hhcGUsIGIuc2hhcGUsICdFcnJvciBpbiBzdWI6ICcpO1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMuc3ViSW50ZXJuYWwoYSwgYikpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBzdWJJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4oYTogVCwgYjogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIE11bHRpcGxpZXMgdHdvIE5EQXJyYXlzIGVsZW1lbnQtd2lzZSAoaGFkYW1hcmQgcHJvZHVjdCksIEEgKiBCLiBJbnB1dHMgbXVzdFxuICAgKiBiZSB0aGUgc2FtZSBzaGFwZS5cbiAgICogQHBhcmFtIGEgVGhlIGZpcnN0IE5EQXJyYXkgdG8gbXVsdGlwbHkgZWxlbWVudC13aXNlLlxuICAgKiBAcGFyYW0gYiBUaGUgc2Vjb25kIE5EQXJyYXkgdG8gbXVsdGlwbHkgZWxlbWVudC13aXNlLlxuICAgKi9cbiAgZWxlbWVudFdpc2VNdWw8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGI6IFQpOiBUIHtcbiAgICB1dGlsLmFzc2VydFNoYXBlc01hdGNoKGEuc2hhcGUsIGIuc2hhcGUsICdFcnJvciBpbiBlbGVtZW50V2lzZU11bDogJyk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5lbGVtZW50V2lzZU11bEludGVybmFsKGEsIGIpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgZWxlbWVudFdpc2VNdWxJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4oYTogVCwgYjogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIERpdmlkZXMgdHdvIE5EQXJyYXlzIGVsZW1lbnQtd2lzZSAoaGFkYW1hcmQgcHJvZHVjdCksIEEgLyBCLiBJbnB1dHMgbXVzdCBiZVxuICAgKiB0aGUgc2FtZSBzaGFwZS5cbiAgICogQHBhcmFtIGEgVGhlIGZpcnN0IE5EQXJyYXkgdG8gZGl2aWRlIGVsZW1lbnQtd2lzZS5cbiAgICogQHBhcmFtIGIgVGhlIHNlY29uZCBOREFycmF5IHRvIGRpdmlkZSBlbGVtZW50LXdpc2UuXG4gICAqL1xuICBkaXZpZGU8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGI6IFQpOiBUIHtcbiAgICB1dGlsLmFzc2VydFNoYXBlc01hdGNoKGEuc2hhcGUsIGIuc2hhcGUsICdFcnJvciBpbiBkaXZpZGU6ICcpO1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMuZGl2aWRlSW50ZXJuYWwoYSwgYikpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBkaXZpZGVJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4oYTogVCwgYjogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIGEgc2NhbGFyIGRpdmlkZWQgYnkgYW4gTkRBcnJheSwgYnJvYWRjYXN0ZWQgb3ZlciB0aGUgTkRBcnJheSwgYyAvXG4gICAqIEEuXG4gICAqIEBwYXJhbSBjIFRoZSBzY2FsYXIgdmFsdWUgaW4gYyAvIEEuXG4gICAqIEBwYXJhbSBhIFRoZSBOREFycmF5IHZhbHVlIGluIGMgLyBBLlxuICAgKi9cbiAgc2NhbGFyRGl2aWRlZEJ5QXJyYXk8VCBleHRlbmRzIE5EQXJyYXk+KGM6IFNjYWxhciwgYTogVCk6IFQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBjLnNpemUgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBzY2FsYXJEaXZpZGVkQnlBcnJheTogZmlyc3QgYXJndW1lbnQgbXVzdCBiZSByYW5rIDAsIGJ1dCBgICtcbiAgICAgICAgICAgIGBnb3QgTkRBcnJheSBvZiByYW5rICR7Yy5yYW5rfS5gKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLnNjYWxhckRpdmlkZWRCeUFycmF5SW50ZXJuYWwoYywgYSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBzY2FsYXJEaXZpZGVkQnlBcnJheUludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihcbiAgICAgIGM6IFNjYWxhciwgYTogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIGFuIE5EQXJyYXkgZGl2aWRlZCBieSBhIHNjYWxhciwgYnJvYWRjYXN0ZWQgb3ZlciB0aGUgTkRBcnJheSwgQSAvXG4gICAqIGMuXG4gICAqIEBwYXJhbSBhIFRoZSBOREFycmF5IHZhbHVlIGluIEEgLyBjLlxuICAgKiBAcGFyYW0gYyBUaGUgc2NhbGFyIHZhbHVlIGluIEEgLyBjLlxuICAgKi9cbiAgYXJyYXlEaXZpZGVkQnlTY2FsYXI8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGM6IFNjYWxhcik6IFQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBjLnNpemUgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBhcnJheURpdmlkZWRCeVNjYWxhcjogc2Vjb25kIGFyZ3VtZW50IG11c3QgYmUgcmFuayAwLCBgICtcbiAgICAgICAgICAgIGBidXQgZ290IE5EQXJyYXkgb2YgcmFuayAke2MucmFua30uYCk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5hcnJheURpdmlkZWRCeVNjYWxhckludGVybmFsKGEsIGMpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgYXJyYXlEaXZpZGVkQnlTY2FsYXJJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4oXG4gICAgICBhOiBULCBjOiBTY2FsYXIpOiBUO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyBleHBvbmVudGlhbCBvZiB0aGUgaW5wdXQgTkRBcnJheSBlbGVtZW50LXdpc2UuIHkgPSBlIF4geFxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheS5cbiAgICovXG4gIGV4cDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQge1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMuZXhwSW50ZXJuYWwobmRhcnJheSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBleHBJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIG5hdHVyYWwgbG9nYXJpdGhtIG9mIHRoZSBpbnB1dCBOREFycmF5IGVsZW1lbnQtd2lzZS4geSA9IGxuKHgpXG4gICAqIEBwYXJhbSBuZGFycmF5IFRoZSBpbnB1dCBOREFycmF5LlxuICAgKi9cbiAgbG9nPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5sb2dJbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGxvZ0ludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgcmVjdGlmaWVkIGxpbmVhciBlbGVtZW50LXdpc2UsIG1heCh4LCAwKS5cbiAgICogQHBhcmFtIG5kYXJyYXkgVGhlIGlucHV0IE5EQXJyYXkuXG4gICAqL1xuICByZWx1PFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5yZWx1SW50ZXJuYWwobmRhcnJheSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCByZWx1SW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyBzaWdtb2lkIGVsZW1lbnQtd2lzZSwgeSA9IDEgLyAoMSArIGV4cCgteCkpLlxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheS5cbiAgICovXG4gIHNpZ21vaWQ8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLnNpZ21vaWRJbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IHNpZ21vaWRJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIGh5cGVyYm9saWMgdGFuZ2VudCBvZiB0aGUgaW5wdXQgTkRBcnJheSBlbGVtZW50LXdpc2UuXG4gICAqIEBwYXJhbSBuZGFycmF5IFRoZSBpbnB1dCBOREFycmF5LlxuICAgKi9cbiAgdGFuaDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQge1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMudGFuaEludGVybmFsKG5kYXJyYXkpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgdGFuaEludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgc2luIG9mIHRoZSBpbnB1dCBOREFycmF5IGVsZW1lbnQtd2lzZSwgeSA9IHNpbih4KS5cbiAgICogQHBhcmFtIG5kYXJyYXkgVGhlIGlucHV0IE5EQXJyYXkuXG4gICAqL1xuICBzaW48VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLnNpbkludGVybmFsKG5kYXJyYXkpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3Qgc2luSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyBzdGVwIG9mIHRoZSBpbnB1dCBOREFycmF5IGVsZW1lbnQtd2lzZSwgeSA9IDEgaWYgeCA+IDAgfCAwIGlmIHggPD1cbiAgICogMFxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheS5cbiAgICovXG4gIHN0ZXA8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLnN0ZXBJbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IHN0ZXBJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIGEgc2NhbGVkIGFycmF5IGFkZCBvcGVyYXRpb24sIGMxICogQSArIGMyICogQi5cbiAgICogQHBhcmFtIGMxIFRoZSBmaXJzdCBzY2FsYXIgaW4gdGhlIHNjYWxlZCBhcnJheSBhZGQgY29tcHV0YXRpb24uXG4gICAqIEBwYXJhbSBhIFRoZSBmaXJzdCBOREFycmF5IGluIHRoZSBzY2FsZWQgYXJyYXkgYWRkIGNvbXB1dGF0aW9uLlxuICAgKiBAcGFyYW0gYzIgVGhlIHNlY29uZCBzY2FsYXIgaW4gdGhlIHNjYWxlZCBhcnJheSBhZGQgY29tcHV0YXRpb24uXG4gICAqIEBwYXJhbSBjYiBUaGUgc2Vjb25kIE5EQXJyYXkgaW4gdGhlIHNjYWxlZCBhcnJheSBhZGQgY29tcHV0YXRpb24uXG4gICAqL1xuICBzY2FsZWRBcnJheUFkZDxUIGV4dGVuZHMgTkRBcnJheT4oYzE6IFNjYWxhciwgYTogVCwgYzI6IFNjYWxhciwgYjogVCk6IFQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBjMS5zaXplID09PSAxLFxuICAgICAgICBgRXJyb3IgaW4gc2NhbGVkQXJyYXlBZGQ6IGZpcnN0IGFyZ3VtZW50IG11c3QgcmFuayAwLCBidXQgZ290IGAgK1xuICAgICAgICAgICAgYCByYW5rICR7YzEucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGMyLnNpemUgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBzY2FsZWRBcnJheUFkZDogdGhpcmQgYXJndW1lbnQgbXVzdCBiZSByYW5rIDAsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgICBgTkRBcnJheSBvZiByYW5rICR7YzIucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnRTaGFwZXNNYXRjaChhLnNoYXBlLCBiLnNoYXBlLCAnRXJyb3IgaW4gc2NhbGVkQXJyYXlBZGQ6ICcpO1xuXG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5zY2FsZWRBcnJheUFkZEludGVybmFsKGMxLCBhLCBjMiwgYikpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBzY2FsZWRBcnJheUFkZEludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihcbiAgICAgIGMxOiBTY2FsYXIsIGE6IFQsIGMyOiBTY2FsYXIsIGI6IFQpOiBUO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyBhIHNjYWxhciB0aW1lcyBhcnJheSBvcGVyYXRpb24gYnJvYWRjYXN0ZWQgb3ZlciB0aGUgTkRBcnJheSwgYyAqXG4gICAqIEEuXG4gICAqIEBwYXJhbSBjIFRoZSBzY2FsYXIgaW4gdGhlIG9wZXJhdGlvbi5cbiAgICogQHBhcmFtIEEgdGhlIE5EQXJyYXkgaW4gdGhlIG9wZXJhdGlvbiB0aGF0IHdpbGwgYmUgYnJvYWRjYXN0ZWQgb3Zlci5cbiAgICovXG4gIHNjYWxhclRpbWVzQXJyYXk8VCBleHRlbmRzIE5EQXJyYXk+KGM6IFNjYWxhciwgYTogVCk6IFQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBjLnNpemUgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBhcnJheURpdmlkZWRCeVNjYWxhcjogZmlyc3QgYXJndW1lbnQgbXVzdCBiZSByYW5rIDAsIGJ1dCBgICtcbiAgICAgICAgICAgIGBnb3QgcmFuayAke2MucmFua30uYCk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5zY2FsYXJUaW1lc0FycmF5SW50ZXJuYWwoYywgYSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBzY2FsYXJUaW1lc0FycmF5SW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KFxuICAgICAgYzogU2NhbGFyLCBhOiBUKTogVDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgYW4gZWxlbWVudC13aXNlIGJyb2FkY2FzdGVkIG11bHRpcGxpY2F0aW9uIG9mIHR3byBtYXRyaWNlcyBBIGFuZFxuICAgKiBCLiBXaWxsIHJldHVybiBhIG5ldyBtYXRyaXggdGhhdCBpcyB0aGUgbWF4IG9mIEEgYW5kIEIsIHdoZXJlIHRoZSBzbWFsbGVyXG4gICAqIG1hdHJpeCB3aWxsIGJyb2FkY2FzdCBvdmVyIHRoZSBsYXJnZXIgbWF0cml4LlxuICAgKiBAcGFyYW0gYyBUaGUgc2NhbGFyIGluIHRoZSBvcGVyYXRpb24uXG4gICAqIEBwYXJhbSBBIHRoZSBOREFycmF5IGluIHRoZSBvcGVyYXRpb24gdGhhdCB3aWxsIGJlIGJyb2FkY2FzdGVkIG92ZXIuXG4gICAqL1xuICBlbGVtZW50V2lzZU11bEJyb2FkY2FzdChhOiBBcnJheTJELCBiOiBBcnJheTJEKTogQXJyYXkyRCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGEucmFuayA9PT0gMixcbiAgICAgICAgYEVycm9yIGluIGVsZW1lbnRXaXNlTXVsQnJvYWRjYXN0OiBmaXJzdCBhcmd1bWVudCBtdXN0IGJlIGAgK1xuICAgICAgICAgICAgYHJhbmsgMiwgYnV0IGdvdCByYW5rICR7YS5yYW5rfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgYi5yYW5rID09PSAyLFxuICAgICAgICBgRXJyb3IgaW4gZWxlbWVudFdpc2VNdWxCcm9hZGNhc3Q6IHNlY29uZCBhcmd1bWVudCBtdXN0IGJlIGAgK1xuICAgICAgICAgICAgYHJhbmsgMiwgYnV0IGdvdCByYW5rICR7Yi5yYW5rfS5gKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmVsZW1lbnRXaXNlTXVsQnJvYWRjYXN0SW50ZXJuYWwoYSwgYikpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBlbGVtZW50V2lzZU11bEJyb2FkY2FzdEludGVybmFsKGE6IEFycmF5MkQsIGI6IEFycmF5MkQpOlxuICAgICAgQXJyYXkyRDtcblxuICAvLy8vLy8vLy8vLy8vLy8vLy8vLy9cbiAgLy8gQ29udm9sdXRpb24gb3BzIC8vXG4gIC8vLy8vLy8vLy8vLy8vLy8vLy8vL1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyBhIDJEIGNvbnZvbHV0aW9uIG92ZXIgdGhlIGlucHV0IHguXG4gICAqIEBwYXJhbSB4IFRoZSBpbnB1dCBpbWFnZSwgbXVzdCBiZSByYW5rIDMsIG9mIHNoYXBlIFtyb3dzLCBjb2xzLCBkZXB0aDFdLlxuICAgKiBAcGFyYW0gd2VpZ2h0cyBUaGUgd2VpZ2h0cyBOREFycmF5LCBtdXN0IGJlIHJhbmsgNCwgb2Ygc2hhcGUgW2YsIGYsIGRlcHRoMSxcbiAgICogZGVwdGgyXS5cbiAgICogQHBhcmFtIGJpYXNlcyBPcHRpb25hbCBiaWFzZXMgTkRBcnJheSwgbXVzdCBiZSByYW5rIDEgb2Ygc2hhcGUgW2RlcHRoMl0uXG4gICAqIEBwYXJhbSBzdHJpZGUgVGhlIHN0cmlkZSBvZiB0aGUgY29udm9sdXRpb24uXG4gICAqIEBwYXJhbSB6ZXJvUGFkIFRoZSB6ZXJvIHBhZGRpbmcgb2YgZWFjaCBzaWRlIG9mIHRoZSBpbnB1dCBOREFycmF5LiBXaWxsIHBhZFxuICAgKiBlcXVhbGx5IG9uIGFsbCBzaWRlcy5cbiAgICovXG4gIGNvbnYyZChcbiAgICAgIHg6IEFycmF5M0QsIHdlaWdodHM6IEFycmF5NEQsIGJpYXNlczogQXJyYXkxRHxudWxsLCBzdHJpZGU6IG51bWJlcixcbiAgICAgIHplcm9QYWQ6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB4LnJhbmsgPT09IDMsXG4gICAgICAgIGBFcnJvciBpbiBjb252MmQ6IHggbXVzdCBiZSByYW5rIDMsIGJ1dCBnb3QgcmFuayAke3gucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHdlaWdodHMucmFuayA9PT0gNCxcbiAgICAgICAgYEVycm9yIGluIGNvbnYyZDogd2VpZ2h0cyBtdXN0IGJlIHJhbmsgNCwgYnV0IGdvdCByYW5rIGAgK1xuICAgICAgICAgICAgYCR7d2VpZ2h0cy5yYW5rfS5gKTtcbiAgICBpZiAoYmlhc2VzICE9IG51bGwpIHtcbiAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgIGJpYXNlcy5yYW5rID09PSAxLFxuICAgICAgICAgIGBFcnJvciBpbiBjb252MmQ6IGJpYXNlcyBtdXN0IGJlIHJhbmsgMSwgYnV0IGdvdCByYW5rIGAgK1xuICAgICAgICAgICAgICBgJHtiaWFzZXMucmFua30uYCk7XG4gICAgfVxuXG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHguc2hhcGVbMl0gPT09IHdlaWdodHMuc2hhcGVbMl0sXG4gICAgICAgIGBFcnJvciBpbiBjb252MmQ6IGRlcHRoIG9mIGlucHV0ICgke3guc2hhcGVbMl19KSBtdXN0IG1hdGNoICBgICtcbiAgICAgICAgICAgIGBpbnB1dCBkZXB0aCBmb3Igd2VpZ2h0cyAke3dlaWdodHMuc2hhcGVbMl19LmApO1xuXG5cbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmNvbnYyZEludGVybmFsKHgsIHdlaWdodHMsIGJpYXNlcywgc3RyaWRlLCB6ZXJvUGFkKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGNvbnYyZEludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgd2VpZ2h0czogQXJyYXk0RCwgYmlhc2VzOiBBcnJheTFEfG51bGwsIHN0cmlkZTogbnVtYmVyLFxuICAgICAgemVyb1BhZDogbnVtYmVyKTogQXJyYXkzRDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIGJhY2twcm9wIG9mIGEgMkQgY29udm9sdXRpb24uXG4gICAqIEBwYXJhbSB4IFRoZSBpbnB1dCBpbWFnZSwgbXVzdCBiZSByYW5rIDMsIG9mIHNoYXBlIFt4cm93cywgeGNvbHMsIGRlcHRoMV0uXG4gICAqIEBwYXJhbSBkeSBUaGUgZHkgaW1hZ2UsIG11c3QgYmUgcmFuayAzLCBvZiBzaGFwZSBbeXJvd3MsIHljb2xzLCBkZXB0aDJdLlxuICAgKiBAcGFyYW0gd2VpZ2h0cyBUaGUgd2VpZ2h0cyBOREFycmF5LCBtdXN0IGJlIHJhbmsgNCwgb2Ygc2hhcGUgW2YsIGYsIGRlcHRoMSxcbiAgICogZGVwdGgyXS5cbiAgICogQHBhcmFtIHN0cmlkZSBUaGUgc3RyaWRlIG9mIHRoZSBvcmlnaW5hbCBjb252b2x1dGlvbi5cbiAgICogQHBhcmFtIHBhZCBUaGUgcGFkZGluZyBvZiB0aGUgb3JpZ2luYWwgY29udm9sdXRpb24uXG4gICAqL1xuICBjb252MmRCYWNrUHJvcChcbiAgICAgIHg6IEFycmF5M0QsIGR5OiBBcnJheTNELCB3ZWlnaHRzOiBBcnJheTRELCBzdHJpZGU6IG51bWJlcixcbiAgICAgIHBhZDogbnVtYmVyKToge2R4OiBBcnJheTNELCBkdzogQXJyYXk0RCwgZGI6IEFycmF5MUR9IHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgeC5yYW5rID09PSAzLFxuICAgICAgICBgRXJyb3IgaW4gY29udjJkQmFja1Byb3A6IHggbXVzdCBiZSByYW5rIDMsIGJ1dCBnb3Qgc2hhcGUgYCArXG4gICAgICAgICAgICBgJHt4LnNoYXBlfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgZHkucmFuayA9PT0gMyxcbiAgICAgICAgYEVycm9yIGluIGNvbnYyZEJhY2tQcm9wOiBkeSBtdXN0IGJlIHJhbmsgMywgYnV0IGdvdCBzaGFwZSBgICtcbiAgICAgICAgICAgIGAke2R5LnNoYXBlfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgd2VpZ2h0cy5yYW5rID09PSA0LFxuICAgICAgICBgRXJyb3IgaW4gY29udjJkQmFja1Byb3A6IHdlaWdodHMgbXVzdCBiZSByYW5rIDQsIGJ1dCBnb3Qgc2hhcGUgYCArXG4gICAgICAgICAgICBgJHt3ZWlnaHRzLnNoYXBlfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgeC5zaGFwZVsyXSA9PT0gd2VpZ2h0cy5zaGFwZVsyXSxcbiAgICAgICAgYEVycm9yIGluIGNvbnYyZEJhY2tQcm9wOiBkZXB0aCBvZiB4ICR7eC5zaGFwZVsyXX0pIG11c3QgYCArXG4gICAgICAgICAgICBgbWF0Y2ggaW5wdXQgZGVwdGggZm9yIHdlaWdodHMgKCR7d2VpZ2h0cy5zaGFwZVsyXX0uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGR5LnNoYXBlWzJdID09PSB3ZWlnaHRzLnNoYXBlWzNdLFxuICAgICAgICBgRXJyb3IgaW4gY29udjJkQmFja1Byb3A6IGRlcHRoIG9mIGR5ICgke2R5LnNoYXBlWzJdfSkgbXVzdCBgICtcbiAgICAgICAgICAgIGBtYXRjaCBvdXRwdXQgZGVwdGggZm9yIHdlaWdodHMgKCR7d2VpZ2h0cy5zaGFwZVszXX0pLmApO1xuXG4gICAgY29uc3QgYmFja3Byb3BSZXN1bHQgPVxuICAgICAgICB0aGlzLmNvbnYyZEJhY2tQcm9wSW50ZXJuYWwoeCwgZHksIHdlaWdodHMsIHN0cmlkZSwgcGFkKTtcblxuICAgIHRoaXMudHJhY2soYmFja3Byb3BSZXN1bHQuZGIpO1xuICAgIHRoaXMudHJhY2soYmFja3Byb3BSZXN1bHQuZHcpO1xuICAgIHRoaXMudHJhY2soYmFja3Byb3BSZXN1bHQuZHgpO1xuXG4gICAgcmV0dXJuIGJhY2twcm9wUmVzdWx0O1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBjb252MmRCYWNrUHJvcEludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgZHk6IEFycmF5M0QsIHdlaWdodHM6IEFycmF5NEQsIHN0cmlkZTogbnVtYmVyLFxuICAgICAgcGFkOiBudW1iZXIpOiB7ZHg6IEFycmF5M0QsIGR3OiBBcnJheTRELCBkYjogQXJyYXkxRH07XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSB0cmFuc3Bvc2VkIDJEIGNvbnZvbHV0aW9uIG9mIGFuIGltYWdlLCBhbHNvIGtub3duIGFzIGFcbiAgICogZGVjb252b2x1dGlvbi5cbiAgICogQHBhcmFtIHggVGhlIGlucHV0IGltYWdlLCBtdXN0IGJlIHJhbmsgMywgb2Ygc2hhcGUgW3hyb3dzLCB4Y29scywgZGVwdGgxXS5cbiAgICogQHBhcmFtIHdlaWdodHMgVGhlIHdlaWdodHMgTkRBcnJheSwgbXVzdCBiZSByYW5rIDQsIG9mIHNoYXBlIFtmLCBmLCBkZXB0aDEsXG4gICAqIGRlcHRoMl0uXG4gICAqIEBwYXJhbSBiaWFzZXMgT3B0aW9uYWwgYmlhc2VzIE5EQXJyYXksIG11c3QgYmUgcmFuayAxIG9mIHNoYXBlIFtkZXB0aDJdLlxuICAgKiBAcGFyYW0gc3RyaWRlIFRoZSBzdHJpZGUgb2YgdGhlIGNvbnZvbHV0aW9uLlxuICAgKiBAcGFyYW0gcGFkIFRoZSBwYWRkaW5nIG9mIGVhY2ggc2lkZSBvZiB0aGUgaW5wdXQgTkRBcnJheS4gV2lsbCBwYWQgZXF1YWxseVxuICAgKiBvbiBhbGwgc2lkZXMuXG4gICAqL1xuICBjb252MmRUcmFuc3Bvc2UoXG4gICAgICB4OiBBcnJheTNELCB3ZWlnaHRzOiBBcnJheTRELCBiaWFzZXM6IEFycmF5MUR8bnVsbCwgc3RyaWRlOiBudW1iZXIsXG4gICAgICBwYWQ6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB4LnJhbmsgPT09IDMsXG4gICAgICAgIGBFcnJvciBpbiBjb252MmRUcmFuc3Bvc2U6IHggbXVzdCBiZSByYW5rIDMsIGJ1dCBnb3QgcmFuayBgICtcbiAgICAgICAgICAgIGAke3gucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHdlaWdodHMucmFuayA9PT0gNCxcbiAgICAgICAgYEVycm9yIGluIGNvbnYyZFRyYW5zcG9zZTogd2VpZ2h0cyBtdXN0IGJlIHJhbmsgNCwgYnV0IGdvdCBgICtcbiAgICAgICAgICAgIGByYW5rICR7d2VpZ2h0cy5yYW5rfWApO1xuICAgIGlmIChiaWFzZXMgIT0gbnVsbCkge1xuICAgICAgdXRpbC5hc3NlcnQoXG4gICAgICAgICAgYmlhc2VzLnJhbmsgPT09IDEsXG4gICAgICAgICAgYEVycm9yIGluIGNvbnYyZFRyYW5zcG9zZTogYmlhc2VzIG11c3QgYmUgcmFuayAxLCBidXQgZ290ICcgK1xuICAgICAgICAgICAgICAncmFuayAke2JpYXNlcy5yYW5rfS5gKTtcbiAgICB9XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHguc2hhcGVbMl0gPT09IHdlaWdodHMuc2hhcGVbM10sXG4gICAgICAgIGBFcnJvciBpbiBjb252MmRUcmFuc3Bvc2U6IGRlcHRoIG9mIGlucHV0ICgke3guc2hhcGVbMl19KSBtdXN0IGAgK1xuICAgICAgICAgICAgYG1hdGNoIGlucHV0IGRlcHRoIGZvciB3ZWlnaHRzICR7d2VpZ2h0cy5zaGFwZVszXX0uYCk7XG5cbiAgICByZXR1cm4gdGhpcy50cmFjayhcbiAgICAgICAgdGhpcy5jb252MmRUcmFuc3Bvc2VJbnRlcm5hbCh4LCB3ZWlnaHRzLCBiaWFzZXMsIHN0cmlkZSwgcGFkKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGNvbnYyZFRyYW5zcG9zZUludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgd2VpZ2h0czogQXJyYXk0RCwgYmlhc2VzOiBBcnJheTFEfG51bGwsIHN0cmlkZTogbnVtYmVyLFxuICAgICAgcGFkOiBudW1iZXIpOiBBcnJheTNEO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgMkQgbWF4IHBvb2xpbmcgb2YgYW4gaW1hZ2UuXG4gICAqIEBwYXJhbSB4IFRoZSBpbnB1dCBpbWFnZSwgbXVzdCBiZSByYW5rIDMuXG4gICAqIEBwYXJhbSBmU2l6ZSBUaGUgZmllbGQgc2l6ZSBvZiB0aGUgbWF4IHBvb2wuXG4gICAqIEBwYXJhbSBzdHJpZGUgVGhlIHN0cmlkZSBvZiB0aGUgbWF4IHBvb2wuXG4gICAqIEBwYXJhbSBwYWQgVGhlIHBhZGRpbmcgb2YgZWFjaCBzaWRlIG9mIHRoZSBpbnB1dCBOREFycmF5LiBXaWxsIHBhZCBlcXVhbGx5XG4gICAqIG9uIGFsbCBzaWRlcy5cbiAgICovXG4gIG1heFBvb2woeDogQXJyYXkzRCwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsIHBhZDogbnVtYmVyKTogQXJyYXkzRCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHgucmFuayA9PT0gMyxcbiAgICAgICAgJ0Vycm9yIGluIG1heFBvb2w6IHggbXVzdCBiZSByYW5rIDMgYnV0IGdvdCByYW5rICcgKyB4LnJhbmsgKyAnLicpO1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMubWF4UG9vbEludGVybmFsKHgsIGZTaXplLCBzdHJpZGUsIHBhZCkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBtYXhQb29sSW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlciwgcGFkOiBudW1iZXIpOiBBcnJheTNEO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgYmFja3Byb3Agb2YgYSBtYXggcG9vbC5cbiAgICogQHBhcmFtIGR5IFRoZSBkeSBlcnJvci5cbiAgICogQHBhcmFtIHggVGhlIGlucHV0IGltYWdlLCBtdXN0IGJlIHJhbmsgMy5cbiAgICogQHBhcmFtIGZTaXplIFRoZSBmaWVsZCBzaXplIG9mIHRoZSBtYXggcG9vbC5cbiAgICogQHBhcmFtIHN0cmlkZSBUaGUgc3RyaWRlIG9mIHRoZSBtYXggcG9vbC5cbiAgICogQHBhcmFtIHBhZCBUaGUgcGFkZGluZyBvZiBlYWNoIHNpZGUgb2YgdGhlIGlucHV0IE5EQXJyYXkuIFdpbGwgcGFkIGVxdWFsbHlcbiAgICogb24gYWxsIHNpZGVzLlxuICAgKi9cbiAgbWF4UG9vbEJhY2twcm9wKFxuICAgICAgZHk6IEFycmF5M0QsIHg6IEFycmF5M0QsIGZTaXplOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLFxuICAgICAgcGFkOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgZHkucmFuayA9PT0gMyxcbiAgICAgICAgYEVycm9yIGluIG1heFBvb2xCYWNrcHJvcDogZHkgbXVzdCBiZSByYW5rIDMgYnV0IGdvdCByYW5rIGAgK1xuICAgICAgICAgICAgYCR7ZHkucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHgucmFuayA9PT0gMyxcbiAgICAgICAgYEVycm9yIGluIG1heFBvb2xCYWNrcHJvcDogeCBtdXN0IGJlIHJhbmsgMyBidXQgZ290IHJhbmsgYCArXG4gICAgICAgICAgICBgJHt4LnJhbmt9LmApO1xuXG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5tYXhQb29sQmFja3Byb3BJbnRlcm5hbChkeSwgeCwgZlNpemUsIHN0cmlkZSwgcGFkKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IG1heFBvb2xCYWNrcHJvcEludGVybmFsKFxuICAgICAgZHk6IEFycmF5M0QsIHg6IEFycmF5M0QsIGZTaXplOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLFxuICAgICAgcGFkOiBudW1iZXIpOiBBcnJheTNEO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgMkQgbWluIHBvb2xpbmcgb2YgYW4gaW1hZ2UuXG4gICAqIEBwYXJhbSB4IFRoZSBpbnB1dCBpbWFnZSwgbXVzdCBiZSByYW5rIDMuXG4gICAqIEBwYXJhbSBmU2l6ZSBUaGUgZmllbGQgc2l6ZSBvZiB0aGUgbWF4IHBvb2wuXG4gICAqIEBwYXJhbSBzdHJpZGUgVGhlIHN0cmlkZSBvZiB0aGUgbWF4IHBvb2wuXG4gICAqIEBwYXJhbSBwYWQgVGhlIHBhZGRpbmcgb2YgZWFjaCBzaWRlIG9mIHRoZSBpbnB1dCBOREFycmF5LiBXaWxsIHBhZCBlcXVhbGx5XG4gICAqIG9uIGFsbCBzaWRlcy5cbiAgICovXG4gIG1pblBvb2woeDogQXJyYXkzRCwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsIHBhZDogbnVtYmVyKTogQXJyYXkzRCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHgucmFuayA9PT0gMyxcbiAgICAgICAgYEVycm9yIGluIG1pblBvb2w6IHggbXVzdCBiZSByYW5rIDMgYnV0IGdvdCByYW5rICR7eC5yYW5rfS5gKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLm1pblBvb2xJbnRlcm5hbCh4LCBmU2l6ZSwgc3RyaWRlLCBwYWQpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgbWluUG9vbEludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsIHBhZDogbnVtYmVyKTogQXJyYXkzRDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIDJEIGF2ZXJhZ2UgcG9vbGluZyBvZiBhbiBpbWFnZS5cbiAgICogQHBhcmFtIHggVGhlIGlucHV0IGltYWdlLCBtdXN0IGJlIHJhbmsgMy5cbiAgICogQHBhcmFtIGZTaXplIFRoZSBmaWVsZCBzaXplIG9mIHRoZSBtYXggcG9vbC5cbiAgICogQHBhcmFtIHN0cmlkZSBUaGUgc3RyaWRlIG9mIHRoZSBtYXggcG9vbC5cbiAgICogQHBhcmFtIHBhZCBUaGUgcGFkZGluZyBvZiBlYWNoIHNpZGUgb2YgdGhlIGlucHV0IE5EQXJyYXkuIFdpbGwgcGFkIGVxdWFsbHlcbiAgICogb24gYWxsIHNpZGVzLlxuICAgKi9cbiAgYXZnUG9vbCh4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlciwgcGFkOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgeC5yYW5rID09PSAzLFxuICAgICAgICBgRXJyb3IgaW4gYXZnUG9vbDogeCBtdXN0IGJlIHJhbmsgMyBidXQgZ290IHJhbmsgJHt4LnJhbmt9LmApO1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMuYXZnUG9vbEludGVybmFsKHgsIGZTaXplLCBzdHJpZGUsIHBhZCkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBhdmdQb29sSW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlciwgcGFkOiBudW1iZXIpOiBBcnJheTNEO1xuXG4gIC8qXG4gICAqIEJpbGluZWFyIHJlc2l6ZSBhIDNEIGFycmF5IHBlciBlYWNoIGNoYW5uZWwgdG8gYSBuZXcgMkQgc2hhcGUuXG4gICAqIEBwYXJhbSB4IFRoZSBpbnB1dCBBcnJheTNELlxuICAgKiBAcGFyYW0gbmV3U2hhcGUyRCBUaGUgbmV3IHNoYXBlIHRvIHJlc2l6ZSB0aGUgQXJyYXkzRCB0by4gRWFjaCBjaGFubmVsIGlzXG4gICAqIHJlc2l6ZWQgaW5kaXZpZHVhbGx5LlxuICAgKiBAcGFyYW0gYWxpZ25Db3JuZXJzIEFuIG9wdGlvbmFsIGJvb2wuIERlZmF1bHRzIHRvIEZhbHNlLiBJZiB0cnVlLCByZXNjYWxlXG4gICAqIGlucHV0IGJ5IChuZXdfaGVpZ2h0IC0gMSkgLyAoaGVpZ2h0IC0gMSksIHdoaWNoIGV4YWN0bHkgYWxpZ25zIHRoZSA0XG4gICAqIGNvcm5lcnMgb2YgaW1hZ2VzIGFuZCByZXNpemVkIGltYWdlcy4gSWYgZmFsc2UsIHJlc2NhbGUgYnkgbmV3X2hlaWdodCAvXG4gICAqIGhlaWdodC4gVHJlYXQgc2ltaWxhcmx5IHRoZSB3aWR0aCBkaW1lbnNpb24uXG4gICAqL1xuICByZXNpemVCaWxpbmVhcjNEKFxuICAgICAgeDogQXJyYXkzRCwgbmV3U2hhcGUyRDogW251bWJlciwgbnVtYmVyXSwgYWxpZ25Db3JuZXJzID0gZmFsc2UpOiBBcnJheTNEIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgeC5yYW5rID09PSAzLFxuICAgICAgICBgRXJyb3IgaW4gcmVzaXplQmlsaW5lYXIzRDogeCBtdXN0IGJlIHJhbmsgMyBidXQgZ290IHJhbmsgJHt4LnJhbmt9LmApO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBuZXdTaGFwZTJELmxlbmd0aCA9PT0gMixcbiAgICAgICAgYEVycm9yIGluIHJlc2l6ZUJpbGluZWFyM0Q6IG5ldyBzaGFwZSBtdXN0IDJELCBidXQgZ290IHNoYXBlIGAgK1xuICAgICAgICAgICAgYCR7bmV3U2hhcGUyRH0uYCk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2soXG4gICAgICAgIHRoaXMucmVzaXplQmlsaW5lYXIzREludGVybmFsKHgsIG5ld1NoYXBlMkQsIGFsaWduQ29ybmVycykpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCByZXNpemVCaWxpbmVhcjNESW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBuZXdTaGFwZTJEOiBbbnVtYmVyLCBudW1iZXJdLCBhbGlnbkNvcm5lcnM6IGJvb2xlYW4pOiBBcnJheTNEO1xuXG4gIC8qKlxuICAgKiBCYXRjaCBub3JtYWxpemF0aW9uIDNELiBNZWFuLCB2YXJpYW5jZSwgc2NhbGUsIGFuZCBvZmZzZXQgY2FuIGJlIG9mIHR3b1xuICAgKiBzaGFwZXM6IDEpIFRoZSBzYW1lIHNoYXBlIGFzIHRoZSBpbnB1dDogYW4gQXJyYXkzRC4gMikgSW4gdGhlIGNvbW1vbiBjYXNlLFxuICAgKiB0aGUgZGVwdGggZGltZW5zaW9uIGlzIHRoZSBsYXN0IGRpbWVuc2lvbiBvZiB4LCBzbyB0aGUgdmFsdWVzIHdvdWxkIGJlIGFuXG4gICAqIEFycmF5MUQgb2Ygc2hhcGUgW2RlcHRoXS5cbiAgICogQHBhcmFtIHggVGhlIGlucHV0IE5EQXJyYXkuXG4gICAqIEBwYXJhbSBtZWFuIEEgbWVhbiBOREFycmF5LlxuICAgKiBAcGFyYW0gdmFyaWFuY2UgQSB2YXJpYW5jZSBOREFycmF5LlxuICAgKiBAcGFyYW0gdmFyaWFuY2VFcHNpbG9uIEEgc21hbGwgZmxvYXQgbnVtYmVyIHRvIGF2b2lkIGRpdmlkaW5nIGJ5IDAuXG4gICAqIEBwYXJhbSBzY2FsZSBBIHNjYWxlIE5EQXJyYXkuXG4gICAqIEBwYXJhbSBvZmZzZXQgQW4gb2Zmc2V0IE5EQXJyYXkuXG4gICAqL1xuICBiYXRjaE5vcm1hbGl6YXRpb24zRChcbiAgICAgIHg6IEFycmF5M0QsIG1lYW46IEFycmF5M0R8QXJyYXkxRCwgdmFyaWFuY2U6IEFycmF5M0R8QXJyYXkxRCxcbiAgICAgIHZhcmlhbmNlRXBzaWxvbiA9IC4wMDEsIHNjYWxlPzogQXJyYXkzRHxBcnJheTFELFxuICAgICAgb2Zmc2V0PzogQXJyYXkzRHxBcnJheTFEKTogQXJyYXkzRCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHgucmFuayA9PT0gMyxcbiAgICAgICAgYEVycm9yIGluIGJhdGNoTm9ybWFsaXphdGlvbjNEOiB4IG11c3QgYmUgcmFuayAzIGJ1dCBnb3QgcmFuayBgICtcbiAgICAgICAgICAgIGAke3gucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIG1lYW4ucmFuayA9PT0gMyB8fCBtZWFuLnJhbmsgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBiYXRjaE5vcm1hbGl6YXRpb24zRDogbWVhbiBtdXN0IGJlIHJhbmsgMyBvciByYW5rIDEgYnV0IGAgK1xuICAgICAgICAgICAgYGdvdCByYW5rICR7bWVhbi5yYW5rfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgdmFyaWFuY2UucmFuayA9PT0gMyB8fCB2YXJpYW5jZS5yYW5rID09PSAxLFxuICAgICAgICBgRXJyb3IgaW4gYmF0Y2hOb3JtYWxpemF0aW9uM0Q6IHZhcmlhbmNlIG11c3QgYmUgcmFuayAzIG9yIHJhbmsgMSBgICtcbiAgICAgICAgICAgIGBidXQgZ290IHJhbmsgJHt2YXJpYW5jZS5yYW5rfS5gKTtcbiAgICBpZiAoc2NhbGUgIT0gbnVsbCkge1xuICAgICAgdXRpbC5hc3NlcnQoXG4gICAgICAgICAgc2NhbGUucmFuayA9PT0gMyB8fCBzY2FsZS5yYW5rID09PSAxLFxuICAgICAgICAgIGBFcnJvciBpbiBiYXRjaE5vcm1hbGl6YXRpb24zRDogc2NhbGUgbXVzdCBiZSByYW5rIDMgb3IgcmFuayAxIGAgK1xuICAgICAgICAgICAgICBgYnV0IGdvdCByYW5rICR7c2NhbGUhLnJhbmt9LmApO1xuICAgIH1cbiAgICBpZiAob2Zmc2V0ICE9IG51bGwpIHtcbiAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgIG9mZnNldC5yYW5rID09PSAzIHx8IG9mZnNldC5yYW5rID09PSAxLFxuICAgICAgICAgIGBFcnJvciBpbiBiYXRjaE5vcm1hbGl6YXRpb24zRDogb2Zmc2V0IG11c3QgYmUgcmFuayAzIG9yIHJhbmsgMSBgICtcbiAgICAgICAgICAgICAgYGJ1dCBnb3QgcmFuayAke29mZnNldCEucmFua30uYCk7XG4gICAgfVxuXG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5iYXRjaE5vcm1hbGl6YXRpb24zREludGVybmFsKFxuICAgICAgICB4LCBtZWFuLCB2YXJpYW5jZSwgdmFyaWFuY2VFcHNpbG9uLCBzY2FsZSwgb2Zmc2V0KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGJhdGNoTm9ybWFsaXphdGlvbjNESW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBtZWFuOiBBcnJheTNEfEFycmF5MUQsIHZhcmlhbmNlOiBBcnJheTNEfEFycmF5MUQsXG4gICAgICB2YXJpYW5jZUVwc2lsb246IG51bWJlciwgc2NhbGU/OiBBcnJheTNEfEFycmF5MUQsXG4gICAgICBvZmZzZXQ/OiBBcnJheTNEfEFycmF5MUQpOiBBcnJheTNEO1xufVxuXG5leHBvcnQgZW51bSBNYXRyaXhPcmllbnRhdGlvbiB7XG4gIFJFR1VMQVIsXG4gIFRSQU5TUE9TRURcbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgY29udl91dGlsIGZyb20gJy4uL21hdGgvY29udl91dGlsJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vdXRpbCc7XG5cbmltcG9ydCAqIGFzIGNvbmNhdDNkX3V0aWwgZnJvbSAnLi9jb25jYXQzZF91dGlsJztcbmltcG9ydCAqIGFzIGNvcHkyRF91dGlsIGZyb20gJy4vY29weTJkX3V0aWwnO1xuaW1wb3J0IHtNYXRyaXhPcmllbnRhdGlvbiwgTkRBcnJheU1hdGh9IGZyb20gJy4vbWF0aCc7XG5pbXBvcnQge0FycmF5MUQsIEFycmF5MkQsIEFycmF5M0QsIEFycmF5NEQsIE5EQXJyYXksIFNjYWxhcn0gZnJvbSAnLi9uZGFycmF5JztcblxuZXhwb3J0IGNsYXNzIE5EQXJyYXlNYXRoQ1BVIGV4dGVuZHMgTkRBcnJheU1hdGgge1xuICBjb25zdHJ1Y3RvcihzYWZlTW9kZSA9IGZhbHNlKSB7XG4gICAgc3VwZXIoc2FmZU1vZGUpO1xuICB9XG5cbiAgcHJvdGVjdGVkIGNsb25lSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KFxuICAgICAgICBuZGFycmF5LnNoYXBlLCB7dmFsdWVzOiBuZXcgRmxvYXQzMkFycmF5KG5kYXJyYXkuZ2V0VmFsdWVzKCkpfSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgcmVzaGFwZUludGVybmFsPFQxIGV4dGVuZHMgTkRBcnJheSwgVDIgZXh0ZW5kcyBOREFycmF5PihcbiAgICAgIG5kYXJyYXk6IFQxLCBuZXdTaGFwZTogbnVtYmVyW10pOiBUMiB7XG4gICAgcmV0dXJuIHRoaXMuY2xvbmVJbnRlcm5hbChuZGFycmF5KS5yZXNoYXBlPFQyPihuZXdTaGFwZSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgc2xpY2UyREludGVybmFsKFxuICAgICAgaW5wdXQ6IEFycmF5MkQsIGJlZ2luUm93Q29sOiBbbnVtYmVyLCBudW1iZXJdLFxuICAgICAgc2l6ZVJvd0NvbDogW251bWJlciwgbnVtYmVyXSk6IEFycmF5MkQge1xuICAgIGNvbnN0IHJlc3VsdCA9IEFycmF5MkQuemVyb3Moc2l6ZVJvd0NvbCk7XG4gICAgdGhpcy5jb3B5MkRJbnRlcm5hbChcbiAgICAgICAgaW5wdXQsIGJlZ2luUm93Q29sLCBzaXplUm93Q29sLCByZXN1bHQsIFswLCAwXSwgc2l6ZVJvd0NvbCk7XG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG4gIHByb3RlY3RlZCBjb3B5MkRJbnRlcm5hbChcbiAgICAgIHNvdXJjZTogQXJyYXkyRCwgc291cmNlQmVnaW5Sb3dDb2w6IFtudW1iZXIsIG51bWJlcl0sXG4gICAgICBzb3VyY2VTaXplUm93Q29sOiBbbnVtYmVyLCBudW1iZXJdLCBkZXN0OiBBcnJheTJELFxuICAgICAgZGVzdEJlZ2luUm93Q29sOiBbbnVtYmVyLCBudW1iZXJdLFxuICAgICAgZGVzdFNpemVSb3dDb2w6IFtudW1iZXIsIG51bWJlcl0pOiB2b2lkIHtcbiAgICBjb3B5MkRfdXRpbC52YWxpZGF0ZVNoYXBlcyhzb3VyY2VTaXplUm93Q29sLCBkZXN0U2l6ZVJvd0NvbCk7XG4gICAgY29uc3Qgc3JjVmFsdWVzID0gc291cmNlLmdldFZhbHVlcygpO1xuICAgIGNvbnN0IGRzdFZhbHVlcyA9IGRlc3QuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgbiA9IHNvdXJjZVNpemVSb3dDb2xbMF0gKiBzb3VyY2VTaXplUm93Q29sWzFdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbjsgKytpKSB7XG4gICAgICBjb25zdCBzcmNSb3cgPSBzb3VyY2VCZWdpblJvd0NvbFswXSArIE1hdGguZmxvb3IoaSAvIHNvdXJjZVNpemVSb3dDb2xbMV0pO1xuICAgICAgY29uc3Qgc3JjQ29sID0gc291cmNlQmVnaW5Sb3dDb2xbMV0gKyAoaSAlIHNvdXJjZVNpemVSb3dDb2xbMV0pO1xuICAgICAgY29uc3Qgc3JjT2ZmID0gc3JjUm93ICogc291cmNlLnNoYXBlWzFdICsgc3JjQ29sO1xuICAgICAgY29uc3QgZHN0Um93ID0gZGVzdEJlZ2luUm93Q29sWzBdICsgTWF0aC5mbG9vcihpIC8gZGVzdFNpemVSb3dDb2xbMV0pO1xuICAgICAgY29uc3QgZHN0Q29sID0gZGVzdEJlZ2luUm93Q29sWzFdICsgKGkgJSBkZXN0U2l6ZVJvd0NvbFsxXSk7XG4gICAgICBjb25zdCBkc3RPZmYgPSBkc3RSb3cgKiBkZXN0LnNoYXBlWzFdICsgZHN0Q29sO1xuICAgICAgZHN0VmFsdWVzW2RzdE9mZl0gPSBzcmNWYWx1ZXNbc3JjT2ZmXTtcbiAgICB9XG4gIH1cblxuICBwcm90ZWN0ZWQgY29uY2F0M0RJbnRlcm5hbCh4MTogQXJyYXkzRCwgeDI6IEFycmF5M0QsIGF4aXM6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIGNvbnN0IG91dHB1dFNoYXBlID1cbiAgICAgICAgY29uY2F0M2RfdXRpbC5jb21wdXRlQ29uY2F0M0RPdXRwdXRTaGFwZSh4MS5zaGFwZSwgeDIuc2hhcGUsIGF4aXMpO1xuXG4gICAgY29uc3QgdmFsdWVzID0gTkRBcnJheS56ZXJvczxBcnJheTNEPihvdXRwdXRTaGFwZSk7XG5cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IG91dHB1dFNoYXBlWzBdOyBpKyspIHtcbiAgICAgIGZvciAobGV0IGogPSAwOyBqIDwgb3V0cHV0U2hhcGVbMV07IGorKykge1xuICAgICAgICBmb3IgKGxldCBrID0gMDsgayA8IG91dHB1dFNoYXBlWzJdOyBrKyspIHtcbiAgICAgICAgICAvLyBTaGFkZXIgYmVnaW5zLlxuICAgICAgICAgIGNvbnN0IGluZGV4OiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbaSwgaiwga107XG4gICAgICAgICAgbGV0IHZhbHVlOiBudW1iZXI7XG4gICAgICAgICAgaWYgKGluZGV4W2F4aXNdIDwgeDEuc2hhcGVbYXhpc10pIHtcbiAgICAgICAgICAgIHZhbHVlID0geDEuZ2V0KGksIGosIGspO1xuICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICBpbmRleFtheGlzXSAtPSB4MS5zaGFwZVtheGlzXTtcbiAgICAgICAgICAgIGNvbnN0IFtpMiwgajIsIGsyXSA9IGluZGV4O1xuICAgICAgICAgICAgdmFsdWUgPSB4Mi5nZXQoaTIsIGoyLCBrMik7XG4gICAgICAgICAgfVxuXG4gICAgICAgICAgdmFsdWVzLnNldCh2YWx1ZSwgaSwgaiwgayk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4gdmFsdWVzO1xuICB9XG5cbiAgcHJvdGVjdGVkIHNjYWxhclBsdXNBcnJheUludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihjOiBTY2FsYXIsIGE6IFQpOiBUIHtcbiAgICBjb25zdCByZXN1bHRWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KGEuc2l6ZSk7XG4gICAgY29uc3QgYVZhbHVlcyA9IGEuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgY1ZhbCA9IGMuZ2V0KCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCByZXN1bHRWYWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIHJlc3VsdFZhbHVlc1tpXSA9IGNWYWwgKyBhVmFsdWVzW2ldO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KGEuc2hhcGUsIHt2YWx1ZXM6IHJlc3VsdFZhbHVlc30pO1xuICB9XG5cbiAgcHJvdGVjdGVkIHNjYWxlZEFycmF5QWRkSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KFxuICAgICAgYzE6IFNjYWxhciwgYTogVCwgYzI6IFNjYWxhciwgYjogVCkge1xuICAgIGNvbnN0IGNWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KGEuc2l6ZSk7XG4gICAgY29uc3QgYVZhbHVlcyA9IGEuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgYlZhbHVlcyA9IGIuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgYzFWYWwgPSBjMS5nZXQoKTtcbiAgICBjb25zdCBjMlZhbCA9IGMyLmdldCgpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgY1ZhbHVlcy5sZW5ndGg7ICsraSkge1xuICAgICAgY1ZhbHVlc1tpXSA9IGMxVmFsICogYVZhbHVlc1tpXSArIGMyVmFsICogYlZhbHVlc1tpXTtcbiAgICB9XG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZTxUPihhLnNoYXBlLCB7dmFsdWVzOiBjVmFsdWVzfSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgc2NhbGFyVGltZXNBcnJheUludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihjOiBTY2FsYXIsIGE6IFQpOiBUIHtcbiAgICBjb25zdCBuZXdWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KGEuc2l6ZSk7XG4gICAgY29uc3QgYVZhbHVlcyA9IGEuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgY1ZhbCA9IGMuZ2V0KCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBhVmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICBuZXdWYWx1ZXNbaV0gPSBjVmFsICogYVZhbHVlc1tpXTtcbiAgICB9XG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZTxUPihhLnNoYXBlLCB7dmFsdWVzOiBuZXdWYWx1ZXN9KTtcbiAgfVxuXG4gIHByb3RlY3RlZCBzY2FsYXJNaW51c0FycmF5SW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KGM6IFNjYWxhciwgYTogVCk6IFQge1xuICAgIGNvbnN0IG5lZ0EgPSB0aGlzLm5lZ0ludGVybmFsKGEpO1xuICAgIGNvbnN0IHJlc3VsdCA9IHRoaXMuc2NhbGFyUGx1c0FycmF5SW50ZXJuYWwoYywgbmVnQSk7XG5cbiAgICBuZWdBLmRpc3Bvc2UoKTtcblxuICAgIHJldHVybiByZXN1bHQ7XG4gIH1cblxuICBwcm90ZWN0ZWQgYXJyYXlNaW51c1NjYWxhckludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBjOiBTY2FsYXIpOiBUIHtcbiAgICBjb25zdCBuZWdDID0gdGhpcy5uZWdJbnRlcm5hbChjKTtcbiAgICBjb25zdCByZXN1bHQgPSB0aGlzLnNjYWxhclBsdXNBcnJheUludGVybmFsKG5lZ0MsIGEpO1xuXG4gICAgbmVnQy5kaXNwb3NlKCk7XG5cbiAgICByZXR1cm4gcmVzdWx0O1xuICB9XG5cbiAgcHJvdGVjdGVkIG5lZ0ludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihhOiBUKTogVCB7XG4gICAgcmV0dXJuIHRoaXMuc2NhbGFyVGltZXNBcnJheUludGVybmFsKFNjYWxhci5ORUdfT05FLCBhKTtcbiAgfVxuXG4gIHByb3RlY3RlZCBhZGRJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4oYTogVCwgYjogVCk6IFQge1xuICAgIHJldHVybiB0aGlzLnNjYWxlZEFycmF5QWRkSW50ZXJuYWw8VD4oU2NhbGFyLk9ORSwgYSwgU2NhbGFyLk9ORSwgYik7XG4gIH1cblxuICBwcm90ZWN0ZWQgc3ViSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGI6IFQpOiBUIHtcbiAgICByZXR1cm4gdGhpcy5zY2FsZWRBcnJheUFkZEludGVybmFsPFQ+KFNjYWxhci5PTkUsIGEsIFNjYWxhci5ORUdfT05FLCBiKTtcbiAgfVxuXG4gIHByb3RlY3RlZCBtYXRNdWxJbnRlcm5hbChcbiAgICAgIGE6IEFycmF5MkQsIGI6IEFycmF5MkQsIGFPcmllbnRhdGlvbiA9IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIsXG4gICAgICBiT3JpZW50YXRpb24gPSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKTogQXJyYXkyRCB7XG4gICAgY29uc3Qgc2hhcmVkRGltID1cbiAgICAgICAgKGFPcmllbnRhdGlvbiA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUikgPyBhLnNoYXBlWzFdIDogYS5zaGFwZVswXTtcblxuICAgIGNvbnN0IGxlZnREaW0gPVxuICAgICAgICAoYU9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/IGEuc2hhcGVbMF0gOiBhLnNoYXBlWzFdO1xuICAgIGNvbnN0IHJpZ2h0RGltID1cbiAgICAgICAgKGJPcmllbnRhdGlvbiA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUikgPyBiLnNoYXBlWzFdIDogYi5zaGFwZVswXTtcblxuICAgIGNvbnN0IG5vcm1hbEdldHRlciA9IChtYXRyaXg6IEFycmF5MkQsIGk6IG51bWJlciwgajogbnVtYmVyKSA9PlxuICAgICAgICBtYXRyaXguZ2V0KGksIGopO1xuICAgIGNvbnN0IHRyYW5zcG9zZWRHZXR0ZXIgPSAobWF0cml4OiBBcnJheTJELCBpOiBudW1iZXIsIGo6IG51bWJlcikgPT5cbiAgICAgICAgbWF0cml4LmdldChqLCBpKTtcblxuICAgIGNvbnN0IGFHZXR0ZXIgPSAoYU9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/XG4gICAgICAgIG5vcm1hbEdldHRlciA6XG4gICAgICAgIHRyYW5zcG9zZWRHZXR0ZXI7XG4gICAgY29uc3QgYkdldHRlciA9IChiT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID9cbiAgICAgICAgbm9ybWFsR2V0dGVyIDpcbiAgICAgICAgdHJhbnNwb3NlZEdldHRlcjtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KGxlZnREaW0gKiByaWdodERpbSk7XG4gICAgbGV0IGluZGV4ID0gMDtcblxuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbGVmdERpbTsgKytpKSB7XG4gICAgICBmb3IgKGxldCBqID0gMDsgaiA8IHJpZ2h0RGltOyArK2opIHtcbiAgICAgICAgbGV0IHN1bSA9IDA7XG4gICAgICAgIGZvciAobGV0IGsgPSAwOyBrIDwgc2hhcmVkRGltOyArK2spIHtcbiAgICAgICAgICAvLyBUT0RPOiBvcHRpbWl6ZSBDUFUgbWF0bXVsLlxuICAgICAgICAgIHN1bSArPSBhR2V0dGVyKGEsIGksIGspICogYkdldHRlcihiLCBrLCBqKTtcbiAgICAgICAgfVxuICAgICAgICB2YWx1ZXNbaW5kZXgrK10gPSBzdW07XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBBcnJheTJELm5ldyhbbGVmdERpbSwgcmlnaHREaW1dLCB2YWx1ZXMpO1xuICB9XG5cbiAgcHJvdGVjdGVkIGVsZW1lbnRXaXNlTXVsSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGI6IFQpOiBUIHtcbiAgICBjb25zdCBuZXdWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KGEuc2l6ZSk7XG4gICAgY29uc3QgYVZhbHVlcyA9IGEuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgYlZhbHVlcyA9IGIuZ2V0VmFsdWVzKCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBhVmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICBuZXdWYWx1ZXNbaV0gPSBhVmFsdWVzW2ldICogYlZhbHVlc1tpXTtcbiAgICB9XG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZTxUPihhLnNoYXBlLCB7dmFsdWVzOiBuZXdWYWx1ZXN9KTtcbiAgfVxuXG4gIHByb3RlY3RlZCBlbGVtZW50V2lzZU11bEJyb2FkY2FzdEludGVybmFsKGE6IEFycmF5MkQsIGI6IEFycmF5MkQpOiBBcnJheTJEIHtcbiAgICBjb25zdCBtYXhSb3cgPSBNYXRoLm1heChhLnNoYXBlWzBdLCBiLnNoYXBlWzBdKTtcbiAgICBjb25zdCBtYXhDb2wgPSBNYXRoLm1heChhLnNoYXBlWzFdLCBiLnNoYXBlWzFdKTtcblxuICAgIGNvbnN0IHZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkobWF4Um93ICogbWF4Q29sKTtcbiAgICBsZXQgaW5kZXggPSAwO1xuICAgIGZvciAobGV0IHJvdyA9IDA7IHJvdyA8IG1heFJvdzsgcm93KyspIHtcbiAgICAgIGZvciAobGV0IGNvbCA9IDA7IGNvbCA8IG1heENvbDsgY29sKyspIHtcbiAgICAgICAgdmFsdWVzW2luZGV4KytdID0gYS5nZXQocm93ICUgYS5zaGFwZVswXSwgY29sICUgYS5zaGFwZVsxXSkgKlxuICAgICAgICAgICAgYi5nZXQocm93ICUgYi5zaGFwZVswXSwgY29sICUgYi5zaGFwZVsxXSk7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBBcnJheTJELm5ldyhbbWF4Um93LCBtYXhDb2xdLCB2YWx1ZXMpO1xuICB9XG5cbiAgcHJvdGVjdGVkIGRpdmlkZUludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBiOiBUKTogVCB7XG4gICAgY29uc3QgbmV3VmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShhLnNpemUpO1xuICAgIGNvbnN0IGFWYWx1ZXMgPSBhLmdldFZhbHVlcygpO1xuICAgIGNvbnN0IGJWYWx1ZXMgPSBiLmdldFZhbHVlcygpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgYVZhbHVlcy5sZW5ndGg7ICsraSkge1xuICAgICAgbmV3VmFsdWVzW2ldID0gYVZhbHVlc1tpXSAvIGJWYWx1ZXNbaV07XG4gICAgfVxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8VD4oYS5zaGFwZSwge3ZhbHVlczogbmV3VmFsdWVzfSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgc2NhbGFyRGl2aWRlZEJ5QXJyYXlJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4oYzogU2NhbGFyLCBhOiBUKTpcbiAgICAgIFQge1xuICAgIGNvbnN0IG5ld1ZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkoYS5zaXplKTtcbiAgICBjb25zdCBhVmFsdWVzID0gYS5nZXRWYWx1ZXMoKTtcbiAgICBjb25zdCBjVmFsdWUgPSBjLmdldCgpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgYVZhbHVlcy5sZW5ndGg7ICsraSkge1xuICAgICAgbmV3VmFsdWVzW2ldID0gY1ZhbHVlIC8gYVZhbHVlc1tpXTtcbiAgICB9XG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZTxUPihhLnNoYXBlLCB7dmFsdWVzOiBuZXdWYWx1ZXN9KTtcbiAgfVxuXG4gIHByb3RlY3RlZCBhcnJheURpdmlkZWRCeVNjYWxhckludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBjOiBTY2FsYXIpOlxuICAgICAgVCB7XG4gICAgY29uc3QgbmV3VmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShhLnNpemUpO1xuICAgIGNvbnN0IGFWYWx1ZXMgPSBhLmdldFZhbHVlcygpO1xuICAgIGNvbnN0IGNWYWx1ZSA9IGMuZ2V0KCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBhVmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICBuZXdWYWx1ZXNbaV0gPSBhVmFsdWVzW2ldIC8gY1ZhbHVlO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KGEuc2hhcGUsIHt2YWx1ZXM6IG5ld1ZhbHVlc30pO1xuICB9XG5cbiAgcHJvdGVjdGVkIHN1bUludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXIge1xuICAgIGxldCBzdW0gPSAwO1xuICAgIGNvbnN0IHZhbHVlcyA9IG5kYXJyYXkuZ2V0VmFsdWVzKCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIHN1bSArPSB2YWx1ZXNbaV07XG4gICAgfVxuICAgIHJldHVybiBTY2FsYXIubmV3KHN1bSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgYXJnTWluSW50ZXJuYWwobmRhcnJheTogTkRBcnJheSk6IFNjYWxhciB7XG4gICAgbGV0IG1pbiA9IE51bWJlci5NQVhfVkFMVUU7XG4gICAgbGV0IG1pbkluZGV4ID0gLTE7XG4gICAgY29uc3QgdmFsdWVzID0gbmRhcnJheS5nZXRWYWx1ZXMoKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHZhbHVlcy5sZW5ndGg7ICsraSkge1xuICAgICAgY29uc3QgdmFsdWUgPSB2YWx1ZXNbaV07XG4gICAgICBpZiAoaXNOYU4odmFsdWUpKSB7XG4gICAgICAgIHJldHVybiBTY2FsYXIubmV3KE5hTik7XG4gICAgICB9XG4gICAgICBpZiAodmFsdWUgPCBtaW4pIHtcbiAgICAgICAgbWluID0gdmFsdWU7XG4gICAgICAgIG1pbkluZGV4ID0gaTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIFNjYWxhci5uZXcobWluSW5kZXgpO1xuICB9XG5cbiAgcHJvdGVjdGVkIGFyZ01heEludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXIge1xuICAgIGxldCBtYXggPSBOdW1iZXIuTkVHQVRJVkVfSU5GSU5JVFk7XG4gICAgbGV0IG1heEluZGV4ID0gLTE7XG4gICAgY29uc3QgdmFsdWVzID0gbmRhcnJheS5nZXRWYWx1ZXMoKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHZhbHVlcy5sZW5ndGg7ICsraSkge1xuICAgICAgY29uc3QgdmFsdWUgPSB2YWx1ZXNbaV07XG4gICAgICBpZiAoaXNOYU4odmFsdWUpKSB7XG4gICAgICAgIHJldHVybiBTY2FsYXIubmV3KE5hTik7XG4gICAgICB9XG4gICAgICBpZiAodmFsdWUgPiBtYXgpIHtcbiAgICAgICAgbWF4ID0gdmFsdWU7XG4gICAgICAgIG1heEluZGV4ID0gaTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIFNjYWxhci5uZXcobWF4SW5kZXgpO1xuICB9XG5cbiAgcHJvdGVjdGVkIGFyZ01heEVxdWFsc0ludGVybmFsKHgxOiBOREFycmF5LCB4MjogTkRBcnJheSk6IFNjYWxhciB7XG4gICAgY29uc3QgYXJnTWF4MSA9IHRoaXMuYXJnTWF4SW50ZXJuYWwoeDEpLmdldCgpO1xuICAgIGNvbnN0IGFyZ01heDIgPSB0aGlzLmFyZ01heEludGVybmFsKHgyKS5nZXQoKTtcbiAgICBpZiAoaXNOYU4oYXJnTWF4MSkgfHwgaXNOYU4oYXJnTWF4MikpIHtcbiAgICAgIHJldHVybiBTY2FsYXIubmV3KE5hTik7XG4gICAgfVxuICAgIHJldHVybiBTY2FsYXIubmV3KCsoYXJnTWF4MSA9PT0gYXJnTWF4MikpO1xuICB9XG5cbiAgcHJvdGVjdGVkIHRvcEtJbnRlcm5hbChuZGFycmF5OiBOREFycmF5LCBrOiBudW1iZXIpOlxuICAgICAge3ZhbHVlczogQXJyYXkxRCwgaW5kaWNlczogQXJyYXkxRH0ge1xuICAgIGNvbnN0IHZhbHVlcyA9IG5kYXJyYXkuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgdmFsdWVzQW5kSW5kaWNlczogQXJyYXk8e3ZhbHVlOiBudW1iZXIsIGluZGV4OiBudW1iZXJ9PiA9IFtdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdmFsdWVzLmxlbmd0aDsgaSsrKSB7XG4gICAgICB2YWx1ZXNBbmRJbmRpY2VzLnB1c2goe3ZhbHVlOiB2YWx1ZXNbaV0sIGluZGV4OiBpfSk7XG4gICAgfVxuICAgIHZhbHVlc0FuZEluZGljZXMuc29ydCgoYSwgYikgPT4ge1xuICAgICAgcmV0dXJuIGIudmFsdWUgLSBhLnZhbHVlO1xuICAgIH0pO1xuICAgIGNvbnN0IHRvcGtWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KGspO1xuICAgIGNvbnN0IHRvcGtJbmRpY2VzID0gbmV3IEZsb2F0MzJBcnJheShrKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGs7IGkrKykge1xuICAgICAgdG9wa1ZhbHVlc1tpXSA9IHZhbHVlc0FuZEluZGljZXNbaV0udmFsdWU7XG4gICAgICB0b3BrSW5kaWNlc1tpXSA9IHZhbHVlc0FuZEluZGljZXNbaV0uaW5kZXg7XG4gICAgfVxuICAgIHJldHVybiB7dmFsdWVzOiBBcnJheTFELm5ldyh0b3BrVmFsdWVzKSwgaW5kaWNlczogQXJyYXkxRC5uZXcodG9wa0luZGljZXMpfTtcbiAgfVxuXG4gIHByb3RlY3RlZCBtaW5JbnRlcm5hbChuZGFycmF5OiBOREFycmF5KTogU2NhbGFyIHtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGxldCBtaW4gPSB2YWx1ZXNbMF07XG4gICAgZm9yIChsZXQgaSA9IDE7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIGNvbnN0IHZhbHVlID0gdmFsdWVzW2ldO1xuICAgICAgaWYgKGlzTmFOKHZhbHVlKSkge1xuICAgICAgICByZXR1cm4gU2NhbGFyLm5ldyhOYU4pO1xuICAgICAgfVxuICAgICAgaWYgKHZhbHVlIDwgbWluKSB7XG4gICAgICAgIG1pbiA9IHZhbHVlO1xuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gU2NhbGFyLm5ldyhtaW4pO1xuICB9XG5cbiAgcHJvdGVjdGVkIG1heEludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXIge1xuICAgIGNvbnN0IHZhbHVlcyA9IG5kYXJyYXkuZ2V0VmFsdWVzKCk7XG4gICAgbGV0IG1heCA9IHZhbHVlc1swXTtcbiAgICBmb3IgKGxldCBpID0gMTsgaSA8IHZhbHVlcy5sZW5ndGg7ICsraSkge1xuICAgICAgY29uc3QgdmFsdWUgPSB2YWx1ZXNbaV07XG4gICAgICBpZiAoaXNOYU4odmFsdWUpKSB7XG4gICAgICAgIHJldHVybiBTY2FsYXIubmV3KE5hTik7XG4gICAgICB9XG4gICAgICBpZiAodmFsdWUgPiBtYXgpIHtcbiAgICAgICAgbWF4ID0gdmFsdWU7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBTY2FsYXIubmV3KG1heCk7XG4gIH1cblxuICBwcm90ZWN0ZWQgZXhwSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGNvbnN0IG5ld1ZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkodmFsdWVzLmxlbmd0aCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIG5ld1ZhbHVlc1tpXSA9IE1hdGguZXhwKHZhbHVlc1tpXSk7XG4gICAgfVxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8VD4obmRhcnJheS5zaGFwZSwge3ZhbHVlczogbmV3VmFsdWVzfSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgbG9nSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGNvbnN0IG5ld1ZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkodmFsdWVzLmxlbmd0aCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIGNvbnN0IHZhbHVlID0gdmFsdWVzW2ldO1xuICAgICAgbmV3VmFsdWVzW2ldID0gTWF0aC5sb2codmFsdWUpO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KG5kYXJyYXkuc2hhcGUsIHt2YWx1ZXM6IG5ld1ZhbHVlc30pO1xuICB9XG5cbiAgcHJvdGVjdGVkIGxvZ1N1bUV4cEludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXIge1xuICAgIGNvbnN0IHhNYXggPSB0aGlzLm1heChuZGFycmF5KTtcbiAgICBjb25zdCBhID0gdGhpcy5hcnJheU1pbnVzU2NhbGFyKG5kYXJyYXksIHhNYXgpO1xuICAgIGNvbnN0IGIgPSB0aGlzLmV4cChhKTtcbiAgICBjb25zdCBjID0gdGhpcy5zdW0oYik7XG4gICAgY29uc3QgZCA9IHRoaXMubG9nKGMpO1xuICAgIGNvbnN0IHJlc3VsdCA9IHRoaXMuYWRkKHhNYXgsIGQpO1xuXG4gICAgeE1heC5kaXNwb3NlKCk7XG4gICAgYS5kaXNwb3NlKCk7XG4gICAgYi5kaXNwb3NlKCk7XG4gICAgYy5kaXNwb3NlKCk7XG4gICAgZC5kaXNwb3NlKCk7XG5cbiAgICByZXR1cm4gcmVzdWx0O1xuICB9XG5cbiAgcHJvdGVjdGVkIHJlbHVJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQge1xuICAgIGNvbnN0IHJlc3VsdFZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkobmRhcnJheS5zaXplKTtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICByZXN1bHRWYWx1ZXNbaV0gPSBNYXRoLm1heCgwLCB2YWx1ZXNbaV0pO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KG5kYXJyYXkuc2hhcGUsIHt2YWx1ZXM6IHJlc3VsdFZhbHVlc30pO1xuICB9XG5cbiAgcHJvdGVjdGVkIHNpZ21vaWRJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQge1xuICAgIGNvbnN0IHJlc3VsdFZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkobmRhcnJheS5zaXplKTtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICByZXN1bHRWYWx1ZXNbaV0gPSAxIC8gKDEgKyBNYXRoLmV4cCgtdmFsdWVzW2ldKSk7XG4gICAgfVxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8VD4obmRhcnJheS5zaGFwZSwge3ZhbHVlczogcmVzdWx0VmFsdWVzfSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgdGFuaEludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgY29uc3QgcmVzdWx0VmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShuZGFycmF5LnNpemUpO1xuICAgIGNvbnN0IHZhbHVlcyA9IG5kYXJyYXkuZ2V0VmFsdWVzKCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIHJlc3VsdFZhbHVlc1tpXSA9IHV0aWwudGFuaCh2YWx1ZXNbaV0pO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KG5kYXJyYXkuc2hhcGUsIHt2YWx1ZXM6IHJlc3VsdFZhbHVlc30pO1xuICB9XG5cbiAgcHJvdGVjdGVkIHNpbkludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgY29uc3QgcmVzdWx0VmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShuZGFycmF5LnNpemUpO1xuICAgIGNvbnN0IHZhbHVlcyA9IG5kYXJyYXkuZ2V0VmFsdWVzKCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIHJlc3VsdFZhbHVlc1tpXSA9IE1hdGguc2luKHZhbHVlc1tpXSk7XG4gICAgfVxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8VD4obmRhcnJheS5zaGFwZSwge3ZhbHVlczogcmVzdWx0VmFsdWVzfSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgc3RlcEludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgY29uc3QgcmVzdWx0VmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShuZGFycmF5LnNpemUpO1xuICAgIGNvbnN0IHZhbHVlcyA9IG5kYXJyYXkuZ2V0VmFsdWVzKCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIGNvbnN0IHZhbHVlID0gdmFsdWVzW2ldO1xuICAgICAgcmVzdWx0VmFsdWVzW2ldID0gdmFsdWUgPiAwID8gMSA6ICh2YWx1ZSA8IDAgPyAwIDogdmFsdWUpO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KG5kYXJyYXkuc2hhcGUsIHt2YWx1ZXM6IHJlc3VsdFZhbHVlc30pO1xuICB9XG5cbiAgLyoqXG4gICAqIGltYWdlIGlzIG9mIHNoYXBlIFtyLCBjLCBkMV0uXG4gICAqIHdlaWdodHMgaXMgb2Ygc2hhcGUgW0YsIEYsIGQxLCBkMl0uXG4gICAqL1xuICBwcm90ZWN0ZWQgY29udjJkSW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCB3ZWlnaHRzOiBBcnJheTRELCBiaWFzZXM6IEFycmF5MUR8bnVsbCwgc3RyaWRlOiBudW1iZXIsXG4gICAgICBwYWQ6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIGNvbnN0IFt4Um93cywgeENvbHMsIGlucHV0RGVwdGhdID0geC5zaGFwZTtcbiAgICBjb25zdCBmaWVsZFNpemUgPSB3ZWlnaHRzLnNoYXBlWzBdO1xuICAgIGNvbnN0IG91dHB1dERlcHRoID0gd2VpZ2h0cy5zaGFwZVszXTtcbiAgICBjb25zdCBvdXRwdXRTaGFwZSA9IGNvbnZfdXRpbC5jb21wdXRlT3V0cHV0U2hhcGUzRChcbiAgICAgICAgW3hSb3dzLCB4Q29scywgaW5wdXREZXB0aF0sIGZpZWxkU2l6ZSwgb3V0cHV0RGVwdGgsIHN0cmlkZSwgcGFkKTtcbiAgICBjb25zdCB5ID0gQXJyYXkzRC56ZXJvcyhvdXRwdXRTaGFwZSk7XG4gICAgZm9yIChsZXQgZDIgPSAwOyBkMiA8IG91dHB1dERlcHRoOyArK2QyKSB7XG4gICAgICBmb3IgKGxldCB5UiA9IDA7IHlSIDwgeS5zaGFwZVswXTsgKyt5Uikge1xuICAgICAgICBjb25zdCB4UkNvcm5lciA9IHlSICogc3RyaWRlIC0gcGFkO1xuICAgICAgICBjb25zdCB4Uk1pbiA9IE1hdGgubWF4KDAsIHhSQ29ybmVyKTtcbiAgICAgICAgY29uc3QgeFJNYXggPSBNYXRoLm1pbih4Um93cywgZmllbGRTaXplICsgeFJDb3JuZXIpO1xuICAgICAgICBmb3IgKGxldCB5QyA9IDA7IHlDIDwgeS5zaGFwZVsxXTsgKyt5Qykge1xuICAgICAgICAgIGNvbnN0IHhDQ29ybmVyID0geUMgKiBzdHJpZGUgLSBwYWQ7XG4gICAgICAgICAgY29uc3QgeENNaW4gPSBNYXRoLm1heCgwLCB4Q0Nvcm5lcik7XG4gICAgICAgICAgY29uc3QgeENNYXggPSBNYXRoLm1pbih4Q29scywgZmllbGRTaXplICsgeENDb3JuZXIpO1xuICAgICAgICAgIGxldCBkb3RQcm9kID0gMDtcbiAgICAgICAgICBmb3IgKGxldCB4UiA9IHhSTWluOyB4UiA8IHhSTWF4OyArK3hSKSB7XG4gICAgICAgICAgICBjb25zdCB3UiA9IHhSIC0geFJDb3JuZXI7XG4gICAgICAgICAgICBmb3IgKGxldCB4QyA9IHhDTWluOyB4QyA8IHhDTWF4OyArK3hDKSB7XG4gICAgICAgICAgICAgIGNvbnN0IHdDID0geEMgLSB4Q0Nvcm5lcjtcbiAgICAgICAgICAgICAgZm9yIChsZXQgZDEgPSAwOyBkMSA8IGlucHV0RGVwdGg7ICsrZDEpIHtcbiAgICAgICAgICAgICAgICBjb25zdCBwaXhlbCA9IHguZ2V0KHhSLCB4QywgZDEpO1xuICAgICAgICAgICAgICAgIGNvbnN0IHdlaWdodCA9IHdlaWdodHMuZ2V0KHdSLCB3QywgZDEsIGQyKTtcbiAgICAgICAgICAgICAgICBkb3RQcm9kICs9IHBpeGVsICogd2VpZ2h0O1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICAgIGNvbnN0IGJpYXMgPSAoYmlhc2VzICE9IG51bGwpID8gYmlhc2VzLmdldChkMikgOiAwO1xuICAgICAgICAgIHkuc2V0KGRvdFByb2QgKyBiaWFzLCB5UiwgeUMsIGQyKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4geTtcbiAgfVxuXG4gIHByb3RlY3RlZCBjb252MmRCYWNrUHJvcEludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgZHk6IEFycmF5M0QsIHdlaWdodHM6IEFycmF5NEQsIHN0cmlkZTogbnVtYmVyLFxuICAgICAgcGFkOiBudW1iZXIpOiB7ZHg6IEFycmF5M0QsIGR3OiBBcnJheTRELCBkYjogQXJyYXkxRH0ge1xuICAgIGNvbnN0IGZTaXplID0gd2VpZ2h0cy5zaGFwZVswXTtcbiAgICBjb25zdCBkdyA9IHRoaXMuY29udjJkRGVyV2VpZ2h0cyh4LCBkeSwgZlNpemUsIHN0cmlkZSwgcGFkKTtcbiAgICBjb25zdCBkYiA9IHRoaXMuY29udjJkRGVyQmlhcyhkeSk7XG4gICAgY29uc3QgZHggPSB0aGlzLmNvbnYyZFRyYW5zcG9zZUludGVybmFsKGR5LCB3ZWlnaHRzLCBudWxsLCBzdHJpZGUsIHBhZCk7XG4gICAgcmV0dXJuIHtkeCwgZGIsIGR3fTtcbiAgfVxuXG4gIC8qKlxuICAgKiBpbWFnZSBpcyBvZiBzaGFwZSBbciwgYywgZDFdLlxuICAgKiB3ZWlnaHRzIGlzIG9mIHNoYXBlIFtGLCBGLCBkMSwgZDJdLlxuICAgKi9cbiAgcHJvdGVjdGVkIGNvbnYyZFRyYW5zcG9zZUludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgd2VpZ2h0czogQXJyYXk0RCwgYmlhc2VzOiBBcnJheTFEfG51bGwsIG9yaWdTdHJpZGU6IG51bWJlcixcbiAgICAgIG9yaWdQYWQ6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIGNvbnN0IGZTaXplID0gd2VpZ2h0cy5zaGFwZVswXTtcbiAgICBjb25zdCBwYWQgPSBmU2l6ZSAtIDEgLSBvcmlnUGFkO1xuICAgIGNvbnN0IG9yaWdJbnB1dERlcHRoID0gd2VpZ2h0cy5zaGFwZVsyXTtcbiAgICBjb25zdCBvcmlnT3V0cHV0RGVwdGggPSB3ZWlnaHRzLnNoYXBlWzNdO1xuICAgIGNvbnN0IFt4Um93cywgeENvbHMsIHhEZXB0aF0gPSB4LnNoYXBlO1xuXG4gICAgLy8gRGlsYXRlIHRoZSBpbnB1dC5cbiAgICBjb25zdCB4Um93c0RpbGF0ZWQgPSAoeFJvd3MgLSAxKSAqIG9yaWdTdHJpZGUgKyAxO1xuICAgIGNvbnN0IHhDb2xzRGlsYXRlZCA9ICh4Q29scyAtIDEpICogb3JpZ1N0cmlkZSArIDE7XG5cbiAgICBjb25zdCBvdXRwdXRTaGFwZSA9IGNvbnZfdXRpbC5jb21wdXRlT3V0cHV0U2hhcGUzRChcbiAgICAgICAgW3hSb3dzRGlsYXRlZCwgeENvbHNEaWxhdGVkLCBvcmlnT3V0cHV0RGVwdGhdLCBmU2l6ZSwgb3JpZ0lucHV0RGVwdGgsIDEsXG4gICAgICAgIHBhZCk7XG4gICAgY29uc3QgeSA9IEFycmF5M0QuemVyb3Mob3V0cHV0U2hhcGUpO1xuICAgIGZvciAobGV0IGQyID0gMDsgZDIgPCBvcmlnSW5wdXREZXB0aDsgKytkMikge1xuICAgICAgZm9yIChsZXQgeVIgPSAwOyB5UiA8IHkuc2hhcGVbMF07ICsreVIpIHtcbiAgICAgICAgY29uc3QgeFJDb3JuZXIgPSB5UiAtIHBhZDtcbiAgICAgICAgY29uc3QgeFJNaW4gPSBNYXRoLm1heCgwLCBNYXRoLmNlaWwoeFJDb3JuZXIgLyBvcmlnU3RyaWRlKSk7XG4gICAgICAgIGNvbnN0IHhSTWF4ID0gTWF0aC5taW4oeFJvd3MsIChmU2l6ZSArIHhSQ29ybmVyKSAvIG9yaWdTdHJpZGUpO1xuXG4gICAgICAgIGZvciAobGV0IHlDID0gMDsgeUMgPCB5LnNoYXBlWzFdOyArK3lDKSB7XG4gICAgICAgICAgY29uc3QgeENDb3JuZXIgPSB5QyAtIHBhZDtcbiAgICAgICAgICBjb25zdCB4Q01pbiA9IE1hdGgubWF4KDAsIE1hdGguY2VpbCh4Q0Nvcm5lciAvIG9yaWdTdHJpZGUpKTtcbiAgICAgICAgICBjb25zdCB4Q01heCA9IE1hdGgubWluKHhDb2xzLCAoZlNpemUgKyB4Q0Nvcm5lcikgLyBvcmlnU3RyaWRlKTtcblxuICAgICAgICAgIGxldCBkb3RQcm9kID0gMDtcbiAgICAgICAgICBmb3IgKGxldCB4UiA9IHhSTWluOyB4UiA8IHhSTWF4OyArK3hSKSB7XG4gICAgICAgICAgICBjb25zdCB3UiA9IHhSICogb3JpZ1N0cmlkZSAtIHhSQ29ybmVyO1xuXG4gICAgICAgICAgICBmb3IgKGxldCB4QyA9IHhDTWluOyB4QyA8IHhDTWF4OyArK3hDKSB7XG4gICAgICAgICAgICAgIGNvbnN0IHdDID0geEMgKiBvcmlnU3RyaWRlIC0geENDb3JuZXI7XG5cbiAgICAgICAgICAgICAgZm9yIChsZXQgZDEgPSAwOyBkMSA8IG9yaWdPdXRwdXREZXB0aDsgKytkMSkge1xuICAgICAgICAgICAgICAgIGNvbnN0IHBpeGVsID0geC5nZXQoeFIsIHhDLCBkMSk7XG4gICAgICAgICAgICAgICAgY29uc3Qgd2VpZ2h0ID1cbiAgICAgICAgICAgICAgICAgICAgd2VpZ2h0cy5nZXQoZlNpemUgLSAxIC0gd1IsIGZTaXplIC0gMSAtIHdDLCBkMiwgZDEpO1xuICAgICAgICAgICAgICAgIGRvdFByb2QgKz0gcGl4ZWwgKiB3ZWlnaHQ7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgICAgY29uc3QgYmlhcyA9IGJpYXNlcyAhPSBudWxsID8gYmlhc2VzLmdldChkMikgOiAwO1xuICAgICAgICAgIHkuc2V0KGRvdFByb2QgKyBiaWFzLCB5UiwgeUMsIGQyKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4geTtcbiAgfVxuXG4gIC8qKlxuICAgKiBpbWFnZSBpcyBvZiBzaGFwZSBbciwgYywgZDFdLlxuICAgKiB3ZWlnaHRzIGlzIG9mIHNoYXBlIFtGLCBGLCBkMSwgZDJdLlxuICAgKi9cbiAgcHJvdGVjdGVkIGNvbnYyZFRyYW5zcG9zZVNoYWRlckxpa2UoXG4gICAgICB4OiBBcnJheTNELCBvcmlnV2VpZ2h0czogQXJyYXk0RCwgb3JpZ1N0cmlkZTogbnVtYmVyLFxuICAgICAgb3JpZ1BhZDogbnVtYmVyKTogQXJyYXkzRCB7XG4gICAgY29uc3QgZlNpemUgPSBvcmlnV2VpZ2h0cy5zaGFwZVswXTtcbiAgICBjb25zdCBwYWQgPSBmU2l6ZSAtIDEgLSBvcmlnUGFkO1xuICAgIGNvbnN0IG9yaWdJbnB1dERlcHRoID0gb3JpZ1dlaWdodHMuc2hhcGVbMl07XG4gICAgY29uc3Qgb3JpZ091dHB1dERlcHRoID0gb3JpZ1dlaWdodHMuc2hhcGVbM107XG4gICAgY29uc3QgW3hSb3dzLCB4Q29scywgeERlcHRoXSA9IHguc2hhcGU7XG5cbiAgICAvLyBEaWxhdGUgdGhlIGlucHV0LlxuICAgIGNvbnN0IHhSb3dzRGlsYXRlZCA9ICh4Um93cyAtIDEpICogb3JpZ1N0cmlkZSArIDE7XG4gICAgY29uc3QgeENvbHNEaWxhdGVkID0gKHhDb2xzIC0gMSkgKiBvcmlnU3RyaWRlICsgMTtcblxuICAgIGNvbnN0IG91dHB1dFNoYXBlID0gY29udl91dGlsLmNvbXB1dGVPdXRwdXRTaGFwZTNEKFxuICAgICAgICBbeFJvd3NEaWxhdGVkLCB4Q29sc0RpbGF0ZWQsIG9yaWdPdXRwdXREZXB0aF0sIGZTaXplLCBvcmlnSW5wdXREZXB0aCwgMSxcbiAgICAgICAgcGFkKTtcbiAgICBjb25zdCB5ID0gQXJyYXkzRC56ZXJvcyhvdXRwdXRTaGFwZSk7XG5cbiAgICBmb3IgKGxldCBkMiA9IDA7IGQyIDwgb3JpZ0lucHV0RGVwdGg7ICsrZDIpIHtcbiAgICAgIGZvciAobGV0IHlSID0gMDsgeVIgPCB5LnNoYXBlWzBdOyArK3lSKSB7XG4gICAgICAgIGZvciAobGV0IHlDID0gMDsgeUMgPCB5LnNoYXBlWzFdOyArK3lDKSB7XG4gICAgICAgICAgLy8gU2hhZGVyIGNvZGUgYmVnaW5zLlxuICAgICAgICAgIGNvbnN0IHhSQ29ybmVyID0geVIgLSBwYWQ7XG4gICAgICAgICAgY29uc3QgeENDb3JuZXIgPSB5QyAtIHBhZDtcbiAgICAgICAgICBsZXQgZG90UHJvZCA9IDA7XG4gICAgICAgICAgZm9yIChsZXQgd1IgPSAwOyB3UiA8IGZTaXplOyArK3dSKSB7XG4gICAgICAgICAgICBjb25zdCB4UiA9ICh4UkNvcm5lciArIHdSKSAvIG9yaWdTdHJpZGU7XG4gICAgICAgICAgICBpZiAoeFIgPCAwIHx8IHhSID49IHhSb3dzIHx8IE1hdGguZmxvb3IoeFIpICE9PSB4Uikge1xuICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGZvciAobGV0IHdDID0gMDsgd0MgPCBmU2l6ZTsgKyt3Qykge1xuICAgICAgICAgICAgICBjb25zdCB4QyA9ICh4Q0Nvcm5lciArIHdDKSAvIG9yaWdTdHJpZGU7XG4gICAgICAgICAgICAgIGlmICh4QyA8IDAgfHwgeEMgPj0geENvbHMgfHwgTWF0aC5mbG9vcih4QykgIT09IHhDKSB7XG4gICAgICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgZm9yIChsZXQgZDEgPSAwOyBkMSA8IG9yaWdPdXRwdXREZXB0aDsgKytkMSkge1xuICAgICAgICAgICAgICAgIGNvbnN0IHBpeGVsID0geC5nZXQoeFIsIHhDLCBkMSk7XG4gICAgICAgICAgICAgICAgY29uc3Qgd2VpZ2h0ID1cbiAgICAgICAgICAgICAgICAgICAgb3JpZ1dlaWdodHMuZ2V0KGZTaXplIC0gMSAtIHdSLCBmU2l6ZSAtIDEgLSB3QywgZDIsIGQxKTtcbiAgICAgICAgICAgICAgICBkb3RQcm9kICs9IHBpeGVsICogd2VpZ2h0O1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICAgIHkuc2V0KGRvdFByb2QsIHlSLCB5QywgZDIpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiB5O1xuICB9XG5cbiAgY29udjJkRGVyV2VpZ2h0cyhcbiAgICAgIHg6IEFycmF5M0QsIGRZOiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlcixcbiAgICAgIHplcm9QYWQ6IG51bWJlcik6IEFycmF5NEQge1xuICAgIGNvbnN0IGlucHV0RGVwdGggPSB4LnNoYXBlWzJdO1xuICAgIGNvbnN0IG91dHB1dERlcHRoID0gZFkuc2hhcGVbMl07XG4gICAgY29uc3Qgd2VpZ2h0c1NoYXBlID1cbiAgICAgICAgY29udl91dGlsLmNvbXB1dGVXZWlnaHRzU2hhcGU0RChpbnB1dERlcHRoLCBvdXRwdXREZXB0aCwgZlNpemUpO1xuICAgIGNvbnN0IGRXID0gQXJyYXk0RC56ZXJvcyh3ZWlnaHRzU2hhcGUpO1xuXG4gICAgY29uc3QgeU51bVJvd3MgPSBkWS5zaGFwZVswXTtcbiAgICBjb25zdCB5TnVtQ29scyA9IGRZLnNoYXBlWzFdO1xuICAgIGNvbnN0IHhOdW1Sb3dzID0geC5zaGFwZVswXTtcbiAgICBjb25zdCB4TnVtQ29scyA9IHguc2hhcGVbMV07XG5cbiAgICBmb3IgKGxldCB3UiA9IDA7IHdSIDwgZlNpemU7ICsrd1IpIHtcbiAgICAgIGNvbnN0IHlSTWluID0gTWF0aC5tYXgoMCwgTWF0aC5jZWlsKCh6ZXJvUGFkIC0gd1IpIC8gc3RyaWRlKSk7XG4gICAgICBjb25zdCB5Uk1heCA9IE1hdGgubWluKHlOdW1Sb3dzLCAoeE51bVJvd3MgKyB6ZXJvUGFkIC0gd1IpIC8gc3RyaWRlKTtcblxuICAgICAgZm9yIChsZXQgd0MgPSAwOyB3QyA8IGZTaXplOyArK3dDKSB7XG4gICAgICAgIGNvbnN0IHlDTWluID0gTWF0aC5tYXgoMCwgTWF0aC5jZWlsKCh6ZXJvUGFkIC0gd0MpIC8gc3RyaWRlKSk7XG4gICAgICAgIGNvbnN0IHlDTWF4ID0gTWF0aC5taW4oeU51bUNvbHMsICh4TnVtQ29scyArIHplcm9QYWQgLSB3QykgLyBzdHJpZGUpO1xuXG4gICAgICAgIGZvciAobGV0IGQxID0gMDsgZDEgPCBpbnB1dERlcHRoOyArK2QxKSB7XG4gICAgICAgICAgZm9yIChsZXQgZDIgPSAwOyBkMiA8IG91dHB1dERlcHRoOyArK2QyKSB7XG4gICAgICAgICAgICAvLyBOZWVkIHRvIGNvbnZvbHZlLlxuICAgICAgICAgICAgbGV0IGRvdFByb2QgPSAwO1xuICAgICAgICAgICAgZm9yIChsZXQgeVIgPSB5Uk1pbjsgeVIgPCB5Uk1heDsgKyt5Uikge1xuICAgICAgICAgICAgICBjb25zdCB4UiA9IHdSICsgeVIgKiBzdHJpZGUgLSB6ZXJvUGFkO1xuICAgICAgICAgICAgICBmb3IgKGxldCB5QyA9IHlDTWluOyB5QyA8IHlDTWF4OyArK3lDKSB7XG4gICAgICAgICAgICAgICAgY29uc3QgeEMgPSB3QyArIHlDICogc3RyaWRlIC0gemVyb1BhZDtcbiAgICAgICAgICAgICAgICBkb3RQcm9kICs9IHguZ2V0KHhSLCB4QywgZDEpICogZFkuZ2V0KHlSLCB5QywgZDIpO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBkVy5zZXQoZG90UHJvZCwgd1IsIHdDLCBkMSwgZDIpO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gZFc7XG4gIH1cblxuICBjb252MmREZXJCaWFzKGRZOiBBcnJheTNEKTogQXJyYXkxRCB7XG4gICAgY29uc3Qgb3V0cHV0RGVwdGggPSBkWS5zaGFwZVsyXTtcbiAgICBjb25zdCBudW1Sb3dzID0gZFkuc2hhcGVbMF07XG4gICAgY29uc3QgbnVtQ29scyA9IGRZLnNoYXBlWzFdO1xuICAgIGNvbnN0IHZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkob3V0cHV0RGVwdGgpO1xuICAgIGZvciAobGV0IGQyID0gMDsgZDIgPCBvdXRwdXREZXB0aDsgKytkMikge1xuICAgICAgbGV0IHN1bSA9IDA7XG4gICAgICBmb3IgKGxldCByID0gMDsgciA8IG51bVJvd3M7ICsrcikge1xuICAgICAgICBmb3IgKGxldCBjID0gMDsgYyA8IG51bUNvbHM7ICsrYykge1xuICAgICAgICAgIHN1bSArPSBkWS5nZXQociwgYywgZDIpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICB2YWx1ZXNbZDJdID0gc3VtO1xuICAgIH1cbiAgICByZXR1cm4gQXJyYXkxRC5uZXcodmFsdWVzKTtcbiAgfVxuXG4gIHByb3RlY3RlZCBzd2l0Y2hEaW1JbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4odDogVCwgbmV3RGltOiBudW1iZXJbXSk6IFQge1xuICAgIGNvbnN0IG5ld1NoYXBlOiBudW1iZXJbXSA9IG5ldyBBcnJheSh0LnJhbmspO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbmV3U2hhcGUubGVuZ3RoOyBpKyspIHtcbiAgICAgIG5ld1NoYXBlW2ldID0gdC5zaGFwZVtuZXdEaW1baV1dO1xuICAgIH1cbiAgICBjb25zdCByZXN1bHRWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KHQuc2l6ZSk7XG4gICAgY29uc3QgdmFsdWVzID0gdC5nZXRWYWx1ZXMoKTtcbiAgICBjb25zdCByZXN1bHQgPSBOREFycmF5Lm1ha2U8VD4obmV3U2hhcGUsIHt2YWx1ZXM6IHJlc3VsdFZhbHVlc30pO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdC5zaXplOyArK2kpIHtcbiAgICAgIGNvbnN0IGxvYyA9IHQuaW5kZXhUb0xvYyhpKTtcblxuICAgICAgLy8gUGVybXV0ZSBsb2NhdGlvbi5cbiAgICAgIGNvbnN0IG5ld0xvYzogbnVtYmVyW10gPSBuZXcgQXJyYXkobG9jLmxlbmd0aCk7XG4gICAgICBmb3IgKGxldCBpID0gMDsgaSA8IG5ld0xvYy5sZW5ndGg7IGkrKykge1xuICAgICAgICBuZXdMb2NbaV0gPSBsb2NbbmV3RGltW2ldXTtcbiAgICAgIH1cblxuICAgICAgY29uc3QgbmV3SW5kZXggPSByZXN1bHQubG9jVG9JbmRleChuZXdMb2MpO1xuICAgICAgcmVzdWx0VmFsdWVzW25ld0luZGV4XSA9IHZhbHVlc1tpXTtcbiAgICB9XG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG4gIHByaXZhdGUgcG9vbChcbiAgICAgIHg6IEFycmF5M0QsIGZTaXplOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLCBwYWQ6IG51bWJlcixcbiAgICAgIHBvb2xUeXBlOiAnbWF4J3wnbWluJ3wnYXZnJykge1xuICAgIGNvbnN0IFt4Um93cywgeENvbHMsIGRlcHRoXSA9IHguc2hhcGU7XG4gICAgY29uc3Qgb3V0cHV0U2hhcGUgPSBjb252X3V0aWwuY29tcHV0ZU91dHB1dFNoYXBlM0QoXG4gICAgICAgIFt4Um93cywgeENvbHMsIGRlcHRoXSwgZlNpemUsIGRlcHRoLCBzdHJpZGUsIHBhZCk7XG4gICAgY29uc3QgeSA9IEFycmF5M0QuemVyb3Mob3V0cHV0U2hhcGUpO1xuICAgIGZvciAobGV0IGQgPSAwOyBkIDwgZGVwdGg7ICsrZCkge1xuICAgICAgZm9yIChsZXQgeVIgPSAwOyB5UiA8IHkuc2hhcGVbMF07ICsreVIpIHtcbiAgICAgICAgY29uc3QgeFJDb3JuZXIgPSB5UiAqIHN0cmlkZSAtIHBhZDtcbiAgICAgICAgY29uc3QgeFJNaW4gPSBNYXRoLm1heCgwLCB4UkNvcm5lcik7XG4gICAgICAgIGNvbnN0IHhSTWF4ID0gTWF0aC5taW4oeFJvd3MsIGZTaXplICsgeFJDb3JuZXIpO1xuICAgICAgICBmb3IgKGxldCB5QyA9IDA7IHlDIDwgeS5zaGFwZVsxXTsgKyt5Qykge1xuICAgICAgICAgIGNvbnN0IHhDQ29ybmVyID0geUMgKiBzdHJpZGUgLSBwYWQ7XG4gICAgICAgICAgY29uc3QgeENNaW4gPSBNYXRoLm1heCgwLCB4Q0Nvcm5lcik7XG4gICAgICAgICAgY29uc3QgeENNYXggPSBNYXRoLm1pbih4Q29scywgZlNpemUgKyB4Q0Nvcm5lcik7XG5cblxuICAgICAgICAgIGxldCBtaW5NYXhWYWx1ZSA9XG4gICAgICAgICAgICAgIChwb29sVHlwZSA9PT0gJ21heCcgPyBOdW1iZXIuTkVHQVRJVkVfSU5GSU5JVFkgOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgTnVtYmVyLlBPU0lUSVZFX0lORklOSVRZKTtcbiAgICAgICAgICBsZXQgYXZnVmFsdWUgPSAwO1xuXG4gICAgICAgICAgZm9yIChsZXQgeFIgPSB4Uk1pbjsgeFIgPCB4Uk1heDsgKyt4Uikge1xuICAgICAgICAgICAgY29uc3Qgd1IgPSB4UiAtIHhSQ29ybmVyO1xuICAgICAgICAgICAgZm9yIChsZXQgeEMgPSB4Q01pbjsgeEMgPCB4Q01heDsgKyt4Qykge1xuICAgICAgICAgICAgICBjb25zdCB3QyA9IHhDIC0geENDb3JuZXI7XG4gICAgICAgICAgICAgIGNvbnN0IHBpeGVsID0geC5nZXQoeFIsIHhDLCBkKTtcbiAgICAgICAgICAgICAgaWYgKGlzTmFOKHBpeGVsKSkge1xuICAgICAgICAgICAgICAgIG1pbk1heFZhbHVlID0gTmFOO1xuICAgICAgICAgICAgICAgIGF2Z1ZhbHVlID0gTmFOO1xuICAgICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIGlmICgocG9vbFR5cGUgPT09ICdtYXgnICYmIHBpeGVsID4gbWluTWF4VmFsdWUpIHx8XG4gICAgICAgICAgICAgICAgICAocG9vbFR5cGUgPT09ICdtaW4nICYmIHBpeGVsIDwgbWluTWF4VmFsdWUpKSB7XG4gICAgICAgICAgICAgICAgbWluTWF4VmFsdWUgPSBwaXhlbDtcbiAgICAgICAgICAgICAgfSBlbHNlIGlmIChwb29sVHlwZSA9PT0gJ2F2ZycpIHtcbiAgICAgICAgICAgICAgICBhdmdWYWx1ZSArPSBwaXhlbCAvIChmU2l6ZSAqIGZTaXplKTtcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgICAgaWYgKGlzTmFOKG1pbk1heFZhbHVlKSkge1xuICAgICAgICAgICAgICBicmVhaztcbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgICAgeS5zZXQocG9vbFR5cGUgPT09ICdhdmcnID8gYXZnVmFsdWUgOiBtaW5NYXhWYWx1ZSwgeVIsIHlDLCBkKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4geTtcbiAgfVxuXG4gIHByb3RlY3RlZCBtYXhQb29sSW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlciwgcGFkOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICByZXR1cm4gdGhpcy5wb29sKHgsIGZTaXplLCBzdHJpZGUsIHBhZCwgJ21heCcpO1xuICB9XG5cbiAgbWF4UG9vbFBvc2l0aW9ucyh4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlciwgcGFkOiBudW1iZXIpIHtcbiAgICBjb25zdCBbeFJvd3MsIHhDb2xzLCBkZXB0aF0gPSB4LnNoYXBlO1xuICAgIGNvbnN0IG91dHB1dFNoYXBlID1cbiAgICAgICAgY29udl91dGlsLmNvbXB1dGVPdXRwdXRTaGFwZTNEKHguc2hhcGUsIGZTaXplLCBkZXB0aCwgc3RyaWRlLCBwYWQpO1xuICAgIGNvbnN0IG1heFBvc2l0aW9ucyA9IEFycmF5M0QuemVyb3Mob3V0cHV0U2hhcGUpO1xuICAgIGZvciAobGV0IGQgPSAwOyBkIDwgZGVwdGg7ICsrZCkge1xuICAgICAgZm9yIChsZXQgeVIgPSAwOyB5UiA8IG91dHB1dFNoYXBlWzBdOyArK3lSKSB7XG4gICAgICAgIGNvbnN0IHhSQ29ybmVyID0geVIgKiBzdHJpZGUgLSBwYWQ7XG4gICAgICAgIGNvbnN0IHhSTWluID0gTWF0aC5tYXgoMCwgeFJDb3JuZXIpO1xuICAgICAgICBjb25zdCB4Uk1heCA9IE1hdGgubWluKHhSb3dzLCBmU2l6ZSArIHhSQ29ybmVyKTtcbiAgICAgICAgZm9yIChsZXQgeUMgPSAwOyB5QyA8IG91dHB1dFNoYXBlWzFdOyArK3lDKSB7XG4gICAgICAgICAgY29uc3QgeENDb3JuZXIgPSB5QyAqIHN0cmlkZSAtIHBhZDtcbiAgICAgICAgICBjb25zdCB4Q01pbiA9IE1hdGgubWF4KDAsIHhDQ29ybmVyKTtcbiAgICAgICAgICBjb25zdCB4Q01heCA9IE1hdGgubWluKHhDb2xzLCBmU2l6ZSArIHhDQ29ybmVyKTtcbiAgICAgICAgICBsZXQgbWF4VmFsdWUgPSBOdW1iZXIuTkVHQVRJVkVfSU5GSU5JVFk7XG4gICAgICAgICAgbGV0IG1heFBvc2l0aW9uID0gLTE7XG4gICAgICAgICAgZm9yIChsZXQgeFIgPSB4Uk1pbjsgeFIgPCB4Uk1heDsgKyt4Uikge1xuICAgICAgICAgICAgY29uc3Qgd1IgPSB4UiAtIHhSQ29ybmVyO1xuICAgICAgICAgICAgZm9yIChsZXQgeEMgPSB4Q01pbjsgeEMgPCB4Q01heDsgKyt4Qykge1xuICAgICAgICAgICAgICBjb25zdCB3QyA9IHhDIC0geENDb3JuZXI7XG4gICAgICAgICAgICAgIGNvbnN0IHBpeGVsID0geC5nZXQoeFIsIHhDLCBkKTtcbiAgICAgICAgICAgICAgaWYgKHBpeGVsID4gbWF4VmFsdWUpIHtcbiAgICAgICAgICAgICAgICBtYXhWYWx1ZSA9IHBpeGVsO1xuICAgICAgICAgICAgICAgIG1heFBvc2l0aW9uID0gd1IgKiBmU2l6ZSArIHdDO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICAgIG1heFBvc2l0aW9ucy5zZXQobWF4UG9zaXRpb24sIHlSLCB5QywgZCk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIG1heFBvc2l0aW9ucztcbiAgfVxuXG4gIHByb3RlY3RlZCBtYXhQb29sQmFja3Byb3BJbnRlcm5hbChcbiAgICAgIGR5OiBBcnJheTNELCB4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBvcmlnU3RyaWRlOiBudW1iZXIsXG4gICAgICBvcmlnUGFkOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICBjb25zdCBtYXhQb3NpdGlvbnMgPSB0aGlzLm1heFBvb2xQb3NpdGlvbnMoeCwgZlNpemUsIG9yaWdTdHJpZGUsIG9yaWdQYWQpO1xuICAgIGNvbnN0IHBhZCA9IGZTaXplIC0gMSAtIG9yaWdQYWQ7XG4gICAgY29uc3QgW2R5Um93cywgZHlDb2xzLCBkZXB0aF0gPSBkeS5zaGFwZTtcblxuICAgIC8vIERpbGF0ZSB0aGUgaW5wdXQuXG4gICAgY29uc3QgZHlSb3dzRGlsYXRlZCA9IChkeVJvd3MgLSAxKSAqIG9yaWdTdHJpZGUgKyAxO1xuICAgIGNvbnN0IGR4Q29sc0RpbGF0ZWQgPSAoZHlDb2xzIC0gMSkgKiBvcmlnU3RyaWRlICsgMTtcblxuICAgIGNvbnN0IG91dHB1dFNoYXBlID0gY29udl91dGlsLmNvbXB1dGVPdXRwdXRTaGFwZTNEKFxuICAgICAgICBbZHlSb3dzRGlsYXRlZCwgZHhDb2xzRGlsYXRlZCwgZGVwdGhdLCBmU2l6ZSwgZGVwdGgsIDEsIHBhZCk7XG4gICAgY29uc3QgZHggPSBBcnJheTNELnplcm9zKG91dHB1dFNoYXBlKTtcblxuICAgIGZvciAobGV0IGQgPSAwOyBkIDwgZGVwdGg7ICsrZCkge1xuICAgICAgZm9yIChsZXQgZHhSID0gMDsgZHhSIDwgZHguc2hhcGVbMF07ICsrZHhSKSB7XG4gICAgICAgIGZvciAobGV0IGR4QyA9IDA7IGR4QyA8IGR4LnNoYXBlWzFdOyArK2R4Qykge1xuICAgICAgICAgIC8vIFNoYWRlciBjb2RlIGJlZ2lucy5cbiAgICAgICAgICBjb25zdCBkeVJDb3JuZXIgPSBkeFIgLSBwYWQ7XG4gICAgICAgICAgY29uc3QgZHlDQ29ybmVyID0gZHhDIC0gcGFkO1xuICAgICAgICAgIGxldCBkb3RQcm9kID0gMDtcbiAgICAgICAgICBmb3IgKGxldCB3UiA9IDA7IHdSIDwgZlNpemU7ICsrd1IpIHtcbiAgICAgICAgICAgIGNvbnN0IGR5UiA9IChkeVJDb3JuZXIgKyB3UikgLyBvcmlnU3RyaWRlO1xuICAgICAgICAgICAgaWYgKGR5UiA8IDAgfHwgZHlSID49IGR5Um93cyB8fCBNYXRoLmZsb29yKGR5UikgIT09IGR5Uikge1xuICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGZvciAobGV0IHdDID0gMDsgd0MgPCBmU2l6ZTsgKyt3Qykge1xuICAgICAgICAgICAgICBjb25zdCBkeUMgPSAoZHlDQ29ybmVyICsgd0MpIC8gb3JpZ1N0cmlkZTtcbiAgICAgICAgICAgICAgaWYgKGR5QyA8IDAgfHwgZHlDID49IGR5Q29scyB8fCBNYXRoLmZsb29yKGR5QykgIT09IGR5Qykge1xuICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIGNvbnN0IG1heFBvcyA9IGZTaXplICogZlNpemUgLSAxIC0gbWF4UG9zaXRpb25zLmdldChkeVIsIGR5QywgZCk7XG4gICAgICAgICAgICAgIGNvbnN0IGN1clBvcyA9IHdSICogZlNpemUgKyB3QztcblxuICAgICAgICAgICAgICBjb25zdCBtYXNrID0gbWF4UG9zID09PSBjdXJQb3MgPyAxIDogMDtcbiAgICAgICAgICAgICAgaWYgKG1hc2sgPT09IDApIHtcbiAgICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAgIGNvbnN0IHBpeGVsID0gZHkuZ2V0KGR5UiwgZHlDLCBkKTtcbiAgICAgICAgICAgICAgZG90UHJvZCArPSBwaXhlbCAqIG1hc2s7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICAgIGR4LnNldChkb3RQcm9kLCBkeFIsIGR4QywgZCk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIGR4O1xuICB9XG5cbiAgcHJvdGVjdGVkIG1pblBvb2xJbnRlcm5hbChcbiAgICAgIHg6IEFycmF5M0QsIGZTaXplOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLCBwYWQ6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIHJldHVybiB0aGlzLnBvb2woeCwgZlNpemUsIHN0cmlkZSwgcGFkLCAnbWluJyk7XG4gIH1cblxuICBwcm90ZWN0ZWQgYXZnUG9vbEludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsIHBhZDogbnVtYmVyKTogQXJyYXkzRCB7XG4gICAgcmV0dXJuIHRoaXMucG9vbCh4LCBmU2l6ZSwgc3RyaWRlLCBwYWQsICdhdmcnKTtcbiAgfVxuXG4gIHByb3RlY3RlZCByZXNpemVCaWxpbmVhcjNESW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBuZXdTaGFwZTJEOiBbbnVtYmVyLCBudW1iZXJdLFxuICAgICAgYWxpZ25Db3JuZXJzOiBib29sZWFuKTogQXJyYXkzRCB7XG4gICAgY29uc3Qgb3V0cHV0ID0gQXJyYXkzRC56ZXJvcyhbbmV3U2hhcGUyRFswXSwgbmV3U2hhcGUyRFsxXSwgeC5zaGFwZVsyXV0pO1xuXG4gICAgY29uc3QgZWZmZWN0aXZlSW5wdXRTaXplID1cbiAgICAgICAgYWxpZ25Db3JuZXJzID8gW3guc2hhcGVbMF0gLSAxLCB4LnNoYXBlWzFdIC0gMSwgeC5zaGFwZVsyXV0gOiB4LnNoYXBlO1xuICAgIGNvbnN0IGVmZmVjdGl2ZU91dHB1dFNpemUgPSBhbGlnbkNvcm5lcnMgP1xuICAgICAgICBbb3V0cHV0LnNoYXBlWzBdIC0gMSwgb3V0cHV0LnNoYXBlWzFdIC0gMSwgb3V0cHV0LnNoYXBlWzJdXSA6XG4gICAgICAgIG91dHB1dC5zaGFwZTtcbiAgICBmb3IgKGxldCByID0gMDsgciA8IG91dHB1dC5zaGFwZVswXTsgcisrKSB7XG4gICAgICBmb3IgKGxldCBjID0gMDsgYyA8IG91dHB1dC5zaGFwZVsxXTsgYysrKSB7XG4gICAgICAgIGZvciAobGV0IGQgPSAwOyBkIDwgb3V0cHV0LnNoYXBlWzJdOyBkKyspIHtcbiAgICAgICAgICAvLyBCZWdpbiBzaGFkZXIuXG5cbiAgICAgICAgICAvLyBDb21wdXRlIHRoZSBmcmFjdGlvbmFsIGluZGV4IG9mIHRoZSBzb3VyY2UuXG4gICAgICAgICAgY29uc3Qgc291cmNlRnJhY1JvdyA9XG4gICAgICAgICAgICAgIChlZmZlY3RpdmVJbnB1dFNpemVbMF0pICogciAvIChlZmZlY3RpdmVPdXRwdXRTaXplWzBdKTtcbiAgICAgICAgICBjb25zdCBzb3VyY2VGcmFjQ29sID1cbiAgICAgICAgICAgICAgKGVmZmVjdGl2ZUlucHV0U2l6ZVsxXSkgKiBjIC8gKGVmZmVjdGl2ZU91dHB1dFNpemVbMV0pO1xuXG4gICAgICAgICAgY29uc3Qgc291cmNlUm93Rmxvb3IgPSBNYXRoLmZsb29yKHNvdXJjZUZyYWNSb3cpO1xuICAgICAgICAgIGNvbnN0IHNvdXJjZVJvd0NlaWwgPVxuICAgICAgICAgICAgICBNYXRoLm1pbih4LnNoYXBlWzBdIC0gMSwgTWF0aC5jZWlsKHNvdXJjZUZyYWNSb3cpKTtcbiAgICAgICAgICBjb25zdCBzb3VyY2VDb2xGbG9vciA9IE1hdGguZmxvb3Ioc291cmNlRnJhY0NvbCk7XG4gICAgICAgICAgY29uc3Qgc291cmNlQ29sQ2VpbCA9XG4gICAgICAgICAgICAgIE1hdGgubWluKHguc2hhcGVbMV0gLSAxLCBNYXRoLmNlaWwoc291cmNlRnJhY0NvbCkpO1xuXG4gICAgICAgICAgY29uc3QgdG9wTGVmdCA9IHguZ2V0KHNvdXJjZVJvd0Zsb29yLCBzb3VyY2VDb2xGbG9vciwgZCk7XG4gICAgICAgICAgY29uc3QgYm90dG9tTGVmdCA9IHguZ2V0KHNvdXJjZVJvd0NlaWwsIHNvdXJjZUNvbEZsb29yLCBkKTtcbiAgICAgICAgICBjb25zdCB0b3BSaWdodCA9IHguZ2V0KHNvdXJjZVJvd0Zsb29yLCBzb3VyY2VDb2xDZWlsLCBkKTtcbiAgICAgICAgICBjb25zdCBib3R0b21SaWdodCA9IHguZ2V0KHNvdXJjZVJvd0NlaWwsIHNvdXJjZUNvbENlaWwsIGQpO1xuXG4gICAgICAgICAgY29uc3Qgcm93RnJhYyA9IHNvdXJjZUZyYWNSb3cgLSBzb3VyY2VSb3dGbG9vcjtcbiAgICAgICAgICBjb25zdCBjb2xGcmFjID0gc291cmNlRnJhY0NvbCAtIHNvdXJjZUNvbEZsb29yO1xuXG4gICAgICAgICAgY29uc3QgdG9wID0gdG9wTGVmdCArICh0b3BSaWdodCAtIHRvcExlZnQpICogY29sRnJhYztcbiAgICAgICAgICBjb25zdCBib3R0b20gPSBib3R0b21MZWZ0ICsgKGJvdHRvbVJpZ2h0IC0gYm90dG9tTGVmdCkgKiBjb2xGcmFjO1xuICAgICAgICAgIGNvbnN0IG5ld1ZhbHVlID0gdG9wICsgKGJvdHRvbSAtIHRvcCkgKiByb3dGcmFjO1xuXG4gICAgICAgICAgb3V0cHV0LnNldChuZXdWYWx1ZSwgciwgYywgZCk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4gb3V0cHV0O1xuICB9XG5cbiAgcHJvdGVjdGVkIGJhdGNoTm9ybWFsaXphdGlvbjNESW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBtZWFuOiBBcnJheTNEfEFycmF5MUQsIHZhcmlhbmNlOiBBcnJheTNEfEFycmF5MUQsXG4gICAgICB2YXJpYW5jZUVwc2lsb24gPSAuMDAxLCBzY2FsZT86IEFycmF5M0R8QXJyYXkxRCxcbiAgICAgIG9mZnNldD86IEFycmF5M0R8QXJyYXkxRCk6IEFycmF5M0Qge1xuICAgIGNvbnN0IHhWYWx1ZXMgPSB4LmdldFZhbHVlcygpO1xuICAgIGNvbnN0IG1lYW5WYWx1ZXMgPSBtZWFuLmdldFZhbHVlcygpO1xuICAgIGNvbnN0IHZhcmlhbmNlVmFsdWVzID0gdmFyaWFuY2UuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3Qgc2NhbGVWYWx1ZXMgPSBzY2FsZSA/IHNjYWxlLmdldFZhbHVlcygpIDogbmV3IEZsb2F0MzJBcnJheShbMV0pO1xuICAgIGNvbnN0IG9mZnNldFZhbHVlcyA9IG9mZnNldCA/IG9mZnNldC5nZXRWYWx1ZXMoKSA6IG5ldyBGbG9hdDMyQXJyYXkoWzBdKTtcbiAgICBjb25zdCBvdXRWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KHhWYWx1ZXMubGVuZ3RoKTtcblxuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgeFZhbHVlcy5sZW5ndGg7IGkrKykge1xuICAgICAgb3V0VmFsdWVzW2ldID0gb2Zmc2V0VmFsdWVzW2kgJSBvZmZzZXRWYWx1ZXMubGVuZ3RoXSArXG4gICAgICAgICAgKHhWYWx1ZXNbaV0gLSBtZWFuVmFsdWVzW2kgJSBtZWFuVmFsdWVzLmxlbmd0aF0pICpcbiAgICAgICAgICAgICAgc2NhbGVWYWx1ZXNbaSAlIHNjYWxlVmFsdWVzLmxlbmd0aF0gL1xuICAgICAgICAgICAgICBNYXRoLnNxcnQoXG4gICAgICAgICAgICAgICAgICB2YXJpYW5jZVZhbHVlc1tpICUgdmFyaWFuY2VWYWx1ZXMubGVuZ3RoXSArIHZhcmlhbmNlRXBzaWxvbik7XG4gICAgfVxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8QXJyYXkzRD4oeC5zaGFwZSwge3ZhbHVlczogb3V0VmFsdWVzfSk7XG4gIH1cbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi91dGlsJztcblxuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4vd2ViZ2wvZ3BncHVfY29udGV4dCc7XG5pbXBvcnQge1RleHR1cmVNYW5hZ2VyfSBmcm9tICcuL3dlYmdsL3RleHR1cmVfbWFuYWdlcic7XG5pbXBvcnQgKiBhcyB3ZWJnbF91dGlsIGZyb20gJy4vd2ViZ2wvd2ViZ2xfdXRpbCc7XG5cbi8vIFRoZXNlIGdsb2JhbCB2YXJpYWJsZXMgbmVlZCB0byBiZSBpbml0aWFsaXplZCB0byBudWxsIHNvIHRoYXQgY2xvc3VyZSBrbm93c1xuLy8gbm90IHRvIHNlYWwgdGhlbS5cbi8qKiBAaGlkZGVuICovXG5leHBvcnQgbGV0IEdQR1BVOiBHUEdQVUNvbnRleHQgPSBudWxsITtcbi8qKiBAaGlkZGVuICovXG5leHBvcnQgbGV0IFRFWFRVUkVfTUFOQUdFUjogVGV4dHVyZU1hbmFnZXIgPSBudWxsITtcblxuLyoqIEBoaWRkZW4gKi9cbmV4cG9ydCBpbnRlcmZhY2UgTkRBcnJheURhdGEge1xuICB2YWx1ZXM/OiBGbG9hdDMyQXJyYXk7XG4gIHRleHR1cmU/OiBXZWJHTFRleHR1cmU7XG4gIC8qKiBbcm93cywgY29sdW1uc10gc2hhcGUgb2YgdGhlIHRleHR1cmUuICovXG4gIHRleHR1cmVTaGFwZVJDPzogW251bWJlciwgbnVtYmVyXTtcbn1cblxuLyoqIEBoaWRkZW4gKi9cbmV4cG9ydCBmdW5jdGlvbiBpbml0aWFsaXplR1BVKFxuICAgIGdwZ3B1OiBHUEdQVUNvbnRleHQsIHRleHR1cmVNYW5hZ2VyOiBUZXh0dXJlTWFuYWdlcikge1xuICBHUEdQVSA9IGdwZ3B1O1xuICBURVhUVVJFX01BTkFHRVIgPSB0ZXh0dXJlTWFuYWdlcjtcbn1cblxuZnVuY3Rpb24gdGhyb3dJZkdQVU5vdEluaXRpYWxpemVkKCkge1xuICBpZiAoR1BHUFUgPT0gbnVsbCB8fCBURVhUVVJFX01BTkFHRVIgPT0gbnVsbCkge1xuICAgIHRocm93IG5ldyBFcnJvcignR1BVIG5vdCBpbnRpYWxpemVkLicpO1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBOREFycmF5IHtcbiAgLyoqIFRoZSBzaGFwZSBvZiB0aGUgbmRhcnJheS4gKi9cbiAgc2hhcGU6IG51bWJlcltdO1xuICAvKiogTnVtYmVyIG9mIGVsZW1lbnRzIGluIHRoZSBuZGFycmF5LiAqL1xuICBzaXplOiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIE51bWJlciBvZiBlbGVtZW50cyB0byBza2lwIGluIGVhY2ggZGltZW5zaW9uIHdoZW4gaW5kZXhpbmcuIFNlZVxuICAgKiBodHRwczovL2RvY3Muc2NpcHkub3JnL2RvYy9udW1weS9yZWZlcmVuY2UvZ2VuZXJhdGVkL251bXB5Lm5kYXJyYXkuc3RyaWRlcy5odG1sXG4gICAqL1xuICBwcm90ZWN0ZWQgc3RyaWRlczogbnVtYmVyW107XG5cbiAgcHJpdmF0ZSBkYXRhOiBOREFycmF5RGF0YTtcblxuICBwcm90ZWN0ZWQgY29uc3RydWN0b3Ioc2hhcGU6IG51bWJlcltdLCBkYXRhOiBOREFycmF5RGF0YSkge1xuICAgIC8vIFNhbml0eSBjaGVja3MuXG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGRhdGEudmFsdWVzICE9IG51bGwgfHwgZGF0YS50ZXh0dXJlICE9IG51bGwsXG4gICAgICAgICdFaXRoZXIgYHZhbHVlc2Agb3IgYHRleHR1cmVgIG11c3QgYmUgZGVmaW5lZCcpO1xuXG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGRhdGEudGV4dHVyZSA9PSBudWxsIHx8IChkYXRhLnRleHR1cmVTaGFwZVJDICE9IG51bGwpLFxuICAgICAgICAnYHRleHR1cmVTaGFwZWAgbXVzdCBiZSBkZWZpbmVkIHdoZW4gYHRleHR1cmVgIGlzIGRlZmluZWQnKTtcblxuICAgIHRoaXMuc2l6ZSA9IHV0aWwuc2l6ZUZyb21TaGFwZShzaGFwZSk7XG5cbiAgICBpZiAoZGF0YS52YWx1ZXMgIT0gbnVsbCkge1xuICAgICAgdXRpbC5hc3NlcnQoXG4gICAgICAgICAgdGhpcy5zaXplID09PSBkYXRhLnZhbHVlcy5sZW5ndGgsXG4gICAgICAgICAgJ0NvbnN0cnVjdGluZyBuZGFycmF5IG9mIHNoYXBlICgnICsgdGhpcy5zaXplICsgJykgc2hvdWxkIG1hdGNoIHRoZScgK1xuICAgICAgICAgICAgICAnIGxlbmd0aCBvZiB2YWx1ZXMgKCcgKyBkYXRhLnZhbHVlcy5sZW5ndGggKyAnKScpO1xuICAgIH1cblxuICAgIHRoaXMuc2hhcGUgPSBzaGFwZTtcbiAgICB0aGlzLmRhdGEgPSBkYXRhO1xuICAgIGNvbnN0IGRpbSA9IHRoaXMuc2hhcGUubGVuZ3RoO1xuXG4gICAgaWYgKGRpbSA8IDIpIHtcbiAgICAgIHRoaXMuc3RyaWRlcyA9IFtdO1xuICAgIH0gZWxzZSB7XG4gICAgICAvLyBMYXN0IGRpbWVuc2lvbiBoYXMgaW1wbGljaXQgc3RyaWRlIG9mIDEsIHRodXMgaGF2aW5nIEQtMSAoaW5zdGVhZCBvZiBEKVxuICAgICAgLy8gc3RyaWRlcy5cbiAgICAgIHRoaXMuc3RyaWRlcyA9IG5ldyBBcnJheShkaW0gLSAxKTtcbiAgICAgIHRoaXMuc3RyaWRlc1tkaW0gLSAyXSA9IHRoaXMuc2hhcGVbZGltIC0gMV07XG4gICAgICBmb3IgKGxldCBpID0gZGltIC0gMzsgaSA+PSAwOyAtLWkpIHtcbiAgICAgICAgdGhpcy5zdHJpZGVzW2ldID0gdGhpcy5zdHJpZGVzW2kgKyAxXSAqIHRoaXMuc2hhcGVbaSArIDFdO1xuICAgICAgfVxuICAgIH1cbiAgfVxuXG4gIC8qKiBDcmVhdGVzIGEgbmRhcnJheSBvZiB6ZXJvcyB3aXRoIHRoZSBzcGVjaWZpZWQgc2hhcGUuICovXG4gIHN0YXRpYyB6ZXJvczxUIGV4dGVuZHMgTkRBcnJheT4oc2hhcGU6IG51bWJlcltdKTogVCB7XG4gICAgY29uc3QgdmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheSh1dGlsLnNpemVGcm9tU2hhcGUoc2hhcGUpKTtcbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KHNoYXBlLCB7dmFsdWVzfSk7XG4gIH1cblxuICAvKiogQ3JlYXRlcyBhIG5kYXJyYXkgb2YgemVyb3Mgd2l0aCB0aGUgc2FtZSBzaGFwZSBhcyB0aGUgc3BlY2lmaWVkIG5kYXJyYXkuXG4gICAqL1xuICBzdGF0aWMgemVyb3NMaWtlPFQgZXh0ZW5kcyBOREFycmF5Pihhbm90aGVyOiBUKTogVCB7XG4gICAgcmV0dXJuIE5EQXJyYXkuemVyb3MoYW5vdGhlci5zaGFwZSkgYXMgVDtcbiAgfVxuXG4gIC8qKiBDcmVhdGVzIGEgbmRhcnJheSB3aXRoIHRoZSBzYW1lIHZhbHVlcy9zaGFwZSBhcyB0aGUgc3BlY2lmaWVkIG5kYXJyYXkuICovXG4gIHN0YXRpYyBsaWtlPFQgZXh0ZW5kcyBOREFycmF5Pihhbm90aGVyOiBUKTogVCB7XG4gICAgY29uc3QgdmFsdWVzID0gYW5vdGhlci5nZXRWYWx1ZXMoKTtcbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KGFub3RoZXIuc2hhcGUsIHt2YWx1ZXM6IG5ldyBGbG9hdDMyQXJyYXkodmFsdWVzKX0pO1xuICB9XG5cbiAgLyoqXG4gICAqIE1ha2VzIGEgbmV3IG5kYXJyYXkgd2l0aCB0aGUgcHJvdmlkZWQgc2hhcGUgYW5kIHZhbHVlcy4gVmFsdWVzIHNob3VsZCBiZSBpblxuICAgKiBhIGZsYXQgYXJyYXkuXG4gICAqL1xuICBzdGF0aWMgbWFrZTxUIGV4dGVuZHMgTkRBcnJheT4oc2hhcGU6IG51bWJlcltdLCBkYXRhOiBOREFycmF5RGF0YSk6IFQge1xuICAgIHN3aXRjaCAoc2hhcGUubGVuZ3RoKSB7XG4gICAgICBjYXNlIDA6XG4gICAgICAgIHJldHVybiBuZXcgU2NhbGFyKGRhdGEpIGFzIFQ7XG4gICAgICBjYXNlIDE6XG4gICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICAgICAgcmV0dXJuIG5ldyBBcnJheTFEKGRhdGEpIGFzIGFueTtcbiAgICAgIGNhc2UgMjpcbiAgICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgICByZXR1cm4gbmV3IEFycmF5MkQoc2hhcGUgYXMgW251bWJlciwgbnVtYmVyXSwgZGF0YSkgYXMgYW55O1xuICAgICAgY2FzZSAzOlxuICAgICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgICAgIHJldHVybiBuZXcgQXJyYXkzRChzaGFwZSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGRhdGEpIGFzIGFueTtcbiAgICAgIGNhc2UgNDpcbiAgICAgICAgcmV0dXJuIG5ldyBBcnJheTREKFxuICAgICAgICAgICAgICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICAgICAgICAgICAgICAgICBzaGFwZSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgZGF0YSkgYXMgYW55O1xuICAgICAgZGVmYXVsdDpcbiAgICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgICByZXR1cm4gbmV3IE5EQXJyYXkoc2hhcGUsIGRhdGEpIGFzIGFueTtcbiAgICB9XG4gIH1cblxuICAvKiogUmVzaGFwZXMgdGhlIGN1cnJlbnQgbmRhcnJheSBpbnRvIHRoZSBwcm92aWRlZCBzaGFwZS4gKi9cbiAgcmVzaGFwZTxUIGV4dGVuZHMgTkRBcnJheT4obmV3U2hhcGU6IG51bWJlcltdKTogVCB7XG4gICAgaWYgKHV0aWwuYXJyYXlzRXF1YWwodGhpcy5zaGFwZSwgbmV3U2hhcGUpKSB7XG4gICAgICAvLyBOby1vcC5cbiAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICAgIHJldHVybiB0aGlzIGFzIGFueTtcbiAgICB9XG5cbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgdGhpcy5zaXplID09PSB1dGlsLnNpemVGcm9tU2hhcGUobmV3U2hhcGUpLFxuICAgICAgICAnbmV3IHNoYXBlIGFuZCBvbGQgc2hhcGUgbXVzdCBoYXZlIHRoZSBzYW1lIG51bWJlciBvZiBlbGVtZW50cy4nKTtcblxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8VD4obmV3U2hhcGUsIHRoaXMuZGF0YSk7XG4gIH1cblxuICBhc1NjYWxhcigpOiBTY2FsYXIge1xuICAgIHV0aWwuYXNzZXJ0KHRoaXMuc2l6ZSA9PT0gMSwgJ1RoZSBhcnJheSBtdXN0IGhhdmUgb25seSAxIGVsZW1lbnQuJyk7XG4gICAgcmV0dXJuIHRoaXMucmVzaGFwZTxTY2FsYXI+KFtdKTtcbiAgfVxuXG4gIGFzMUQoKTogQXJyYXkxRCB7XG4gICAgcmV0dXJuIHRoaXMucmVzaGFwZTxBcnJheTFEPihbdGhpcy5zaXplXSk7XG4gIH1cblxuICBhczJEKHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTogQXJyYXkyRCB7XG4gICAgcmV0dXJuIHRoaXMucmVzaGFwZTxBcnJheTJEPihbcm93cywgY29sdW1uc10pO1xuICB9XG5cbiAgYXMzRChyb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlciwgZGVwdGg6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIHJldHVybiB0aGlzLnJlc2hhcGU8QXJyYXkzRD4oW3Jvd3MsIGNvbHVtbnMsIGRlcHRoXSk7XG4gIH1cblxuICBhczREKHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyLCBkZXB0aDogbnVtYmVyLCBkZXB0aDI6IG51bWJlcik6IEFycmF5NEQge1xuICAgIHJldHVybiB0aGlzLnJlc2hhcGU8QXJyYXk0RD4oW3Jvd3MsIGNvbHVtbnMsIGRlcHRoLCBkZXB0aDJdKTtcbiAgfVxuXG4gIGdldCByYW5rKCk6IG51bWJlciB7XG4gICAgcmV0dXJuIHRoaXMuc2hhcGUubGVuZ3RoO1xuICB9XG5cbiAgZ2V0KC4uLmxvY3M6IG51bWJlcltdKSB7XG4gICAgbGV0IGluZGV4ID0gbG9jc1tsb2NzLmxlbmd0aCAtIDFdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbG9jcy5sZW5ndGggLSAxOyArK2kpIHtcbiAgICAgIGluZGV4ICs9IHRoaXMuc3RyaWRlc1tpXSAqIGxvY3NbaV07XG4gICAgfVxuICAgIHJldHVybiB0aGlzLmdldFZhbHVlcygpW2luZGV4XTtcbiAgfVxuXG4gIGFkZCh2YWx1ZTogbnVtYmVyLCAuLi5sb2NzOiBudW1iZXJbXSkge1xuICAgIHRoaXMuc2V0KHRoaXMuZ2V0KC4uLmxvY3MpICsgdmFsdWUsIC4uLmxvY3MpO1xuICB9XG5cbiAgc2V0KHZhbHVlOiBudW1iZXIsIC4uLmxvY3M6IG51bWJlcltdKSB7XG4gICAgbGV0IGluZGV4ID0gbG9jc1tsb2NzLmxlbmd0aCAtIDFdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbG9jcy5sZW5ndGggLSAxOyArK2kpIHtcbiAgICAgIGluZGV4ICs9IHRoaXMuc3RyaWRlc1tpXSAqIGxvY3NbaV07XG4gICAgfVxuICAgIHRoaXMuZ2V0VmFsdWVzKClbaW5kZXhdID0gdmFsdWU7XG4gIH1cblxuICBsb2NUb0luZGV4KGxvY3M6IG51bWJlcltdKTogbnVtYmVyIHtcbiAgICBsZXQgaW5kZXggPSBsb2NzW2xvY3MubGVuZ3RoIC0gMV07XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBsb2NzLmxlbmd0aCAtIDE7ICsraSkge1xuICAgICAgaW5kZXggKz0gdGhpcy5zdHJpZGVzW2ldICogbG9jc1tpXTtcbiAgICB9XG4gICAgcmV0dXJuIGluZGV4O1xuICB9XG5cbiAgaW5kZXhUb0xvYyhpbmRleDogbnVtYmVyKTogbnVtYmVyW10ge1xuICAgIGNvbnN0IGxvY3M6IG51bWJlcltdID0gbmV3IEFycmF5KHRoaXMuc2hhcGUubGVuZ3RoKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGxvY3MubGVuZ3RoIC0gMTsgKytpKSB7XG4gICAgICBsb2NzW2ldID0gTWF0aC5mbG9vcihpbmRleCAvIHRoaXMuc3RyaWRlc1tpXSk7XG4gICAgICBpbmRleCAtPSBsb2NzW2ldICogdGhpcy5zdHJpZGVzW2ldO1xuICAgIH1cbiAgICBsb2NzW2xvY3MubGVuZ3RoIC0gMV0gPSBpbmRleDtcbiAgICByZXR1cm4gbG9jcztcbiAgfVxuXG4gIGZpbGwodmFsdWU6IG51bWJlcikge1xuICAgIHRoaXMuZ2V0VmFsdWVzKCkuZmlsbCh2YWx1ZSk7XG4gIH1cblxuICBnZXREYXRhKCk6IE5EQXJyYXlEYXRhIHtcbiAgICByZXR1cm4gdGhpcy5kYXRhO1xuICB9XG5cbiAgZ2V0VmFsdWVzKCk6IEZsb2F0MzJBcnJheSB7XG4gICAgaWYgKHRoaXMuZGF0YS52YWx1ZXMgPT0gbnVsbCkge1xuICAgICAgdGhyb3dJZkdQVU5vdEluaXRpYWxpemVkKCk7XG4gICAgICB0aGlzLmRhdGEudmFsdWVzID0gR1BHUFUuZG93bmxvYWRNYXRyaXhGcm9tVGV4dHVyZShcbiAgICAgICAgICB0aGlzLmRhdGEudGV4dHVyZSEsIHRoaXMuZGF0YS50ZXh0dXJlU2hhcGVSQyFbMF0sXG4gICAgICAgICAgdGhpcy5kYXRhLnRleHR1cmVTaGFwZVJDIVsxXSk7XG4gICAgICB0aGlzLmRpc3Bvc2VUZXh0dXJlKCk7XG4gICAgfVxuICAgIHJldHVybiB0aGlzLmRhdGEudmFsdWVzO1xuICB9XG5cbiAgcHJpdmF0ZSB1cGxvYWRUb0dQVShwcmVmZXJyZWRUZXhTaGFwZT86IFtudW1iZXIsIG51bWJlcl0pIHtcbiAgICB0aHJvd0lmR1BVTm90SW5pdGlhbGl6ZWQoKTtcbiAgICB0aGlzLmRhdGEudGV4dHVyZVNoYXBlUkMgPSB3ZWJnbF91dGlsLmdldFRleHR1cmVTaGFwZUZyb21Mb2dpY2FsU2hhcGUoXG4gICAgICAgIEdQR1BVLmdsLCB0aGlzLnNoYXBlLCBwcmVmZXJyZWRUZXhTaGFwZSk7XG4gICAgdGhpcy5kYXRhLnRleHR1cmUgPVxuICAgICAgICBURVhUVVJFX01BTkFHRVIuYWNxdWlyZVRleHR1cmUodGhpcy5kYXRhLnRleHR1cmVTaGFwZVJDKTtcblxuICAgIEdQR1BVLnVwbG9hZE1hdHJpeFRvVGV4dHVyZShcbiAgICAgICAgdGhpcy5kYXRhLnRleHR1cmUsIHRoaXMuZGF0YS50ZXh0dXJlU2hhcGVSQ1swXSxcbiAgICAgICAgdGhpcy5kYXRhLnRleHR1cmVTaGFwZVJDWzFdLCB0aGlzLmRhdGEudmFsdWVzISk7XG5cbiAgICB0aGlzLmRhdGEudmFsdWVzID0gbnVsbCE7XG4gIH1cblxuICBnZXRUZXh0dXJlKHByZWZlcnJlZFNoYXBlUkM/OiBbbnVtYmVyLCBudW1iZXJdKTogV2ViR0xUZXh0dXJlIHtcbiAgICBpZiAodGhpcy5kYXRhLnRleHR1cmUgPT0gbnVsbCkge1xuICAgICAgdGhpcy51cGxvYWRUb0dQVShwcmVmZXJyZWRTaGFwZVJDKTtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMuZGF0YS50ZXh0dXJlITtcbiAgfVxuXG4gIGdldFRleHR1cmVTaGFwZVJDKHByZWZlcnJlZFNoYXBlUkM/OiBbbnVtYmVyLCBudW1iZXJdKTogW251bWJlciwgbnVtYmVyXSB7XG4gICAgaWYgKHRoaXMuZGF0YS50ZXh0dXJlU2hhcGVSQyA9PSBudWxsKSB7XG4gICAgICB0aGlzLnVwbG9hZFRvR1BVKHByZWZlcnJlZFNoYXBlUkMpO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcy5kYXRhLnRleHR1cmVTaGFwZVJDITtcbiAgfVxuXG4gIGRpc3Bvc2UoKTogdm9pZCB7XG4gICAgdGhpcy5kYXRhLnZhbHVlcyA9IG51bGwhO1xuICAgIHRoaXMuc2hhcGUgPSBudWxsITtcbiAgICBpZiAodGhpcy5kYXRhLnRleHR1cmUgIT0gbnVsbCkge1xuICAgICAgdGhpcy5kaXNwb3NlVGV4dHVyZSgpO1xuICAgIH1cbiAgfVxuXG4gIHByaXZhdGUgZGlzcG9zZVRleHR1cmUoKSB7XG4gICAgdGhyb3dJZkdQVU5vdEluaXRpYWxpemVkKCk7XG4gICAgVEVYVFVSRV9NQU5BR0VSLnJlbGVhc2VUZXh0dXJlKFxuICAgICAgICB0aGlzLmRhdGEudGV4dHVyZSEsIHRoaXMuZGF0YS50ZXh0dXJlU2hhcGVSQyEpO1xuICAgIHRoaXMuZGF0YS50ZXh0dXJlID0gbnVsbCE7XG4gICAgdGhpcy5kYXRhLnRleHR1cmVTaGFwZVJDID0gbnVsbCE7XG4gIH1cblxuICBpbkdQVSgpOiBib29sZWFuIHtcbiAgICByZXR1cm4gdGhpcy5kYXRhLnRleHR1cmUgIT0gbnVsbDtcbiAgfVxuXG4gIGVxdWFscyh0OiBOREFycmF5KTogYm9vbGVhbiB7XG4gICAgcmV0dXJuIHV0aWwuYXJyYXlzRXF1YWwodGhpcy5zaGFwZSwgdC5zaGFwZSkgJiZcbiAgICAgICAgdXRpbC5hcnJheXNFcXVhbCh0aGlzLmdldFZhbHVlcygpLCB0LmdldFZhbHVlcygpKTtcbiAgfVxuXG4gIHN0YXRpYyByYW5kPFQgZXh0ZW5kcyBOREFycmF5PihzaGFwZTogbnVtYmVyW10sIHJhbmRGdW5jdGlvbjogKCkgPT4gbnVtYmVyKTpcbiAgICAgIFQge1xuICAgIGNvbnN0IHNpemUgPSB1dGlsLnNpemVGcm9tU2hhcGUoc2hhcGUpO1xuICAgIGNvbnN0IHZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkoc2l6ZSk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBzaXplOyBpKyspIHtcbiAgICAgIHZhbHVlc1tpXSA9IHJhbmRGdW5jdGlvbigpO1xuICAgIH1cblxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8VD4oc2hhcGUsIHt2YWx1ZXN9KTtcbiAgfVxuXG4gIHN0YXRpYyByYW5kTm9ybWFsPFQgZXh0ZW5kcyBOREFycmF5PihzaGFwZTogbnVtYmVyW10sIG1lYW4gPSAwLCBzdGREZXYgPSAxKSB7XG4gICAgcmV0dXJuIE5EQXJyYXkucmFuZDxUPihzaGFwZSwgKCkgPT4gdXRpbC5yYW5kR2F1c3MobWVhbiwgc3RkRGV2KSk7XG4gIH1cblxuICBzdGF0aWMgcmFuZFRydW5jYXRlZE5vcm1hbDxUIGV4dGVuZHMgTkRBcnJheT4oXG4gICAgICBzaGFwZTogbnVtYmVyW10sIG1lYW4gPSAwLCBzdGREZXYgPSAxKSB7XG4gICAgcmV0dXJuIE5EQXJyYXkucmFuZDxUPihzaGFwZSwgKCkgPT4gdXRpbC5yYW5kR2F1c3MobWVhbiwgc3RkRGV2LCB0cnVlKSk7XG4gIH1cblxuICBzdGF0aWMgcmFuZFVuaWZvcm08VCBleHRlbmRzIE5EQXJyYXk+KHNoYXBlOiBudW1iZXJbXSwgYTogbnVtYmVyLCBiOiBudW1iZXIpIHtcbiAgICByZXR1cm4gTkRBcnJheS5yYW5kPFQ+KHNoYXBlLCAoKSA9PiB1dGlsLnJhbmRVbmlmb3JtKGEsIGIpKTtcbiAgfVxufVxuXG5leHBvcnQgY2xhc3MgU2NhbGFyIGV4dGVuZHMgTkRBcnJheSB7XG4gIGNvbnN0cnVjdG9yKGRhdGE6IE5EQXJyYXlEYXRhKSB7XG4gICAgaWYgKGRhdGEudGV4dHVyZSAhPSBudWxsKSB7XG4gICAgICBkYXRhLnRleHR1cmVTaGFwZVJDID0gWzEsIDFdO1xuICAgIH1cbiAgICBzdXBlcihbXSwgZGF0YSk7XG4gIH1cblxuICBzdGF0aWMgbmV3KHZhbHVlOiBudW1iZXIpIHtcbiAgICByZXR1cm4gbmV3IFNjYWxhcih7dmFsdWVzOiBuZXcgRmxvYXQzMkFycmF5KFt2YWx1ZV0pfSk7XG4gIH1cblxuICBzdGF0aWMgWkVSTyA9IFNjYWxhci5uZXcoMCk7XG4gIHN0YXRpYyBPTkUgPSBTY2FsYXIubmV3KDEpO1xuICBzdGF0aWMgVFdPID0gU2NhbGFyLm5ldygyKTtcbiAgc3RhdGljIE5FR19PTkUgPSBTY2FsYXIubmV3KC0xKTtcblxuICBnZXQoKTogbnVtYmVyIHtcbiAgICByZXR1cm4gdGhpcy5nZXRWYWx1ZXMoKVswXTtcbiAgfVxuXG4gIHNldCh2YWx1ZTogbnVtYmVyKSB7XG4gICAgdGhpcy5nZXRWYWx1ZXMoKVswXSA9IHZhbHVlO1xuICB9XG5cbiAgYWRkKHZhbHVlOiBudW1iZXIpIHtcbiAgICB0aGlzLmdldFZhbHVlcygpWzBdICs9IHZhbHVlO1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBBcnJheTFEIGV4dGVuZHMgTkRBcnJheSB7XG4gIHNoYXBlOiBbbnVtYmVyXTtcblxuICBjb25zdHJ1Y3RvcihkYXRhOiBOREFycmF5RGF0YSkge1xuICAgIGNvbnN0IHNoYXBlID0gKGRhdGEudmFsdWVzICE9IG51bGwpID9cbiAgICAgICAgW2RhdGEudmFsdWVzLmxlbmd0aF0gOlxuICAgICAgICBbdXRpbC5zaXplRnJvbVNoYXBlKGRhdGEudGV4dHVyZVNoYXBlUkMhKV07XG4gICAgc3VwZXIoc2hhcGUsIGRhdGEpO1xuICB9XG5cbiAgc3RhdGljIG5ldyh2YWx1ZXM6IEZsb2F0MzJBcnJheXxudW1iZXJbXSkge1xuICAgIGlmICghKHZhbHVlcyBpbnN0YW5jZW9mIEZsb2F0MzJBcnJheSkpIHtcbiAgICAgIGNvbnN0IGluZmVycmVkU2hhcGUgPSB1dGlsLmluZmVyU2hhcGUodmFsdWVzKTtcbiAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgIGluZmVycmVkU2hhcGUubGVuZ3RoID09PSAxLFxuICAgICAgICAgIGBFcnJvciBjb25zdHJ1Y3RpbmcgQXJyYXkxRC4gU2hhcGUgb2YgdmFsdWVzICR7aW5mZXJyZWRTaGFwZX0gaXMgYCArXG4gICAgICAgICAgICAgIGBub3QgMSBkaW1lbnNpb25hbC5gKTtcbiAgICB9XG4gICAgcmV0dXJuIG5ldyBBcnJheTFEKHt2YWx1ZXM6IHRvVHlwZWRBcnJheSh2YWx1ZXMpfSk7XG4gIH1cblxuICBnZXQoaTogbnVtYmVyKTogbnVtYmVyIHtcbiAgICByZXR1cm4gdGhpcy5nZXRWYWx1ZXMoKVtpXTtcbiAgfVxuXG4gIHNldCh2YWx1ZTogbnVtYmVyLCBpOiBudW1iZXIpIHtcbiAgICB0aGlzLmdldFZhbHVlcygpW2ldID0gdmFsdWU7XG4gIH1cblxuICBhZGQodmFsdWU6IG51bWJlciwgaTogbnVtYmVyKSB7XG4gICAgdGhpcy5nZXRWYWx1ZXMoKVtpXSArPSB2YWx1ZTtcbiAgfVxuXG4gIGxvY1RvSW5kZXgobG9jOiBbbnVtYmVyXSk6IG51bWJlciB7XG4gICAgcmV0dXJuIGxvY1swXTtcbiAgfVxuXG4gIGluZGV4VG9Mb2MoaW5kZXg6IG51bWJlcik6IFtudW1iZXJdIHtcbiAgICByZXR1cm4gW2luZGV4XTtcbiAgfVxuXG4gIHN0YXRpYyB6ZXJvcyhzaGFwZTogW251bWJlcl0pOiBBcnJheTFEIHtcbiAgICByZXR1cm4gTkRBcnJheS56ZXJvczxBcnJheTFEPihzaGFwZSk7XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIEFycmF5MkQgZXh0ZW5kcyBOREFycmF5IHtcbiAgc2hhcGU6IFtudW1iZXIsIG51bWJlcl07XG5cbiAgcHJpdmF0ZSBzdHJpZGUwOiBudW1iZXI7XG5cbiAgY29uc3RydWN0b3Ioc2hhcGU6IFtudW1iZXIsIG51bWJlcl0sIGRhdGE6IE5EQXJyYXlEYXRhKSB7XG4gICAgdXRpbC5hc3NlcnQoc2hhcGUubGVuZ3RoID09PSAyLCAnU2hhcGUgc2hvdWxkIGJlIG9mIGxlbmd0aCAyJyk7XG4gICAgc3VwZXIoc2hhcGUsIGRhdGEpO1xuICAgIHRoaXMuc3RyaWRlMCA9IHRoaXMuc3RyaWRlc1swXTtcbiAgfVxuXG4gIHN0YXRpYyBuZXcoXG4gICAgICBzaGFwZTogW251bWJlciwgbnVtYmVyXSwgdmFsdWVzOiBGbG9hdDMyQXJyYXl8bnVtYmVyW118bnVtYmVyW11bXSkge1xuICAgIGlmICghKHZhbHVlcyBpbnN0YW5jZW9mIEZsb2F0MzJBcnJheSkpIHtcbiAgICAgIGNvbnN0IGluZmVycmVkU2hhcGUgPSB1dGlsLmluZmVyU2hhcGUodmFsdWVzKTtcbiAgICAgIGlmIChpbmZlcnJlZFNoYXBlLmxlbmd0aCA+IDEpIHtcbiAgICAgICAgdXRpbC5hc3NlcnRTaGFwZXNNYXRjaChcbiAgICAgICAgICAgIHNoYXBlLCBpbmZlcnJlZFNoYXBlLFxuICAgICAgICAgICAgYEVycm9yIHdoZW4gY29uc3RydWN0aW5nIEFycmF5MkQuIFNoYXBlIG9mIHZhbHVlcyBgICtcbiAgICAgICAgICAgICAgICBgJHtpbmZlcnJlZFNoYXBlfSBkb2VzIG5vdCBtYXRjaCB0aGUgcHJvdmlkZWQgc2hhcGUgYCArXG4gICAgICAgICAgICAgICAgYCR7c2hhcGV9LiBgKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIG5ldyBBcnJheTJEKHNoYXBlLCB7dmFsdWVzOiB0b1R5cGVkQXJyYXkodmFsdWVzKX0pO1xuICB9XG5cbiAgZ2V0KGk6IG51bWJlciwgajogbnVtYmVyKSB7XG4gICAgcmV0dXJuIHRoaXMuZ2V0VmFsdWVzKClbdGhpcy5zdHJpZGUwICogaSArIGpdO1xuICB9XG5cbiAgc2V0KHZhbHVlOiBudW1iZXIsIGk6IG51bWJlciwgajogbnVtYmVyKSB7XG4gICAgdGhpcy5nZXRWYWx1ZXMoKVt0aGlzLnN0cmlkZTAgKiBpICsgal0gPSB2YWx1ZTtcbiAgfVxuXG4gIGFkZCh2YWx1ZTogbnVtYmVyLCBpOiBudW1iZXIsIGo6IG51bWJlcikge1xuICAgIHRoaXMuZ2V0VmFsdWVzKClbdGhpcy5zdHJpZGUwICogaSArIGpdICs9IHZhbHVlO1xuICB9XG5cbiAgbG9jVG9JbmRleChsb2NzOiBbbnVtYmVyLCBudW1iZXJdKTogbnVtYmVyIHtcbiAgICByZXR1cm4gdGhpcy5zdHJpZGUwICogbG9jc1swXSArIGxvY3NbMV07XG4gIH1cblxuICBpbmRleFRvTG9jKGluZGV4OiBudW1iZXIpOiBbbnVtYmVyLCBudW1iZXJdIHtcbiAgICByZXR1cm4gW01hdGguZmxvb3IoaW5kZXggLyB0aGlzLnN0cmlkZTApLCBpbmRleCAlIHRoaXMuc3RyaWRlMF07XG4gIH1cblxuICBzdGF0aWMgemVyb3Moc2hhcGU6IFtudW1iZXIsIG51bWJlcl0pOiBBcnJheTJEIHtcbiAgICByZXR1cm4gTkRBcnJheS56ZXJvczxBcnJheTJEPihzaGFwZSk7XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIEFycmF5M0QgZXh0ZW5kcyBOREFycmF5IHtcbiAgc2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgcHJpdmF0ZSBzdHJpZGUwOiBudW1iZXI7XG4gIHByaXZhdGUgc3RyaWRlMTogbnVtYmVyO1xuXG4gIGNvbnN0cnVjdG9yKHNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGRhdGE6IE5EQXJyYXlEYXRhKSB7XG4gICAgdXRpbC5hc3NlcnQoc2hhcGUubGVuZ3RoID09PSAzLCAnU2hhcGUgc2hvdWxkIGJlIG9mIGxlbmd0aCAzJyk7XG4gICAgc3VwZXIoc2hhcGUsIGRhdGEpO1xuICAgIHRoaXMuc3RyaWRlMCA9IHRoaXMuc3RyaWRlc1swXTtcbiAgICB0aGlzLnN0cmlkZTEgPSB0aGlzLnN0cmlkZXNbMV07XG4gIH1cblxuICBzdGF0aWMgbmV3KFxuICAgICAgc2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSxcbiAgICAgIHZhbHVlczogRmxvYXQzMkFycmF5fG51bWJlcltdfG51bWJlcltdW11bXSkge1xuICAgIGlmICghKHZhbHVlcyBpbnN0YW5jZW9mIEZsb2F0MzJBcnJheSkpIHtcbiAgICAgIGNvbnN0IGluZmVycmVkU2hhcGUgPSB1dGlsLmluZmVyU2hhcGUodmFsdWVzKTtcbiAgICAgIGlmIChpbmZlcnJlZFNoYXBlLmxlbmd0aCA+IDEpIHtcbiAgICAgICAgdXRpbC5hc3NlcnRTaGFwZXNNYXRjaChcbiAgICAgICAgICAgIHNoYXBlLCBpbmZlcnJlZFNoYXBlLFxuICAgICAgICAgICAgYEVycm9yIHdoZW4gY29uc3RydWN0aW5nIEFycmF5M0QuIFNoYXBlIG9mIHZhbHVlcyBgICtcbiAgICAgICAgICAgICAgICBgJHtpbmZlcnJlZFNoYXBlfSBkb2VzIG5vdCBtYXRjaCB0aGUgcHJvdmlkZWQgc2hhcGUgYCArXG4gICAgICAgICAgICAgICAgYCR7c2hhcGV9LiBgKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIG5ldyBBcnJheTNEKHNoYXBlLCB7dmFsdWVzOiB0b1R5cGVkQXJyYXkodmFsdWVzKX0pO1xuICB9XG5cbiAgZ2V0KGk6IG51bWJlciwgajogbnVtYmVyLCBrOiBudW1iZXIpIHtcbiAgICByZXR1cm4gdGhpcy5nZXRWYWx1ZXMoKVt0aGlzLnN0cmlkZTAgKiBpICsgdGhpcy5zdHJpZGUxICogaiArIGtdO1xuICB9XG5cbiAgc2V0KHZhbHVlOiBudW1iZXIsIGk6IG51bWJlciwgajogbnVtYmVyLCBrOiBudW1iZXIpIHtcbiAgICB0aGlzLmdldFZhbHVlcygpW3RoaXMuc3RyaWRlMCAqIGkgKyB0aGlzLnN0cmlkZTEgKiBqICsga10gPSB2YWx1ZTtcbiAgfVxuXG4gIGFkZCh2YWx1ZTogbnVtYmVyLCBpOiBudW1iZXIsIGo6IG51bWJlciwgazogbnVtYmVyKSB7XG4gICAgdGhpcy5nZXRWYWx1ZXMoKVt0aGlzLnN0cmlkZTAgKiBpICsgdGhpcy5zdHJpZGUxICogaiArIGtdICs9IHZhbHVlO1xuICB9XG5cbiAgbG9jVG9JbmRleChsb2NzOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0pOiBudW1iZXIge1xuICAgIHJldHVybiB0aGlzLnN0cmlkZTAgKiBsb2NzWzBdICsgdGhpcy5zdHJpZGUxICogbG9jc1sxXSArIGxvY3NbMl07XG4gIH1cblxuICBpbmRleFRvTG9jKGluZGV4OiBudW1iZXIpOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0ge1xuICAgIGNvbnN0IGkgPSBNYXRoLmZsb29yKGluZGV4IC8gdGhpcy5zdHJpZGUwKTtcbiAgICBpbmRleCAtPSBpICogdGhpcy5zdHJpZGUwO1xuICAgIHJldHVybiBbaSwgTWF0aC5mbG9vcihpbmRleCAvIHRoaXMuc3RyaWRlMSksIGluZGV4ICUgdGhpcy5zdHJpZGUxXTtcbiAgfVxuXG4gIHN0YXRpYyB6ZXJvcyhzaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdKTogQXJyYXkzRCB7XG4gICAgcmV0dXJuIE5EQXJyYXkuemVyb3M8QXJyYXkzRD4oc2hhcGUpO1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBBcnJheTREIGV4dGVuZHMgTkRBcnJheSB7XG4gIHNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgcHJpdmF0ZSBzdHJpZGUwOiBudW1iZXI7XG4gIHByaXZhdGUgc3RyaWRlMTogbnVtYmVyO1xuICBwcml2YXRlIHN0cmlkZTI6IG51bWJlcjtcblxuICBjb25zdHJ1Y3RvcihzaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGRhdGE6IE5EQXJyYXlEYXRhKSB7XG4gICAgdXRpbC5hc3NlcnQoc2hhcGUubGVuZ3RoID09PSA0LCAnU2hhcGUgc2hvdWxkIGJlIG9mIGxlbmd0aCA0Jyk7XG4gICAgc3VwZXIoc2hhcGUsIGRhdGEpO1xuICAgIHRoaXMuc3RyaWRlMCA9IHRoaXMuc3RyaWRlc1swXTtcbiAgICB0aGlzLnN0cmlkZTEgPSB0aGlzLnN0cmlkZXNbMV07XG4gICAgdGhpcy5zdHJpZGUyID0gdGhpcy5zdHJpZGVzWzJdO1xuICB9XG5cbiAgc3RhdGljIG5ldyhcbiAgICAgIHNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSxcbiAgICAgIHZhbHVlczogRmxvYXQzMkFycmF5fG51bWJlcltdfG51bWJlcltdW11bXVtdKSB7XG4gICAgaWYgKCEodmFsdWVzIGluc3RhbmNlb2YgRmxvYXQzMkFycmF5KSkge1xuICAgICAgY29uc3QgaW5mZXJyZWRTaGFwZSA9IHV0aWwuaW5mZXJTaGFwZSh2YWx1ZXMpO1xuICAgICAgaWYgKGluZmVycmVkU2hhcGUubGVuZ3RoID4gMSkge1xuICAgICAgICB1dGlsLmFzc2VydFNoYXBlc01hdGNoKFxuICAgICAgICAgICAgc2hhcGUsIGluZmVycmVkU2hhcGUsXG4gICAgICAgICAgICBgRXJyb3Igd2hlbiBjb25zdHJ1Y3RpbmcgQXJyYXk0RC4gU2hhcGUgb2YgdmFsdWVzIGAgK1xuICAgICAgICAgICAgICAgIGAke2luZmVycmVkU2hhcGV9IGRvZXMgbm90IG1hdGNoIHRoZSBwcm92aWRlZCBzaGFwZSBgICtcbiAgICAgICAgICAgICAgICBgJHtzaGFwZX0uIGApO1xuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gbmV3IEFycmF5NEQoc2hhcGUsIHt2YWx1ZXM6IHRvVHlwZWRBcnJheSh2YWx1ZXMpfSk7XG4gIH1cblxuICBnZXQoaTogbnVtYmVyLCBqOiBudW1iZXIsIGs6IG51bWJlciwgbDogbnVtYmVyKSB7XG4gICAgcmV0dXJuIHRoaXMuZ2V0VmFsdWVzKClcbiAgICAgICAgW3RoaXMuc3RyaWRlMCAqIGkgKyB0aGlzLnN0cmlkZTEgKiBqICsgdGhpcy5zdHJpZGUyICogayArIGxdO1xuICB9XG5cbiAgc2V0KHZhbHVlOiBudW1iZXIsIGk6IG51bWJlciwgajogbnVtYmVyLCBrOiBudW1iZXIsIGw6IG51bWJlcikge1xuICAgIHRoaXMuZ2V0VmFsdWVzKClcbiAgICAgICAgW3RoaXMuc3RyaWRlMCAqIGkgKyB0aGlzLnN0cmlkZTEgKiBqICsgdGhpcy5zdHJpZGUyICogayArIGxdID0gdmFsdWU7XG4gIH1cblxuICBhZGQodmFsdWU6IG51bWJlciwgaTogbnVtYmVyLCBqOiBudW1iZXIsIGs6IG51bWJlciwgbDogbnVtYmVyKSB7XG4gICAgdGhpcy5nZXRWYWx1ZXMoKVxuICAgICAgICBbdGhpcy5zdHJpZGUwICogaSArIHRoaXMuc3RyaWRlMSAqIGogKyB0aGlzLnN0cmlkZTIgKiBrICsgbF0gKz0gdmFsdWU7XG4gIH1cblxuICBsb2NUb0luZGV4KGxvY3M6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdKTogbnVtYmVyIHtcbiAgICByZXR1cm4gdGhpcy5zdHJpZGUwICogbG9jc1swXSArIHRoaXMuc3RyaWRlMSAqIGxvY3NbMV0gK1xuICAgICAgICB0aGlzLnN0cmlkZTIgKiBsb2NzWzJdICsgbG9jc1szXTtcbiAgfVxuXG4gIGluZGV4VG9Mb2MoaW5kZXg6IG51bWJlcik6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdIHtcbiAgICBjb25zdCBpID0gTWF0aC5mbG9vcihpbmRleCAvIHRoaXMuc3RyaWRlMCk7XG4gICAgaW5kZXggLT0gaSAqIHRoaXMuc3RyaWRlMDtcbiAgICBjb25zdCBqID0gTWF0aC5mbG9vcihpbmRleCAvIHRoaXMuc3RyaWRlMSk7XG4gICAgaW5kZXggLT0gaiAqIHRoaXMuc3RyaWRlMTtcbiAgICByZXR1cm4gW2ksIGosIE1hdGguZmxvb3IoaW5kZXggLyB0aGlzLnN0cmlkZTIpLCBpbmRleCAlIHRoaXMuc3RyaWRlMl07XG4gIH1cblxuICBzdGF0aWMgemVyb3Moc2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdKTogQXJyYXk0RCB7XG4gICAgcmV0dXJuIE5EQXJyYXkuemVyb3M8QXJyYXk0RD4oc2hhcGUpO1xuICB9XG59XG5cbnR5cGUgQXJyYXlEYXRhID0gRmxvYXQzMkFycmF5fG51bWJlcltdfG51bWJlcltdW118bnVtYmVyW11bXVtdfG51bWJlcltdW11bXVtdO1xuXG5mdW5jdGlvbiB0b1R5cGVkQXJyYXkoYTogQXJyYXlEYXRhKTogRmxvYXQzMkFycmF5IHtcbiAgcmV0dXJuIChhIGluc3RhbmNlb2YgRmxvYXQzMkFycmF5KSA/IGEgOiBuZXcgRmxvYXQzMkFycmF5KHV0aWwuZmxhdHRlbihhKSk7XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIGNvbnZfdXRpbCBmcm9tICcuLi9jb252X3V0aWwnO1xuXG5pbXBvcnQgKiBhcyBjb252X2dwdSBmcm9tICcuL2NvbnZfZ3B1JztcbmltcG9ydCB7R1BHUFVDb250ZXh0fSBmcm9tICcuL2dwZ3B1X2NvbnRleHQnO1xuXG5leHBvcnQgZnVuY3Rpb24gZ2V0RnJhZ21lbnRTaGFkZXJEZXJXZWlnaHRzU291cmNlKFxuICAgIHhTaGFwZVJvd0NvbERlcHRoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGZTaXplOiBudW1iZXIsXG4gICAgb3V0cHV0RGVwdGg6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsIHplcm9QYWQ6IG51bWJlcikge1xuICBjb25zdCBnZXRNYXRyaXhWYWx1ZU9yWmVyb1BhZCA9XG4gICAgICBjb252X2dwdS5nZXRGcmFnbWVudFNoYWRlckdldE1hdHJpeFZhbHVlT3JaZXJvUGFkU291cmNlKCk7XG4gIGNvbnN0IGlucHV0RGVwdGggPSB4U2hhcGVSb3dDb2xEZXB0aFsyXTtcblxuICBjb25zdCB4VGV4U2hhcGVSQyA9IGNvbnZfdXRpbC5jb21wdXRlVGV4U2hhcGVGcm9tM0QoeFNoYXBlUm93Q29sRGVwdGgpO1xuXG4gIGNvbnN0IHlTaGFwZSA9IGNvbnZfdXRpbC5jb21wdXRlT3V0cHV0U2hhcGUzRChcbiAgICAgIHhTaGFwZVJvd0NvbERlcHRoLCBmU2l6ZSwgb3V0cHV0RGVwdGgsIHN0cmlkZSwgemVyb1BhZCk7XG4gIGNvbnN0IHlOdW1Sb3dzID0geVNoYXBlWzBdO1xuICBjb25zdCB5TnVtQ29scyA9IHlTaGFwZVsxXTtcbiAgY29uc3QgeVRleFNoYXBlUkMgPSBjb252X3V0aWwuY29tcHV0ZVRleFNoYXBlRnJvbTNEKHlTaGFwZSk7XG5cbiAgY29uc3QgZlNpemVUaW1lc0lucHV0RGVwdGggPSBmU2l6ZSAqIGlucHV0RGVwdGg7XG5cbiAgY29uc3QgcHJvbG9ndWUgPSBgXG4gICAgcHJlY2lzaW9uIGhpZ2hwIGZsb2F0O1xuICAgIHVuaWZvcm0gc2FtcGxlcjJEIHg7XG4gICAgdW5pZm9ybSBzYW1wbGVyMkQgZHk7XG4gIGA7XG5cbiAgcmV0dXJuIHByb2xvZ3VlICsgJ1xcbicgKyBnZXRNYXRyaXhWYWx1ZU9yWmVyb1BhZCArICdcXG4nICtcbiAgICAgIGBcbiAgICBjb25zdCB2ZWMyIGhhbGZDUiA9IHZlYzIoMC41LCAwLjUpO1xuICAgIGNvbnN0IHZlYzIgeFNoYXBlQ1IgPSB2ZWMyKCR7eFRleFNoYXBlUkNbMV19LCAke3hUZXhTaGFwZVJDWzBdfSk7XG4gICAgY29uc3QgdmVjMiBkeVNoYXBlQ1IgPSB2ZWMyKCR7eVRleFNoYXBlUkNbMV19LCAke3lUZXhTaGFwZVJDWzBdfSk7XG5cbiAgICB2b2lkIG1haW4oKSB7XG4gICAgICB2ZWMyIHdUZXhDUiA9IGZsb29yKGdsX0ZyYWdDb29yZC54eSk7XG5cbiAgICAgIC8vIE1hcCBmcm9tIDJEICh3VGV4Uiwgd1RleEMpIHRvIDREICh3Uiwgd0MsIGQxLCBkMikuXG4gICAgICBmbG9hdCB3UiA9IGZsb29yKHdUZXhDUi55IC8gJHtmU2l6ZVRpbWVzSW5wdXREZXB0aH0uMCk7XG4gICAgICBmbG9hdCB3VGV4UkxlZnRvdmVyID0gd1RleENSLnkgLSB3UiAqICR7ZlNpemVUaW1lc0lucHV0RGVwdGh9LjA7XG4gICAgICBmbG9hdCB3QyA9IGZsb29yKHdUZXhSTGVmdG92ZXIgLyAke2lucHV0RGVwdGh9LjApO1xuICAgICAgZmxvYXQgZDEgPSBtb2Qod1RleFJMZWZ0b3ZlciwgJHtpbnB1dERlcHRofS4wKTtcbiAgICAgIGZsb2F0IGQyID0gd1RleENSLng7XG5cbiAgICAgIC8vIENvbnZvbHZlIHgoPywgPywgZDEpIHdpdGggZHkoOiwgOiwgZDIpIHRvIGdldCBkdyh3Uiwgd0MsIGQxLCBkMikuXG4gICAgICAvLyA/ID0gdG8gYmUgZGV0ZXJtaW5lZC4gOiA9IGFjcm9zcyBhbGwgdmFsdWVzIGluIHRoYXQgYXhpcy5cbiAgICAgIGZsb2F0IGRvdFByb2QgPSAwLjA7XG4gICAgICBmb3IgKGZsb2F0IHlSID0gMC4wOyB5UiA8ICR7eU51bVJvd3N9LjA7IHlSICs9IDEuMCkge1xuICAgICAgICBmbG9hdCB4UiA9IHdSICsgeVIgKiAke3N0cmlkZX0uMCAtICR7emVyb1BhZH0uMDtcbiAgICAgICAgZmxvYXQgeFRleFIgPSB4UjtcbiAgICAgICAgZmxvYXQgeVRleFIgPSB5UjtcbiAgICAgICAgZm9yIChmbG9hdCB5QyA9IDAuMDsgeUMgPCAke3lOdW1Db2xzfS4wOyB5QyArPSAxLjApIHtcbiAgICAgICAgICBmbG9hdCB4QyA9IHdDICsgeUMgKiAke3N0cmlkZX0uMCAtICR7emVyb1BhZH0uMDtcblxuICAgICAgICAgIC8vIE1hcCBmcm9tIDNEICh4UiwgeEMsIGQxKSB0byAyRCAoeFRleFIsIHhUZXhDKS5cbiAgICAgICAgICAvLyBNYXAgZnJvbSAzRCAoeVIsIHlDLCBkMikgdG8gMkQgKHlUZXhSLCB5VGV4QykuXG4gICAgICAgICAgdmVjMiB4eVRleEMgPSB2ZWMyKHhDLCB5QykgKiB2ZWMyKCR7aW5wdXREZXB0aH0uMCwgJHtvdXRwdXREZXB0aH0uMCkgK1xuICAgICAgICAgICAgICAgICAgICAgICAgdmVjMihkMSwgZDIpO1xuICAgICAgICAgIGZsb2F0IHhUZXhDID0geHlUZXhDLng7XG4gICAgICAgICAgZmxvYXQgeVRleEMgPSB4eVRleEMueTtcblxuICAgICAgICAgIC8vIFJlYWQgZHkoeVIsIHlDLCBkMikuXG4gICAgICAgICAgdmVjMiBkeVVWID0gKHZlYzIoeVRleEMsIHlUZXhSKSArIGhhbGZDUikgLyBkeVNoYXBlQ1I7XG4gICAgICAgICAgZmxvYXQgZHlWYWx1ZSA9IHRleHR1cmUyRChkeSwgZHlVVikucjtcblxuICAgICAgICAgIC8vIFJlYWQgeCh4UiwgeEMsIGQxKSAocG90ZW50aWFsbHkgemVyby1wYWRkZWQpLlxuICAgICAgICAgIGZsb2F0IHhWYWx1ZSA9XG4gICAgICAgICAgICBnZXRNYXRyaXhWYWx1ZU9yWmVyb1BhZCh4LCB4U2hhcGVDUiwgdmVjMih4VGV4QywgeFRleFIpKTtcblxuICAgICAgICAgIGRvdFByb2QgKz0gKHhWYWx1ZSAqIGR5VmFsdWUpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICBnbF9GcmFnQ29sb3IgPSB2ZWM0KGRvdFByb2QsIDAsIDAsIDApO1xuICAgIH1gO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0RnJhZ21lbnRTaGFkZXJDb252VHJhbnNwb3NlU291cmNlKFxuICAgIHhTaGFwZVJDRDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCBmU2l6ZTogbnVtYmVyLCBvcmlnSW5wdXREZXB0aDogbnVtYmVyLFxuICAgIG9yaWdTdHJpZGU6IG51bWJlciwgb3JpZ1BhZDogbnVtYmVyLCBoYXNCaWFzOiBib29sZWFuKSB7XG4gIGNvbnN0IHBhZCA9IGZTaXplIC0gMSAtIG9yaWdQYWQ7XG4gIGNvbnN0IFt4Um93cywgeENvbHMsIG9yaWdPdXRwdXREZXB0aF0gPSB4U2hhcGVSQ0Q7XG5cbiAgY29uc3QgeFRleFNoYXBlUkMgPSBjb252X3V0aWwuY29tcHV0ZVRleFNoYXBlRnJvbTNEKHhTaGFwZVJDRCk7XG4gIGNvbnN0IHdUZXhTaGFwZVJDID1cbiAgICAgIGNvbnZfdXRpbC5jb21wdXRlV2VpZ2h0c1RleFNoYXBlKG9yaWdJbnB1dERlcHRoLCBvcmlnT3V0cHV0RGVwdGgsIGZTaXplKTtcblxuICBjb25zdCBnZXRCaWFzVmFsdWUgPSBoYXNCaWFzID9cbiAgICAgIGNvbnZfZ3B1LmdldEZyYWdtZW50U2hhZGVyR2V0Qmlhc1ZhbHVlU291cmNlKG9yaWdJbnB1dERlcHRoKSA6XG4gICAgICAnJztcbiAgY29uc3QgYmlhc1Byb2xvZ3VlID0gaGFzQmlhcyA/ICd1bmlmb3JtIHNhbXBsZXIyRCBiaWFzZXM7JyA6ICcnO1xuICBjb25zdCBiaWFzT3BlcmF0aW9uID0gaGFzQmlhcyA/ICdkb3RQcm9kICs9IGdldEJpYXNWYWx1ZShiaWFzZXMsIGQyKTsnIDogJyc7XG5cbiAgY29uc3QgcHJvbG9ndWUgPSBgXG4gICAgcHJlY2lzaW9uIGhpZ2hwIGZsb2F0O1xuICAgIHVuaWZvcm0gc2FtcGxlcjJEIHg7XG4gICAgdW5pZm9ybSBzYW1wbGVyMkQgd2VpZ2h0cztcbiAgICAke2JpYXNQcm9sb2d1ZX1cbiAgICBgO1xuXG4gIHJldHVybiBwcm9sb2d1ZSArICdcXG4nICsgZ2V0Qmlhc1ZhbHVlICsgJ1xcbicgK1xuICAgICAgYFxuICAgIGNvbnN0IHZlYzIgaGFsZkNSID0gdmVjMigwLjUsIDAuNSk7XG4gICAgY29uc3QgdmVjMiB4U2hhcGVDUiA9IHZlYzIoJHt4VGV4U2hhcGVSQ1sxXX0sICR7eFRleFNoYXBlUkNbMF19KTtcbiAgICBjb25zdCB2ZWMyIHdTaGFwZUNSID0gdmVjMigke3dUZXhTaGFwZVJDWzFdfSwgJHt3VGV4U2hhcGVSQ1swXX0pO1xuXG4gICAgdm9pZCBtYWluKCkge1xuICAgICAgdmVjMiB5VGV4Q1IgPSBmbG9vcihnbF9GcmFnQ29vcmQueHkpO1xuXG4gICAgICAvLyBNYXAgZnJvbSAyRCAoeVRleFIsIHlUZXhDKSB0byAzRCAoeVIsIHlDLCBkMikuXG4gICAgICBmbG9hdCB5UiA9IHlUZXhDUi55O1xuICAgICAgZmxvYXQgeUMgPSBmbG9vcih5VGV4Q1IueCAvICR7b3JpZ0lucHV0RGVwdGh9LjApO1xuICAgICAgZmxvYXQgZDIgPSBtb2QoeVRleENSLngsICR7b3JpZ0lucHV0RGVwdGh9LjApO1xuXG4gICAgICB2ZWMyIHhSQ0Nvcm5lciA9IHZlYzIoeVIsIHlDKSAtIHZlYzIoJHtwYWR9LjAsICR7cGFkfS4wKTtcbiAgICAgIGZsb2F0IHhSQ29ybmVyID0geFJDQ29ybmVyLng7XG4gICAgICBmbG9hdCB4Q0Nvcm5lciA9IHhSQ0Nvcm5lci55O1xuXG4gICAgICAvLyBDb252b2x2ZSB4KD8sID8sIGQxKSB3aXRoIHcoOiwgOiwgZDIsIGQxKSB0byBnZXQgeSh5UiwgeUMsIGQyKS5cbiAgICAgIC8vID8gPSB0byBiZSBkZXRlcm1pbmVkLiA6ID0gYWNyb3NzIGFsbCB2YWx1ZXMgaW4gdGhhdCBheGlzLlxuICAgICAgZmxvYXQgZG90UHJvZCA9IDAuMDtcbiAgICAgIGZvciAoZmxvYXQgd1IgPSAwLjA7IHdSIDwgJHtmU2l6ZX0uMDsgd1IgKz0gMS4wKSB7XG5cbiAgICAgICAgZmxvYXQgeFIgPSAoeFJDb3JuZXIgKyB3UikgLyAke29yaWdTdHJpZGV9LjA7XG4gICAgICAgIC8vIFRPRE8oc21pbGtvdik6IFNwbGljZSB0aGlzIHdpdGggYW5vdGhlciB2ZXJzaW9uIHdoZXJlIHlvdSBjYWxsXG4gICAgICAgIC8vIGdldE1hdHJpeFZhbHVlT3JaZXJvUGFkKCkuIEhlcmUgYW5kIGJlbG93LlxuICAgICAgICBpZiAoeFIgPCAwLjAgfHwgeFIgPj0gJHt4Um93c30uMCB8fCBmcmFjdCh4UikgPiAwLjApIHtcbiAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgfVxuXG4gICAgICAgIGZsb2F0IHdSUGVybSA9ICR7ZlNpemV9LjAgLSAxLjAgLSB3UjtcbiAgICAgICAgZmxvYXQgeFRleFIgPSB4UjtcblxuICAgICAgICBmb3IgKGZsb2F0IHdDID0gMC4wOyB3QyA8ICR7ZlNpemV9LjA7IHdDICs9IDEuMCkge1xuXG4gICAgICAgICAgZmxvYXQgeEMgPSAoeENDb3JuZXIgKyB3QykgLyAke29yaWdTdHJpZGV9LjA7XG4gICAgICAgICAgaWYgKHhDIDwgMC4wIHx8IHhDID49ICR7eENvbHN9LjAgfHwgZnJhY3QoeEMpID4gMC4wKSB7XG4gICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICB9XG5cbiAgICAgICAgICBmbG9hdCB3Q1Blcm0gPSAke2ZTaXplfS4wIC0gMS4wIC0gd0M7XG4gICAgICAgICAgZmxvYXQgd1RleFIgPSB3UlBlcm0gKiAke2ZTaXplfS4wICogJHtvcmlnSW5wdXREZXB0aH0uMCArXG4gICAgICAgICAgICAgICAgICAgICAgICB3Q1Blcm0gKiAke29yaWdJbnB1dERlcHRofS4wICsgZDI7XG5cbiAgICAgICAgICBmb3IgKGZsb2F0IGQxID0gMC4wOyBkMSA8ICR7b3JpZ091dHB1dERlcHRofS4wOyBkMSArPSAxLjApIHtcbiAgICAgICAgICAgIGZsb2F0IHhUZXhDID0geEMgKiAke29yaWdPdXRwdXREZXB0aH0uMCArIGQxO1xuICAgICAgICAgICAgZmxvYXQgd1RleEMgPSBkMTtcblxuICAgICAgICAgICAgLy8gUmVhZCB4KHhSLCB4QywgZDEpLlxuICAgICAgICAgICAgdmVjMiB4VVYgPSAodmVjMih4VGV4QywgeFRleFIpICsgaGFsZkNSKSAvIHhTaGFwZUNSO1xuICAgICAgICAgICAgZmxvYXQgeFZhbHVlID0gdGV4dHVyZTJEKHgsIHhVVikucjtcblxuICAgICAgICAgICAgLy8gUmVhZCB3KHdSUGVybSwgd0NQZXJtLCBkMiwgZDEpLlxuICAgICAgICAgICAgdmVjMiB3VVYgPSAodmVjMih3VGV4Qywgd1RleFIpICsgaGFsZkNSKSAvIHdTaGFwZUNSO1xuICAgICAgICAgICAgZmxvYXQgd1ZhbHVlID0gdGV4dHVyZTJEKHdlaWdodHMsIHdVVikucjtcblxuICAgICAgICAgICAgZG90UHJvZCArPSB4VmFsdWUgKiB3VmFsdWU7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgICAke2JpYXNPcGVyYXRpb259XG4gICAgICBnbF9GcmFnQ29sb3IgPSB2ZWM0KGRvdFByb2QsIDAsIDAsIDApO1xuICAgIH1gO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0RnJhZ21lbnRTaGFkZXJEZXJCaWFzU291cmNlKFxuICAgIGR5U2hhcGVSQ0Q6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSkge1xuICBjb25zdCBkeVRleFNoYXBlUkMgPSBjb252X3V0aWwuY29tcHV0ZVRleFNoYXBlRnJvbTNEKGR5U2hhcGVSQ0QpO1xuICBjb25zdCBbeU51bVJvd3MsIHlOdW1Db2xzLCBvdXRwdXREZXB0aF0gPSBkeVNoYXBlUkNEO1xuXG4gIHJldHVybiBgXG4gICAgcHJlY2lzaW9uIGhpZ2hwIGZsb2F0O1xuICAgIHVuaWZvcm0gc2FtcGxlcjJEIGR5O1xuXG4gICAgY29uc3QgdmVjMiBoYWxmQ1IgPSB2ZWMyKDAuNSwgMC41KTtcbiAgICBjb25zdCB2ZWMyIGR5U2hhcGVDUiA9IHZlYzIoJHtkeVRleFNoYXBlUkNbMV19LCAke2R5VGV4U2hhcGVSQ1swXX0pO1xuXG4gICAgdm9pZCBtYWluKCkge1xuICAgICAgdmVjMiBiaWFzVGV4Q1IgPSBmbG9vcihnbF9GcmFnQ29vcmQueHkpO1xuXG4gICAgICAvLyBUaGUgYmlhcyB0ZXh0dXJlIFJDIHNoYXBlIGlzIFsxLCBkMl0uXG4gICAgICBmbG9hdCBkMiA9IGJpYXNUZXhDUi54O1xuXG4gICAgICBmbG9hdCBkZXJCaWFzID0gMC4wO1xuICAgICAgZm9yIChmbG9hdCB5UiA9IDAuMDsgeVIgPCAke3lOdW1Sb3dzfS4wOyB5UiArPSAxLjApIHtcbiAgICAgICAgZmxvYXQgeVRleFIgPSB5UjtcblxuICAgICAgICBmb3IgKGZsb2F0IHlDID0gMC4wOyB5QyA8ICR7eU51bUNvbHN9LjA7IHlDICs9IDEuMCkge1xuICAgICAgICAgIC8vIE1hcCBmcm9tIDNEICh5UiwgeUMsIGQyKSB0byAyRCAoeVRleFIsIHlUZXhDKS5cbiAgICAgICAgICBmbG9hdCB5VGV4QyA9IHlDICogJHtvdXRwdXREZXB0aH0uMCArIGQyO1xuXG4gICAgICAgICAgLy8gUmVhZCBkeSh5UiwgeUMsIGQyKS5cbiAgICAgICAgICB2ZWMyIGR5VVYgPSAodmVjMih5VGV4QywgeVRleFIpICsgaGFsZkNSKSAvIGR5U2hhcGVDUjtcbiAgICAgICAgICBmbG9hdCBkeVZhbHVlID0gdGV4dHVyZTJEKGR5LCBkeVVWKS5yO1xuXG4gICAgICAgICAgZGVyQmlhcyArPSBkeVZhbHVlO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICBnbF9GcmFnQ29sb3IgPSB2ZWM0KGRlckJpYXMsIDAsIDAsIDApO1xuICAgIH1gO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZGVyQmlhcyhcbiAgICBncGdwdTogR1BHUFVDb250ZXh0LCBwcm9ncmFtOiBXZWJHTFByb2dyYW0sIGR5VGV4OiBXZWJHTFRleHR1cmUsXG4gICAgcmVzdWx0OiBXZWJHTFRleHR1cmUsIHJlc3VsdFRleFNoYXBlUkM6IFtudW1iZXIsIG51bWJlcl0pIHtcbiAgZ3BncHUuc2V0T3V0cHV0TWF0cml4VGV4dHVyZShcbiAgICAgIHJlc3VsdCwgcmVzdWx0VGV4U2hhcGVSQ1swXSwgcmVzdWx0VGV4U2hhcGVSQ1sxXSk7XG4gIGdwZ3B1LnNldFByb2dyYW0ocHJvZ3JhbSk7XG4gIGdwZ3B1LnNldElucHV0TWF0cml4VGV4dHVyZShkeVRleCwgJ2R5JywgMCk7XG4gIGdwZ3B1LmV4ZWN1dGVQcm9ncmFtKCk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBkZXJXZWlnaHRzKFxuICAgIGdwZ3B1OiBHUEdQVUNvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSwgeFRleDogV2ViR0xUZXh0dXJlLFxuICAgIGR5VGV4OiBXZWJHTFRleHR1cmUsIHJlc3VsdDogV2ViR0xUZXh0dXJlLFxuICAgIHJlc3VsdFRleFNoYXBlUkM6IFtudW1iZXIsIG51bWJlcl0pIHtcbiAgZ3BncHUuc2V0T3V0cHV0TWF0cml4VGV4dHVyZShcbiAgICAgIHJlc3VsdCwgcmVzdWx0VGV4U2hhcGVSQ1swXSwgcmVzdWx0VGV4U2hhcGVSQ1sxXSk7XG4gIGdwZ3B1LnNldFByb2dyYW0ocHJvZ3JhbSk7XG4gIGdwZ3B1LnNldElucHV0TWF0cml4VGV4dHVyZSh4VGV4LCAneCcsIDApO1xuICBncGdwdS5zZXRJbnB1dE1hdHJpeFRleHR1cmUoZHlUZXgsICdkeScsIDEpO1xuICBncGdwdS5leGVjdXRlUHJvZ3JhbSgpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY29udlRyYW5zcG9zZShcbiAgICBncGdwdTogR1BHUFVDb250ZXh0LCBwcm9ncmFtOiBXZWJHTFByb2dyYW0sIHhUZXg6IFdlYkdMVGV4dHVyZSxcbiAgICB3ZWlnaHRzVGV4OiBXZWJHTFRleHR1cmUsIGJpYXNlc1RleDogV2ViR0xUZXh0dXJlfG51bGwsXG4gICAgcmVzdWx0VGV4OiBXZWJHTFRleHR1cmUsIHJlc3VsdFRleFNoYXBlUkM6IFtudW1iZXIsIG51bWJlcl0pIHtcbiAgZ3BncHUuc2V0T3V0cHV0TWF0cml4VGV4dHVyZShcbiAgICAgIHJlc3VsdFRleCwgcmVzdWx0VGV4U2hhcGVSQ1swXSwgcmVzdWx0VGV4U2hhcGVSQ1sxXSk7XG4gIGdwZ3B1LnNldFByb2dyYW0ocHJvZ3JhbSk7XG4gIGdwZ3B1LnNldElucHV0TWF0cml4VGV4dHVyZSh4VGV4LCAneCcsIDApO1xuICBncGdwdS5zZXRJbnB1dE1hdHJpeFRleHR1cmUod2VpZ2h0c1RleCwgJ3dlaWdodHMnLCAxKTtcbiAgaWYgKGJpYXNlc1RleCAhPSBudWxsKSB7XG4gICAgZ3BncHUuc2V0SW5wdXRNYXRyaXhUZXh0dXJlKGJpYXNlc1RleCwgJ2JpYXNlcycsIDIpO1xuICB9XG4gIGdwZ3B1LmV4ZWN1dGVQcm9ncmFtKCk7XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIGNvbnZfdXRpbCBmcm9tICcuLi9jb252X3V0aWwnO1xuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4vZ3BncHVfY29udGV4dCc7XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRGcmFnbWVudFNoYWRlclByb2xvZ3VlU291cmNlKCk6IHN0cmluZyB7XG4gIHJldHVybiBgXG4gICAgcHJlY2lzaW9uIGhpZ2hwIGZsb2F0O1xuICAgIHVuaWZvcm0gc2FtcGxlcjJEIHg7XG4gICAgdW5pZm9ybSBzYW1wbGVyMkQgd2VpZ2h0cztcbiAgICB1bmlmb3JtIHNhbXBsZXIyRCBiaWFzZXM7XG4gICAgdmFyeWluZyB2ZWMyIHJlc3VsdFVWO2A7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRGcmFnbWVudFNoYWRlckdldE1hdHJpeFZhbHVlT3JaZXJvUGFkU291cmNlKCk6IHN0cmluZyB7XG4gIHJldHVybiBgXG4gICAgZmxvYXQgZ2V0TWF0cml4VmFsdWVPclplcm9QYWQoaW4gc2FtcGxlcjJEIG1hdHJpeCwgdmVjMiBtYXRyaXhTaGFwZUNSLFxuICAgICAgICB2ZWMyIHJlcXVlc3RlZENSKSB7XG4gICAgICB2ZWMyIHV2ID0gKHJlcXVlc3RlZENSICsgdmVjMigwLjUsIDAuNSkpIC8gbWF0cml4U2hhcGVDUjtcbiAgICAgIGZsb2F0IHZhbHVlID0gdGV4dHVyZTJEKG1hdHJpeCwgdXYpLnI7XG4gICAgICBib29sIGxlc3NUaGFuWmVybyA9IGFueShsZXNzVGhhbih1diwgdmVjMigwLCAwKSkpO1xuICAgICAgYm9vbCBncmVhdGVyVGhhbk9uZSA9IGFueShncmVhdGVyVGhhbih1diwgdmVjMigxLCAxKSkpO1xuICAgICAgYm9vbCBvdXRzaWRlID0gbGVzc1RoYW5aZXJvIHx8IGdyZWF0ZXJUaGFuT25lO1xuICAgICAgcmV0dXJuIG1peCh2YWx1ZSwgMC4wLCBmbG9hdChvdXRzaWRlKSk7XG4gICAgfWA7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRGcmFnbWVudFNoYWRlckNvbnZvbHZlU291cmNlKFxuICAgIHhTaGFwZVJDRDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCBmU2l6ZTogbnVtYmVyLCBvdXRwdXREZXB0aDogbnVtYmVyLFxuICAgIHN0cmlkZTogbnVtYmVyLCBwYWQ6IG51bWJlciwgaGFzQmlhczogYm9vbGVhbikge1xuICBjb25zdCBbeFJvd3MsIHhDb2xzLCBpbnB1dERlcHRoXSA9IHhTaGFwZVJDRDtcblxuICBjb25zdCB4VGV4U2hhcGVSQyA9IGNvbnZfdXRpbC5jb21wdXRlVGV4U2hhcGVGcm9tM0QoeFNoYXBlUkNEKTtcbiAgY29uc3Qgd1RleFNoYXBlUkMgPVxuICAgICAgY29udl91dGlsLmNvbXB1dGVXZWlnaHRzVGV4U2hhcGUoaW5wdXREZXB0aCwgb3V0cHV0RGVwdGgsIGZTaXplKTtcblxuICByZXR1cm4gYFxuICAgIGNvbnN0IHZlYzIgaGFsZkNSID0gdmVjMigwLjUsIDAuNSk7XG4gICAgY29uc3QgdmVjMiB4U2hhcGVDUiA9IHZlYzIoJHt4VGV4U2hhcGVSQ1sxXX0sICR7eFRleFNoYXBlUkNbMF19KTtcbiAgICBjb25zdCB2ZWMyIHdTaGFwZUNSID0gdmVjMigke3dUZXhTaGFwZVJDWzFdfSwgJHt3VGV4U2hhcGVSQ1swXX0pO1xuXG4gICAgdm9pZCBtYWluKCkge1xuICAgICAgdmVjMiB5VGV4Q1IgPSBmbG9vcihnbF9GcmFnQ29vcmQueHkpO1xuXG4gICAgICAvLyBNYXAgZnJvbSAyRCAoeVRleFIsIHlUZXhDKSB0byAzRCAoeVIsIHlDLCBkMikuXG4gICAgICBmbG9hdCB5UiA9IHlUZXhDUi55O1xuICAgICAgZmxvYXQgeUMgPSBmbG9vcih5VGV4Q1IueCAvICR7b3V0cHV0RGVwdGh9LjApO1xuICAgICAgZmxvYXQgZDIgPSBtb2QoeVRleENSLngsICR7b3V0cHV0RGVwdGh9LjApO1xuICAgICAgZmxvYXQgd1RleEMgPSBkMjtcblxuICAgICAgdmVjMiB4UkNDb3JuZXIgPSB2ZWMyKHlSLCB5QykgKiB2ZWMyKCR7c3RyaWRlfSwgJHtzdHJpZGV9KSAtXG4gICAgICAgICAgdmVjMigke3BhZH0uMCwgJHtwYWR9LjApO1xuICAgICAgZmxvYXQgeFJDb3JuZXIgPSB4UkNDb3JuZXIueDtcbiAgICAgIGZsb2F0IHhDQ29ybmVyID0geFJDQ29ybmVyLnk7XG5cbiAgICAgIC8vIENvbnZvbHZlIHgoPywgPywgZDEpIHdpdGggdyg6LCA6LCBkMSwgZDIpIHRvIGdldCB5KHlSLCB5QywgZDIpLlxuICAgICAgLy8gPyA9IHRvIGJlIGRldGVybWluZWQuIDogPSBhY3Jvc3MgYWxsIHZhbHVlcyBpbiB0aGF0IGF4aXMuXG4gICAgICBmbG9hdCBkb3RQcm9kID0gMC4wO1xuICAgICAgZm9yIChmbG9hdCB3UiA9IDAuMDsgd1IgPCAke2ZTaXplfS4wOyB3UiArPSAxLjApIHtcbiAgICAgICAgZmxvYXQgeFIgPSB4UkNvcm5lciArIHdSO1xuICAgICAgICBmbG9hdCB4VGV4UiA9IHhSO1xuXG4gICAgICAgIGZvciAoZmxvYXQgd0MgPSAwLjA7IHdDIDwgJHtmU2l6ZX0uMDsgd0MgKz0gMS4wKSB7XG4gICAgICAgICAgZmxvYXQgeEMgPSB4Q0Nvcm5lciArIHdDO1xuXG4gICAgICAgICAgZm9yIChmbG9hdCBkMSA9IDAuMDsgZDEgPCAke2lucHV0RGVwdGh9LjA7IGQxICs9IDEuMCkge1xuICAgICAgICAgICAgZmxvYXQgeFRleEMgPSB4QyAqICR7aW5wdXREZXB0aH0uMCArIGQxO1xuICAgICAgICAgICAgZmxvYXQgd1RleFIgPSB3UiAqICR7ZlNpemUgKiBpbnB1dERlcHRofS4wICtcbiAgICAgICAgICAgICAgICB3QyAqICR7aW5wdXREZXB0aH0uMCArIGQxO1xuXG4gICAgICAgICAgICBmbG9hdCB4VmFsdWUgPVxuICAgICAgICAgICAgICAgIGdldE1hdHJpeFZhbHVlT3JaZXJvUGFkKHgsIHhTaGFwZUNSLCB2ZWMyKHhUZXhDLCB4VGV4UikpO1xuXG4gICAgICAgICAgICAvLyBSZWFkIHcod1IsIHdDLCBkMSwgZDIpLlxuICAgICAgICAgICAgdmVjMiB3VVYgPSAodmVjMih3VGV4Qywgd1RleFIpICsgaGFsZkNSKSAvIHdTaGFwZUNSO1xuICAgICAgICAgICAgZmxvYXQgd1ZhbHVlID0gdGV4dHVyZTJEKHdlaWdodHMsIHdVVikucjtcblxuICAgICAgICAgICAgZG90UHJvZCArPSB4VmFsdWUgKiB3VmFsdWU7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgICBpZiAoJHtoYXNCaWFzfSkge1xuICAgICAgICBkb3RQcm9kICs9IGdldEJpYXNWYWx1ZShiaWFzZXMsIGQyKTtcbiAgICAgIH1cbiAgICAgIGdsX0ZyYWdDb2xvciA9IHZlYzQoZG90UHJvZCwgMCwgMCwgMCk7XG4gICAgfWA7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRGcmFnbWVudFNoYWRlckdldEJpYXNWYWx1ZVNvdXJjZShvdXRwdXREZXB0aDogbnVtYmVyKTpcbiAgICBzdHJpbmcge1xuICByZXR1cm4gYFxuICAgIGZsb2F0IGdldEJpYXNWYWx1ZShpbiBzYW1wbGVyMkQgYmlhcywgZmxvYXQgYmlhc0MpIHtcbiAgICAgIGNvbnN0IHZlYzIgYmlhc1NoYXBlQ1IgPSB2ZWMyKCR7b3V0cHV0RGVwdGh9LCAxKTtcbiAgICAgIHZlYzIgYmlhc0NSID0gdmVjMihtb2QoYmlhc0MsICR7b3V0cHV0RGVwdGh9LjApLCAwKTtcbiAgICAgIHZlYzIgYmlhc1VWID0gKGJpYXNDUiArIHZlYzIoMC41LCAwLjUpKSAvIGJpYXNTaGFwZUNSO1xuICAgICAgcmV0dXJuIHRleHR1cmUyRChiaWFzLCBiaWFzVVYpLnI7XG4gICAgfWA7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRGcmFnbWVudFNoYWRlclNvdXJjZShcbiAgICBhU2hhcGVSb3dDb2xEZXB0aDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCByZXN1bHREZXB0aDogbnVtYmVyLFxuICAgIGZpZWxkU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlciwgemVyb1BhZDogbnVtYmVyLFxuICAgIGhhc0JpYXM6IGJvb2xlYW4pOiBzdHJpbmcge1xuICBjb25zdCBhU2hhcGVSQzogW251bWJlciwgbnVtYmVyXSA9XG4gICAgICBjb252X3V0aWwuY29tcHV0ZVRleFNoYXBlRnJvbTNEKGFTaGFwZVJvd0NvbERlcHRoKTtcblxuICBjb25zdCB3ZWlnaHRTaGFwZVJDOiBbbnVtYmVyLCBudW1iZXJdID0gY29udl91dGlsLmNvbXB1dGVXZWlnaHRzVGV4U2hhcGUoXG4gICAgICBhU2hhcGVSb3dDb2xEZXB0aFsyXSwgcmVzdWx0RGVwdGgsIGZpZWxkU2l6ZSk7XG5cbiAgY29uc3QgcHJvbG9ndWUgPSBnZXRGcmFnbWVudFNoYWRlclByb2xvZ3VlU291cmNlKCk7XG4gIGNvbnN0IGdldE1hdHJpeFZhbHVlT3JaZXJvUGFkID1cbiAgICAgIGdldEZyYWdtZW50U2hhZGVyR2V0TWF0cml4VmFsdWVPclplcm9QYWRTb3VyY2UoKTtcbiAgY29uc3QgY29udm9sdmUgPSBnZXRGcmFnbWVudFNoYWRlckNvbnZvbHZlU291cmNlKFxuICAgICAgYVNoYXBlUm93Q29sRGVwdGgsIGZpZWxkU2l6ZSwgcmVzdWx0RGVwdGgsIHN0cmlkZSwgemVyb1BhZCwgaGFzQmlhcyk7XG4gIGNvbnN0IGdldEJpYXNWYWx1ZSA9IGdldEZyYWdtZW50U2hhZGVyR2V0Qmlhc1ZhbHVlU291cmNlKHJlc3VsdERlcHRoKTtcblxuICByZXR1cm4gW1xuICAgIHByb2xvZ3VlLFxuICAgIGdldE1hdHJpeFZhbHVlT3JaZXJvUGFkLFxuICAgIGdldEJpYXNWYWx1ZSxcbiAgICBjb252b2x2ZSxcbiAgXS5qb2luKCdcXG4nKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNvbnZvbHZlKFxuICAgIGdwZ3B1OiBHUEdQVUNvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSwgYTogV2ViR0xUZXh0dXJlLFxuICAgIHdlaWdodHM6IFdlYkdMVGV4dHVyZSwgYmlhc2VzOiBXZWJHTFRleHR1cmV8bnVsbCwgcmVzdWx0OiBXZWJHTFRleHR1cmUsXG4gICAgcmVzdWx0U2hhcGVSb3dDb2w6IFtudW1iZXIsIG51bWJlcl0pIHtcbiAgZ3BncHUuc2V0T3V0cHV0TWF0cml4VGV4dHVyZShcbiAgICAgIHJlc3VsdCwgcmVzdWx0U2hhcGVSb3dDb2xbMF0sIHJlc3VsdFNoYXBlUm93Q29sWzFdKTtcbiAgZ3BncHUuc2V0UHJvZ3JhbShwcm9ncmFtKTtcbiAgZ3BncHUuc2V0SW5wdXRNYXRyaXhUZXh0dXJlKGEsICd4JywgMCk7XG4gIGdwZ3B1LnNldElucHV0TWF0cml4VGV4dHVyZSh3ZWlnaHRzLCAnd2VpZ2h0cycsIDEpO1xuICBpZiAoYmlhc2VzICE9IG51bGwpIHtcbiAgICBncGdwdS5zZXRJbnB1dE1hdHJpeFRleHR1cmUoYmlhc2VzLCAnYmlhc2VzJywgMik7XG4gIH1cbiAgZ3BncHUuZXhlY3V0ZVByb2dyYW0oKTtcbn0iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIGdwZ3B1X3V0aWwgZnJvbSAnLi9ncGdwdV91dGlsJztcbmltcG9ydCAqIGFzIHRleF91dGlsIGZyb20gJy4vdGV4X3V0aWwnO1xuaW1wb3J0ICogYXMgd2ViZ2xfdXRpbCBmcm9tICcuL3dlYmdsX3V0aWwnO1xuXG5pbXBvcnQge1dlYkdMTG9zZUNvbnRleHRFeHRlbnNpb259IGZyb20gJy4vd2ViZ2xfdXRpbCc7XG5cbmV4cG9ydCBjbGFzcyBHUEdQVUNvbnRleHQge1xuICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0O1xuICB0ZXh0dXJlRmxvYXRFeHRlbnNpb246IHt9O1xuICBjb2xvckJ1ZmZlckZsb2F0RXh0ZW5zaW9uOiB7fTtcbiAgbG9zZUNvbnRleHRFeHRlbnNpb246IFdlYkdMTG9zZUNvbnRleHRFeHRlbnNpb247XG4gIHZlcnRleEJ1ZmZlcjogV2ViR0xCdWZmZXI7XG4gIGluZGV4QnVmZmVyOiBXZWJHTEJ1ZmZlcjtcbiAgZnJhbWVidWZmZXI6IFdlYkdMRnJhbWVidWZmZXI7XG4gIG91dHB1dFRleHR1cmU6IFdlYkdMVGV4dHVyZXxudWxsID0gbnVsbDtcbiAgcHJvZ3JhbTogV2ViR0xQcm9ncmFtfG51bGwgPSBudWxsO1xuICBwcml2YXRlIGRpc3Bvc2VkID0gZmFsc2U7XG4gIHByaXZhdGUgYXV0b0RlYnVnVmFsaWRhdGUgPSBmYWxzZTtcblxuICBjb25zdHJ1Y3RvcihnbD86IFdlYkdMUmVuZGVyaW5nQ29udGV4dCkge1xuICAgIGlmIChnbCAhPSBudWxsKSB7XG4gICAgICB0aGlzLmdsID0gZ2w7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMuZ2wgPSBncGdwdV91dGlsLmNyZWF0ZVdlYkdMQ29udGV4dCgpO1xuICAgIH1cblxuICAgIC8vIFdlYkdMIDIuMCBlbmFibGVzIHRleHR1cmUgZmxvYXRzIHdpdGhvdXQgYW4gZXh0ZW5zaW9uLlxuICAgIGlmICghd2ViZ2xfdXRpbC5pc1dlYkdMMkVuYWJsZWQoKSkge1xuICAgICAgdGhpcy50ZXh0dXJlRmxvYXRFeHRlbnNpb24gPVxuICAgICAgICAgIHdlYmdsX3V0aWwuZ2V0RXh0ZW5zaW9uT3JUaHJvdyh0aGlzLmdsLCAnT0VTX3RleHR1cmVfZmxvYXQnKTtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy5jb2xvckJ1ZmZlckZsb2F0RXh0ZW5zaW9uID1cbiAgICAgICAgICB3ZWJnbF91dGlsLmdldEV4dGVuc2lvbk9yVGhyb3codGhpcy5nbCwgJ0VYVF9jb2xvcl9idWZmZXJfZmxvYXQnKTtcbiAgICB9XG5cbiAgICB0aGlzLmxvc2VDb250ZXh0RXh0ZW5zaW9uID1cbiAgICAgICAgd2ViZ2xfdXRpbC5nZXRFeHRlbnNpb25PclRocm93KHRoaXMuZ2wsICdXRUJHTF9sb3NlX2NvbnRleHQnKSBhc1xuICAgICAgICBXZWJHTExvc2VDb250ZXh0RXh0ZW5zaW9uO1xuICAgIHRoaXMudmVydGV4QnVmZmVyID0gZ3BncHVfdXRpbC5jcmVhdGVWZXJ0ZXhCdWZmZXIodGhpcy5nbCk7XG4gICAgdGhpcy5pbmRleEJ1ZmZlciA9IGdwZ3B1X3V0aWwuY3JlYXRlSW5kZXhCdWZmZXIodGhpcy5nbCk7XG4gICAgdGhpcy5mcmFtZWJ1ZmZlciA9IHdlYmdsX3V0aWwuY3JlYXRlRnJhbWVidWZmZXIodGhpcy5nbCk7XG4gIH1cblxuICBwdWJsaWMgZGlzcG9zZSgpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGlmICh0aGlzLnByb2dyYW0gIT0gbnVsbCkge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICdEaXNwb3NpbmcgYSBHUEdQVUNvbnRleHQgdGhhdCBzdGlsbCBoYXMgYSBib3VuZCBXZWJHTFByb2dyYW0uJyArXG4gICAgICAgICAgJyBUaGlzIGlzIHByb2JhYmx5IGEgcmVzb3VyY2UgbGVhaywgZGVsZXRlIHRoZSBwcm9ncmFtIHdpdGggJyArXG4gICAgICAgICAgJ0dQR1BVQ29udGV4dC5kZWxldGVQcm9ncmFtIGJlZm9yZSBkaXNwb3NpbmcuJyk7XG4gICAgfVxuICAgIGlmICh0aGlzLm91dHB1dFRleHR1cmUgIT0gbnVsbCkge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICdEaXNwb3NpbmcgYSBHUEdQVUNvbnRleHQgdGhhdCBzdGlsbCBoYXMgYSBib3VuZCBvdXRwdXQgbWF0cml4ICcgK1xuICAgICAgICAgICd0ZXh0dXJlLiAgVGhpcyBpcyBwcm9iYWJseSBhIHJlc291cmNlIGxlYWssIGRlbGV0ZSB0aGUgb3V0cHV0ICcgK1xuICAgICAgICAgICdtYXRyaXggdGV4dHVyZSB3aXRoIEdQR1BVQ29udGV4dC5kZWxldGVNYXRyaXhUZXh0dXJlIGJlZm9yZSAnICtcbiAgICAgICAgICAnZGlzcG9zaW5nLicpO1xuICAgIH1cbiAgICBjb25zdCBnbCA9IHRoaXMuZ2w7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmZpbmlzaCgpKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZEZyYW1lYnVmZmVyKGdsLkZSQU1FQlVGRkVSLCBudWxsKSk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmRlbGV0ZUZyYW1lYnVmZmVyKHRoaXMuZnJhbWVidWZmZXIpKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZEJ1ZmZlcihnbC5BUlJBWV9CVUZGRVIsIG51bGwpKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZGVsZXRlQnVmZmVyKHRoaXMudmVydGV4QnVmZmVyKSk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soXG4gICAgICAgIGdsLCAoKSA9PiBnbC5iaW5kQnVmZmVyKGdsLkVMRU1FTlRfQVJSQVlfQlVGRkVSLCBudWxsKSk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmRlbGV0ZUJ1ZmZlcih0aGlzLmluZGV4QnVmZmVyKSk7XG4gICAgdGhpcy5sb3NlQ29udGV4dEV4dGVuc2lvbi5sb3NlQ29udGV4dCgpO1xuICAgIHRoaXMuZGlzcG9zZWQgPSB0cnVlO1xuICB9XG5cbiAgcHVibGljIGVuYWJsZUF1dG9tYXRpY0RlYnVnVmFsaWRhdGlvbihlbmFibGVkOiBib29sZWFuKSB7XG4gICAgdGhpcy5hdXRvRGVidWdWYWxpZGF0ZSA9IGVuYWJsZWQ7XG4gICAgd2ViZ2xfdXRpbC5lbmFibGVEZWJ1Z1dlYkdMRXJyb3JDaGVja2luZyhlbmFibGVkKTtcbiAgfVxuXG4gIHB1YmxpYyBjcmVhdGVNYXRyaXhUZXh0dXJlKHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTogV2ViR0xUZXh0dXJlIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHJldHVybiBncGdwdV91dGlsLmNyZWF0ZU1hdHJpeFRleHR1cmUodGhpcy5nbCwgcm93cywgY29sdW1ucyk7XG4gIH1cblxuICBwdWJsaWMgdXBsb2FkUGl4ZWxEYXRhVG9UZXh0dXJlKFxuICAgICAgdGV4dHVyZTogV2ViR0xUZXh0dXJlLFxuICAgICAgcGl4ZWxzOiBJbWFnZURhdGF8SFRNTEltYWdlRWxlbWVudHxIVE1MQ2FudmFzRWxlbWVudHxIVE1MVmlkZW9FbGVtZW50KSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICBncGdwdV91dGlsLnVwbG9hZFBpeGVsRGF0YVRvVGV4dHVyZSh0aGlzLmdsLCB0ZXh0dXJlLCBwaXhlbHMpO1xuICB9XG5cbiAgcHVibGljIGNyZWF0ZVBhY2tlZE1hdHJpeFRleHR1cmUocm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOlxuICAgICAgV2ViR0xUZXh0dXJlIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHJldHVybiBncGdwdV91dGlsLmNyZWF0ZVBhY2tlZE1hdHJpeFRleHR1cmUodGhpcy5nbCwgcm93cywgY29sdW1ucyk7XG4gIH1cblxuICBwdWJsaWMgZGVsZXRlTWF0cml4VGV4dHVyZSh0ZXh0dXJlOiBXZWJHTFRleHR1cmUpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGlmICh0aGlzLm91dHB1dFRleHR1cmUgPT09IHRleHR1cmUpIHtcbiAgICAgIHdlYmdsX3V0aWwudW5iaW5kQ29sb3JUZXh0dXJlRnJvbUZyYW1lYnVmZmVyKHRoaXMuZ2wsIHRoaXMuZnJhbWVidWZmZXIpO1xuICAgICAgdGhpcy5vdXRwdXRUZXh0dXJlID0gbnVsbDtcbiAgICB9XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2sodGhpcy5nbCwgKCkgPT4gdGhpcy5nbC5kZWxldGVUZXh0dXJlKHRleHR1cmUpKTtcbiAgfVxuXG4gIHB1YmxpYyB1cGxvYWRNYXRyaXhUb1RleHR1cmUoXG4gICAgICB0ZXh0dXJlOiBXZWJHTFRleHR1cmUsIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyLFxuICAgICAgbWF0cml4OiBGbG9hdDMyQXJyYXkpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGNvbnN0IG51bUNoYW5uZWxzID0gMTtcbiAgICByZXR1cm4gZ3BncHVfdXRpbC51cGxvYWRNYXRyaXhUb1RleHR1cmUoXG4gICAgICAgIHRoaXMuZ2wsIHRleHR1cmUsIHJvd3MsIGNvbHVtbnMsIG1hdHJpeCwgbnVtQ2hhbm5lbHMpO1xuICB9XG5cbiAgcHVibGljIHVwbG9hZE1hdHJpeFRvUGFja2VkVGV4dHVyZShcbiAgICAgIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIsXG4gICAgICBtYXRyaXg6IEZsb2F0MzJBcnJheSkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgcmV0dXJuIGdwZ3B1X3V0aWwudXBsb2FkTWF0cml4VG9QYWNrZWRUZXh0dXJlKFxuICAgICAgICB0aGlzLmdsLCB0ZXh0dXJlLCByb3dzLCBjb2x1bW5zLCBtYXRyaXgpO1xuICB9XG5cbiAgcHVibGljIGRvd25sb2FkTWF0cml4RnJvbVRleHR1cmUoXG4gICAgICB0ZXh0dXJlOiBXZWJHTFRleHR1cmUsIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTogRmxvYXQzMkFycmF5IHtcbiAgICByZXR1cm4gdGhpcy5kb3dubG9hZE1hdHJpeERyaXZlcihcbiAgICAgICAgdGV4dHVyZSxcbiAgICAgICAgKCkgPT5cbiAgICAgICAgICAgIGdwZ3B1X3V0aWwuZG93bmxvYWRNYXRyaXhGcm9tT3V0cHV0VGV4dHVyZSh0aGlzLmdsLCByb3dzLCBjb2x1bW5zKSk7XG4gIH1cblxuICBwdWJsaWMgZG93bmxvYWRNYXRyaXhGcm9tUGFja2VkVGV4dHVyZShcbiAgICAgIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBGbG9hdDMyQXJyYXkge1xuICAgIHJldHVybiB0aGlzLmRvd25sb2FkTWF0cml4RHJpdmVyKFxuICAgICAgICB0ZXh0dXJlLFxuICAgICAgICAoKSA9PiBncGdwdV91dGlsLmRvd25sb2FkTWF0cml4RnJvbVBhY2tlZE91dHB1dFRleHR1cmUoXG4gICAgICAgICAgICB0aGlzLmdsLCByb3dzLCBjb2x1bW5zKSk7XG4gIH1cblxuICBwdWJsaWMgY3JlYXRlUHJvZ3JhbShmcmFnbWVudFNoYWRlclNvdXJjZTogc3RyaW5nKTogV2ViR0xQcm9ncmFtIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGNvbnN0IGdsID0gdGhpcy5nbDtcbiAgICBjb25zdCBmcmFnbWVudFNoYWRlcjogV2ViR0xTaGFkZXIgPVxuICAgICAgICB3ZWJnbF91dGlsLmNyZWF0ZUZyYWdtZW50U2hhZGVyKGdsLCBmcmFnbWVudFNoYWRlclNvdXJjZSk7XG4gICAgY29uc3QgdmVydGV4U2hhZGVyOiBXZWJHTFNoYWRlciA9IGdwZ3B1X3V0aWwuY3JlYXRlVmVydGV4U2hhZGVyKGdsKTtcbiAgICBjb25zdCBwcm9ncmFtOiBXZWJHTFByb2dyYW0gPSB3ZWJnbF91dGlsLmNyZWF0ZVByb2dyYW0oZ2wpO1xuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5hdHRhY2hTaGFkZXIocHJvZ3JhbSwgdmVydGV4U2hhZGVyKSk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmF0dGFjaFNoYWRlcihwcm9ncmFtLCBmcmFnbWVudFNoYWRlcikpO1xuICAgIHdlYmdsX3V0aWwubGlua1Byb2dyYW0oZ2wsIHByb2dyYW0pO1xuICAgIGlmICh0aGlzLmF1dG9EZWJ1Z1ZhbGlkYXRlKSB7XG4gICAgICB3ZWJnbF91dGlsLnZhbGlkYXRlUHJvZ3JhbShnbCwgcHJvZ3JhbSk7XG4gICAgfVxuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5kZXRhY2hTaGFkZXIocHJvZ3JhbSwgdmVydGV4U2hhZGVyKSk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmRlbGV0ZVNoYWRlcih2ZXJ0ZXhTaGFkZXIpKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZGV0YWNoU2hhZGVyKHByb2dyYW0sIGZyYWdtZW50U2hhZGVyKSk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmRlbGV0ZVNoYWRlcihmcmFnbWVudFNoYWRlcikpO1xuICAgIHJldHVybiBwcm9ncmFtO1xuICB9XG5cbiAgcHVibGljIGRlbGV0ZVByb2dyYW0ocHJvZ3JhbTogV2ViR0xQcm9ncmFtKSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICBpZiAocHJvZ3JhbSA9PT0gdGhpcy5wcm9ncmFtKSB7XG4gICAgICB0aGlzLnByb2dyYW0gPSBudWxsO1xuICAgIH1cbiAgICBpZiAocHJvZ3JhbSAhPSBudWxsKSB7XG4gICAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayh0aGlzLmdsLCAoKSA9PiB0aGlzLmdsLmRlbGV0ZVByb2dyYW0ocHJvZ3JhbSkpO1xuICAgIH1cbiAgfVxuXG4gIHB1YmxpYyBzZXRQcm9ncmFtKHByb2dyYW06IFdlYkdMUHJvZ3JhbXxudWxsKSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICB0aGlzLnByb2dyYW0gPSBwcm9ncmFtO1xuICAgIGlmICgodGhpcy5wcm9ncmFtICE9IG51bGwpICYmIHRoaXMuYXV0b0RlYnVnVmFsaWRhdGUpIHtcbiAgICAgIHdlYmdsX3V0aWwudmFsaWRhdGVQcm9ncmFtKHRoaXMuZ2wsIHRoaXMucHJvZ3JhbSk7XG4gICAgfVxuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKHRoaXMuZ2wsICgpID0+IHRoaXMuZ2wudXNlUHJvZ3JhbShwcm9ncmFtKSk7XG4gIH1cblxuICBwdWJsaWMgZ2V0VW5pZm9ybUxvY2F0aW9uKHVuaWZvcm1OYW1lOiBzdHJpbmcpOiBXZWJHTFVuaWZvcm1Mb2NhdGlvbiB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICB0aGlzLnRocm93SWZOb1Byb2dyYW0oKTtcbiAgICByZXR1cm4gd2ViZ2xfdXRpbC5nZXRQcm9ncmFtVW5pZm9ybUxvY2F0aW9uT3JUaHJvdyhcbiAgICAgICAgdGhpcy5nbCwgdGhpcy5wcm9ncmFtISwgdW5pZm9ybU5hbWUpO1xuICB9XG5cbiAgcHVibGljIHNldElucHV0TWF0cml4VGV4dHVyZShcbiAgICAgIGlucHV0TWF0cml4VGV4dHVyZTogV2ViR0xUZXh0dXJlLCB1bmlmb3JtTmFtZTogc3RyaW5nLFxuICAgICAgdGV4dHVyZVVuaXQ6IG51bWJlcikge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgdGhpcy50aHJvd0lmTm9Qcm9ncmFtKCk7XG4gICAgd2ViZ2xfdXRpbC5iaW5kVGV4dHVyZVRvUHJvZ3JhbVVuaWZvcm1TYW1wbGVyKFxuICAgICAgICB0aGlzLmdsLCB0aGlzLnByb2dyYW0hLCBpbnB1dE1hdHJpeFRleHR1cmUsIHVuaWZvcm1OYW1lLCB0ZXh0dXJlVW5pdCk7XG4gIH1cblxuICBwdWJsaWMgc2V0T3V0cHV0TWF0cml4VGV4dHVyZShcbiAgICAgIG91dHB1dE1hdHJpeFRleHR1cmU6IFdlYkdMVGV4dHVyZSwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpIHtcbiAgICB0aGlzLnNldE91dHB1dE1hdHJpeFRleHR1cmVEcml2ZXIob3V0cHV0TWF0cml4VGV4dHVyZSwgY29sdW1ucywgcm93cyk7XG4gIH1cblxuICBwdWJsaWMgc2V0T3V0cHV0UGFja2VkTWF0cml4VGV4dHVyZShcbiAgICAgIG91dHB1dFBhY2tlZE1hdHJpeFRleHR1cmU6IFdlYkdMVGV4dHVyZSwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGNvbnN0IFt3aWR0aCwgaGVpZ2h0XSA9XG4gICAgICAgIHRleF91dGlsLmdldFBhY2tlZE1hdHJpeFRleHR1cmVTaGFwZVdpZHRoSGVpZ2h0KHJvd3MsIGNvbHVtbnMpO1xuICAgIHRoaXMuc2V0T3V0cHV0TWF0cml4VGV4dHVyZURyaXZlcihvdXRwdXRQYWNrZWRNYXRyaXhUZXh0dXJlLCB3aWR0aCwgaGVpZ2h0KTtcbiAgfVxuXG4gIHB1YmxpYyBzZXRPdXRwdXRNYXRyaXhXcml0ZVJlZ2lvbihcbiAgICAgIHN0YXJ0Um93OiBudW1iZXIsIG51bVJvd3M6IG51bWJlciwgc3RhcnRDb2x1bW46IG51bWJlcixcbiAgICAgIG51bUNvbHVtbnM6IG51bWJlcikge1xuICAgIHRoaXMuc2V0T3V0cHV0TWF0cml4V3JpdGVSZWdpb25Ecml2ZXIoXG4gICAgICAgIHN0YXJ0Q29sdW1uLCBzdGFydFJvdywgbnVtQ29sdW1ucywgbnVtUm93cyk7XG4gIH1cblxuICBwdWJsaWMgc2V0T3V0cHV0UGFja2VkTWF0cml4V3JpdGVSZWdpb24oXG4gICAgICBzdGFydFJvdzogbnVtYmVyLCBudW1Sb3dzOiBudW1iZXIsIHN0YXJ0Q29sdW1uOiBudW1iZXIsXG4gICAgICBudW1Db2x1bW5zOiBudW1iZXIpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ3NldE91dHB1dFBhY2tlZE1hdHJpeFdyaXRlUmVnaW9uIG5vdCBpbXBsZW1lbnRlZC4nKTtcbiAgfVxuXG4gIHB1YmxpYyBkZWJ1Z1ZhbGlkYXRlKCkge1xuICAgIGlmICh0aGlzLnByb2dyYW0gIT0gbnVsbCkge1xuICAgICAgd2ViZ2xfdXRpbC52YWxpZGF0ZVByb2dyYW0odGhpcy5nbCwgdGhpcy5wcm9ncmFtKTtcbiAgICB9XG4gICAgd2ViZ2xfdXRpbC52YWxpZGF0ZUZyYW1lYnVmZmVyKHRoaXMuZ2wpO1xuICB9XG5cbiAgcHVibGljIGV4ZWN1dGVQcm9ncmFtKCkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgdGhpcy50aHJvd0lmTm9Qcm9ncmFtKCk7XG4gICAgY29uc3QgZ2wgPSB0aGlzLmdsO1xuICAgIGdwZ3B1X3V0aWwuYmluZFZlcnRleFByb2dyYW1BdHRyaWJ1dGVTdHJlYW1zKFxuICAgICAgICBnbCwgdGhpcy5wcm9ncmFtISwgdGhpcy52ZXJ0ZXhCdWZmZXIpO1xuICAgIGlmICh0aGlzLmF1dG9EZWJ1Z1ZhbGlkYXRlKSB7XG4gICAgICB0aGlzLmRlYnVnVmFsaWRhdGUoKTtcbiAgICB9XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soXG4gICAgICAgIGdsLCAoKSA9PiBnbC5kcmF3RWxlbWVudHMoZ2wuVFJJQU5HTEVTLCA2LCBnbC5VTlNJR05FRF9TSE9SVCwgMCkpO1xuICB9XG5cbiAgcHVibGljIGJsb2NrVW50aWxBbGxQcm9ncmFtc0NvbXBsZXRlZCgpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKHRoaXMuZ2wsICgpID0+IHRoaXMuZ2wuZmluaXNoKCkpO1xuICB9XG5cbiAgcHJpdmF0ZSBkb3dubG9hZE1hdHJpeERyaXZlcihcbiAgICAgIHRleHR1cmU6IFdlYkdMVGV4dHVyZSxcbiAgICAgIGRvd25sb2FkQW5kRGVjb2RlOiAoKSA9PiBGbG9hdDMyQXJyYXkpOiBGbG9hdDMyQXJyYXkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgd2ViZ2xfdXRpbC5iaW5kQ29sb3JUZXh0dXJlVG9GcmFtZWJ1ZmZlcihcbiAgICAgICAgdGhpcy5nbCwgdGV4dHVyZSwgdGhpcy5mcmFtZWJ1ZmZlcik7XG4gICAgY29uc3QgcmVzdWx0ID0gZG93bmxvYWRBbmREZWNvZGUoKTtcbiAgICBpZiAodGhpcy5vdXRwdXRUZXh0dXJlICE9IG51bGwpIHtcbiAgICAgIHdlYmdsX3V0aWwuYmluZENvbG9yVGV4dHVyZVRvRnJhbWVidWZmZXIoXG4gICAgICAgICAgdGhpcy5nbCwgdGhpcy5vdXRwdXRUZXh0dXJlLCB0aGlzLmZyYW1lYnVmZmVyKTtcbiAgICAgIGlmICh0aGlzLmF1dG9EZWJ1Z1ZhbGlkYXRlKSB7XG4gICAgICAgIHdlYmdsX3V0aWwudmFsaWRhdGVGcmFtZWJ1ZmZlcih0aGlzLmdsKTtcbiAgICAgIH1cbiAgICB9IGVsc2Uge1xuICAgICAgd2ViZ2xfdXRpbC51bmJpbmRDb2xvclRleHR1cmVGcm9tRnJhbWVidWZmZXIodGhpcy5nbCwgdGhpcy5mcmFtZWJ1ZmZlcik7XG4gICAgfVxuICAgIHJldHVybiByZXN1bHQ7XG4gIH1cblxuICBwcml2YXRlIHNldE91dHB1dE1hdHJpeFRleHR1cmVEcml2ZXIoXG4gICAgICBvdXRwdXRNYXRyaXhUZXh0dXJlTWF5YmVQYWNrZWQ6IFdlYkdMVGV4dHVyZSwgd2lkdGg6IG51bWJlcixcbiAgICAgIGhlaWdodDogbnVtYmVyKSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICBjb25zdCBnbCA9IHRoaXMuZ2w7XG4gICAgd2ViZ2xfdXRpbC5iaW5kQ29sb3JUZXh0dXJlVG9GcmFtZWJ1ZmZlcihcbiAgICAgICAgZ2wsIG91dHB1dE1hdHJpeFRleHR1cmVNYXliZVBhY2tlZCwgdGhpcy5mcmFtZWJ1ZmZlcik7XG4gICAgaWYgKHRoaXMuYXV0b0RlYnVnVmFsaWRhdGUpIHtcbiAgICAgIHdlYmdsX3V0aWwudmFsaWRhdGVGcmFtZWJ1ZmZlcihnbCk7XG4gICAgfVxuICAgIHRoaXMub3V0cHV0VGV4dHVyZSA9IG91dHB1dE1hdHJpeFRleHR1cmVNYXliZVBhY2tlZDtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wudmlld3BvcnQoMCwgMCwgd2lkdGgsIGhlaWdodCkpO1xuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5zY2lzc29yKDAsIDAsIHdpZHRoLCBoZWlnaHQpKTtcbiAgfVxuXG4gIHByaXZhdGUgc2V0T3V0cHV0TWF0cml4V3JpdGVSZWdpb25Ecml2ZXIoXG4gICAgICB4OiBudW1iZXIsIHk6IG51bWJlciwgd2lkdGg6IG51bWJlciwgaGVpZ2h0OiBudW1iZXIpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgICB0aGlzLmdsLCAoKSA9PiB0aGlzLmdsLnNjaXNzb3IoeCwgeSwgd2lkdGgsIGhlaWdodCkpO1xuICB9XG5cbiAgcHJpdmF0ZSB0aHJvd0lmRGlzcG9zZWQoKSB7XG4gICAgaWYgKHRoaXMuZGlzcG9zZWQpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignQXR0ZW1wdGVkIHRvIHVzZSBkaXNwb3NlZCBHUEdQVUNvbnRleHQuJyk7XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSB0aHJvd0lmTm9Qcm9ncmFtKCkge1xuICAgIGlmICh0aGlzLnByb2dyYW0gPT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKCdObyBHUFUgcHJvZ3JhbSBpcyBjdXJyZW50bHkgc2V0LicpO1xuICAgIH1cbiAgfVxufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQgKiBhcyB0ZXhfdXRpbCBmcm9tICcuL3RleF91dGlsJztcbmltcG9ydCAqIGFzIHdlYmdsX3V0aWwgZnJvbSAnLi93ZWJnbF91dGlsJztcblxuZXhwb3J0IGZ1bmN0aW9uIGdldFdlYkdMQ29udGV4dEF0dHJpYnV0ZXMoKTogV2ViR0xDb250ZXh0QXR0cmlidXRlcyB7XG4gIHJldHVybiB7XG4gICAgYWxwaGE6IGZhbHNlLFxuICAgIGFudGlhbGlhczogZmFsc2UsXG4gICAgcHJlbXVsdGlwbGllZEFscGhhOiBmYWxzZSxcbiAgICBwcmVzZXJ2ZURyYXdpbmdCdWZmZXI6IGZhbHNlLFxuICAgIGRlcHRoOiBmYWxzZSxcbiAgICBzdGVuY2lsOiBmYWxzZSxcbiAgICBmYWlsSWZNYWpvclBlcmZvcm1hbmNlQ2F2ZWF0OiB0cnVlXG4gIH07XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVXZWJHTENvbnRleHQoY2FudmFzPzogSFRNTENhbnZhc0VsZW1lbnQpIHtcbiAgY29uc3QgYXR0cmlidXRlcyA9IGdldFdlYkdMQ29udGV4dEF0dHJpYnV0ZXMoKTtcbiAgbGV0IGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQ7XG4gIGlmIChjYW52YXMgIT0gbnVsbCkge1xuICAgIGdsID0gd2ViZ2xfdXRpbC5jcmVhdGVXZWJHTFJlbmRlcmluZ0NvbnRleHRGcm9tQ2FudmFzKGNhbnZhcywgYXR0cmlidXRlcyk7XG4gIH0gZWxzZSB7XG4gICAgZ2wgPSB3ZWJnbF91dGlsLmNyZWF0ZVdlYkdMUmVuZGVyaW5nQ29udGV4dChhdHRyaWJ1dGVzKTtcbiAgfVxuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZGlzYWJsZShnbC5ERVBUSF9URVNUKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5kaXNhYmxlKGdsLlNURU5DSUxfVEVTVCkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZGlzYWJsZShnbC5CTEVORCkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZGlzYWJsZShnbC5ESVRIRVIpKTtcbiAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmRpc2FibGUoZ2wuUE9MWUdPTl9PRkZTRVRfRklMTCkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZGlzYWJsZShnbC5TQU1QTEVfQ09WRVJBR0UpKTtcbiAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmVuYWJsZShnbC5TQ0lTU09SX1RFU1QpKTtcbiAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmVuYWJsZShnbC5DVUxMX0ZBQ0UpKTtcbiAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmN1bGxGYWNlKGdsLkJBQ0spKTtcbiAgcmV0dXJuIGdsO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlVmVydGV4U2hhZGVyKGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQpOiBXZWJHTFNoYWRlciB7XG4gIGNvbnN0IHZlcnRleFNoYWRlclNvdXJjZSA9IGBcbiAgICBwcmVjaXNpb24gaGlnaHAgZmxvYXQ7XG4gICAgYXR0cmlidXRlIHZlYzMgY2xpcFNwYWNlUG9zO1xuICAgIGF0dHJpYnV0ZSB2ZWMyIHV2O1xuICAgIHZhcnlpbmcgdmVjMiByZXN1bHRVVjtcblxuICAgIHZvaWQgbWFpbigpIHtcbiAgICAgIGdsX1Bvc2l0aW9uID0gdmVjNChjbGlwU3BhY2VQb3MsIDEpO1xuICAgICAgcmVzdWx0VVYgPSB1djtcbiAgICB9YDtcbiAgcmV0dXJuIHdlYmdsX3V0aWwuY3JlYXRlVmVydGV4U2hhZGVyKGdsLCB2ZXJ0ZXhTaGFkZXJTb3VyY2UpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlVmVydGV4QnVmZmVyKGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQpOiBXZWJHTEJ1ZmZlciB7XG4gIC8vIFt4IHkgeiB1IHZdICogW3VwcGVyLWxlZnQsIGxvd2VyLWxlZnQsIHVwcGVyLXJpZ2h0LCBsb3dlci1yaWdodF1cbiAgY29uc3QgdmVydGV4QXJyYXkgPSBuZXcgRmxvYXQzMkFycmF5KFxuICAgICAgWy0xLCAxLCAwLCAwLCAxLCAtMSwgLTEsIDAsIDAsIDAsIDEsIDEsIDAsIDEsIDEsIDEsIC0xLCAwLCAxLCAwXSk7XG4gIHJldHVybiB3ZWJnbF91dGlsLmNyZWF0ZVN0YXRpY1ZlcnRleEJ1ZmZlcihnbCwgdmVydGV4QXJyYXkpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlSW5kZXhCdWZmZXIoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCk6IFdlYkdMQnVmZmVyIHtcbiAgLy8gT3BlbkdMIChhbmQgV2ViR0wpIGhhdmUgXCJDQ1cgPT0gZnJvbnRcIiB3aW5kaW5nXG4gIGNvbnN0IHRyaWFuZ2xlVmVydGV4SW5kaWNlcyA9IG5ldyBVaW50MTZBcnJheShbMCwgMSwgMiwgMiwgMSwgM10pO1xuICByZXR1cm4gd2ViZ2xfdXRpbC5jcmVhdGVTdGF0aWNJbmRleEJ1ZmZlcihnbCwgdHJpYW5nbGVWZXJ0ZXhJbmRpY2VzKTtcbn1cblxuZnVuY3Rpb24gZ2V0VGV4dHVyZUludGVybmFsRm9ybWF0KFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIG51bUNoYW5uZWxzOiBudW1iZXIpOiBudW1iZXIge1xuICBpZiAod2ViZ2xfdXRpbC5pc1dlYkdMMkVuYWJsZWQoKSkge1xuICAgIGlmIChudW1DaGFubmVscyA9PT0gNCkge1xuICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgcmV0dXJuIChnbCBhcyBhbnkpLlJHQkEzMkY7XG4gICAgfVxuICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICByZXR1cm4gKGdsIGFzIGFueSkuUjMyRjtcbiAgfVxuICByZXR1cm4gZ2wuUkdCQTtcbn1cblxuZnVuY3Rpb24gZ2V0VGV4dHVyZUZvcm1hdChcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBudW1DaGFubmVsczogbnVtYmVyKTogbnVtYmVyIHtcbiAgaWYgKHdlYmdsX3V0aWwuaXNXZWJHTDJFbmFibGVkKCkgJiYgbnVtQ2hhbm5lbHMgPT09IDEpIHtcbiAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgcmV0dXJuIChnbCBhcyBhbnkpLlJFRDtcbiAgfVxuICByZXR1cm4gZ2wuUkdCQTtcbn1cblxuZnVuY3Rpb24gY3JlYXRlQW5kQ29uZmlndXJlVGV4dHVyZShcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCB3aWR0aDogbnVtYmVyLCBoZWlnaHQ6IG51bWJlcixcbiAgICBudW1DaGFubmVsczogbnVtYmVyKTogV2ViR0xUZXh0dXJlIHtcbiAgd2ViZ2xfdXRpbC52YWxpZGF0ZVRleHR1cmVTaXplKGdsLCB3aWR0aCwgaGVpZ2h0KTtcbiAgY29uc3QgdGV4dHVyZSA9IHdlYmdsX3V0aWwuY3JlYXRlVGV4dHVyZShnbCk7XG5cbiAgY29uc3QgdGV4MmQgPSBnbC5URVhUVVJFXzJEO1xuICBjb25zdCBpbnRlcm5hbEZvcm1hdCA9IGdldFRleHR1cmVJbnRlcm5hbEZvcm1hdChnbCwgbnVtQ2hhbm5lbHMpO1xuICBjb25zdCBmb3JtYXQgPSBnZXRUZXh0dXJlRm9ybWF0KGdsLCBudW1DaGFubmVscyk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kVGV4dHVyZSh0ZXgyZCwgdGV4dHVyZSkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgIGdsLCAoKSA9PiBnbC50ZXhQYXJhbWV0ZXJpKHRleDJkLCBnbC5URVhUVVJFX1dSQVBfUywgZ2wuQ0xBTVBfVE9fRURHRSkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgIGdsLCAoKSA9PiBnbC50ZXhQYXJhbWV0ZXJpKHRleDJkLCBnbC5URVhUVVJFX1dSQVBfVCwgZ2wuQ0xBTVBfVE9fRURHRSkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgIGdsLCAoKSA9PiBnbC50ZXhQYXJhbWV0ZXJpKHRleDJkLCBnbC5URVhUVVJFX01JTl9GSUxURVIsIGdsLk5FQVJFU1QpKTtcbiAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soXG4gICAgICBnbCwgKCkgPT4gZ2wudGV4UGFyYW1ldGVyaSh0ZXgyZCwgZ2wuVEVYVFVSRV9NQUdfRklMVEVSLCBnbC5ORUFSRVNUKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsXG4gICAgICAoKSA9PiBnbC50ZXhJbWFnZTJEKFxuICAgICAgICAgIHRleDJkLCAwLCBpbnRlcm5hbEZvcm1hdCwgd2lkdGgsIGhlaWdodCwgMCwgZm9ybWF0LCBnbC5GTE9BVCwgbnVsbCkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgbnVsbCkpO1xuICByZXR1cm4gdGV4dHVyZTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZU1hdHJpeFRleHR1cmUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBXZWJHTFRleHR1cmUge1xuICBjb25zdCBbd2lkdGgsIGhlaWdodF0gPVxuICAgICAgdGV4X3V0aWwuZ2V0VW5wYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChyb3dzLCBjb2x1bW5zKTtcbiAgY29uc3QgbnVtQ2hhbm5lbHMgPSAxO1xuICByZXR1cm4gY3JlYXRlQW5kQ29uZmlndXJlVGV4dHVyZShnbCwgd2lkdGgsIGhlaWdodCwgbnVtQ2hhbm5lbHMpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlQ29sb3JNYXRyaXhUZXh0dXJlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTogV2ViR0xUZXh0dXJlIHtcbiAgY29uc3QgW3dpZHRoLCBoZWlnaHRdID1cbiAgICAgIHRleF91dGlsLmdldENvbG9yTWF0cml4VGV4dHVyZVNoYXBlV2lkdGhIZWlnaHQocm93cywgY29sdW1ucyk7XG4gIGNvbnN0IG51bUNoYW5uZWxzID0gNDtcbiAgcmV0dXJuIGNyZWF0ZUFuZENvbmZpZ3VyZVRleHR1cmUoZ2wsIHdpZHRoLCBoZWlnaHQsIG51bUNoYW5uZWxzKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVBhY2tlZE1hdHJpeFRleHR1cmUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBXZWJHTFRleHR1cmUge1xuICBjb25zdCBbd2lkdGgsIGhlaWdodF0gPVxuICAgICAgdGV4X3V0aWwuZ2V0UGFja2VkTWF0cml4VGV4dHVyZVNoYXBlV2lkdGhIZWlnaHQocm93cywgY29sdW1ucyk7XG4gIGNvbnN0IG51bUNoYW5uZWxzID0gNDtcbiAgcmV0dXJuIGNyZWF0ZUFuZENvbmZpZ3VyZVRleHR1cmUoZ2wsIHdpZHRoLCBoZWlnaHQsIG51bUNoYW5uZWxzKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGJpbmRWZXJ0ZXhQcm9ncmFtQXR0cmlidXRlU3RyZWFtcyhcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBwcm9ncmFtOiBXZWJHTFByb2dyYW0sXG4gICAgdmVydGV4QnVmZmVyOiBXZWJHTEJ1ZmZlcikge1xuICBjb25zdCBwb3NPZmZzZXQgPSAwOyAgICAgICAgICAgICAgIC8vIHggaXMgdGhlIGZpcnN0IGJ1ZmZlciBlbGVtZW50XG4gIGNvbnN0IHV2T2Zmc2V0ID0gMyAqIDQ7ICAgICAgICAgICAgLy8gdXYgY29tZXMgYWZ0ZXIgW3ggeSB6XVxuICBjb25zdCBzdHJpZGUgPSAoMyAqIDQpICsgKDIgKiA0KTsgIC8vIHh5eiArIHV2LCBlYWNoIGVudHJ5IGlzIDQtYnl0ZSBmbG9hdC5cbiAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soXG4gICAgICBnbCwgKCkgPT4gZ2wuYmluZEJ1ZmZlcihnbC5BUlJBWV9CVUZGRVIsIHZlcnRleEJ1ZmZlcikpO1xuICB3ZWJnbF91dGlsLmJpbmRWZXJ0ZXhCdWZmZXJUb1Byb2dyYW1BdHRyaWJ1dGUoXG4gICAgICBnbCwgcHJvZ3JhbSwgJ2NsaXBTcGFjZVBvcycsIHZlcnRleEJ1ZmZlciwgMywgc3RyaWRlLCBwb3NPZmZzZXQpO1xuICB0cnkge1xuICAgIHdlYmdsX3V0aWwuYmluZFZlcnRleEJ1ZmZlclRvUHJvZ3JhbUF0dHJpYnV0ZShcbiAgICAgICAgZ2wsIHByb2dyYW0sICd1dicsIHZlcnRleEJ1ZmZlciwgMiwgc3RyaWRlLCB1dk9mZnNldCk7XG4gIH0gY2F0Y2ggKGUpIHtcbiAgICAvLyBQcm9ncmFtcyB3aXRoIDF4MSBvdXRwdXQgdGV4dHVyZXMgZG9uJ3QgdXNlIHRoZSB1diBhdHRyaWJ1dGUuXG4gICAgLy8gVGhpcyBjYW4gY2F1c2UgdGhlIHNoYWRlciBsaW5rZXIgdG8gZGVhZC1zdHJpcCBpdCwgc28gd2Ugc2hvdWxkbid0XG4gICAgLy8gY29tcGxhaW4gb3IgZmFpbCBpZiBpdCdzIG5vdCBwcmVzZW50LlxuICAgIGlmICghZS5oYXNPd25Qcm9wZXJ0eSgnbmFtZWRWZXJ0ZXhBdHRyaWJ1dGVOb3RGb3VuZCcpKSB7XG4gICAgICB0aHJvdyBlO1xuICAgIH1cbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gdXBsb2FkUGl4ZWxEYXRhVG9UZXh0dXJlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHRleHR1cmU6IFdlYkdMVGV4dHVyZSxcbiAgICBwaXhlbHM6IEltYWdlRGF0YXxIVE1MSW1hZ2VFbGVtZW50fEhUTUxDYW52YXNFbGVtZW50fEhUTUxWaWRlb0VsZW1lbnQpIHtcbiAgY29uc3QgbnVtQ2hhbm5lbHMgPSA0O1xuICBjb25zdCBpbnRlcm5hbEZvcm1hdCA9IGdldFRleHR1cmVJbnRlcm5hbEZvcm1hdChnbCwgbnVtQ2hhbm5lbHMpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgdGV4dHVyZSkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgIGdsLFxuICAgICAgKCkgPT4gZ2wudGV4SW1hZ2UyRChcbiAgICAgICAgICBnbC5URVhUVVJFXzJELCAwLCBpbnRlcm5hbEZvcm1hdCwgZ2wuUkdCQSwgZ2wuRkxPQVQsIHBpeGVscykpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgbnVsbCkpO1xufVxuXG5mdW5jdGlvbiB1cGxvYWREYXRhVG9UZXh0dXJlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgd2lkdGg6IG51bWJlcixcbiAgICBoZWlnaHQ6IG51bWJlciwgZGF0YTogRmxvYXQzMkFycmF5LCBudW1DaGFubmVsczogbnVtYmVyKSB7XG4gIGNvbnN0IHRleHR1cmVGb3JtYXQgPSBnZXRUZXh0dXJlRm9ybWF0KGdsLCBudW1DaGFubmVscyk7XG5cbiAgd2ViZ2xfdXRpbC52YWxpZGF0ZVRleHR1cmVTaXplKGdsLCB3aWR0aCwgaGVpZ2h0KTtcbiAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRUZXh0dXJlKGdsLlRFWFRVUkVfMkQsIHRleHR1cmUpKTtcbiAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soXG4gICAgICBnbCxcbiAgICAgICgpID0+IGdsLnRleFN1YkltYWdlMkQoXG4gICAgICAgICAgZ2wuVEVYVFVSRV8yRCwgMCwgMCwgMCwgd2lkdGgsIGhlaWdodCwgdGV4dHVyZUZvcm1hdCwgZ2wuRkxPQVQsXG4gICAgICAgICAgZGF0YSkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgbnVsbCkpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gdXBsb2FkTWF0cml4VG9UZXh0dXJlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgcm93czogbnVtYmVyLFxuICAgIGNvbHVtbnM6IG51bWJlciwgbWF0cml4OiBGbG9hdDMyQXJyYXksIG51bUNoYW5uZWxzOiBudW1iZXIpIHtcbiAgY29uc3QgW3csIGhdID1cbiAgICAgIHRleF91dGlsLmdldFVucGFja2VkTWF0cml4VGV4dHVyZVNoYXBlV2lkdGhIZWlnaHQocm93cywgY29sdW1ucyk7XG5cbiAgY29uc3QgY2hhbm5lbHNQZXJUZXh0dXJlID1cbiAgICAgIG51bUNoYW5uZWxzID09PSAxID8gd2ViZ2xfdXRpbC5nZXRDaGFubmVsc1BlclRleHR1cmUoKSA6IG51bUNoYW5uZWxzO1xuICBjb25zdCB1bnBhY2tlZEFycmF5ID1cbiAgICAgIG5ldyBGbG9hdDMyQXJyYXkodGV4X3V0aWwuZ2V0VW5wYWNrZWRBcnJheVNpemVGcm9tTWF0cml4U2l6ZShcbiAgICAgICAgICBtYXRyaXgubGVuZ3RoLCBjaGFubmVsc1BlclRleHR1cmUpKTtcbiAgdGV4X3V0aWwuZW5jb2RlTWF0cml4VG9VbnBhY2tlZEFycmF5KFxuICAgICAgbWF0cml4LCB1bnBhY2tlZEFycmF5LCBjaGFubmVsc1BlclRleHR1cmUpO1xuXG4gIHVwbG9hZERhdGFUb1RleHR1cmUoZ2wsIHRleHR1cmUsIHcsIGgsIHVucGFja2VkQXJyYXksIG51bUNoYW5uZWxzKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHVwbG9hZE1hdHJpeFRvUGFja2VkVGV4dHVyZShcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCB0ZXh0dXJlOiBXZWJHTFRleHR1cmUsIHJvd3M6IG51bWJlcixcbiAgICBjb2x1bW5zOiBudW1iZXIsIG1hdHJpeDogRmxvYXQzMkFycmF5KSB7XG4gIGNvbnN0IFt3LCBoXSA9IHRleF91dGlsLmdldFBhY2tlZE1hdHJpeFRleHR1cmVTaGFwZVdpZHRoSGVpZ2h0KHJvd3MsIGNvbHVtbnMpO1xuICBjb25zdCBwYWNrZWRSR0JBID0gbmV3IEZsb2F0MzJBcnJheShcbiAgICAgIHRleF91dGlsLmdldFBhY2tlZFJHQkFBcnJheVNpemVGcm9tTWF0cml4U2hhcGUocm93cywgY29sdW1ucykpO1xuICB0ZXhfdXRpbC5lbmNvZGVNYXRyaXhUb1BhY2tlZFJHQkEobWF0cml4LCByb3dzLCBjb2x1bW5zLCBwYWNrZWRSR0JBKTtcbiAgY29uc3QgbnVtQ2hhbm5lbHMgPSA0O1xuICB1cGxvYWREYXRhVG9UZXh0dXJlKGdsLCB0ZXh0dXJlLCB3LCBoLCBwYWNrZWRSR0JBLCBudW1DaGFubmVscyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBkb3dubG9hZE1hdHJpeEZyb21PdXRwdXRUZXh0dXJlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTogRmxvYXQzMkFycmF5IHtcbiAgY29uc3QgW3csIGhdID1cbiAgICAgIHRleF91dGlsLmdldFVucGFja2VkTWF0cml4VGV4dHVyZVNoYXBlV2lkdGhIZWlnaHQocm93cywgY29sdW1ucyk7XG5cbiAgY29uc3QgY2hhbm5lbHNQZXJUZXh0dXJlID0gNDtcbiAgY29uc3QgdW5wYWNrZWRBcnJheSA9XG4gICAgICBuZXcgRmxvYXQzMkFycmF5KHRleF91dGlsLmdldFVucGFja2VkQXJyYXlTaXplRnJvbU1hdHJpeFNpemUoXG4gICAgICAgICAgcm93cyAqIGNvbHVtbnMsIGNoYW5uZWxzUGVyVGV4dHVyZSkpO1xuICBjb25zdCB0ZXh0dXJlRm9ybWF0ID0gZ2V0VGV4dHVyZUZvcm1hdChnbCwgY2hhbm5lbHNQZXJUZXh0dXJlKTtcblxuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgIGdsLCAoKSA9PiBnbC5yZWFkUGl4ZWxzKDAsIDAsIHcsIGgsIGdsLlJHQkEsIGdsLkZMT0FULCB1bnBhY2tlZEFycmF5KSk7XG5cbiAgY29uc3QgbWF0cml4ID0gbmV3IEZsb2F0MzJBcnJheShyb3dzICogY29sdW1ucyk7XG4gIHRleF91dGlsLmRlY29kZU1hdHJpeEZyb21VbnBhY2tlZEFycmF5KFxuICAgICAgdW5wYWNrZWRBcnJheSwgbWF0cml4LCBjaGFubmVsc1BlclRleHR1cmUpO1xuICByZXR1cm4gbWF0cml4O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZG93bmxvYWRNYXRyaXhGcm9tUGFja2VkT3V0cHV0VGV4dHVyZShcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IEZsb2F0MzJBcnJheSB7XG4gIGNvbnN0IFt3LCBoXSA9IHRleF91dGlsLmdldFBhY2tlZE1hdHJpeFRleHR1cmVTaGFwZVdpZHRoSGVpZ2h0KHJvd3MsIGNvbHVtbnMpO1xuICBjb25zdCBwYWNrZWRSR0JBID0gbmV3IEZsb2F0MzJBcnJheShcbiAgICAgIHRleF91dGlsLmdldFBhY2tlZFJHQkFBcnJheVNpemVGcm9tTWF0cml4U2hhcGUocm93cywgY29sdW1ucykpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgIGdsLCAoKSA9PiBnbC5yZWFkUGl4ZWxzKDAsIDAsIHcsIGgsIGdsLlJHQkEsIGdsLkZMT0FULCBwYWNrZWRSR0JBKSk7XG4gIGNvbnN0IG1hdHJpeCA9IG5ldyBGbG9hdDMyQXJyYXkocm93cyAqIGNvbHVtbnMpO1xuICByZXR1cm4gdGV4X3V0aWwuZGVjb2RlTWF0cml4RnJvbVBhY2tlZFJHQkEocGFja2VkUkdCQSwgcm93cywgY29sdW1ucywgbWF0cml4KTtcbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4vZ3BncHVfY29udGV4dCc7XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRGcmFnbWVudFNoYWRlclNvdXJjZShyb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IHN0cmluZyB7XG4gIHJldHVybiBgXG4gICAgcHJlY2lzaW9uIGhpZ2hwIGZsb2F0O1xuICAgIHVuaWZvcm0gc2FtcGxlcjJEIG1hdHJpeEE7XG4gICAgdmFyeWluZyB2ZWMyIHJlc3VsdFVWO1xuXG4gICAgY29uc3QgdmVjMiBhRGltQ1IgPSB2ZWMyKCR7Y29sdW1uc30uMCwgJHtyb3dzfS4wKTtcbiAgICBjb25zdCB2ZWMyIGhhbGZDUiA9IHZlYzIoMC41LCAwLjUpO1xuXG4gICAgdm9pZCBtYWluKCkge1xuICAgICAgZmxvYXQgYU1heCA9IHRleHR1cmUyRChtYXRyaXhBLCBoYWxmQ1IgLyBhRGltQ1IpLnI7XG4gICAgICBmb3IgKGZsb2F0IHIgPSAwLjA7IHIgPCBhRGltQ1IueTsgciArPSAxLjApIHtcbiAgICAgICAgZm9yIChmbG9hdCBjID0gMC4wOyBjIDwgYURpbUNSLng7IGMgKz0gMS4wKSB7XG4gICAgICAgICAgdmVjMiB1diA9ICh2ZWMyKGMsIHIpICsgaGFsZkNSKSAvIGFEaW1DUjtcbiAgICAgICAgICBmbG9hdCBhQ3VyID0gdGV4dHVyZTJEKG1hdHJpeEEsIHV2KS5yO1xuICAgICAgICAgIGFNYXggPSBtYXgoYU1heCwgYUN1cik7XG4gICAgICAgIH1cbiAgICAgIH1cblxuICAgICAgZmxvYXQgZXhwU3VtID0gMC4wO1xuICAgICAgZm9yIChmbG9hdCByID0gMC4wOyByIDwgYURpbUNSLnk7IHIgKz0gMS4wKSB7XG4gICAgICAgIGZvciAoZmxvYXQgYyA9IDAuMDsgYyA8IGFEaW1DUi54OyBjICs9IDEuMCkge1xuICAgICAgICAgIHZlYzIgdXYgPSAodmVjMihjLCByKSArIGhhbGZDUikgLyBhRGltQ1I7XG4gICAgICAgICAgZmxvYXQgYUN1ciA9IHRleHR1cmUyRChtYXRyaXhBLCB1dikucjtcbiAgICAgICAgICBleHBTdW0gKz0gZXhwKGFDdXIgLSBhTWF4KTtcbiAgICAgICAgfVxuICAgICAgfVxuXG4gICAgICBnbF9GcmFnQ29sb3IgPSB2ZWM0KGFNYXggKyBsb2coZXhwU3VtKSwgMCwgMCwgMCk7XG4gICAgfWA7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBsb2dTdW1FeHAoXG4gICAgZ3BncHU6IEdQR1BVQ29udGV4dCwgbG9nU3VtRXhwUHJvZ3JhbTogV2ViR0xQcm9ncmFtLCBhOiBXZWJHTFRleHR1cmUsXG4gICAgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIsIHJlc3VsdDogV2ViR0xUZXh0dXJlKSB7XG4gIGdwZ3B1LnNldE91dHB1dE1hdHJpeFRleHR1cmUocmVzdWx0LCAxLCAxKTtcbiAgZ3BncHUuc2V0UHJvZ3JhbShsb2dTdW1FeHBQcm9ncmFtKTtcbiAgZ3BncHUuc2V0SW5wdXRNYXRyaXhUZXh0dXJlKGEsICdtYXRyaXhBJywgMCk7XG4gIGdwZ3B1LmV4ZWN1dGVQcm9ncmFtKCk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB1cGxvYWRMb2dTdW1FeHBEb3dubG9hZChcbiAgICBhOiBGbG9hdDMyQXJyYXksIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTogbnVtYmVyIHtcbiAgY29uc3QgZ3BncHUgPSBuZXcgR1BHUFVDb250ZXh0KCk7XG4gIGNvbnN0IHByb2dyYW0gPSBncGdwdS5jcmVhdGVQcm9ncmFtKGdldEZyYWdtZW50U2hhZGVyU291cmNlKHJvd3MsIGNvbHVtbnMpKTtcbiAgY29uc3QgYVRleHR1cmUgPSBncGdwdS5jcmVhdGVNYXRyaXhUZXh0dXJlKHJvd3MsIGNvbHVtbnMpO1xuICBjb25zdCByZXN1bHRUZXh0dXJlID0gZ3BncHUuY3JlYXRlTWF0cml4VGV4dHVyZSgxLCAxKTtcbiAgZ3BncHUudXBsb2FkTWF0cml4VG9UZXh0dXJlKGFUZXh0dXJlLCByb3dzLCBjb2x1bW5zLCBhKTtcbiAgbG9nU3VtRXhwKGdwZ3B1LCBwcm9ncmFtLCBhVGV4dHVyZSwgcm93cywgY29sdW1ucywgcmVzdWx0VGV4dHVyZSk7XG4gIGNvbnN0IHJlc3VsdCA9IGdwZ3B1LmRvd25sb2FkTWF0cml4RnJvbVRleHR1cmUocmVzdWx0VGV4dHVyZSwgMSwgMSk7XG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUoYVRleHR1cmUpO1xuICBncGdwdS5kZWxldGVNYXRyaXhUZXh0dXJlKHJlc3VsdFRleHR1cmUpO1xuICBncGdwdS5kZWxldGVQcm9ncmFtKHByb2dyYW0pO1xuICBncGdwdS5kaXNwb3NlKCk7XG4gIHJldHVybiByZXN1bHRbMF07XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIGNvbnZfdXRpbCBmcm9tICcuLi9jb252X3V0aWwnO1xuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4vZ3BncHVfY29udGV4dCc7XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRGcmFnbWVudFNoYWRlck1heFBvb2xCYWNrcHJvcChcbiAgICBkeVNoYXBlUkNEOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGZTaXplOiBudW1iZXIsIG9yaWdTdHJpZGU6IG51bWJlcixcbiAgICBvcmlnUGFkOiBudW1iZXIpIHtcbiAgY29uc3Qgb3JpZ0lucHV0RGVwdGggPSBkeVNoYXBlUkNEWzJdO1xuICBjb25zdCBwYWQgPSBmU2l6ZSAtIDEgLSBvcmlnUGFkO1xuICBjb25zdCBbZHlSb3dzLCBkeUNvbHMsIGRlcHRoXSA9IGR5U2hhcGVSQ0Q7XG5cbiAgY29uc3QgZHlUZXhTaGFwZVJDID0gY29udl91dGlsLmNvbXB1dGVUZXhTaGFwZUZyb20zRChkeVNoYXBlUkNEKTtcblxuICByZXR1cm4gYFxuICAgIHByZWNpc2lvbiBoaWdocCBmbG9hdDtcbiAgICB1bmlmb3JtIHNhbXBsZXIyRCBkeTtcbiAgICB1bmlmb3JtIHNhbXBsZXIyRCBtYXhQb3M7XG5cbiAgICBjb25zdCB2ZWMyIGhhbGZDUiA9IHZlYzIoMC41LCAwLjUpO1xuICAgIGNvbnN0IHZlYzIgZHlTaGFwZUNSID0gdmVjMigke2R5VGV4U2hhcGVSQ1sxXX0sICR7ZHlUZXhTaGFwZVJDWzBdfSk7XG5cbiAgICB2b2lkIG1haW4oKSB7XG4gICAgICB2ZWMyIGR4VGV4Q1IgPSBmbG9vcihnbF9GcmFnQ29vcmQueHkpO1xuXG4gICAgICAvLyBNYXAgZnJvbSAyRCAoZHhUZXhSLCBkeFRleEMpIHRvIDNEIChkeFIsIGR4QywgZCkuXG4gICAgICBmbG9hdCBkeFIgPSBkeFRleENSLnk7XG4gICAgICBmbG9hdCBkeEMgPSBmbG9vcihkeFRleENSLnggLyAke29yaWdJbnB1dERlcHRofS4wKTtcbiAgICAgIGZsb2F0IGQgPSBtb2QoZHhUZXhDUi54LCAke29yaWdJbnB1dERlcHRofS4wKTtcblxuICAgICAgdmVjMiBkeVJDQ29ybmVyID0gdmVjMihkeFIsIGR4QykgLSB2ZWMyKCR7cGFkfS4wLCAke3BhZH0uMCk7XG4gICAgICBmbG9hdCBkeVJDb3JuZXIgPSBkeVJDQ29ybmVyLng7XG4gICAgICBmbG9hdCBkeUNDb3JuZXIgPSBkeVJDQ29ybmVyLnk7XG5cbiAgICAgIC8vIENvbnZvbHZlIGR5KD8sID8sIGQpIHdpdGggcG9zIG1hc2soOiwgOiwgZCkgdG8gZ2V0IGR4KHlSLCBkeEMsIGQpLlxuICAgICAgLy8gPyA9IHRvIGJlIGRldGVybWluZWQuIDogPSBhY3Jvc3MgYWxsIHZhbHVlcyBpbiB0aGF0IGF4aXMuXG4gICAgICBmbG9hdCBkb3RQcm9kID0gMC4wO1xuICAgICAgZm9yIChmbG9hdCB3UiA9IDAuMDsgd1IgPCAke2ZTaXplfS4wOyB3UiArPSAxLjApIHtcblxuICAgICAgICBmbG9hdCBkeVIgPSAoZHlSQ29ybmVyICsgd1IpIC8gJHtvcmlnU3RyaWRlfS4wO1xuICAgICAgICAvLyBUT0RPKG5zdGhvcmF0KTogU3BsaWNlIHRoaXMgd2l0aCBhbm90aGVyIHZlcnNpb24gd2hlcmUgeW91IGNhbGxcbiAgICAgICAgLy8gZ2V0TWF0cml4VmFsdWVPclplcm9QYWQoKS4gSGVyZSBhbmQgYmVsb3cuXG4gICAgICAgIGlmIChkeVIgPCAwLjAgfHwgZHlSID49ICR7ZHlSb3dzfS4wIHx8IGZyYWN0KGR5UikgPiAwLjApIHtcbiAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgfVxuXG4gICAgICAgIGZsb2F0IGR5VGV4UiA9IGR5UjtcblxuICAgICAgICBmb3IgKGZsb2F0IHdDID0gMC4wOyB3QyA8ICR7ZlNpemV9LjA7IHdDICs9IDEuMCkge1xuXG4gICAgICAgICAgZmxvYXQgZHlDID0gKGR5Q0Nvcm5lciArIHdDKSAvICR7b3JpZ1N0cmlkZX0uMDtcbiAgICAgICAgICBpZiAoZHlDIDwgMC4wIHx8IGR5QyA+PSAke2R5Q29sc30uMCB8fCBmcmFjdChkeUMpID4gMC4wKSB7XG4gICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICB9XG5cbiAgICAgICAgICBmbG9hdCBkeVRleEMgPSBkeUMgKiAke2RlcHRofS4wICsgZDtcblxuICAgICAgICAgIC8vIFJlYWQgZHkoZHlSLCBkeUMsIGQpLlxuICAgICAgICAgIHZlYzIgZHlVViA9ICh2ZWMyKGR5VGV4QywgZHlUZXhSKSArIGhhbGZDUikgLyBkeVNoYXBlQ1I7XG4gICAgICAgICAgZmxvYXQgZHlWYWx1ZSA9IHRleHR1cmUyRChkeSwgZHlVVikucjtcblxuICAgICAgICAgIC8vIFJlYWQgbWF4UG9zKGR5UiwgZHlDLCBkKS5cbiAgICAgICAgICBmbG9hdCBtYXhQb3NWYWx1ZSA9XG4gICAgICAgICAgICAgICR7ZlNpemUgKiBmU2l6ZSAtIDF9LjAgLSB0ZXh0dXJlMkQobWF4UG9zLCBkeVVWKS5yO1xuXG4gICAgICAgICAgLy8gR2V0IHRoZSBjdXJyZW50IHZhbHVlLCBjaGVjayBpdCBhZ2FpbnN0IHRoZSB2YWx1ZSBmcm9tIHRoZVxuICAgICAgICAgIC8vIHBvc2l0aW9uIG1hdHJpeC5cbiAgICAgICAgICBmbG9hdCBjdXJQb3NWYWx1ZSA9IHdSICogJHtmU2l6ZX0uMCArIHdDO1xuICAgICAgICAgIGZsb2F0IG1hc2sgPSBmbG9hdChtYXhQb3NWYWx1ZSA9PSBjdXJQb3NWYWx1ZSA/IDEuMCA6IDAuMCk7XG5cbiAgICAgICAgICBkb3RQcm9kICs9IGR5VmFsdWUgKiBtYXNrO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICBnbF9GcmFnQ29sb3IgPSB2ZWM0KGRvdFByb2QsIDAsIDAsIDApO1xuICAgIH1gO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gbWF4UG9vbEJhY2twcm9wKFxuICAgIGdwZ3B1OiBHUEdQVUNvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSwgZHlUZXg6IFdlYkdMVGV4dHVyZSxcbiAgICBtYXhQb3NpdGlvbnNUZXg6IFdlYkdMVGV4dHVyZSwgcmVzdWx0VGV4OiBXZWJHTFRleHR1cmUsXG4gICAgcmVzdWx0VGV4U2hhcGVSQzogW251bWJlciwgbnVtYmVyXSkge1xuICBncGdwdS5zZXRPdXRwdXRNYXRyaXhUZXh0dXJlKFxuICAgICAgcmVzdWx0VGV4LCByZXN1bHRUZXhTaGFwZVJDWzBdLCByZXN1bHRUZXhTaGFwZVJDWzFdKTtcbiAgZ3BncHUuc2V0UHJvZ3JhbShwcm9ncmFtKTtcbiAgZ3BncHUuc2V0SW5wdXRNYXRyaXhUZXh0dXJlKGR5VGV4LCAnZHknLCAwKTtcbiAgZ3BncHUuc2V0SW5wdXRNYXRyaXhUZXh0dXJlKG1heFBvc2l0aW9uc1RleCwgJ21heFBvcycsIDEpO1xuICBncGdwdS5leGVjdXRlUHJvZ3JhbSgpO1xufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQge0dQR1BVQ29udGV4dH0gZnJvbSAnLi9ncGdwdV9jb250ZXh0JztcbmltcG9ydCAqIGFzIHBvb2xfZ3B1IGZyb20gJy4vcG9vbF9ncHUnO1xuXG5leHBvcnQgZnVuY3Rpb24gZ2V0RnJhZ21lbnRTaGFkZXJNYXhQb29sUG9zaXRpb25zU291cmNlKFxuICAgIHhTaGFwZVJDRDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlcixcbiAgICBwYWQ6IG51bWJlcikge1xuICByZXR1cm4gZ2V0RnJhZ21lbnRTaGFkZXJNYXhQb29sQ29tbW9uU291cmNlKFxuICAgICAgeFNoYXBlUkNELCBmU2l6ZSwgc3RyaWRlLCBwYWQsIHRydWUpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0RnJhZ21lbnRTaGFkZXJNYXhQb29sU291cmNlKFxuICAgIHhTaGFwZVJDRDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlcixcbiAgICBwYWQ6IG51bWJlcikge1xuICByZXR1cm4gZ2V0RnJhZ21lbnRTaGFkZXJNYXhQb29sQ29tbW9uU291cmNlKFxuICAgICAgeFNoYXBlUkNELCBmU2l6ZSwgc3RyaWRlLCBwYWQsIGZhbHNlKTtcbn1cblxuZnVuY3Rpb24gZ2V0RnJhZ21lbnRTaGFkZXJNYXhQb29sQ29tbW9uU291cmNlKFxuICAgIHhTaGFwZVJDRDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlcixcbiAgICBwYWQ6IG51bWJlciwgY29tcHV0ZU1heFBvc2l0aW9uczogYm9vbGVhbikge1xuICByZXR1cm4gcG9vbF9ncHUuZ2V0RnJhZ21lbnRTaGFkZXJQb29sQ29tbW9uU291cmNlKFxuICAgICAgeFNoYXBlUkNELCBmU2l6ZSwgc3RyaWRlLCBwYWQsICdtYXgnLCBjb21wdXRlTWF4UG9zaXRpb25zKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIG1heFBvb2xDb21tb24oXG4gICAgZ3BncHU6IEdQR1BVQ29udGV4dCwgcHJvZ3JhbTogV2ViR0xQcm9ncmFtLCB4OiBXZWJHTFRleHR1cmUsXG4gICAgcmVzdWx0OiBXZWJHTFRleHR1cmUsIHJlc3VsdFNoYXBlUm93Q29sOiBbbnVtYmVyLCBudW1iZXJdKSB7XG4gIHBvb2xfZ3B1LnBvb2xDb21tb24oZ3BncHUsIHByb2dyYW0sIHgsIHJlc3VsdCwgcmVzdWx0U2hhcGVSb3dDb2wpO1xufSIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0IHtNYXRyaXhPcmllbnRhdGlvbn0gZnJvbSAnLi4vbWF0aCc7XG5pbXBvcnQge0FycmF5MkR9IGZyb20gJy4uL25kYXJyYXknO1xuXG5pbXBvcnQge0dQR1BVQ29udGV4dH0gZnJvbSAnLi9ncGdwdV9jb250ZXh0JztcbmltcG9ydCAqIGFzIHNoYWRlcl9jb21waWxlciBmcm9tICcuL3NoYWRlcl9jb21waWxlcic7XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRGcmFnbWVudFNoYWRlcihcbiAgICBhOiBBcnJheTJELCBiOiBBcnJheTJELCBvdXQ6IEFycmF5MkQsIGFPcmllbnRhdGlvbjogTWF0cml4T3JpZW50YXRpb24sXG4gICAgYk9yaWVudGF0aW9uOiBNYXRyaXhPcmllbnRhdGlvbik6IHN0cmluZyB7XG4gIGNvbnN0IHNoYXJlZERpbSA9XG4gICAgICAoYU9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSID8gYS5zaGFwZVsxXSA6IGEuc2hhcGVbMF0pO1xuICBjb25zdCBhU25pcHBldCA9XG4gICAgICAoYU9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/ICdhUm93LCBpJyA6ICdpLCBhUm93JztcbiAgY29uc3QgYlNuaXBwZXQgPVxuICAgICAgKGJPcmllbnRhdGlvbiA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUikgPyAnaSwgYkNvbCcgOiAnYkNvbCwgaSc7XG5cbiAgY29uc3QgaW5wdXRzID0gW3tuYW1lOiAnbWF0cml4QScsIGFycmF5OiBhfSwge25hbWU6ICdtYXRyaXhCJywgYXJyYXk6IGJ9XTtcbiAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgY29uc3QgZmxvYXQgc2hhcmVkRGltID0gJHtzaGFyZWREaW19LjA7XG5cbiAgICBmbG9hdCBkb3RBUm93QkNvbChmbG9hdCBhUm93LCBmbG9hdCBiQ29sKSB7XG4gICAgICBmbG9hdCByZXN1bHQgPSAwLjA7XG4gICAgICBmb3IgKGZsb2F0IGkgPSAwLjA7IGkgPCBzaGFyZWREaW07IGkgKz0gMS4wKSB7XG4gICAgICAgIGZsb2F0IGEgPSBnZXRNYXRyaXhBKCR7YVNuaXBwZXR9KTtcbiAgICAgICAgZmxvYXQgYiA9IGdldE1hdHJpeEIoJHtiU25pcHBldH0pO1xuICAgICAgICByZXN1bHQgKz0gKGEgKiBiKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiByZXN1bHQ7XG4gICAgfVxuXG4gICAgdm9pZCBtYWluKCkge1xuICAgICAgdmVjMiByZXNSQyA9IGdldE91dHB1dENvb3JkcygpO1xuICAgICAgc2V0T3V0cHV0KGRvdEFSb3dCQ29sKHJlc1JDLngsIHJlc1JDLnkpKTtcbiAgICB9XG4gIGA7XG4gIHJldHVybiBzaGFkZXJfY29tcGlsZXIubWFrZVNoYWRlcihpbnB1dHMsIG91dCwgdXNlckNvZGUpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gbXVsdGlwbHlNYXRyaXgoXG4gICAgZ3BncHU6IEdQR1BVQ29udGV4dCwgbXVsdGlwbHlQcm9ncmFtOiBXZWJHTFByb2dyYW0sIGE6IFdlYkdMVGV4dHVyZSxcbiAgICBiOiBXZWJHTFRleHR1cmUsIHJlc3VsdDogV2ViR0xUZXh0dXJlLCBvdXRUZXhTaGFwZTogW251bWJlciwgbnVtYmVyXSkge1xuICBncGdwdS5zZXRPdXRwdXRNYXRyaXhUZXh0dXJlKHJlc3VsdCwgb3V0VGV4U2hhcGVbMF0sIG91dFRleFNoYXBlWzFdKTtcbiAgZ3BncHUuc2V0UHJvZ3JhbShtdWx0aXBseVByb2dyYW0pO1xuICBncGdwdS5zZXRJbnB1dE1hdHJpeFRleHR1cmUoYSwgJ21hdHJpeEEnLCAwKTtcbiAgZ3BncHUuc2V0SW5wdXRNYXRyaXhUZXh0dXJlKGIsICdtYXRyaXhCJywgMSk7XG4gIGdwZ3B1LmV4ZWN1dGVQcm9ncmFtKCk7XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCB7TWF0cml4T3JpZW50YXRpb259IGZyb20gJy4uL21hdGgnO1xuXG5pbXBvcnQge0dQR1BVQ29udGV4dH0gZnJvbSAnLi9ncGdwdV9jb250ZXh0JztcblxuZXhwb3J0IGZ1bmN0aW9uIGdldEZyYWdtZW50U2hhZGVyU291cmNlKFxuICAgIHNoYXJlZERpbWVuc2lvbjogbnVtYmVyLCBhT3JpZW50YXRpb246IE1hdHJpeE9yaWVudGF0aW9uLFxuICAgIGJPcmllbnRhdGlvbjogTWF0cml4T3JpZW50YXRpb24pOiBzdHJpbmcge1xuICAvKlxuICAgICAgQSA9IFswIDEgICBCID0gWzAgMSAgb3V0ID0gW0EwKkIwK0ExKkIyIEEwKkIxK0ExKkIzXG4gICAgICAgICAgIDIgM10gICAgICAgMiAzXSAgICAgICAgQTIqQjArQTEqQjIgQTIqQjErQXcqQjNdXG4gICAgICBvdXQuMCA9IEEwICogQjAgKyBBMSAqIEIyXG4gICAgICBvdXQuMSA9IEEwICogQjEgKyBBMSAqIEIzXG4gICAgICBvdXQuMiA9IEEyICogQjAgKyBBMyAqIEIyXG4gICAgICBvdXQuMyA9IEEyICogQjEgKyBBMyAqIEIzXG5cbiAgICAgIEEqQiAgICAgPSBBLnh4enogKiBCLnh5eHkgKyBBLnl5d3cgKiBCLnp3endcbiAgICAgIEFedCpCICAgPSBBLnh4eXkgKiBCLnh5eHkgKyBBLnp6d3cgKiBCLnp3endcbiAgICAgIEEqQl50ICAgPSBBLnh4enogKiBCLnh6eHogKyBBLnl5d3cgKiBCLnl3eXdcbiAgICAgIEFedCpCXnQgPSBBLnh4eXkgKiBCLnh6eHogKyBBLnp6d3cgKiBCLnl3eXdcbiAgICovXG4gIGNvbnN0IHNoYXJlZERpbWVuc2lvblBhY2tlZCA9IE1hdGguY2VpbChzaGFyZWREaW1lbnNpb24gLyAyKTtcbiAgY29uc3QgYVNhbXBsZSA9IChhT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID9cbiAgICAgICdjZW50ZXIsIHJlc3VsdFVWLnQnIDpcbiAgICAgICdyZXN1bHRVVi50LCBjZW50ZXInO1xuICBjb25zdCBiU2FtcGxlID0gKGJPcmllbnRhdGlvbiA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUikgP1xuICAgICAgJ3Jlc3VsdFVWLnMsIGNlbnRlcicgOlxuICAgICAgJ2NlbnRlciwgcmVzdWx0VVYucyc7XG4gIGNvbnN0IGFTd2l6emxlOiBbc3RyaW5nLCBzdHJpbmddID1cbiAgICAgIChhT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID8gWydhLnh4enonLCAnYS55eXd3J10gOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBbJ2EueHh5eScsICdhLnp6d3cnXTtcbiAgY29uc3QgYlN3aXp6bGU6IFtzdHJpbmcsIHN0cmluZ10gPVxuICAgICAgKGJPcmllbnRhdGlvbiA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUikgPyBbJ2IueHl4eScsICdiLnp3encnXSA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFsnYi54enh6JywgJ2IueXd5dyddO1xuICByZXR1cm4gYFxuICAgIHByZWNpc2lvbiBoaWdocCBmbG9hdDtcbiAgICB1bmlmb3JtIHNhbXBsZXIyRCBtYXRyaXhBO1xuICAgIHVuaWZvcm0gc2FtcGxlcjJEIG1hdHJpeEI7XG4gICAgdmFyeWluZyB2ZWMyIHJlc3VsdFVWO1xuXG4gICAgY29uc3QgZmxvYXQgc2hhcmVkRGltZW5zaW9uID0gJHtzaGFyZWREaW1lbnNpb25QYWNrZWR9LjA7XG5cbiAgICB2ZWM0IGRvdDJ4MkFSb3dCQ29sKCkge1xuICAgICAgdmVjNCByZXN1bHQgPSB2ZWM0KDAsIDAsIDAsIDApO1xuICAgICAgZm9yIChmbG9hdCBpID0gMC4wOyBpIDwgc2hhcmVkRGltZW5zaW9uOyBpICs9IDEuMCkge1xuICAgICAgICBmbG9hdCBjZW50ZXIgPSAoaSArIDAuNSkgLyBzaGFyZWREaW1lbnNpb247XG4gICAgICAgIHZlYzQgYSA9IHRleHR1cmUyRChtYXRyaXhBLCB2ZWMyKCR7YVNhbXBsZX0pKTtcbiAgICAgICAgdmVjNCBiID0gdGV4dHVyZTJEKG1hdHJpeEIsIHZlYzIoJHtiU2FtcGxlfSkpO1xuICAgICAgICByZXN1bHQgKz1cbiAgICAgICAgICAoJHthU3dpenpsZVswXX0gKiAke2JTd2l6emxlWzBdfSkgKyAoJHthU3dpenpsZVsxXX0gKiAke2JTd2l6emxlWzFdfSk7XG4gICAgICB9XG4gICAgICByZXR1cm4gcmVzdWx0O1xuICAgIH1cblxuICAgIHZvaWQgbWFpbigpIHtcbiAgICAgIGdsX0ZyYWdDb2xvciA9IGRvdDJ4MkFSb3dCQ29sKCk7XG4gICAgfWA7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBtdWx0aXBseU1hdHJpeFBhY2tlZChcbiAgICBncGdwdTogR1BHUFVDb250ZXh0LCBtdWx0aXBseVByb2dyYW06IFdlYkdMUHJvZ3JhbSwgYTogV2ViR0xUZXh0dXJlLFxuICAgIGI6IFdlYkdMVGV4dHVyZSwgcmVzdWx0OiBXZWJHTFRleHR1cmUsXG4gICAgcmVzdWx0U2hhcGVSb3dDb2w6IFtudW1iZXIsIG51bWJlcl0pIHtcbiAgZ3BncHUuc2V0T3V0cHV0UGFja2VkTWF0cml4VGV4dHVyZShcbiAgICAgIHJlc3VsdCwgcmVzdWx0U2hhcGVSb3dDb2xbMF0sIHJlc3VsdFNoYXBlUm93Q29sWzFdKTtcbiAgZ3BncHUuc2V0UHJvZ3JhbShtdWx0aXBseVByb2dyYW0pO1xuICBncGdwdS5zZXRJbnB1dE1hdHJpeFRleHR1cmUoYSwgJ21hdHJpeEEnLCAwKTtcbiAgZ3BncHUuc2V0SW5wdXRNYXRyaXhUZXh0dXJlKGIsICdtYXRyaXhCJywgMSk7XG4gIGdwZ3B1LmV4ZWN1dGVQcm9ncmFtKCk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB1cGxvYWRNdWx0aXBseU1hdHJpeFBhY2tlZERvd25sb2FkKFxuICAgIGE6IEZsb2F0MzJBcnJheSwgYVNoYXBlUm93Q29sOiBbbnVtYmVyLCBudW1iZXJdLCBiOiBGbG9hdDMyQXJyYXksXG4gICAgYlNoYXBlUm93Q29sOiBbbnVtYmVyLCBudW1iZXJdLCBhT3JpZW50YXRpb24gPSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSLFxuICAgIGJPcmllbnRhdGlvbiA9IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpOiBGbG9hdDMyQXJyYXkge1xuICBjb25zdCByZXN1bHROdW1Sb3dzID0gKGFPcmllbnRhdGlvbiA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUikgP1xuICAgICAgYVNoYXBlUm93Q29sWzBdIDpcbiAgICAgIGFTaGFwZVJvd0NvbFsxXTtcbiAgY29uc3QgcmVzdWx0TnVtQ29scyA9IChiT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID9cbiAgICAgIGJTaGFwZVJvd0NvbFsxXSA6XG4gICAgICBiU2hhcGVSb3dDb2xbMF07XG4gIGNvbnN0IHNoYXJlZERpbWVuc2lvbiA9IChhT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID9cbiAgICAgIGFTaGFwZVJvd0NvbFsxXSA6XG4gICAgICBhU2hhcGVSb3dDb2xbMF07XG5cbiAgY29uc3QgZ3BncHUgPSBuZXcgR1BHUFVDb250ZXh0KCk7XG4gIGNvbnN0IHByb2dyYW06IFdlYkdMUHJvZ3JhbSA9IGdwZ3B1LmNyZWF0ZVByb2dyYW0oXG4gICAgICBnZXRGcmFnbWVudFNoYWRlclNvdXJjZShzaGFyZWREaW1lbnNpb24sIGFPcmllbnRhdGlvbiwgYk9yaWVudGF0aW9uKSk7XG5cbiAgY29uc3QgYVRleHR1cmU6IFdlYkdMVGV4dHVyZSA9XG4gICAgICBncGdwdS5jcmVhdGVQYWNrZWRNYXRyaXhUZXh0dXJlKGFTaGFwZVJvd0NvbFswXSwgYVNoYXBlUm93Q29sWzFdKTtcbiAgY29uc3QgYlRleHR1cmU6IFdlYkdMVGV4dHVyZSA9XG4gICAgICBncGdwdS5jcmVhdGVQYWNrZWRNYXRyaXhUZXh0dXJlKGJTaGFwZVJvd0NvbFswXSwgYlNoYXBlUm93Q29sWzFdKTtcbiAgY29uc3QgcmVzdWx0VGV4dHVyZTogV2ViR0xUZXh0dXJlID1cbiAgICAgIGdwZ3B1LmNyZWF0ZVBhY2tlZE1hdHJpeFRleHR1cmUocmVzdWx0TnVtUm93cywgcmVzdWx0TnVtQ29scyk7XG5cbiAgZ3BncHUudXBsb2FkTWF0cml4VG9QYWNrZWRUZXh0dXJlKFxuICAgICAgYVRleHR1cmUsIGFTaGFwZVJvd0NvbFswXSwgYVNoYXBlUm93Q29sWzFdLCBhKTtcbiAgZ3BncHUudXBsb2FkTWF0cml4VG9QYWNrZWRUZXh0dXJlKFxuICAgICAgYlRleHR1cmUsIGJTaGFwZVJvd0NvbFswXSwgYlNoYXBlUm93Q29sWzFdLCBiKTtcblxuICBtdWx0aXBseU1hdHJpeFBhY2tlZChcbiAgICAgIGdwZ3B1LCBwcm9ncmFtLCBhVGV4dHVyZSwgYlRleHR1cmUsIHJlc3VsdFRleHR1cmUsXG4gICAgICBbcmVzdWx0TnVtUm93cywgcmVzdWx0TnVtQ29sc10pO1xuXG4gIGNvbnN0IHJlc3VsdCA9IGdwZ3B1LmRvd25sb2FkTWF0cml4RnJvbVBhY2tlZFRleHR1cmUoXG4gICAgICByZXN1bHRUZXh0dXJlLCByZXN1bHROdW1Sb3dzLCByZXN1bHROdW1Db2xzKTtcblxuICBncGdwdS5kZWxldGVNYXRyaXhUZXh0dXJlKGFUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZShiVGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUocmVzdWx0VGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZVByb2dyYW0ocHJvZ3JhbSk7XG4gIGdwZ3B1LmRpc3Bvc2UoKTtcblxuICByZXR1cm4gcmVzdWx0O1xufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQgKiBhcyBjb252X3V0aWwgZnJvbSAnLi4vY29udl91dGlsJztcbmltcG9ydCB7R1BHUFVDb250ZXh0fSBmcm9tICcuL2dwZ3B1X2NvbnRleHQnO1xuaW1wb3J0IHtJU19OQU5fU0hBREVSX0ZVTkN9IGZyb20gJy4vd2ViZ2xfdXRpbCc7XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRGcmFnbWVudFNoYWRlclBvb2xDb21tb25Tb3VyY2UoXG4gICAgeFNoYXBlUkNEOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGZTaXplOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLFxuICAgIHBhZDogbnVtYmVyLCBwb29sVHlwZTogJ21heCd8J21pbid8J2F2ZycsIGNvbXB1dGVQb3NpdGlvbnM6IGJvb2xlYW4pIHtcbiAgaWYgKHBvb2xUeXBlID09PSAnYXZnJyAmJiBjb21wdXRlUG9zaXRpb25zKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdDYW5ub3QgY29tcHV0ZSBwb3NpdGlvbnMgZm9yIGF2ZXJhZ2UgcG9vbC4nKTtcbiAgfVxuXG4gIGNvbnN0IGRlcHRoID0geFNoYXBlUkNEWzJdO1xuXG4gIGNvbnN0IHhUZXhTaGFwZVJDID0gY29udl91dGlsLmNvbXB1dGVUZXhTaGFwZUZyb20zRCh4U2hhcGVSQ0QpO1xuXG4gIGxldCByZXR1cm5WYWx1ZSA9ICdtaW5NYXhWYWx1ZSc7XG4gIGlmIChjb21wdXRlUG9zaXRpb25zKSB7XG4gICAgcmV0dXJuVmFsdWUgPSAnbWluTWF4UG9zaXRpb24nO1xuICB9IGVsc2UgaWYgKHBvb2xUeXBlID09PSAnYXZnJykge1xuICAgIHJldHVyblZhbHVlID0gJ2F2Z1ZhbHVlJztcbiAgfVxuXG4gIHJldHVybiBgXG4gICAgcHJlY2lzaW9uIGhpZ2hwIGZsb2F0O1xuICAgIHVuaWZvcm0gc2FtcGxlcjJEIHg7XG4gICAgdmFyeWluZyB2ZWMyIHJlc3VsdFVWO1xuXG4gICAgY29uc3QgdmVjMiBoYWxmQ1IgPSB2ZWMyKDAuNSwgMC41KTtcbiAgICBjb25zdCB2ZWMyIHhTaGFwZUNSID0gdmVjMigke3hUZXhTaGFwZVJDWzFdfSwgJHt4VGV4U2hhcGVSQ1swXX0pO1xuXG4gICAgJHtJU19OQU5fU0hBREVSX0ZVTkN9XG5cbiAgICB2b2lkIG1haW4oKSB7XG4gICAgICB2ZWMyIHlUZXhDUiA9IGZsb29yKGdsX0ZyYWdDb29yZC54eSk7XG5cbiAgICAgIC8vIE1hcCBmcm9tIDJEICh5VGV4UiwgeVRleEMpIHRvIDNEICh5UiwgeUMsIGQyKS5cbiAgICAgIGZsb2F0IHlSID0geVRleENSLnk7XG4gICAgICBmbG9hdCB5QyA9IGZsb29yKHlUZXhDUi54IC8gJHtkZXB0aH0uMCk7XG4gICAgICBmbG9hdCBkID0gbW9kKHlUZXhDUi54LCAke2RlcHRofS4wKTtcblxuICAgICAgdmVjMiB4UkNDb3JuZXIgPSB2ZWMyKHlSLCB5QykgKiB2ZWMyKCR7c3RyaWRlfSwgJHtzdHJpZGV9KSAtXG4gICAgICAgICAgdmVjMigke3BhZH0uMCwgJHtwYWR9LjApO1xuICAgICAgZmxvYXQgeFJDb3JuZXIgPSB4UkNDb3JuZXIueDtcbiAgICAgIGZsb2F0IHhDQ29ybmVyID0geFJDQ29ybmVyLnk7XG5cbiAgICAgIC8vIG1heC9taW4geCg/LCA/LCBkKSB0byBnZXQgeSh5UiwgeUMsIGQpLlxuICAgICAgLy8gPyA9IHRvIGJlIGRldGVybWluZWRcbiAgICAgIGZsb2F0IG1pbk1heFZhbHVlID0gMC4wO1xuICAgICAgZmxvYXQgbWluTWF4VmFsdWVGb3VuZCA9IDAuMDtcbiAgICAgIGZsb2F0IG1pbk1heFBvc2l0aW9uID0gMC4wO1xuICAgICAgZmxvYXQgYXZnVmFsdWUgPSAwLjA7XG5cbiAgICAgIGZvciAoZmxvYXQgd1IgPSAwLjA7IHdSIDwgJHtmU2l6ZX0uMDsgd1IgKz0gMS4wKSB7XG4gICAgICAgIGZsb2F0IHhSID0geFJDb3JuZXIgKyB3UjtcbiAgICAgICAgZmxvYXQgeFRleFIgPSB4UjtcblxuICAgICAgICBmb3IgKGZsb2F0IHdDID0gMC4wOyB3QyA8ICR7ZlNpemV9LjA7IHdDICs9IDEuMCkge1xuICAgICAgICAgIGZsb2F0IHhDID0geENDb3JuZXIgKyB3QztcbiAgICAgICAgICBmbG9hdCB4VGV4QyA9IHhDICogJHtkZXB0aH0uMCArIGQ7XG5cbiAgICAgICAgICB2ZWMyIHRleENSID0gdmVjMih4VGV4QywgeFRleFIpO1xuXG4gICAgICAgICAgLy8gQ2hlY2sgaWYgdGhlIHJlcXVlc3RlZCBVViBpcyBpbnZhbGlkLlxuICAgICAgICAgIHZlYzIgdXYgPSAodGV4Q1IgKyBoYWxmQ1IpIC8geFNoYXBlQ1I7XG4gICAgICAgICAgYm9vbCBsZXNzVGhhblplcm8gPSBhbnkobGVzc1RoYW4odXYsIHZlYzIoMCwgMCkpKTtcbiAgICAgICAgICBib29sIGdyZWF0ZXJUaGFuT25lID0gYW55KGdyZWF0ZXJUaGFuKHV2LCB2ZWMyKDEsIDEpKSk7XG4gICAgICAgICAgYm9vbCBvdXRzaWRlID0gbGVzc1RoYW5aZXJvIHx8IGdyZWF0ZXJUaGFuT25lO1xuICAgICAgICAgIGlmIChvdXRzaWRlKSB7XG4gICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICB9XG5cbiAgICAgICAgICBmbG9hdCB2YWx1ZSA9IHRleHR1cmUyRCh4LCB1dikucjtcbiAgICAgICAgICBpZiAoaXNOYU4odmFsdWUpKSB7XG4gICAgICAgICAgICBnbF9GcmFnQ29sb3IgPSB2ZWM0KHZhbHVlLCAwLCAwLCAwKTtcbiAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgICB9XG4gICAgICAgICAgaWYgKCR7cG9vbFR5cGUgPT09ICdhdmcnfSkge1xuICAgICAgICAgICAgYXZnVmFsdWUgKz0gdmFsdWUgLyAke2ZTaXplICogZlNpemV9LjA7XG4gICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIC8vIElmIGEgbWluIC8gbWF4IHZhbHVlIGhhcyBhbHJlYWR5IGJlZW4gZm91bmQsIHVzZSBpdC4gSWYgbm90LCB1c2VcbiAgICAgICAgICAgIC8vIHRoZSBjdXJyZW50IHZhbHVlLlxuICAgICAgICAgICAgZmxvYXQgY3VycmVudE1pbk1heFZhbHVlID0gbWl4KFxuICAgICAgICAgICAgICAgIHZhbHVlLCBtaW5NYXhWYWx1ZSwgbWluTWF4VmFsdWVGb3VuZCk7XG4gICAgICAgICAgICBpZiAodmFsdWUgJHtwb29sVHlwZSA9PT0gJ21pbicgPyAnPD0nIDogJz49J30gY3VycmVudE1pbk1heFZhbHVlKSB7XG4gICAgICAgICAgICAgIG1pbk1heFZhbHVlID0gdmFsdWU7XG4gICAgICAgICAgICAgIG1pbk1heFZhbHVlRm91bmQgPSAxLjA7XG4gICAgICAgICAgICAgIGlmICgke2NvbXB1dGVQb3NpdGlvbnN9KSB7XG4gICAgICAgICAgICAgICAgbWluTWF4UG9zaXRpb24gPSB3UiAqICR7ZlNpemV9LjAgKyB3QztcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgZ2xfRnJhZ0NvbG9yID0gdmVjNCgke3JldHVyblZhbHVlfSwgMCwgMCwgMCk7XG4gICAgfWA7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBwb29sQ29tbW9uKFxuICAgIGdwZ3B1OiBHUEdQVUNvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSwgeDogV2ViR0xUZXh0dXJlLFxuICAgIHJlc3VsdDogV2ViR0xUZXh0dXJlLCByZXN1bHRTaGFwZVJvd0NvbDogW251bWJlciwgbnVtYmVyXSkge1xuICBncGdwdS5zZXRPdXRwdXRNYXRyaXhUZXh0dXJlKFxuICAgICAgcmVzdWx0LCByZXN1bHRTaGFwZVJvd0NvbFswXSwgcmVzdWx0U2hhcGVSb3dDb2xbMV0pO1xuICBncGdwdS5zZXRQcm9ncmFtKHByb2dyYW0pO1xuICBncGdwdS5zZXRJbnB1dE1hdHJpeFRleHR1cmUoeCwgJ3gnLCAwKTtcbiAgZ3BncHUuZXhlY3V0ZVByb2dyYW0oKTtcbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi8uLi91dGlsJztcbmltcG9ydCB7TkRBcnJheX0gZnJvbSAnLi4vbmRhcnJheSc7XG5cbmV4cG9ydCB0eXBlIElucHV0ID0ge1xuICBuYW1lOiBzdHJpbmc7IGFycmF5OiBOREFycmF5O1xufTtcblxuZXhwb3J0IGZ1bmN0aW9uIG1ha2VTaGFkZXJLZXkoaW5wdXRzOiBOREFycmF5W10sIG91dHB1dDogTkRBcnJheSk6IHN0cmluZyB7XG4gIGNvbnN0IGlucyA9IGlucHV0cy5tYXAoeCA9PiB4LnNoYXBlICsgJ18nICsgeC5nZXRUZXh0dXJlU2hhcGVSQygpKTtcbiAgcmV0dXJuIGlucy5qb2luKCdfJykgKyAnXycgKyBvdXRwdXQuc2hhcGUgKyAnXycgKyBvdXRwdXQuZ2V0VGV4dHVyZVNoYXBlUkMoKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIG1ha2VTaGFkZXIoXG4gICAgaW5wdXRzOiBJbnB1dFtdLCBvdXRwdXQ6IE5EQXJyYXksIHVzZXJDb2RlOiBzdHJpbmcpOiBzdHJpbmcge1xuICBjb25zdCBpbnB1dFByZWZpeFNuaXBwZXQgPVxuICAgICAgaW5wdXRzLm1hcCh4ID0+IGB1bmlmb3JtIHNhbXBsZXIyRCAke3gubmFtZX07YCkuam9pbignXFxuJyk7XG4gIGNvbnN0IGlucHV0U2FtcGxpbmdTbmlwcGV0ID1cbiAgICAgIGlucHV0cy5tYXAoeCA9PiBnZXRJbnB1dFNhbXBsaW5nU25pcHBldCh4KSkuam9pbignXFxuJyk7XG4gIGNvbnN0IG91dFRleFNoYXBlID0gb3V0cHV0LmdldFRleHR1cmVTaGFwZVJDKCk7XG4gIGNvbnN0IG91dHB1dFNhbXBsaW5nU25pcHBldCA9XG4gICAgICBnZXRPdXRwdXRTYW1wbGluZ1NuaXBwZXQob3V0cHV0LnNoYXBlLCBvdXRUZXhTaGFwZSk7XG4gIGNvbnN0IHNvdXJjZSA9IFtcbiAgICBTSEFERVJfUFJFRklYLCBpbnB1dFByZWZpeFNuaXBwZXQsIFNBTVBMRV8yRF9TTklQUEVULCBpbnB1dFNhbXBsaW5nU25pcHBldCxcbiAgICBvdXRwdXRTYW1wbGluZ1NuaXBwZXQsIHVzZXJDb2RlXG4gIF0uam9pbignXFxuJyk7XG4gIHJldHVybiBzb3VyY2U7XG59XG5cbmZ1bmN0aW9uIGdldElucHV0U2FtcGxpbmdTbmlwcGV0KGlucHV0OiBJbnB1dCkge1xuICBjb25zdCBhcnIgPSBpbnB1dC5hcnJheTtcbiAgY29uc3Qgc2hhcGUgPSBhcnIuc2hhcGU7XG4gIGNvbnN0IHRleFNoYXBlID0gYXJyLmdldFRleHR1cmVTaGFwZVJDKHNoYXBlIGFzIFtudW1iZXIsIG51bWJlcl0pO1xuICBzd2l0Y2ggKHNoYXBlLmxlbmd0aCkge1xuICAgIGNhc2UgMjpcbiAgICAgIHJldHVybiBnZXRTYW1wbGVyMkQoaW5wdXQubmFtZSwgc2hhcGUgYXMgW251bWJlciwgbnVtYmVyXSwgdGV4U2hhcGUpO1xuICAgIGRlZmF1bHQ6XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYCR7YXJyLnJhbmt9LUQgaW5wdXQgc2FtcGxpbmcgaXMgbm90IHlldCBzdXBwb3J0ZWRgKTtcbiAgfVxufVxuXG5mdW5jdGlvbiBnZXRPdXRwdXRTYW1wbGluZ1NuaXBwZXQoXG4gICAgb3V0U2hhcGU6IG51bWJlcltdLCBvdXRUZXhTaGFwZTogW251bWJlciwgbnVtYmVyXSk6IHN0cmluZyB7XG4gIHN3aXRjaCAob3V0U2hhcGUubGVuZ3RoKSB7XG4gICAgY2FzZSAyOlxuICAgICAgcmV0dXJuIGdldE91dHB1dDJEQ29vcmRzKG91dFNoYXBlIGFzIFtudW1iZXIsIG51bWJlcl0sIG91dFRleFNoYXBlKTtcbiAgICBkZWZhdWx0OlxuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGAke291dFNoYXBlLmxlbmd0aH0tRCBvdXRwdXQgc2FtcGxpbmcgaXMgbm90IHlldCBzdXBwb3J0ZWRgKTtcbiAgfVxufVxuXG5jb25zdCBTSEFERVJfUFJFRklYID0gYFxuICBwcmVjaXNpb24gaGlnaHAgZmxvYXQ7XG4gIHZhcnlpbmcgdmVjMiByZXN1bHRVVjtcbiAgY29uc3QgdmVjMiBoYWxmQ1IgPSB2ZWMyKDAuNSwgMC41KTtcblxuICB2b2lkIHNldE91dHB1dChmbG9hdCB2YWwpIHtcbiAgICBnbF9GcmFnQ29sb3IgPSB2ZWM0KHZhbCwgMCwgMCwgMCk7XG4gIH1cbmA7XG5cbmNvbnN0IFNBTVBMRV8yRF9TTklQUEVUID0gYFxuICBmbG9hdCBzYW1wbGUyRChzYW1wbGVyMkQgdGV4dHVyZSwgZmxvYXQgdGV4TnVtUiwgZmxvYXQgdGV4TnVtQywgZmxvYXQgbnVtQyxcbiAgICAgIGZsb2F0IHJvdywgZmxvYXQgY29sKSB7XG4gICAgZmxvYXQgaW5kZXggPSBkb3QodmVjMihyb3csIGNvbCksIHZlYzIobnVtQywgMS4wKSk7XG4gICAgZmxvYXQgdGV4UiA9IGZsb29yKGluZGV4IC8gdGV4TnVtQyk7XG4gICAgZmxvYXQgdGV4QyA9IG1vZChpbmRleCwgdGV4TnVtQyk7XG4gICAgdmVjMiB1diA9ICh2ZWMyKHRleEMsIHRleFIpICsgaGFsZkNSKSAvIHZlYzIodGV4TnVtQywgdGV4TnVtUik7XG4gICAgcmV0dXJuIHRleHR1cmUyRCh0ZXh0dXJlLCB1dikucjtcbiAgfVxuYDtcblxuZnVuY3Rpb24gZ2V0T3V0cHV0MkRDb29yZHMoXG4gICAgc2hhcGU6IFtudW1iZXIsIG51bWJlcl0sIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdKSB7XG4gIGlmICh1dGlsLmFycmF5c0VxdWFsKHNoYXBlLCB0ZXhTaGFwZSkpIHtcbiAgICByZXR1cm4gYFxuICAgICAgdmVjMiBnZXRPdXRwdXRDb29yZHMoKSB7XG4gICAgICAgIHJldHVybiBmbG9vcihnbF9GcmFnQ29vcmQueXgpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgcmV0dXJuIGBcbiAgICB2ZWMyIGdldE91dHB1dENvb3JkcygpIHtcbiAgICAgIHZlYzIgcmVzVGV4UkMgPSBmbG9vcihnbF9GcmFnQ29vcmQueXgpO1xuICAgICAgZmxvYXQgaW5kZXggPSBkb3QocmVzVGV4UkMsIHZlYzIoJHt0ZXhTaGFwZVsxXX0uMCwgMS4wKSk7XG4gICAgICBmbG9hdCByID0gZmxvb3IoaW5kZXggLyAke3NoYXBlWzFdfS4wKTtcbiAgICAgIGZsb2F0IGMgPSBtb2QoaW5kZXgsICR7c2hhcGVbMV19LjApO1xuICAgICAgcmV0dXJuIHZlYzIociwgYyk7XG4gICAgfVxuICBgO1xufVxuXG5mdW5jdGlvbiBnZXRTYW1wbGVyMkQoXG4gICAgdGV4TmFtZTogc3RyaW5nLCBzaGFwZTogW251bWJlciwgbnVtYmVyXSwgdGV4U2hhcGU6IFtudW1iZXIsIG51bWJlcl0pIHtcbiAgY29uc3QgZnVuY05hbWUgPSAnZ2V0JyArIHRleE5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkgKyB0ZXhOYW1lLnNsaWNlKDEpO1xuICBjb25zdCB0UiA9IHRleFNoYXBlWzBdO1xuICBjb25zdCB0QyA9IHRleFNoYXBlWzFdO1xuICBpZiAodXRpbC5hcnJheXNFcXVhbChzaGFwZSwgdGV4U2hhcGUpKSB7XG4gICAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGZsb2F0IHJvdywgZmxvYXQgY29sKSB7XG4gICAgICAgIHZlYzIgdXYgPSAodmVjMihjb2wsIHJvdykgKyBoYWxmQ1IpIC8gdmVjMigke3RDfS4wLCAke3RSfS4wKTtcbiAgICAgICAgcmV0dXJuIHRleHR1cmUyRCgke3RleE5hbWV9LCB1dikucjtcbiAgICAgIH1cbiAgICBgO1xuICB9XG4gIHJldHVybiBgXG4gICAgZmxvYXQgJHtmdW5jTmFtZX0oZmxvYXQgcm93LCBmbG9hdCBjb2wpIHtcbiAgICAgIHJldHVybiBzYW1wbGUyRCgke3RleE5hbWV9LCAke3RSfS4wLCAke3RDfS4wLCAke3NoYXBlWzFdfS4wLCByb3csIGNvbCk7XG4gICAgfVxuICBgO1xufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5leHBvcnQgZnVuY3Rpb24gZ2V0VW5wYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChcbiAgICByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IFtudW1iZXIsIG51bWJlcl0ge1xuICByZXR1cm4gW2NvbHVtbnMsIHJvd3NdO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0VW5wYWNrZWRBcnJheVNpemVGcm9tTWF0cml4U2l6ZShcbiAgICBtYXRyaXhTaXplOiBudW1iZXIsIGNoYW5uZWxzUGVyVGV4dHVyZTogbnVtYmVyKTogbnVtYmVyIHtcbiAgcmV0dXJuIG1hdHJpeFNpemUgKiBjaGFubmVsc1BlclRleHR1cmU7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRDb2xvck1hdHJpeFRleHR1cmVTaGFwZVdpZHRoSGVpZ2h0KFxuICAgIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTogW251bWJlciwgbnVtYmVyXSB7XG4gIHJldHVybiBbY29sdW1ucyAqIDQsIHJvd3NdO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0TWF0cml4U2l6ZUZyb21VbnBhY2tlZEFycmF5U2l6ZShcbiAgICB1bnBhY2tlZFNpemU6IG51bWJlciwgY2hhbm5lbHNQZXJUZXh0dXJlOiBudW1iZXIpOiBudW1iZXIge1xuICBpZiAodW5wYWNrZWRTaXplICUgY2hhbm5lbHNQZXJUZXh0dXJlICE9PSAwKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAndW5wYWNrZWRTaXplICgnICsgdW5wYWNrZWRTaXplICsgJykgbXVzdCBiZSBhIG11bHRpcGxlIG9mICcgK1xuICAgICAgICBjaGFubmVsc1BlclRleHR1cmUpO1xuICB9XG4gIHJldHVybiB1bnBhY2tlZFNpemUgLyBjaGFubmVsc1BlclRleHR1cmU7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBlbmNvZGVNYXRyaXhUb1VucGFja2VkQXJyYXkoXG4gICAgbWF0cml4OiBGbG9hdDMyQXJyYXksIHVucGFja2VkQXJyYXk6IEZsb2F0MzJBcnJheSxcbiAgICBjaGFubmVsc1BlclRleHR1cmU6IG51bWJlcikge1xuICBjb25zdCByZXF1aXJlZFNpemUgPVxuICAgICAgZ2V0VW5wYWNrZWRBcnJheVNpemVGcm9tTWF0cml4U2l6ZShtYXRyaXgubGVuZ3RoLCBjaGFubmVsc1BlclRleHR1cmUpO1xuICBpZiAodW5wYWNrZWRBcnJheS5sZW5ndGggPCByZXF1aXJlZFNpemUpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICd1bnBhY2tlZEFycmF5IGxlbmd0aCAoJyArIHVucGFja2VkQXJyYXkubGVuZ3RoICtcbiAgICAgICAgJykgbXVzdCBiZSA+PSAnICsgcmVxdWlyZWRTaXplKTtcbiAgfVxuICBsZXQgZHN0ID0gMDtcbiAgZm9yIChsZXQgc3JjID0gMDsgc3JjIDwgbWF0cml4Lmxlbmd0aDsgKytzcmMpIHtcbiAgICB1bnBhY2tlZEFycmF5W2RzdF0gPSBtYXRyaXhbc3JjXTtcbiAgICBkc3QgKz0gY2hhbm5lbHNQZXJUZXh0dXJlO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBkZWNvZGVNYXRyaXhGcm9tVW5wYWNrZWRBcnJheShcbiAgICB1bnBhY2tlZEFycmF5OiBGbG9hdDMyQXJyYXksIG1hdHJpeDogRmxvYXQzMkFycmF5LFxuICAgIGNoYW5uZWxzUGVyVGV4dHVyZTogbnVtYmVyKSB7XG4gIGNvbnN0IHJlcXVpcmVkU2l6ZSA9IGdldE1hdHJpeFNpemVGcm9tVW5wYWNrZWRBcnJheVNpemUoXG4gICAgICB1bnBhY2tlZEFycmF5Lmxlbmd0aCwgY2hhbm5lbHNQZXJUZXh0dXJlKTtcbiAgaWYgKG1hdHJpeC5sZW5ndGggPCByZXF1aXJlZFNpemUpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICdtYXRyaXggbGVuZ3RoICgnICsgbWF0cml4Lmxlbmd0aCArICcpIG11c3QgYmUgPj0gJyArIHJlcXVpcmVkU2l6ZSk7XG4gIH1cbiAgbGV0IGRzdCA9IDA7XG4gIGZvciAobGV0IHNyYyA9IDA7IHNyYyA8IHVucGFja2VkQXJyYXkubGVuZ3RoOyBzcmMgKz0gY2hhbm5lbHNQZXJUZXh0dXJlKSB7XG4gICAgbWF0cml4W2RzdCsrXSA9IHVucGFja2VkQXJyYXlbc3JjXTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0UGFja2VkTWF0cml4VGV4dHVyZVNoYXBlV2lkdGhIZWlnaHQoXG4gICAgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBbbnVtYmVyLCBudW1iZXJdIHtcbiAgcmV0dXJuIFtNYXRoLmNlaWwoY29sdW1ucyAvIDIpLCBNYXRoLmNlaWwocm93cyAvIDIpXTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldFBhY2tlZFJHQkFBcnJheVNpemVGcm9tTWF0cml4U2hhcGUoXG4gICAgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBudW1iZXIge1xuICBjb25zdCBbdywgaF0gPSBnZXRQYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChyb3dzLCBjb2x1bW5zKTtcbiAgcmV0dXJuIHcgKiBoICogNDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGVuY29kZU1hdHJpeFRvUGFja2VkUkdCQShcbiAgICBtYXRyaXg6IEZsb2F0MzJBcnJheSwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIsXG4gICAgcGFja2VkUkdCQTogRmxvYXQzMkFycmF5KSB7XG4gIGNvbnN0IHJlcXVpcmVkU2l6ZSA9IGdldFBhY2tlZFJHQkFBcnJheVNpemVGcm9tTWF0cml4U2hhcGUocm93cywgY29sdW1ucyk7XG4gIGlmIChwYWNrZWRSR0JBLmxlbmd0aCA8IHJlcXVpcmVkU2l6ZSkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgJ3BhY2tlZFJHQkEgbGVuZ3RoICgnICsgcGFja2VkUkdCQS5sZW5ndGggK1xuICAgICAgICAnKSBtdXN0IGJlID49ICcgKyByZXF1aXJlZFNpemUpO1xuICB9XG4gIC8qXG4gICAgVW5wYWNrZWQgbWF0cml4LCByb3ctbWFqb3Igb3JkZXIgaW4gRmxvYXQzMkFycmF5WzE2XTogIEEgQiBDIERcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgRSBGIEcgSFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBJIEogSyBMXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIE0gTiBPIFBcblxuICAgIFBhY2tlZCBtYXRyaXgsIDJ4MiBSR0JBMzIgdGV4dHVyZSAobWVtb3J5IHZpZXcpOiAgICAgICBBQkVGIENER0ggSUpNTiBLTE9QXG5cbiAgICBQYWNrZWQgbWF0cml4LCAyeDIgUkdCQTMyIHRleHR1cmUgKG1hdHJpeCB2aWV3KTogICAgICAgQUJ8Q0RcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgRUZ8R0hcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgLS0rLS1cbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgSUp8S0xcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgTU58T1BcbiAgICovXG4gIGNvbnN0IFt0ZXh0dXJlV2lkdGgsIHRleHR1cmVIZWlnaHRdID1cbiAgICAgIGdldFBhY2tlZE1hdHJpeFRleHR1cmVTaGFwZVdpZHRoSGVpZ2h0KHJvd3MsIGNvbHVtbnMpO1xuICBjb25zdCBvZGRXaWR0aCA9IChjb2x1bW5zICUgMikgPT09IDE7XG4gIGNvbnN0IG9kZEhlaWdodCA9IChyb3dzICUgMikgPT09IDE7XG4gIGNvbnN0IHdpZHRoSW5GdWxsQmxvY2tzID0gTWF0aC5mbG9vcihjb2x1bW5zIC8gMik7XG4gIGNvbnN0IGhlaWdodEluRnVsbEJsb2NrcyA9IE1hdGguZmxvb3Iocm93cyAvIDIpO1xuXG4gIC8vIGxvb3Agb3ZlciBmdWxsIDJ4MiBibG9ja3NcbiAge1xuICAgIGNvbnN0IGRzdFN0cmlkZSA9IChvZGRXaWR0aCA/IDQgOiAwKTtcbiAgICBjb25zdCBvbmVSb3cgPSBjb2x1bW5zO1xuICAgIGxldCBkc3QgPSAwO1xuICAgIGZvciAobGV0IGJsb2NrWSA9IDA7IGJsb2NrWSA8IGhlaWdodEluRnVsbEJsb2NrczsgKytibG9ja1kpIHtcbiAgICAgIGNvbnN0IG1hdHJpeFNyY1JvdyA9IChibG9ja1kgKiAyICogY29sdW1ucyk7XG4gICAgICBmb3IgKGxldCBibG9ja1ggPSAwOyBibG9ja1ggPCB3aWR0aEluRnVsbEJsb2NrczsgKytibG9ja1gpIHtcbiAgICAgICAgY29uc3QgbWF0cml4U3JjQ29sID0gYmxvY2tYICogMjtcbiAgICAgICAgY29uc3Qgc3JjID0gbWF0cml4U3JjUm93ICsgbWF0cml4U3JjQ29sO1xuICAgICAgICBwYWNrZWRSR0JBW2RzdF0gPSBtYXRyaXhbc3JjXTtcbiAgICAgICAgcGFja2VkUkdCQVtkc3QgKyAxXSA9IG1hdHJpeFtzcmMgKyAxXTtcbiAgICAgICAgcGFja2VkUkdCQVtkc3QgKyAyXSA9IG1hdHJpeFtzcmMgKyBvbmVSb3ddO1xuICAgICAgICBwYWNrZWRSR0JBW2RzdCArIDNdID0gbWF0cml4W3NyYyArIG9uZVJvdyArIDFdO1xuICAgICAgICBkc3QgKz0gNDtcbiAgICAgIH1cbiAgICAgIGRzdCArPSBkc3RTdHJpZGU7XG4gICAgfVxuICB9XG5cbiAgLy8gbG9vcCBkb3duIGZpbmFsIG9kZCBjb2x1bW5cbiAgaWYgKG9kZFdpZHRoKSB7XG4gICAgbGV0IHNyYyA9IGNvbHVtbnMgLSAxO1xuICAgIGxldCBkc3QgPSAodGV4dHVyZVdpZHRoIC0gMSkgKiA0O1xuICAgIGNvbnN0IHNyY1N0cmlkZSA9IDIgKiBjb2x1bW5zO1xuICAgIGNvbnN0IGRzdFN0cmlkZSA9IHRleHR1cmVXaWR0aCAqIDQ7XG4gICAgZm9yIChsZXQgYmxvY2tZID0gMDsgYmxvY2tZIDwgaGVpZ2h0SW5GdWxsQmxvY2tzOyArK2Jsb2NrWSkge1xuICAgICAgcGFja2VkUkdCQVtkc3RdID0gbWF0cml4W3NyY107XG4gICAgICBwYWNrZWRSR0JBW2RzdCArIDJdID0gbWF0cml4W3NyYyArIGNvbHVtbnNdO1xuICAgICAgc3JjICs9IHNyY1N0cmlkZTtcbiAgICAgIGRzdCArPSBkc3RTdHJpZGU7XG4gICAgfVxuICB9XG5cbiAgLy8gbG9vcCBhY3Jvc3MgZmluYWwgcm93XG4gIGlmIChvZGRIZWlnaHQpIHtcbiAgICBsZXQgc3JjID0gKHJvd3MgLSAxKSAqIGNvbHVtbnM7XG4gICAgbGV0IGRzdCA9ICh0ZXh0dXJlSGVpZ2h0IC0gMSkgKiB0ZXh0dXJlV2lkdGggKiA0O1xuICAgIGZvciAobGV0IGJsb2NrWCA9IDA7IGJsb2NrWCA8IHdpZHRoSW5GdWxsQmxvY2tzOyArK2Jsb2NrWCkge1xuICAgICAgcGFja2VkUkdCQVtkc3QrK10gPSBtYXRyaXhbc3JjKytdO1xuICAgICAgcGFja2VkUkdCQVtkc3QrK10gPSBtYXRyaXhbc3JjKytdO1xuICAgICAgZHN0ICs9IDI7XG4gICAgfVxuICB9XG5cbiAgLy8gZmlsbCBpbiBib3R0b20tcmlnaHQgdGV4ZWxcbiAgaWYgKG9kZFdpZHRoICYmIG9kZEhlaWdodCkge1xuICAgIHBhY2tlZFJHQkFbcGFja2VkUkdCQS5sZW5ndGggLSA0XSA9IG1hdHJpeFttYXRyaXgubGVuZ3RoIC0gMV07XG4gIH1cblxuICByZXR1cm4gcGFja2VkUkdCQTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGRlY29kZU1hdHJpeEZyb21QYWNrZWRSR0JBKFxuICAgIHBhY2tlZFJHQkE6IEZsb2F0MzJBcnJheSwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIsXG4gICAgbWF0cml4OiBGbG9hdDMyQXJyYXkpOiBGbG9hdDMyQXJyYXkge1xuICBjb25zdCByZXF1aXJlZFNpemUgPSByb3dzICogY29sdW1ucztcbiAgaWYgKHJlcXVpcmVkU2l6ZSA8IG1hdHJpeC5sZW5ndGgpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICdtYXRyaXggbGVuZ3RoICgnICsgbWF0cml4Lmxlbmd0aCArICcpIG11c3QgYmUgPj0gJyArIHJlcXVpcmVkU2l6ZSk7XG4gIH1cbiAgY29uc3Qgb2RkV2lkdGggPSAoY29sdW1ucyAlIDIpID09PSAxO1xuICBjb25zdCBvZGRIZWlnaHQgPSAocm93cyAlIDIpID09PSAxO1xuICBjb25zdCB3aWR0aEluRnVsbEJsb2NrcyA9IE1hdGguZmxvb3IoY29sdW1ucyAvIDIpO1xuICBjb25zdCBoZWlnaHRJbkZ1bGxCbG9ja3MgPSBNYXRoLmZsb29yKHJvd3MgLyAyKTtcbiAgY29uc3QgW3RleHR1cmVXaWR0aCwgdGV4dHVyZUhlaWdodF0gPVxuICAgICAgZ2V0UGFja2VkTWF0cml4VGV4dHVyZVNoYXBlV2lkdGhIZWlnaHQocm93cywgY29sdW1ucyk7XG5cbiAgLy8gbG9vcCBvdmVyIGZ1bGwgMngyIGJsb2Nrc1xuICB7XG4gICAgY29uc3Qgc3JjU3RyaWRlID0gb2RkV2lkdGggPyA0IDogMDtcbiAgICBjb25zdCBkc3RTdHJpZGUgPSBjb2x1bW5zICsgKG9kZFdpZHRoID8gMSA6IDApO1xuICAgIGxldCBzcmMgPSAwO1xuICAgIGxldCBkc3RSb3cxID0gMDtcbiAgICBsZXQgZHN0Um93MiA9IGNvbHVtbnM7XG4gICAgZm9yIChsZXQgYmxvY2tZID0gMDsgYmxvY2tZIDwgaGVpZ2h0SW5GdWxsQmxvY2tzOyArK2Jsb2NrWSkge1xuICAgICAgZm9yIChsZXQgYmxvY2tYID0gMDsgYmxvY2tYIDwgd2lkdGhJbkZ1bGxCbG9ja3M7ICsrYmxvY2tYKSB7XG4gICAgICAgIG1hdHJpeFtkc3RSb3cxKytdID0gcGFja2VkUkdCQVtzcmMrK107XG4gICAgICAgIG1hdHJpeFtkc3RSb3cxKytdID0gcGFja2VkUkdCQVtzcmMrK107XG4gICAgICAgIG1hdHJpeFtkc3RSb3cyKytdID0gcGFja2VkUkdCQVtzcmMrK107XG4gICAgICAgIG1hdHJpeFtkc3RSb3cyKytdID0gcGFja2VkUkdCQVtzcmMrK107XG4gICAgICB9XG4gICAgICBzcmMgKz0gc3JjU3RyaWRlO1xuICAgICAgZHN0Um93MSArPSBkc3RTdHJpZGU7XG4gICAgICBkc3RSb3cyICs9IGRzdFN0cmlkZTtcbiAgICB9XG4gIH1cblxuICAvLyBsb29wIGRvd24gZmluYWwgY29sdW1uXG4gIGlmIChvZGRXaWR0aCkge1xuICAgIGxldCBzcmMgPSAodGV4dHVyZVdpZHRoIC0gMSkgKiA0O1xuICAgIGxldCBkc3QgPSBjb2x1bW5zIC0gMTtcbiAgICBjb25zdCBzcmNTdHJpZGUgPSB0ZXh0dXJlV2lkdGggKiA0O1xuICAgIGNvbnN0IGRzdFN0cmlkZSA9IDIgKiBjb2x1bW5zO1xuICAgIGZvciAobGV0IGJsb2NrWSA9IDA7IGJsb2NrWSA8IGhlaWdodEluRnVsbEJsb2NrczsgKytibG9ja1kpIHtcbiAgICAgIG1hdHJpeFtkc3RdID0gcGFja2VkUkdCQVtzcmNdO1xuICAgICAgbWF0cml4W2RzdCArIGNvbHVtbnNdID0gcGFja2VkUkdCQVtzcmMgKyAyXTtcbiAgICAgIHNyYyArPSBzcmNTdHJpZGU7XG4gICAgICBkc3QgKz0gZHN0U3RyaWRlO1xuICAgIH1cbiAgfVxuXG4gIC8vIGxvb3AgYWNyb3NzIGZpbmFsIHJvd1xuICBpZiAob2RkSGVpZ2h0KSB7XG4gICAgbGV0IHNyYyA9ICh0ZXh0dXJlSGVpZ2h0IC0gMSkgKiB0ZXh0dXJlV2lkdGggKiA0O1xuICAgIGxldCBkc3QgPSAocm93cyAtIDEpICogY29sdW1ucztcbiAgICBmb3IgKGxldCBibG9ja1ggPSAwOyBibG9ja1ggPCB3aWR0aEluRnVsbEJsb2NrczsgKytibG9ja1gpIHtcbiAgICAgIG1hdHJpeFtkc3QrK10gPSBwYWNrZWRSR0JBW3NyYysrXTtcbiAgICAgIG1hdHJpeFtkc3QrK10gPSBwYWNrZWRSR0JBW3NyYysrXTtcbiAgICAgIHNyYyArPSAyO1xuICAgIH1cbiAgfVxuXG4gIC8vIGZpbGwgaW4gYm90dG9tLXJpZ2h0IGNlbGxcbiAgaWYgKG9kZFdpZHRoICYmIG9kZEhlaWdodCkge1xuICAgIG1hdHJpeFttYXRyaXgubGVuZ3RoIC0gMV0gPSBwYWNrZWRSR0JBW3BhY2tlZFJHQkEubGVuZ3RoIC0gNF07XG4gIH1cblxuICByZXR1cm4gbWF0cml4O1xufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5sZXQgVVNFX1dFQkdMMl9XSEVOX0FWQUlMQUJMRSA9IGZhbHNlO1xubGV0IFdFQkdMMl9FTkFCTEVEOiBib29sZWFufHVuZGVmaW5lZCA9IG51bGwhO1xubGV0IE1BWF9URVhUVVJFX1NJWkU6IG51bWJlciA9IG51bGwhO1xuXG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uLy4uL3V0aWwnO1xuXG5leHBvcnQgaW50ZXJmYWNlIFdlYkdMQ29udGV4dEF0dHJpYnV0ZXMge1xuICBhbHBoYT86IGJvb2xlYW47XG4gIGFudGlhbGlhcz86IGJvb2xlYW47XG4gIHByZW11bHRpcGxpZWRBbHBoYT86IGJvb2xlYW47XG4gIHByZXNlcnZlRHJhd2luZ0J1ZmZlcj86IGJvb2xlYW47XG4gIGRlcHRoPzogYm9vbGVhbjtcbiAgc3RlbmNpbD86IGJvb2xlYW47XG4gIGZhaWxJZk1ham9yUGVyZm9ybWFuY2VDYXZlYXQ/OiBib29sZWFuO1xufVxuXG4vKiogQGhpZGRlbiAqL1xuZXhwb3J0IGNvbnN0IElTX05BTl9TSEFERVJfRlVOQyA9IGBcbmJvb2wgaXNOYU4oZmxvYXQgdmFsKSB7XG4gIHJldHVybiB2YWwgPT0gdmFsID8gZmFsc2UgOiB0cnVlO1xufVxuYDtcblxuZXhwb3J0IGludGVyZmFjZSBXZWJHTExvc2VDb250ZXh0RXh0ZW5zaW9uIHsgbG9zZUNvbnRleHQoKTogdm9pZDsgfVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlV2ViR0xSZW5kZXJpbmdDb250ZXh0KGF0dHJpYnV0ZXM6IFdlYkdMQ29udGV4dEF0dHJpYnV0ZXMpOlxuICAgIFdlYkdMUmVuZGVyaW5nQ29udGV4dCB7XG4gIGNvbnN0IGNhbnZhcyA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2NhbnZhcycpO1xuICBjYW52YXMud2lkdGggPSAxO1xuICBjYW52YXMuaGVpZ2h0ID0gMTtcbiAgcmV0dXJuIGNyZWF0ZVdlYkdMUmVuZGVyaW5nQ29udGV4dEZyb21DYW52YXMoY2FudmFzLCBhdHRyaWJ1dGVzKTtcbn1cblxuLyoqXG4gKiBGb3JjZSB0aGUgbGlicmFyeSB0byBwcmVmZXIgV2ViR0wgMS4wIGluc3RlYWQgb2YgV2ViR0wgMi4wIGV2ZW4gd2hlbiBXZWJHTFxuICogMi4wIGlzIGF2YWlsYWJsZS5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHByZWZlcldlYkdMMSgpIHtcbiAgVVNFX1dFQkdMMl9XSEVOX0FWQUlMQUJMRSA9IGZhbHNlO1xuICBXRUJHTDJfRU5BQkxFRCA9IHVuZGVmaW5lZDtcbn1cblxuLyoqXG4gKiBQcmVmZXIgV2ViR0wgMi4wIHRvIFdlYkdMIDEuMC4gVGhpcyBpcyB0aGUgZGVmYXVsdCBjb25maWd1cmF0aW9uLlxuICovXG5leHBvcnQgZnVuY3Rpb24gcHJlZmVyV2ViR0wyKCkge1xuICBVU0VfV0VCR0wyX1dIRU5fQVZBSUxBQkxFID0gdHJ1ZTtcbiAgV0VCR0wyX0VOQUJMRUQgPSB1bmRlZmluZWQ7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBpc1dlYkdMMkVuYWJsZWQoKSB7XG4gIGlmICghVVNFX1dFQkdMMl9XSEVOX0FWQUlMQUJMRSkge1xuICAgIHJldHVybiBmYWxzZTtcbiAgfVxuXG4gIGlmIChXRUJHTDJfRU5BQkxFRCA9PT0gdW5kZWZpbmVkKSB7XG4gICAgY29uc3QgdGVtcENhbnZhcyA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2NhbnZhcycpO1xuICAgIGNvbnN0IGdsID0gdGVtcENhbnZhcy5nZXRDb250ZXh0KCd3ZWJnbDInKTtcbiAgICBpZiAoZ2wgIT0gbnVsbCkge1xuICAgICAgV0VCR0wyX0VOQUJMRUQgPSB0cnVlO1xuXG4gICAgICBjb25zdCBsb3NlQ29udGV4dEV4dGVuc2lvbiA9XG4gICAgICAgICAgZ2V0RXh0ZW5zaW9uT3JUaHJvdyhcbiAgICAgICAgICAgICAgZ2wgYXMgV2ViR0xSZW5kZXJpbmdDb250ZXh0LCAnV0VCR0xfbG9zZV9jb250ZXh0JykgYXNcbiAgICAgICAgICBXZWJHTExvc2VDb250ZXh0RXh0ZW5zaW9uO1xuICAgICAgbG9zZUNvbnRleHRFeHRlbnNpb24ubG9zZUNvbnRleHQoKTtcbiAgICB9IGVsc2Uge1xuICAgICAgV0VCR0wyX0VOQUJMRUQgPSBmYWxzZTtcbiAgICB9XG4gIH1cbiAgcmV0dXJuIFdFQkdMMl9FTkFCTEVEO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlV2ViR0xSZW5kZXJpbmdDb250ZXh0RnJvbUNhbnZhcyhcbiAgICBjYW52YXM6IEhUTUxDYW52YXNFbGVtZW50LFxuICAgIGF0dHJpYnV0ZXM6IFdlYkdMQ29udGV4dEF0dHJpYnV0ZXMpOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQge1xuICBsZXQgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dDtcbiAgaWYgKGlzV2ViR0wyRW5hYmxlZCgpKSB7XG4gICAgZ2wgPSBjYW52YXMuZ2V0Q29udGV4dCgnd2ViZ2wyJywgYXR0cmlidXRlcykgYXMgV2ViR0xSZW5kZXJpbmdDb250ZXh0O1xuICB9IGVsc2Uge1xuICAgIGdsID0gKGNhbnZhcy5nZXRDb250ZXh0KCd3ZWJnbCcsIGF0dHJpYnV0ZXMpIHx8XG4gICAgICAgICAgY2FudmFzLmdldENvbnRleHQoJ2V4cGVyaW1lbnRhbC13ZWJnbCcsIGF0dHJpYnV0ZXMpKSBhc1xuICAgICAgICBXZWJHTFJlbmRlcmluZ0NvbnRleHQ7XG4gIH1cblxuICBpZiAoZ2wgPT0gbnVsbCkge1xuICAgIHRocm93IG5ldyBFcnJvcignVGhpcyBicm93c2VyIGRvZXMgbm90IHN1cHBvcnQgV2ViR0wuJyk7XG4gIH1cbiAgcmV0dXJuIGdsO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY2FsbEFuZENoZWNrPFQ+KGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIGZ1bmM6ICgpID0+IFQpOiBUIHtcbiAgY29uc3QgcmV0dXJuVmFsdWUgPSBmdW5jKCk7XG4gIGNoZWNrV2ViR0xFcnJvcihnbCk7XG4gIHJldHVybiByZXR1cm5WYWx1ZTtcbn1cblxubGV0IHdlYkdMRGVidWdFcnJvckNoZWNraW5nRW5hYmxlZCA9IGZhbHNlO1xuXG5leHBvcnQgZnVuY3Rpb24gZW5hYmxlRGVidWdXZWJHTEVycm9yQ2hlY2tpbmcoZW5hYmxlZDogYm9vbGVhbikge1xuICB3ZWJHTERlYnVnRXJyb3JDaGVja2luZ0VuYWJsZWQgPSBlbmFibGVkO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY2hlY2tXZWJHTEVycm9yKGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQpIHtcbiAgaWYgKHdlYkdMRGVidWdFcnJvckNoZWNraW5nRW5hYmxlZCkge1xuICAgIGNvbnN0IGVycm9yID0gZ2wuZ2V0RXJyb3IoKTtcbiAgICBpZiAoZXJyb3IgIT09IGdsLk5PX0VSUk9SKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ1dlYkdMIEVycm9yOiAnICsgZ2V0V2ViR0xFcnJvck1lc3NhZ2UoZ2wsIGVycm9yKSk7XG4gICAgfVxuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRXZWJHTEVycm9yTWVzc2FnZShcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBzdGF0dXM6IG51bWJlcik6IHN0cmluZyB7XG4gIHN3aXRjaCAoc3RhdHVzKSB7XG4gICAgY2FzZSBnbC5OT19FUlJPUjpcbiAgICAgIHJldHVybiAnTk9fRVJST1InO1xuICAgIGNhc2UgZ2wuSU5WQUxJRF9FTlVNOlxuICAgICAgcmV0dXJuICdJTlZBTElEX0VOVU0nO1xuICAgIGNhc2UgZ2wuSU5WQUxJRF9WQUxVRTpcbiAgICAgIHJldHVybiAnSU5WQUxJRF9WQUxVRSc7XG4gICAgY2FzZSBnbC5JTlZBTElEX09QRVJBVElPTjpcbiAgICAgIHJldHVybiAnSU5WQUxJRF9PUEVSQVRJT04nO1xuICAgIGNhc2UgZ2wuSU5WQUxJRF9GUkFNRUJVRkZFUl9PUEVSQVRJT046XG4gICAgICByZXR1cm4gJ0lOVkFMSURfRlJBTUVCVUZGRVJfT1BFUkFUSU9OJztcbiAgICBjYXNlIGdsLk9VVF9PRl9NRU1PUlk6XG4gICAgICByZXR1cm4gJ09VVF9PRl9NRU1PUlknO1xuICAgIGNhc2UgZ2wuQ09OVEVYVF9MT1NUX1dFQkdMOlxuICAgICAgcmV0dXJuICdDT05URVhUX0xPU1RfV0VCR0wnO1xuICAgIGRlZmF1bHQ6XG4gICAgICByZXR1cm4gJ1Vua25vd24gZXJyb3IgY29kZSAnICsgc3RhdHVzO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRFeHRlbnNpb25PclRocm93KFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIGV4dGVuc2lvbk5hbWU6IHN0cmluZyk6IHt9IHtcbiAgcmV0dXJuIHRocm93SWZOdWxsPHt9PihcbiAgICAgIGdsLCAoKSA9PiBnbC5nZXRFeHRlbnNpb24oZXh0ZW5zaW9uTmFtZSksXG4gICAgICAnRXh0ZW5zaW9uIFwiJyArIGV4dGVuc2lvbk5hbWUgKyAnXCIgbm90IHN1cHBvcnRlZCBvbiB0aGlzIGJyb3dzZXIuJyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVWZXJ0ZXhTaGFkZXIoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgdmVydGV4U2hhZGVyU291cmNlOiBzdHJpbmcpOiBXZWJHTFNoYWRlciB7XG4gIGNvbnN0IHZlcnRleFNoYWRlcjogV2ViR0xTaGFkZXIgPSB0aHJvd0lmTnVsbDxXZWJHTFNoYWRlcj4oXG4gICAgICBnbCwgKCkgPT4gZ2wuY3JlYXRlU2hhZGVyKGdsLlZFUlRFWF9TSEFERVIpLFxuICAgICAgJ1VuYWJsZSB0byBjcmVhdGUgdmVydGV4IFdlYkdMU2hhZGVyLicpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLnNoYWRlclNvdXJjZSh2ZXJ0ZXhTaGFkZXIsIHZlcnRleFNoYWRlclNvdXJjZSkpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmNvbXBpbGVTaGFkZXIodmVydGV4U2hhZGVyKSk7XG4gIGlmIChnbC5nZXRTaGFkZXJQYXJhbWV0ZXIodmVydGV4U2hhZGVyLCBnbC5DT01QSUxFX1NUQVRVUykgPT09IGZhbHNlKSB7XG4gICAgY29uc29sZS5sb2coZ2wuZ2V0U2hhZGVySW5mb0xvZyh2ZXJ0ZXhTaGFkZXIpKTtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ0ZhaWxlZCB0byBjb21waWxlIHZlcnRleCBzaGFkZXIuJyk7XG4gIH1cbiAgcmV0dXJuIHZlcnRleFNoYWRlcjtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZUZyYWdtZW50U2hhZGVyKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIGZyYWdtZW50U2hhZGVyU291cmNlOiBzdHJpbmcpOiBXZWJHTFNoYWRlciB7XG4gIGNvbnN0IGZyYWdtZW50U2hhZGVyOiBXZWJHTFNoYWRlciA9IHRocm93SWZOdWxsPFdlYkdMU2hhZGVyPihcbiAgICAgIGdsLCAoKSA9PiBnbC5jcmVhdGVTaGFkZXIoZ2wuRlJBR01FTlRfU0hBREVSKSxcbiAgICAgICdVbmFibGUgdG8gY3JlYXRlIGZyYWdtZW50IFdlYkdMU2hhZGVyLicpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLnNoYWRlclNvdXJjZShmcmFnbWVudFNoYWRlciwgZnJhZ21lbnRTaGFkZXJTb3VyY2UpKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5jb21waWxlU2hhZGVyKGZyYWdtZW50U2hhZGVyKSk7XG4gIGlmIChnbC5nZXRTaGFkZXJQYXJhbWV0ZXIoZnJhZ21lbnRTaGFkZXIsIGdsLkNPTVBJTEVfU1RBVFVTKSA9PT0gZmFsc2UpIHtcbiAgICBjb25zb2xlLmxvZyhnbC5nZXRTaGFkZXJJbmZvTG9nKGZyYWdtZW50U2hhZGVyKSk7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdGYWlsZWQgdG8gY29tcGlsZSBmcmFnbWVudCBzaGFkZXIuJyk7XG4gIH1cbiAgcmV0dXJuIGZyYWdtZW50U2hhZGVyO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlUHJvZ3JhbShnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0KTogV2ViR0xQcm9ncmFtIHtcbiAgcmV0dXJuIHRocm93SWZOdWxsPFdlYkdMUHJvZ3JhbT4oXG4gICAgICBnbCwgKCkgPT4gZ2wuY3JlYXRlUHJvZ3JhbSgpLCAnVW5hYmxlIHRvIGNyZWF0ZSBXZWJHTFByb2dyYW0uJyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBsaW5rUHJvZ3JhbShnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBwcm9ncmFtOiBXZWJHTFByb2dyYW0pIHtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5saW5rUHJvZ3JhbShwcm9ncmFtKSk7XG4gIGlmIChnbC5nZXRQcm9ncmFtUGFyYW1ldGVyKHByb2dyYW0sIGdsLkxJTktfU1RBVFVTKSA9PT0gZmFsc2UpIHtcbiAgICBjb25zb2xlLmxvZyhnbC5nZXRQcm9ncmFtSW5mb0xvZyhwcm9ncmFtKSk7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdGYWlsZWQgdG8gbGluayB2ZXJ0ZXggYW5kIGZyYWdtZW50IHNoYWRlcnMuJyk7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHZhbGlkYXRlUHJvZ3JhbShcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBwcm9ncmFtOiBXZWJHTFByb2dyYW0pIHtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC52YWxpZGF0ZVByb2dyYW0ocHJvZ3JhbSkpO1xuICBpZiAoZ2wuZ2V0UHJvZ3JhbVBhcmFtZXRlcihwcm9ncmFtLCBnbC5WQUxJREFURV9TVEFUVVMpID09PSBmYWxzZSkge1xuICAgIGNvbnNvbGUubG9nKGdsLmdldFByb2dyYW1JbmZvTG9nKHByb2dyYW0pKTtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ1NoYWRlciBwcm9ncmFtIHZhbGlkYXRpb24gZmFpbGVkLicpO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVTdGF0aWNWZXJ0ZXhCdWZmZXIoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgZGF0YTogRmxvYXQzMkFycmF5KTogV2ViR0xCdWZmZXIge1xuICBjb25zdCBidWZmZXI6IFdlYkdMQnVmZmVyID0gdGhyb3dJZk51bGw8V2ViR0xCdWZmZXI+KFxuICAgICAgZ2wsICgpID0+IGdsLmNyZWF0ZUJ1ZmZlcigpLCAnVW5hYmxlIHRvIGNyZWF0ZSBXZWJHTEJ1ZmZlcicpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRCdWZmZXIoZ2wuQVJSQVlfQlVGRkVSLCBidWZmZXIpKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5idWZmZXJEYXRhKGdsLkFSUkFZX0JVRkZFUiwgZGF0YSwgZ2wuU1RBVElDX0RSQVcpKTtcbiAgcmV0dXJuIGJ1ZmZlcjtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVN0YXRpY0luZGV4QnVmZmVyKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIGRhdGE6IFVpbnQxNkFycmF5KTogV2ViR0xCdWZmZXIge1xuICBjb25zdCBidWZmZXI6IFdlYkdMQnVmZmVyID0gdGhyb3dJZk51bGw8V2ViR0xCdWZmZXI+KFxuICAgICAgZ2wsICgpID0+IGdsLmNyZWF0ZUJ1ZmZlcigpLCAnVW5hYmxlIHRvIGNyZWF0ZSBXZWJHTEJ1ZmZlcicpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRCdWZmZXIoZ2wuRUxFTUVOVF9BUlJBWV9CVUZGRVIsIGJ1ZmZlcikpO1xuICBjYWxsQW5kQ2hlY2soXG4gICAgICBnbCwgKCkgPT4gZ2wuYnVmZmVyRGF0YShnbC5FTEVNRU5UX0FSUkFZX0JVRkZFUiwgZGF0YSwgZ2wuU1RBVElDX0RSQVcpKTtcbiAgcmV0dXJuIGJ1ZmZlcjtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHF1ZXJ5TWF4VGV4dHVyZVNpemUoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCk6IG51bWJlciB7XG4gIGlmIChNQVhfVEVYVFVSRV9TSVpFICE9IG51bGwpIHtcbiAgICByZXR1cm4gTUFYX1RFWFRVUkVfU0laRTtcbiAgfVxuICBNQVhfVEVYVFVSRV9TSVpFID1cbiAgICAgIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2whLmdldFBhcmFtZXRlcihnbCEuTUFYX1RFWFRVUkVfU0laRSkpO1xuICByZXR1cm4gTUFYX1RFWFRVUkVfU0laRTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldENoYW5uZWxzUGVyVGV4dHVyZSgpOiBudW1iZXIge1xuICBpZiAoaXNXZWJHTDJFbmFibGVkKCkpIHtcbiAgICByZXR1cm4gMTtcbiAgfVxuICByZXR1cm4gNDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVRleHR1cmUoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCk6IFdlYkdMVGV4dHVyZSB7XG4gIHJldHVybiB0aHJvd0lmTnVsbDxXZWJHTFRleHR1cmU+KFxuICAgICAgZ2wsICgpID0+IGdsLmNyZWF0ZVRleHR1cmUoKSwgJ1VuYWJsZSB0byBjcmVhdGUgV2ViR0xUZXh0dXJlLicpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gdmFsaWRhdGVUZXh0dXJlU2l6ZShcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCB3aWR0aDogbnVtYmVyLCBoZWlnaHQ6IG51bWJlcikge1xuICBjb25zdCBtYXhUZXh0dXJlU2l6ZTogbnVtYmVyID0gcXVlcnlNYXhUZXh0dXJlU2l6ZShnbCk7XG4gIGlmICgod2lkdGggPD0gMCkgfHwgKGhlaWdodCA8PSAwKSkge1xuICAgIGNvbnN0IHJlcXVlc3RlZCA9ICdbJyArIHdpZHRoICsgJ3gnICsgaGVpZ2h0ICsgJ10nO1xuICAgIHRocm93IG5ldyBFcnJvcignUmVxdWVzdGVkIHRleHR1cmUgc2l6ZSAnICsgcmVxdWVzdGVkICsgJyBpcyBpbnZhbGlkLicpO1xuICB9XG4gIGlmICgod2lkdGggPiBtYXhUZXh0dXJlU2l6ZSkgfHwgKGhlaWdodCA+IG1heFRleHR1cmVTaXplKSkge1xuICAgIGNvbnN0IHJlcXVlc3RlZCA9ICdbJyArIHdpZHRoICsgJ3gnICsgaGVpZ2h0ICsgJ10nO1xuICAgIGNvbnN0IG1heCA9ICdbJyArIG1heFRleHR1cmVTaXplICsgJ3gnICsgbWF4VGV4dHVyZVNpemUgKyAnXSc7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAnUmVxdWVzdGVkIHRleHR1cmUgc2l6ZSAnICsgcmVxdWVzdGVkICtcbiAgICAgICAgJyBncmVhdGVyIHRoYW4gV2ViR0wgbWF4aW11bSBvbiB0aGlzIGJyb3dzZXIgLyBHUFUgJyArIG1heCArICcuJyk7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZUZyYW1lYnVmZmVyKGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQpOiBXZWJHTEZyYW1lYnVmZmVyIHtcbiAgcmV0dXJuIHRocm93SWZOdWxsPFdlYkdMRnJhbWVidWZmZXI+KFxuICAgICAgZ2wsICgpID0+IGdsLmNyZWF0ZUZyYW1lYnVmZmVyKCksICdVbmFibGUgdG8gY3JlYXRlIFdlYkdMRnJhbWVidWZmZXIuJyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBiaW5kVmVydGV4QnVmZmVyVG9Qcm9ncmFtQXR0cmlidXRlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSwgYXR0cmlidXRlOiBzdHJpbmcsXG4gICAgYnVmZmVyOiBXZWJHTEJ1ZmZlciwgYXJyYXlFbnRyaWVzUGVySXRlbTogbnVtYmVyLCBpdGVtU3RyaWRlSW5CeXRlczogbnVtYmVyLFxuICAgIGl0ZW1PZmZzZXRJbkJ5dGVzOiBudW1iZXIpIHtcbiAgY29uc3QgbG9jID0gZ2wuZ2V0QXR0cmliTG9jYXRpb24ocHJvZ3JhbSwgYXR0cmlidXRlKTtcbiAgaWYgKGxvYyA9PT0gLTEpIHtcbiAgICBjb25zdCBlcnJvciA9IG5ldyBFcnJvcihcbiAgICAgICAgJ1VuYWJsZSB0byBnZXQgYXR0cmlidXRlIFwiJyArIGF0dHJpYnV0ZSArICdcIiBvbiBXZWJHTFByb2dyYW0uJyk7XG4gICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgIChlcnJvciBhcyBhbnkpLm5hbWVkVmVydGV4QXR0cmlidXRlTm90Rm91bmQgPSBhdHRyaWJ1dGU7XG4gICAgdGhyb3cgZXJyb3I7XG4gIH1cbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kQnVmZmVyKGdsLkFSUkFZX0JVRkZFUiwgYnVmZmVyKSk7XG4gIGNhbGxBbmRDaGVjayhcbiAgICAgIGdsLFxuICAgICAgKCkgPT4gZ2wudmVydGV4QXR0cmliUG9pbnRlcihcbiAgICAgICAgICBsb2MsIGFycmF5RW50cmllc1Blckl0ZW0sIGdsLkZMT0FULCBmYWxzZSwgaXRlbVN0cmlkZUluQnl0ZXMsXG4gICAgICAgICAgaXRlbU9mZnNldEluQnl0ZXMpKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5lbmFibGVWZXJ0ZXhBdHRyaWJBcnJheShsb2MpKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGJpbmRUZXh0dXJlVW5pdChcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCB0ZXh0dXJlOiBXZWJHTFRleHR1cmUsIHRleHR1cmVVbml0OiBudW1iZXIpIHtcbiAgdmFsaWRhdGVUZXh0dXJlVW5pdChnbCwgdGV4dHVyZVVuaXQpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmFjdGl2ZVRleHR1cmUoZ2wuVEVYVFVSRTAgKyB0ZXh0dXJlVW5pdCkpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRUZXh0dXJlKGdsLlRFWFRVUkVfMkQsIHRleHR1cmUpKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHVuYmluZFRleHR1cmVVbml0KFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHRleHR1cmVVbml0OiBudW1iZXIpIHtcbiAgdmFsaWRhdGVUZXh0dXJlVW5pdChnbCwgdGV4dHVyZVVuaXQpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmFjdGl2ZVRleHR1cmUoZ2wuVEVYVFVSRTAgKyB0ZXh0dXJlVW5pdCkpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRUZXh0dXJlKGdsLlRFWFRVUkVfMkQsIG51bGwpKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldFByb2dyYW1Vbmlmb3JtTG9jYXRpb25PclRocm93KFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSxcbiAgICB1bmlmb3JtTmFtZTogc3RyaW5nKTogV2ViR0xVbmlmb3JtTG9jYXRpb24ge1xuICByZXR1cm4gdGhyb3dJZk51bGw8V2ViR0xVbmlmb3JtTG9jYXRpb24+KFxuICAgICAgZ2wsICgpID0+IGdsLmdldFVuaWZvcm1Mb2NhdGlvbihwcm9ncmFtLCB1bmlmb3JtTmFtZSksXG4gICAgICAndW5pZm9ybSBcIicgKyB1bmlmb3JtTmFtZSArICdcIiBub3QgcHJlc2VudCBpbiBwcm9ncmFtLicpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYmluZFRleHR1cmVUb1Byb2dyYW1Vbmlmb3JtU2FtcGxlcihcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBwcm9ncmFtOiBXZWJHTFByb2dyYW0sIHRleHR1cmU6IFdlYkdMVGV4dHVyZSxcbiAgICB1bmlmb3JtU2FtcGxlck5hbWU6IHN0cmluZywgdGV4dHVyZVVuaXQ6IG51bWJlcikge1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGJpbmRUZXh0dXJlVW5pdChnbCwgdGV4dHVyZSwgdGV4dHVyZVVuaXQpKTtcbiAgY29uc3Qgc2FtcGxlckxvY2F0aW9uID1cbiAgICAgIGdldFByb2dyYW1Vbmlmb3JtTG9jYXRpb25PclRocm93KGdsLCBwcm9ncmFtLCB1bmlmb3JtU2FtcGxlck5hbWUpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLnVuaWZvcm0xaShzYW1wbGVyTG9jYXRpb24sIHRleHR1cmVVbml0KSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBiaW5kQ2FudmFzVG9GcmFtZWJ1ZmZlcihnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0KSB7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZEZyYW1lYnVmZmVyKGdsLkZSQU1FQlVGRkVSLCBudWxsKSk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wudmlld3BvcnQoMCwgMCwgZ2wuY2FudmFzLndpZHRoLCBnbC5jYW52YXMuaGVpZ2h0KSk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuc2Npc3NvcigwLCAwLCBnbC5jYW52YXMud2lkdGgsIGdsLmNhbnZhcy5oZWlnaHQpKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGJpbmRDb2xvclRleHR1cmVUb0ZyYW1lYnVmZmVyKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHRleHR1cmU6IFdlYkdMVGV4dHVyZSxcbiAgICBmcmFtZWJ1ZmZlcjogV2ViR0xGcmFtZWJ1ZmZlcikge1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRGcmFtZWJ1ZmZlcihnbC5GUkFNRUJVRkZFUiwgZnJhbWVidWZmZXIpKTtcbiAgY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsXG4gICAgICAoKSA9PiBnbC5mcmFtZWJ1ZmZlclRleHR1cmUyRChcbiAgICAgICAgICBnbC5GUkFNRUJVRkZFUiwgZ2wuQ09MT1JfQVRUQUNITUVOVDAsIGdsLlRFWFRVUkVfMkQsIHRleHR1cmUsIDApKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHVuYmluZENvbG9yVGV4dHVyZUZyb21GcmFtZWJ1ZmZlcihcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBmcmFtZWJ1ZmZlcjogV2ViR0xGcmFtZWJ1ZmZlcikge1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRGcmFtZWJ1ZmZlcihnbC5GUkFNRUJVRkZFUiwgZnJhbWVidWZmZXIpKTtcbiAgY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsXG4gICAgICAoKSA9PiBnbC5mcmFtZWJ1ZmZlclRleHR1cmUyRChcbiAgICAgICAgICBnbC5GUkFNRUJVRkZFUiwgZ2wuQ09MT1JfQVRUQUNITUVOVDAsIGdsLlRFWFRVUkVfMkQsIG51bGwsIDApKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHZhbGlkYXRlRnJhbWVidWZmZXIoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCkge1xuICBjb25zdCBzdGF0dXMgPSBnbC5jaGVja0ZyYW1lYnVmZmVyU3RhdHVzKGdsLkZSQU1FQlVGRkVSKTtcbiAgaWYgKHN0YXR1cyAhPT0gZ2wuRlJBTUVCVUZGRVJfQ09NUExFVEUpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICdFcnJvciBiaW5kaW5nIGZyYW1lYnVmZmVyOiAnICsgZ2V0RnJhbWVidWZmZXJFcnJvck1lc3NhZ2UoZ2wsIHN0YXR1cykpO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRGcmFtZWJ1ZmZlckVycm9yTWVzc2FnZShcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBzdGF0dXM6IG51bWJlcik6IHN0cmluZyB7XG4gIHN3aXRjaCAoc3RhdHVzKSB7XG4gICAgY2FzZSBnbC5GUkFNRUJVRkZFUl9JTkNPTVBMRVRFX0FUVEFDSE1FTlQ6XG4gICAgICByZXR1cm4gJ0ZSQU1FQlVGRkVSX0lOQ09NUExFVEVfQVRUQUNITUVOVCc7XG4gICAgY2FzZSBnbC5GUkFNRUJVRkZFUl9JTkNPTVBMRVRFX01JU1NJTkdfQVRUQUNITUVOVDpcbiAgICAgIHJldHVybiAnRlJBTUVCVUZGRVJfSU5DT01QTEVURV9NSVNTSU5HX0FUVEFDSE1FTlQnO1xuICAgIGNhc2UgZ2wuRlJBTUVCVUZGRVJfSU5DT01QTEVURV9ESU1FTlNJT05TOlxuICAgICAgcmV0dXJuICdGUkFNRUJVRkZFUl9JTkNPTVBMRVRFX0RJTUVOU0lPTlMnO1xuICAgIGNhc2UgZ2wuRlJBTUVCVUZGRVJfVU5TVVBQT1JURUQ6XG4gICAgICByZXR1cm4gJ0ZSQU1FQlVGRkVSX1VOU1VQUE9SVEVEJztcbiAgICBkZWZhdWx0OlxuICAgICAgcmV0dXJuICd1bmtub3duIGVycm9yICcgKyBzdGF0dXM7XG4gIH1cbn1cblxuZnVuY3Rpb24gdGhyb3dJZk51bGw8VD4oXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgcmV0dXJuVE9yTnVsbDogKCkgPT4gVCB8IG51bGwsXG4gICAgZmFpbHVyZU1lc3NhZ2U6IHN0cmluZyk6IFQge1xuICBjb25zdCB0T3JOdWxsOiBUfG51bGwgPSBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IHJldHVyblRPck51bGwoKSk7XG4gIGlmICh0T3JOdWxsID09IG51bGwpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoZmFpbHVyZU1lc3NhZ2UpO1xuICB9XG4gIHJldHVybiB0T3JOdWxsIGFzIFQ7XG59XG5cbmZ1bmN0aW9uIHZhbGlkYXRlVGV4dHVyZVVuaXQoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgdGV4dHVyZVVuaXQ6IG51bWJlcikge1xuICBjb25zdCBtYXhUZXh0dXJlVW5pdCA9IGdsLk1BWF9DT01CSU5FRF9URVhUVVJFX0lNQUdFX1VOSVRTIC0gMTtcbiAgY29uc3QgZ2xUZXh0dXJlVW5pdCA9IHRleHR1cmVVbml0ICsgZ2wuVEVYVFVSRTA7XG4gIGlmIChnbFRleHR1cmVVbml0IDwgZ2wuVEVYVFVSRTAgfHwgZ2xUZXh0dXJlVW5pdCA+IG1heFRleHR1cmVVbml0KSB7XG4gICAgY29uc3QgdGV4dHVyZVVuaXRSYW5nZSA9ICdbZ2wuVEVYVFVSRTAsIGdsLlRFWFRVUkUnICsgbWF4VGV4dHVyZVVuaXQgKyAnXSc7XG4gICAgdGhyb3cgbmV3IEVycm9yKCd0ZXh0dXJlVW5pdCBtdXN0IGJlIGluICcgKyB0ZXh0dXJlVW5pdFJhbmdlICsgJy4nKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0VGV4dHVyZVNoYXBlRnJvbUxvZ2ljYWxTaGFwZShcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBsb2dpY2FsU2hhcGU6IG51bWJlcltdLFxuICAgIHByZWZlcnJlZFRleFNoYXBlPzogW251bWJlciwgbnVtYmVyXSk6IFtudW1iZXIsIG51bWJlcl0ge1xuICBjb25zdCBtYXhUZXhTaXplID0gcXVlcnlNYXhUZXh0dXJlU2l6ZShnbCk7XG4gIGNvbnN0IHNpemUgPSB1dGlsLnNpemVGcm9tU2hhcGUobG9naWNhbFNoYXBlKTtcbiAgaWYgKHByZWZlcnJlZFRleFNoYXBlICE9IG51bGwpIHtcbiAgICBjb25zdCBzaXplUHJlZmVycmVkID0gdXRpbC5zaXplRnJvbVNoYXBlKHByZWZlcnJlZFRleFNoYXBlKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgc2l6ZSA9PT0gc2l6ZVByZWZlcnJlZCxcbiAgICAgICAgYFNpemUgb2Ygc2hhcGUgKCR7c2l6ZX0pIG11c3QgbWF0Y2ggc2l6ZSBvZiBgICtcbiAgICAgICAgICAgIGBwcmVmZXJyZWRTaGFwZSAoJHtzaXplUHJlZmVycmVkfSlgKTtcbiAgICBpZiAocHJlZmVycmVkVGV4U2hhcGVbMF0gPD0gbWF4VGV4U2l6ZSAmJlxuICAgICAgICBwcmVmZXJyZWRUZXhTaGFwZVsxXSA8PSBtYXhUZXhTaXplKSB7XG4gICAgICByZXR1cm4gcHJlZmVycmVkVGV4U2hhcGU7XG4gICAgfVxuICB9XG5cbiAgaWYgKGxvZ2ljYWxTaGFwZS5sZW5ndGggPD0gMSAmJiBzaXplIDw9IG1heFRleFNpemUpIHtcbiAgICByZXR1cm4gW3NpemUsIDFdO1xuICB9IGVsc2UgaWYgKFxuICAgICAgbG9naWNhbFNoYXBlLmxlbmd0aCA9PT0gMiAmJiBsb2dpY2FsU2hhcGVbMF0gPD0gbWF4VGV4U2l6ZSAmJlxuICAgICAgbG9naWNhbFNoYXBlWzFdIDw9IG1heFRleFNpemUpIHtcbiAgICByZXR1cm4gbG9naWNhbFNoYXBlIGFzIFtudW1iZXIsIG51bWJlcl07XG4gIH0gZWxzZSBpZiAoXG4gICAgICBsb2dpY2FsU2hhcGUubGVuZ3RoID09PSAzICYmIGxvZ2ljYWxTaGFwZVswXSA8PSBtYXhUZXhTaXplICYmXG4gICAgICBsb2dpY2FsU2hhcGVbMV0gKiBsb2dpY2FsU2hhcGVbMl0gPD0gbWF4VGV4U2l6ZSkge1xuICAgIHJldHVybiBbbG9naWNhbFNoYXBlWzBdLCBsb2dpY2FsU2hhcGVbMV0gKiBsb2dpY2FsU2hhcGVbMl1dO1xuICB9IGVsc2Uge1xuICAgIHJldHVybiB1dGlsLnNpemVUb1NxdWFyaXNoU2hhcGUoc2l6ZSk7XG4gIH1cbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuZXhwb3J0IGZ1bmN0aW9uIGV4cGVjdEFycmF5c0Nsb3NlKFxuICAgIGFjdHVhbDogRmxvYXQzMkFycmF5LCBleHBlY3RlZDogRmxvYXQzMkFycmF5LCBlcHNpbG9uOiBudW1iZXIpIHtcbiAgaWYgKGFjdHVhbC5sZW5ndGggIT09IGV4cGVjdGVkLmxlbmd0aCkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgJ01hdHJpY2VzIGhhdmUgZGlmZmVyZW50IGxlbmd0aHMgKCcgKyBhY3R1YWwubGVuZ3RoICsgJyB2cyAnICtcbiAgICAgICAgZXhwZWN0ZWQubGVuZ3RoICsgJykuJyk7XG4gIH1cbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBleHBlY3RlZC5sZW5ndGg7ICsraSkge1xuICAgIGNvbnN0IGEgPSBhY3R1YWxbaV07XG4gICAgY29uc3QgZSA9IGV4cGVjdGVkW2ldO1xuICAgIGlmIChpc05hTihhKSAmJiBpc05hTihlKSkge1xuICAgICAgY29udGludWU7XG4gICAgfVxuICAgIGlmIChpc05hTihhKSB8fCBpc05hTihlKSB8fCBNYXRoLmFicyhhIC0gZSkgPiBlcHNpbG9uKSB7XG4gICAgICBjb25zdCBhY3R1YWxTdHIgPSAnYWN0dWFsWycgKyBpICsgJ10gPT09ICcgKyBhO1xuICAgICAgY29uc3QgZXhwZWN0ZWRTdHIgPSAnZXhwZWN0ZWRbJyArIGkgKyAnXSA9PT0gJyArIGU7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ0FycmF5cyBkaWZmZXI6ICcgKyBhY3R1YWxTdHIgKyAnLCAnICsgZXhwZWN0ZWRTdHIpO1xuICAgIH1cbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gcmFuZG9tQXJyYXlJblJhbmdlKFxuICAgIG46IG51bWJlciwgbWluVmFsdWU6IG51bWJlciwgbWF4VmFsdWU6IG51bWJlcik6IEZsb2F0MzJBcnJheSB7XG4gIGNvbnN0IHYgPSBuZXcgRmxvYXQzMkFycmF5KG4pO1xuICBjb25zdCByYW5nZSA9IG1heFZhbHVlIC0gbWluVmFsdWU7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgbjsgKytpKSB7XG4gICAgdltpXSA9IChNYXRoLnJhbmRvbSgpICogcmFuZ2UpICsgbWluVmFsdWU7XG4gIH1cbiAgcmV0dXJuIHY7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBtYWtlSWRlbnRpdHkobjogbnVtYmVyKTogRmxvYXQzMkFycmF5IHtcbiAgY29uc3QgaSA9IG5ldyBGbG9hdDMyQXJyYXkobiAqIG4pO1xuICBmb3IgKGxldCBqID0gMDsgaiA8IG47ICsraikge1xuICAgIGlbKGogKiBuKSArIGpdID0gMTtcbiAgfVxuICByZXR1cm4gaTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHNldFZhbHVlKFxuICAgIG06IEZsb2F0MzJBcnJheSwgbU51bVJvd3M6IG51bWJlciwgbU51bUNvbHM6IG51bWJlciwgdjogbnVtYmVyLCByb3c6IG51bWJlcixcbiAgICBjb2x1bW46IG51bWJlcikge1xuICBpZiAocm93ID49IG1OdW1Sb3dzKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdyb3cgKCcgKyByb3cgKyAnKSBtdXN0IGJlIGluIFswICcgKyBtTnVtUm93cyArICddLicpO1xuICB9XG4gIGlmIChjb2x1bW4gPj0gbU51bUNvbHMpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ2NvbHVtbiAoJyArIGNvbHVtbiArICcpIG11c3QgYmUgaW4gWzAgJyArIG1OdW1Db2xzICsgJ10uJyk7XG4gIH1cbiAgbVsocm93ICogbU51bUNvbHMpICsgY29sdW1uXSA9IHY7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcHVNdWx0aXBseU1hdHJpeChcbiAgICBhOiBGbG9hdDMyQXJyYXksIGFSb3c6IG51bWJlciwgYUNvbDogbnVtYmVyLCBiOiBGbG9hdDMyQXJyYXksIGJSb3c6IG51bWJlcixcbiAgICBiQ29sOiBudW1iZXIpIHtcbiAgY29uc3QgcmVzdWx0ID0gbmV3IEZsb2F0MzJBcnJheShhUm93ICogYkNvbCk7XG4gIGZvciAobGV0IHIgPSAwOyByIDwgYVJvdzsgKytyKSB7XG4gICAgZm9yIChsZXQgYyA9IDA7IGMgPCBiQ29sOyArK2MpIHtcbiAgICAgIGxldCBkID0gMDtcbiAgICAgIGZvciAobGV0IGsgPSAwOyBrIDwgYUNvbDsgKytrKSB7XG4gICAgICAgIGQgKz0gYVsociAqIGFDb2wpICsga10gKiBiWyhrICogYkNvbCkgKyBjXTtcbiAgICAgIH1cbiAgICAgIHJlc3VsdFsociAqIGJDb2wpICsgY10gPSBkO1xuICAgIH1cbiAgfVxuICByZXR1cm4gcmVzdWx0O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3B1RG90UHJvZHVjdChhOiBGbG9hdDMyQXJyYXksIGI6IEZsb2F0MzJBcnJheSk6IG51bWJlciB7XG4gIGlmIChhLmxlbmd0aCAhPT0gYi5sZW5ndGgpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ2NwdURvdFByb2R1Y3Q6IGluY29tcGF0aWJsZSB2ZWN0b3JzLicpO1xuICB9XG4gIGxldCBkID0gMDtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBhLmxlbmd0aDsgKytpKSB7XG4gICAgZCArPSBhW2ldICogYltpXTtcbiAgfVxuICByZXR1cm4gZDtcbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuZXhwb3J0IHR5cGUgVmVjdG9yID0gbnVtYmVyW10gfCBGbG9hdDY0QXJyYXkgfCBGbG9hdDMyQXJyYXkgfCBJbnQzMkFycmF5IHxcbiAgICBJbnQ4QXJyYXkgfCBJbnQxNkFycmF5O1xuXG4vKiogU2h1ZmZsZXMgdGhlIGFycmF5IHVzaW5nIEZpc2hlci1ZYXRlcyBhbGdvcml0aG0uICovXG4vLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG5leHBvcnQgZnVuY3Rpb24gc2h1ZmZsZShhcnJheTogYW55W118VWludDMyQXJyYXl8SW50MzJBcnJheXxcbiAgICAgICAgICAgICAgICAgICAgICAgIEZsb2F0MzJBcnJheSk6IHZvaWQge1xuICBsZXQgY291bnRlciA9IGFycmF5Lmxlbmd0aDtcbiAgbGV0IHRlbXAgPSAwO1xuICBsZXQgaW5kZXggPSAwO1xuICAvLyBXaGlsZSB0aGVyZSBhcmUgZWxlbWVudHMgaW4gdGhlIGFycmF5XG4gIHdoaWxlIChjb3VudGVyID4gMCkge1xuICAgIC8vIFBpY2sgYSByYW5kb20gaW5kZXhcbiAgICBpbmRleCA9IChNYXRoLnJhbmRvbSgpICogY291bnRlcikgfCAwO1xuICAgIC8vIERlY3JlYXNlIGNvdW50ZXIgYnkgMVxuICAgIGNvdW50ZXItLTtcbiAgICAvLyBBbmQgc3dhcCB0aGUgbGFzdCBlbGVtZW50IHdpdGggaXRcbiAgICB0ZW1wID0gYXJyYXlbY291bnRlcl07XG4gICAgYXJyYXlbY291bnRlcl0gPSBhcnJheVtpbmRleF07XG4gICAgYXJyYXlbaW5kZXhdID0gdGVtcDtcbiAgfVxufVxuXG4vKiogQ2xhbXBzIGEgdmFsdWUgdG8gYSBzcGVjaWZpZWQgcmFuZ2UuICovXG5leHBvcnQgZnVuY3Rpb24gY2xhbXAobWluOiBudW1iZXIsIHg6IG51bWJlciwgbWF4OiBudW1iZXIpOiBudW1iZXIge1xuICByZXR1cm4gTWF0aC5tYXgobWluLCBNYXRoLm1pbih4LCBtYXgpKTtcbn1cblxuLyoqIFJldHVybnMgYSBzYW1wbGUgZnJvbSBhIHVuaWZvcm0gW2EsIGJdIGRpc3RyaWJ1dGlvbi4gKi9cbmV4cG9ydCBmdW5jdGlvbiByYW5kVW5pZm9ybShhOiBudW1iZXIsIGI6IG51bWJlcikge1xuICByZXR1cm4gTWF0aC5yYW5kb20oKSAqIChiIC0gYSkgKyBhO1xufVxuXG4vKipcbiAqIFNhbXBsZXMgZnJvbSBhIGdhdXNzaWFuIGRpc3RyaWJ1dGlvbi5cbiAqXG4gKiBAcGFyYW0gbWVhbiBUaGUgbWVhbi4gRGVmYXVsdCBpcyAwLlxuICogQHBhcmFtIHN0ZERldiBUaGUgc3RhbmRhcmQgZGV2aWF0aW9uLiBEZWZhdWx0IGlzIDEuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiByYW5kR2F1c3MobWVhbiA9IDAsIHN0ZERldiA9IDEsIHRydW5jYXRlZCA9IGZhbHNlKTogbnVtYmVyIHtcbiAgbGV0IHYxOiBudW1iZXIsIHYyOiBudW1iZXIsIHM6IG51bWJlcjtcbiAgZG8ge1xuICAgIHYxID0gMiAqIE1hdGgucmFuZG9tKCkgLSAxO1xuICAgIHYyID0gMiAqIE1hdGgucmFuZG9tKCkgLSAxO1xuICAgIHMgPSB2MSAqIHYxICsgdjIgKiB2MjtcbiAgfSB3aGlsZSAocyA+IDEpO1xuXG4gIGNvbnN0IHJlc3VsdCA9IE1hdGguc3FydCgtMiAqIE1hdGgubG9nKHMpIC8gcykgKiB2MTtcbiAgaWYgKHRydW5jYXRlZCAmJiByZXN1bHQgPiAyKSB7XG4gICAgcmV0dXJuIHJhbmRHYXVzcyhtZWFuLCBzdGREZXYsIHRydWUpO1xuICB9XG4gIHJldHVybiBtZWFuICsgc3RkRGV2ICogcmVzdWx0O1xufVxuXG4vKiogUmV0dXJucyBzcXVhcmVkIGV1Y2xlZGlhbiBkaXN0YW5jZSBiZXR3ZWVuIHR3byB2ZWN0b3JzLiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGRpc3RTcXVhcmVkKGE6IFZlY3RvciwgYjogVmVjdG9yKTogbnVtYmVyIHtcbiAgbGV0IHJlc3VsdCA9IDA7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgYS5sZW5ndGg7IGkrKykge1xuICAgIGNvbnN0IGRpZmYgPSBhW2ldIC0gYltpXTtcbiAgICByZXN1bHQgKz0gZGlmZiAqIGRpZmY7XG4gIH1cbiAgcmV0dXJuIHJlc3VsdDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGFzc2VydChleHByOiBib29sZWFuLCBtc2c6IHN0cmluZykge1xuICBpZiAoIWV4cHIpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IobXNnKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gYXNzZXJ0U2hhcGVzTWF0Y2goXG4gICAgc2hhcGVBOiBudW1iZXJbXSwgc2hhcGVCOiBudW1iZXJbXSwgZXJyb3JNZXNzYWdlUHJlZml4ID0gJycpOiB2b2lkIHtcbiAgYXNzZXJ0KFxuICAgICAgYXJyYXlzRXF1YWwoc2hhcGVBLCBzaGFwZUIpLFxuICAgICAgZXJyb3JNZXNzYWdlUHJlZml4ICsgYFNoYXBlcyAke3NoYXBlQX0gYW5kICR7c2hhcGVCfSBtdXN0IG1hdGNoYCk7XG59XG5cbi8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbmV4cG9ydCBmdW5jdGlvbiBmbGF0dGVuKGFycjogYW55W10sIHJldD86IG51bWJlcltdKTogbnVtYmVyW10ge1xuICByZXQgPSAocmV0ID09PSB1bmRlZmluZWQgPyBbXSA6IHJldCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgYXJyLmxlbmd0aDsgKytpKSB7XG4gICAgaWYgKEFycmF5LmlzQXJyYXkoYXJyW2ldKSkge1xuICAgICAgZmxhdHRlbihhcnJbaV0sIHJldCk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHJldC5wdXNoKGFycltpXSk7XG4gICAgfVxuICB9XG4gIHJldHVybiByZXQ7XG59XG5cbmV4cG9ydCB0eXBlIEFycmF5RGF0YSA9IG51bWJlcnxudW1iZXJbXXxudW1iZXJbXVtdfG51bWJlcltdW11bXXxudW1iZXJbXVtdW11bXTtcblxuZXhwb3J0IGZ1bmN0aW9uIGluZmVyU2hhcGUoYXJyOiBBcnJheURhdGEpOiBudW1iZXJbXSB7XG4gIGNvbnN0IHNoYXBlOiBudW1iZXJbXSA9IFtdO1xuICB3aGlsZSAoYXJyIGluc3RhbmNlb2YgQXJyYXkpIHtcbiAgICBzaGFwZS5wdXNoKGFyci5sZW5ndGgpO1xuICAgIGFyciA9IGFyclswXTtcbiAgfVxuICByZXR1cm4gc2hhcGU7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBzaXplRnJvbVNoYXBlKHNoYXBlOiBudW1iZXJbXSk6IG51bWJlciB7XG4gIGlmIChzaGFwZS5sZW5ndGggPT09IDApIHtcbiAgICAvLyBTY2FsYXIuXG4gICAgcmV0dXJuIDE7XG4gIH1cbiAgbGV0IHNpemUgPSBzaGFwZVswXTtcbiAgZm9yIChsZXQgaSA9IDE7IGkgPCBzaGFwZS5sZW5ndGg7IGkrKykge1xuICAgIHNpemUgKj0gc2hhcGVbaV07XG4gIH1cbiAgcmV0dXJuIHNpemU7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBpc1NjYWxhclNoYXBlKHNoYXBlOiBudW1iZXJbXSk6IGJvb2xlYW4ge1xuICByZXR1cm4gc2hhcGUubGVuZ3RoID09PSAwO1xufVxuXG4vLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG5leHBvcnQgZnVuY3Rpb24gYXJyYXlzRXF1YWwobjE6IGFueVtdfEZsb2F0MzJBcnJheSwgbjI6IGFueVtdfEZsb2F0MzJBcnJheSkge1xuICBpZiAobjEubGVuZ3RoICE9PSBuMi5sZW5ndGgpIHtcbiAgICByZXR1cm4gZmFsc2U7XG4gIH1cbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBuMS5sZW5ndGg7IGkrKykge1xuICAgIGlmIChuMVtpXSAhPT0gbjJbaV0pIHtcbiAgICAgIHJldHVybiBmYWxzZTtcbiAgICB9XG4gIH1cbiAgcmV0dXJuIHRydWU7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBpc0ludChhOiBudW1iZXIpOiBib29sZWFuIHtcbiAgcmV0dXJuIGEgJSAxID09PSAwO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gdGFuaCh4OiBudW1iZXIpOiBudW1iZXIge1xuICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gIGlmICgoTWF0aCBhcyBhbnkpLnRhbmggIT0gbnVsbCkge1xuICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICByZXR1cm4gKE1hdGggYXMgYW55KS50YW5oKHgpO1xuICB9XG4gIGlmICh4ID09PSBJbmZpbml0eSkge1xuICAgIHJldHVybiAxO1xuICB9IGVsc2UgaWYgKHggPT09IC1JbmZpbml0eSkge1xuICAgIHJldHVybiAtMTtcbiAgfSBlbHNlIHtcbiAgICBjb25zdCBlMnggPSBNYXRoLmV4cCgyICogeCk7XG4gICAgcmV0dXJuIChlMnggLSAxKSAvIChlMnggKyAxKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gc2l6ZVRvU3F1YXJpc2hTaGFwZShzaXplOiBudW1iZXIpOiBbbnVtYmVyLCBudW1iZXJdIHtcbiAgZm9yIChsZXQgYSA9IE1hdGguZmxvb3IoTWF0aC5zcXJ0KHNpemUpKTsgYSA+IDE7IC0tYSkge1xuICAgIGlmIChzaXplICUgYSA9PT0gMCkge1xuICAgICAgcmV0dXJuIFthLCBzaXplIC8gYV07XG4gICAgfVxuICB9XG4gIHJldHVybiBbMSwgc2l6ZV07XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVTaHVmZmxlZEluZGljZXMobjogbnVtYmVyKTogVWludDMyQXJyYXkge1xuICBjb25zdCBzaHVmZmxlZEluZGljZXMgPSBuZXcgVWludDMyQXJyYXkobik7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgbjsgKytpKSB7XG4gICAgc2h1ZmZsZWRJbmRpY2VzW2ldID0gaTtcbiAgfVxuICBzaHVmZmxlKHNodWZmbGVkSW5kaWNlcyk7XG4gIHJldHVybiBzaHVmZmxlZEluZGljZXM7XG59XG4iXX0=
