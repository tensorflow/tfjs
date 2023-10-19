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

const DEFAULT_MODEL_NAME = 'model.json';
const TFHUB_SEARCH_PARAM = '?tfjs-format=file';

const sentences = [
  'add clem burke in my playlist Pre-Party R&B Jams',
  'Add Live from Aragon Ballroom to Trapeo',
  'add Unite and Win to my night out',
  'Add track to my Digster Future Hits',
  'add the piano bar to my Cindy Wilson',
  'Add Spanish Harlem Incident to cleaning the house',
  'add The Greyest of Blue Skies in Indie Español my playlist',
  'Add the name kids in the street to the plylist New Indie Mix',
  'add album radar latino',
  'Add Tranquility to the Latin Pop Rising playlist.',
  'play something from the twenties',
  'Play The View From The Afternoon by Malese Jow on Last Fm',
  'play songs by Sammy Fain',
  'Play music from the year 1964',
  'Play the heinz strobl ep from 2016 on Groove Shark',
  'Play me Leonid Soybelman on Vimeo.',
  'Play a song from my workout playlist on Groove Shark',
  'play some Alte Kameraden music',
  'Will it be warm 1 week from now in DC',
  'what is the forecast for temperate conditions in Thailand in Lopeno',
  'Is the weather colder in Costa Rica',
  'Will it be colder in Delaware?',
  '"I need to know the weather for Hamorton, TN"',
  'What will the weather be in Albania at 11:56.',
  'Is it going to hail in Mount San Jacinto State Park',
  'What\'s the forecast for Walker Bay Nature Reserve for next year ? ',
  'is it supposed to be sunny here?',
  'in California will it be cold in East Trenton Heights',
  'What is the weather like in Wallis and Futuna? What will the weather be in Romania at 4?',
  'What is the weather going to be like in Reidland New Mexico next Jun.?',
  'How cold is it in Cargray, Argentina?',
  'Is the forecast chillier in 1 hour in Mali',
  'Tell me if there will be wind in NE Will it be cloudy not far  from Allenton Will there be a blizzard in AR what is the New Caledonia forecast for Bagnell on sep. the 5th Weather for apr. the thirteenth in Djibouti',
  'Can you give me the weather forecast in Tajikistan? How cold is it going to be in San Marcial, AK in one second? What will the weather be in a month from now at my current location?',
  'What is the weather like in IA in april How windy is it in Anderson Lake State Fish and Wildlife Area? Is it going to be stormy in Austin Creek State Recreation Area at 09:42?',
  'When will the weather be temperate like it is now in Stansbury Park in Tuvalu, What is the weather in neighboring OH, What\'s the weather forecast for Spain ? ',
  'Play the music Hands Up',
  'Play some twenties theme music on Google Music.',
  'How will the weather be in New Mexico around 00:09:07 am?',
  'What will the humidity be in AR in 49 weeks and a half from now',
  'Is it humid in Parc national de Killarney',
  'is it supposed to get colder here on 12/28/2019',
  'How is the forecast for OK?',
  'what is the Posey Island State Park forecast for colder temps at meal time',
  'Is it supposed to be chilly in Kuwait?',
  'Tell me if it\'ll be chilly here at 0 pm',
  'what is the forecast for colder conditions within the same area of this current place',
  'Will it hail today in West Point at 11:36:48',
  'Is it overcast in South Carolina',
  'Will the sun be out close-by Admiralty Island National Monument?',
  'What will the weather be in Wakarusa',
  'How temperate will it be here this week?',
  'what is the forecast for here at tea time',
];

function predictFunction(input) {
  return model => model.predict(input);
}

// Common predict function for MobileNetV3 and MobileNetV2Lite
function commonMobileNetPredictFunc() {
  const input = tf.randomNormal([1, 224, 224, 3]);
  const inputData = input.dataSync();
  if (typeof isTflite === 'function' && isTflite()) {
    return async () => {
      // Do copy from 'inputData' in each predict as its buffer
      // will be detached after transferring to worker.
      const input = inputData.slice(0);
      return await tfliteModel.predict(
          Comlink.transfer(input, [input.buffer]));
    };
  } else {
    return predictFunction(input);
  }
}

const benchmarks = {
  'MobileNetV3': {
    type: 'GraphModel',
    architectures: ['small_075', 'small_100', 'large_075', 'large_100'],
    load: async (inputResolution = 224, modelArchitecture = 'small_075') => {
      const url = `https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_${
          modelArchitecture}_224/classification/5/default/1`;
      return tf.loadGraphModel(url, {fromTFHub: true});
    },
    loadTflite: async (
        enableProfiling = false, modelArchitecture = 'small_075') => {
      const url = `https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_${
          modelArchitecture}_224/classification/5/metadata/1`;
      return await tfliteWorkerAPI.loadTFLiteModel(url, {enableProfiling});
    },
    predictFunc: commonMobileNetPredictFunc,
  },
  'MobileNetV2': {
    type: 'GraphModel',
    architectures: ['050', '075', '100'],
    load: async (inputResolution = 224, modelArchitecture = '050') => {
      const url = `https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_${
          modelArchitecture}_224/classification/3/default/1`;
      return tf.loadGraphModel(url, {fromTFHub: true});
    },
    predictFunc: () => {
      const input = tf.randomNormal([1, 224, 224, 3]);
      return predictFunction(input);
    },
  },
  // Currently, for mobilenet_v2, only the architectures with alpha=100 has
  // tflite model. Since users could tune the alpha for 'mobilenet_v2' tfjs
  // models, while we could only provides mobilnet_v2_lite with alpha=100 on the
  // tflite backend, so mibilnet_v2_lite is separated from mibilnet_v2 and fixes
  // alpha=100; othwise it would confuse users.
  'MobileNetV2Lite': {
    type: 'GraphModel',
    load: async () => {
      throw new Error(`Please set tflite as the backend to run this model.`);
    },
    loadTflite: async (enableProfiling = false) => {
      const url =
          'https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224/1/metadata/1';
      return await tfliteWorkerAPI.loadTFLiteModel(url, {enableProfiling});
    },
    predictFunc: commonMobileNetPredictFunc,
  },
  'HandPoseDetector': {
    type: 'GraphModel',
    architectures: ['lite', 'full'],
    load: async (inputResolution = 192, modelArchitecture = 'lite') => {
      const url =
          `https://tfhub.dev/mediapipe/tfjs-model/handpose_3d/detector/${
              modelArchitecture}/1`;
      return tf.loadGraphModel(url, {fromTFHub: true});
    },
    predictFunc: () => {
      const input = tf.zeros([1, 192, 192, 3]);
      return predictFunction(input);
    },
  },
  'HandPoseLandmark': {
    type: 'GraphModel',
    architectures: ['lite', 'full'],
    load: async (inputResolution = 224, modelArchitecture = 'lite') => {
      const url =
          `https://tfhub.dev/mediapipe/tfjs-model/handpose_3d/landmark/${
              modelArchitecture}/1`;
      return tf.loadGraphModel(url, {fromTFHub: true});
    },
    predictFunc: () => {
      const input = tf.zeros([1, 224, 224, 3]);
      return predictFunction(input);
    },
  },
  'MoveNet-SinglePose': {
    type: 'GraphModel',
    inputSizes: [192, 256],
    load: async (inputResolution = 192) => {
      const modelType = inputResolution === 192 ? 'lightning' : 'thunder';
      const url = `https://tfhub.dev/google/tfjs-model/movenet/singlepose/${
          modelType}/4`;
      return tf.loadGraphModel(url, {fromTFHub: true});
    },
    predictFunc: (inputResolution = 192) => {
      const input = tf.zeros([1, inputResolution, inputResolution, 3], 'int32');
      return predictFunction(input);
    },
  },
  'MoveNet-MultiPose': {
    type: 'GraphModel',
    load: async () => {
      const url =
          'https://tfhub.dev/google/tfjs-model/movenet/multipose/lightning/1';
      return tf.loadGraphModel(url, {fromTFHub: true});
    },
    predictFunc: () => {
      const input = tf.zeros([1, 256, 256, 3], 'int32');
      return predictFunction(input);
    },
  },
  'BlazePoseDetector': {
    type: 'GraphModel',
    load: async () => {
      const url =
          'https://tfhub.dev/mediapipe/tfjs-model/blazeposedetector/1/default/1';
      return tf.loadGraphModel(url, {fromTFHub: true});
    },
    predictFunc: () => {
      const input = tf.zeros([1, 224, 224, 3]);
      return predictFunction(input);
    },
  },
  'BlazePoseLandmark': {
    type: 'GraphModel',
    architectures: ['lite', 'full', 'heavy'],
    load: async (inputResolution = 256, modelArchitecture = 'lite') => {
      const url = `https://tfhub.dev/mediapipe/tfjs-model/blazeposelandmark_${
          modelArchitecture}/2/default/2`;
      return tf.loadGraphModel(url, {fromTFHub: true});
    },
    predictFunc: () => {
      const input = tf.zeros([1, 256, 256, 3]);
      return predictFunction(input);
    },
  },
  'Coco-SSD': {
    type: 'GraphModel',
    // The model has has the dynamic ops, so it is supposed to use executeAsync.
    architectures: ['MobileNetV2', 'MobileNetV1', 'liteMobileNetV2'],
    load: async (inputResolution = 227, modelArchitecture = 'MobileNetV2') => {
      const tfliteBased = modelArchitecture.split('MobileNetV')[0];
      const mobileNetVersion = modelArchitecture.split('MobileNetV')[1];
      const url = `https://storage.googleapis.com/tfjs-models/savedmodel/ssd${
          tfliteBased}_mobilenet_v${mobileNetVersion}/model.json`;
      return tf.loadGraphModel(url);
    },
    predictFunc: () => {
      const input = tf.zeros([1, 227, 227, 3], 'int32');
      return model => model.executeAsync(input);
    },
  },
  'DeepLabV3': {
    type: 'GraphModel',
    // TODO: Add cityscapes architecture.
    // https://github.com/tensorflow/tfjs/issues/6733
    architectures: ['pascal', 'ade20k'],
    load: async (inputResolution = 227, modelArchitecture = 'pascal') => {
      const url =
          `https://storage.googleapis.com/tfhub-tfjs-modules/tensorflow/tfjs-model/deeplab/${
              modelArchitecture}/1/quantized/2/1/model.json`;
      return tf.loadGraphModel(url);
    },
    predictFunc: () => {
      const input = tf.zeros([1, 227, 500, 3], 'int32');
      return predictFunction(input);
    },
  },
  'FaceDetection': {
    type: 'GraphModel',
    inputSizes: [128, 192],
    load: async (inputResolution = 128) => {
      const modelSize = inputResolution === 128 ? 'short' : 'full';
      const url = `https://tfhub.dev/mediapipe/tfjs-model/face_detection/${
          modelSize}/1`;
      return tf.loadGraphModel(url, {fromTFHub: true});
    },
    predictFunc: (inputResolution = 128) => {
      const input = tf.zeros([1, inputResolution, inputResolution, 3]);
      return predictFunction(input);
    },
  },
  'FaceLandmarkDetection': {
    type: 'GraphModel',
    architectures: ['attention_mesh', 'face_mesh'],
    load: async (
        inputResolution = 192, modelArchitecture = 'attention_mesh') => {
      const url =
          `https://tfhub.dev/mediapipe/tfjs-model/face_landmarks_detection/${
              modelArchitecture}/1`;
      return tf.loadGraphModel(url, {fromTFHub: true});
    },
    predictFunc: () => {
      const input = tf.zeros([1, 192, 192, 3]);
      return predictFunction(input);
    },
  },
  'ArPortraitDepth': {
    type: 'GraphModel',
    load: async () => {
      const url = 'https://tfhub.dev/tensorflow/tfjs-model/ar_portrait_depth/1';
      return tf.loadGraphModel(url, {fromTFHub: true});
    },
    predictFunc: () => {
      const input = tf.zeros([1, 256, 192, 3]);
      return predictFunction(input);
    },
  },
  'SelfieSegmentation-General': {
    type: 'GraphModel',
    load: async () => {
      const url =
          'https://tfhub.dev/mediapipe/tfjs-model/selfie_segmentation/general/1';
      return tf.loadGraphModel(url, {fromTFHub: true});
    },
    predictFunc: () => {
      const input = tf.zeros([1, 256, 256, 3]);
      return predictFunction(input);
    },
  },
  'SelfieSegmentation-Landscape': {
    type: 'GraphModel',
    load: async () => {
      const url =
          'https://tfhub.dev/mediapipe/tfjs-model/selfie_segmentation/landscape/1';
      return tf.loadGraphModel(url, {fromTFHub: true});
    },
    predictFunc: () => {
      const input = tf.zeros([1, 144, 256, 3]);
      return predictFunction(input);
    },
  },
  'AutoML Image': {
    type: 'GraphModel',
    load: async () => {
      const url =
          'https://storage.googleapis.com/tfjs-testing/tfjs-automl/img_classification/model.json';
      return tf.loadGraphModel(url);
    },
    predictFunc: () => {
      const zeros = tf.zeros([1, 224, 224, 3]);
      return predictFunction(zeros);
    }
  },
  'AutoML Object': {
    type: 'GraphModel',
    load: async () => {
      const url =
          'https://storage.googleapis.com/tfjs-testing/tfjs-automl/object_detection/model.json';
      return tf.loadGraphModel(url);
    },
    predictFunc: () => {
      const OUTPUT_NODE_NAMES =
          ['Postprocessor/convert_scores', 'Postprocessor/Decode/transpose_1'];
      const feedDict = {ToFloat: tf.zeros([1, 224, 224, 3])};
      return model => model.executeAsync(feedDict, OUTPUT_NODE_NAMES);
    }
  },
  'USE - batchsize 30': {
    type: 'GraphModel',
    load: async () => {
      return use.load();
    },
    predictFunc: () => {
      const sentences30 = sentences.slice(0, 30);
      return async model => {
        const res = await model.embed(sentences30);
        return res;
      };
    }
  },
  'USE - batchsize 1': {
    type: 'GraphModel',
    load: async () => {
      return use.load();
    },
    predictFunc: () => {
      return async model => {
        const next = [sentences[0]];
        const res = await model.embed(next);
        return res;
      };
    }
  },
  'TextToxicity': {
    type: 'GraphModel',
    // The model has has the dynamic ops, so it is supposed to use executeAsync.
    load: async () => {
      const url =
          'https://storage.googleapis.com/tfhub-tfjs-modules/tensorflow/tfjs-model/toxicity/1/default/1/model.json';
      return tf.loadGraphModel(url);
    },
    predictFunc: () => {
      const indices = tf.tensor2d(
          [0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9], [10, 2],
          'int32');
      const values = tf.randomUniform([10], 0, 1000, 'int32');
      const input = {Placeholder_1: indices, Placeholder: values};
      return model => model.executeAsync(input);
    }
  },
  'MobileBert': {
    type: 'GraphModel',
    load: async () => {
      const url = 'https://tfhub.dev/tensorflow/tfjs-model/mobilebert/1';
      return tf.loadGraphModel(url, {fromTFHub: true});
    },
    predictFunc: () => {
      const batchSize = 1;
      const INPUT_SIZE = 384;
      const inputIds = tf.ones([batchSize, INPUT_SIZE], 'int32');
      const segmentIds = tf.ones([1, INPUT_SIZE], 'int32');
      const inputMask = tf.ones([1, INPUT_SIZE], 'int32');
      const input = {
        input_ids: inputIds,
        segment_ids: segmentIds,
        input_mask: inputMask,
        global_step: tf.scalar(1, 'int32')
      };
      return predictFunction(input);
    }
  },
  'posenet': {
    type: 'GraphModel',
    inputSizes: [128, 256, 512, 1024],
    architectures: ['MobileNetV1', 'ResNet50'],
    inputTypes: ['image', 'tensor'],
    load: async (
        inputResolution = 128, modelArchitecture = 'MobileNetV1',
        inputType = 'image') => {
      let config = null;
      if (modelArchitecture === 'MobileNetV1') {
        config = {
          architecture: modelArchitecture,
          outputStride: 16,
          multiplier: 0.75,
          inputResolution: inputResolution,
        };
      } else if (modelArchitecture === 'ResNet50') {
        config = {
          architecture: modelArchitecture,
          outputStride: 32,
          quantBytes: 2,
          inputResolution: inputResolution,
        };
      }
      const model = await posenet.load(config);
      if (inputType === 'tensor') {
        model.input = tf.zeros([inputResolution, inputResolution, 3]);
      } else {
        model.input = await loadImage('tennis_standing.jpg');
      }
      return model;
    },
    predictFunc: () => {
      return async model => {
        return model.estimateSinglePose(model.input);
      };
    }
  },
  'bodypix': {
    type: 'GraphModel',
    // The ratio to the default camera size [480, 640].
    inputSizes: [0.25, 0.5, 0.75, 1.0],
    architectures: ['ResNet50'],
    inputTypes: ['image', 'tensor'],
    load: async (
        internalResolution, modelArchitecture = 'MobileNetV1',
        inputType = 'image') => {
      let config = {
        architecture: 'ResNet50',
        outputStride: 32,
        quantBytes: 4,
      };
      const model = await bodyPix.load(config);
      if (inputType === 'tensor') {
        model.input =
            tf.zeros([480 * internalResolution, 640 * internalResolution, 3]);
      } else {
        model.input = await loadImage('tennis_standing.jpg');
      }
      return model;
    },
    predictFunc: (internalResolution = 0.5) => {
      return async model => {
        const PERSON_INFERENCE_CONFIG = {
          internalResolution: internalResolution,
        };
        return model.segmentPerson(model.input, PERSON_INFERENCE_CONFIG);
      };
    }
  },
  'speech-commands': {
    // Layer model doesn't support dump. TODO(xing.xu@intel.com):
    // https://github.com/tensorflow/tfjs/issues/7010
    supportDump: false,
    load: async () => {
      const recognizer = speechCommands.create('BROWSER_FFT');
      await recognizer.ensureModelLoaded();
      return recognizer;
    },
    predictFunc: () => {
      return async (model) => {
        const shape = model.modelInputShape();
        // Cannot use tf.util.sizeFromShape because shape includes null.
        const mySpectrogramData = new Float32Array(shape.reduce((acc, curr) => {
          if (curr == null) {
            return acc;
          }
          return acc * curr;
        }, 1));
        const x = tf.tensor4d(mySpectrogramData, [1].concat(shape.slice(1)));
        return await model.recognize(x);
      }
    }
  },
  'custom': {
    type: '',
    load: async () => {
      return loadModelByUrlWithState(state.modelUrl, {}, state);
    },
    loadTflite: async (enableProfiling = false) => {
      return await tfliteWorkerAPI.loadTFLiteModel(state.modelUrl, {enableProfiling});
    },
    predictFunc: () => {
      return async (model, customInput) => {
        let inferenceInput;
        try {
          inferenceInput = customInput ||
              generateInputFromDef(
                               state.inputs, model instanceof tf.GraphModel);
          if (typeof isTflite === 'function' && isTflite()) {
            const inputDataArray = [];
            inferenceInput = inferenceInput instanceof
                Array ? inferenceInput : [inferenceInput];
            for (let tensor of inferenceInput) {
              const inputData = tensor.dataSync();
              // Do copy from 'inputData' in each predict as its buffer
              // will be detached after transferring to worker.
              inputDataArray.push(inputData.slice(0));
            }

            const buffer = inputDataArray.map(data => data.buffer);
            return await tfliteModel.predict(
                Comlink.transfer(inputDataArray, buffer));
          } else {
            const predict = getPredictFnForModel(model, inferenceInput);
            return await predict();
          }
        } finally {
          // dispose input tensors
          if (!customInput && inferenceInput) {
            tf.dispose(inferenceInput);
          }
        }
      };
    },
  },
};


const imageBucket =
    'https://storage.googleapis.com/tfjs-models/assets/posenet/';
async function loadImage(imagePath) {
  const image = new Image();
  const promise = new Promise((resolve, reject) => {
    image.crossOrigin = '';
    image.onload = () => {
      resolve(image);
    };
  });

  image.src = `${imageBucket}${imagePath}`;
  return promise;
}

function findIOHandler(path, loadOptions = {}) {
  let handler;
  if (path.load != null) {
    handler = path;
  } else if (loadOptions.requestInit != null) {
    handler = tf.io.browserHTTPRequest(path, loadOptions);
  } else {
    const handlers = tf.io.getLoadHandlers(path, loadOptions);
    if (handlers.length === 0) {
      handlers.push(tf.io.browserHTTPRequest(path, loadOptions));
    } else if (handlers.length > 1) {
      throw new Error(
          `Found more than one (${handlers.length}) load handlers for ` +
          `URL '${[path]}'.`);
    }
    handler = handlers[0];
  }
  return handler;
}

async function tryAllLoadingMethods(
    modelHandler, loadOptions = {}, state = {}) {
  let model;
  // TODO: download weights once
  try {
    model = await tf.loadGraphModel(modelHandler, loadOptions);
    state.modelType = 'GraphModel';
    return model;
  } catch (e) {
  }

  try {
    model = await tf.loadLayersModel(modelHandler, loadOptions);
    state.modelType = 'LayersModel';
    return model;
  } catch (e) {
  }

  throw new Error(`Didn't find a fit loading method for this model.`);
}

/**
 * Load a graph model or a a model composed of Layer objects and record the
 * model type (GraphModel or LayersModel) at `state.modelType`, given a URL to
 * the model definition.
 *
 * @param {string} modelUrl
 * @param {io.LoadOptions} loadOptions
 * @param {object} state  The object that is used to record the model type.
 *     This can be extended with more model information if needed.
 */
async function loadModelByUrlWithState(modelUrl, loadOptions = {}, state = {}) {
  let model, ioHandler, modelType;

  const supportedSchemes = /^(https?|localstorage|indexeddb):\/\/.+$/;
  if (!supportedSchemes.test(modelUrl)) {
    throw new Error(`Please use a valid URL, such as 'https://'.`);
  }

  const tfHubUrl = /^https:\/\/tfhub.dev\/.+$/;
  if (loadOptions.fromTFHub || tfHubUrl.test(modelUrl)) {
    if (!modelUrl.endsWith('/')) {
      modelUrl = modelUrl + '/';
    }
    modelUrl = `${modelUrl}${DEFAULT_MODEL_NAME}${TFHUB_SEARCH_PARAM}`;
  }

  // Convert URL to IOHandler and parse the model type
  try {
    ioHandler = findIOHandler(modelUrl, loadOptions);
    const artifacts = await ioHandler.load();
    modelType = artifacts.format;
  } catch (e) {
    console.error(`Failed to fetch or parse 'model.json' file.`);
    throw e;
  }

  // load models
  try {
    if (modelType === 'graph-model') {
      model = await tf.loadGraphModel(ioHandler, loadOptions);
      state.modelType = 'GraphModel';
    } else if (modelType === 'layers-model') {
      model = await tf.loadLayersModel(ioHandler, loadOptions);
      state.modelType = 'LayersModel';
    } else {
      model = await tryAllLoadingMethods(ioHandler, loadOptions, state);
    }
  } catch (e) {
    console.error('Failed to load the model.');
    throw e;
  }

  return model;
}

async function loadModelByUrl(modelUrl, loadOptions = {}) {
  const state = {};
  return loadModelByUrlWithState(modelUrl, loadOptions, state);
}
