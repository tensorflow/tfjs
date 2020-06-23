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

const benchmarks = {
  'mobilenet_v2': {
    load: async () => {
      const url =
          'https://storage.googleapis.com/learnjs-data/mobilenet_v2_100_fused/model.json';
      return tf.loadGraphModel(url);
    },
    predictFunc: () => {
      const input = tf.randomNormal([1, 224, 224, 3]);
      return model => model.predict(input);
    }
  },
  'mesh_128': {
    load: async () => {
      const url =
          'https://storage.googleapis.com/learnjs-data/mesh_128_shift30_fixed_batch/model.json';
      return tf.loadGraphModel(url);
    },
    predictFunc: () => {
      const zeros = tf.zeros([1, 128, 128, 3]);
      return model => {
        return model.predict(zeros)[0];
      };
    },
  },
  'face_detector': {
    load: async () => {
      const url =
          'https://storage.googleapis.com/learnjs-data/face_detector_front/model.json';
      return tf.loadGraphModel(url);
    },
    predictFunc: () => {
      const zeros = tf.zeros([1, 128, 128, 3]);
      return model => {
        return model.predict(zeros);
      };
    },
  },
  'AutoML Image': {
    load: async () => {
      const url =
          'https://storage.googleapis.com/tfjs-testing/tfjs-automl/img_classification/model.json';
      return tf.automl.loadImageClassification(url);
    },
    predictFunc: () => {
      const zeros = tf.zeros([224, 224, 3]);
      return model => model.classify(zeros);
    }
  },
  'AutoML Object': {
    load: async () => {
      const url =
          'https://storage.googleapis.com/tfjs-testing/tfjs-automl/object_detection/model.json';
      return tf.automl.loadObjectDetection(url);
    },
    predictFunc: () => {
      const zeros = tf.zeros([224, 224, 3]);
      return model => model.detect(zeros);
    }
  },
  'USE - batchsize 30': {
    load: async () => {
      return use.load();
    },
    predictFunc: () => {
      const sentences30 = sentences.slice(0, 30);
      return async model => {
        const res = await model.embed(sentences30);
        return await res.data();
      }
    }
  },
  'USE - batchsize 1': {
    load: async () => {
      return use.load();
    },
    predictFunc: () => {
      let nextIdx = 0;

      return async model => {
        const next = [sentences[(nextIdx % sentences.length)]];
        const res = await model.embed(next);
        nextIdx += 1;
        return await res.data();
      }
    }
  },
  'posenet': {
    load: async () => {
      const model = await posenet.load();
      model.image = await loadImage('tennis_standing.jpg');
      return model;
    },
    predictFunc: () => {
      return async model => {
        return model.estimateSinglePose(model.image);
      }
    }
  },
  'bodypix': {
    load: async () => {
      const model = await bodyPix.load();
      model.image = await loadImage('tennis_standing.jpg');
      return model;
    },
    predictFunc: () => {
      return async model => {
        return model.segmentPerson(model.image);
      }
    }
  },
  'custom model': {
    load: async () => {
      return loadModelByUrl(state.modelUrl);
    },
    predictFunc: () => {
      return async model => {
        const inferenceInputs = [];
        try {
          for (let inferenceInputIndex = 0; inferenceInputIndex < model.inputs.length; inferenceInputIndex++) {
            // Construct the input tensor shape
            const inferenceInput = model.inputs[inferenceInputIndex];
            const inputShape = [];
            for (let dimension = 0; dimension < inferenceInput.shape.length; dimension++) {
              const shapeValue = inferenceInput.shape[dimension];
              if (shapeValue == null || shapeValue < 0) {
                inputShape.push(1);
              } else if (shapeValue == 0) {
                await showMsg('Warning: one dimension of an input tensor is zero');
                inputShape.push(shapeValue);
              } else {
                inputShape.push(shapeValue);
              }
            }

            // Construct the input tensor
            let inputTensor;
            if (inferenceInput.dtype == 'float32' || inferenceInput.dtype == 'int32') {
              inputTensor = tf.randomNormal(inputShape, 0, 1, inferenceInput.dtype);
            } else {
              throw new Error(`${inferenceInput.dtype} dtype is not supported`);
            }
            inferenceInputs.push(inputTensor);
          }

          let resultTensor;
          if (model instanceof tf.GraphModel && model.executeAsync != null) {
            resultTensor = await model.executeAsync(inferenceInputs);
          } else if (model.predict != null) {
            resultTensor = model.predict(inferenceInputs);
          } else {
            throw new Error("Predict function was not found");
          }
          return resultTensor;
        } catch (e) {
          showMsg('Error: ' + e.message);
          throw e;
        } finally {
          // dispose input tensors
          for (let tensorIndex = 0; tensorIndex < inferenceInputs.length; tensorIndex++) {
            if (inferenceInputs[tensorIndex] instanceof tf.Tensor) {
              inferenceInputs[tensorIndex].dispose();
            }
          }
        }
      }
    }
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
          `URL '${[path]}'`);
    }
    handler = handlers[0];
  }
  return handler;
}

async function tryAllLoadingMethods(modelHandler, loadOptions = {}) {
  let model;
  // TODO: download weights once
  model = await tf.loadGraphModel(modelHandler, loadOptions).then(model => {
    state.modelType = 'GraphModel';
    return model;
  }).catch(e => {});

  if (model == null) {
    model = await tf.loadLayersModel(modelHandler, loadOptions).then(model => {
      state.modelType = 'LayersModel';
      return model;
    });
  }
  return model;
}

async function loadModelByUrl(modelUrl, loadOptions = {}) {
  let model, ioHandler, modelType;
  state.modelType = '';

  const supportedSchemes =  /^(https?|localstorage|indexeddb):\/\/.+$/;
  if (!supportedSchemes.test(modelUrl)) {
    throw new Error(`Please use a valid URL, such as 'https://'`);
  }

  const tfHubUrl =  /^https:\/\/tfhub.dev\/.+$/;
  if (loadOptions.fromTFHub || tfHubUrl.test(modelUrl)) {
    if (!modelUrl.endsWith('/')) {
      modelUrl = modelUrl + '/';
    }
    modelUrl = `${modelUrl}${DEFAULT_MODEL_NAME}${TFHUB_SEARCH_PARAM}`;
  }

  // Convert URL to IOHandler and parse the model type
  try {
    ioHandler = findIOHandler(modelUrl, loadOptions);
    modelType = await ioHandler.load().then(artifacts => artifacts.format);
  } catch (e) {
    throw new Error(`Failed to fetch or parse 'model.json' file`);
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
      model = await tryAllLoadingMethods(ioHandler, loadOptions);
    }
  } catch (e) {
    throw new Error('Failed to load the model');
  }

  return model;
}
