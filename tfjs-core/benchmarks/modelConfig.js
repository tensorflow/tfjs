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
      const zeros = tf.zeros([1, 224, 224, 3]);
      return model => model.predict(zeros);
    }
  },
  'facemesh': {
    load: async () => {
      const url =
          'https://storage.googleapis.com/learnjs-data/facemesh_staging/facemesh_facecontours_faceflag-blaze_shift30-2019_01_14-v0.hdf5_tfjs_fixed_batch/model.json';
      return tf.loadGraphModel(url);
    },
    predictFunc: () => {
      const zeros = tf.zeros([1, 192, 192, 3]);
      return model => {
        return model.predict(zeros)[0];
      };
    },
  },
  'blazeface': {
    load: async () => {
      const url =
        'https://tfhub.dev/tensorflow/tfjs-model/blazeface/1/default/1';
      return tf.loadGraphModel(url, {fromTFHub: true});
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
  'cocossd': {
    load: async () => {
      const model = await cocoSsd.load();
      model.image = await loadImage('tennis_standing.jpg');
      return model;
    },
    predictFunc: () => {
      return async model => {
        return model.detect(model.image);
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
  'deeplab': {
    load: async () => {
      const modelName = 'pascal';   // set to your preferred model, either `pascal`, `cityscapes` or `ade20k`
      const quantizationBytes = 2;  // either 1, 2 or 4
      return await deeplab.load({base: modelName, quantizationBytes});
    },
    predictFunc: () => {
      const zeros = tf.zeros([227, 500, 3]);
      return model => model.segment(zeros);
    }
  },
  'knn': {
    load: async() => {
      const url =
          'https://storage.googleapis.com/learnjs-data/mobilenet_v2_100_fused/model.json';
      window.mobilenetModule = await tf.loadGraphModel(url);
      const classifier = knnClassifier.create();

      const logits0 = mobilenetModule.predict(tf.zeros([1, 224, 224, 3]), 'conv_preds');
      classifier.addExample(logits0, 0);

      // window.img1 = await loadImage('tennis_standing.jpg');
      window.img1 = tf.ones([1, 224, 224, 3]);

      const logits1 = mobilenetModule.predict(img1, 'conv_preds');
      classifier.addExample(logits1, 1);
      return classifier;
    },
    predictFunc: () => {
      const logits0 = mobilenetModule.predict(window.img1, 'conv_preds');
      return async model => {
        return model.predictClass(logits0);
      }
    }
  },
  'speech-commands': {
    load: async () => {
      const recognizer = speechCommands.create('BROWSER_FFT');
      await recognizer.ensureModelLoaded();
      return recognizer;
    },
    predictFunc: (model) => {
      const shape = model.modelInputShape();
      const mySpectrogramData = new Float32Array(shape.reduce((acc, curr) => {
        if(curr == null) {
          return acc;
        }
        return acc * curr;
      }, 1));
      const x = tf.tensor4d(
        mySpectrogramData, [1].concat(shape.slice(1)));
      return async () => {
        return await model.recognize(x);
      }
    }
  },
  'toxicity': {
    load: async () => {
      return toxicity.load();
    },
    predictFunc: model => {
      const sentences = ['you suck'];
      return async () => {
        return model.classify(sentences);
      }
    }
  },
  'qna': {
    load: async () => {
      return qna.load();
    },
    predictFunc: () => {
      const passage = "Google LLC is an American multinational technology company that specializes in Internet-related services and products, which include online advertising technologies, search engine, cloud computing, software, and hardware. It is considered one of the Big Four technology companies, alongside Amazon, Apple, and Facebook. Google was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a California privately held company on September 4, 1998, in California. Google was then reincorporated in Delaware on October 22, 2002. An initial public offering (IPO) took place on August 19, 2004, and Google moved to its headquarters in Mountain View, California, nicknamed the Googleplex. In August 2015, Google announced plans to reorganize its various interests as a conglomerate called Alphabet Inc. Google is Alphabet's leading subsidiary and will continue to be the umbrella company for Alphabet's Internet interests. Sundar Pichai was appointed CEO of Google, replacing Larry Page who became the CEO of Alphabet.";
const question = "Who is the CEO of Google?";
      return async model => {
        return model.findAnswers(question, passage);
      }
    }
  }
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
