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
(function(global, factory) {
typeof exports === 'object' && typeof module !== 'undefined' ?
    factory(
        exports, require('@tensorflow/tfjs-core'),
        require('@tensorflow/tfjs-converter')) :
    typeof define === 'function' && define.amd ?
    define(
        ['exports', '@tensorflow/tfjs-core', '@tensorflow/tfjs-converter'],
        factory) :
    (global = global || self,
     factory(global.posenet = {}, global.tf, global.tf));
}(this, function(exports, tf, tfconv) {
'use strict';

const VALID_OUTPUT_STRIDES = [8, 16, 32];
function assertValidOutputStride(outputStride) {
  tf.util.assert(
      typeof outputStride === 'number', () => 'outputStride is not a number');
  tf.util.assert(
      VALID_OUTPUT_STRIDES.indexOf(outputStride) >= 0,
      () => `outputStride of ${outputStride} is invalid. ` +
          `It must be either 8, 16, or 32`);
}
function assertValidResolution(resolution, outputStride) {
  tf.util.assert(
      typeof resolution === 'number', () => 'resolution is not a number');
  tf.util.assert(
      (resolution - 1) % outputStride === 0,
      () => `resolution of ${resolution} is invalid for output stride ` +
          `${outputStride}.`);
}
function toFloatIfInt(input) {
  return tf.tidy(() => {
    if (input.dtype === 'int32') input = input.toFloat();
    input = tf.div(input, 127.5);
    return tf.sub(input, 1.0);
  });
}
class MobileNet {
  constructor(model, outputStride) {
    this.model = model;
    const inputShape = this.model.inputs[0].shape;
    tf.util.assert(
        (inputShape[1] === -1) && (inputShape[2] === -1),
        () => `Input shape [${inputShape[1]}, ${inputShape[2]}] ` +
            `must both be -1`);
    this.outputStride = outputStride;
  }
  predict(input) {
    return tf.tidy(() => {
      const asFloat = toFloatIfInt(input);
      const asBatch = asFloat.expandDims(0);
      const [offsets4d, heatmaps4d, displacementFwd4d, displacementBwd4d] =
          this.model.predict(asBatch);
      const heatmaps = heatmaps4d.squeeze();
      const heatmapScores = heatmaps.sigmoid();
      const offsets = offsets4d.squeeze();
      const displacementFwd = displacementFwd4d.squeeze();
      const displacementBwd = displacementBwd4d.squeeze();
      return {
        heatmapScores,
        offsets: offsets,
        displacementFwd: displacementFwd,
        displacementBwd: displacementBwd
      };
    });
  }
  dispose() {
    this.model.dispose();
  }
}

function half(k) {
  return Math.floor(k / 2);
}
class MaxHeap {
  constructor(maxSize, getElementValue) {
    this.priorityQueue = new Array(maxSize);
    this.numberOfElements = -1;
    this.getElementValue = getElementValue;
  }
  enqueue(x) {
    this.priorityQueue[++this.numberOfElements] = x;
    this.swim(this.numberOfElements);
  }
  dequeue() {
    const max = this.priorityQueue[0];
    this.exchange(0, this.numberOfElements--);
    this.sink(0);
    this.priorityQueue[this.numberOfElements + 1] = null;
    return max;
  }
  empty() {
    return this.numberOfElements === -1;
  }
  size() {
    return this.numberOfElements + 1;
  }
  all() {
    return this.priorityQueue.slice(0, this.numberOfElements + 1);
  }
  max() {
    return this.priorityQueue[0];
  }
  swim(k) {
    while (k > 0 && this.less(half(k), k)) {
      this.exchange(k, half(k));
      k = half(k);
    }
  }
  sink(k) {
    while (2 * k <= this.numberOfElements) {
      let j = 2 * k;
      if (j < this.numberOfElements && this.less(j, j + 1)) {
        j++;
      }
      if (!this.less(k, j)) {
        break;
      }
      this.exchange(k, j);
      k = j;
    }
  }
  getValueAt(i) {
    return this.getElementValue(this.priorityQueue[i]);
  }
  less(i, j) {
    return this.getValueAt(i) < this.getValueAt(j);
  }
  exchange(i, j) {
    const t = this.priorityQueue[i];
    this.priorityQueue[i] = this.priorityQueue[j];
    this.priorityQueue[j] = t;
  }
}

function scoreIsMaximumInLocalWindow(
    keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores) {
  const [height, width] = scores.shape;
  let localMaximum = true;
  const yStart = Math.max(heatmapY - localMaximumRadius, 0);
  const yEnd = Math.min(heatmapY + localMaximumRadius + 1, height);
  for (let yCurrent = yStart; yCurrent < yEnd; ++yCurrent) {
    const xStart = Math.max(heatmapX - localMaximumRadius, 0);
    const xEnd = Math.min(heatmapX + localMaximumRadius + 1, width);
    for (let xCurrent = xStart; xCurrent < xEnd; ++xCurrent) {
      if (scores.get(yCurrent, xCurrent, keypointId) > score) {
        localMaximum = false;
        break;
      }
    }
    if (!localMaximum) {
      break;
    }
  }
  return localMaximum;
}
function buildPartWithScoreQueue(scoreThreshold, localMaximumRadius, scores) {
  const [height, width, numKeypoints] = scores.shape;
  const queue = new MaxHeap(height * width * numKeypoints, ({score}) => score);
  for (let heatmapY = 0; heatmapY < height; ++heatmapY) {
    for (let heatmapX = 0; heatmapX < width; ++heatmapX) {
      for (let keypointId = 0; keypointId < numKeypoints; ++keypointId) {
        const score = scores.get(heatmapY, heatmapX, keypointId);
        if (score < scoreThreshold) {
          continue;
        }
        if (scoreIsMaximumInLocalWindow(
                keypointId, score, heatmapY, heatmapX, localMaximumRadius,
                scores)) {
          queue.enqueue({score, part: {heatmapY, heatmapX, id: keypointId}});
        }
      }
    }
  }
  return queue;
}

const partNames = [
  'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder',
  'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist',
  'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'
];
const NUM_KEYPOINTS = partNames.length;
const partIds = partNames.reduce((result, jointName, i) => {
  result[jointName] = i;
  return result;
}, {});
const connectedPartNames = [
  ['leftHip', 'leftShoulder'], ['leftElbow', 'leftShoulder'],
  ['leftElbow', 'leftWrist'], ['leftHip', 'leftKnee'],
  ['leftKnee', 'leftAnkle'], ['rightHip', 'rightShoulder'],
  ['rightElbow', 'rightShoulder'], ['rightElbow', 'rightWrist'],
  ['rightHip', 'rightKnee'], ['rightKnee', 'rightAnkle'],
  ['leftShoulder', 'rightShoulder'], ['leftHip', 'rightHip']
];
const poseChain = [
  ['nose', 'leftEye'], ['leftEye', 'leftEar'], ['nose', 'rightEye'],
  ['rightEye', 'rightEar'], ['nose', 'leftShoulder'],
  ['leftShoulder', 'leftElbow'], ['leftElbow', 'leftWrist'],
  ['leftShoulder', 'leftHip'], ['leftHip', 'leftKnee'],
  ['leftKnee', 'leftAnkle'], ['nose', 'rightShoulder'],
  ['rightShoulder', 'rightElbow'], ['rightElbow', 'rightWrist'],
  ['rightShoulder', 'rightHip'], ['rightHip', 'rightKnee'],
  ['rightKnee', 'rightAnkle']
];
const connectedPartIndices = connectedPartNames.map(
    ([jointNameA, jointNameB]) => ([partIds[jointNameA], partIds[jointNameB]]));
const partChannels = [
  'left_face',
  'right_face',
  'right_upper_leg_front',
  'right_lower_leg_back',
  'right_upper_leg_back',
  'left_lower_leg_front',
  'left_upper_leg_front',
  'left_upper_leg_back',
  'left_lower_leg_back',
  'right_feet',
  'right_lower_leg_front',
  'left_feet',
  'torso_front',
  'torso_back',
  'right_upper_arm_front',
  'right_upper_arm_back',
  'right_lower_arm_back',
  'left_lower_arm_front',
  'left_upper_arm_front',
  'left_upper_arm_back',
  'left_lower_arm_back',
  'right_hand',
  'right_lower_arm_front',
  'left_hand'
];

function getOffsetPoint(y, x, keypoint, offsets) {
  return {
    y: offsets.get(y, x, keypoint),
    x: offsets.get(y, x, keypoint + NUM_KEYPOINTS)
  };
}
function getImageCoords(part, outputStride, offsets) {
  const {heatmapY, heatmapX, id: keypoint} = part;
  const {y, x} = getOffsetPoint(heatmapY, heatmapX, keypoint, offsets);
  return {
    x: part.heatmapX * outputStride + x,
    y: part.heatmapY * outputStride + y
  };
}
function clamp(a, min, max) {
  if (a < min) {
    return min;
  }
  if (a > max) {
    return max;
  }
  return a;
}
function squaredDistance(y1, x1, y2, x2) {
  const dy = y2 - y1;
  const dx = x2 - x1;
  return dy * dy + dx * dx;
}
function addVectors(a, b) {
  return {x: a.x + b.x, y: a.y + b.y};
}

const parentChildrenTuples = poseChain.map(
    ([parentJoinName, childJoinName]) =>
        ([partIds[parentJoinName], partIds[childJoinName]]));
const parentToChildEdges =
    parentChildrenTuples.map(([, childJointId]) => childJointId);
const childToParentEdges = parentChildrenTuples.map(([
                                                      parentJointId,
                                                    ]) => parentJointId);
function getDisplacement(edgeId, point, displacements) {
  const numEdges = displacements.shape[2] / 2;
  return {
    y: displacements.get(point.y, point.x, edgeId),
    x: displacements.get(point.y, point.x, numEdges + edgeId)
  };
}
function getStridedIndexNearPoint(point, outputStride, height, width) {
  return {
    y: clamp(Math.round(point.y / outputStride), 0, height - 1),
    x: clamp(Math.round(point.x / outputStride), 0, width - 1)
  };
}
function traverseToTargetKeypoint(
    edgeId, sourceKeypoint, targetKeypointId, scoresBuffer, offsets,
    outputStride, displacements, offsetRefineStep = 2) {
  const [height, width] = scoresBuffer.shape;
  const sourceKeypointIndices = getStridedIndexNearPoint(
      sourceKeypoint.position, outputStride, height, width);
  const displacement =
      getDisplacement(edgeId, sourceKeypointIndices, displacements);
  let displacedPoint = addVectors(sourceKeypoint.position, displacement);
  let targetKeypoint = displacedPoint;
  for (let i = 0; i < offsetRefineStep; i++) {
    const targetKeypointIndices =
        getStridedIndexNearPoint(targetKeypoint, outputStride, height, width);
    const offsetPoint = getOffsetPoint(
        targetKeypointIndices.y, targetKeypointIndices.x, targetKeypointId,
        offsets);
    targetKeypoint = addVectors(
        {
          x: targetKeypointIndices.x * outputStride,
          y: targetKeypointIndices.y * outputStride
        },
        {x: offsetPoint.x, y: offsetPoint.y});
  }
  const targetKeyPointIndices =
      getStridedIndexNearPoint(targetKeypoint, outputStride, height, width);
  const score = scoresBuffer.get(
      targetKeyPointIndices.y, targetKeyPointIndices.x, targetKeypointId);
  return {position: targetKeypoint, part: partNames[targetKeypointId], score};
}
function decodePose(
    root, scores, offsets, outputStride, displacementsFwd, displacementsBwd) {
  const numParts = scores.shape[2];
  const numEdges = parentToChildEdges.length;
  const instanceKeypoints = new Array(numParts);
  const {part: rootPart, score: rootScore} = root;
  const rootPoint = getImageCoords(rootPart, outputStride, offsets);
  instanceKeypoints[rootPart.id] = {
    score: rootScore,
    part: partNames[rootPart.id],
    position: rootPoint
  };
  for (let edge = numEdges - 1; edge >= 0; --edge) {
    const sourceKeypointId = parentToChildEdges[edge];
    const targetKeypointId = childToParentEdges[edge];
    if (instanceKeypoints[sourceKeypointId] &&
        !instanceKeypoints[targetKeypointId]) {
      instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(
          edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores,
          offsets, outputStride, displacementsBwd);
    }
  }
  for (let edge = 0; edge < numEdges; ++edge) {
    const sourceKeypointId = childToParentEdges[edge];
    const targetKeypointId = parentToChildEdges[edge];
    if (instanceKeypoints[sourceKeypointId] &&
        !instanceKeypoints[targetKeypointId]) {
      instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(
          edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores,
          offsets, outputStride, displacementsFwd);
    }
  }
  return instanceKeypoints;
}

function withinNmsRadiusOfCorrespondingPoint(
    poses, squaredNmsRadius, {x, y}, keypointId) {
  return poses.some(({keypoints}) => {
    const correspondingKeypoint = keypoints[keypointId].position;
    return squaredDistance(
               y, x, correspondingKeypoint.y, correspondingKeypoint.x) <=
        squaredNmsRadius;
  });
}
function getInstanceScore(existingPoses, squaredNmsRadius, instanceKeypoints) {
  let notOverlappedKeypointScores =
      instanceKeypoints.reduce((result, {position, score}, keypointId) => {
        if (!withinNmsRadiusOfCorrespondingPoint(
                existingPoses, squaredNmsRadius, position, keypointId)) {
          result += score;
        }
        return result;
      }, 0.0);
  return notOverlappedKeypointScores /= instanceKeypoints.length;
}
const kLocalMaximumRadius = 1;
function decodeMultiplePoses(
    scoresBuffer, offsetsBuffer, displacementsFwdBuffer, displacementsBwdBuffer,
    outputStride, maxPoseDetections, scoreThreshold = 0.5, nmsRadius = 20) {
  const poses = [];
  const queue = buildPartWithScoreQueue(
      scoreThreshold, kLocalMaximumRadius, scoresBuffer);
  const squaredNmsRadius = nmsRadius * nmsRadius;
  while (poses.length < maxPoseDetections && !queue.empty()) {
    const root = queue.dequeue();
    const rootImageCoords =
        getImageCoords(root.part, outputStride, offsetsBuffer);
    if (withinNmsRadiusOfCorrespondingPoint(
            poses, squaredNmsRadius, rootImageCoords, root.part.id)) {
      continue;
    }
    const keypoints = decodePose(
        root, scoresBuffer, offsetsBuffer, outputStride, displacementsFwdBuffer,
        displacementsBwdBuffer);
    const score = getInstanceScore(poses, squaredNmsRadius, keypoints);
    poses.push({keypoints, score});
  }
  return poses;
}

function eitherPointDoesntMeetConfidence(a, b, minConfidence) {
  return (a < minConfidence || b < minConfidence);
}
function getAdjacentKeyPoints(keypoints, minConfidence) {
  return connectedPartIndices.reduce((result, [leftJoint, rightJoint]) => {
    if (eitherPointDoesntMeetConfidence(
            keypoints[leftJoint].score, keypoints[rightJoint].score,
            minConfidence)) {
      return result;
    }
    result.push([keypoints[leftJoint], keypoints[rightJoint]]);
    return result;
  }, []);
}
const {NEGATIVE_INFINITY, POSITIVE_INFINITY} = Number;
function getBoundingBox(keypoints) {
  return keypoints.reduce(({maxX, maxY, minX, minY}, {position: {x, y}}) => {
    return {
      maxX: Math.max(maxX, x),
      maxY: Math.max(maxY, y),
      minX: Math.min(minX, x),
      minY: Math.min(minY, y)
    };
  }, {
    maxX: NEGATIVE_INFINITY,
    maxY: NEGATIVE_INFINITY,
    minX: POSITIVE_INFINITY,
    minY: POSITIVE_INFINITY
  });
}
function getBoundingBoxPoints(keypoints) {
  const {minX, minY, maxX, maxY} = getBoundingBox(keypoints);
  return [
    {x: minX, y: minY}, {x: maxX, y: minY}, {x: maxX, y: maxY},
    {x: minX, y: maxY}
  ];
}
async function toTensorBuffer(tensor, type = 'float32') {
  const tensorData = await tensor.data();
  return tf.buffer(tensor.shape, type, tensorData);
}
async function toTensorBuffers3D(tensors) {
  return Promise.all(tensors.map(tensor => toTensorBuffer(tensor, 'float32')));
}
function scalePose(pose, scaleY, scaleX, offsetY = 0, offsetX = 0) {
  return {
    score: pose.score,
    keypoints: pose.keypoints.map(({score, part, position}) => ({
                                    score,
                                    part,
                                    position: {
                                      x: position.x * scaleX + offsetX,
                                      y: position.y * scaleY + offsetY
                                    }
                                  }))
  };
}
function scalePoses(poses, scaleY, scaleX, offsetY = 0, offsetX = 0) {
  if (scaleX === 1 && scaleY === 1 && offsetY === 0 && offsetX === 0) {
    return poses;
  }
  return poses.map(pose => scalePose(pose, scaleY, scaleX, offsetY, offsetX));
}
function flipPoseHorizontal(pose, imageWidth) {
  return {
    score: pose.score,
    keypoints: pose.keypoints.map(
        ({score, part, position}) => ({
          score,
          part,
          position: {x: imageWidth - 1 - position.x, y: position.y}
        }))
  };
}
function flipPosesHorizontal(poses, imageWidth) {
  if (imageWidth <= 0) {
    return poses;
  }
  return poses.map(pose => flipPoseHorizontal(pose, imageWidth));
}
function getInputTensorDimensions(input) {
  return input instanceof tf.Tensor ? [input.shape[0], input.shape[1]] :
                                      [input.height, input.width];
}
function toInputTensor(input) {
  return input instanceof tf.Tensor ? input : tf.browser.fromPixels(input);
}
function padAndResizeTo(input, [targetH, targetW]) {
  const [height, width] = getInputTensorDimensions(input);
  const targetAspect = targetW / targetH;
  const aspect = width / height;
  let [padT, padB, padL, padR] = [0, 0, 0, 0];
  if (aspect < targetAspect) {
    padT = 0;
    padB = 0;
    padL = Math.round(0.5 * (targetAspect * height - width));
    padR = Math.round(0.5 * (targetAspect * height - width));
  } else {
    padT = Math.round(0.5 * ((1.0 / targetAspect) * width - height));
    padB = Math.round(0.5 * ((1.0 / targetAspect) * width - height));
    padL = 0;
    padR = 0;
  }
  const resized = tf.tidy(() => {
    let imageTensor = toInputTensor(input);
    imageTensor = tf.pad3d(imageTensor, [[padT, padB], [padL, padR], [0, 0]]);
    return imageTensor.resizeBilinear([targetH, targetW]);
  });
  return {resized, padding: {top: padT, left: padL, right: padR, bottom: padB}};
}
function scaleAndFlipPoses(
    poses, [height, width], [inputResolutionHeight, inputResolutionWidth],
    padding, flipHorizontal) {
  const scaleY =
      (height + padding.top + padding.bottom) / (inputResolutionHeight);
  const scaleX =
      (width + padding.left + padding.right) / (inputResolutionWidth);
  const scaledPoses =
      scalePoses(poses, scaleY, scaleX, -padding.top, -padding.left);
  if (flipHorizontal) {
    return flipPosesHorizontal(scaledPoses, width);
  } else {
    return scaledPoses;
  }
}

function mod(a, b) {
  return tf.tidy(() => {
    const floored = a.div(tf.scalar(b, 'int32'));
    return a.sub(floored.mul(tf.scalar(b, 'int32')));
  });
}
function argmax2d(inputs) {
  const [height, width, depth] = inputs.shape;
  return tf.tidy(() => {
    const reshaped = inputs.reshape([height * width, depth]);
    const coords = reshaped.argMax(0);
    const yCoords = coords.div(tf.scalar(width, 'int32')).expandDims(1);
    const xCoords = mod(coords, width).expandDims(1);
    return tf.concat([yCoords, xCoords], 1);
  });
}

function getPointsConfidence(heatmapScores, heatMapCoords) {
  const numKeypoints = heatMapCoords.shape[0];
  const result = new Float32Array(numKeypoints);
  for (let keypoint = 0; keypoint < numKeypoints; keypoint++) {
    const y = heatMapCoords.get(keypoint, 0);
    const x = heatMapCoords.get(keypoint, 1);
    result[keypoint] = heatmapScores.get(y, x, keypoint);
  }
  return result;
}
function getOffsetPoint$1(y, x, keypoint, offsetsBuffer) {
  return {
    y: offsetsBuffer.get(y, x, keypoint),
    x: offsetsBuffer.get(y, x, keypoint + NUM_KEYPOINTS)
  };
}
function getOffsetVectors(heatMapCoordsBuffer, offsetsBuffer) {
  const result = [];
  for (let keypoint = 0; keypoint < NUM_KEYPOINTS; keypoint++) {
    const heatmapY = heatMapCoordsBuffer.get(keypoint, 0).valueOf();
    const heatmapX = heatMapCoordsBuffer.get(keypoint, 1).valueOf();
    const {x, y} =
        getOffsetPoint$1(heatmapY, heatmapX, keypoint, offsetsBuffer);
    result.push(y);
    result.push(x);
  }
  return tf.tensor2d(result, [NUM_KEYPOINTS, 2]);
}
function getOffsetPoints(heatMapCoordsBuffer, outputStride, offsetsBuffer) {
  return tf.tidy(() => {
    const offsetVectors = getOffsetVectors(heatMapCoordsBuffer, offsetsBuffer);
    return heatMapCoordsBuffer.toTensor()
        .mul(tf.scalar(outputStride, 'int32'))
        .toFloat()
        .add(offsetVectors);
  });
}

async function decodeSinglePose(heatmapScores, offsets, outputStride) {
  let totalScore = 0.0;
  const heatmapValues = argmax2d(heatmapScores);
  const [scoresBuffer, offsetsBuffer, heatmapValuesBuffer] = await Promise.all([
    toTensorBuffer(heatmapScores), toTensorBuffer(offsets),
    toTensorBuffer(heatmapValues, 'int32')
  ]);
  const offsetPoints =
      getOffsetPoints(heatmapValuesBuffer, outputStride, offsetsBuffer);
  const offsetPointsBuffer = await toTensorBuffer(offsetPoints);
  const keypointConfidence =
      Array.from(getPointsConfidence(scoresBuffer, heatmapValuesBuffer));
  const keypoints = keypointConfidence.map((score, keypointId) => {
    totalScore += score;
    return {
      position: {
        y: offsetPointsBuffer.get(keypointId, 0),
        x: offsetPointsBuffer.get(keypointId, 1)
      },
      part: partNames[keypointId],
      score
    };
  });
  heatmapValues.dispose();
  offsetPoints.dispose();
  return {keypoints, score: totalScore / keypoints.length};
}

const MOBILENET_BASE_URL =
    'https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/';
const RESNET50_BASE_URL =
    'https://storage.googleapis.com/tfjs-models/savedmodel/posenet/resnet50/';
function resNet50Checkpoint(stride, quantBytes) {
  const graphJson = `model-stride${stride}.json`;
  if (quantBytes == 4) {
    return RESNET50_BASE_URL + `float/` + graphJson;
  } else {
    return RESNET50_BASE_URL + `quant${quantBytes}/` + graphJson;
  }
}
function mobileNetCheckpoint(stride, multiplier, quantBytes) {
  const toStr = {1.0: '100', 0.75: '075', 0.50: '050'};
  const graphJson = `model-stride${stride}.json`;
  if (quantBytes == 4) {
    return MOBILENET_BASE_URL + `float/${toStr[multiplier]}/` + graphJson;
  } else {
    return MOBILENET_BASE_URL + `quant${quantBytes}/${toStr[multiplier]}/` +
        graphJson;
  }
}

function toFloatIfInt$1(input) {
  return tf.tidy(() => {
    if (input.dtype === 'int32') {
      input = input.toFloat();
    }
    const imageNetMean = tf.tensor([-123.15, -115.90, -103.06]);
    return input.add(imageNetMean);
  });
}
class ResNet {
  constructor(model, outputStride) {
    this.model = model;
    const inputShape = this.model.inputs[0].shape;
    tf.util.assert(
        (inputShape[1] === -1) && (inputShape[2] === -1),
        () => `Input shape [${inputShape[1]}, ${inputShape[2]}] ` +
            `must both be equal to or -1`);
    this.outputStride = outputStride;
  }
  predict(input) {
    // return tf.tidy(() => {
    const asFloat = toFloatIfInt$1(input);
    const asBatch = asFloat.expandDims(0);
    const [displacementFwd4d, displacementBwd4d, offsets4d, heatmaps4d] =
        this.model.predict(asBatch);
    const heatmaps = heatmaps4d.squeeze();
    const heatmapScores = heatmaps.sigmoid();
    const offsets = offsets4d.squeeze();
    const displacementFwd = displacementFwd4d.squeeze();
    const displacementBwd = displacementBwd4d.squeeze();
    return {
      heatmapScores,
      offsets: offsets,
      displacementFwd: displacementFwd,
      displacementBwd: displacementBwd
    };
    // });
  }
  dispose() {
    this.model.dispose();
  }
}

const RESNET_CONFIG = {
  architecture: 'ResNet50',
  outputStride: 32,
  multiplier: 1.0,
  inputResolution: 257,
};
const VALID_ARCHITECTURE = ['MobileNetV1', 'ResNet50'];
const VALID_STRIDE = {
  'MobileNetV1': [8, 16, 32],
  'ResNet50': [32, 16]
};
const VALID_INPUT_RESOLUTION =
    [161, 193, 257, 289, 321, 353, 385, 417, 449, 481, 513, 801];
const VALID_MULTIPLIER = {
  'MobileNetV1': [0.50, 0.75, 1.0],
  'ResNet50': [1.0]
};
const VALID_QUANT_BYTES = [1, 2, 4];
function validateModelConfig(config) {
  config = config || RESNET_CONFIG;
  if (config.architecture == null) {
    config.architecture = 'MobileNetV1';
  }
  if (VALID_ARCHITECTURE.indexOf(config.architecture) < 0) {
    throw new Error(
        `Invalid architecture ${config.architecture}. ` +
        `Should be one of ${VALID_ARCHITECTURE}`);
  }
  if (config.inputResolution == null) {
    config.inputResolution = 257;
  }
  if (VALID_INPUT_RESOLUTION.indexOf(config.inputResolution) < 0) {
    throw new Error(
        `Invalid inputResolution ${config.inputResolution}. ` +
        `Should be one of ${VALID_INPUT_RESOLUTION}`);
  }
  if (config.outputStride == null) {
    config.outputStride = 16;
  }
  if (VALID_STRIDE[config.architecture].indexOf(config.outputStride) < 0) {
    throw new Error(
        `Invalid outputStride ${config.outputStride}. ` +
        `Should be one of ${VALID_STRIDE[config.architecture]} ` +
        `for architecutre ${config.architecture}.`);
  }
  if (config.multiplier == null) {
    config.multiplier = 1.0;
  }
  if (VALID_MULTIPLIER[config.architecture].indexOf(config.multiplier) < 0) {
    throw new Error(
        `Invalid multiplier ${config.multiplier}. ` +
        `Should be one of ${VALID_MULTIPLIER[config.architecture]} ` +
        `for architecutre ${config.architecture}.`);
  }
  if (config.quantBytes == null) {
    config.quantBytes = 4;
  }
  if (VALID_QUANT_BYTES.indexOf(config.quantBytes) < 0) {
    throw new Error(
        `Invalid quantBytes ${config.quantBytes}. ` +
        `Should be one of ${VALID_QUANT_BYTES} ` +
        `for architecutre ${config.architecture}.`);
  }
  return config;
}
const SINGLE_PERSON_INFERENCE_CONFIG = {
  flipHorizontal: false
};
const MULTI_PERSON_INFERENCE_CONFIG = {
  flipHorizontal: false,
  maxDetections: 5,
  scoreThreshold: 0.5,
  nmsRadius: 20
};
function validateMultiPersonInputConfig(config) {
  const {maxDetections, scoreThreshold, nmsRadius} = config;
  if (maxDetections <= 0) {
    throw new Error(
        `Invalid maxDetections ${maxDetections}. ` +
        `Should be > 0`);
  }
  if (scoreThreshold < 0.0 || scoreThreshold > 1.0) {
    throw new Error(
        `Invalid scoreThreshold ${scoreThreshold}. ` +
        `Should be in range [0.0, 1.0]`);
  }
  if (nmsRadius <= 0) {
    throw new Error(`Invalid nmsRadius ${nmsRadius}.`);
  }
}
class PoseNet {
  constructor(net, inputResolution) {
    this.baseModel = net;
    this.inputResolution = inputResolution;
  }
  async estimateMultiplePoses(input, config = MULTI_PERSON_INFERENCE_CONFIG) {
    const configWithDefaults =
        Object.assign({}, MULTI_PERSON_INFERENCE_CONFIG, config);
    validateMultiPersonInputConfig(config);
    const outputStride = this.baseModel.outputStride;
    const inputResolution = this.inputResolution;
    assertValidOutputStride(outputStride);
    assertValidResolution(this.inputResolution, outputStride);
    const [height, width] = getInputTensorDimensions(input);
    const {resized, padding} =
        padAndResizeTo(input, [inputResolution, inputResolution]);
    const {heatmapScores, offsets, displacementFwd, displacementBwd} =
        this.baseModel.predict(resized);
    const [scoresBuffer, offsetsBuffer, displacementsFwdBuffer, displacementsBwdBuffer] =
        await toTensorBuffers3D(
            [heatmapScores, offsets, displacementFwd, displacementBwd]);
    const poses = await decodeMultiplePoses(
        scoresBuffer, offsetsBuffer, displacementsFwdBuffer,
        displacementsBwdBuffer, outputStride, configWithDefaults.maxDetections,
        configWithDefaults.scoreThreshold, configWithDefaults.nmsRadius);
    const resultPoses = scaleAndFlipPoses(
        poses, [height, width], [inputResolution, inputResolution], padding,
        configWithDefaults.flipHorizontal);
    heatmapScores.dispose();
    offsets.dispose();
    displacementFwd.dispose();
    displacementBwd.dispose();
    resized.dispose();
    return resultPoses;
  }
  async estimateSinglePose(input, config = SINGLE_PERSON_INFERENCE_CONFIG) {
    const configWithDefaults =
        Object.assign({}, SINGLE_PERSON_INFERENCE_CONFIG, config);
    const outputStride = this.baseModel.outputStride;
    const inputResolution = this.inputResolution;
    assertValidOutputStride(outputStride);
    assertValidResolution(inputResolution, outputStride);
    const [height, width] = getInputTensorDimensions(input);
    const {resized, padding} =
        padAndResizeTo(input, [inputResolution, inputResolution]);
    const {heatmapScores, offsets, displacementFwd, displacementBwd} =
        this.baseModel.predict(resized);
    const pose = await decodeSinglePose(heatmapScores, offsets, outputStride);
    const poses = [pose];
    const resultPoses = scaleAndFlipPoses(
        poses, [height, width], [inputResolution, inputResolution], padding,
        configWithDefaults.flipHorizontal);
    heatmapScores.dispose();
    offsets.dispose();
    displacementFwd.dispose();
    displacementBwd.dispose();
    resized.dispose();
    return resultPoses[0];
  }
  async estimatePoses(input, config) {
    if (config.decodingMethod == 'single-person') {
      const pose = await this.estimateSinglePose(input, config);
      return [pose];
    } else {
      return this.estimateMultiplePoses(input, config);
    }
  }
  dispose() {
    this.baseModel.dispose();
  }
}
async function loadMobileNet(config) {
  const outputStride = config.outputStride;
  const quantBytes = config.quantBytes;
  const multiplier = config.multiplier;
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this
        model.`);
  }
  const url = mobileNetCheckpoint(outputStride, multiplier, quantBytes);
  const graphModel = await tfconv.loadGraphModel(config.modelUrl || url);
  const mobilenet = new MobileNet(graphModel, outputStride);
  return new PoseNet(mobilenet, config.inputResolution);
}
async function loadResNet(config) {
  const outputStride = config.outputStride;
  const quantBytes = config.quantBytes;
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this
        model.`);
  }
  const url = resNet50Checkpoint(outputStride, quantBytes);
  const graphModel = await tfconv.loadGraphModel(config.modelUrl || url);
  const resnet = new ResNet(graphModel, outputStride);
  return new PoseNet(resnet, config.inputResolution);
}
async function load(config = RESNET_CONFIG) {
  config = validateModelConfig(config);
  if (config.architecture === 'ResNet50') {
    return loadResNet(config);
  } else if (config.architecture === 'MobileNetV1') {
    return loadMobileNet(config);
  } else {
    return null;
  }
}

exports.MobileNet = MobileNet;
exports.PoseNet = PoseNet;
exports.VALID_INPUT_RESOLUTION = VALID_INPUT_RESOLUTION;
exports.decodeMultiplePoses = decodeMultiplePoses;
exports.decodeSinglePose = decodeSinglePose;
exports.getAdjacentKeyPoints = getAdjacentKeyPoints;
exports.getBoundingBox = getBoundingBox;
exports.getBoundingBoxPoints = getBoundingBoxPoints;
exports.load = load;
exports.partChannels = partChannels;
exports.partIds = partIds;
exports.partNames = partNames;
exports.poseChain = poseChain;
exports.scalePose = scalePose;

Object.defineProperty(exports, '__esModule', {value: true});
}));
