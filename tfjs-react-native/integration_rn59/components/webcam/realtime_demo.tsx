/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import React from 'react';
import { ActivityIndicator, StyleSheet, View, Text, Dimensions ,PixelRatio, LayoutChangeEvent } from 'react-native';
import Svg, { Circle, Line} from 'react-native-svg';

import * as Permissions from 'expo-permissions';
import { Camera } from 'expo-camera';
import { GLView, ExpoWebGLRenderingContext } from 'expo-gl';

import * as tf from '@tensorflow/tfjs';
import * as posenet from '@tensorflow-models/posenet';
import {fromTexture, toTexture, renderToGLView, decodeJpeg, fetch} from '@tensorflow/tfjs-react-native';

interface ScreenProps {
  returnToMain: () => void;
}

interface ScreenState {
  hasCameraPermission?: boolean;
  // tslint:disable-next-line: no-any
  cameraType: any;
  isLoading: boolean;
  posenetModel?: posenet.PoseNet;
  pose?: posenet.Pose;
}

let cameraPreviewWidth: number;
let cameraPreviewHeight: number;

const inputTensorWidth = 152;
const inputTensorHeight = 200;

export class RealtimeDemo extends React.Component<ScreenProps,ScreenState> {
  private camera?: Camera|null;
  private glView?: GLView;

  constructor(props: ScreenProps) {
    super(props);
    this.state = {
      isLoading: true,
      cameraType: Camera.Constants.Type.front,
    };
  }

  async loadPosenetModel() {
    const model =  await posenet.load({
      architecture: 'MobileNetV1',
      outputStride: 16,
      inputResolution: { width: inputTensorWidth, height: inputTensorHeight },
      multiplier: 0.75
    });
    return model;
  }

  async componentDidMount() {
    const { status } = await Permissions.askAsync(Permissions.CAMERA);
    const posenetModel = await this.loadPosenetModel();
    this.setState({
      hasCameraPermission: status === 'granted',
      isLoading: false,
      posenetModel,
    });
  }

  async createCameraTexture(): Promise<WebGLTexture> {
    const { status } = await Permissions.askAsync(Permissions.CAMERA);
    if (status !== 'granted') {
      throw new Error('Denied camera permissions!');
    }
    //@ts-ignore
    return this.glView!.createCameraTextureAsync(this.camera!);
  }

  onCameraLayout(event: LayoutChangeEvent) {
    const {x, y, width, height} = event.nativeEvent.layout;
    cameraPreviewHeight = height;
    cameraPreviewWidth = width;
    console.log('onCameraLayout', x, y, width, height);
  }

  onGLViewLayout(event: LayoutChangeEvent) {
    const {x, y, width, height} = event.nativeEvent.layout;
    console.log('onGLViewLayout', x, y, width, height);
  }

  startRenderLoop(gl: ExpoWebGLRenderingContext, inputTexture: WebGLTexture,
    sourceDims:{width: number, height: number, depth:number}) {
    const pixelRatio = PixelRatio.get();
    const width = Math.floor(cameraPreviewWidth * pixelRatio);
    const height = Math.floor(cameraPreviewHeight * pixelRatio);

    const targetDims = {
      height: inputTensorHeight,
      width: inputTensorWidth,
      depth: 3,
    };

    const previewLoop = async () => {
      renderToGLView(gl, inputTexture, { width, height });
      gl.endFrameEXP();
      requestAnimationFrame(previewLoop);
    };

    setTimeout(() => {
      previewLoop();
    }, 100);

    const poseLoop = async () => {
      const inputTensor = fromTexture(gl, inputTexture, sourceDims, targetDims);

      const pose = await this.state.posenetModel!.estimateSinglePose(
        inputTensor, { flipHorizontal: false });
      this.setState({pose});

      requestAnimationFrame(poseLoop);
    };

    setTimeout(() => {
      poseLoop();
    }, 1000);
  }

  async getImageTensor() {
    const imageUrl = 'https://storage.googleapis.com/tfjs-models/assets/posenet/backpackman.jpg';
    const response = await fetch(imageUrl, {}, { isBinary: true });
    const rawImageData = await response.arrayBuffer();

    const imageTensor = decodeJpeg(new Uint8Array(rawImageData), 3);
    const imageTensorWAlpha = imageTensor.pad([[0,0],[0,0],[0,1]], 255);

    tf.dispose(imageTensor);
    return imageTensorWAlpha;
  }

 async onContextCreate(gl: ExpoWebGLRenderingContext) {
    const targetTexture = await this.createCameraTexture();
    const sourceDims = {
      height: 600,
      width: 800,
      depth: 4,
    };

    // const imageTensor = await this.getImageTensor();
    // const targetTexture = await toTexture(gl, imageTensor);
    // const sourceDims = {
    //   height: 513,
    //   width: 513,
    //   depth: 4,
    // };

    this.startRenderLoop(gl, targetTexture, sourceDims);
  }

  renderPose() {
    const MIN_KEYPOINT_SCORE = 0.2;
    const {pose} = this.state;
    if(pose != null) {
      const keypoints = pose.keypoints
        .filter(k => k.score > MIN_KEYPOINT_SCORE)
        .map((k,i) => {
          return <Circle
            key={`skeletonkp_${i}`}
            cx={k.position.x}
            cy={k.position.y}
            r='5'
            strokeWidth='0'
            fill='blue'
          />;
        });

      const adjacentKeypoints =
        posenet.getAdjacentKeyPoints(pose.keypoints, MIN_KEYPOINT_SCORE);

      const skeleton = adjacentKeypoints.map(([from, to], i) => {
        return <Line
          key={`skeletonls_${i}`}
          x1={from.position.x}
          y1={from.position.y}
          x2={to.position.x}
          y2={to.position.y}
          stroke='magenta'
          strokeWidth='1'
        />;
      });

      return <Svg height='100%' width='100%'>
          {skeleton}
          {keypoints}
        </Svg>;
    } else {
      return null;
    }
  }

  render() {
    const {isLoading} = this.state;
    const camView = <View style={styles.cameraContainer}>
        <Camera
          style={styles.camera}
          type={this.state.cameraType}
          zoom={0}
          ref={ref => this.camera = ref!}
          onLayout={this.onCameraLayout.bind(this)}
        />
        <GLView
          style={styles.camera}
          onLayout={this.onGLViewLayout.bind(this)}
          onContextCreate={this.onContextCreate.bind(this)}
          ref={ref => this.glView = ref!}
        />
        <View style={styles.camera}>
          {this.renderPose()}
        </View>

        </View>;
    return (
      <View style={{width:'100%'}}>
        {isLoading ? <View style={[styles.loadingIndicator]}>
          <ActivityIndicator size='large' color='#FF0266' />
        </View> : camView}
      </View>
    );
  }
}

const styles = StyleSheet.create({
  loadingIndicator: {
    position: 'absolute',
    top: 20,
    right: 20,
    zIndex: 200,
  },
  cameraContainer: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
    height: '100%',
    backgroundColor: '#fff',
  },
  camera : {
    display: 'flex',
    width: '50%',
    height: '30%',
    // backgroundColor: '#f0F',
    zIndex: 1,
    borderWidth: 2,
    borderRadius: 2,
    borderColor: '#f0f',
  }
});
