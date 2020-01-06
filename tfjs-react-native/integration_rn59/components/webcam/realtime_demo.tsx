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
import { ActivityIndicator, StyleSheet, View, PixelRatio, LayoutChangeEvent } from 'react-native';
import Svg, { Circle, Rect, G, Line} from 'react-native-svg';

import * as Permissions from 'expo-permissions';
import { Camera } from 'expo-camera';
import { GLView, ExpoWebGLRenderingContext } from 'expo-gl';

import * as tf from '@tensorflow/tfjs';
import * as blazeface from '@tensorflow-models/blazeface';
import * as posenet from '@tensorflow-models/posenet';
import {fromTexture, renderToGLView} from '@tensorflow/tfjs-react-native';

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
  // tslint:disable-next-line: no-any
  faceDetector?: any;
  faces?: blazeface.NormalizedFace[];
  xScale?: number;
  yScale?: number;
  drawWidth?: number;
  drawHeight?: number;
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
      cameraType: Camera.Constants.Type.back,
    };
  }

  async loadPosenetModel() {
    const model =  await posenet.load({
      architecture: 'MobileNetV1',
      outputStride: 16,
      inputResolution: { width: inputTensorWidth, height: inputTensorHeight },
      multiplier: 0.75,
      quantBytes: 2
    });
    return model;
  }

  async loadBlazefaceModel() {
    const model =  await blazeface.load();
    return model;
  }

  async componentDidMount() {
    const { status } = await Permissions.askAsync(Permissions.CAMERA);

    const [blazefaceModel, posenetModel] =
      await Promise.all([this.loadBlazefaceModel(), this.loadPosenetModel()]);

    this.setState({
      hasCameraPermission: status === 'granted',
      isLoading: false,
      faceDetector: blazefaceModel,
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
    sourceDims:{width: number, height: number, depth:number}, model: string) {
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
    previewLoop();

    const poseLoop = async () => {
      const inputTensor = fromTexture(gl, inputTexture, sourceDims, targetDims);
      const pose = await this.state.posenetModel!.estimateSinglePose(
        inputTensor, { flipHorizontal: true });

      this.setState({pose});
      tf.dispose(inputTensor);

      requestAnimationFrame(poseLoop);
    };

    const faceDetectorLoop = async () => {
      const inputTensor = fromTexture(gl, inputTexture, sourceDims, targetDims);
      const faces = await this.state.faceDetector.estimateFaces(
        inputTensor, false);

      this.setState({faces});
      tf.dispose(inputTensor);

      requestAnimationFrame(faceDetectorLoop);
    };

    setTimeout(() => {
      if(model === 'posenet') {
        poseLoop();
      } else if (model=== 'blazeface') {
        faceDetectorLoop();
      }
    }, 100);
  }

 async onContextCreate(gl: ExpoWebGLRenderingContext) {
    const targetTexture = await this.createCameraTexture();
    const sourceDims = {
      height: 600,
      width: 800,
      depth: 4,
    };

    this.setState({
      xScale: sourceDims.width / inputTensorWidth,
      yScale: sourceDims.height / inputTensorHeight,
      drawWidth: sourceDims.width,
      drawHeight: sourceDims.height,
    });

    this.startRenderLoop(gl, targetTexture, sourceDims, 'posenet');
  }

  renderPose() {
    const MIN_KEYPOINT_SCORE = 0.2;
    const {pose, xScale, yScale, drawHeight, drawWidth} = this.state;
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

      return <Svg height='100%' width='100%'
        viewBox={`0 0 ${drawWidth! / xScale!} ${drawHeight! / yScale!}`}>
          {skeleton}
          {keypoints}
        </Svg>;
    } else {
      return null;
    }
  }

  renderFaces() {
    const {faces, xScale, yScale, drawHeight, drawWidth} = this.state;
    if(faces != null) {
      const faceBoxes = faces.map((f, fIndex) => {
        const topLeft = f.topLeft as number[];
        const bottomRight = f.bottomRight as number[];

        const landmarks = (f.landmarks as number[][]).map((l, lIndex) => {
          return <Circle
            key={`landmark_${fIndex}_${lIndex}`}
            cx={l[0]}
            cy={l[1]}
            r='2'
            strokeWidth='0'
            fill='blue'
            />;
        });

        return <G key={`facebox_${fIndex}`}>
          <Rect
            x={topLeft[0]}
            y={topLeft[1]}
            fill={'red'}
            fillOpacity={0.2}
            width={(bottomRight[0] - topLeft[0])}
            height={(bottomRight[1] - topLeft[1])}
          />
          {landmarks}
        </G>;
      });

      return <Svg height='100%' width='100%'
        viewBox={`0 0 ${drawWidth! / xScale!} ${drawHeight! / yScale!}`}>
          {faceBoxes}
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
          {/* {this.renderFaces()} */}
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
