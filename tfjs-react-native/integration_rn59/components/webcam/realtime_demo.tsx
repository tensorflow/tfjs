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
import { ActivityIndicator, StyleSheet, View, PixelRatio, LayoutChangeEvent, Dimensions, Platform } from 'react-native';
import Svg, { Circle, Rect, G, Line} from 'react-native-svg';

import * as Permissions from 'expo-permissions';
import { Camera } from 'expo-camera';
import { GLView, ExpoWebGLRenderingContext } from 'expo-gl';

import * as tf from '@tensorflow/tfjs';
import * as blazeface from '@tensorflow-models/blazeface';
import * as posenet from '@tensorflow-models/posenet';
import {fromTexture, renderToGLView, detectGLCapabilities} from '@tensorflow/tfjs-react-native';

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

    const deviceDims =  {
      width: Dimensions.get('window').width,
      height: Dimensions.get('window').height
    };
    console.log('device dims', deviceDims);

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
    const width = PixelRatio.getPixelSizeForLayoutSize(cameraPreviewWidth);
    const height = PixelRatio.getPixelSizeForLayoutSize(cameraPreviewHeight);

    const targetDepth = 3;
    const targetDims = {
      height: inputTensorHeight,
      width: inputTensorWidth,
      depth: targetDepth,
    };

    const flipHorizontal = Platform.OS === 'ios' ? false : true;

    const poseLoop = async (inputTensor: tf.Tensor3D) => {
      const pose = await this.state.posenetModel!.estimateSinglePose(
        inputTensor, { flipHorizontal });

      this.setState({pose});
      tf.dispose([inputTensor, inputTensor]);
    };

    const faceDetectorLoop = async (inputTensor: tf.Tensor3D) => {
      const faces = await this.state.faceDetector.estimateFaces(
        inputTensor, false);

      this.setState({faces});
      tf.dispose(inputTensor);
    };



    const previewLoop = async () => {
      renderToGLView(gl, inputTexture, {width, height}, flipHorizontal);

      const inputTensor = fromTexture(gl, inputTexture, sourceDims, targetDims);
      if(model === 'posenet') {
        await poseLoop(inputTensor);
      } else if (model=== 'blazeface') {
        await faceDetectorLoop(inputTensor);
      }

      // Note that call gl.enfFrameExp() here synchronizes the display of the
      // video texture and the react rendered markers. To desync them
      // (i.e. draw the video as fast as possible, move this line to below the
      // renderToGLView call)
      gl.endFrameEXP();
      requestAnimationFrame(previewLoop);
    };
    setTimeout(() => {
      previewLoop();
    }, 200);
  }

 async onContextCreate(gl: ExpoWebGLRenderingContext) {


    let textureDims;
    if (Platform.OS === 'ios') {
      textureDims = {
        height: 1920,
        width: 1080,
        depth: 4,
      };
    } else {
      textureDims = {
        height: 1200,
        width: 1600,
        depth: 4,
      };
    }
    const targetTexture = await this.createCameraTexture();
    await detectGLCapabilities(gl);
    this.startRenderLoop(gl, targetTexture, textureDims, 'posenet');
  }

  renderPose() {
    const MIN_KEYPOINT_SCORE = 0.2;
    const {pose} = this.state;
    if(pose != null) {
      // console.log('Pose.keypoints', pose.keypoints[0]);
      const keypoints = pose.keypoints
        .filter(k => k.score > MIN_KEYPOINT_SCORE)
        .map((k,i) => {
          return <Circle
            key={`skeletonkp_${i}`}
            cx={k.position.x}
            cy={k.position.y}
            r='2'
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
        viewBox={`0 0 ${inputTensorWidth} ${inputTensorHeight}`}>
          {skeleton}
          {keypoints}
        </Svg>;
    } else {
      return null;
    }
  }

  renderFaces() {
    const {faces} = this.state;
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

      const flipHorizontal = Platform.OS === 'ios' ? 1 : -1;
      return <Svg height='100%' width='100%'
        viewBox={`0 0 ${inputTensorWidth} ${inputTensorHeight}`}
        scaleX={flipHorizontal}>
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
          {this.renderFaces()}
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
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
    height: '100%',
    backgroundColor: '#fff',
  },
  camera : {
    position:'absolute',
    left: 50,
    top: 100,
    width: 600/2,
    height: 800/2,
    zIndex: 1,
    borderWidth: 1,
    borderColor: 'black',
    borderRadius: 0,
  }
});
