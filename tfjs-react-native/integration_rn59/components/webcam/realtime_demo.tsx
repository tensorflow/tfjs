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
import Svg, { Circle, Rect, G} from 'react-native-svg';

import * as Permissions from 'expo-permissions';
import { Camera } from 'expo-camera';
import { GLView, ExpoWebGLRenderingContext } from 'expo-gl';

import * as tf from '@tensorflow/tfjs';
import * as blazeface from '@tensorflow-models/blazeface';
import {fromTexture, renderToGLView, decodeJpeg, fetch} from '@tensorflow/tfjs-react-native';

interface ScreenProps {
  returnToMain: () => void;
}

interface ScreenState {
  hasCameraPermission?: boolean;
  // tslint:disable-next-line: no-any
  cameraType: any;
  isLoading: boolean;
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

  async loadBlazefaceModel() {
    const model =  await blazeface.load();
    return model;
  }

  async componentDidMount() {
    const { status } = await Permissions.askAsync(Permissions.CAMERA);
    const blazefaceModel = await this.loadBlazefaceModel();
    this.setState({
      hasCameraPermission: status === 'granted',
      isLoading: false,
      faceDetector: blazefaceModel,
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

    const faceDetectorLoop = async () => {
      let start;
      let end;

      start = Date.now();
      const inputTensor = fromTexture(gl, inputTexture, sourceDims, targetDims);
      end = Date.now();
      // console.log('fromTexture time', end - start);

      start = Date.now();
      const faces = await this.state.faceDetector.estimateFaces(
        inputTensor, false);
      end = Date.now();
      // console.log('faces', faces)
      // console.log('facedetector time', end - start);
      this.setState({faces});

      tf.dispose(inputTensor);
      requestAnimationFrame(faceDetectorLoop);
    };

    setTimeout(() => {
      faceDetectorLoop();
    }, 500);
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

    this.setState({
      xScale: sourceDims.width / inputTensorWidth,
      yScale: sourceDims.height / inputTensorHeight,
      drawWidth: sourceDims.width,
      drawHeight: sourceDims.height,
    });

    // Test image
    // const imageTensor = await this.getImageTensor();
    // const targetTexture = await toTexture(gl, imageTensor);
    // const sourceDims = {
    //   height: 513,
    //   width: 513,
    //   depth: 4,
    // };

    this.startRenderLoop(gl, targetTexture, sourceDims);
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
