/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import { ActivityIndicator, StyleSheet, View, Image, Text, TouchableHighlight } from 'react-native';
import * as Permissions from 'expo-permissions';
import { Camera } from 'expo-camera';
import {StyleTranfer} from './style_transfer';
import {base64ImageToTensor, tensorToImageUrl, resizeImage, toDataUri} from './image_utils';
import * as tf from '@tensorflow/tfjs';
import { GLView, ExpoWebGLRenderingContext } from 'expo-gl';
import {fromCamera} from '@tensorflow/tfjs-react-native';
import {Platform} from 'react-native';

export function getWebGLErrorMessage(
    gl: WebGLRenderingContext, status: number): string {
  switch (status) {
    //@ts-ignore
    case gl.NO_ERROR:
      return 'NO_ERROR';
      //@ts-ignore
    case gl.INVALID_ENUM:
      return 'INVALID_ENUM';
      //@ts-ignore
    case gl.INVALID_VALUE:
      return 'INVALID_VALUE';
      //@ts-ignore
    case gl.INVALID_OPERATION:
      return 'INVALID_OPERATION';
      //@ts-ignore
    case gl.INVALID_FRAMEBUFFER_OPERATION:
      return 'INVALID_FRAMEBUFFER_OPERATION';
      //@ts-ignore
    case gl.OUT_OF_MEMORY:
      return 'OUT_OF_MEMORY';
      //@ts-ignore
    case gl.CONTEXT_LOST_WEBGL:
      return 'CONTEXT_LOST_WEBGL';
    default:
      return `Unknown error code ${status}`;
  }
}

const vertShaderSource = `#version 300 es
precision highp float;
in vec2 position;
out vec2 uv;
void main() {
  uv = position;
  gl_Position = vec4(1.0 - 2.0 * position, 0, 1);
}`;

const fragShaderSource = `#version 300 es
precision highp float;
uniform sampler2D cameraTexture;
in vec2 uv;
out vec4 fragColor;
void main() {
  // fragColor = vec4(1.0 - texture(cameraTexture, uv).rgb, 1.0);
  fragColor = texture(cameraTexture, uv);
}`;

interface ScreenProps {
  returnToMain: () => void;
}

interface ScreenState {
  mode: 'results' | 'newStyleImage' | 'newContentImage';
  resultImage?: string;
  styleImage?: string;
  contentImage?: string;
  hasCameraPermission?: boolean;
  // tslint:disable-next-line: no-any
  cameraType: any;
  isLoading: boolean;
}

export class WebcamDemo extends React.Component<ScreenProps,ScreenState> {
  private camera?: Camera|null;
  private styler: StyleTranfer;
  private glView?: GLView;
  private texture?: WebGLTexture;
  private _rafID?: number;

  constructor(props: ScreenProps) {
    super(props);
    this.state = {
      mode: 'results',
      cameraType: Camera.Constants.Type.back,
      isLoading: true,
    };
    this.styler = new StyleTranfer();
  }

  async componentDidMount() {
    // await this.styler.init();
    const { status } = await Permissions.askAsync(Permissions.CAMERA);
    // this.camTexture = await GLView.createCameraTextureAsync();

    this.setState({
      hasCameraPermission: status === 'granted',
      isLoading: false
    });
  }

  showResults() {
    this.setState({ mode: 'results' });
  }

  takeStyleImage() {
    this.setState({ mode: 'newStyleImage' });
  }

  takeContentImage() {
    this.setState({ mode: 'newContentImage' });
  }

  flipCamera() {
    const newState = this.state.cameraType === Camera.Constants.Type.back
          ? Camera.Constants.Type.front
          : Camera.Constants.Type.back;
    this.setState({
      cameraType: newState,
    });
  }

  renderStyleImagePreview() {
    const {styleImage} = this.state;
    if(styleImage == null) {
      return (
        <View>
          <Text style={styles.instructionText}>Style</Text>
          <Text style={{fontSize: 48, paddingLeft: 0}}>💅🏽</Text>
        </View>
      );
    } else {
      return (
        <View>
          <Image
            style={styles.imagePreview}
            source={{uri: toDataUri(styleImage)}} />
            <Text style={styles.centeredText}>Style</Text>
        </View>
      );
    }
  }

  renderContentImagePreview() {
    const {contentImage} = this.state;
    if(contentImage == null) {
      return (
        <View>
          <Text style={styles.instructionText}>Stuff</Text>
          <Text style={{fontSize: 48, paddingLeft: 0}}>🖼️</Text>
        </View>
      );
    } else {
      return (
        <View>
          <Image
            style={styles.imagePreview}
            source={{uri: toDataUri(contentImage)}} />
            <Text style={styles.centeredText}>Stuff</Text>
        </View>
      );
    }
  }

  async stylize(contentImage: string, styleImage: string):
    Promise<string> {
    const contentTensor = await base64ImageToTensor(contentImage);
    const styleTensor = await base64ImageToTensor(styleImage);
    const stylizedResult = this.styler.stylize(
      styleTensor, contentTensor);
    const stylizedImage = await tensorToImageUrl(stylizedResult);
    tf.dispose([contentTensor, styleTensor, stylizedResult]);
    return stylizedImage;
  }

  async handleCameraCapture() {
    const {mode} = this.state;
    let {styleImage, contentImage, resultImage} = this.state;
    this.setState({
      isLoading: true,
    });
    let image = await this.camera!.takePictureAsync({
      skipProcessing: true,
    });
    image = await resizeImage(image.uri, 240);

    if(mode === 'newStyleImage' && image.base64 != null) {
      styleImage = image.base64;
      if(contentImage == null) {
        this.setState({
          styleImage,
          mode: 'results',
          isLoading: false,
        });
      } else {
        resultImage = await this.stylize(contentImage, styleImage),
        this.setState({
          styleImage,
          contentImage,
          resultImage,
          mode: 'results',
          isLoading: false,
        });
      }
    } else if (mode === 'newContentImage' && image.base64 != null) {
      contentImage = image.base64;
      if(styleImage == null) {
        this.setState({
          contentImage,
          mode: 'results',
          isLoading: false,
        });
      } else {
        resultImage = await this.stylize(contentImage, styleImage);
        this.setState({
          contentImage,
          styleImage,
          resultImage,
          mode: 'results',
          isLoading: false,
        });
      }
    }
  }

  async createCameraTexture(): Promise<WebGLTexture> {
    const { status } = await Permissions.askAsync(Permissions.CAMERA);
    if (status !== 'granted') {
      throw new Error('Denied camera permissions!');
    }
    //@ts-ignore
    return this.glView!.createCameraTextureAsync(this.camera!);
  }

  async getCameraTextureFromHiddenContext() : Promise<WebGLTexture> {
    // Create texture asynchronously on hiden context
    const createCamTex = this.glView!.createCameraTextureAsync;
    console.log('createCamTex', createCamTex);
    //@ts-ignore
    const fakeExGlView = {exglCtxId: global.glContext.__exglCtxId};
    const boundCamTex = createCamTex.bind(fakeExGlView);
    //@ts-ignore
    const cameraTexture = await boundCamTex(this.camera);
    //@ts-ignore
    const gl = global.glContext as ExpoWebGLRenderingContext;

    return cameraTexture;
  }

  async onContextCreate(gl: ExpoWebGLRenderingContext) {
    //@ts-ignore
    console.log('on context create', gl.sentinel);

    //@ts-ignore
    const getExt = gl.getExtension.bind(gl);
    const shimGetExt = (name: string) => {
      if (name === 'EXT_color_buffer_float') {
        if (Platform.OS === 'ios') {
          // iOS does not support EXT_color_buffer_float
          return null;
        } else {
          return {};
        }
      }

      if (name === 'EXT_color_buffer_half_float') {
        return {};
      }
      return getExt(name);
    };

    const shimFenceSync = () => {
      return {};
    };
    const shimClientWaitSync = () => gl.CONDITION_SATISFIED;

    // @ts-ignore
    gl.getExtension = shimGetExt.bind(gl);
    // @ts-ignore
    gl.fenceSync = shimFenceSync.bind(gl);
    gl.clientWaitSync = shimClientWaitSync.bind(gl);

    tf.webgl.setWebGLContext(2, gl);
    //@ts-ignore
    tf.backend().gpgpu.gl = gl;
    //@ts-ignore
    tf.backend().gpgpu.init();

    // gl.disable(gl.DEPTH_TEST);
    // gl.disable(gl.STENCIL_TEST);
    // gl.disable(gl.BLEND);
    // gl.disable(gl.DITHER);
    // gl.disable(gl.POLYGON_OFFSET_FILL);
    // gl.disable(gl.SAMPLE_COVERAGE);
    // gl.enable(gl.SCISSOR_TEST);
    // gl.enable(gl.CULL_FACE);
    // gl.cullFace(gl.BACK);

    //@ts-ignore
    // console.log('GLOBAL GLCONTEXT', global.glContext.__exglCtxId);
    //@ts-ignore
    this.texture = await this.createCameraTexture();
    const cameraTexture = this.texture;
    console.log('cameratexture', cameraTexture);

    // gl.activeTexture(gl.TEXTURE0 + 10);
    // let error = gl.getError();
    // console.log('onContextCreate activeTexture', error);
    // if (error !== gl.NO_ERROR) {
    //   throw new Error('WebGL Error: ' + getWebGLErrorMessage(gl, error));
    // }
    // gl.bindTexture(gl.TEXTURE_2D, cameraTexture);
    // error = gl.getError();
    // console.log('onContextCreate bindTexture', error);
    // if (error !== gl.NO_ERROR) {
    //   throw new Error('WebGL Error: ' + getWebGLErrorMessage(gl, error));
    // }

    // setInterval(() => {
    //   console.log('tf.test', tf.log(Array(1000).fill(549)).dataSync()[2]);
    // }, 2000);


    //@ts-ignore
    // const gl = global.glContext as ExpoWebGLRenderingContext

    // const myTexture = gl.createTexture();
    // gl.activeTexture(gl.TEXTURE0 + 10);
    // gl.bindTexture(gl.TEXTURE_2D, myTexture);

    // const error = gl.getError();
    // console.log('mytexture bind', error);
    // if (error !== gl.NO_ERROR) {
    //   throw new Error('WebGL Error: ' + getWebGLErrorMessage(gl, error));
    // }

    // const alignment = 1;
    // gl.pixelStorei(gl.UNPACK_ALIGNMENT, alignment);

    // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // const texData = new Uint8Array(4*4*4);
    // for(let i = 0; i < texData.length; i++) {
    //   texData[i] = i;
    // }

    // gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 4, 4, 0, gl.RGBA, gl.UNSIGNED_BYTE,
    //           texData);

    // setTimeout(() => {
    //   console.log(`My texture`);
    //   const asTensor = fromCamera(cameraTexture!, 4, 4, 4) as tf.Tensor;
    //   console.log(asTensor.shape);
    //   const imageData = asTensor.dataSync();
    //   console.log('fromCamera Result');
    //   console.log(imageData.length, Array.from(imageData.slice(0, 0 + 20)));
    // }, 3000);

    // setInterval(() => {
    //   console.log(`My texture 2`);
    //   const asTensor = fromCamera(cameraTexture!, 4, 4, 4) as tf.Tensor;
    //   console.log(asTensor.shape);
    //   const imageData = asTensor.dataSync();
    //   console.log('fromCamera Result');
    //   console.log(imageData.length, Array.from(imageData.slice(0, 0 + 20)));
    // }, 5000);


    // Compile vertex and fragment shaders

    const w = 512;
    const h = 512;
    const d = 3;
    const pixels = new Uint8Array(w*h*d);


      const vertShader = gl.createShader(gl.VERTEX_SHADER)!;
      gl.shaderSource(vertShader, vertShaderSource);
      gl.compileShader(vertShader);

      const fragShader = gl.createShader(gl.FRAGMENT_SHADER)!;
      gl.shaderSource(fragShader, fragShaderSource);
      gl.compileShader(fragShader);

      // Link, use program, save and enable attributes
      const program = gl.createProgram()!;
      gl.attachShader(program, vertShader);
      gl.attachShader(program, fragShader);
      gl.linkProgram(program);
      gl.validateProgram(program);

      gl.useProgram(program);

      const positionAttrib = gl.getAttribLocation(program, 'position');
      gl.enableVertexAttribArray(positionAttrib);

      // Create, bind, fill buffer
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      const verts = new Float32Array([-2, 0, 0, -2, 2, 2]);
      gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);

      // Bind 'position' attribute
      gl.vertexAttribPointer(positionAttrib, 2, gl.FLOAT, false, 0, 0);

      // Set 'cameraTexture' uniform
      gl.uniform1i(gl.getUniformLocation(program, 'cameraTexture'), 0);

      // // Activate unit 0
      gl.activeTexture(gl.TEXTURE0);


    // Render loop
    const loop = () => {
      this._rafID = requestAnimationFrame(loop);


      console.log('loop');
       if (gl.isContextLost()) {
         console.log('GL CONTEXT IS LOST');
      }

      // Draw
      gl.clearColor(0, 0, 0, 1);
      // tslint:disable-next-line: no-bitwise
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.bindTexture(gl.TEXTURE_2D, cameraTexture);
      gl.drawArrays(gl.TRIANGLES, 0, verts.length / 2);

      // Just downloading
      const fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(
          gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
          gl.TEXTURE_2D, cameraTexture, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);

      const start = Date.now();
      gl.readPixels(0, 0, w, h, gl.RGB, gl.UNSIGNED_BYTE, pixels);
      const end = Date.now();
      console.log('pixels read', pixels.length, end - start); // Uint8Array
      console.log('pixels sample', Array.from(pixels.slice(100, 120)));

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      gl.endFrameEXP();
    };
    loop();
  }

  renderCameraCapture() {
    const {hasCameraPermission} = this.state;

    if (hasCameraPermission === null) {
      return <View />;
    } else if (hasCameraPermission === false) {
      return <Text>No access to camera</Text>;
    }
    return (
      <View  style={styles.cameraContainer}>
        <Camera
          style={styles.camera}
          type={this.state.cameraType}
          ref={ref => { this.camera = ref; }}>
        </Camera>
        <View style={styles.cameraControls}>
            <TouchableHighlight
              style={styles.flipCameraBtn}
              onPress={() => {this.flipCamera();}}
              underlayColor='#FFDE03'>
              <Text style={{fontSize: 16, color: 'white'}}>
                FLIP
              </Text>
            </TouchableHighlight>
            <TouchableHighlight
              style={styles.takeImageBtn}
              onPress={() => { this.handleCameraCapture(); }}
              underlayColor='#FFDE03'>
              <Text style={{fontSize: 16, color: 'white', fontWeight: 'bold'}}>
                TAKE
              </Text>
            </TouchableHighlight>
            <TouchableHighlight
              style={styles.cancelBtn}
              onPress={() => {this.showResults(); }}
              underlayColor='#FFDE03'>
              <Text style={{fontSize: 16, color: 'white'}}>
                BACK
              </Text>
            </TouchableHighlight>
          </View>
        </View>
    );
  }

  renderResults() {
    const {resultImage} = this.state;
    return (
      <View>
        <View style={styles.resultImageContainer}>
          {resultImage == null ?
            <Text style={styles.introText}>
              Tap the squares below to add style and content
              images and see the magic!
            </Text>
            :
            <Image
              style={styles.resultImage}
              resizeMode='contain'
              source={{uri: toDataUri(resultImage)}} />
          }
          <TouchableHighlight
            style={styles.styleImageContainer}
            onPress={() => this.takeStyleImage()}
            underlayColor='white'>
              {this.renderStyleImagePreview()}
          </TouchableHighlight>

          <TouchableHighlight
            style={styles.contentImageContainer}
            onPress={() => this.takeContentImage()}
            underlayColor='white'>
            {this.renderContentImagePreview()}
          </TouchableHighlight>

        </View>
      </View>
    );
  }

  render() {
    const {isLoading} = this.state;
    const camV = <View style={styles.cameraContainer}>
        <Camera
          style={StyleSheet.absoluteFill}
          type={this.state.cameraType}
          zoom={0}
          ref={ref => this.camera = ref!}
        />
        <GLView
          style={styles.camera}
          onContextCreate={this.onContextCreate.bind(this)}
          ref={ref => this.glView = ref!}
        />
        </View>;
    return (
      <View style={{width:'100%'}}>
        {isLoading ? <View style={[styles.loadingIndicator]}>
          <ActivityIndicator size='large' color='#FF0266' />
        </View> : camV}
        {/* {mode === 'results' ?
              this.renderResults() : this.renderCameraCapture()} */}
      </View>
    );
  }
}

const styles = StyleSheet.create({
   container: {
    flex: 1,
    flexDirection: 'column',
  },
  sectionContainer: {
    marginTop: 32,
    paddingHorizontal: 24
  },
  centeredText: {
    textAlign: 'center',
    fontSize: 14,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: '600',
    color: 'black',
    marginBottom: 6
  },
  loadingIndicator: {
    position: 'absolute',
    top: 20,
    right: 20,
    // flexDirection: 'row',
    // justifyContent: 'flex-end',
    zIndex: 200,
    // width: '100%'
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
    width: '92%',
    height: '64%',
    backgroundColor: '#f0F',
    zIndex: 1,
    borderWidth: 20,
    borderRadius: 40,
    borderColor: '#f0f',
  },
  cameraControls: {
    display: 'flex',
    flexDirection: 'row',
    width: '92%',
    justifyContent: 'space-between',
    marginTop: 40,
    zIndex: 100,
    backgroundColor: 'transparent',
  },
  flipCameraBtn: {
    backgroundColor: '#424242',
    width: 75,
    height: 75,
    borderRadius:16,
    display: 'flex',
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  takeImageBtn: {
    backgroundColor: '#FF0266',
    width: 75,
    height: 75,
    borderRadius:50,
    display: 'flex',
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  cancelBtn: {
    backgroundColor: '#424242',
    width: 75,
    height: 75,
    borderRadius:4,
    display: 'flex',
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  resultImageContainer : {
    width: '100%',
    height: '100%',
    padding:5,
    margin:0,
    backgroundColor: '#fff',
    zIndex: 1,
  },
  resultImage: {
    width: '98%',
    height: '98%',
  },
  styleImageContainer: {
    position:'absolute',
    width: 80,
    height: 150,
    bottom: 30,
    left: 20,
    zIndex: 10,
    borderRadius:10,
    backgroundColor: 'rgba(176, 222, 255, 0.5)',
    borderWidth: 1,
    borderColor: 'rgba(176, 222, 255, 0.7)',
  },
  contentImageContainer: {
    position:'absolute',
    width: 80,
    height: 150,
    bottom:30,
    right: 20,
    zIndex: 10,
    borderRadius:10,
    backgroundColor: 'rgba(255, 197, 161, 0.5)',
    borderWidth: 1,
    borderColor: 'rgba(255, 197, 161, 0.7)',
  },
  imagePreview: {
    width: 78,
    height: 148,
    borderRadius:10,
  },
  instructionText: {
    fontSize: 28,
    fontWeight:'bold',
    paddingLeft: 5
  },
  introText: {
    fontSize: 52,
    fontWeight:'bold',
    padding: 20,
    textAlign: 'left',
  }

});
