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

import * as React from 'react';
import * as tf from '@tensorflow/tfjs-core';
import {
  StyleSheet,
  PixelRatio,
  LayoutChangeEvent,
  Platform
} from 'react-native';
import { Camera } from 'expo-camera';
import { GLView, ExpoWebGLRenderingContext } from 'expo-gl';
import { fromTexture, renderToGLView, detectGLCapabilities } from './camera';

interface WrappedComponentProps {
  onLayout?: (event: LayoutChangeEvent) => void;
  // tslint:disable-next-line: no-any
  [index: string]: any;
}

interface Props {
  cameraTextureWidth: number;
  cameraTextureHeight: number;
  resizeWidth: number;
  resizeHeight: number;
  resizeDepth: number;
  autorender: boolean;
  onReady: (
    images: IterableIterator<tf.Tensor3D>,
    updateCameraPreview: () => void,
    gl: ExpoWebGLRenderingContext,
    cameraTexture: WebGLTexture,
  ) => void;
}

interface State {
  cameraLayout: { x: number; y: number; width: number; height: number };
}

const DEFAULT_AUTORENDER = true;
const DEFAULT_RESIZE_DEPTH = 3;

/**
 * A higher-order-component (HOC) that augments the [Expo.Camera](https://docs.expo.io/versions/latest/sdk/camera/)
 * component with the ability to yield tensors representing the camera stream.
 *
 * Because the camera data will be consumed in the process, the original
 * camera component will not render any content. This component provides
 * options that can be used to render the camera preview.
 *
 * Notably the component allows on-the-fly resizing of the camera image to
 * smaller dimensions, this speeds up data transfer between the native and
 * javascript threads immensely.
 *
 * __In addition to__ all the props taken by Expo.Camera. The returned
 * component takes the following props
 *
 * - __cameraTextureWidth__: number — the width the camera preview texture
 *   (see example and note below)
 * - __cameraTextureHeight__: number — the height the camera preview texture
 *   (see example and note below)
 * - __resizeWidth__: number — the width of the output tensor
 * - __resizeHeight__: number — the height of the output tensor
 * - __resizeDepth__: number — the depth (num of channels) of the output tensor.
 *    Should be 3 or 4.
 * - __autorender__: boolean — if true the view will be automatically updated
 *   with the contents of the camera. Set this to false if you want more direct
 *   control on when rendering happens.
 * - __onReady__: (
 *    images: IterableIterator<tf.Tensor3D>,
 *    updateCameraPreview: () => void,
 *    gl: ExpoWebGLRenderingContext,
 *    cameraTexture: WebGLTexture
 *  ) => void — When the component is mounted and ready this callback will
 *  be called and recieve the following 3 elements:
 *    - __images__ is a (iterator)[https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Iterators_and_Generators]
 *      that yields tensors representing the camera image on demand.
 *    - __updateCameraPreview__ is a function that will update the WebGL render
 *      buffer with the contents of the camera. Not needed when `autorender`
 *      is true
 *    - __gl__ is the ExpoWebGl context used to do the rendering. After calling
 *      `updateCameraPreview` and any other operations you want to synchronize
 *      to the camera rendering you must call gl.endFrameExp() to display it
 *      on the screen. This is also provided in case you want to do other
 *      rendering using WebGL. Not needed when `autorender` is true.
 *    - __cameraTexture__ The underlying cameraTexture. This can be used to
 *      implement your own __updateCameraPreview__.
 *
 * ```js
 * import { Camera } from 'expo-camera';
 * import { cameraWithTensors } from '@tensorflow/tfjs-react-native';
 *
 * const TensorCamera = cameraWithTensors(Camera);
 *
 * class MyComponent {
 *
 *   handleCameraStream(images, updatePreview, gl) {
 *     const loop = async () => {
 *       const nextImageTensor = images.next().value
 *
 *       //
 *       // do something with tensor here
 *       //
 *
 *       // if autorender is false you need the following two lines.
 *       // updatePreview();
 *       // gl.endFrameEXP();
 *
 *       requestAnimationFrame(loop);
 *     }
 *     loop();
 *   }
 *
 *   render() {
 *    // Currently expo does not support automatically determining the
 *    // resolution of the camera texture used. So it must be determined
 *    // empirically for the supported devices and preview size.
 *
 *    let textureDims;
 *    if (Platform.OS === 'ios') {
 *     textureDims = {
 *       height: 1920,
 *       width: 1080,
 *     };
 *    } else {
 *     textureDims = {
 *       height: 1200,
 *       width: 1600,
 *     };
 *    }
 *
 *    return <View>
 *      <TensorCamera
 *       // Standard Camera props
 *       style={styles.camera}
 *       type={Camera.Constants.Type.front}
 *       // Tensor related props
 *       cameraTextureHeight={textureDims.height}
 *       cameraTextureWidth={textureDims.width}
 *       resizeHeight={200}
 *       resizeWidth={152}
 *       resizeDepth={3}
 *       onReady={this.handleCameraStream}
 *       autorender={true}
 *      />
 *    </View>
 *   }
 * }
 * ```
 *
 * @param CameraComponent an expo Camera component constructor
 */
/** @doc {heading: 'Media', subheading: 'Camera'} */
export function cameraWithTensors<T extends WrappedComponentProps>(
  // tslint:disable-next-line: variable-name
  CameraComponent: React.ComponentType<T>,
) {
  return class CameraWithTensorStream
    extends React.Component<T & Props, State> {
    camera: Camera;
    glView: GLView;
    glContext: ExpoWebGLRenderingContext;
    rafID: number;

    constructor(props: T & Props) {
      super(props);
      this.onCameraLayout = this.onCameraLayout.bind(this);
      this.onGLContextCreate = this.onGLContextCreate.bind(this);

      this.state = {
        cameraLayout: null,
      };
    }

    componentWillUnmount() {
      cancelAnimationFrame(this.rafID);
      if(this.glContext) {
        GLView.destroyContextAsync(this.glContext);
      }
      this.camera = null;
      this.glView = null;
      this.glContext = null;
    }

    /*
     * Measure the camera component when it is laid out so that we can overlay
     * the GLView.
     */
    onCameraLayout(event: LayoutChangeEvent) {
      const { x, y, width, height } = event.nativeEvent.layout;
      this.setState({
        cameraLayout: { x, y, width, height },
      });
    }

    /**
     * Creates a WebGL texture that is updated by the underlying platform to
     * contain the contents of the camera.
     */
    async createCameraTexture(): Promise<WebGLTexture> {
      if (this.glView != null && this.camera != null) {
        //@ts-ignore
        return this.glView.createCameraTextureAsync(this.camera);
      } else {
        throw new Error('Expo GL context or camera not available');
      }
    }

    /**
     * Callback for GL context creation. We do mose of the work of setting
     * up the component here.
     * @param gl
     */
    async onGLContextCreate(gl: ExpoWebGLRenderingContext) {
      this.glContext = gl;
      const cameraTexture = await this.createCameraTexture();
      await detectGLCapabilities(gl);

      // Optionally set up a render loop that just displays the camera texture
      // to the GLView.
      const autorender =
        this.props.autorender != null
          ? this.props.autorender
          : DEFAULT_AUTORENDER;
      const updatePreview = this.previewUpdateFunc(gl, cameraTexture);
      if (autorender) {
        const renderLoop = () => {
          updatePreview();
          gl.endFrameEXP();
          this.rafID = requestAnimationFrame(renderLoop);
        };
        renderLoop();
      }

      const {
        resizeHeight,
        resizeWidth,
        resizeDepth,
        cameraTextureHeight,
        cameraTextureWidth,
      } = this.props;

      //
      //  Set up a generator function that yields tensors representing the
      // camera on demand.
      //
      const cameraStreamView = this;
      function* nextFrameGenerator() {
        const RGBA_DEPTH = 4;
        const textureDims = {
          height: cameraTextureHeight,
          width: cameraTextureWidth,
          depth: RGBA_DEPTH,
        };

        const targetDims = {
          height: resizeHeight,
          width: resizeWidth,
          depth: resizeDepth || DEFAULT_RESIZE_DEPTH,
        };

        while (cameraStreamView.glContext != null) {
          const imageTensor = fromTexture(
            gl,
            cameraTexture,
            textureDims,
            targetDims
          );
          yield imageTensor;
        }
      }
      const nextFrameIterator = nextFrameGenerator();

      // Pass the utility functions to the caller provided callback
      this.props.onReady(nextFrameIterator, updatePreview, gl, cameraTexture);
    }

    /**
     * Helper function that can be used to update the GLView framebuffer.
     *
     * @param gl the open gl texture to render to
     * @param cameraTexture the texture to draw.
     */
    previewUpdateFunc(
      gl: ExpoWebGLRenderingContext,
      cameraTexture: WebGLTexture
    ) {
      const renderFunc = () => {
        const { cameraLayout } = this.state;
        const width = PixelRatio.getPixelSizeForLayoutSize(cameraLayout.width);
        const height = PixelRatio.getPixelSizeForLayoutSize(
          cameraLayout.height
        );
        const isFrontCamera =
          this.camera.props.type === Camera.Constants.Type.front;
        const flipHorizontal =
          Platform.OS === 'ios' && isFrontCamera ? false : true;

        renderToGLView(gl, cameraTexture, { width, height }, flipHorizontal);
      };

      return renderFunc.bind(this);
    }

    /**
     * Render the component
     */
    render() {
      const { cameraLayout } = this.state;

      // Before passing props into the original wrapped component we want to
      // remove the props that we augment the component with.

      // Use this object to use typescript to check that we are removing
      // all the tensorCamera properties.
      const tensorCameraPropMap: Props = {
        cameraTextureWidth: null,
        cameraTextureHeight: null,
        resizeWidth: null,
        resizeHeight: null,
        resizeDepth: null,
        autorender: null,
        onReady: null,
      };
      const tensorCameraPropKeys = Object.keys(tensorCameraPropMap);

      const cameraProps: WrappedComponentProps = {};
      const allProps = Object.keys(this.props);
      for (let i = 0; i < allProps.length; i++) {
        const key = allProps[i];
        if (!tensorCameraPropKeys.includes(key)) {
          cameraProps[key] = this.props[key];
        }
      }

      // Set up an on layout handler
      const onlayout = this.props.onLayout ? (e: LayoutChangeEvent) => {
        this.props.onLayout(e);
        this.onCameraLayout(e);
      } : this.onCameraLayout;

      cameraProps.onLayout = onlayout;

      const cameraComp = (
        //@ts-ignore see https://github.com/microsoft/TypeScript/issues/30650
        <CameraComponent
          key='camera-with-tensor-camera-view'
          {...(cameraProps)}
          ref={(ref: Camera) => (this.camera = ref)}
        />
      );

      // Create the glView if the camera has mounted.
      let glViewComponent = null;
      if (cameraLayout != null) {
        const styles = StyleSheet.create({
          glView: {
            position: 'absolute',
            left: cameraLayout.x,
            top: cameraLayout.y,
            width: cameraLayout.width,
            height: cameraLayout.height,
            zIndex: this.props.style.zIndex ?
              parseInt(this.props.style.zIndex, 10) + 10 : 10,
          }

        });

        glViewComponent = (
          <GLView
            key='camera-with-tensor-gl-view'
            style={styles.glView}
            onContextCreate={this.onGLContextCreate}
            ref={ref => (this.glView = ref)}
          />
        );
      }

      return [cameraComp, glViewComponent];
    }
  };
}
