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
    gl: ExpoWebGLRenderingContext
  ) => void;
}

interface State {
  cameraLayout: { x: number; y: number; width: number; height: number };
}

const DEFAULT_AUTORENDER = true;
const DEFAULT_RESIZE_DEPTH = 3;

export function cameraWithTensors<T extends WrappedComponentProps>(
  // tslint:disable-next-line: variable-name
  CameraComponent: React.ComponentType<T>,
) {
  return class CameraWithTensorStream
    extends React.Component<T & Props, State> {
    camera: Camera;
    glView: GLView;
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
      this.camera = null;
      this.glView = null;
      cancelAnimationFrame(this.rafID);
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
        throw new Error('glView or camera not available');
      }
    }

    /**
     * Callback for GL context creation. We do mose of the work of setting
     * up the component here.
     * @param gl
     */
    async onGLContextCreate(gl: ExpoWebGLRenderingContext) {
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

        while (true) {
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
      this.props.onReady(nextFrameIterator, updatePreview, gl);
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
        const flipHorizontal = Platform.OS === 'ios' ? false : true;

        renderToGLView(gl, cameraTexture, { width, height }, flipHorizontal);
      };

      return renderFunc.bind(this);
    }

    /**
     * Render the component
     */
    render() {
      const { cameraLayout } = this.state;

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
        if(!tensorCameraPropKeys.includes(key)) {
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
            zIndex: 10,
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
