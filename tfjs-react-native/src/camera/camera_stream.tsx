import * as tf from "@tensorflow/tfjs-core";
import {CameraView} from "expo-camera";
import {ExpoWebGLRenderingContext, GLView} from "expo-gl";
import {
  Fragment,
  useCallback,
  useEffect,
  useRef,
  useState,
  type ComponentType,
  type FC,
} from "react";
import {LayoutChangeEvent, PixelRatio, Platform} from "react-native";
import {detectGLCapabilities, fromTexture, renderToGLView} from "./camera";
import {Rotation} from "./types";

interface WrappedComponentProps {
  onLayout?: (event: LayoutChangeEvent) => void;
  // tslint:disable-next-line: no-any
  [index: string]: any;
}

interface Props {
  useCustomShadersToResize: boolean;
  cameraTextureWidth: number;
  cameraTextureHeight: number;
  resizeWidth: number;
  resizeHeight: number;
  resizeDepth: number;
  autorender: boolean;
  rotation?: Rotation;
  onReady: (
    images: IterableIterator<tf.Tensor3D>,
    updateCameraPreview: () => void,
    gl: ExpoWebGLRenderingContext,
    cameraTexture: WebGLTexture,
  ) => void;
}

const DEFAULT_AUTORENDER = true;
const DEFAULT_RESIZE_DEPTH = 3;
const DEFAULT_USE_CUSTOM_SHADERS_TO_RESIZE = false;

type CameraComponentProps<P> = Omit<
  P & Props,
  | "rotation"
  | "style"
  | "useCustomShadersToResize"
  | "cameraTextureWidth"
  | "cameraTextureHeight"
  | "resizeWidth"
  | "resizeHeight"
  | "resizeDepth"
  | "autorender"
  | "onReady"
>;

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
 * - __use_custom_shaders_to_resize__: boolean — whether to use custom shaders
 *   to resize the camera image to smaller dimensions that fit the output
 *   tensor.
 *   - If it is set to false (default and recommended), the resize will be done
 *     by the underlying GL system when drawing the camera image texture to the
 *     target output texture with TEXTURE_MIN_FILTER/TEXTURE_MAG_FILTER set to
 *     gl.LINEAR, and there is no need to provide `cameraTextureWidth` and
 *     `cameraTextureHeight` props below.
 *   - If it is set to true (legacy), the resize will be done by the custom
 *     shaders defined in `resize_bilinear_program_info.ts`. Setting it to true
 *     also requires that client provide the correct `cameraTextureWidth` and
 *     `cameraTextureHeight` props below. Unfortunately there is no official API
 *     to get the camera texture size programmatically so they have to be
 *     decided empirically. From our experience, it is hard to cover all cases
 *     in this way because different devices models and/or preview sizes might
 *     produce different camera texture sizes.
 * - __cameraTextureWidth__: number — the width the camera preview texture
 *   (see note above)
 * - __cameraTextureHeight__: number — the height the camera preview texture
 *   (see note above)
 * - __resizeWidth__: number — the width of the output tensor
 * - __resizeHeight__: number — the height of the output tensor
 * - __resizeDepth__: number — the depth (num of channels) of the output tensor.
 *    Should be 3 or 4.
 * - __autorender__: boolean — if true the view will be automatically updated
 *   with the contents of the camera. Set this to false if you want more direct
 *   control on when rendering happens.
 * - __rotation__: number — the degrees that the internal camera texture and
 *   preview will be rotated. Accepted values: 0, +/- 90, +/- 180, +/- 270 or
 *   360.
 * - __onReady__: (
 *    images: IterableIterator<tf.Tensor3D>,
 *    updateCameraPreview: () => void,
 *    gl: ExpoWebGLRenderingContext,
 *    cameraTexture: WebGLTexture
 *  ) => void — When the component is mounted and ready this callback will
 *  be called and receive the following 3 elements:
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
 * import { CameraView } from 'expo-camera';
 * import { cameraWithTensors } from '@tensorflow/tfjs-react-native';
 *
 * const TensorCamera = cameraWithTensors(CameraView);
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
 *    return <View>
 *      <TensorCamera
 *       // Standard Camera props
 *       style={styles.camera}
 *       type={Camera.Constants.Type.front}
 *       // Tensor related props
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
  CameraComponent: ComponentType<CameraComponentProps<T>>,
) {
  const CameraWithTensorStream: FC<T & Props> = (props) => {
    const cameraRef = useRef<CameraView>(null);
    const glViewRef = useRef<GLView>(null);
    const glContextRef = useRef<ExpoWebGLRenderingContext>(null);
    const rafIDRef = useRef<number>(0);

    const [cameraLayout, setCameraLayout] = useState<{
      x: number;
      y: number;
      width: number;
      height: number;
    } | null>(null);

    const onCameraLayout = useCallback((event: LayoutChangeEvent) => {
      const {x, y, width, height} = event.nativeEvent.layout;
      setCameraLayout({x, y, width, height});
    }, []);

    const createCameraTexture = useCallback(async (): Promise<WebGLTexture> => {
      if (glViewRef.current != null && cameraRef.current != null) {
        //@ts-ignore
        return glViewRef.current.createCameraTextureAsync(cameraRef.current);
      } else {
        throw new Error("Expo GL context or camera not available");
      }
    }, []);

    const previewUpdateFunc = useCallback(
      (gl: ExpoWebGLRenderingContext, cameraTexture: WebGLTexture) => {
        const renderFunc = () => {
          if (!cameraLayout) return;
          const {rotation} = props;
          const width = PixelRatio.getPixelSizeForLayoutSize(
            cameraLayout.width,
          );
          const height = PixelRatio.getPixelSizeForLayoutSize(
            cameraLayout.height,
          );
          const isFrontCamera = cameraRef.current?.props.facing === "front";
          const flipHorizontal =
            Platform.OS === "ios" && isFrontCamera ? false : true;

          renderToGLView(
            gl,
            cameraTexture,
            {width, height},
            flipHorizontal,
            rotation,
          );
        };

        return renderFunc;
      },
      [cameraLayout.height, cameraLayout.width, props.rotation],
    );

    const onGLContextCreate = useCallback(
      async (gl: ExpoWebGLRenderingContext) => {
        glContextRef.current = gl;
        const cameraTexture = await createCameraTexture();
        await detectGLCapabilities(gl);

        // Optionally set up a render loop that just displays the camera texture to the GLView.
        const autorender =
          props.autorender != null ? props.autorender : DEFAULT_AUTORENDER;
        const updatePreview = previewUpdateFunc(gl, cameraTexture);
        if (autorender) {
          const renderLoop = () => {
            updatePreview();
            gl.endFrameEXP();
            rafIDRef.current = requestAnimationFrame(renderLoop);
          };
          renderLoop();
        }

        const {resizeDepth} = props;

        // cameraTextureHeight and cameraTextureWidth props can be omitted when
        // useCustomShadersToResize is set to false. Setting a default value to
        // them here.
        const cameraTextureHeight =
          props.cameraTextureHeight != null ? props.cameraTextureHeight : 0;
        const cameraTextureWidth =
          props.cameraTextureWidth != null ? props.cameraTextureWidth : 0;
        const useCustomShadersToResize =
          props.useCustomShadersToResize != null
            ? props.useCustomShadersToResize
            : DEFAULT_USE_CUSTOM_SHADERS_TO_RESIZE;

        //
        // Set up a generator function that yields tensors representing the
        // camera on demand.
        //
        function* nextFrameGenerator() {
          const RGBA_DEPTH = 4;
          const textureDims = {
            height: cameraTextureHeight,
            width: cameraTextureWidth,
            depth: RGBA_DEPTH,
          };

          while (glContextRef.current != null) {
            if (!cameraLayout) return;
            const targetDims = {
              height: props.resizeHeight,
              width: props.resizeWidth,
              depth: resizeDepth || DEFAULT_RESIZE_DEPTH,
            };

            const imageTensor = fromTexture(
              gl,
              cameraTexture,
              textureDims,
              targetDims,
              useCustomShadersToResize,
              {rotation: props.rotation},
            );
            yield imageTensor;
          }
        }
        const nextFrameIterator = nextFrameGenerator();

        // Pass the utility functions to the caller provided callback
        props.onReady(nextFrameIterator, updatePreview, gl, cameraTexture);
      },
      [
        props.autorender,
        props.resizeDepth,
        props.cameraTextureHeight,
        props.cameraTextureWidth,
        props.useCustomShadersToResize,
        props.rotation,
        props.resizeHeight,
        props.resizeWidth,
        props.onReady,
        createCameraTexture,
        previewUpdateFunc,
        !cameraLayout,
      ],
    );

    useEffect(() => {
      return () => {
        cancelAnimationFrame(rafIDRef.current);
        if (glContextRef.current) {
          GLView.destroyContextAsync(glContextRef.current);
        }
        cameraRef.current = null;
        glViewRef.current = null;
        glContextRef.current = null;
      };
    }, []);

    const {onLayout: propOnLayout, } = props;

    const onlayout = useCallback(
      (e: LayoutChangeEvent) => {
        propOnLayout?.(e);
        onCameraLayout(e);
      },
      [propOnLayout, onCameraLayout],
    );

    const {
      useCustomShadersToResize,
      cameraTextureWidth,
      cameraTextureHeight,
      resizeWidth,
      resizeHeight,
      resizeDepth,
      autorender,
      onReady,
      rotation,
      style,
      ...cameraProps
    } = props;

    return (
      <Fragment>
        <CameraComponent
          {...cameraProps}
          style={style}
          onLayout={onlayout}
          ref={cameraRef}
        />

        {cameraLayout && (
          <GLView
            style={[
              {
                position: "absolute",
                left: cameraLayout.x,
                top: cameraLayout.y,
                width: cameraLayout.width,
                height: cameraLayout.height,
                zIndex: style?.zIndex
                  ? parseInt(String(style.zIndex), 10) + 10
                  : 10,
              },
            ]}
            onContextCreate={onGLContextCreate}
            ref={glViewRef}
          />
        )}
      </Fragment>
    );
  };
  CameraWithTensorStream.displayName = "CameraWithTensorStream"; // Optional: Set a display name for debugging

  return CameraWithTensorStream;
}
