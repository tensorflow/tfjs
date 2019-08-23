import * as tf from '@tensorflow/tfjs';
import {fetch} from '@tensorflow/tfjs-react-native';
import * as ImageManipulator from 'expo-image-manipulator';
// import * as FileSystem from 'expo-file-system';
import * as jpeg from 'jpeg-js';
import {Image} from 'react-native';

export async function resizeImage(
    imageUrl: string, width: number): Promise<ImageManipulator.ImageResult> {
  const actions = [{
    resize: {
      width,
    },
  }];
  const saveOptions = {
    compress: 0.8,
    format: ImageManipulator.SaveFormat.JPEG,
    base64: true,
  };
  const res =
      await ImageManipulator.manipulateAsync(imageUrl, actions, saveOptions);
  console.log('image manip res', res.uri, res.base64);
  return res;
}

export async function imageUrlToTensor(
    imageUrl: string, base64?: string): Promise<tf.Tensor3D> {
  let rawImageData;
  if (base64 != null) {
    rawImageData = tf.util.encodeString(base64, 'base64');
  } else {
    const imageAssetPath = Image.resolveAssetSource({uri: imageUrl});
    try {
      const response = await fetch(imageAssetPath.uri, {}, {isBinary: true});
      rawImageData = await response.arrayBuffer();
    } catch (e) {
      console.log(`Error fetching ${imageUrl}`, e);
      throw e;
    }
  }

  const TO_UINT8ARRAY = true;
  const {width, height, data} = jpeg.decode(rawImageData, TO_UINT8ARRAY);
  // Drop the alpha channel info
  const buffer = new Uint8Array(width * height * 3);
  let offset = 0;  // offset into original data
  for (let i = 0; i < buffer.length; i += 3) {
    buffer[i] = data[offset];
    buffer[i + 1] = data[offset + 1];
    buffer[i + 2] = data[offset + 2];

    offset += 4;
  }

  return tf.tensor3d(buffer, [height, width, 3]);
}

// export async function tensorToImageUrl(imageTensor: tf.Tensor3D):
//     Promise<string> {
//   const [width, height] = imageTensor.shape;
//   const buffer = await imageTensor.cast('int32').data();

//   const rawImageData = {
//     data: buffer,
//     width,
//     height,
//   };
//   const jpegImageData = jpeg.encode(rawImageData, 50);
//   console.log(jpegImageData);
// }

export async function tensorToImageUrl(imageTensor: tf.Tensor3D):
    Promise<string> {
  const [height, width] = imageTensor.shape;
  const buffer = await imageTensor.toInt().data();
  const frameData = new Uint8Array(width * height * 4);

  let offset = 0;
  for (let i = 0; i < frameData.length; i += 4) {
    frameData[i] = buffer[offset];
    frameData[i + 1] = buffer[offset + 1];
    frameData[i + 2] = buffer[offset + 2];
    frameData[i + 3] = 0xFF;

    offset += 3;
  }

  const rawImageData = {
    data: frameData,
    width,
    height,
  };
  const jpegImageData = jpeg.encode(rawImageData, 80);

  // Write to temporary file.
  const base64Encoding = tf.util.decodeString(jpegImageData.data, 'base64');
  console.log(
      'yyy base64encoding', base64Encoding.length, base64Encoding.slice(0, 20));
  return `data:image/jpeg;base64,${base64Encoding}`;
}
