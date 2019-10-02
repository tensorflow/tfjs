import { Tensor3D, Tensor4D, tensor3d, util } from '@tensorflow/tfjs-core';
import * as jpeg from 'jpeg-js';

enum ImageType {
  JPEG = 'jpeg',
  PNG = 'png',
  GIF = 'gif',
  BMP = 'BMP'
}

/**
 * Decode a JPEG-encoded image to a 3D Tensor of dtype `int32`.
 *
 * @param contents The JPEG-encoded image in an Uint8Array.
 * @param channels An optional int. Defaults to 0. Accepted values are
 *     0: use the number of channels in the PNG-encoded image.
 *     1: output a grayscale image.
 *     3: output an RGB image.
 * @param ratio An optional int. Defaults to 1. Downscaling ratio. It is used
 *     when image is type Jpeg.
 * @returns A 3D Tensor of dtype `int32` with shape [height, width, 1/3].
 */
/**
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'node'}
 */
export function decodeJpeg(contents: Uint8Array | ArrayBuffer, channels: 0 | 1 | 3 = 0, ratio?: 1 | 2 | 4 | 8): Tensor3D {
  const TO_UINT8ARRAY = true;
  const { width, height, data } = jpeg.decode(contents, TO_UINT8ARRAY);
  // Drop the alpha channel info for mobilenet
  const buffer = new Uint8Array(width * height * 3);
  let offset = 0; // offset into original data
  for (let i = 0; i < buffer.length; i += 3) {
    buffer[i] = data[offset];
    buffer[i + 1] = data[offset + 1];
    buffer[i + 2] = data[offset + 2];

    offset += 4;
  }

  return tensor3d(buffer, [height, width, channels]);
}

/**
 * Given the encoded bytes of an image, it returns a 3D or 4D tensor of the
 * decoded image. Supports JPEG formats, BMP, PNG, and GIF support may be added
 * at a future date.
 *
 * @param content The encoded image in an Uint8Array.
 * @param channels An optional int. Defaults to 0, use the number of channels in
 *     the image. Number of color channels for the decoded image. It is used
 *     when image is type Png, Bmp, or Jpeg.
 * @param dtype The data type of the result. Only `int32` is supported at this
 *     time.
 * @param expandAnimations A boolean which controls the shape of the returned
 *     op's output. If True, the returned op will produce a 3-D tensor for PNG,
 *     JPEG, and BMP files; and a 4-D tensor for all GIFs, whether animated or
 *     not. If, False, the returned op will produce a 3-D tensor for all file
 *     types and will truncate animated GIFs to the first frame.
 * @returns A Tensor with dtype `int32` and a 3- or 4-dimensional shape,
 *     depending on the file type. For gif file the returned Tensor shape is
 *     [num_frames, height, width, 3], and for jpeg/png/bmp the returned Tensor
 *     shape is []height, width, channels]
 */
/**
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'node'}
 */
export function decodeImage(content: Uint8Array, channels: 0 | 1 | 3 = 0, dtype = 'int32', expandAnimations = true): Tensor3D | Tensor4D {
  util.assert(
    dtype === 'int32',
    () => 'decodeImage could only return Tensor of type `int32` for now.');

  const imageType = getImageType(content);

  // The return tensor has dtype uint8, which is not supported in
  // TensorFlow.js, casting it to int32 which is the default dtype for image
  // tensor. If the image is BMP, JPEG or PNG type, expanding the tensors
  // shape so it becomes Tensor4D, which is the default tensor shape for image
  // ([batch,imageHeight,imageWidth, depth]).
  switch (imageType) {
    case ImageType.JPEG:
      return decodeJpeg(content, channels);
    case ImageType.PNG:
      // return decodePng(content, channels);
      return null;
    case ImageType.GIF:
      // If not to expand animations, take first frame of the gif and return
      // as a 3D tensor.
      // return tidy(() => {
      //   const img = decodeGif(content);
      //   return expandAnimations ? img : img.slice(0, 1).squeeze([0]);
      // });
      return null;
    case ImageType.BMP:
      // return decodeBmp(content, channels);
      return null;
    default:
      return null;
  }
}

/**
 * Helper function to get image type based on starting bytes of the image file.
 */
function getImageType(content: Uint8Array): string {
  // Classify the contents of a file based on starting bytes (aka magic number:
  // tslint:disable-next-line:max-line-length
  // https://en.wikipedia.org/wiki/Magic_number_(programming)#Magic_numbers_in_files)
  // This aligns with TensorFlow Core code:
  // tslint:disable-next-line:max-line-length
  // https://github.com/tensorflow/tensorflow/blob/4213d5c1bd921f8d5b7b2dc4bbf1eea78d0b5258/tensorflow/core/kernels/decode_image_op.cc#L44
  if (content.length > 3 && content[0] === 255 && content[1] === 216 &&
    content[2] === 255) {
    // JPEG byte chunk starts with `ff d8 ff`
    return ImageType.JPEG;
  } else if (
    content.length > 4 && content[0] === 71 && content[1] === 73 &&
    content[2] === 70 && content[3] === 56) {
    // GIF byte chunk starts with `47 49 46 38`
    return ImageType.GIF;
  } else if (
    content.length > 8 && content[0] === 137 && content[1] === 80 &&
    content[2] === 78 && content[3] === 71 && content[4] === 13 &&
    content[5] === 10 && content[6] === 26 && content[7] === 10) {
    // PNG byte chunk starts with `\211 P N G \r \n \032 \n (89 50 4E 47 0D 0A
    // 1A 0A)`
    return ImageType.PNG;
  } else if (content.length > 3 && content[0] === 66 && content[1] === 77) {
    // BMP byte chunk starts with `42 4d`
    return ImageType.BMP;
  } else {
    throw new Error(
      'Expected image (JPEG, PNG, or GIF), but got unsupported image type');
  }
}
