/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Tensor, Tensor3D, Tensor4D, tidy, util} from '@tensorflow/tfjs';
import {ensureTensorflowBackend, nodeBackend} from './nodejs_kernel_backend';

export enum ImageType {
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
 *     0: use the number of channels in the JPEG-encoded image.
 *     1: output a grayscale image.
 *     3: output an RGB image.
 * @param ratio An optional int. Defaults to 1. Downscaling ratio. It is used
 *     when image is type Jpeg.
 * @param fancyUpscaling An optional bool. Defaults to True. If true use a
 *     slower but nicer upscaling of the chroma planes. It is used when image is
 *     type Jpeg.
 * @param tryRecoverTruncated An optional bool. Defaults to False. If true try
 *     to recover an image from truncated input. It is used when image is type
 *     Jpeg.
 * @param acceptableFraction An optional float. Defaults to 1. The minimum
 *     required fraction of lines before a truncated input is accepted. It is
 *     used when image is type Jpeg.
 * @param dctMethod An optional string. Defaults to "". string specifying a hint
 *     about the algorithm used for decompression. Defaults to "" which maps to
 *     a system-specific default. Currently valid values are ["INTEGER_FAST",
 *     "INTEGER_ACCURATE"]. The hint may be ignored (e.g., the internal jpeg
 *     library changes to a version that does not have that specific option.) It
 *     is used when image is type Jpeg.
 * @returns A 3D Tensor of dtype `int32` with shape [height, width, 1/3].
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'node'}
 */
export function decodeJpeg(
    contents: Uint8Array, channels = 0, ratio = 1, fancyUpscaling = true,
    tryRecoverTruncated = false, acceptableFraction = 1,
    dctMethod = ''): Tensor3D {
  ensureTensorflowBackend();
  return tidy(() => {
    return nodeBackend()
        .decodeJpeg(
            contents, channels, ratio, fancyUpscaling, tryRecoverTruncated,
            acceptableFraction, dctMethod)
        .toInt();
  });
}

/**
 * Decode a PNG-encoded image to a 3D Tensor of dtype `int32`.
 *
 * @param contents The PNG-encoded image in an Uint8Array.
 * @param channels An optional int. Defaults to 0. Accepted values are
 *      0: use the number of channels in the PNG-encoded image.
 *      1: output a grayscale image.
 *      3: output an RGB image.
 *      4: output an RGBA image.
 * @param dtype The data type of the result. Only `int32` is supported at this
 *     time.
 * @returns A 3D Tensor of dtype `int32` with shape [height, width, 1/3/4].
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'node'}
 */
export function decodePng(
    contents: Uint8Array, channels = 0, dtype = 'int32'): Tensor3D {
  util.assert(
      dtype === 'int32',
      () => 'decodeImage could only return Tensor of type `int32` for now.');
  ensureTensorflowBackend();
  return tidy(() => {
    return nodeBackend().decodePng(contents, channels).toInt();
  });
}

/**
 * Decode the first frame of a BMP-encoded image to a 3D Tensor of dtype
 * `int32`.
 *
 * @param contents The BMP-encoded image in an Uint8Array.
 * @param channels An optional int. Defaults to 0. Accepted values are
 *      0: use the number of channels in the BMP-encoded image.
 *      3: output an RGB image.
 *      4: output an RGBA image.
 * @returns A 3D Tensor of dtype `int32` with shape [height, width, 3/4].
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'node'}
 */
export function decodeBmp(contents: Uint8Array, channels = 0): Tensor3D {
  ensureTensorflowBackend();
  return tidy(() => {
    return nodeBackend().decodeBmp(contents, channels).toInt();
  });
}

/**
 * Decode the frame(s) of a GIF-encoded image to a 4D Tensor of dtype `int32`.
 *
 * @param contents The GIF-encoded image in an Uint8Array.
 * @returns A 4D Tensor of dtype `int32` with shape [num_frames, height, width,
 *     3]. RGB channel order.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'node'}
 */
export function decodeGif(contents: Uint8Array): Tensor4D {
  ensureTensorflowBackend();
  return tidy(() => {
    return nodeBackend().decodeGif(contents).toInt();
  });
}

/**
 * Given the encoded bytes of an image, it returns a 3D or 4D tensor of the
 * decoded image. Supports BMP, GIF, JPEG and PNG formats.
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
 *     shape is [height, width, channels]
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'node'}
 */
export function decodeImage(
    content: Uint8Array, channels = 0, dtype = 'int32',
    expandAnimations = true): Tensor3D|Tensor4D {
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
      return decodePng(content, channels);
    case ImageType.GIF:
      // If not to expand animations, take first frame of the gif and return
      // as a 3D tensor.
      return tidy(() => {
        const img = decodeGif(content);
        return expandAnimations ? img : img.slice(0, 1).squeeze([0]);
      });
    case ImageType.BMP:
      return decodeBmp(content, channels);
    default:
      return null;
  }
}

/**
 * Encodes an image tensor to JPEG.
 *
 * @param image A 3-D uint8 Tensor of shape [height, width, channels].
 * @param format An optional string from: "", "grayscale", "rgb".
 *     Defaults to "". Per pixel image format.
 *     - '': Use a default format based on the number of channels in the image.
 *     - grayscale: Output a grayscale JPEG image. The channels dimension of
 *       image must be 1.
 *     - rgb: Output an RGB JPEG image. The channels dimension of image must
 *       be 3.
 * @param quality An optional int. Defaults to 95. Quality of the compression
 *     from 0 to 100 (higher is better and slower).
 * @param progressive An optional bool. Defaults to False. If True, create a
 *     JPEG that loads progressively (coarse to fine).
 * @param optimizeSize An optional bool. Defaults to False. If True, spend
 *     CPU/RAM to reduce size with no quality change.
 * @param chromaDownsampling  An optional bool. Defaults to True.
 *     See http://en.wikipedia.org/wiki/Chroma_subsampling.
 * @param densityUnit An optional string from: "in", "cm". Defaults to "in".
 *     Unit used to specify x_density and y_density: pixels per inch ('in') or
 *     centimeter ('cm').
 * @param xDensity An optional int. Defaults to 300. Horizontal pixels per
 *     density unit.
 * @param yDensity An optional int. Defaults to 300. Vertical pixels per
 *     density unit.
 * @param xmpMetadata An optional string. Defaults to "". If not empty, embed
 *     this XMP metadata in the image header.
 * @returns The JPEG encoded data as an Uint8Array.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'node'}
 */
export async function encodeJpeg(
    image: Tensor3D, format: ''|'grayscale'|'rgb' = '', quality = 95,
    progressive = false, optimizeSize = false, chromaDownsampling = true,
    densityUnit: 'in'|'cm' = 'in', xDensity = 300, yDensity = 300,
    xmpMetadata = ''): Promise<Uint8Array> {
  ensureTensorflowBackend();

  const backendEncodeImage = (imageData: Uint8Array) =>
      nodeBackend().encodeJpeg(
          imageData, image.shape, format, quality, progressive, optimizeSize,
          chromaDownsampling, densityUnit, xDensity, yDensity, xmpMetadata);

  return encodeImage(image, backendEncodeImage);
}

/**
 * Encodes an image tensor to PNG.
 *
 * @param image A 3-D uint8 Tensor of shape [height, width, channels].
 * @param compression An optional int. Defaults to -1. Compression level.
 * @returns The PNG encoded data as an Uint8Array.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'node'}
 */
export async function encodePng(
    image: Tensor3D, compression = 1): Promise<Uint8Array> {
  ensureTensorflowBackend();

  const backendEncodeImage = (imageData: Uint8Array) =>
      nodeBackend().encodePng(imageData, image.shape, compression);
  return encodeImage(image, backendEncodeImage);
}

async function encodeImage(
    image: Tensor3D, backendEncodeImage: (imageData: Uint8Array) => Tensor):
    Promise<Uint8Array> {
  const encodedDataTensor =
      backendEncodeImage(new Uint8Array(await image.data()));

  const encodedPngData =
      (
          // tslint:disable-next-line:no-any
          await encodedDataTensor.data())[0] as any as Uint8Array;
  encodedDataTensor.dispose();
  return encodedPngData;
}

/**
 * Helper function to get image type based on starting bytes of the image file.
 */
export function getImageType(content: Uint8Array): string {
  // Classify the contents of a file based on starting bytes (aka magic number:
  // https://en.wikipedia.org/wiki/Magic_number_(programming)#Magic_numbers_in_files)
  // This aligns with TensorFlow Core code:
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
        'Expected image (BMP, JPEG, PNG, or GIF), but got unsupported ' +
        'image type');
  }
}
