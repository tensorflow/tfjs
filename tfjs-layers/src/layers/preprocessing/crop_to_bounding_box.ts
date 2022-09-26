const tf = require('@tensorflow/tfjs')
const assert = require('assert')
const imageGet = require('get-image-data');

// function to load image data into tensor
async function loadLocalImage(filename: any) {
  return new Promise((res, rej) => {
    imageGet(filename, (err : Error, info: any) => {
      if (err) {
        rej(err);
        return;
      }
      const image = tf.fromPixels(info.data)
      console.log(image, '127');
      res(image);
    });
  });
}



  //convert image to tensor
  // need height and width of image, which is the offsets
  // the area we slice out


  //crop tensor based on the off_seet height from the top, and offset_width



