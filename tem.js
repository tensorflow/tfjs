model = await benchmarks['MobileNetV3'].load();
i = tf.ones([1, 224, 224, 3]);
o = model.predict(i)
