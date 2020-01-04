import tensorflow as tf
tf.enable_eager_execution()
input_ = tf.ones([2, 3, 4, 5, 2])
filters = [8]
paddings = ['valid', 'same']
kernels = [[2, 2, 2]]
strides = [2]

for filter_ in filters:
  for padding in paddings:
    for kernel in kernels:
      for stride in strides:
        if not stride:
          stride = kernel
        print(' padding: {padding}\n kernel: {kernel}\n stride: {stride}'.format(filter_=filter_, padding=padding, kernel=kernel, stride=stride))
        t = tf.keras.layers.Conv3DTranspose(filter_, kernel, stride, padding, kernel_initializer='ones', bias_initializer='ones', data_format='channels_last').apply(input_)
        print(t.shape)
        t = tf.keras.layers.Conv3DTranspose(filter_, kernel, stride, padding, kernel_initializer='ones', bias_initializer='ones', data_format='channels_first').apply(input_)
        print(t.shape)

