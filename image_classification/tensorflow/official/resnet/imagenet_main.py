# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random

import numpy.random
import tensorflow as tf  # pylint: disable=g-bad-import-order

from mlperf_compliance import mlperf_log
from official.resnet import imagenet_preprocessing
from official.resnet import resnet_model
from official.resnet import resnet_run_loop

from nvidia import dali
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf
from nvidia.dali.pipeline import Pipeline

import horovod.tensorflow as hvd
# Initialize Horovod
hvd.init()

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_CLASSES = 1001

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 1500


_BASE_LR = 0.128

###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(data_dir, 'train-%05d-of-01024' % (i+1))
        for i in range(_NUM_TRAIN_FILES)]
  else:
    return [
        os.path.join(data_dir, 'validation-%05d-of-00128' % (i+1))
        for i in range(128)]


def _parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields (values are included as examples):

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  return features['image/encoded'], label


def parse_record(raw_record, is_training, dtype):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: data type to use for images/features.

  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  image_buffer, label = _parse_example_proto(raw_record)


  image = imagenet_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      output_height=_DEFAULT_IMAGE_SIZE,
      output_width=_DEFAULT_IMAGE_SIZE,
      num_channels=_NUM_CHANNELS,
      is_training=is_training)
  image = tf.cast(image, dtype)

  return image, label

########## DALI #######################################


_mean_pixel = [255 * x for x in (0.485, 0.456, 0.406)]
_std_pixel  = [255 * x for x in (0.229, 0.224, 0.225)]

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, rec_path, idx_path,
                 shard_id, num_shards, crop_shape, 
                 min_random_area, max_random_area,
                 min_random_aspect_ratio, max_random_aspect_ratio,
                 nvjpeg_padding, prefetch_queue=3,
                 seed=12,
                 output_layout=types.NCHW, pad_output=True, dtype='float16',
                 mlperf_print=True, use_roi_decode=False, cache_size=0):
        super(HybridTrainPipe, self).__init__(
                batch_size, num_threads, device_id, 
                seed = seed + device_id, 
                prefetch_queue_depth = prefetch_queue)

        if cache_size > 0:
            self.input = ops.MXNetReader(path = [rec_path], index_path=[idx_path],
                                         random_shuffle=True, shard_id=shard_id, num_shards=num_shards,
                                         stick_to_shard=True, lazy_init=True, skip_cached_images=True)
        else:  # stick_to_shard might not exist in this version of DALI.
            self.input = ops.MXNetReader(path = [rec_path], index_path=[idx_path],
                                         random_shuffle=True, shard_id=shard_id, num_shards=num_shards)

        if use_roi_decode and cache_size == 0:
            self.decode = ops.nvJPEGDecoderRandomCrop(device = "mixed", output_type = types.RGB,
                                                      device_memory_padding = nvjpeg_padding,
                                                      host_memory_padding = nvjpeg_padding,
                                                      random_area = [
                                                          min_random_area,
                                                          max_random_area],
                                                      random_aspect_ratio = [
                                                          min_random_aspect_ratio,
                                                          max_random_aspect_ratio])
            self.rrc = ops.Resize(device = "gpu", resize_x=crop_shape[0], resize_y=crop_shape[1])
        else:
            if cache_size > 0:
                self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB,
                                                device_memory_padding = nvjpeg_padding,
                                                host_memory_padding = nvjpeg_padding,
                                                cache_type='threshold',
                                                cache_size=cache_size,
                                                cache_threshold=0,
                                                cache_debug=False)
            else:
                self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB,
                                                device_memory_padding = nvjpeg_padding,
                                                host_memory_padding = nvjpeg_padding)
            
            self.rrc = ops.RandomResizedCrop(device = "gpu",
                                             random_area = [
                                                 min_random_area,
                                                 max_random_area],
                                             random_aspect_ratio = [
                                                 min_random_aspect_ratio,
                                                 max_random_aspect_ratio],
                                             size = crop_shape)

        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT16 if dtype == 'float16' else types.FLOAT,
                                            output_layout = output_layout,
                                            crop = crop_shape,
                                            pad_output = pad_output,
                                            image_type = types.RGB,
                                            mean = _mean_pixel,
                                            std =  _std_pixel)
        self.coin = ops.CoinFlip(probability = 0.5)

        
    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name = "Reader")

        images = self.decode(self.jpegs)
        images = self.rrc(images)
        output = self.cmnp(images, mirror = rng)
        return (output, self.labels.gpu())



class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, rec_path, idx_path,
                 shard_id, num_shards, crop_shape, 
                 nvjpeg_padding, prefetch_queue=3,
                 seed=12, resize_shp=None,
                 output_layout=types.NCHW, pad_output=True, dtype='float16',
                 mlperf_print=True, cache_size=0):

        super(HybridValPipe, self).__init__(
                batch_size, num_threads, device_id, 
                seed = seed + device_id,
                prefetch_queue_depth = prefetch_queue)

        if cache_size > 0:
            self.input = ops.MXNetReader(path = [rec_path], index_path=[idx_path],
                                         random_shuffle=False, shard_id=shard_id, num_shards=num_shards,
                                         stick_to_shard=True, lazy_init=True, skip_cached_images=True)
        else:  # stick_to_shard might not exist in this version of DALI.
            self.input = ops.MXNetReader(path = [rec_path], index_path=[idx_path],
                                         random_shuffle=False, shard_id=shard_id, num_shards=num_shards)

        if cache_size > 0:
            self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB,
                                            device_memory_padding = nvjpeg_padding,
                                            host_memory_padding = nvjpeg_padding,
                                            cache_type='threshold',
                                            cache_size=cache_size,
                                            cache_threshold=0,
                                            cache_debug=False)
        else:
            self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB,
                                            device_memory_padding = nvjpeg_padding,
                                            host_memory_padding = nvjpeg_padding)

        self.resize = ops.Resize(device = "gpu", resize_shorter=resize_shp) if resize_shp else None

        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT16 if dtype == 'float16' else types.FLOAT,
                                            output_layout = output_layout,
                                            crop = crop_shape,
                                            pad_output = pad_output,
                                            image_type = types.RGB,
                                            mean = _mean_pixel,
                                            std =  _std_pixel)

        
    def define_graph(self):
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        if self.resize:
            images = self.resize(images)
        output = self.cmnp(images)
        return (output, self.labels.gpu())



def build_input_pipeline(
        data_train, data_train_idx, data_val, data_val_idx,
        batch_size=128, target_shape=(224,224, 3), seed=1, is_training=True):
    # resize is default base length of shorter edge for dataset;
    # all images will be reshaped to this size
    resize = 256 
    # target shape is final shape of images pipelined to network;
    # all images will be cropped to this size
    #target_shape = tuple([int(l) for l in args.image_shape.split(',')])

    pad_output = target_shape[0] == 4
    #gpus = list(map(int, filter(None, args.gpus.split(',')))) # filter to not encount eventually empty strings
    #batch_size = args.batch_size//len(gpus)

    #mx_resnet_print(
    #        key=mlperf_constants.MODEL_BN_SPAN,
    #        val=batch_size)

    num_threads = 3 #args.dali_threads

    # the input_layout w.r.t. the model is the output_layout of the image pipeline
    output_layout = types.NHWC #types.NHWC if args.input_layout == 'NHWC' else types.NCHW


    data_paths = {}
    data_paths["train_data_tmp"] = data_train
    data_paths["train_idx_tmp"] = data_train_idx
    data_paths["val_data_tmp"] = data_val
    data_paths["val_idx_tmp"] = data_val_idx

    if is_training:
        pipe = HybridTrainPipe(batch_size      = batch_size,
                                  num_threads     = num_threads,
                                  device_id       = hvd.local_rank(),
                                  rec_path       = data_paths["train_data_tmp"],
                                  idx_path       = data_paths["train_idx_tmp"],
                                  shard_id        = hvd.local_rank(),
                                  num_shards      = hvd.size(),
                                  crop_shape      = target_shape[:-1],
                                  min_random_area = 0.05,
                                  max_random_area = 1.0,
                                  min_random_aspect_ratio = 3./4.,
                                  max_random_aspect_ratio = 4./3.,
                                  nvjpeg_padding  = 64 * 1024 * 1024,
                                  prefetch_queue  = 2,
                                  seed            = seed,
                                  output_layout   = output_layout,
                                  pad_output      = pad_output,
                                  dtype           = 'float32',
                                  mlperf_print    = True,
                                  use_roi_decode  = 0,
                                  cache_size      = 6144)

    else:
        pipe =  HybridValPipe(batch_size     = batch_size,
                              num_threads    = num_threads,
                              device_id      = hvd.local_rank(),
                              rec_path       = data_paths["val_data_tmp"],
                              idx_path       = data_paths["val_idx_tmp"],
                              shard_id       = 0,
                              num_shards     = 1,
                              crop_shape     = target_shape[:-1],
                              nvjpeg_padding = 64 * 1024 * 1024,
                              prefetch_queue = 2,
                              seed           = seed,
                              resize_shp     = resize,
                              output_layout  = output_layout,
                              pad_output     = pad_output,
                              dtype          = 'float32',
                              mlperf_print   = True,
                              cache_size     = 6144)
    

    return pipe



class DALIPreprocessor(object):
    def __init__(self,
                 pipe,
                 batch_size,
                 height,
                 width):
        
        device_id = hvd.local_rank()

        daliop = dali_tf.DALIIterator()

        with tf.device("/gpu:0"):
            self.images, self.labels = daliop(
                pipeline=pipe,
                shapes=[(batch_size, height, width, 3), (batch_size, 1)],
                dtypes=[tf.float32, tf.float32],
                device_id=device_id)

    def get_device_minibatches(self):
        return self.images, tf.cast(self.labels, tf.int64)


def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None,
             dtype=tf.float32):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.
    dtype: Data type to use for images/features

  Returns:
    A dataset that can be used for iteration.
  """
  mlperf_log.resnet_print(key=mlperf_log.INPUT_ORDER)

  data_train = '/data/train100k.rec'
  data_train_idx = '/data/train100k.idx'
  data_val = '/data/val.rec'
  data_val_idx = '/data/val.idx'   
    
  pipeline = build_input_pipeline(
        data_train, data_train_idx, data_val, data_val_idx,
        batch_size=batch_size, target_shape=(224,224, 3), seed=1, is_training=is_training)
    
  preprocessor = DALIPreprocessor(pipeline, batch_size, 224, 224)

  images, labels = preprocessor.get_device_minibatches()
  
  return (images, labels)


def get_synth_input_fn():
  return resnet_run_loop.get_synth_input_fn(
      _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS, _NUM_CLASSES)


###############################################################################
# Running the model
###############################################################################
class ImagenetModel(resnet_model.Model):
  """Model class with appropriate defaults for Imagenet data."""

  def __init__(self, resnet_size, data_format=None, num_classes=_NUM_CLASSES,
               version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE):
    """These are the parameters that work for Imagenet data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      version: Integer representing which version of the ResNet network to use.
        See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
      bottleneck = False
      final_size = 512
    else:
      bottleneck = True
      final_size = 2048

    super(ImagenetModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        second_pool_size=7,
        second_pool_stride=1,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        final_size=final_size,
        version=version,
        data_format=data_format,
        dtype=dtype
    )


def _get_block_sizes(resnet_size):
  """Retrieve the size of each block_layer in the ResNet model.

  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.

  Args:
    resnet_size: The number of convolutional layers needed in the model.

  Returns:
    A list of block sizes to use in building the model.

  Raises:
    KeyError: if invalid resnet_size is received.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)


def imagenet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""

  # Warmup and higher lr may not be valid for fine tuning with small batches
  # and smaller numbers of training images.
  if params['fine_tune']:
    base_lr = .1
  else:
    base_lr = .128
  global_batch_size = params['batch_size'] * hvd.size()
  learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
      batch_size=global_batch_size, batch_denom=256,
      num_images=_NUM_IMAGES['train'], boundary_epochs=[30, 60, 80, 90],
      decay_rates=[1, 0.1, 0.01, 0.001, 1e-4], base_lr=_BASE_LR,
      enable_lars=params['enable_lars'])

  return resnet_run_loop.resnet_model_fn(
      features=features,
      labels=labels,
      mode=mode,
      model_class=ImagenetModel,
      resnet_size=params['resnet_size'],
      weight_decay=params['weight_decay'],
      learning_rate_fn=learning_rate_fn,
      momentum=0.9,
      data_format=params['data_format'],
      version=params['version'],
      loss_scale=params['loss_scale'],
      loss_filter_fn=None,
      dtype=params['dtype'],
      label_smoothing=params['label_smoothing'],
      enable_lars=params['enable_lars']
  )


def main(argv):
  parser = resnet_run_loop.ResnetArgParser(
      resnet_size_choices=[18, 34, 50, 101, 152, 200])

  parser.set_defaults(
       train_epochs=90,
       version=1
  )

  flags = parser.parse_args(args=argv[2:])

  seed = int(argv[1])
  print('Setting random seed = ', seed)
  print('special seeding')
  mlperf_log.resnet_print(key=mlperf_log.RUN_SET_RANDOM_SEED, value=seed)
  random.seed(seed)
  tf.set_random_seed(seed)
  numpy.random.seed(seed)

  mlperf_log.resnet_print(key=mlperf_log.PREPROC_NUM_TRAIN_EXAMPLES,
                          value=_NUM_IMAGES['train'])
  mlperf_log.resnet_print(key=mlperf_log.PREPROC_NUM_EVAL_EXAMPLES,
                          value=_NUM_IMAGES['validation'])
  input_function = flags.use_synthetic_data and get_synth_input_fn() or input_fn

  resnet_run_loop.resnet_main(seed,
      flags, imagenet_model_fn, input_function,
      shape=[_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS])


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  mlperf_log.ROOT_DIR_RESNET = os.path.split(os.path.abspath(__file__))[0]
  main(argv=sys.argv)
