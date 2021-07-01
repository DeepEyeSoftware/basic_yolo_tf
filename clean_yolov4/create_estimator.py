import os
import json
import logging
import argparse
import tensorflow as tf
from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.dataset import Dataset
# from core.dataset_api import DatasetAPI
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all


def parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR")) # /opt/ml/model
    parser.add_argument("--sm-output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")) # /opt/ml/output/data
    # parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS")))
    # parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST"))
    
    # Input data and model directory (on S3)
    parser.add_argument('--model_dir', type=str) # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--log_file', type=str, default='/opt/ml/output/logfile.log')
    
    # Input channels in the TensorFlow estimator’s fit call
    # parser.add_argument('--annots-train', type=str, default=os.environ.get('ANNOTS_TRAIN')) 
    parser.add_argument('--common', type=str, default=os.environ.get('SM_CHANNEL_COMMON')) 
    # parser.add_argument('--annots-eval', type=str, default=os.environ.get('ANNOTS_EVAL')) 
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN')) 
    parser.add_argument('--eval', type=str, default=os.environ.get('SM_CHANNEL_EVAL'))

    return parser.parse_known_args()


def custom_loss(y_pred, y_true_label_s, y_true_label_m, y_true_label_l, y_true_bboxes_s, y_true_bboxes_m, y_true_bboxes_l):
        giou_loss = conf_loss = prob_loss = 0.0
        
        loss_items = compute_loss(y_pred[1], y_pred[0], y_true_label_s, y_true_bboxes_s, STRIDE=8, NUM_CLASS=80, IOU_LOSS_THRESH=0.5)

        giou_loss += loss_items[0]
        conf_loss += loss_items[1]
        prob_loss += loss_items[2]
        
        loss_items = compute_loss(y_pred[3], y_pred[2], y_true_label_m, y_true_bboxes_m, STRIDE=16, NUM_CLASS=80, IOU_LOSS_THRESH=0.5)

        giou_loss += loss_items[0]
        conf_loss += loss_items[1]
        prob_loss += loss_items[2]
        
        loss_items = compute_loss(y_pred[5], y_pred[4], y_true_label_l, y_true_bboxes_l, STRIDE=32, NUM_CLASS=80, IOU_LOSS_THRESH=0.5)

        giou_loss += loss_items[0]
        conf_loss += loss_items[1]
        prob_loss += loss_items[2]
        
        total_loss = giou_loss + conf_loss + prob_loss
        
        return total_loss

class LossLayer(tf.keras.layers.Layer):
    name = 'custom_loss'
    def call(self, y_pred, y_true_label_s, y_true_label_m, y_true_label_l, y_true_bboxes_s, y_true_bboxes_m, y_true_bboxes_l):
        self.add_loss(custom_loss(y_pred, y_true_label_s, y_true_label_m, y_true_label_l, y_true_bboxes_s, y_true_bboxes_m, y_true_bboxes_l))
        return y_pred

def get_model():
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(cfg)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
    
    train_output_sizes = cfg.TRAIN.INPUT_SIZE // STRIDES
    
    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3], name='image')
    
    y_true_label_s = tf.keras.layers.Input([train_output_sizes[0], train_output_sizes[0], 3, 85], name='label_s')
    y_true_label_m = tf.keras.layers.Input([train_output_sizes[1], train_output_sizes[1], 3, 85], name='label_m')
    y_true_label_l = tf.keras.layers.Input([train_output_sizes[2], train_output_sizes[2], 3, 85], name='label_l')
    y_true_bboxes_s = tf.keras.layers.Input([150, 4], name='bboxes_s')
    y_true_bboxes_m = tf.keras.layers.Input([150, 4], name='bboxes_m')
    y_true_bboxes_l = tf.keras.layers.Input([150, 4], name='bboxes_l')
    
    freeze_layers = utils.load_freeze_layer(cfg.FLAGS.MODEL, cfg.FLAGS.TINY)

    feature_maps = YOLO(input_layer, NUM_CLASS, cfg.FLAGS.MODEL, cfg.FLAGS.TINY) # returns [conv_sbbox, conv_mbbox, conv_lbbox]
    
    if cfg.FLAGS.TINY:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    else:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            elif i == 1:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

    outputs = LossLayer()(bbox_tensors, y_true_label_s, y_true_label_m, y_true_label_l, y_true_bboxes_s, y_true_bboxes_m, y_true_bboxes_l)
    
    model = tf.keras.Model(inputs=[input_layer,
                                   y_true_label_s, y_true_bboxes_s,
                                   y_true_label_m, y_true_bboxes_m,
                                   y_true_label_l, y_true_bboxes_l], 
                           outputs=outputs)
    
    return model

def my_model_fn(args):
    model = get_model()
    # print(model.output_names)
    # print(model.summary())

    optim = tf.keras.optimizers.Adam()
    model.compile(optimizer=optim)

    export_outputs = {'custom_loss': tf.estimator.export.PredictOutput}
                      
    # estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir='.', export_outputs=export_outputs)
    estimator = tf.keras.estimator.model_to_estimator(keras_model=model, export_outputs=export_outputs, model_dir=args.model_dir)

    return estimator


def my_training_fn(training_flag=True):
    trainset = Dataset(cfg, is_training=training_flag)
    # trainset = DatasetAPI(cfg, is_training=training_flag)

    batch_size = int(cfg.TRAIN.BATCH_SIZE)
    output_sizes = [int(cfg.TRAIN.INPUT_SIZE/cfg.YOLO.STRIDES[i]) for i in [0,1,2]]

    train_ds = tf.data.Dataset.from_generator(trainset.__iter__, 
                        output_signature=(
                                {'image': tf.TensorSpec(shape=(None, 416, 416, 3), dtype=np.float32),
                                'label_s': tf.TensorSpec(shape=(None, output_sizes[0], output_sizes[0], 3, 85), dtype=np.float32),
                                'bboxes_s': tf.TensorSpec(shape=(None, 150, 4), dtype=np.float32),
                                'label_m': tf.TensorSpec(shape=(None, output_sizes[1], output_sizes[1], 3, 85), dtype=np.float32),
                                'bboxes_m': tf.TensorSpec(shape=(None, 150, 4), dtype=np.float32),
                                'label_l': tf.TensorSpec(shape=(None, output_sizes[2], output_sizes[2], 3, 85), dtype=np.float32),
                                'bboxes_l': tf.TensorSpec(shape=(None, 150, 4), dtype=np.float32)}
                        ))
                        
    return train_ds

def serving_input_fn(train_output_sizes):
    inputs = {'image': tf.compat.v1.placeholder(tf.float32, [None, 416, 416, 3]),
              'bboxes_s': tf.compat.v1.placeholder(tf.float32, [None, 150, 4]),
              'label_s': tf.compat.v1.placeholder(tf.float32, [None, train_output_sizes[0], train_output_sizes[0], 3, 85]),
              'bboxes_m': tf.compat.v1.placeholder(tf.float32, [None, 150, 4]),
              'label_m': tf.compat.v1.placeholder(tf.float32, [None, train_output_sizes[1], train_output_sizes[1], 3, 85]),
              'bboxes_l': tf.compat.v1.placeholder(tf.float32, [None, 150, 4]),
              'label_l': tf.compat.v1.placeholder(tf.float32, [None, train_output_sizes[2], train_output_sizes[2], 3, 85])
    }
    
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

if __name__ == "__main__":
    args, unknown = parse_args()
    # logging.basicConfig(filename=args.log_file, level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)
    tf.get_logger().setLevel('ERROR')
    
    try:
        cfg.TRAIN.ANNOT_PATH = os.path.join(args.common, cfg.TRAIN.ANNOT_PATH)
        cfg.TRAIN.TRAIN_PATH = args.train
        
        cfg.TEST.ANNOT_PATH = os.path.join(args.common, cfg.TEST.ANNOT_PATH)
        cfg.TEST.EVAL_PATH = args.eval
        
        cfg.YOLO.CLASSES = os.path.join(args.common, cfg.YOLO.CLASSES)
        
        estimator = my_model_fn(args)
        
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: my_training_fn(), max_steps=1)
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: my_training_fn(False))

        #TODO: Printeaza iar os.walk
        for root, dirs, files in os.walk("/opt/ml/input", topdown=False):
            if 'annotations' in root or 'images' in root or 'android' in root or '.git' in root:
                continue
            for name in files:
                print(os.path.join(root, name))
            for name in dirs:
                print(os.path.join(root, name))

        logging.info("Training started.")
        # # print("Training started.")
        estimator.train(input_fn=lambda: my_training_fn(), steps=1)
        # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        logging.info("Training done.")
        # # print("Training done.")
        
        # # if args.current_host == args.hosts[0]:
        # estimator.export_saved_model(args.sm_model_dir, lambda: serving_input_fn(cfg.TRAIN.INPUT_SIZE // np.array(cfg.YOLO.STRIDES)))
    except Exception as e:
        logging.error(e, exc_info=True)
    finally:
        # Save log file
        pass