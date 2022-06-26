import base64
import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
import numpy as np
import cv2
from app.shadows_estimator import MultiTank

from fastai.vision import *
import torchvision.transforms as T
import base64
import PIL
#from PIL import Image




def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])
    return x


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x


def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)

    return yolo_conv


# yolo_head
def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)

    return yolo_output


def YoloV3(size, channels, classes, masks=np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])):
    x = inputs = Input([size, size, channels], name='input')

    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    # channel 1024
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    # channel 512
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    # channel 256
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

    return Model(inputs, (output_0, output_1, output_2), name='yolov3')


def load_darknet_weights(model, weights_file):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    layers = ['yolo_darknet', 'yolo_conv_0', 'yolo_output_0', 'yolo_conv_1',
              'yolo_output_1', 'yolo_conv_2', 'yolo_output_2']

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            # print("{}/{} {}".format(sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def transform_images(x_train, size=416):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)


def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    # (x,y) -> centre of bounding box, (w,h) -> height and width
    grid_size = tf.shape(pred)[1:3]  # [grid grid] => [13,13] or [26,26] or [52,52]

    # Splitting pred output by last channel (x, y, w, h, obj, ...classes)
    # box_xy: (batch_size, grid, grid, anchors, (x,y))
    # box_wh: (batch_size, grid, grid, anchors, (w,h))
    # objectness: (batch_size, grid, grid, anchors, obj)
    # class_prob: (batch_size, grid, grid, anchors, classes)
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    # Applying sigmoid function to each splitted output except height and width
    box_xy = tf.sigmoid(box_xy)  # [tx, ty]
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)

    # pred_box : shape->(batch_size, grid, grid, anchors, (x, y, w, h))
    # pred_box : [tx, ty, tw, th]
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    # It will return two 2D tensors of shape(13,13) filled with [[0,...,12],[0,...,12],..
    # ..,[0,...,12]] and [[0,..,0], [1,..,1],..,[12,..,12]]
    # for more info refer to https://www.tensorflow.org/api_docs/python/tf/meshgrid
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))

    # gred: shape->(grid, grid, 1, 2)
    # tf.stack(grid, axis=-1) will stack two tensor of grid and output shape will be(13,13,2)
    # tf.expand_dim(.., axis=2) will and 1 dimention on axis=2
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)  # [bx, by]
    box_wh = anchors * tf.exp(box_wh)  # [bw, bh]

    # converting x,y,w,h into xmin,ymin,xmax,ymax
    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)  # x1y1x2y2
    return bbox, objectness, class_probs, pred_box


def detect(model, img_raw, size=416):
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, size)
    output = model.predict(img)

    class_names = {0: 'Tank', 1: 'Tank Cluster', 2: 'Floating Head Tank'}
    yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    gt_classes = ['Floating Head Tank', 'Tank Cluster', 'Tank']

    num_classes = len(class_names.values())
    # img_name = img_path.split('/')[-1]
    yolo_anchors = np.array([(15, 15), (24, 24), (27, 73), (36, 36), (52, 52),
                             (72, 26), (72, 66), (87, 94), (125, 122)],
                            np.float32) / 512

    boxes_0 = yolo_boxes(output[0], yolo_anchors[yolo_anchor_masks[0]], classes=num_classes)
    boxes_1 = yolo_boxes(output[1], yolo_anchors[yolo_anchor_masks[1]], classes=num_classes)
    boxes_2 = yolo_boxes(output[2], yolo_anchors[yolo_anchor_masks[2]], classes=num_classes)
    outputs = yolo_nms(
        (boxes_0[:3], boxes_1[:3], boxes_2[:3]),
        classes=num_classes
    )
    return outputs


def yolo_nms(outputs, classes):
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))
    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    yolo_max_boxes = 100
    yolo_iou_threshold = 0.5
    yolo_score_threshold = 0.4

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),  # (1, 10647, 1, 4)
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),  # (1, 10647, 3)
        max_output_size_per_class=yolo_max_boxes,
        max_total_size=yolo_max_boxes,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold
    )
    return boxes, scores, classes, valid_detections


def create_model(size=416, channels=3):
    tf.keras.backend.clear_session()
    pret_model = YoloV3(size, channels, classes=80)
    # load_darknet_weights(pret_model, 'Pretrained_Model/yolov3.weights')
    print('\nPretrained Weight Loaded')

    model = YoloV3(size, channels, classes=3)
    model.get_layer('yolo_darknet').set_weights(
        pret_model.get_layer('yolo_darknet').get_weights())
    print('Yolo DarkNet weight loaded')

    freeze_all(model.get_layer('yolo_darknet'))
    print('Frozen DarkNet layers')
    return model


def create_trained_model(weights_path, size=416):
    model = create_model(size)
    model.load_weights(weights_path)

    return model

def draw_outputs(img, outputs, class_names):
    convert_name = {
        'Tank': 'T',
        'Tank Cluster': 'TC',
        'Floating Head Tank': 'FHT'
    }

    boxes, scores, volumes, classes, nums = outputs
    volumes = [float(i) for i in volumes]
    scores = np.array(scores)
    boxes, scores, classes, nums = boxes[0], scores[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        color = [0,0,0]
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        color[int(classes[i])]=255
        img = cv2.rectangle(img, x1y1, x2y2, tuple(color), 2)
        img = cv2.putText(img, '{} {:.2f} V {:.2f}'.format(
            convert_name[class_names[int(classes[i])]],
            scores[i],
            volumes[i]),
            x1y1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2)
    return img

def make_prediction(MODEL, image, file_suffix, size=416, save_image=True):
    class_names = {0: 'Tank', 1: 'Tank Cluster', 2: 'Floating Head Tank'}
    gt_classes = ['Floating Head Tank', 'Tank', 'Tank Cluster']

    # 2. Collecting Predicted Bounding Boxes
    prediction = [[] for i in range(3)]

    boxes, scores, classes, nums = detect(MODEL, image, size)

    width = 512
    height = 512

    Bboxes, Classes = [], []
    for i in range(nums[0]):
        xmin, ymin, xmax, ymax = tuple(np.array(boxes[0][i]))

        xmin = int(round(xmin*width))
        ymin = int(round(ymin*height))
        xmax = int(round(xmax*width))
        ymax = int(round(ymax*height))

        class_name = str(class_names[int(classes[0][i])])
        #Format: ymin, xmin, ymax, xmax
        if(class_name == 'Floating Head Tank'):
            Bboxes.append([ymin, xmin, ymax, xmax])
            Classes.append(class_name)

    im = Image(pil2tensor(image, dtype=np.float32).div_(255))

    all_tanks = MultiTank(Bboxes, im)
    volumes = all_tanks.get_volumes()

    for i in range(nums[0]):
        score = float(scores[0][i])
        coor = np.array(boxes[0][i])
        volume = float(volumes[i])
        class_name = class_names[int(classes[0][i])]
        xmin, ymin, xmax, ymax = list(map(str, coor))
        bbox = xmin + " " + ymin + " " + xmax + " " + ymax
        prediction[gt_classes.index(class_name)].append(
            {"confidence": str(score),
             "volumes": str(volume),
             "file_id": str(class_name),
             "bbox": str(bbox)
             })

    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, volumes, classes, nums), class_names)
    if save_image:
        cv2.imwrite('app/1.jpg', img)

    retval, buffer = cv2.imencode(file_suffix, img)
    encoded_img = base64.b64encode(buffer)

    return {"data": prediction, "encoded_img": encoded_img}



if __name__ == "__main__":
    pass
