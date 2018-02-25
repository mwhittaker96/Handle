import os
import numpy as np
import cv2
import sys
import tensorflow as tf
import argparse
import sys

from threading import Thread, Lock
import time

mutex = Lock()

classify_threshold = 65

global message
message = ''


labels = ['Heart: ', 'Thumb: ', 'Peace: ', 'E: ', 'C: ', 'I: ', 'A: ', 'O: ', 'U: ']
global percentages
percentages = [0,0,0,0,0,0,0,0,0]

"""
This will be run on the other thread
"""
def display_results():
    result = classify_image('hot_frame.jpg')
    global message
    max_value = sys.float_info.min
    indexFound = 0
    count = 0
    for value in result[1]:
        result[1][count] = value * 100
        if max_value < value:
            max_value = value
            indexFound = count
        count += 1


    if(result[1][indexFound] >= classify_threshold):
        message = str(result[0][indexFound] +": "+str(result[1][indexFound])+"%")
    else:
        message = ''

    global percentages
    percentages = list(result[1])

    mutex.release()




###############################################################################

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
    return graph

model_file = "/tmp/output_graph.pb"
graph = load_graph(model_file)



def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):

    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')

    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def classify_image(image_file_name):

    file_name = image_file_name
    model_file = "/tmp/output_graph.pb"
    label_file = "/tmp/output_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 128
    input_std = 128
    input_layer = "Mul"
    output_layer = 'final_result'

    t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0],
                        {input_operation.outputs[0]: t})

    results = np.squeeze(results)
    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    return [labels, results]


##############################################################################




#video streaming and processing
cap = cv2.VideoCapture(0)
hsvSkinThreshold = [0,48,80]
rgbSkinThreshold = [20, 255, 255]

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,400)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2



fontSmall              = cv2.FONT_HERSHEY_SIMPLEX
small                  = (10,400)
fontColorSmall         = (255,0,0)
pos = [
    (10, 40),
    (10, 80),
    (10, 120),
    (10, 160),
    (10, 200),
    (10, 240),
    (10, 280),
    (10, 320),
    (10, 360)
]

while(True):
        #capture frame-by-frame
        ret, frame = cap.read()
        blur = cv2.blur(frame, (3,3))
        hsvframe = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsvframe, np.array(hsvSkinThreshold), np.array(rgbSkinThreshold))

        cv2.putText(frame, message,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType)

        if not mutex.locked():
            mutex.acquire()
            cv2.imwrite('hot_frame.jpg', mask2)
            image_classify_thread = Thread(target=display_results)
            image_classify_thread.start()


        #display frame
        cv2.imshow('frame', frame)
        cv2.imshow('mask2', mask2)

        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        for i in range(0, len(labels)):
            cv2.putText(gray, (str(labels[i])+str(percentages[i])+"%"),
                pos[i],
                fontSmall,
                fontScale,
                fontColorSmall,
                lineType)

        cv2.imshow('results', gray)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#when done
cap.release()
cv2.destroyAllWindows()
