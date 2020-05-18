import cv2
import numpy as np
import math
import random
import os
from matplotlib import pyplot as plt
from PIL import Image
from keras.models import load_model
import tensorflow as tf
import chardetection
import boto3
from botocore.exceptions import NoCredentialsError
from subprocess import call
import datetime
import reverse_geocoder as rg 
import pymongo
import d

ACCESS_KEY = 'AKIAJTZBE5AEHK23XTZQ'
SECRET_KEY = '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'
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["license"]
mycol = mydb["vehicalsdata"]


def load_image_into_numpy_array(image):
    '''
    This function converts the image into the numpy array for prediction
    '''

    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# Detection
def run_inference_for_single_image(image, graph):
    '''
    This function will detect the number plate in the image using the SSD trained model

    @input: Image to process, tensorflow graph
    @output: List containing all the information
    '''

    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()

            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            # Do the preprocessing for detection mask
            # The following processing is only for single image
            if 'detection_masks' in tensor_dict:
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                
                # Follow the convention by adding back the batch dimension
                
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
                
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                    feed_dict={image_tensor: np.expand_dims(image, 0)})


            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict

# Image Segmentation
def CapturePlatesFromImage(image):
    # Loading the model
    MODEL_NAME = 'plate_detector' # This is the model we will use here.
    
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb' # Path to save the downloaded model.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    
    
    # Loading The image
    img_Original = cv2.imread(image)
    img = Image.open(image)    
    image_np = load_image_into_numpy_array(img)
    
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    
    d = output_dict['detection_boxes'][0].tolist()
    
    (ymin, xmin, ymax, xmax) = d
    
    im_width, im_height = img.size
    
    (left, right, top, bottom) = (xmin*im_width, xmax*im_width,ymin*im_height, ymax*im_height)
    
    imgResult = img_Original[math.floor(top):math.ceil(bottom),math.floor(left):math.ceil(right)]
    
    return imgResult

def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

for i in range(1,6):    
    img_plate = CapturePlatesFromImage("img"+str(i)+".jpg")
    ptime = str(datetime.datetime.now()).split('.')[0]
    cv2.imwrite("exampleabc.png",img_plate)
    uploaded = upload_to_aws('exampleabc.png', 'alprdata', 'img2.png')
    number_plate=chardetection.imagetotext()
    locations=d.locs()
    dbrow={"Numberplate":number_plate,"Rtime":ptime,"Location":locations}
    mycol.insert_one(dbrow)
    print("The detected numberplate is " + number_plate)

