from styx_msgs.msg import TrafficLight
import numpy as np
import rospkg
import cv2
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Input, Dense
import tensorflow as tf
import json
import rospy


class TLClassifier(object):
    def __init__(self):
        # Lower and Upper threshold for color extraction
        self.model = None
        self.create_model()
        self.load_ssd_model()
        if not self.model:
            rospy.logerr("Failed to traffic light classifier model")

        self.colors = [TrafficLight.RED,
                       TrafficLight.YELLOW,
                       TrafficLight.GREEN,
                       TrafficLight.UNKNOWN]

        self.num_pixels = 950

        self.lower = np.array([150, 100, 150])
        self.upper = np.array([180, 255, 255])

        # Define red pixels in hsv color space
        self.lower_red_1 = np.array([0, 70, 50], dtype="uint8")
        self.upper_red_1 = np.array([10, 255, 255], dtype="uint8")

        self.lower_red_2 = np.array([170, 70, 50], dtype="uint8")
        self.upper_red_2 = np.array([180, 255, 255], dtype="uint8")

    def create_model(self):
        self.model =  Sequential()
        self.model.add(Dense(200, activation='relu', input_shape=(30000,)))
        self.model.add(Dense(3, activation='softmax'))

        rospack = rospkg.RosPack()
        path_v = rospack.get_path('styx')
        model_file = path_v+ \
               '/../tl_detector/light_classification/model/tl-model-sim.h5'
        self.model.load_weights(model_file)
        self.graph = tf.get_default_graph()

    def load_ssd_model(self):
        rospack = rospkg.RosPack()
        path_v = rospack.get_path('styx')
        PATH_TO_CKPT = path_v + \
                     '/../tl_detector/light_classification/model/ssd_sim.pb'
        PATH_TO_LABELS = path_v + \
                     '/../tl_detector/light_classification/model/tl_label_map.pbtxt'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, image):
        """
        Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_new = image[100:700, 50:550]
        #dim = (100, 26)
        r = 100.0 / image_new.shape[1]
        dim = (100, int(image_new.shape[0] * r))
        resized = cv2.resize(image_new, dim)
        image_data = np.array([resized.flatten().tolist()])
        image_data /= 255
        with self.graph.as_default():
             classes = self.model.predict(image_data, batch_size=1)
             return self.colors[np.argmax(classes[0])]
        return TrafficLight.UNKNOWN
