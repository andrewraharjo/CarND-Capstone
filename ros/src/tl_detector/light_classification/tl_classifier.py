from styx_msgs.msg import TrafficLight
from PIL import Image
import numpy as np
import rospkg
import cv2
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Input, Dense
import tensorflow as tf
import json
import rospy

ACTIVATE_SIM = True

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

        #self.lower = np.array([150, 100, 150])
        #self.upper = np.array([180, 255, 255])

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

        ## Load Simulator model here
        if ACTIVATE_SIM:
            model_path = path_v + \
                     '/../tl_detector/light_classification/model/ssd_sim.pb'
        else:
            model_path = path_v + \
                     '/../tl_detector/light_classification/model/ssd_real.pb'

        label_path = path_v + \
                     '/../tl_detector/light_classification/model/label_map.pbtxt'
        self.d_graph = tf.Graph()
        with self.d_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.d_graph)

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

    def get_traffic_classification(self, image):
        image_tensor = self.d_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = self.d_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = self.d_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.d_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.d_graph.get_tensor_by_name('num_detections:0')

        # convert from cv2 to PIL
        cv2_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(cv2_rgb)

        (im_width, im_height) = image.size
        image_np = np.array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        boxes_new = np.squeeze(boxes)
        scores_new = np.squeeze(scores)
        min_score_thresh = .5
        image_np_new = image_np.copy()

        found_box = False  # limit to 1 traffic light

        boxes_new_shape = boxes_new.shape

        for i in range(boxes_new_shape[0]):
            if scores_new[i] > min_score_thresh and not found_box:

                found_box = True

                ymin = boxes_new[i, 0]
                xmin = boxes_new[i, 1]
                ymax = boxes_new[i, 2]
                xmax = boxes_new[i, 3]
                (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width),
                                              int(ymin * im_height), int(ymax * im_height))
                img_light_np = image_np_new[top:bottom, left:right]

                img_light = Image.fromarray(img_light_np)
                img_light_gray = img_light.convert('L')

                img_light_bw_np = np.asarray(img_light_gray).copy()

                if ACTIVATE_SIM:
                    img_light_bw_np[img_light_bw_np < 100] = 0  # Black
                    img_light_bw_np[img_light_bw_np >= 100] = 255  # White
                else:
                    img_light_bw_np[img_light_bw_np < 190] = 0  # Black
                    img_light_bw_np[img_light_bw_np >= 190] = 255  # White

                img_light_bw = Image.fromarray(img_light_bw_np)
                w, h = img_light_bw.size
                h_w_ratio = h / (w * 1.0)

                light_colors = []  # red, yellow, green
                single_light_pixel_count = int(h * w / 3)

                if ACTIVATE_SIM:
                    nzCountRed = np.count_nonzero(np.array(img_light_bw)[int(h / 10):int(h / 3), :]) / (
                        single_light_pixel_count * 1.0)
                    nzCountYellow = np.count_nonzero(np.array(img_light_bw)[int(h / 3):int(h * 2 / 3), :]) / (
                        single_light_pixel_count * 1.0)
                    nzCountGreen = np.count_nonzero(np.array(img_light_bw)[int(h * 2 / 3):int(h * 9 / 10), :]) / (
                        single_light_pixel_count * 1.0)
                else:
                    nzCountRed = np.count_nonzero(np.array(img_light_bw)[0:int(h / 3), :]) / (
                    single_light_pixel_count * 1.0)
                    nzCountYellow = np.count_nonzero(np.array(img_light_bw)[int(h / 3):int(h * 2 / 3), :]) / (
                        single_light_pixel_count * 1.0)
                    nzCountGreen = np.count_nonzero(np.array(img_light_bw)[int(h * 2 / 3):h, :]) / (
                        single_light_pixel_count * 1.0)

                light_colors.extend([nzCountRed, nzCountYellow, nzCountGreen])

                max_i = max(enumerate(light_colors), key=lambda x: x[1])[0]

                if ACTIVATE_SIM:
                    if light_colors[max_i] > 0.05:
                        if max_i == 0:
                            return TrafficLight.RED
                        elif max_i == 1:
                            return TrafficLight.RED
                        elif max_i == 2:
                            return TrafficLight.GREEN
                else:
                    if (0.10 < light_colors[max_i] < 0.97) and (h_w_ratio > 1.75):
                        if max_i == 0:
                            return TrafficLight.RED
                        elif max_i == 1:
                            return TrafficLight.RED
                        elif max_i == 2:
                            return TrafficLight.GREEN

        return TrafficLight.UNKNOWN
