#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
import tf.transformations
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import numpy as np
import yaml

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.car_pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.traffic_positions = self.get_given_traffic_lights()

        # self.last_traffic_light_state = TrafficLight.UNKNOWN
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.last_reported_traffic_light_id = None
        self.last_reported_traffic_light_time = None

        self.traffic_lights = None
        self.image = None

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.bridge = CvBridge()

        self.experiment_environment = rospy.get_param('/experiment_environment', "site")
        self.light_classifier = TLClassifier(self.experiment_environment)
        # self.light_classifier = TLClassifierCV()

        self.listener = tf.TransformListener()

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)

        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1)
        rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        self.upcoming_stop_light_pub = rospy.Publisher(
            '/upcoming_stop_light_position', geometry_msgs.msg.Point, queue_size=1)
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.image_pub = rospy.Publisher('/camera/my_image', Image, queue_size=1)

        rospy.spin()

    def get_given_traffic_lights(self):
        """
        Return given traffic light positions
        :return: TrafficLightArray
        """
        traffic_lights = TrafficLightArray()

        traffic_light_list = []

        # tl_height = rospy.get_param("/tl_height")
        config_string = rospy.get_param("/traffic_light_config")
        traffic_light_positions = yaml.load(config_string)["light_positions"]

        for traffic_light_index, traffic_light_position in enumerate(traffic_light_positions):
            traffic_light = TrafficLight()

            traffic_light.pose.pose.position.x = traffic_light_position[0]
            traffic_light.pose.pose.position.y = traffic_light_position[1]
            # traffic_light.pose.pose.position.z = tl_height
            traffic_light.state = TrafficLight.UNKNOWN
            traffic_light_list.append(traffic_light)

            traffic_lights.lights = traffic_light_list

        return traffic_lights

    def get_waypoints_matrix(self, waypoints):
        """
        Converts waypoints listt to numpy matrix
        :param waypoints: list of styx_msgs.msg.Waypoint instances
        :return: 2D numpy array
        """

        waypoints_matrix = np.zeros(shape=(len(waypoints), 2), dtype=np.float32)

        for index, waypoint in enumerate(waypoints):
            waypoints_matrix[index, 0] = waypoint.pose.pose.position.x
            waypoints_matrix[index, 1] = waypoint.pose.pose.position.y

        return waypoints_matrix

    def pose_cb(self, msg):
        self.pose = msg
        arguments = [self.traffic_positions, self.car_pose, self.waypoints, self.image]
        are_arguments_available = all([x is not None for x in arguments])
        if are_arguments_available:

            # Get closest traffic light
            traffic_light = self.get_closest_traffic_light_ahead_of_car(
                self.traffic_positions.lights, self.car_pose.position, self.waypoints)

            # These values seem so be wrong - Udacity keeps on putting in config different values that what camera
            # actually publishes.
            # image_width = self.config["camera_info"]["image_width"]
            # image_height = self.config["camera_info"]["image_height"]

            # Therefore simply check image size
            self.camera_image = self.image
            self.camera_image.encoding = "rgb8"
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

            image_height = cv_image.shape[0]
            image_width = cv_image.shape[1]

            x, y = self.project_to_image_plane(
                traffic_light.pose.pose.position, self.car_pose, image_width, image_height)

            simulator_traffic_light_in_view = 0 < x < image_width and 0 < y < image_height
            # As of this writing, site camera mapping is broken (thanks, Udacity...), so we will just process all
            # images on site
            site_traffic_light_in_view = True

            traffic_light_in_view = simulator_traffic_light_in_view if self.experiment_environment == "simulator" \
                else site_traffic_light_in_view

            # Only try to classify image if traffic light is within it
            if traffic_light_in_view:

                traffic_light_state = self.light_classifier.get_classification(cv_image)

                # lights_map = {0: "Red", 1: "Yellow", 2: "Green"}
                # rospy.logwarn("Detected light: {}".format(lights_map.get(traffic_light_state, "Other")))

                cv2.circle(cv_image, (x, y), radius=50, color=(255, 0, 0), thickness=12)
                marked_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
                self.image_pub.publish(marked_image)

                if traffic_light_state == TrafficLight.RED or traffic_light == TrafficLight.YELLOW:
                    self.upcoming_stop_light_pub.publish(traffic_light.pose.pose.position)

    def get_closest_waypoint_index(self, position, waypoints_matrix):
        """
        Given a pose and waypoints list, return index of waypoint closest to pose
        :param position: geometry_msgs.msgs.Position instance
        :param waypoints_matrix: numpy matrix with waypoints coordinates
        :return: integer index
        """

        x_distances = waypoints_matrix[:, 0] - position.x
        y_distances = waypoints_matrix[:, 1] - position.y

        squared_distances = x_distances ** 2 + y_distances ** 2
        return np.argmin(squared_distances)

    def waypoints_cb(self, waypoints):
        # self.waypoints = waypoints
        self.current_waypoints = waypoints.waypoints
        self.base_waypoints_sub.unregister()

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        return 0

    def get_road_distance(self, waypoints):
        """
        Get road distance covered when following waypoints
        :param waypoints: list of styx_msgs.msg.Waypoint instances
        :return: float
        """

        total_distance = 0.0

        for index in range(1, len(waypoints)):

            x_distance = waypoints[index].pose.pose.position.x - waypoints[index - 1].pose.pose.position.x
            y_distance = waypoints[index].pose.pose.position.y - waypoints[index - 1].pose.pose.position.y

            distance = np.sqrt((x_distance**2) + (y_distance**2))

            total_distance += distance

        return total_distance

    def project_to_image_plane(self, point_in_world, car_pose, image_width, image_height):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        #TODO Use tranform and rotation to calculate 2D position of light in image
        world_coordinates_point = np.array(
            [point_in_world.x, point_in_world.y, point_in_world.z], dtype=np.float32).reshape(3, 1)

        car_position = np.array([car_pose.position.x, car_pose.position.y, car_pose.position.z],
                                dtype=np.float32).reshape(3, 1)
        camera_offset = np.array([1.0, 0, 1.2], dtype=np.float32).reshape(3, 1)
        # translation_vector = np.array(trans, dtype=np.float32).reshape(3, 1)
        translation_vector = car_position + camera_offset

        # Move point to camera origin
        world_coordinates_point_shifted_to_camera_coordinates = world_coordinates_point - translation_vector

        homogenous_vector = np.ones(shape=(4, 1), dtype=np.float32)
        homogenous_vector[:3] = world_coordinates_point_shifted_to_camera_coordinates

        quaternion = np.array([
            car_pose.orientation.x, car_pose.orientation.y, car_pose.orientation.z, car_pose.orientation.w],
            dtype=np.float32)

        euler_angles = tf.transformations.euler_from_quaternion(quaternion)
        rotation_matrix = tf.transformations.euler_matrix(*euler_angles)

        point_in_camera_coordinates = np.dot(rotation_matrix, homogenous_vector)

        x = (fx * point_in_camera_coordinates[0] * point_in_camera_coordinates[2]) + (image_width / 2)
        y = (fy * point_in_camera_coordinates[1] * point_in_camera_coordinates[2]) + (image_height / 2)

        return (x, y)

    def get_closest_traffic_light_ahead_of_car(self, traffic_lights, car_position, waypoints):
        """
        Given list of traffic lights, car position and waypoints, return closest traffic light
        ahead of the car. This function wraps around the track, so that if car is at the end of the track,
        and closest traffic light is at track's beginning, it will be correctly reported
        :param traffic_lights: list of styx_msgs.msg.TrafficLight instances
        :param car_position: geometry_msgs.msgs.Pose instance
        :param waypoints: list of styx_msgs.msg.Waypoint instances
        :return: styx_msgs.msg.TrafficLight instance
        """

        waypoints_matrix = self.get_waypoints_matrix(waypoints)
        car_index = self.get_closest_waypoint_index(car_position, waypoints_matrix)

        # Arrange track waypoints so they start at car position
        waypoints_ahead = waypoints[car_index:] + waypoints[:car_index]
        waypoints_ahead_matrix = self.get_waypoints_matrix(waypoints_ahead)

        distances = []

        for traffic_light in traffic_lights:

            waypoint_index = self.get_closest_waypoint_index(traffic_light.pose.pose.position, waypoints_ahead_matrix)

            distance = self.get_road_distance(waypoints_ahead[:waypoint_index])
            distances.append(distance)

        closest_traffic_light_index = np.argmin(distances)

        return traffic_lights[closest_traffic_light_index]

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        x, y = self.project_to_image_plane(light.pose.pose.position)

        #TODO use light location to zoom in on traffic light in image

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        light_positions = self.config['light_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)

        #TODO find the closest visible traffic light (if one exists)

        if light:
            state = self.get_light_state(light)
            return light_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
