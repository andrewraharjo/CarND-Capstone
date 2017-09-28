#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray, TrafficLight
import numpy as np
import tf
import math
import std_msgs.msg
from std_msgs.msg import Bool, Float64, Int32
from scipy.interpolate import interp1d

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
DEBUG         = True
SPEED_LIMIT   = 6  # 10 mp/h
STOP_DIST     = 30 # wp count

class WaypointUpdater(object):
    def __init__(self):
        self.cur_pose = None
        self.base_waypoints = None
        self.next_waypoints = None
        self.is_signal_red = False
        self.prev_pose = None
        self.move_car = False
        self.f_sp = None

        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.is_signal_red_pub = rospy.Publisher('is_signal_red', Bool, queue_size=1)
        self.publish()
        rospy.spin()

    def publish(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if (self.cur_pose is not None) and (self.base_waypoints is not None):
                waypoints = self.base_waypoints.waypoints
                nb_waypoints = len(waypoints)
                next_waypoints = []

                next_wp_i = self.next_waypoint(self.cur_pose.pose, waypoints)
                to_red_wp_count = 0
                if self.is_signal_red == True:
                    # count waypoints between current and red
                    to_red_wp_count = self.wp_count(next_wp_i, self.red_wp_i+1)
                    if DEBUG:
                        rospy.logerr('to red: {}; cur wp: {}, red_wp: {}'.format(to_red_wp_count, next_wp_i, self.red_wp_i+1))

                if (self.is_signal_red == True) and (0 < to_red_wp_count < STOP_DIST):
                    if self.f_sp == None:
                        # sp_x = [waypoints[next_wp_i].pose.pose.position.x, waypoints[self.red_wp_i].pose.pose.position.x]
                        sp_wp_i = [to_red_wp_count, 0]
                        next_wp_velocity = self.get_waypoint_velocity(self.base_waypoints.waypoints[next_wp_i])
                        if next_wp_velocity > SPEED_LIMIT:
                            next_wp_velocity = SPEED_LIMIT
                        sp_v = [next_wp_velocity, 0.0]
                        self.f_sp = interp1d(sp_wp_i, sp_v)
                    for cur_wp_i in range(to_red_wp_count):
                        px = waypoints[next_wp_i].pose.pose.position.x
                        next_waypoints.append(waypoints[next_wp_i])
                        remaining_wp_to_red = to_red_wp_count - cur_wp_i
                        if DEBUG:
                            rospy.logerr('value of next waypoint %s red count %s cur_wp_i %s',remaining_wp_to_red, to_red_wp_count, cur_wp_i)
                        self.set_waypoint_velocity(next_waypoints, cur_wp_i, self.f_sp(remaining_wp_to_red))
                        next_wp_i = (next_wp_i + 1) % nb_waypoints
                    self.f_sp = None
                    self.set_waypoint_velocity(next_waypoints, cur_wp_i, 0.0)
                    if DEBUG:
                        rospy.loginfo("set velocity to 0")
                else:
                    if DEBUG:
                        rospy.logerr('normal drive')
                    for cur_wp_i in range(LOOKAHEAD_WPS):
                        next_wp_velocity = self.get_waypoint_velocity(self.base_waypoints.waypoints[next_wp_i])
                        if next_wp_velocity > SPEED_LIMIT:
                            next_wp_velocity = SPEED_LIMIT
                        next_waypoints.append(waypoints[next_wp_i])
                        self.set_waypoint_velocity(next_waypoints, cur_wp_i, SPEED_LIMIT)
                        next_wp_i = (next_wp_i + 1) % nb_waypoints
                    self.f_sp = None

                # next_waypoints = waypoints[next_wp_i:next_wp_i+LOOKAHEAD_WPS]

                if DEBUG:
                    rospy.logerr('# of wps published: {}'.format(len(next_waypoints)))

                # publish
                final_waypoints_msg = Lane()
                final_waypoints_msg.header.frame_id = '/world'
                final_waypoints_msg.header.stamp = rospy.Time(0)
                final_waypoints_msg.waypoints = next_waypoints
                self.final_waypoints_pub.publish(final_waypoints_msg)

                self.is_signal_red_pub.publish(self.is_signal_red)

            rate.sleep()

    def wp_count(self, wp_i_1, wp_i_2):
        if (wp_i_2 > wp_i_1):
            return wp_i_2 - wp_i_1
        else:
            return len(self.base_waypoints.waypoints) - wp_i_1 + wp_i_2

    def pose_cb(self, msg):
        self.cur_pose = msg

    def waypoints_cb(self, msg):
        self.base_waypoints = msg
        self.base_waypoints_sub.unregister()

    def traffic_cb(self, msg):
        if DEBUG:
            rospy.logerr('Got TL')
            rospy.logerr("message = %s", msg)

        if msg.data  >=  0:
            self.is_signal_red = True
            self.red_wp_i = msg.data
            self.move_car = False
            if DEBUG:
                rospy.loginfo("data %s signal  = true", msg.data)
        else:
            #self.prev_pose = self.cur_pose
            self.move_car = True
            self.red_wp_i = None
            self.is_signal_red = False

            #if self.prev_pose == None:
            #    self.prev_pose = self.cur_pose
            #else:
            #    if (self.prev_pose.pose.position.x == self.cur_pose.pose.position.x) and (self.prev_pose.pose.position.y == self.cur_pose.pose.position.y):
            #        self.is_signal_red = False
            #        self.red_wp_i = None
            #        self.move_car = True
            #        self.prev_pose = self.cur_pose
            #        if DEBUG:
            #            rospy.loginfo("velocity 0 hence changing the state")
            #    self.prev_pose = self.cur_pose

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, wp_idx, velocity):
        waypoints[wp_idx].twist.twist.linear.x = velocity


    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def closest_waypoint(self, pose, waypoints):
        closest_len = 100000
        closest_wp_i = 0
        dl = lambda a, b: (a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2
        for i in range(len(waypoints)):
            dist = dl(pose.position, waypoints[i].pose.pose.position)
            if (dist < closest_len):
                closest_len = dist
                closest_wp_i = i
        return closest_wp_i

    def next_waypoint(self, pose, waypoints):
        closest_wp_i = self.closest_waypoint(pose, waypoints)
        map_x = waypoints[closest_wp_i].pose.pose.position.x
        map_y = waypoints[closest_wp_i].pose.pose.position.y

        heading = math.atan2((map_y - pose.position.y), (map_x - pose.position.x))

        pose_quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        (_, _, yaw) = tf.transformations.euler_from_quaternion(pose_quaternion)
        angle = math.fabs(heading - yaw)

        if DEBUG:
            rospy.logerr('current_pose - x:{}, y:{},z:{}'.format(pose.position.x, pose.position.y, pose.position.z))
            rospy.logerr('ego yaw: {}'.format(yaw))
            rospy.logerr('heading: {}, angle: {}'.format(heading, angle))
            rospy.logerr('closest wp: {}; {}-{}'.format(closest_wp_i, waypoints[closest_wp_i].pose.pose.position.x, waypoints[closest_wp_i].pose.pose.position.y))

        if angle > (math.pi / 4):
            closest_wp_i = (closest_wp_i + 1) % len(waypoints)
            if DEBUG:
                rospy.logerr('corrected wp: {}; {}-{}'.format(closest_wp_i, waypoints[closest_wp_i].pose.pose.position.x, waypoints[closest_wp_i].pose.pose.position.y))

        if DEBUG:
            rospy.logerr(' ')

        return closest_wp_i


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
