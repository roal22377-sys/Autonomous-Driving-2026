#!/usr/bin/env python

# Author: christoph.roesmann@tu-dortmund.de
# Modifier: Donghee Han, hdh7485@kaist.ac.kr

import rclpy
from rclpy.node import Node
import math

from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped
from collections import deque
import numpy as np
EPS = 1e-3
QUEUE_SIZE = 1

class KeyToAck(Node):
  def __init__(self):
    super().__init__('keyboard_to_ackermann_drive')

    self.speed_queue = deque(maxlen=QUEUE_SIZE)
    self.steering_queue = deque(maxlen=QUEUE_SIZE)
    self.speed_queue.extend(np.zeros(QUEUE_SIZE,))
    self.steering_queue.extend(np.zeros(QUEUE_SIZE,))

    self.pub = self.create_publisher(AckermannDriveStamped, 'teleop', 1)
    self.sub = self.create_subscription(
      Twist,
      'key_vel',
      self._process_key_vel,
      1
    )

    # self.pub_timer = self.create_timer(0.1, self._publish_teleop)

  def _process_key_vel(self, msg):
    self.speed_queue.append(msg.linear.x)
    if msg.angular.z >= EPS:
      self.steering_queue.append(0.6)
    elif msg.angular.z <= -EPS:
      self.steering_queue.append(-0.6)
    else:
      self.steering_queue.append(0.0)  

    msg = AckermannDriveStamped()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.drive.steering_angle = sum(self.steering_queue)/len(self.steering_queue)
    msg.drive.speed = sum(self.speed_queue)/len(self.speed_queue)
    self.pub.publish(msg)

  def _publish_teleop(self):
    msg = AckermannDriveStamped()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.drive.steering_angle = sum(self.steering_queue)/len(self.steering_queue)
    msg.drive.speed = sum(self.speed_queue)/len(self.speed_queue)
    self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    p = KeyToAck()
    rclpy.spin(p)

if __name__ == '__main__':
    main()
