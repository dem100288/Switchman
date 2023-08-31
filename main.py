DEBUG = False
TRUE_ROBOT = False
MAX_DISTANCE_CENTER = 50
MAX_FRAME_SKIPPING = 5
FRAME_HEIGHT = 240
FRAME_WIDTH = 320
CENTER_FRAME_X = FRAME_WIDTH // 2
CENTER_FRAME_Y = FRAME_HEIGHT // 2
NEAR_COORD_DIST = 5
NEAR_COLOR = 50
NEAR_NUM = 5
PER_ARROW_CENTER_CIRCLE_CENTER = 0.35
ARROW_COLOR = (60, 150, 210)
CIRCLE_COLOR = (30, 30, 120)
MOVE_TO_POINT_SPEED = 0.2
MOVE_TO_SIDE_SPEED = 0.15
MOVE_TO_INC_DEPTH_SPEED = 0.05
MOVE_SPEED = 0.15

if not DEBUG:
    import pymurapi as mur
import time
import cv2 as cv
import numpy as np
import math

sdf = cv.HOUGH_GRADIENT


class auv_test:

    def set_motor_power(self, index_motor, power):
        pass

    def get_depth(self):
        return 0

    def get_yaw(self):
        return 0

    def get_image_front(self):
        img = cv.imread('arrow.png', cv.IMREAD_COLOR)
        # cv.imshow('front', img)
        return img

    def get_image_bottom(self):
        img = cv.imread('src.png', cv.IMREAD_COLOR)
        # cv.imshow('bottom', img)
        return img


auv = auv_test() if DEBUG else mur.mur_init()


def angle_between_vect(vect1, vect2, deg=True):
    v1 = vect1 / np.linalg.norm(vect1)
    v2 = vect2 / np.linalg.norm(vect2)
    rad = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    if deg:
        return np.degrees([rad.real])[0]
    return rad


def middle_point(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)


def dist_between_points(p1, p2):
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


def dist_between_points_y(p1, p2):
    return abs(p1[1] - p2[1])


def frame_coord_to_true_cord(coord):
    return coord[0] - CENTER_FRAME_X, (coord[1] - CENTER_FRAME_Y) * -1


def near_coord(coord, target_cord):
    return dist_between_points(coord, target_cord) <= NEAR_COORD_DIST


def near_coord_y(coord, target_cord):
    return dist_between_points_y(coord, target_cord) <= NEAR_COORD_DIST


def near_color(color, target_color):
    return ((abs(color[0] - target_color[0]) <= NEAR_COLOR)
            and (abs(color[1] - target_color[1]) <= NEAR_COLOR)
            and (abs(color[2] - target_color[2]) <= NEAR_COLOR))


def get_normal_curse(course):
    if -45 < course < 45:
        return 0
    elif 45 < course < 135:
        return 90
    elif -135 < course < -45:
        return -90
    else:
        return 180


def near_num(num, target_num, delta):
    return (abs(num - target_num) <= delta)


def point_in_conture(contur, point):
    point = (int(point[0]), int(point[1]))
    res = cv.pointPolygonTest(contur, point, False)
    if res > 0:
        return point
    point = (int(point[0]) + 1, int(point[1]))
    res = cv.pointPolygonTest(contur, point, False)
    if res > 0:
        return point
    point = (int(point[0]) - 1, int(point[1]))
    res = cv.pointPolygonTest(contur, point, False)
    if res > 0:
        return point
    point = (int(point[0]), int(point[1]) + 1)
    res = cv.pointPolygonTest(contur, point, False)
    if res > 0:
        return point
    point = (int(point[0]), int(point[1]) - 1)
    res = cv.pointPolygonTest(contur, point, False)
    if res > 0:
        return point
    return None


def color_in_conture(contur, point, src):
    point = (int(point[0]) + 1, int(point[1]))
    res = cv.pointPolygonTest(contur, point, False)
    if res > 0:
        i = 1
        c = src[point[1]][point[0]]
        color = [int(c[0]), int(c[1]), int(c[2])]
        while True:
            point = (int(point[0]) + i, int(point[1]))
            res = cv.pointPolygonTest(contur, point, False)
            if res > 0:
                c = src[point[1]][point[0]]
                color[0] += c[0]
                color[1] += c[1]
                color[2] += c[2]
            else:
                break
            i += 1
        color[0] = color[0] // i
        color[1] = color[1] // i
        color[2] = color[2] // i
        return color
    point = (int(point[0]) - 1, int(point[1]))
    res = cv.pointPolygonTest(contur, point, False)
    if res > 0:
        i = 1
        c = src[point[1]][point[0]]
        color = [int(c[0]), int(c[1]), int(c[2])]
        while True:
            point = (int(point[0]) - i, int(point[1]))
            res = cv.pointPolygonTest(contur, point, False)
            if res > 0:
                c = src[point[1]][point[0]]
                color[0] += c[0]
                color[1] += c[1]
                color[2] += c[2]
            else:
                break
            i += 1
        color[0] = color[0] // i
        color[1] = color[1] // i
        color[2] = color[2] // i
        return color
    point = (int(point[0]), int(point[1]) + 1)
    res = cv.pointPolygonTest(contur, point, False)
    if res > 0:
        i = 1
        c = src[point[1]][point[0]]
        color = [int(c[0]), int(c[1]), int(c[2])]
        while True:
            point = (int(point[0]), int(point[1]) + i)
            res = cv.pointPolygonTest(contur, point, False)
            if res > 0:
                c = src[point[1]][point[0]]
                color[0] += c[0]
                color[1] += c[1]
                color[2] += c[2]
            else:
                break
            i += 1
        color[0] = color[0] // i
        color[1] = color[1] // i
        color[2] = color[2] // i
        return color
    point = (int(point[0]), int(point[1]) - 1)
    res = cv.pointPolygonTest(contur, point, False)
    if res > 0:
        i = 1
        c = src[point[1]][point[0]]
        color = [int(c[0]), int(c[1]), int(c[2])]
        while True:
            point = (int(point[0]), int(point[1]) - i)
            res = cv.pointPolygonTest(contur, point, False)
            if res > 0:
                c = src[point[1]][point[0]]
                color[0] += c[0]
                color[1] += c[1]
                color[2] += c[2]
            else:
                break
            i += 1
        color[0] = color[0] // i
        color[1] = color[1] // i
        color[2] = color[2] // i
        return color
    return None


def sign_num(num):
    return -1 if num < 0 else (0 if num == 0 else 1)


def clip_num(num, min_value, max_value):
    return min(max(num, min_value), max_value)


def course_to_angle(course):
    return abs(course % 360)


def angle_to_course(angle):
    return 180 - angle


def dif_num(num1, num2):
    return num1 - num2


def dif_angle_360(target_angle, cur_angle):
    target_angle = (target_angle - cur_angle) % 360
    return target_angle if target_angle <= 180 else target_angle - 360


def dif_angle_180(target_angle, cur_angle):
    target_angle = (target_angle - cur_angle)
    return target_angle if abs(target_angle) <= 180 else target_angle - 360


def checkidentity(obj, collect):
    for obj_collect in collect:
        if (dist_between_points(obj.center, obj_collect.center) <= MAX_DISTANCE_CENTER
                and type(obj) == type(obj_collect)):
            return obj_collect
    return None


class ChangeValueLogger:

    def __init__(self, func_dif=dif_num, koef=1):
        self.koef = koef
        self.velocity = 0
        self.velocity_without_time = 0
        self.acceleration = 0
        self.acceleration_time = 0
        self.value = 0
        self.min_velocity = 0.01
        self.min_acceleration = 0.01
        self.func_dif = func_dif

    def isRelax(self):
        return self.velocity < self.min_velocity and self.acceleration < self.min_acceleration

    def updateValue(self, value, elapsed):
        prev_value = self.value
        self.value = value
        self.velocity_without_time = self.func_dif(value, prev_value)
        prev_velocity = self.velocity
        self.velocity = self.velocity_without_time / elapsed if elapsed > 0 else 0
        self.acceleration_time = self.velocity - prev_velocity
        self.acceleration = self.acceleration_time / elapsed if elapsed > 0 else 0

    def getVelosityWithKoef(self):
        return self.velocity * self.koef


class DetectedObject:

    def __init__(self, center, array, color=[0, 0, 0]):
        self.frame_coord = center
        self.center = frame_coord_to_true_cord(center)
        self.array = array
        self.in_frame = True
        self.not_frame_count = 0
        self.color = color
        self.number = 0

    def set_color(self, color):
        self.color = color

    def update_property(self, new_obj):
        self.center = new_obj.center
        self.frame_coord = new_obj.frame_coord
        self.array = new_obj.array
        self.in_frame = True
        self.not_frame_count = 0

    def clear_in_frame(self):
        self.in_frame = False

    def delete_frame(self):
        if not self.in_frame:
            self.not_frame_count += 1
        return self.not_frame_count >= MAX_FRAME_SKIPPING


class CircleObject(DetectedObject):

    def __init__(self, center, radius, color=[0, 0, 0]):
        self.radius = radius
        super().__init__(center, math.pi * math.pow(radius, 2), color)

    def update_property(self, new_obj):
        super().update_property(new_obj)
        self.radius = new_obj.radius


class ArrowObject(DetectedObject):

    def __init__(self, center, direction, color=[0, 0, 0]):
        super().__init__(center, 0, color)
        self.direction = frame_coord_to_true_cord(direction)
        self.vector = (direction[0] - center[0], (direction[1] - center[1]) * -1)
        self.angle = get_normal_curse(angle_between_vect(self.vector, (0, 1))) * sign_num(self.vector[0])

    def update_property(self, new_obj):
        super().update_property(new_obj)
        self.direction = new_obj.direction
        self.vector = new_obj.vector
        self.angle = new_obj.angle


class FindedContour:

    def __init__(self, countour, frame_object, with_defects=False):
        self.frame_object = frame_object
        self.contour = countour
        self.hull = None
        self.around_circle = None
        self.defects = None
        self.contour_poly = None
        self.find_around_circle()
        self.find_hull()
        if with_defects: self.find_defects()

    def find_hull(self):
        if self.hull is None:
            self.hull = cv.convexHull(self.contour, returnPoints=False)

    def find_contour_poly(self):
        if self.contour_poly is None:
            self.contour_poly = cv.approxPolyDP(self.contour, 3, True)

    def find_around_circle(self):
        if self.around_circle is None:
            self.find_contour_poly()
            center, radius = cv.minEnclosingCircle(self.contour_poly)
            self.around_circle = CircleObject(center, radius)

    def find_defects(self):
        if self.defects is None:
            self.find_hull()
            try:
                self.defects = cv.convexityDefects(self.contour, self.hull)
            except Exception:
                self.defects = None


class FrameObject:

    def __init__(self, frame):
        self.src = frame  # cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.grayscale = None
        self.canny = None
        self.circles = None
        self.contours = None

    def CalcGrayscale(self):
        if self.grayscale is None:
            self.grayscale = cv.cvtColor(self.src, cv.COLOR_BGR2GRAY)

    def CalcCanny(self):
        self.CalcGrayscale()
        if self.canny is None:
            self.canny = cv.Canny(self.grayscale, 50, 200, None, 3)

    def FindContours(self):
        if self.contours is None:
            self.CalcCanny()
            contours, _ = cv.findContours(self.canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            contours = list(map(lambda x: FindedContour(x, self), contours))
            unique_contours = []
            for cont in contours:
                flag = False
                for ucont in unique_contours:
                    if (cont.around_circle is not None and
                            near_coord(cont.around_circle.center, ucont.around_circle.center) and
                            near_num(cont.around_circle.radius, ucont.around_circle.radius, NEAR_NUM)):
                        flag = True
                if not flag:
                    unique_contours.append(cont)
                    cont.find_defects()
            self.contours = unique_contours

    def FindCircles(self):
        if self.circles is None:
            self.CalcGrayscale()
            circles = cv.HoughCircles(self.grayscale, cv.HOUGH_GRADIENT, 1, 20, param1=10, param2=40, minRadius=10,
                                      maxRadius=0)
            if circles is None:
                self.circles = []
            else:
                self.circles = list(
                    map(lambda x: CircleObject((x[0], x[1]), x[2]), np.round(circles[0, :]).astype("int")))
                for circle in self.circles:
                    a = circle.radius // 4
                    x0 = clip_num(circle.frame_coord[0] - a, 0, FRAME_WIDTH)
                    y0 = clip_num(circle.frame_coord[1] - a, 0, FRAME_HEIGHT)
                    color = [0, 0, 0]
                    i = 0
                    for x in range(x0, clip_num(circle.frame_coord[0] + a, 0, FRAME_WIDTH)):
                        for y in range(y0, clip_num(circle.frame_coord[1] + a, 0, FRAME_HEIGHT)):
                            c = self.src[y, x]
                            color[0] += c[0]
                            color[1] += c[1]
                            color[2] += c[2]
                            i += 1
                    if i > 0:
                        color[0] = color[0] // i
                        color[1] = color[1] // i
                        color[2] = color[2] // i
                    circle.set_color(color)


class Stage:

    def __init__(self, task):
        self.task = task
        self.end = False

    def init(self):
        pass

    def proc_stage(self):
        pass

    def processing(self):
        self.proc_stage()
        return not self.end

    def end_stage(self):
        self.task.move = 0
        self.end = True


class ObjectDetector:

    def __init__(self, name):
        self.name = name
        self.objects = []
        self.last_number = 0

    def objects_after_number(self, number):
        return list(filter(lambda x: x.number > number, self.objects))

    def find_objects(self, frame_object):
        return []

    def detect(self, frame_object):
        new_objects = self.find_objects(frame_object)
        for object in self.objects:
            object.clear_in_frame()
        for new_object in new_objects:
            object_in_collect = checkidentity(new_object, self.objects)
            if object_in_collect:
                object_in_collect.update_property(new_object)
            else:
                self.last_number += 1
                new_object.number = self.last_number
                self.objects.append(new_object)
        for i in range(len(self.objects) - 1, -1, -1):
            if self.objects[i].delete_frame():
                del self.objects[i]


class RedCirclesDetector(ObjectDetector):

    def find_objects(self, frame_object):
        global circle_num
        frame_object.FindCircles()
        if isinstance(frame_object.circles, list):
            cir = list(filter(lambda x: near_color(x.color, CIRCLE_COLOR), frame_object.circles))
            for c in cir:
                cv.circle(frame_object.src, (c.frame_coord[0], c.frame_coord[1]), c.radius, (0, 255, 0), 2)
            return list(filter(lambda x: near_color(x.color, CIRCLE_COLOR), frame_object.circles))
        else:
            return []


class ArrowDetector(ObjectDetector):

    def find_objects(self, frame_object):
        frame_object.FindContours()
        arrows = []
        if not isinstance(frame_object.contours, list): return arrows
        for contour in frame_object.contours:
            if contour.defects is not None:
                for j in range(contour.defects.shape[0]):
                    s, e, f, d = contour.defects[j, 0]
                    start = tuple(contour.contour[s][0])
                    end = tuple(contour.contour[e][0])
                    far = tuple(contour.contour[f][0])
                    line_center = middle_point(start, end)
                    dist = dist_between_points(line_center, contour.around_circle.frame_coord)
                    color = color_in_conture(contour.contour, far, frame_object.src)
                    if color is None:
                        continue
                    if dist < (contour.around_circle.radius * PER_ARROW_CENTER_CIRCLE_CENTER) and near_color(color,
                                                                                                             ARROW_COLOR):
                        arrows.append(ArrowObject(middle_point(line_center, far), far, ARROW_COLOR))
        return arrows


class Task:

    def __init__(self, robot):
        self.course = 0
        self.move = 0
        self.target_depth = 3.2
        self.stages = []
        self.end = False
        self.current_stage = -1
        self.robot = robot
        self.bottom_detectors = []
        self.forward_detectors = []

    def move_to_point(self, point_3d, dist=0):
        if len(point_3d) == 2:
            point_3d = (point_3d[0], point_3d[1], 0)
        y_sign = sign_num(point_3d[1])
        z_sign = sign_num(point_3d[2])
        self.target_depth += MOVE_TO_INC_DEPTH_SPEED * z_sign * -1
        self.move = (MOVE_TO_POINT_SPEED + dist * 1 / CENTER_FRAME_Y) * y_sign
        if near_coord(point_3d[:2], (0, 0)):
            self.clear_motor_koef()
        else:
            self.clear_motor_koef()
            x_sign = sign_num(point_3d[0])
            koef_sign = y_sign * -1
            if x_sign < 0:
                self.robot.set_motor_koef(MOVE_TO_SIDE_SPEED * koef_sign, (MOVE_TO_SIDE_SPEED / 2) * koef_sign)
            elif x_sign > 0:
                self.robot.set_motor_koef((MOVE_TO_SIDE_SPEED / 2) * koef_sign, MOVE_TO_SIDE_SPEED * koef_sign)

    def move_to_point_simple(self, point_3d, dist=0):
        if len(point_3d) == 2:
            point_3d = (point_3d[0], point_3d[1], 0)
        y_sign = sign_num(point_3d[1])
        z_sign = sign_num(point_3d[2])
        self.target_depth += MOVE_TO_INC_DEPTH_SPEED * z_sign * -1
        self.move = clip_num((dist * 0.5 / CENTER_FRAME_Y) * y_sign, -MOVE_TO_POINT_SPEED, MOVE_TO_POINT_SPEED)

    def add_stage(self, stage):
        self.stages.append(stage)
        if len(self.stages) == 1:
            self.current_stage = 0
            self.stages[self.current_stage].init()

    def add_detector_front(self, detector):
        self.forward_detectors.append(detector)

    def add_detector_bottom(self, detector):
        self.bottom_detectors.append(detector)

    def del_detector(self, detector):
        try:
            self.bottom_detectors.remove(detector)
        except Exception:
            pass
        try:
            self.forward_detectors.remove(detector)
        except Exception:
            pass

    def del_detector_name(self, name_detector):
        for i in range(len(self.bottom_detectors) - 1, -1, -1):
            if self.bottom_detectors[i].name == name_detector:
                del self.bottom_detectors[i]
        for i in range(len(self.forward_detectors) - 1, -1, -1):
            if self.forward_detectors[i].name == name_detector:
                del self.forward_detectors[i]

    def reaching_the_position(self):
        pass

    def clear_motor_koef(self):
        self.robot.set_motor_koef(0, 0)

    def processing_task(self, bottom_frame, forward_frame):
        bottom_frame_object = FrameObject(bottom_frame)
        for detector in self.bottom_detectors:
            detector.detect(bottom_frame_object)
        forward_frame_object = FrameObject(forward_frame)
        for detector in self.forward_detectors:
            detector.detect(forward_frame_object)
        if self.current_stage >= 0:
            running = self.stages[self.current_stage].processing()
            if not running:
                self.clear_motor_koef()
                self.current_stage += 1
                if self.current_stage >= len(self.stages):
                    return False
                else:
                    self.stages[self.current_stage].init()
        return True


class StartCircle(Stage):

    def init(self):
        super().init()
        self.red_circles_detector = RedCirclesDetector('red_circles')
        self.task.add_detector_bottom(self.red_circles_detector)
        self.last_start_coord = None

    def proc_stage(self):
        if len(self.red_circles_detector.objects) > 0:
            start = self.red_circles_detector.objects[0]
            self.last_start_coord = start.center
            coord_f = near_coord(start.center, (0, 0))
            if not coord_f:
                self.task.move_to_point((start.center[0], start.center[1], 0),
                                        dist_between_points(start.center, (0, 0)))
            else:
                self.task.clear_motor_koef()
                self.task.move = 0
            depth_f = near_num(self.task.robot.depth(), self.task.target_depth,
                               0.5) and self.task.robot.speed_depth_change() < 0.1
            course_f = near_num(self.task.robot.course(), self.task.course,
                                0.5) and self.task.robot.speed_course_change() < 0.1
            if coord_f and depth_f and course_f:
                self.task.del_detector(self.red_circles_detector)
                self.end_stage()
        else:
            if self.last_start_coord is None:
                self.task.move = MOVE_SPEED
            else:
                self.task.move_to_point((self.last_start_coord[0], self.last_start_coord[1], 0),
                                        dist_between_points(self.last_start_coord, (0, 0)))


class StartCircleSimple(StartCircle):

    def proc_stage(self):
        if len(self.red_circles_detector.objects) > 0:
            start = self.red_circles_detector.objects[0]
            self.last_start_coord = start.center
            coord_f = near_coord_y(start.center, (0, 0))
            if not coord_f:
                self.task.move_to_point_simple((start.center[0], start.center[1], 0),
                                               dist_between_points_y(start.center, (0, 0)))
            else:
                self.task.move = 0
            depth_f = near_num(self.task.robot.depth(), self.task.target_depth,
                               0.5) and self.task.robot.speed_depth_change() < 0.1
            course_f = near_num(self.task.robot.course(), self.task.course,
                                0.5) and self.task.robot.speed_course_change() < 0.1
            if coord_f and depth_f and course_f:
                self.task.del_detector(self.red_circles_detector)
                self.end_stage()
        else:
            if self.last_start_coord is None:
                self.task.move = MOVE_SPEED
            else:
                self.task.move_to_point_simple((self.last_start_coord[0], self.last_start_coord[1], 0),
                                               dist_between_points_y(self.last_start_coord, (0, 0)))


class MoveToStartPath(Stage):

    def init(self):
        super().init()
        self.arrow_detector = ArrowDetector('arrow')
        self.task.add_detector_bottom(self.arrow_detector)
        self.last_arrow_coord = None

    def proc_stage(self):
        if len(self.arrow_detector.objects) > 0:
            arrow = self.arrow_detector.objects[0]
            self.last_arrow_coord = arrow.center
            coord_f = near_coord(arrow.center, (0, 0))
            if not coord_f:
                self.task.move_to_point((arrow.center[0], arrow.center[1], 0),
                                        dist_between_points(arrow.center, (0, 0)))
            else:
                self.task.clear_motor_koef()
                self.task.move = 0
            course_f = near_num(self.task.robot.course(), self.task.course,
                                0.5) and self.task.robot.speed_course_change() < 0.1
            if coord_f and course_f:
                self.task.del_detector(self.arrow_detector)
                self.end_stage()
        else:
            if self.last_arrow_coord is None:
                self.task.move = MOVE_SPEED
            else:
                self.task.move_to_point((self.last_arrow_coord[0], self.last_arrow_coord[1], 0),
                                        dist_between_points(self.last_arrow_coord, (0, 0)))


class MoveToStartPathSimple(MoveToStartPath):

    def proc_stage(self):
        if len(self.arrow_detector.objects) > 0:
            arrow = self.arrow_detector.objects[0]
            self.last_arrow_coord = arrow.center
            coord_f = near_coord_y(arrow.center, (0, 0)) and abs(self.task.move) < 0.01
            if not coord_f:
                self.task.move_to_point_simple((arrow.center[0], arrow.center[1], 0),
                                               dist_between_points_y(arrow.center, (0, 0)))
            else:
                self.task.move = 0
            course_f = near_num(self.task.robot.course(), self.task.course,
                                0.5) and self.task.robot.speed_course_change() < 0.1
            if coord_f and course_f:
                self.task.del_detector(self.arrow_detector)
                self.end_stage()
        else:
            if self.last_arrow_coord is None:
                self.task.move = MOVE_SPEED
            else:
                self.task.move_to_point_simple((self.last_arrow_coord[0], self.last_arrow_coord[1], 0),
                                               dist_between_points_y(self.last_arrow_coord, (0, 0)))


class ArrowPath(Stage):

    def init(self):
        super().init()
        self.arrow_detector = ArrowDetector('arrow')
        self.red_circles_detector = RedCirclesDetector('red_circles')
        self.task.add_detector_bottom(self.arrow_detector)
        self.task.add_detector_bottom(self.red_circles_detector)
        self.move_to_arrow = False
        self.rotate = False
        self.last_arrow_number = 0
        self.last_arrow = None

    def proc_stage(self):
        if len(self.red_circles_detector.objects) > 0:
            self.task.del_detector(self.arrow_detector)
            self.task.del_detector(self.red_circles_detector)
            self.end_stage()
            return
        next_arrows = list(filter(lambda x: abs(x.angle) == 90,
                                  self.arrow_detector.objects_after_number(self.last_arrow_number)))
        if len(next_arrows) > 0:
            arrow = next_arrows[0]
            for i in range(1, len(next_arrows)):
                i_arrow = next_arrows[i]
                if abs(i_arrow.center[0]) < abs(arrow.center[0]):
                    arrow = i_arrow

            self.last_arrow = arrow
            coord_f = near_coord(arrow.center, (0, -20))
            if not coord_f:

                self.task.move_to_point((arrow.center[0], arrow.center[1] + 20, 0),
                                        dist_between_points(arrow.center, (0, -20)))
            else:
                self.task.clear_motor_koef()
                self.task.move = 0
            if coord_f:
                angle = angle_between_vect(arrow.vector, (0, 1))
                self.task.course = get_normal_curse(self.task.robot.course() + angle)
                course_f = near_num(self.task.robot.course(), self.task.course,
                                    0.5) and self.task.robot.speed_course_change() < 0.1
                if course_f:
                    self.last_arrow_number = arrow.number
                    self.last_arrow = None
        else:
            if self.last_arrow is None:
                self.task.move = MOVE_SPEED
            else:
                self.task.move_to_point((self.last_arrow.center[0], self.last_arrow.center[1], 0),
                                        dist_between_points(self.last_arrow.center, (0, 0)))


class ArrowPathSimple(ArrowPath):

    def init(self):
        super().init()
        self.to_center = True

    def proc_stage(self):
        if len(self.red_circles_detector.objects) > 0:
            self.task.del_detector(self.arrow_detector)
            self.task.del_detector(self.red_circles_detector)
            self.end_stage()
            return
        next_arrows = list(filter(lambda x: abs(x.angle) == 90,
                                  self.arrow_detector.objects_after_number(self.last_arrow_number)))
        if len(next_arrows) > 0:
            arrow = next_arrows[0]
            for i in range(1, len(next_arrows)):
                i_arrow = next_arrows[i]
                if abs(i_arrow.center[0]) < abs(arrow.center[0]):
                    arrow = i_arrow
            self.last_arrow = arrow

            coord_f = near_coord_y(arrow.center, (0, 0))
            if not coord_f and self.to_center:
                self.task.move_to_point_simple((arrow.center[0], arrow.center[1], 0),
                                               dist_between_points_y(arrow.center, (0, 0)))
            else:
                self.task.move = 0
                self.to_center = False
            if coord_f and abs(self.task.move) < 0.01:
                self.task.course = get_normal_curse(self.task.robot.course() + arrow.angle)
                self.last_arrow_number = arrow.number
                self.last_arrow = None
        else:
            course_f = near_num(self.task.robot.course(), self.task.course,
                                0.5) and self.task.robot.speed_course_change() < 0.1
            if course_f:
                self.task.move = MOVE_SPEED
                self.to_center = True


class EndCircle(Stage):

    def init(self):
        super().init()
        self.end_circle_detector = RedCirclesDetector('red_circles')
        self.task.add_detector_bottom(self.end_circle_detector)
        self.ascent = False

    def proc_stage(self):
        end_circle = None
        if len(self.end_circle_detector.objects) > 0:
            end_circle = self.end_circle_detector.objects[0]
        if end_circle is not None:
            self.task.move_to_point((end_circle.center[0], end_circle.center[1], 0),
                                    dist_between_points(end_circle.center, (0, -50)))
        coord_f = near_coord(end_circle.center, (0, -50))
        if coord_f and not self.ascent:
            self.task.move = 0
            self.ascent = True
            self.task.robot.allow_surfacing()
            self.task.target_depth = 0
        elif not end_circle:
            self.task.move = MOVE_SPEED


class EndCircleSimple(EndCircle):

    def proc_stage(self):
        end_circle = None
        if not self.ascent:
            if len(self.end_circle_detector.objects) > 0:
                end_circle = self.end_circle_detector.objects[0]
            if end_circle is not None:
                self.task.move_to_point_simple((end_circle.center[0], end_circle.center[1] + 20, 0),
                                               dist_between_points_y(end_circle.center, (0, 20)))
                coord_f = near_coord_y(end_circle.center, (0, -20))
                if coord_f and not self.ascent:
                    self.task.move = 0
                    self.ascent = True
                    self.task.robot.allow_surfacing()
                    self.task.target_depth = 0
            else:
                self.task.move = MOVE_SPEED


class Switchman(Task):

    def __init__(self, robot):
        super().__init__(robot)
        self.add_stage(StartCircle(self))
        self.add_stage(MoveToStartPath(self))
        self.add_stage(ArrowPath(self))
        self.add_stage(EndCircle(self))


class SwitchmanSimple(Task):

    def __init__(self, robot):
        super().__init__(robot)
        self.add_stage(StartCircleSimple(self))
        self.add_stage(MoveToStartPathSimple(self))
        self.add_stage(ArrowPathSimple(self))
        self.add_stage(EndCircleSimple(self))


class MotorControl:
    max_motor_power = 100
    motor_depth_1 = 2
    motor_depth_2 = 3
    motor_left = 0
    motor_right = 1
    koef_move = 0.5
    koef_course = 0.5
    koef_left = 0
    koef_right = 0

    def __init__(self, sim):
        self.sim = sim

    def set_koef_motor(self, left, right):
        self.koef_left = left
        self.koef_right = right

    def set_motor_depth_power(self, vect):
        power = vect * self.max_motor_power
        self.sim.set_motor_power(self.motor_depth_1, power)
        self.sim.set_motor_power(self.motor_depth_2, power)

    def set_motor_cource_move_power(self, left, right):
        power_left = left * self.max_motor_power
        power_right = right * self.max_motor_power
        self.sim.set_motor_power(self.motor_left, power_left)
        self.sim.set_motor_power(self.motor_right, power_right)

    def set_motor_power(self, course_vect, move_vect, depth_vect):
        self.set_motor_depth_power(depth_vect)
        left_p = course_vect * self.koef_course + move_vect * self.koef_move + self.koef_left
        right_p = -course_vect * self.koef_course + move_vect * self.koef_move + self.koef_right
        self.set_motor_cource_move_power(left_p, right_p)


class DepthControl:
    max_calc_difference = 0.5
    koef_dif = 0.5
    max_target_depth = 0.1
    min_target_depth = 4
    max_len_vector = 1

    def __init__(self, target_depth=0):
        self.target_depth = 0
        self.current_depth = 0
        self.clipped_depth_difference = 0
        self.supportingTarget_depth = True
        self.calc_difference = self.max_len_vector / self.max_calc_difference
        self.setTargetDepth(target_depth)
        self.valueLog = ChangeValueLogger()

    def updateValues(self):
        depth_dif = dif_num(self.current_depth, self.target_depth) + self.valueLog.getVelosityWithKoef()
        self.clipped_depth_difference = clip_num(depth_dif, -self.max_calc_difference,
                                                 self.max_calc_difference)

    def setTargetDepth(self, depth):
        self.target_depth = clip_num(depth, self.max_target_depth, self.min_target_depth)

    def startCalc(self):
        self.supportingTarget_depth = True

    def stopCalc(self):
        self.supportingTarget_depth = False

    def getVect(self):
        val = self.calc_difference * self.clipped_depth_difference
        return val if self.supportingTarget_depth else 0

    def updateDepth(self, depth, elapsed):
        self.valueLog.updateValue(depth, elapsed)
        self.current_depth = depth
        self.updateValues()


class MoveControl:

    def __init__(self):
        self.move = 0

    def setMove(self, move):
        self.move = move

    def getVect(self):
        return self.move


class CourseControl:
    max_calc_difference = 20
    max_len_vector = 1

    def __init__(self, target_course=0):
        self.course_log = ChangeValueLogger(dif_angle_180)
        self.angle_log = ChangeValueLogger(dif_angle_360)
        self.clipped_curse_difference = 0
        self.clipped_angle_difference = 0
        self.target_course = 0
        self.current_course = 0
        self.current_angle = 0
        self.setTargetCourse(target_course)
        self.updateCourse(target_course, 0)
        self.calc_difference = self.max_len_vector / self.max_calc_difference

    def setTargetCourse(self, target_course):
        self.setTargetAngle(course_to_angle(target_course))

    def setTargetAngle(self, target_angle):
        self.target_angle = target_angle
        self.target_course = angle_to_course(self.target_angle)
        self.updateValues()

    def updateCourse(self, course, elapsed):
        self.current_course = course
        self.current_angle = course_to_angle(self.current_course)
        self.course_log.updateValue(self.current_course, elapsed)
        self.angle_log.updateValue(self.current_angle, elapsed)
        self.updateValues()

    def updateValues(self):
        course_dif = dif_angle_180(self.target_course, self.current_course) - self.course_log.getVelosityWithKoef()
        angle_dif = dif_angle_360(self.target_angle, self.current_angle) - self.angle_log.getVelosityWithKoef()
        self.cliped_curse_difference = clip_num(course_dif,
                                                -self.max_calc_difference,
                                                self.max_calc_difference)
        self.clipped_angle_difference = clip_num(angle_dif,
                                                 -self.max_calc_difference,
                                                 self.max_calc_difference)

    def getVect(self):
        val = self.calc_difference * self.clipped_angle_difference
        return val


class PositionController:

    def __init__(self, sim, use_def_control=True):
        self.sim = sim
        self.depth_controller = None
        self.motor_controller = None
        self.course_controller = None
        self.move_controller = None
        self.x = 0
        self.y = 0
        self.depth = 0
        self.course = 0
        if use_def_control:
            self.setDefualtControllers()

    def setDefualtControllers(self, motor=True, depth=True, course=True, move=True):
        if motor:
            self.replaceMotorController(MotorControl(self.sim))
        if depth:
            self.replaceDepthController(DepthControl(self.sim.get_depth()))
        if course:
            self.replaceCourseController(CourseControl())
        if move:
            self.replaceMoveController(MoveControl())

    def replaceDepthController(self, depth_controller):
        self.depth_controller = depth_controller

    def replaceMotorController(self, motor_controller):
        self.motor_controller = motor_controller

    def replaceCourseController(self, course_controller):
        self.course_controller = course_controller

    def replaceMoveController(self, move_controller):
        self.move_controller = move_controller

    def update(self, elapsed):
        self.depth_controller.updateDepth(self.sim.get_depth(), elapsed)
        d_vect = self.depth_controller.getVect() if self.depth_controller else 0
        self.course_controller.updateCourse(self.sim.get_yaw(), elapsed)
        c_vect = self.course_controller.getVect() if self.course_controller else 0
        m_vect = self.move_controller.getVect() if self.move_controller else 0
        self.motor_controller.set_motor_power(c_vect, m_vect, d_vect)

    def setTargetDepth(self, depth):
        self.depth_controller.setTargetDepth(depth)

    def setTargetCourse(self, course):
        self.course_controller.setTargetCourse(course)

    def setTargetMove(self, move):
        self.move_controller.setMove(move)


class Bobot:

    def __init__(self):
        self.task = SwitchmanSimple(self)
        self.position = PositionController(auv)
        if TRUE_ROBOT:
            self.cam_front = cv.VideoCapture(0)
            self.cam_bottom = cv.VideoCapture(1)

    def depth(self):
        return self.position.depth_controller.current_depth

    def course(self):
        return self.position.course_controller.current_course

    def allow_surfacing(self):
        DepthControl.max_target_depth = 0

    def speed_depth_change(self):
        return self.position.depth_controller.valueLog.velocity

    def speed_course_change(self):
        return self.position.course_controller.course_log.velocity

    def set_motor_koef(self, left, right):
        if self.position.motor_controller is not None:
            self.position.motor_controller.set_koef_motor(left, right)

    # основной цикл робота
    def cicle(self, elapsed):
        if not TRUE_ROBOT:
            front = auv.get_image_front()
            bottom = auv.get_image_bottom()
        else:
            ret, front = self.cam_front.read()
            ret, bottom = self.cam_bottom.read()
        self.task.processing_task(bottom, front)
        self.position.setTargetDepth(self.task.target_depth)
        self.position.setTargetCourse(self.task.course)
        self.position.setTargetMove(self.task.move)
        self.position.update(elapsed)


if __name__ == '__main__':
    bobotik = Bobot()
    last_time = time.time()
    while (True):
        cur_time = time.time()
        bobotik.cicle(cur_time - last_time)
        last_time = cur_time
        time.sleep(0.05)