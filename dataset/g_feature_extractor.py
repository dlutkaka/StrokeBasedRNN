# encoding: utf-8

"""
@file: g_feature_extractor.py
@time: 2019/3/17 09:03

"""

from __future__ import print_function
from __future__ import division
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math


def number_sign_changes(discrete_data):
    if discrete_data is None:
        return None
    sign_changes = 0
    for (index, data) in enumerate(discrete_data):
        if index - 1 >= 0:
            sign_changes += 1 if discrete_data[index - 1] * data < 0 else 0
    return sign_changes


def derivative(discrete_data):
    if discrete_data is None:
        return None
    return np.gradient(discrete_data)


class Signature(object):
    def __init__(self, x, y, p, t, normalize=True):
        self.x = Signature.normalize(x)
        self.y = Signature.normalize(y)
        self.p = Signature.normalize(p)
        self.t = t
        self.x_derivative = derivative(self.x)
        self.y_derivative = derivative(self.y)
        self.p_derivative = derivative(self.p)
        self.x_second_derivative = derivative(self.x_derivative)
        self.y_second_derivative = derivative(self.y_derivative)
        self.p_second_derivative = derivative(self.p_derivative)

    @classmethod
    def from_mat_file(cls, mat_file_path):
        data = loadmat(mat_file_path)
        x = data['x'][0].tolist()
        y = data['y'][0].tolist()
        p = data['p'][0].tolist()
        t = None
        if hasattr(data, 't'):
            t = data['t'][0].tolist()
        if x is None or y is None or p is None:
            raise ValueError('x,y and p can not be None')
        return Signature(x, y, p, t, normalize=True)

    @staticmethod
    def normalize(list_data):
        if list_data is None:
            return None
        scaler = MinMaxScaler()
        data = np.array(list_data, dtype=np.float64)
        data = data.reshape(-1, 1)
        data = scaler.fit_transform(data)
        data = data.reshape(-1)
        return data.tolist()


class FeatureExtractor(object):
    def total_duration(self):
        duration = 0
        if self.signature.t is not None:
            duration = self.signature.t[len(
                self.signature.t) - 1] - self.signature.t[0]
        return duration

    def total_stroke_number(self):
        total_number = 1
        pre_p = None
        for p in self.signature.p:
            if p != pre_p and p == 0:
                total_number += 1
            pre_p = p
        return total_number

    def total_touch_duration(self):
        duration = 0
        if self.signature.t is not None:
            pre_p = None
            stroke_start_time = self.signature.t[0]
            for (index, p) in self.signature.p:
                if p != pre_p and p == 0:
                    stroke_end_time = self.signature.t[index]
                    duration += stroke_end_time - stroke_start_time
                if pre_p == 0 and p != 0:
                    stroke_start_time = self.signature.t[index]
            if stroke_start_time > stroke_end_time:
                duration += self.signature.t[len(self.signature.t) -
                                             1] - stroke_start_time
        return duration

    def time_percent_touch(self):
        total_time = self.total_duration()
        touch_time = self.total_touch_duration()
        return touch_time / total_time

    def sign_changes(self):
        result = list()
        result.append(number_sign_changes(self.signature.x_derivative))
        result.append(number_sign_changes(self.signature.y_derivative))
        result.append(number_sign_changes(self.signature.p_derivative))
        result.append(number_sign_changes(self.signature.x_second_derivative))
        result.append(number_sign_changes(self.signature.y_second_derivative))
        result.append(number_sign_changes(self.signature.p_second_derivative))
        return result

    def sign_mean(self):
        result = list()
        x_mean = np.mean(self.signature.x_derivative)
        y_mean = np.mean(self.signature.y_derivative)
        x_mean1 = np.mean(self.signature.x_second_derivative)
        y_mean1 = np.mean(self.signature.y_second_derivative)
        p_mean = np.mean(self.signature.p_derivative)
        result.append(x_mean)
        result.append(y_mean)
        result.append(x_mean1)
        result.append(y_mean1)
        result.append(p_mean)
        return result

    def sign_standard_derivation(self):
        result = list()
        x_std = np.std(self.signature.x_derivative, ddof=1)
        y_std = np.std(self.signature.y_derivative, ddof=1)
        x_std1 = np.std(self.signature.x_second_derivative, ddof=1)
        y_std1 = np.std(self.signature.y_second_derivative, ddof=1)
        result.append(x_std)
        result.append(y_std)
        result.append(x_std1)
        result.append(y_std1)
        return result

    def maximum_pressure_derivative(self):
        return np.max(self.signature.p_derivative)

    def mean_velocity(self):
        distance = list()
        for index, value in enumerate(self.signature.x):
            if index == 0:
                continue
            x_diff = value - self.signature.x[index - 1]
            y_diff = self.signature.y[index] - self.signature.y[index - 1]
            dis = math.sqrt(math.pow(x_diff, 2) + math.pow(y_diff, 2))
            distance.append(dis)
        if self.signature.t is not None:
            result = np.sum(distance) / \
                (self.signature.t[index] - self.signature.t[0])
        else:
            result = np.sum(distance) / index
        return result

    def mean_velocity_x(self):
        return np.mean(np.abs(self.signature.x_derivative))

    def mean_velocity_y(self):
        return np.mean(np.abs(self.signature.y_derivative))

    def time_maximal_value(self):
        result = [0, 0, 0, 0, 0, 0, 0]
        max_x = float('-Inf')
        max_y = float('-Inf')
        max_x_derivative = float('-Inf')
        max_y_derivative = float('-Inf')
        max_x_second_derivative = float('-Inf')
        max_y_second_derivative = float('-Inf')
        max_p_derivative = float('-Inf')
        for i in range(len(self.signature.x)):
            if max_x < self.signature.x[i]:
                max_x = self.signature.x[i]
                result[0] = i
            if max_y < self.signature.y[i]:
                max_y = self.signature.y[i]
                result[1] = i
            if max_x_derivative < self.signature.x_derivative[i]:
                max_x_derivative = self.signature.x_derivative[i]
                result[2] = i
            if max_y_derivative < self.signature.y_derivative[i]:
                max_y_derivative = self.signature.y_derivative[i]
                result[3] = i
            if max_x_second_derivative < self.signature.x_second_derivative[i]:
                max_x_second_derivative = self.signature.x_second_derivative[i]
                result[4] = i
            if max_y_second_derivative < self.signature.y_second_derivative[i]:
                max_y_second_derivative = self.signature.y_second_derivative[i]
                result[5] = i
            if max_p_derivative < self.signature.p_derivative[i]:
                max_p_derivative = self.signature.p_derivative[i]
                result[6] = i
        if self.signature.t is not None:
            total_time = self.signature.t[len(
                self.signature.t) - 1] - self.signature.t[0]
            result = [(self.signature.t[r] - self.signature.t[0]) /
                      total_time for r in result]
        else:
            result = [r / len(self.signature.x) for r in result]
        return result

    def time_minimal_value(self):
        result = [0, 0, 0, 0, 0, 0]
        min_x = float('+Inf')
        min_y = float('+Inf')
        min_x_derivative = float('+Inf')
        min_y_derivative = float('+Inf')
        min_x_second_derivative = float('+Inf')
        min_y_second_derivative = float('+Inf')
        for i in range(len(self.signature.x)):
            if min_x > self.signature.x[i]:
                min_x = self.signature.x[i]
                result[0] = i
            if min_y < self.signature.y[i]:
                min_y = self.signature.y[i]
                result[1] = i
            if min_x_derivative < self.signature.x_derivative[i]:
                min_x_derivative = self.signature.x_derivative[i]
                result[2] = i
            if min_y_derivative < self.signature.y_derivative[i]:
                min_y_derivative = self.signature.y_derivative[i]
                result[3] = i
            if min_x_second_derivative < self.signature.x_second_derivative[i]:
                min_x_second_derivative = self.signature.x_second_derivative[i]
                result[4] = i
            if min_y_second_derivative < self.signature.y_second_derivative[i]:
                min_y_second_derivative = self.signature.y_second_derivative[i]
                result[5] = i
        if self.signature.t is not None:
            total_time = self.signature.t[len(
                self.signature.t) - 1] - self.signature.t[0]
            result = [(self.signature.t[r] - self.signature.t[0]) /
                      total_time for r in result]
        else:
            result = [r / len(self.signature.x) for r in result]
        return result

    def mean_acceleration(self):
        velocity = list()
        for index, value in enumerate(self.signature.x):
            if index == 0:
                continue
            x_diff = value - self.signature.x[index - 1]
            y_diff = self.signature.y[index] - self.signature.y[index - 1]
            dis = math.sqrt(math.pow(x_diff, 2) + math.pow(y_diff, 2))
            vel = dis
            if self.signature.t is not None:
                vel = dis / self.signature.t[index] - \
                    self.signature.t[index - 1]
            velocity.append(vel)
        acceleration = derivative(velocity)
        return np.mean(acceleration)

    def start_direction(self):
        if len(self.signature.y) < 2:
            return 0
        start_y_diff = self.signature.y[1] - self.signature.y[0]
        start_x_diff = self.signature.x[1] - self.signature.x[0]
        return math.atan2(start_y_diff, start_x_diff)

    def angle_histogram(self):
        size = 8
        result = [0 for _ in range(size)]
        for i in range(len(self.signature.x)):
            if i + 1 < len(self.signature.x):
                x_diff = self.signature.x[i + 1] - self.signature.x[i]
                y_diff = self.signature.y[i + 1] - self.signature.y[i]
                angle = math.atan2(y_diff, x_diff)
                index = (angle + math.pi) / (math.pi * 2 / size)
                index = int(math.ceil(index))
                if index < size:
                    result[index] += 1
        return result

    def velocity_histogram(self):
        size = 16
        result = [0 for _ in range(size)]
        velocities = list()
        for i in range(len(self.signature.x_derivative)):
            velocity1 = math.pow(
                self.signature.x_derivative[i], 2) + math.pow(self.signature.y_derivative[i], 2)
            velocity = math.sqrt(velocity1)
            velocities.append(velocity)
        mean = np.mean(velocities)
        std = np.std(velocities, ddof=1)
        range_max = mean + std * 3
        for v in velocities:
            index = v / (range_max / size)
            index = int(math.ceil(index))
            if index < size:
                result[index] += 1
        return result

    def extract_global_features(self, signature):
        try:
            self.signature = signature
        except Exception as e:
            print(e)
            raise e
        result = list()
        writing_duration = self.total_duration()
        stroke_number = self.total_stroke_number()
        touch_duration = self.total_touch_duration()
        sign_changes = self.sign_changes()
        sign_mean = self.sign_mean()
        sign_std = self.sign_standard_derivation()
        max_p = self.maximum_pressure_derivative()
        mean_velocity = self.mean_velocity()
        mean_acceleration = self.mean_acceleration()
        mean_velocity_x = self.mean_velocity_x()
        mean_velocity_y = self.mean_velocity_y()
        time_maximal_value = self.time_maximal_value()
        time_minimal_value = self.time_minimal_value()
        start_direction = self.start_direction()
        angle_histogram = self.angle_histogram()
        velocity_histogram = self.velocity_histogram()

        result.append(writing_duration)
        result.append(stroke_number)
        result.append(touch_duration)
        result.extend(sign_changes)
        result.extend(sign_mean)
        result.extend(sign_std)
        result.append(max_p)
        result.append(mean_velocity)
        result.append(mean_acceleration)
        result.append(mean_velocity_x)
        result.append(mean_velocity_y)
        result.extend(time_maximal_value)
        result.extend(time_minimal_value)
        result.append(start_direction)
        result.extend(angle_histogram)
        result.extend(velocity_histogram)
        return result


def extract_global_feature(x, y, p, t, normalize=False):
    signature = Signature(x, y, p, t, normalize=normalize)
    extractor = FeatureExtractor()
    return extractor.extract_global_features(signature)


def main():
    feature_extractor = FeatureExtractor()
    feature_extractor.extract_global_features(Signature.from_mat_file(
        '/Users/Bruce/Desktop/BiosecurID-SONOF-DB/OnlineReal/u1001s0001_sg0001.mat'))


if __name__ == '__main__':
    main()
