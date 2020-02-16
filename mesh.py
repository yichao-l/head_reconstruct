import numpy as np


class Triangle():
    def init(self, p, rgb):
        self.p = p
        v1 = p[2] - p[0]
        v2 = p[1] - p[0]
        self.norm = np.cross(v1, v2)
        length = np.linalg.norm
        if np.linalg.norm != 0:
            self.norm /= length
        self.center = np.mean(p, axis=0)
        self.rgb = np.mean(rgb, axis=0)

    def dist(self, p):
        return np.abs(np.inner(p - self.center, self.norm))


def x2line(p0, p1, x):
    x - np.inner(p0 - p1, x - p1) * (p0 - p1) / np.linalg.norm(p0 - p1)


class Mesh():

    def init(self, tr):
        self.trs = [tr]

    def add_point(self, x):
        p0 = self.p[0]
        p1 = self.p[1]
        p2 = self.p[2]
        outside01 = np.inner(x2line(p0, p1, x), x2line(p0, p1, p2)) < 0
        outside02 = np.inner(x2line(p0, p2, x), x2line(p0, p2, p1)) < 0
        outside12 = np.inner(x2line(p1, p2, x), x2line(p1, p2, p0)) < 0
        inside = not (outside01 or outside02 or outside12)
