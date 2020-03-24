import numpy as np


class Link():
    def __init__(self, left, right):
        '''
        left and right are the frame id of the heads
        '''
        self.left = left
        self.right = right

    def add_matches(self, matches):
        self.matches = matches

    def reset(self):
        if hasattr(self, "tform"):
            del self.tform
        if hasattr(self, "matches"):
            del self.matches
        if hasattr(self, "inliers"):
            del self.inliers
        if hasattr(self, "err"):
            del self.err

    def add_ransac_results(self, tform, inliers, err, matches):
        self.tform = tform
        self.inliers = inliers
        self.err = err
        self.matches = matches

    def print(self):
        print("left:", self.left, "\tright:", self.right)
        if hasattr(self, "tform"):
            print(self.tform)
        if hasattr(self, "matches"):
            print(f"# Matches {len(self.matches)}")
        if hasattr(self, "inliers"):
            print(f"# Inliers {len(self.inliers)}")

    def print_short(self):
        print(f"{self.left}-{self.right}, Count={np.sum(self.inliers)}, Err={self.err:.4f}")
