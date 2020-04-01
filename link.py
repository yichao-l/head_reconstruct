import numpy as np
'''
The Link object contains all required variables to store the results of SIFT matches between two subsequent frames.

It is defined by:
The IDs of two Single Heads, self.left and self.right
the result of the RANAC operation based on the SIFT matches:
    The matches object, self.macthes: indicating which  
    the inliers object, self.inliers: a boolean filter to select the calcualted matches from self.matches
    the resulting transformation, tform and the error metric, indicating how good the match is.
'''

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
        if hasattr(self, "sample_matches_cvg"):
            del self.sample_matches_cvg
        if hasattr(self, "pct_coverage"):
            del self.pct_coverage
        if hasattr(self, "sample_matches_mchs"):
            del self.sample_matches_mchs
        if hasattr(self, "err_matches"):
            del self.err_matches
        if hasattr(self, "matches"):
            del self.matches

    def add_ransac_results(self, sample_matches_cvg, pct_coverage, sample_matches_mchs, err_matches, matches):
        '''
        Store the results from the RANSAC computation to the object.
        '''
        # the sample matches with the best coverage amount all points
        self.sample_matches_cvg = sample_matches_cvg
        self.pct_coverage = pct_coverage
        # the sample matches with the best inlier matches numbers
        self.sample_matches_mchs = sample_matches_mchs
        self.err_matches = err_matches
        self.matches = matches

    def print(self):
        print("left:", self.left, "\tright:", self.right)
        if hasattr(self, "matches"):
            print(f"# Matches {len(self.matches)}")
        if hasattr(self, "sample_matches_mchs"):
            print(f"# Inliers {len(self.sample_matches_mchs)}")
        if hasattr(self, "pct_coverage"):
            print(f"Coverage {100 * self.pct_coverage:.0f}%")
        if hasattr(self, "err_matches"):
            print(f"Err Matches {self.err_matches:.4f}%")

    def print_short(self):
        print(
            f"{self.left}-{self.right}, Count={np.sum(self.sample_matches_mchs)}, Err Matches={self.err_matches:.4f}, Cov={100 * self.pct_coverage:.1f}")
