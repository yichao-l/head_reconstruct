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
        if hasattr(self, "inliers_all_points"):
            del self.inliers_all_points
        if hasattr(self, "coverage_all_points"):
            del self.coverage_all_points
        if hasattr(self, "kp_sample_matches"):
            del self.kp_sample_matches
        if hasattr(self, "err_matches"):
            del self.err_matches
        if hasattr(self, "matches"):
            del self.matches

    def add_ransac_results(self, sample_matches_cvg, pct_coverage, sample_matches_mchs, err_matches, matches):
        '''
        Store the results from the RANSAC computation to the object.
        '''
        self.inliers_all_points = sample_matches_cvg
        self.coverage_all_points = pct_coverage
        self.kp_sample_matches = sample_matches_mchs
        self.err_matches = err_matches
        self.matches = matches

    def print(self):
        print("left:", self.left, "\tright:", self.right)
        if hasattr(self, "matches"):
            print(f"# Matches {len(self.matches)}")
        if hasattr(self, "kp_sample_matches"):
            print(f"# Inliers {len(self.kp_sample_matches)}")
        if hasattr(self, "coverage_all_points"):
            print(f"Coverage {100 * self.coverage_all_points:.0f}%")
        if hasattr(self, "err_matches"):
            print(f"Err Matches {self.err_matches:.4f}%")

    def print_short(self):
        print(
            f"{self.left}-{self.right}, Count={np.sum(self.kp_sample_matches)}, Err Matches={self.err_matches:.4f}, Cov={100 * self.coverage_all_points:.1f}")
