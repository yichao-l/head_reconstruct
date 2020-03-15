from icp import nearest_neighbor
import numpy as np
from tqdm import tqdm_notebook as tqdm


def calc_distances(mhead, A, B):
    head1 = mhead.heads[mhead.head_id_from_frame_id(A)]
    head2 = mhead.heads[mhead.head_id_from_frame_id(B)]
    A = head1.xyz
    B = head2.xyz
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation

    prev_error = -1
    distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)
    return distances


def refine(mhead, A, B, offset, step, axis):
    best_count = -1

    head2 = mhead.heads[mhead.head_id_from_frame_id(B)]
    head1 = mhead.heads[mhead.head_id_from_frame_id(A)]
    l = head2.xyz.shape[0]
    filter2 = np.random.random((l)) < 0.03
    l = head1.xyz.shape[0]

    for d in np.arange(-offset, offset, step):
        t = np.array(axis) * d

        head2.xyz = head2.xyz + t

        xyz1 = head1.xyz
        xyz2 = head2.xyz[filter2]
        m = xyz2.shape[1]

        # make points homogeneous, copy them to maintain the originals
        dst = np.ones((m + 1, xyz1.shape[0]))
        src = np.ones((m + 1, xyz2.shape[0]))
        dst[:m, :] = np.copy(xyz1.T)
        src[:m, :] = np.copy(xyz2.T)

        # apply the initial pose estimation

        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        count = np.sum(distances < offset)
        if count > best_count:
            best_t = t
            best_count = count
        head2.xyz = head2.xyz - t
    head2.xyz = head2.xyz + best_t
    return best_t


def calc_R(phi, axis):
    if axis == 0:
        return np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    elif axis == 1:
        return np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [- np.sin(phi), 0, np.cos(phi)]])
    elif axis == 2:
        return np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    return None


def refine_phi(mhead, A, B, offset, step, axis):
    best_count = -1

    head2 = mhead.heads[mhead.head_id_from_frame_id(B)]
    head1 = mhead.heads[mhead.head_id_from_frame_id(A)]
    l = head2.xyz.shape[0]
    filter2 = np.random.random((l)) < 0.1

    for phi in np.arange(-offset * np.pi / 180, offset * np.pi / 180, step * np.pi / 180):
        R = calc_R(phi, axis)
        head2.xyz = np.dot(head2.xyz, R)
        xyz1 = head1.xyz
        xyz2 = head2.xyz[filter2]
        m = xyz2.shape[1]

        # make points homogeneous, copy them to maintain the originals
        dst = np.ones((m + 1, xyz1.shape[0]))
        src = np.ones((m + 1, xyz2.shape[0]))
        dst[:m, :] = np.copy(xyz1.T)
        src[:m, :] = np.copy(xyz2.T)

        # apply the initial pose estimation

        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        count = np.sum(distances < 0.01)
        if count > best_count:
            best_phi = phi
            best_count = count
        R = calc_R(-phi, axis)
        print(f"{axis}, {phi * 180 / np.pi:.1f}, {best_count}, {best_phi * 180 / np.pi:.1f}")
        head2.xyz = np.dot(head2.xyz, R)
    R = calc_R(best_phi, axis)
    head2.xyz = np.dot(head2.xyz, R)
    return best_phi


def refine3d(mhead, A, B):
    t = np.zeros(3)
    wide_r = 0.05
    wide_s = 0.02
    med_r = 0.025
    med_s = 0.005
    fine_r = 0.007
    fine_s = 0.001
    phi_r = 10
    phi_s = 1

    with tqdm(total=9) as progressbar:
        for r, s in zip([wide_r, med_r, fine_r], [wide_s, med_s, fine_s]):
            for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
                t = t + refine(mhead, A, B, r, s, axis)
                progressbar.set_description(f"{A} to {B} t  := {t[0]:3f},{t[1]:3f},{t[2]:3f}")
                progressbar.update(1)
    with tqdm(total=3) as progressbar:
        for r, s in zip([phi_r], [phi_s]):
            for axis in [0, 1, 2]:
                t = t + refine_phi(mhead, A, B, r, s, axis)
                progressbar.set_description(f"{A} to {B} phi:= {t[0]:3f},{t[1]:3f},{t[2]:3f}")
                progressbar.update(1)
