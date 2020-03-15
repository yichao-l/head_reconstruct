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
    if (np.array(axis) == np.array([1, 0, 0])).all():
        return np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    elif (np.array(axis) == np.array([0, 1, 0])).all():
        return np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [- np.sin(phi), 0, np.cos(phi)]])
    elif (np.array(axis) == np.array([0, 0, 1])).all():
        return np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    return None


def refine_phi(mhead, A, B, offset, step, axis):
    best_count = -1

    head1 = mhead.heads[mhead.head_id_from_frame_id(A)]
    head2 = mhead.heads[mhead.head_id_from_frame_id(B)]
    l = head2.xyz.shape[0]
    filter2 = np.random.random((l)) < 0.1

    for phi in np.arange(-offset * np.pi / 180, offset * np.pi / 180, step * np.pi / 180):
        R = calc_R(phi, axis)
        head2.xyz = np.dot(head2.xyz, R)
        xyz1 = head1.xyz
        xyz2 = head2.xyz[filter2]
        distances, indices = nearest_neighbor(xyz2, xyz1)
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


def refine_phi2(mhead, A, B, offset, step, axis):
    head1 = mhead.heads[mhead.head_id_from_frame_id(A)]
    head2 = mhead.heads[mhead.head_id_from_frame_id(B)]
    l = head2.xyz.shape[0]
    filter2 = np.random.random((l)) < 0.1
    lower_count = 0
    higher_count = 0
    middle_count = 0
    phi = 0
    init = 3
    bounce_count = 0
    move_down = True
    while lower_count >= middle_count or higher_count >= middle_count or init > 0:
        if lower_count > middle_count or init > 0:
            s = -step * np.pi / 180
            move_down = True
        elif higher_count > middle_count:
            s = step * np.pi / 180
            move_down = False
        elif lower_count == middle_count:
            s = -step * np.pi / 180
            if move_down == False:
                bounce_count += 1
            move_down = True

        elif higher_count == middle_count:
            s = step * np.pi / 180
            move_down = False
        else:
            break
        if bounce_count == 2:
            break
        init -= 1
        phi = phi + s
        R = calc_R(2 * s, axis)
        head2.xyz = np.dot(head2.xyz, R)
        xyz1 = head1.xyz
        xyz2 = head2.xyz[filter2]
        distances, indices = nearest_neighbor(xyz2, xyz1)
        count = np.sum(distances < 0.01)
        R = calc_R(-s, axis)
        head2.xyz = np.dot(head2.xyz, R)

        if move_down:
            higher_count = middle_count
            middle_count = lower_count
            lower_count = count
        elif not move_down:
            lower_count = middle_count
            middle_count = higher_count
            higher_count = count

        print(
            f"{axis}, {phi * 180 / np.pi:.1f}, {count} ,{bounce_count}|| {lower_count}, {middle_count} , {higher_count}")
    return phi


def refine_3(mhead, A, B, step, axis, angle):
    head1 = mhead.heads[mhead.head_id_from_frame_id(A)]
    head2 = mhead.heads[mhead.head_id_from_frame_id(B)]
    l = head2.xyz.shape[0]
    filter2 = np.random.random((l)) < 0.1
    lower_count = 0
    higher_count = 0
    middle_count = 0
    phi = 0
    init = 3
    bounce_count = 0
    move_down = True
    while lower_count >= middle_count or higher_count >= middle_count or init > 0:
        if lower_count > middle_count or init > 0:
            s = -step * np.pi / 180
            move_down = True
        elif higher_count > middle_count:
            s = step * np.pi / 180
            move_down = False
        elif lower_count == middle_count:
            s = -step * np.pi / 180
            if move_down == False:
                bounce_count += 1
            move_down = True

        elif higher_count == middle_count:
            s = step * np.pi / 180
            move_down = False
        else:
            break
        if bounce_count == 2:
            break
        init -= 1
        phi = phi + s

        if angle:
            R = calc_R(2 * s, axis)
            head2.xyz = np.dot(head2.xyz, R)
        else:
            t = np.array(axis) * 2 * s
            head2.xyz = head2.xyz + t

        xyz1 = head1.xyz
        xyz2 = head2.xyz[filter2]
        distances, indices = nearest_neighbor(xyz2, xyz1)
        count = np.sum(distances < 0.01)
        if angle:

            R = calc_R(-s, axis)
            head2.xyz = np.dot(head2.xyz, R)
        else:
            t = np.array(axis) * -1 * s
            head2.xyz = head2.xyz + t

        if move_down:
            higher_count = middle_count
            middle_count = lower_count
            lower_count = count
        elif not move_down:
            lower_count = middle_count
            middle_count = higher_count
            higher_count = count

        print(
            f"{axis}, {phi * 180 / np.pi:.1f}, {count} ,{bounce_count}|| {lower_count}, {middle_count} , {higher_count}")
    return phi * np.array(axis)


def refine3d(mhead, A, B):
    t = np.zeros(3)
    wide_r = 0.05
    wide_s = 0.02
    med_r = 0.025
    med_s = 0.005
    fine_r = 0.007
    fine_s = 0.001
    phi_r = 10
    phi_s = 0.01

    # with tqdm(total=9) as progressbar:
    #     for r, s in zip([wide_r, med_r, fine_r], [wide_s, med_s, fine_s]):
    #         for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
    #             t = t + refine(mhead, A, B, r, s, axis)
    #             progressbar.set_description(f"{A} to {B} t  := {t[0]:3f},{t[1]:3f},{t[2]:3f}")
    #             progressbar.update(1)
    with tqdm(total=3) as progressbar:
        for r, s in zip([phi_r], [phi_s]):
            for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
                # t = t + refine_phi2(mhead, A, B, r, s, axis)

                t = t + refine_3(mhead, A, B, s, axis, angle=True)

                progressbar.set_description(f"{A} to {B} phi:= {t[0]:3f},{t[1]:3f},{t[2]:3f}")
                progressbar.update(1)
