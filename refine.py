from icp import nearest_neighbor
import numpy as np
from tqdm.autonotebook import tqdm


def calc_R(phi, axis):
    '''
    :param phi: angle, in degrees
    :param axis: axis for rotation 3D array
    :return:  rotation matrix, as function of angle and selected axis.
    '''
    phi_rad = phi * np.pi / 180
    if (np.array(axis) == np.array([1, 0, 0])).all():
        return np.array([[1, 0, 0], [0, np.cos(phi_rad), -np.sin(phi_rad)], [0, np.sin(phi_rad), np.cos(phi_rad)]])
    elif (np.array(axis) == np.array([0, 1, 0])).all():
        return np.array([[np.cos(phi_rad), 0, np.sin(phi_rad)], [0, 1, 0], [- np.sin(phi_rad), 0, np.cos(phi_rad)]])
    elif (np.array(axis) == np.array([0, 0, 1])).all():
        return np.array([[np.cos(phi_rad), -np.sin(phi_rad), 0], [np.sin(phi_rad), np.cos(phi_rad), 0], [0, 0, 1]])
    raise ValueError("invalid axis")


def refine_over_range(mhead, A, B, range, step, axis, filter, angle, max_distance=0.005):
    '''

    :param mhead:
    :param A:
    :param B:
    :param range:
    :param step:
    :param axis:
    :param filter:
    :param angle:
    :param max_distance:
    :return:
    '''
    max_distance_between_points = max_distance
    best_count = -1

    head1 = mhead.heads[mhead.head_id_from_frame_id(A)]
    head2 = mhead.heads[mhead.head_id_from_frame_id(B)]
    l = head2.xyz.shape[0]

    if filter is None:
        filter2 = np.random.random((l)) < 0.1
    else:
        filter2 = filter

    for value in np.arange(-range, range, step):
        if angle:
            CoG = head2.xyz.mean(axis=0)
            R = calc_R(value, axis)
            head2.xyz = np.dot(head2.xyz - CoG, R) + CoG
        else:
            head2.xyz = head2.xyz + value
        xyz1 = head1.xyz
        xyz2 = head2.xyz[filter2]
        distances, indices = nearest_neighbor(xyz2, xyz1)
        count = np.sum(distances < max_distance_between_points)
        if count > best_count:
            best_value = value
            best_count = count
        if angle:
            R = calc_R(-value, axis)
            head2.xyz = np.dot(head2.xyz - CoG, R) + CoG
        else:
            head2.xyz = head2.xyz - value
        # print(value * np.array(axis), count / np.size(xyz2), best_count / np.size(xyz2))
    if angle:
        R = calc_R(best_value, axis)
        head2.xyz = np.dot(head2.xyz - CoG, R) + CoG
    else:
        head2.xyz = head2.xyz + best_value
    return best_value * np.array(axis), best_count / np.size(xyz2), filter2


def refine_local(mhead, A, B, step, axis, angle, filter=None, max_distance=0.01):
    '''

    :param mhead: Multi Head Object
    :param A: Index of Head A
    :param B: Index of Head B
    :param step: step size
    :param axis: axis (can be carthsesian or angles)
    :param angle: angular(True) or  carthesian(False)
    :param filter: subset of  points of Head B that  are actually used
    :param max_distance: max distance between points on head B to head A
    :return:

    algorithm looks for local optimum: it scans over the given axis (angular or carthesian) and look for local optimum by either increasing or decreasing the middle position,
    tracking values at the boundaries

    '''

    max_distance_between_points = max_distance
    head1 = mhead.heads[mhead.head_id_from_frame_id(A)]
    head2 = mhead.heads[mhead.head_id_from_frame_id(B)]
    l = head2.xyz.shape[0]
    if filter is None:
        filter2 = np.random.random((l)) < 0.1
    else:
        filter2 = filter
    lower_count = 0
    higher_count = 0
    middle_count = 0
    value = 0
    init = 3
    bounce_count = 0
    move_down = True
    while lower_count >= middle_count or higher_count >= middle_count or init > 0:
        if lower_count > middle_count or init > 0:
            s = -step
            move_down = True
        elif higher_count > middle_count:
            s = step
            move_down = False
        elif lower_count == middle_count:
            s = -step
            if move_down == False:
                bounce_count += 1
            move_down = True
        elif higher_count == middle_count:
            s = step
            move_down = False
        else:
            break
        if bounce_count == 2:
            break
        init -= 1
        value = value + s

        # apply transformation, looking either at higher or lower position
        if angle:
            CoG = head2.xyz.mean(axis=0)
            R = calc_R(2 * s, axis)
            head2.xyz = np.dot(head2.xyz - CoG, R) + CoG
        else:
            t = np.array(axis) * 2 * s
            head2.xyz = head2.xyz + t
        xyz1 = head1.xyz
        xyz2 = head2.xyz[filter2]
        # calculate distances
        distances, indices = nearest_neighbor(xyz2, xyz1)
        # count how many distance are within range
        count = np.sum(distances < max_distance_between_points)
        # move back to middle position
        if angle:

            R = calc_R(-s, axis)
            head2.xyz = np.dot(head2.xyz - CoG, R) + CoG
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
    return value * np.array(axis), middle_count / np.size(xyz2), filter2


def refine_6D(mhead, A, B, angle_over_range=False, pos_over_range=False, filter=None):
    '''

    :param mhead: Multi Head Object
    :param A: Index of Head A
    :param B: Index of Head B
    :param angle_over_range: Whether we refine the angle locally or over a range
    :param pos_over_range: Whether we refine the position locally or over a range
    :param filter: subset of  points of Head B that  are actually used
    :return: the filter and the score (fraction of points of Head B that are within wide/fine max distance of a point on head A) after 6D refinement

    Transforms Head B over 3 axes and 3 angles to maximise number of overlapping points with Head A

    '''
    t = np.zeros(3)

    wide_s = 0.005
    fine_s = 0.002
    wide_phi = 2
    fine_phi = 0.4
    wide_max_distance = 0.02
    fine_max_distance = 0.01

    if angle_over_range:
        with tqdm(total=3) as progressbar:
            for axis in [[0, 1, 0], [1, 0, 0], [0, 0, 1]]:
                x, score, filter = refine_over_range(mhead, A, B, range=25, step=1, axis=axis, filter=filter,
                                                     angle=True, max_distance=wide_max_distance)
                t = t + x
                progressbar.set_description(
                    f"scan {A} to {B} phi:{t[0]: 3.1f},{t[1]: 3.1f},{t[2]: 3.1f} Scr={100 * score:2.2f}%")
                progressbar.update(1)
    else:
        with tqdm(total=3) as progressbar:
            for s in [wide_phi, fine_phi]:
                for axis in [[0, 1, 0], [1, 0, 0], [0, 0, 1]]:
                    x, score, filter = refine_local(mhead, A, B, s, axis, angle=True, filter=filter,
                                                    max_distance=fine_max_distance)
                    t = t + x
                    progressbar.set_description(
                        f"refine {A} to {B} phi:{t[0]: 3.1f},{t[1]: 3.1f},{t[2]: 3.1f} Scr={100 * score:2.2f}%")
                    progressbar.update(1)
    t = np.zeros(3)

    if pos_over_range:
        with tqdm(total=3) as progressbar:
            for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
                x, score, filter = refine_over_range(mhead, A, B, range=0.1, step=0.02, axis=axis, filter=filter,
                                                     angle=False, max_distance=wide_max_distance)
                t = t + 1000 * x
                progressbar.set_description(
                    f"scan {A} to {B} t  :{t[0]: 3.1f},{t[1]: 3.1f},{t[2]: 3.1f} Scr={100 * score:2.2f}%")
                progressbar.update(1)
    else:
        with tqdm(total=6) as progressbar:
            for s in [wide_s, fine_s]:
                for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
                    x, score, filter = refine_local(mhead, A, B, s, axis, angle=False, filter=filter,
                                                    max_distance=fine_max_distance)
                    t = t + 1000 * x
                    progressbar.set_description(
                        f"refine {A} to {B} t  :{t[0]: 3.1f},{t[1]: 3.1f},{t[2]: 3.1f} Scr={100 * score:2.2f}%")
                    progressbar.update(1)

    return filter, score
