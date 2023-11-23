import numpy as np


def get_link_mask(link_ids, camera):
    """

    :param link_ids:
    :param camera:
    :return:
    """
    link_seg = camera.get_uint32_texture('Segmentation')
    link_seg_id = link_seg[..., 1]
    link_mask = np.zeros((link_seg_id.shape[0], link_seg_id.shape[1])).astype(np.uint8)
    for idx, lid in enumerate(link_ids):
        cur_link_pixels = int(np.sum(link_seg_id==lid))
        if cur_link_pixels > 0:
            link_mask[link_seg_id == lid] = lid + 1
    return link_mask


def get_point_cloud_from_mask(points, link_mask, link_id):
    link_pixels = np.nonzero(link_mask == link_id + 1)
    link_point_cloud = points[link_pixels]
    return link_point_cloud


def process_angle_limit(x):
    if np.isneginf(x):
        x = -10
    if np.isinf(x):
        x = 10
    return x


def get_random_number(l, r):
    return np.random.rand() * (r - l) + l