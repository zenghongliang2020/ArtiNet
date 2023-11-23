import os
import json
import sapien.core as sapien
from sapien.core import Pose
import numpy as np
from utils import process_angle_limit, get_random_number
from PIL import Image, ImageColor
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from transforms3d.euler import mat2euler
from sapien.utils.viewer import Viewer
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--data_type', type=str, default='train')
parser.add_argument('--object_position_offset', type=float, default=0.0)
parser.add_argument('--out_dir', type=str, default='../data/')
args = parser.parse_args()


def load_object(object, object_position_offset):
    pose = Pose([object_position_offset, 0, 0], [1, 0, 0, 0])
    object.set_pose(pose)
    all_link_ids = [l.get_id() for l in object.get_links()]
    movable_link_ids = []
    joint_angles = []
    joint_idx = -1
    for j in object.get_joints():
        # set joint property
        j.set_drive_property(stiffness=0, damping=10)
        if j.get_dof() == 1:
            movable_link_ids.append(j.get_child_link().get_id())
            joint_idx = len(joint_angles)
            l = process_angle_limit(j.get_limits()[0, 0])
            r = process_angle_limit(j.get_limits()[0, 1])
            joint_angles.append(float(get_random_number(l, r)))
    object.set_qpos(joint_angles)
    return joint_angles, movable_link_ids


def get_movable_link_mask(link_ids, camera):
    link_seg = camera.get_uint32_texture('Segmentation')
    link_seg_id = link_seg[..., 1]
    link_mask = np.zeros((link_seg_id.shape[0], link_seg_id.shape[1])).astype(np.uint8)
    for idx, lid in enumerate(link_ids):
        cur_link_pixels = int(np.sum(link_seg_id==lid))
        if cur_link_pixels > 0:
            link_mask[link_seg_id == lid] = lid + 1
    return link_mask


def get_joint_origins(object, camera):
    joint_pos_p = []
    joint_axes = []
    for j in object.get_joints():
        if j.get_dof() == 1:
            pos = j.get_global_pose()
            mat = pos.to_transformation_matrix()
            joint_ax = [float(-mat[0, 0]), float(-mat[1, 0]), float(mat[2, 0])]
            joint_pos_p.append(pos.p.tolist())
            joint_axes.append(joint_ax)
    if joint_pos_p is None:
        raise ValueError('joint origins error!')

    return joint_pos_p, joint_axes

def get_point_cloud_from_mask(points, link_mask, link_id):
    link_pixels = np.nonzero(link_mask == link_id + 1)
    link_point_cloud = points[link_pixels]
    print(link_point_cloud.shape)
    return link_point_cloud

def creat_data(data_type):
    train_shape_list, val_shape_list, cur_shape_list = [], [], []
    train_file_dir = "../stats/train_MOVABLE_train_data_list.txt"
    val_file_dir = "../stats/train_MOVABLE_val_data_list.txt"
    all_shape_list = []
    all_cat_list = ['StorageFurniture', 'Microwave', 'Refrigerator', 'Door']
    len_shape, len_train_shape, len_val_shape = {}, {}, {}
    shape_cat_dict = {}

    if data_type == 'train':
        out_dir = os.path.join(args.out_dir, 'train')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(train_file_dir, 'r') as fin:
            for l in fin.readlines():
                shape_id, cat = l.rstrip().split()
                if cat not in all_cat_list:
                    continue
                train_shape_list.append(shape_id)
                cur_shape_list.append(shape_id)
                all_shape_list.append(shape_id)
                shape_cat_dict[shape_id] = cat

    elif data_type == 'val':
        out_dir = os.path.join(args.out_dir, 'val')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(val_file_dir, 'r') as fin:
            for l in fin.readlines():
                shape_id, cat = l.rstrip().split()
                if cat not in all_cat_list:
                    continue
                val_shape_list.append(shape_id)
                cur_shape_list.append(shape_id)
                all_shape_list.append(shape_id)
                shape_cat_dict[shape_id] = cat

    else:
        raise ValueError('ERROR: data_type %s is unkown!' % data_type)

    engine = sapien.Engine()
    renderer = sapien.SapienRenderer(offscreen_only=True)
    engine.set_renderer(renderer)

    for shape_id in cur_shape_list:

        scene = engine.create_scene()
        scene.set_timestep(1 / 100.0)
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)
        loader = scene.create_urdf_loader()

        urdf_path = '../data/original_sapien_dataset/%s/mobility_vhacd.urdf' % shape_id
        material = scene.create_physical_material(4, 4, 0.01)
        object = loader.load(urdf_path, {"material": material})
        joint_angles, movable_link_ids = load_object(object, 0.0)
        print(np.degrees(joint_angles))

        # ---------------------------------------------------------------------------- #
        # Camera
        # ---------------------------------------------------------------------------- #
        near, far = 0.1, 100
        width, height = 640, 480
        camera = scene.add_camera(
            name="camera",
            width=width,
            height=height,
            fovy=np.deg2rad(35),
            near=near,
            far=far,
        )
        camera.set_pose(sapien.Pose(p=[0, 0, 0]))

        #print('Intrinsic matrix\n', camera.get_intrinsic_matrix())

        camera_mount_actor = scene.create_actor_builder().build_kinematic()
        camera.set_parent(parent=camera_mount_actor, keep_pose=False)

        # Compute the camera pose by specifying forward(x), left(y) and up(z)
        cam_pos = np.array([-2, -2, 3])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))

        scene.step()  # make everything set
        scene.update_render()
        camera.take_picture()

        # ---------------------------------------------------------------------------- #
        # RGBA
        # ---------------------------------------------------------------------------- #
        rgba = camera.get_float_texture('Color')  # [H, W, 4]
        # An alias is also provided
        # rgba = camera.get_color_rgba()  # [H, W, 4]
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        rgba_pil = Image.fromarray(rgba_img)
        save_dir = os.path.join(out_dir, '%s_%s' % (cat, shape_id))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'rgb.png')
        rgba_pil.save(save_path)

        # ---------------------------------------------------------------------------- #
        # Points
        # ---------------------------------------------------------------------------- #
        movable_link_mask = get_movable_link_mask(movable_link_ids, camera)
        position = camera.get_float_texture('Position')
        for i in movable_link_ids:
            link_position = get_point_cloud_from_mask(position, movable_link_mask, i)
            link_points_opengl = link_position[..., :3][link_position[..., 3] < 1]
            pc_save_dir = os.path.join(save_dir, '%d' %i)
            if not os.path.exists(pc_save_dir):
                os.makedirs(pc_save_dir)
            pc_save_path = os.path.join(pc_save_dir, 'pcs.xyz')
            np.savetxt(pc_save_path, link_points_opengl, delimiter=' ', fmt='%.6f')
            link_points_camera = link_points_opengl[..., [2, 0, 1]] * [-1, -1, 1]
            points_color = rgba[position[..., 3] < 1][..., :3]
            model_matrix = camera.get_model_matrix()
            points_world = link_points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]

            pcd = o3d.geometry.PointCloud()
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
            pcd.points = o3d.utility.Vector3dVector(points_world)
            pcd.colors = o3d.utility.Vector3dVector(points_color)

            # o3d.visualization.draw_geometries([pcd, coord_frame])

        # ---------------------------------------------------------------------------- #
        # joint_angel joint_position
        # ---------------------------------------------------------------------------- #
        joint_pos_p, joint_axes = get_joint_origins(object, camera)
        model_matrix = camera.get_model_matrix()
        joint_pos_p_camera = (joint_pos_p - model_matrix[:3, 3]) @ np.linalg.inv(model_matrix[:3, :3]).T
        joint_axes_camera = joint_axes @ np.linalg.inv(model_matrix[:3, :3])
        joint_axes_t = joint_axes_camera @ model_matrix[:3, :3]

        unit_u = joint_axes / np.linalg.norm(joint_axes, axis=1, keepdims=True)
        unit_v = joint_axes_t / np.linalg.norm(joint_axes_t, axis=1, keepdims=True)
        #
        # angles = np.arccos(np.einsum('ij,ij->i', unit_u, unit_v))
        # print(np.degrees(angles))
        # #
        # print('origin------------------------')
        # print(joint_axes)
        # print('trans-------------------------')
        # print(joint_axes_camera)
        # print('tran-ori----------------------')
        # print(joint_axes_t)






        scene.remove_articulation(object)
        scene.remove_camera(camera)


creat_data(args.data_type)