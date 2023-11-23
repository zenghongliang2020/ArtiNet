import os
import sys
import sapien.core as sapien
from sapien.core import Pose
import numpy as np
import pandas as pd
from PIL import Image
import argparse

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
path = os.path.join(rootPath, 'code')
sys.path.append(path)
from utils import process_angle_limit, get_random_number, get_link_mask, get_point_cloud_from_mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cat_index', default='11,24,31,36', help='train cat index [default: 11, 24, 31, 36]')
    parser.add_argument('--test_cat_index', default='45', help='test cat index [default: 45]')
    parser.add_argument('--model_name', default='movable', help='Training model name [default: movable]')
    parser.add_argument('--create_train', action='store_true', help='create train data or not')
    parser.add_argument('--create_test', action='store_false', help='create test data or not')
    parser.add_argument('--re_split', action='store_false', help='Repartitioning the dataset')
    args = parser.parse_args()
    return args


args = parse_args()
ALL_CAT_LIST = ['Bottle', 'Box', 'Bucket', 'Camera', 'Cart', 'Chair', 'Clock', 'CoffeeMachine',
                'Dishwasher', 'Dispenser', 'Display', 'Door', 'Eyeglasses', 'Fan', 'Faucet',
                'FoldingChair', 'Globe', 'Kettle', 'Keyboard', 'KitchenPot', 'Knife', 'Lamp',
                'Laptop', 'Lighter', 'Microwave', 'Mouse', 'Oven', 'Pen', 'Phone', 'Pliers',
                'Printer', 'Refrigerator', 'Remote', 'Safe', 'Scissors', 'Stapler', 'StorageFurniture',
                'Suitcase', 'Switch', 'Table', 'Toaster', 'Toilet', 'TrashCan', 'USB', 'WashingMachine', 'Window']
TRAIN_CAT_LIST = [ALL_CAT_LIST[int(x)] for x in args.train_cat_index.split(',')]
TEST_CAT_LIST = [ALL_CAT_LIST[int(x)] for x in args.test_cat_index.split(',')]
MODEL_NAME = args.model_name
STATS_PATH = os.path.join(rootPath, 'stats')
MAP_DIR = os.path.join(STATS_PATH, 'all_shapeid_cat_map.txt')
train_file = os.path.join(STATS_PATH, 'train_%s_cat_idx[%s].txt' % (MODEL_NAME, args.train_cat_index))
val_file = os.path.join(STATS_PATH, 'val_%s_cat_idx[%s].txt' % (MODEL_NAME, args.train_cat_index))
test_file = os.path.join(STATS_PATH, 'test_%s_cat_idx[%s].txt' % (MODEL_NAME, args.test_cat_index))


def split_train_val_test():
    if args.create_train:
        with open(train_file, 'w') as fr:
            fr.write('')
        with open(val_file, 'w') as fr:
            fr.write('')
        with open(MAP_DIR, 'r') as fin:
            for l in fin.readlines():
                shape_id, cat = l.rstrip().split()
                if cat not in TRAIN_CAT_LIST:
                    continue
                if np.random.rand() < 0.7:
                    with open(train_file, 'a') as f:
                        f.write(f"{shape_id} {cat}\n")
                else:
                    with open(val_file, 'a') as f:
                        f.write(f"{shape_id} {cat}\n")
    if args.create_test:
        with open(test_file, 'w') as fr:
            fr.write('')
        with open(MAP_DIR, 'r') as fin:
            for l in fin.readlines():
                shape_id, cat = l.rstrip().split()
                if cat not in TEST_CAT_LIST:
                    continue
                with open(test_file, 'a') as f:
                    f.write(f"{shape_id} {cat}\n")


def load_object(object, object_position_offset):
    pose = Pose([object_position_offset, 0, 0], [1, 0, 0, 0])
    object.set_pose(pose)
    all_link_ids = [l.get_id() for l in object.get_links()]
    movable_link_ids = []
    joint_angles = []
    for j in object.get_joints():
        # set joint property
        j.set_drive_property(stiffness=0, damping=10)
        if j.get_dof() == 1:
            movable_link_ids.append(j.get_child_link().get_id())
            l = process_angle_limit(j.get_limits()[0, 0])
            r = process_angle_limit(j.get_limits()[0, 1])
            joint_angles.append(float(get_random_number(l, r)))
    object.set_qpos(joint_angles)
    return joint_angles, movable_link_ids, all_link_ids


def creat_movable_data(data_type, out_dir):
    train_shape_list, val_shape_list, test_shape_list, cur_shape_list = [], [], [], []
    shape_cat_dict = {}
    if data_type == 'train':
        out_dir = os.path.join(out_dir, 'movable_train_cat_idx[%s]' % args.train_cat_index)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(train_file, 'r') as fin:
            for l in fin.readlines():
                shape_id, cat = l.rstrip().split()
                shape_cat_dict[shape_id] = cat
                train_shape_list.append(shape_id)
                cur_shape_list.append(shape_id)
        print('the length of train shape list is %s' % len(train_shape_list))

    elif data_type == 'val':
        out_dir = os.path.join(out_dir, 'movable_val_cat_idx[%s]' % args.train_cat_index)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(val_file, 'r') as fin:
            for l in fin.readlines():
                shape_id, cat = l.rstrip().split()
                shape_cat_dict[shape_id] = cat
                val_shape_list.append(shape_id)
                cur_shape_list.append(shape_id)
        print('the length of val shape list is %s' % len(val_shape_list))

    elif data_type == 'test':
        out_dir = os.path.join(out_dir, 'movable_test_cat_idx[%s]' % args.test_cat_index)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(test_file, 'r') as fin:
            for l in fin.readlines():
                shape_id, cat = l.rstrip().split()
                shape_cat_dict[shape_id] = cat
                val_shape_list.append(shape_id)
                cur_shape_list.append(shape_id)
        print('the length of test shape list is %s' % len(val_shape_list))
    else:
        raise ValueError('ERROR: data_type %s is unkown!' % data_type)

    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    data_num = 0
    for shape_id in cur_shape_list:
        for cam_posx in [2, -2]:
            for cam_posy in [2, -2]:
                for cam_posz in [2, 3]:

                    scene = engine.create_scene()
                    scene.set_timestep(1 / 100.0)
                    scene.set_ambient_light([0.5, 0.5, 0.5])
                    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
                    scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
                    scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
                    scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)
                    loader = scene.create_urdf_loader()
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

                    camera_mount_actor = scene.create_actor_builder().build_kinematic()
                    camera.set_parent(parent=camera_mount_actor, keep_pose=False)
                    # Compute the camera pose by specifying forward(x), left(y) and up(z)
                    cam_pos = np.array([cam_posx, cam_posy, cam_posz])
                    forward = -cam_pos / np.linalg.norm(cam_pos)
                    left = np.cross([0, 0, 1], forward)
                    left = left / np.linalg.norm(left)
                    up = np.cross(forward, left)
                    mat44 = np.eye(4)
                    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
                    mat44[:3, 3] = cam_pos
                    camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))

                    # urdf_path = '../data/original_sapien_dataset/%s/mobility_vhacd.urdf' % shape_id
                    urdf_path = '../data/partnet-mobility-v0/dataset/%s/mobility.urdf' % shape_id
                    material = scene.create_physical_material(4, 4, 0.01)
                    object = loader.load(urdf_path, {"material": material})
                    joint_angles, movable_link_ids, all_link_ids = load_object(object, 0.0)

                    scene.step()  # make everything set
                    scene.update_render()
                    camera.take_picture()

                    # ---------------------------------------------------------------------------- #
                    # RGBA & Pcs
                    # ---------------------------------------------------------------------------- #
                    rgba = camera.get_float_texture('Color')  # [H, W, 4]
                    # An alias is also provided
                    # rgba = camera.get_color_rgba()  # [H, W, 4]
                    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
                    rgba_pil = Image.fromarray(rgba_img)
                    link_mask = get_link_mask(all_link_ids, camera)
                    position = camera.get_float_texture('Position')

                    # Save
                    for id in all_link_ids:
                        link_position = get_point_cloud_from_mask(position, link_mask, id)
                        link_points_opengl = link_position[..., :3][link_position[..., 3] < 1]
                        if link_points_opengl.shape[0] < 3000:
                            continue
                        save_dir = os.path.join(out_dir, '%s_%s_(%s,%s,%s)_id%s' % (
                        shape_cat_dict[shape_id], shape_id, cam_posx, cam_posy, cam_posz, id))
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        # Save RGBA
                        save_rgb_path = os.path.join(save_dir, 'rgb.png')
                        rgba_pil.save(save_rgb_path)
                        # Save Pcs
                        pc_save_path = os.path.join(save_dir, 'pcs.xyz')
                        np.savetxt(pc_save_path, link_points_opengl, delimiter=' ', fmt='%.6f')
                        # Save Lable
                        save_lable_path = os.path.join(save_dir, 'labels.csv')
                        if id in movable_link_ids:
                            lable = {'movable': [1]}
                        else:
                            lable = {'movable': [0]}
                        df = pd.DataFrame(lable)
                        df.to_csv(save_lable_path, index=False)
                        data_num += 1
                        if data_num % 100 == 0:
                            print('now data num:%s' % data_num)

                scene.remove_articulation(object)
                scene.remove_camera(camera)


if __name__ == '__main__':
    out_dir = '../data/'
    if args.re_split:
        split_train_val_test()
    if MODEL_NAME == 'movable':
        if args.create_train:
            creat_movable_data('train', out_dir)
            creat_movable_data('val', out_dir)
        if args.create_test:
            creat_movable_data('test', out_dir)
