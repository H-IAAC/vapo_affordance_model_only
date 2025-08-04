import json
import os

import cv2
import hydra
from hydra.utils import get_original_cwd
import numpy as np
from omegaconf.listconfig import ListConfig
import torch
import tqdm

from vapo.affordance.dataset_creation.core.utils import get_files, get_files_regex
from vapo.affordance.utils.img_utils import get_aff_imgs, transform_and_predict
from vapo.affordance.utils.utils import get_abs_path, get_transforms, load_from_hydra, torch_to_numpy


def get_filenames(data_dir, get_eval_files=False, cam_type=""):
    files = []
    np_comprez = False
    if isinstance(data_dir, ListConfig):
        for dir_i in data_dir:
            dir_i = get_abs_path(dir_i)
            if not os.path.exists(dir_i):
                print("Path does not exist: %s" % dir_i)
                continue
            files += get_files(dir_i, "npz")
            if len(files) > 0:
                np_comprez = True
            files += get_files(dir_i, "jpg")
            files += get_files(dir_i, "png")
    else:
        if get_eval_files:
            files, np_comprez = get_validation_files(data_dir, cam_type)
        else:
            data_dir = get_abs_path(data_dir)
            if not os.path.exists(data_dir):
                print("Path does not exist: %s" % data_dir)
                return [], False
            for ext in ["npz", "jpg", "png"]:
                search_str = "**/%s*/*.%s" % (cam_type, ext)
                files += get_files_regex(data_dir, search_str, recursive=True)
                if len(files) > 0 and ext == "npz":
                    np_comprez = True
    return files, np_comprez


# Load validation files for custom datase
def get_validation_files(data_dir, cam_type):
    data_dir = os.path.join(get_original_cwd(), data_dir)
    data_dir = os.path.abspath(data_dir)
    json_file = os.path.join(data_dir, "episodes_split.json")
    with open(json_file) as f:
        data = json.load(f)
    d = []
    for ep, imgs in data["validation"].items():
        im_lst = [data_dir + "/%s/data/%s.npz" % (ep, img_path) for img_path in imgs if cam_type in img_path]
        d.extend(im_lst)
    return d, True


# Run the code: python ./scripts/viz_affordances.py data_dir=datasets/playdata/demo_affordance/npz_files
@hydra.main(config_path="../config", config_name="viz_affordances")
def viz(cfg):
    # Create output directory if save_images
    if not os.path.exists(cfg.output_dir) and cfg.save_images:
        os.makedirs(cfg.output_dir)
    
    # Load model based on hydra config file
    original_dir = hydra.utils.get_original_cwd()
    run_dir = os.path.join(original_dir, cfg.test.folder_name)
    run_dir = os.path.abspath(run_dir)
    model, run_cfg = load_from_hydra(os.path.join(run_dir, ".hydra/config.yaml"), cfg)

    # Transforms
    if "cam_data" in cfg:
        cam_type = cfg.cam_data
    else:
        cam_type = run_cfg.dataset.cam

    transforms_cfg = run_cfg.affordance.transforms
    img_size = run_cfg.dataset.img_resize[cam_type]
    
    aff_transforms = get_transforms(transforms_cfg.validation, img_size)

    # Image files input and preprocessing
    files, np_comprez = get_filenames(cfg.data_dir, get_eval_files=cfg.get_eval_files, cam_type=cam_type)
    out_shape = (cfg.out_size, cfg.out_size)
    for filename in tqdm.tqdm(files):
        if np_comprez:
            data = np.load(filename)
            rgb_img = data["frame"]
            d_img = data["d_img"] # depth image
            # gt == ground truth
            gt_centers = data["centers"]
            gt_mask = data["mask"].squeeze()
            gt_mask = (gt_mask / 255).astype("uint8")
            gt_directions = data["directions"]
        else:
            rgb_img = cv2.imread(filename, cv2.COLOR_BGR2RGB)
            out_shape = np.shape(rgb_img)[:2]
        
        res = transform_and_predict(model, aff_transforms, rgb_img)
        centers, mask, directions, aff_probs, object_masks = res
        affordance_mask, aff_over_img, flow_over_img, flow_img = get_aff_imgs(
            rgb_img,
            mask,
            directions,
            centers,
            out_shape,
            cam=cam_type,
            n_classes=aff_probs.shape[-1],
        )



        '''
        TODO: TRANSFER TARGET PREDICTION AND SELECTION CODE
        FROM _compute_target_aff() AT target_search.py STARTING
        ON LINE 155
        '''

        # Calculate ground truth, if file contains this data
        if np_comprez:
            gt_aff, gt_aff_img, gt_flow_img, gt_flow = get_aff_imgs(
            rgb_img,
            gt_mask.squeeze(),
            gt_directions,
            gt_centers,
            out_shape,
            cam=cam_type,
            n_classes=model.n_classes,
        )

        # Save and show
        if cfg.save_images:
            _, tail = os.path.split(filename)
            split = tail.split(".")
            name = "".join(split[:-1])

            output_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.output_dir)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, name + ".png")

            cv2.imwrite(output_file, flow_over_img[:, :, ::-1])  # Flip channel order
            print("Saved %s" % output_file)

        if cfg.imshow:
            cv2.imshow("Affordance mask", affordance_mask[:, ::-1])
            cv2.imshow("Affordance over image", aff_over_img[:, :, ::-1])
            cv2.imshow("Flow", flow_img[:, :, ::-1])
            cv2.imshow("Flow over image", flow_over_img[:, :, ::-1])
            if np_comprez:
                cv2.imshow("gt mask", gt_aff[:, ::-1])
                cv2.imshow("gt mask over image", gt_aff_img[:, :, ::-1])
                cv2.imshow("gt flow", gt_flow[:, :, ::-1])
                cv2.imshow("gt flow over image", gt_flow_img[:, :, ::-1])
            cv2.waitKey(0)


if __name__ == "__main__":
    viz()
