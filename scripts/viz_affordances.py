import json
import os

import cv2
import hydra
from hydra.utils import get_original_cwd
import numpy as np
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf
import tqdm

from vapo.affordance.dataset_creation.core.utils import get_files, get_files_regex
from vapo.affordance.utils.img_utils import get_aff_imgs, resize_center, transform_and_predict
from vapo.affordance.utils.utils import get_abs_path, get_transforms, load_from_hydra


class VizAffordances:

    def __init__(self, cfg):
        self.model, self.cfg = self.init_model(cfg)

        # Create output directory if save_images
        if not os.path.exists(self.cfg.output_dir) and self.cfg.save_images:
            os.makedirs(self.cfg.output_dir)
        
        self.cam_type = self.cfg.dataset.cam
        cam_dir = get_abs_path(self.cfg.paths.camera_config)
        cam_cfg = OmegaConf.load(cam_dir + "/tabletop_sideview.yaml")
        self.cam = hydra.utils.instantiate(cam_cfg)
        
        # Transforms
        self.img_size = self.cfg.dataset.img_resize[self.cam_type]
        self.aff_transforms = get_transforms(self.cfg.affordance.transforms.validation, self.img_size)
        # Image files input and preprocessing
        self.files, self.np_comprez = self.get_filenames()
        self.calculate_gt = self.cfg.calculate_gt
        self.target_height_override = self.cfg.target_height_override
        self.find_handle_by_depth = self.cfg.find_handle_by_depth
        self.out_shape = (self.cfg.out_size, self.cfg.out_size)
    

    # Load model based on hydra config file
    def init_model(self, cfg):
        original_dir = hydra.utils.get_original_cwd()
        run_dir = os.path.join(original_dir, cfg.test.folder_name)
        run_dir = os.path.abspath(run_dir)
        return load_from_hydra(os.path.join(run_dir, ".hydra/config.yaml"), cfg)


    def run(self):
        for filename in tqdm.tqdm(self.files):

            if self.np_comprez:
                rgb_img, d_img, gt_centers, gt_mask, gt_directions = self.unpack_npz(filename)
            else:
                rgb_img = cv2.imread(filename, cv2.COLOR_BGR2RGB)
                self.out_shape = np.shape(rgb_img)[:2]
            
            # Affordance prediction
            res = transform_and_predict(self.model, self.aff_transforms, rgb_img)
            centers, mask, directions, aff_probs, object_masks = res
            affordance_mask, aff_over_img, flow_over_img, flow_img = get_aff_imgs(
                rgb_img,
                mask,
                directions,
                centers,
                self.out_shape,
                cam=self.cam_type,
                n_classes=aff_probs.shape[-1],
            )

            # Calculate for ground truth if applicable
            if self.np_comprez and self.calculate_gt:
                gt_aff, gt_aff_img, gt_flow_img, gt_flow = get_aff_imgs(
                rgb_img,
                gt_mask.squeeze(),
                gt_directions,
                gt_centers,
                self.out_shape,
                cam=self.cam_type,
                n_classes=self.model.n_classes,
            )
            else:
                gt_aff = None
                gt_aff_img = None
                gt_flow_img = None
                gt_flow = None

            res = self.compute_target(rgb_img, d_img, centers, mask, aff_probs, object_masks)
            target_pos, no_target, world_pts, target_img = res
            print("Target pos: ", target_pos, " Num_world_pts: ", len(world_pts), " World pts: ", world_pts)

            # Save and show
            if self.cfg.save_images:
                self.save_images(filename, flow_over_img)
            if self.cfg.imshow:
                self.show_images(
                    affordance_mask,
                    aff_over_img,
                    flow_img,
                    flow_over_img,
                    target_img,
                    no_target,
                    (self.np_comprez and self.calculate_gt),
                    gt_aff,
                    gt_aff_img,
                    gt_flow,
                    gt_flow_img,
                )


    # Derived from vapo/agent/core/target_search.py
    def compute_target(self, rgb_img, d_img, centers, mask, aff_probs, object_masks):

        # No center detected
        no_target = len(centers) <= 0
        if no_target:
            return np.array(None), no_target, [], None

        max_robustness = 0
        obj_class = np.unique(object_masks)[1:]
        obj_class = obj_class[obj_class != 0]  # remove background class

        # World coords
        world_pts = []
        pred_shape = mask.shape[:2]
        new_shape = d_img.shape[:2]
        for o in centers:
            o = resize_center(o, pred_shape, new_shape)
            world_pt = self.get_world_pt(o, self.cam, d_img)
            world_pts.append(world_pt)

        # If flag is true on config file, select highest point instead of most robust
        if self.target_height_override:
            max_height = -1
            target_idx = 0
            for i, pt in enumerate(world_pts):
                if pt[-1] > max_height:
                    target_pos = pt
                    max_height = pt[-1]
                    target_idx = i

        
        # Look for most likely/robust center
        else:
            target_idx = 0
            for i, o in enumerate(centers):
                # Mean prob of being class 1 (foreground)
                robustness = np.mean(aff_probs[object_masks == obj_class[i], 1])
                if robustness > max_robustness:
                    max_robustness = robustness
                    target_idx = i
            target_pos = world_pts[target_idx]


        # Recover target
        v, u = resize_center(centers[target_idx], pred_shape, new_shape)
        target_img = cv2.drawMarker(
            np.array(rgb_img),
            (u, v),
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=15,
            thickness=3,
            line_type=cv2.LINE_AA,
        )
        
        return target_pos, no_target, world_pts, target_img


    def get_world_pt(self, pixel, cam, depth):
        x = pixel
        v, u = pixel
        if self.find_handle_by_depth:
            # Searches in an n×n window around the original pixel
            # for the point with minimum depth (closest object surface).
            n = 10
            depth_window = depth[x[0] - n : x[0] + n, x[1] - n : x[1] + n]
            proposal = np.argwhere(depth_window == np.min(depth_window))[0]
            v = x[0] - n + proposal[0]
            u = x[1] - n + proposal[1]
        world_pt = np.array(cam.deproject([u, v], depth))
        return world_pt


    def unpack_npz(self, filename):
        data = np.load(filename)
        rgb_img = data["frame"]
        d_img = data["d_img"] # depth image
        if self.calculate_gt:
            # gt == ground truth
            gt_centers = data["centers"]
            gt_mask = data["mask"].squeeze()
            gt_mask = (gt_mask / 255).astype("uint8")
            gt_directions = data["directions"]
        else:
            gt_centers = None
            gt_mask = None
            gt_directions = None
        return rgb_img, d_img, gt_centers, gt_mask, gt_directions


    def get_filenames(self):
        data_dir = self.cfg.data_dir
        get_eval_files = self.cfg.get_eval_files
        cam_type = self.cam_type
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
                files, np_comprez = self.get_validation_files(self)
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


    # Load validation files for custom dataset
    def get_validation_files(self):
        data_dir = self.cfg.data_dir
        cam_type = self.cam_type
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


    def save_images(self, filename, img):
        _, tail = os.path.split(filename)
        split = tail.split(".")
        name = "".join(split[:-1])

        output_dir = os.path.join(hydra.utils.get_original_cwd(), self.cfg.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, name + ".png")

        cv2.imwrite(output_file, img[:, :, ::-1])  # Flip channel order
        print("Saved %s" % output_file)
        return


    def show_images(
            self,
            affordance_mask,
            aff_over_img,
            flow_img,
            flow_over_img,
            target_img,
            no_target,
            flag_gt,
            gt_aff,
            gt_aff_img,
            gt_flow,
            gt_flow_img
        ):
        cv2.imshow("Affordance mask", affordance_mask[:, ::-1])
        cv2.imshow("Affordance over image", aff_over_img[:, :, ::-1])
        cv2.imshow("Flow", flow_img[:, :, ::-1])
        cv2.imshow("Flow over image", flow_over_img[:, :, ::-1])
        if not no_target:
            cv2.imshow("Target image", target_img[:, :, ::-1])

        if flag_gt:
            cv2.imshow("gt mask", gt_aff[:, ::-1])
            cv2.imshow("gt mask over image", gt_aff_img[:, :, ::-1])
            cv2.imshow("gt flow", gt_flow[:, :, ::-1])
            cv2.imshow("gt flow over image", gt_flow_img[:, :, ::-1])
        cv2.waitKey(0)
        return


# Run the code: python ./scripts/viz_affordances.py data_dir=datasets/playdata/demo_affordance/npz_files
@hydra.main(config_path="../config", config_name="viz_affordances")
def main(cfg):
    viz = VizAffordances(cfg)
    viz.run()


if __name__ == "__main__":
    main()
