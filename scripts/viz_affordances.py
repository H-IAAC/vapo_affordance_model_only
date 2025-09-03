import os

import cv2
import hydra
import numpy as np
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf
import tqdm

from vapo.affordance.dataset_creation.core.utils import get_files_regex
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
        self.npz_files, self.img_dict = self.get_filenames()
        self.target_height_override = self.cfg.target_height_override
        self.find_handle_by_depth = self.cfg.find_handle_by_depth
        self.out_shape = None #general for all image sizes
    

    # Load model based on hydra config file
    def init_model(self, cfg):
        original_dir = hydra.utils.get_original_cwd()
        run_dir = os.path.join(original_dir, cfg.test.folder_name)
        run_dir = os.path.abspath(run_dir)
        return load_from_hydra(os.path.join(run_dir, ".hydra/config.yaml"), cfg)


    def run(self):
        if len(self.npz_files) > 0:
            for filename in tqdm.tqdm(self.npz_files):
                print(filename)
                rgb_img, d_img = self.unpack_npz(filename)
                self.compute_aff_target(rgb_img, d_img, filename)
                cv2.waitKey(0)
        else:
            print("No NPZ files found.")

        cv2.destroyAllWindows()
        if len(self.img_dict) > 0:
            for base, paths in tqdm.tqdm(self.img_dict.items()):
                filename_rgb = paths['rgb']
                filename_depth = paths['depth']
                rgb_img = cv2.cvtColor(cv2.imread(filename_rgb), cv2.COLOR_BGR2RGB)

                d_img = cv2.imread(filename_depth, cv2.IMREAD_UNCHANGED)
                if d_img is None:
                    d_img = np.zeros(rgb_img.shape[:2], dtype=np.float32)  # fallback
                    print(f"Warning: Depth image use of {filename_depth} failed. Using fallback.")

                elif d_img.dtype == np.uint16:
                    d_img = d_img.astype(np.float32) / 1000.0  # meter conversion
                
                self.compute_aff_target(rgb_img, d_img, filename_rgb)
                cv2.waitKey(0)


        else:
            print("No RGB files found.")
        return

    def compute_aff_target(self, rgb_img, d_img, filename, return_data = False):
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

        # target_pos is an element of world_pts; world_pts is a list
        # target_pos is a numpy.ndarray with 3 numbers of float64 type
        # no_target is a boolean
        res = self.compute_target(rgb_img, d_img, centers, mask, aff_probs, object_masks)
        target_pos, no_target, world_pts, target_img = res

        # Show info on terminal
        rounded_world_pts = [np.round(pt, 2) for pt in world_pts]
        if not no_target:
            print("\nTarget position: ", target_pos, "\nNumber of world points: ", len(world_pts), "\nWorld points: ")
        else:
            print("\nNo target found. Number of world points: ", len(world_pts), "\nWorld points: ")
        for pt in rounded_world_pts:
            print(pt)

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
            )
        if return_data:
            return target_pos, no_target, world_pts

        return

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
        return rgb_img, d_img

    # Data folder example:
    # Script may be called with parameter datasets/playdata/viz_affordances/input_files
    # Then, inside input files, there must be <cam_type>/<file_type>/<files.ext>
    # as in static/rgb/0001.png
    def get_filenames(self):
        data_dirs = self.cfg.data_dir
        cam_type = self.cam_type
        npz_files = []
        rgb_files = []
        depth_files = []

        # Ensure data_dirs is always a list
        if not isinstance(data_dirs, (list, ListConfig)):
            data_dirs = [data_dirs]

        for dir_i in data_dirs:
            dir_i = get_abs_path(dir_i)
            if not os.path.exists(dir_i):
                print("Path does not exist: %s" % dir_i)
                continue

            for ext in ["npz", "jpg", "png"]:
                for file_type in ["npz", "rgb", "depth"]:
                    search_str = f"**/{cam_type}*/{file_type}/*.{ext}"
                    found_files = get_files_regex(dir_i, search_str, recursive=True, print_warning=False)
                    if file_type == "npz" and ext == "npz":
                        npz_files += found_files
                    elif file_type == "rgb" and ext in ["jpg", "png"]:
                        rgb_files += found_files
                    elif file_type == "depth" and ext in ["jpg", "png"]:
                        depth_files += found_files

        # Build dict assuming rgb_files and depth_files are ordered and correspond
        img_dict = {}
        for rgb_path, depth_path in zip(rgb_files, depth_files):
            base = os.path.splitext(os.path.basename(rgb_path))[0]
            img_dict[base] = {'rgb': rgb_path, 'depth': depth_path}

        return npz_files, img_dict

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
        ):

        def resize_for_display(img, max_dim=480):
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                return cv2.resize(img, new_size)
            return img
    
        imgs = [
            resize_for_display(cv2.cvtColor(affordance_mask, cv2.COLOR_GRAY2BGR)),
            resize_for_display(aff_over_img[:, :, ::-1]),
            resize_for_display(flow_img[:, :, ::-1]),
            resize_for_display(flow_over_img[:, :, ::-1]),
        ]
        if not no_target:
            imgs.append(resize_for_display(target_img[:, :, ::-1]))

        # Calculate number of columns per row
        n = len(imgs)
        num_rows = 3
        cols = (n + num_rows - 1) // num_rows  # Ceiling division

        # Pad with blank images if needed
        blank = np.zeros_like(imgs[0])
        imgs_padded = imgs + [blank] * (num_rows * cols - n)

        # Build rows
        rows = []
        for i in range(num_rows):
            row_imgs = imgs_padded[i*cols:(i+1)*cols]
            row = np.hstack(row_imgs)
            rows.append(row)

        # Stack rows vertically
        cv2.imshow("Affordance Visualizations", np.vstack(rows))
        return


# Run the code: python ./scripts/viz_affordances.py data_dir=datasets/playdata/demo_affordance/npz_files
@hydra.main(config_path="../config", config_name="viz_affordances")
def main(cfg):
    viz = VizAffordances(cfg)
    viz.run()


if __name__ == "__main__":
    main()
