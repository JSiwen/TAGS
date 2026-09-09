#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import os
import shutil
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass


def _mean_sq_dist_to_knn_cpu(points_np: np.ndarray, k: int = 3) -> torch.Tensor:
    """
    Same intent as simple_knn.distCUDA2: per-point mean of k smallest squared
    distances to other points (SimpleKNN uses k=3). CPU path for drivers/GPU
    that crash (e.g. SIGBUS) in distCUDA2; set TAGGS_CPU_KNN_INIT=1.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError as e:
        raise RuntimeError(
            "TAGGS_CPU_KNN_INIT requires scipy. Install with: pip install scipy"
        ) from e

    pts = np.ascontiguousarray(points_np, dtype=np.float64)
    tree = cKDTree(pts)
    try:
        dists, _ = tree.query(pts, k=k + 1, workers=-1)
    except TypeError:
        dists, _ = tree.query(pts, k=k + 1)
    neigh = dists[:, 1 : k + 1]
    mean_sq = (neigh.astype(np.float64) ** 2).mean(axis=1)
    return torch.from_numpy(mean_sq.astype(np.float32))


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.densify_max_new_points_per_iter = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        if os.environ.get("TAGGS_CPU_KNN_INIT", "").lower() in ("1", "true", "yes"):
            print("[ TAGGS_CPU_KNN_INIT ] Using scipy cKDTree for initial neighbor distances (distCUDA2 skipped).")
            dist2 = torch.clamp_min(_mean_sq_dist_to_knn_cpu(np.asarray(pcd.points)).cuda(), 0.0000001)
        else:
            # Reuse fused_point_cloud — avoids a second full GPU copy from pts_cpu.cuda() before SimpleKNN runs.
            try:
                dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.contiguous()), 0.0000001)
            except (RuntimeError, MemoryError) as e:
                msg = str(e).lower()
                if (
                    "out of memory" in msg
                    or "bad_alloc" in msg
                    or "cudaerrormemoryallocation" in msg.replace(" ", "")
                ):
                    torch.cuda.empty_cache()
                    print(
                        "[GaussianModel] distCUDA2 failed (likely GPU OOM); "
                        "falling back to CPU kNN init (same as TAGGS_CPU_KNN_INIT=1). scipy required."
                    )
                    dist2 = torch.clamp_min(_mean_sq_dist_to_knn_cpu(np.asarray(pcd.points)).cuda(), 0.0000001)
                else:
                    raise
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.densify_max_new_points_per_iter = max(
            0, int(getattr(training_args, "densify_max_new_points_per_iter", 0))
        )
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        # Rough size check (vertex + normals + f_dc + f_rest + opacity + scale + rot as float32)
        try:
            n = int(self._xyz.shape[0])
            est_bytes = n * 4 * (
                3 + 3 + self._features_dc.shape[1] * self._features_dc.shape[2]
                + self._features_rest.shape[1] * self._features_rest.shape[2]
                + 1 + self._scaling.shape[1] + self._rotation.shape[1]
            )
            margin = int(os.environ.get("TAGGS_SAVE_PLY_MARGIN_BYTES", str(512 * 1024 * 1024)))
            du = shutil.disk_usage(os.path.dirname(os.path.abspath(path)))
            if du.free < est_bytes + margin:
                raise OSError(
                    f"[save_ply] Insufficient disk space for {path!r}: need ~{(est_bytes + margin) // (1024 * 1024)} MiB free, "
                    f"have {du.free // (1024 * 1024)} MiB (set TAGGS_SAVE_PLY_MARGIN_BYTES to adjust margin)."
                )
        except OSError:
            raise
        except Exception:
            pass

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        n = mask.shape[0]
        for group in self.optimizer.param_groups:
            p = group["params"][0]
            if p.shape[0] != n:
                raise RuntimeError(
                    f"Gaussian state is inconsistent: param group '{group['name']}' has {p.shape[0]} rows "
                    f"but prune/densify mask length is {n}. This usually happens after a partial CUDA OOM "
                    "mid-densification (optimizer updated for some attributes but not others). "
                    "Restore from the last good checkpoint, or retry with the fixed atomic "
                    "`cat_tensors_to_optimizer`."
                )
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _assert_optimizer_row_consistency(self, where="unknown"):
        """
        Lightweight consistency check for Gaussian parameter row counts.
        Catches partial updates early by validating optimizer groups, Adam states,
        and side buffers all share the same number of points.
        """
        n = self.get_xyz.shape[0]
        required_groups = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")
        seen = set()
        for group in self.optimizer.param_groups:
            name = group.get("name", "<unnamed>")
            p = group["params"][0]
            if p.shape[0] != n:
                raise RuntimeError(
                    f"[GaussianConsistency:{where}] param group '{name}' rows={p.shape[0]} "
                    f"but xyz rows={n}."
                )
            state = self.optimizer.state.get(p, None)
            if state is not None:
                exp_avg = state.get("exp_avg", None)
                exp_avg_sq = state.get("exp_avg_sq", None)
                if exp_avg is not None and exp_avg.shape[0] != n:
                    raise RuntimeError(
                        f"[GaussianConsistency:{where}] state exp_avg rows={exp_avg.shape[0]} "
                        f"for group '{name}', expected {n}."
                    )
                if exp_avg_sq is not None and exp_avg_sq.shape[0] != n:
                    raise RuntimeError(
                        f"[GaussianConsistency:{where}] state exp_avg_sq rows={exp_avg_sq.shape[0]} "
                        f"for group '{name}', expected {n}."
                    )
            seen.add(name)

        for group_name in required_groups:
            if group_name not in seen:
                raise RuntimeError(
                    f"[GaussianConsistency:{where}] missing optimizer group '{group_name}'."
                )

        if self.xyz_gradient_accum.shape[0] != n:
            raise RuntimeError(
                f"[GaussianConsistency:{where}] xyz_gradient_accum rows={self.xyz_gradient_accum.shape[0]} expected {n}."
            )
        if self.denom.shape[0] != n:
            raise RuntimeError(
                f"[GaussianConsistency:{where}] denom rows={self.denom.shape[0]} expected {n}."
            )
        if self.max_radii2D.shape[0] != n:
            raise RuntimeError(
                f"[GaussianConsistency:{where}] max_radii2D rows={self.max_radii2D.shape[0]} expected {n}."
            )
        tr = getattr(self, "tmp_radii", None)
        if tr is not None and tr.shape[0] != n:
            raise RuntimeError(
                f"[GaussianConsistency:{where}] tmp_radii rows={tr.shape[0]} expected {n}."
            )

    def prune_points(self, mask):
        valid_points_mask = ~mask
        # Keep at least one Gaussian alive to avoid downstream renderer/backward
        # shape issues when tensors become [0, ...].
        if valid_points_mask.numel() > 0 and not torch.any(valid_points_mask):
            keep_idx = torch.argmax(self.get_opacity.squeeze(-1))
            valid_points_mask[keep_idx] = True
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        # tmp_radii only exists during densify_and_prune; sparse/budget pruning runs after and clears it.
        tr = getattr(self, "tmp_radii", None)
        if tr is not None:
            self.tmp_radii = tr[valid_points_mask]
        self._assert_optimizer_row_consistency(where="prune_points")

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        Append new Gaussians to all optimizer param groups. All torch.cat allocations
        complete before any optimizer mutation, so CUDA OOM cannot leave groups with
        mismatched row counts (which caused IndexError in _prune_optimizer).
        """
        staged = []
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            old_param = group["params"][0]
            stored_state = self.optimizer.state.get(old_param, None)
            new_values = torch.cat((old_param, extension_tensor), dim=0)
            if stored_state is not None:
                new_state = {
                    **{
                        k: v
                        for k, v in stored_state.items()
                        if k not in ("exp_avg", "exp_avg_sq")
                    },
                    "exp_avg": torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    ),
                    "exp_avg_sq": torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    ),
                }
            else:
                new_state = None
            staged.append((group, old_param, new_values, new_state))

        optimizable_tensors = {}
        for group, old_param, new_values, new_state in staged:
            new_param = nn.Parameter(new_values.requires_grad_(True))
            self.optimizer.state.pop(old_param, None)
            group["params"][0] = new_param
            if new_state is not None:
                self.optimizer.state[new_param] = new_state
            optimizable_tensors[group["name"]] = new_param

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        # Pre-allocate auxiliary buffers before optimizer mutation. If CUDA OOM
        # happens here, we fail fast without partially updating model/optimizer states.
        target_n = self.get_xyz.shape[0] + new_xyz.shape[0]
        next_tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        next_xyz_gradient_accum = torch.zeros((target_n, 1), device="cuda")
        next_denom = torch.zeros((target_n, 1), device="cuda")
        next_max_radii2D = torch.zeros((target_n), device="cuda")

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = next_tmp_radii
        self.xyz_gradient_accum = next_xyz_gradient_accum
        self.denom = next_denom
        self.max_radii2D = next_max_radii2D
        self._assert_optimizer_row_consistency(where="densification_postfix")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if self.densify_max_new_points_per_iter > 0:
            max_selected = self.densify_max_new_points_per_iter // max(1, N)
            selected_count = int(selected_pts_mask.sum().item())
            if selected_count > max_selected:
                idx = torch.nonzero(selected_pts_mask, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    keep_idx = idx[torch.randperm(idx.numel(), device=idx.device)[:max_selected]]
                    selected_pts_mask = torch.zeros_like(selected_pts_mask)
                    selected_pts_mask[keep_idx] = True

        if not torch.any(selected_pts_mask):
            return

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )
        if self.densify_max_new_points_per_iter > 0:
            selected_count = int(selected_pts_mask.sum().item())
            if selected_count > self.densify_max_new_points_per_iter:
                idx = torch.nonzero(selected_pts_mask, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    keep_idx = idx[torch.randperm(idx.numel(), device=idx.device)[: self.densify_max_new_points_per_iter]]
                    selected_pts_mask = torch.zeros_like(selected_pts_mask)
                    selected_pts_mask[keep_idx] = True

        if not torch.any(selected_pts_mask):
            return

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii
        )

    def _densify_cuda_oom(self, err):
        msg = str(err).lower()
        return "out of memory" in msg or type(err).__name__ in ("OutOfMemoryError", "OutOfMemory")

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii

        min_free_mb = int(os.environ.get("TAGGS_DENSIFY_MIN_FREE_MB", "0"))
        if min_free_mb > 0 and torch.cuda.is_available():
            free_b, _ = torch.cuda.mem_get_info()
            if free_b < min_free_mb * 1024 * 1024:
                print(
                    f"[Densify] GPU free memory {free_b // (1024 * 1024)} MiB < "
                    f"{min_free_mb} MiB (TAGGS_DENSIFY_MIN_FREE_MB); skipping clone/split this round (prune still runs)."
                )
            else:
                torch.cuda.empty_cache()
                try:
                    self.densify_and_clone(grads, max_grad, extent)
                except RuntimeError as e:
                    if self._densify_cuda_oom(e):
                        print("[Densify] CUDA OOM during densify_and_clone; skipping clone and split this round.")
                        torch.cuda.empty_cache()
                    else:
                        raise
                else:
                    torch.cuda.empty_cache()
                    try:
                        self.densify_and_split(grads, max_grad, extent)
                    except RuntimeError as e:
                        if self._densify_cuda_oom(e):
                            print("[Densify] CUDA OOM during densify_and_split; continuing (clone may have succeeded).")
                            torch.cuda.empty_cache()
                        else:
                            raise
        else:
            torch.cuda.empty_cache()
            try:
                self.densify_and_clone(grads, max_grad, extent)
                self.densify_and_split(grads, max_grad, extent)
            except RuntimeError as e:
                if self._densify_cuda_oom(e):
                    print("[Densify] CUDA OOM during densify; skipping this densify step and continuing.")
                    torch.cuda.empty_cache()
                else:
                    raise

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
