# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch.nn import functional as F
from training.gaussian import generate_gaussian_map
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing
# from torchvision.utils import save_image
import os
import cv2
from .face_parser import FaceParser, Erosion2d, Dilation2d
#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0, gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased'):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)
        self.step = 0
        
        self.face_parser = FaceParser(device)
        self.erosion = Erosion2d(1, 1, 7, soft_max=False).to(device)
        self.eye_erosion = Erosion2d(1, 1, 3, soft_max=False).to(device)
        self.dilation = Dilation2d(1, 1, 7, soft_max=False).to(device)
            
        list_kernels = []
        num_neighboors = 3
        init_neighboor = torch.zeros((num_neighboors, num_neighboors), dtype=torch.float32)
        init_neighboor[1,1] = 1
        
        for i in range(num_neighboors):
            for j in range(num_neighboors):
                if (i == 1 and j== 1) : continue
                tmp = init_neighboor.clone()
                tmp[i,j] = -1.
                list_kernels.append(tmp)
        list_kernels = torch.stack(list_kernels, 0).unsqueeze(1)
        list_kernels.require_grad = False
        self.diff_kernel = list_kernels.to(device)
        
        # diff depth
        

    def run_G(self, z, c, swapping_prob, neural_rendering_resolution, update_emas=False):
        # import pdb; pdb.set_trace()
        # self.G.eval().requires_grad_(False)
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)
        
       
        ws = self.G.mapping(z, c_gen_conditioning, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        gen_output = self.G.synthesis(ws, c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        '''self.step += 1
        run_dir = 'test_img/save'
        images =gen_output['image']
        images= (images - images.min()) / (images.max() - images.min())
        images_raw = gen_output['image_raw']
        images_raw= (images_raw - images_raw.min()) / (images_raw.max() - images_raw.min())
        images_depth = gen_output['image_depth']
        images_depth= (images_depth - images_depth.min()) / (images_depth.max() - images_depth.min())
        save_image(images, os.path.join(run_dir, f'fakes{self.step}.png'))
        save_image(images_raw, os.path.join(run_dir, f'fakes{self.step}_raw.png') )
        save_image(images_depth, os.path.join(run_dir, f'fakes{self.step}_depth.png'))
        torch.save(z,os.path.join(run_dir, f'z{self.step}.pt') )
        torch.save(c_gen_conditioning ,os.path.join(run_dir, f'c{self.step}.pt') )
        # assert False'''
        return gen_output, ws

    def smooth_seg_loss(self, diff_depth, mask, topk=0.7):
        mask.require_grad = True

        mask[:, :, 0] = 0
        mask[:, :, 255] = 0
        mask[:, :, :, 255] = 0
        mask[:, :, :, 0] = 0
        neighbor_loss = mask * ((diff_depth**2).sum(1, keepdims=True)) ** 0.5

        self.step += 1
        # assert False
        neighbor_loss = neighbor_loss[mask > 0]
        valid_loss, _ = torch.topk(neighbor_loss, int(topk * neighbor_loss.size()[0]))
        neighbor_loss = valid_loss.mean()
        if torch.isnan(neighbor_loss):
            return 0
        return neighbor_loss

    def compute_angle_from_matrix(self, matrix3x3):
        M = matrix3x3
        theta_y = torch.asin(-M[:, 2, 0])
        theta_z = torch.atan2(M[:, 1, 0], M[:, 0, 0])
        theta_x = torch.atan2(M[:, 2, 1], M[:, 2, 2])
        return (theta_x, theta_y, theta_z)

    def compute_angle_from_c(self, gen_c):
        
        extrinsic = gen_c[:, :16].detach()
        extrinsic = extrinsic.reshape((-1, 4, 4))
        angle = self.compute_angle_from_matrix(extrinsic[:, :3, :3])
        
        return angle

    def load_cz(self, swapping_prob, device, gen_z, gen_c):
        
        out, _ = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=128)
        angles = self.compute_angle_from_c(gen_c)

        frontal_ids = angles[1].abs() < (np.pi/ 12)
        
        
        # Compute depth diff
        depth_map = out['image_depth']
        depth_map = F.interpolate(depth_map, size=(256, 256), mode='bilinear', align_corners=True)
        diff_depth = torch.nn.functional.conv2d(depth_map, self.diff_kernel, padding=(1,1))

        # FACE PARSING
        image = out["image"].detach()
        seg_mask = self.face_parser.parse(image) # B x 1 x 512 x 512
        neighbor_loss = 0

        # For skin
        skin_mask = (seg_mask == 1).float()
            
        skin_mask = F.interpolate(skin_mask, size=(256, 256), mode='nearest')
        with torch.no_grad():
            skin_mask = self.erosion(skin_mask)
        skin_loss = 1.0 * self.smooth_seg_loss(diff_depth, skin_mask, 0.7)

        eye_mask = ((seg_mask == 4) | (seg_mask == 5)).float()
        eye_mask = F.interpolate(eye_mask, size=(256, 256), mode='nearest')
        with torch.no_grad():
            eye_mask = self.eye_erosion(eye_mask)
        eye_mask[~frontal_ids] = 0.0
        eye_loss = 1.0 * self.smooth_seg_loss(diff_depth, eye_mask, 1.0)
        # print(skin_loss, eye_loss)
        neighbor_loss = skin_loss + eye_loss
        return neighbor_loss

        # For eye
        eye_loss = 0
        high_res_depth = F.interpolate(depth_map, size=(512, 512), mode='bilinear', align_corners=True)

        # TODO: Debug
        # high_res_image = F.interpolate(image, size=(512, 512), mode='bilinear', align_corners=True)
        # min_img = high_res_image.flatten(2).min(2)[0][:, :, None, None]
        # max_img = high_res_image.flatten(2).max(2)[0][:, :, None, None]
        # high_res_image = (high_res_image - min_img) / (max_img - min_img)
        # high_res_image = high_res_image.permute(0, 2, 3, 1)
        # eye_mask = (seg_mask == 4) | (seg_mask == 5)
        # for mask, image, depth in zip(eye_mask, high_res_image, high_res_depth):
        #     image = image.cpu().numpy()[:, :, ::-1] * 255
        #     image = image.astype('uint8')

        #     mask = mask[0, :, :, None].cpu().numpy()
        #     eye_mask_color = np.zeros_like(image)
        #     eye_mask_color[:, :, 0] = mask[:, :, 0] * 255
        #     eye_mask_color[:, :, 2] = mask[:, :, 0] * 255
            
        #     image = image * (1 - mask) + mask * (image * 0.7 + 0.3 * eye_mask_color)
        #     cv2.imwrite("image.png", image)
        #     depth = depth[0].cpu().detach().numpy()
        #     depth = 1.0 - (depth - depth.min()) / (depth.max() - depth.min())
        #     depth[depth < 0.5] = 0.5
        #     depth = (depth - depth.min()) / (depth.max() - depth.min())
        #     depth = (depth * 255).astype('uint8')
        #     depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        #     cv2.imwrite("depth.png", depth)
        #     import pdb; pdb.set_trace()

        for eye_id in [4, 5]:
            eye_mask = (seg_mask == eye_id).float()

            # Dilation to get surrounding mask
            with torch.no_grad():
                surround_eye_mask = self.dilation(eye_mask)

            valid_indices = (eye_mask.sum((1,2,3)) > 3) & frontal_ids
            
            if valid_indices.sum() == 0:
                continue    
            eye_depth = high_res_depth[valid_indices].float()
            eye_mask = eye_mask[valid_indices].float()

            # Find the eye center & radius
            coords = torch.nonzero(eye_mask[:, 0])
            

            centers = []
            radius = []
            min_depth = []
            max_depth = []
            keep_indices = []
            for batch_i in range(eye_depth.shape[0]):
                b_coords = coords[coords[:, 0] == batch_i][:, 1:].cpu().numpy()
                b_depth = eye_depth[batch_i][eye_mask[batch_i] > 0]
                max_depth.append(b_depth.max().item())
                min_depth.append(b_depth.min().item())

                y1, y2, x1, x2 = b_coords[:, 0].min(), b_coords[:, 0].max(), b_coords[:, 1].min(), b_coords[:, 1].max()
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                if eye_mask[batch_i, 0, center[1], center[0]] > 0:
                    keep_indices.append(True)
                else:
                    keep_indices.append(False)
                
                centers.append(center)
                radius.append(max(x2 - x1 + 1, y2 - y1 + 1))

            min_depth = torch.tensor(min_depth, device=depth_map.device)[:, None, None, None]
            max_depth = torch.tensor(max_depth, device=depth_map.device)[:, None, None, None]

            # Compute gaussian map
            gaussian_map = 1.0 - generate_gaussian_map(centers, radius, size=(512, 512))
            gaussian_map = torch.from_numpy(gaussian_map).to(depth_map.device)

            # Scale to depth max and min value
            gaussian_map = gaussian_map * (max_depth - min_depth) + min_depth
            # import pdb; pdb.set_trace()
            gaussian_map = gaussian_map * eye_mask

            # import pdb; pdb.set_trace()
            # for batch_i in range(eye_depth.shape[0]):
            #     vis_gt = gaussian_map[batch_i,0].cpu().numpy()
            #     vis_gt[vis_gt > 0] = 0.5 + (vis_gt[vis_gt > 0] - vis_gt[vis_gt > 0].min()) / (2 * (vis_gt[vis_gt > 0].max() - vis_gt[vis_gt > 0].min()))

            #     vis_pred = (eye_depth * eye_mask)[batch_i, 0].detach().cpu().numpy()
            #     vis_pred[vis_pred > 0] = 0.5 + (vis_pred[vis_pred > 0] - vis_pred[vis_pred > 0].min()) / (2 * (vis_pred[vis_pred > 0].max() - vis_pred[vis_pred > 0].min()))
                
            #     vis_depth = eye_depth[batch_i, 0].detach().cpu().numpy()
            #     vis_depth = (vis_depth - vis_depth.min()) / ((vis_depth.max() - vis_depth.min()))
            #     vis_depth = 1.0 - vis_depth

            #     cv2.imwrite(f"eye_{batch_i}.png", vis_gt * 255)
            #     cv2.imwrite(f"eye_out{batch_i}.png", vis_pred * 255)
            #     cv2.imwrite(f"depth_{batch_i}.png", vis_depth * 255)
                # import pdb; pdb.set_trace()

            # Contraint eyes to globe shaped
            one_eye_loss = torch.sqrt(F.mse_loss(gaussian_map, eye_depth, reduction='none')[keep_indices][eye_mask[keep_indices] > 0]).mean()

            # Constraint eyes to not higher than surrounding region
            surround_eye_mask = surround_eye_mask[valid_indices]
            surround_eye_mask[eye_mask > 0] = 0
            surround_eye_depth = surround_eye_mask * eye_depth
            # surround_eye_depth[surround_eye_mask == 0] = 999
            max_constraint_value = surround_eye_depth.flatten(1).max(1)[0].detach()
            
            max_constraint_value = max_constraint_value[:, None, None, None] * torch.ones_like(gaussian_map)

            error_ids = (gaussian_map < max_constraint_value) & (eye_mask > 0)
            # cv2.imwrite("error.png", error_ids[0,0].cpu().numpy() * 255)
            eye_upper_bound = 0
            if torch.any(error_ids):
                eye_upper_bound = torch.sqrt(F.mse_loss(gaussian_map[error_ids], max_constraint_value[error_ids], reduction='none')).mean()
            # print(eye_upper_bound)
            # import pdb; pdb.set_trace()
            # Constraint eyes to not higher than any skin region
            # skin_mask = (seg_mask == 1).float()[valid_indices]
            # with torch.no_grad():
            #     skin_mask = self.erosion(skin_mask)
            # skin_depth = skin_mask * eye_depth
            # skin_depth[skin_mask == 0] = 9999
            # max_constraint_value = skin_depth.flatten(1).min(1)[0].detach()
            
            # max_constraint_value = max_constraint_value[:, None, None, None] * torch.ones_like(gaussian_map)

            # error_ids = (gaussian_map < max_constraint_value) & (eye_mask > 0)
            # cv2.imwrite("error.png", error_ids[2,0].cpu().numpy() * 255)
            # eye_upper_bound2 = 0
            # if torch.any(error_ids):
            #     eye_upper_bound2 = torch.sqrt(F.mse_loss(gaussian_map[error_ids], max_constraint_value[error_ids], reduction='none')).mean()

            # print(eye_upper_bound, eye_upper_bound2)
            # import pdb; pdb.set_trace()
            
            if not torch.isnan(one_eye_loss):
                eye_loss = eye_loss + one_eye_loss + eye_upper_bound #+ eye_upper_bound2

            # print(eye_loss)
        neighbor_loss = skin_loss + 0.3 * eye_loss
        return neighbor_loss

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def sym_loss(self, phase, real_img, gen_z, gen_c ):
        G = phase
        cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth','Gsmooth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw}

        # Smothness loss
        # if phase == 'Greg': import pdb; pdb.set_trace()
        if phase in ['Gsmooth']:

        # if :
            loss = 0
            
            loss = self.load_cz(swapping_prob, real_img_raw.device, gen_z, gen_c)
            training_stats.report('Loss/smooth', loss)
            loss.mul(100).backward()
            

        # Gmain: Maximize logits for generated images.
        
        if phase in ['Gmain', 'Gboth']:
        # if True:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()
                # return gen_img

        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':
        # if True:
            
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            
            TVloss.mul(gain).backward()


        
        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-detach':
        # if True:   
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial.detach() - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-fixed':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}

                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
                    else: # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
