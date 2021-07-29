import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import utils
import algorithms.modules as m
import algorithms.modules_3d as m3d

from algorithms.rot_utils import euler2mat


class SACv2_3D(object):
    def __init__(self, obs_shape, action_shape, args):
        self.discount = args.discount
        self.update_freq = args.update_freq
        self.tau = args.tau

        self.train_rl = args.train_rl
        self.train_3d = args.train_3d
        self.prop_to_3d = args.prop_to_3d
        self.log_3d_imgs = args.log_3d_imgs

        assert not args.from_state and not args.use_vit, 'not supported yet'

        shared = m.SharedCNN(obs_shape, args.num_shared_layers, args.num_filters)
        head = m.HeadCNN(shared.out_shape, args.num_head_layers, args.num_filters)
        self.encoder_rl = m.Encoder(
            shared,
            head,
            m.Identity(out_dim=head.out_shape[0])
        ).cuda()

        """
        RL Networks
        """
        self.actor = m.EfficientActor(self.encoder_rl.out_dim, args.projection_dim, action_shape, args.hidden_dim,
                                      args.actor_log_std_min, args.actor_log_std_max).cuda()
        self.critic = m.EfficientCritic(self.encoder_rl.out_dim, args.projection_dim, action_shape,
                                        args.hidden_dim).cuda()
        self.critic_target = m.EfficientCritic(self.encoder_rl.out_dim, args.projection_dim, action_shape,
                                               args.hidden_dim).cuda()
        self.critic_target.load_state_dict(self.critic.state_dict())

        """
        3D Networks
        """
        self.encoder_3d = m3d.Encoder_3d(args).cuda()
        self.decoder_3d = m3d.Decoder_3d(args).cuda()
        self.rotate_3d = m3d.Rotate_3d(args).cuda()
        self.pose_3d = m3d.Posenet_3d().cuda()

        self.log_alpha = torch.tensor(np.log(args.init_temperature)).cuda()
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)

        """
        RL Optimizers
        """
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(
            itertools.chain(self.encoder_3d.parameters(), self.critic.parameters()),
            lr=args.lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999))

        """
        3D Optimizers
        """
        enc3d_params = list(self.encoder_3d.parameters())
        recond3d_params = list(self.rotate_3d.parameters()) + \
                          list(self.decoder_3d.parameters())

        pose3d_params = list(self.pose_3d.parameters())

        self.enc3d_optimizer = torch.optim.Adam(enc3d_params, lr=args.lr_3d)
        self.recon3d_optimizer = torch.optim.Adam(recond3d_params, lr=args.lr_3dc)
        self.pose3d_optimizer = torch.optim.Adam(pose3d_params, lr=args.lr_3dp)

        self.aug = m.RandomShiftsAug(pad=4)
        self.train()
        print("3D Encoder:", utils.count_parameters(self.encoder_3d))
        print('Encoder:', utils.count_parameters(self.encoder_rl))
        print('Actor:', utils.count_parameters(self.actor))
        print('Critic:', utils.count_parameters(self.critic))

    def train(self, training=True):
        self.training = training
        for p in [self.encoder_rl, self.actor, self.critic, self.critic_target]:
            p.train(training)
        for p in [self.encoder_3d, self.decoder_3d, self.rotate_3d, self.pose_3d]:
            p.train(training)

    def eval(self):
        self.train(False)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _obs_to_input(self, obs):
        if isinstance(obs, utils.LazyFrames):
            _obs = np.array(obs)
        else:
            _obs = obs
        _obs = torch.FloatTensor(_obs).cuda()
        _obs = _obs.unsqueeze(0)
        return _obs

    def select_action(self, obs):
        obs = obs[0] # Take first view
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            _obs = self.encoder_3d(_obs)
            mu, _, _, _ = self.actor(self.encoder_rl(_obs), compute_pi=False, compute_log_pi=False)
        return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        obs = obs[0] # Take first view
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            _obs = self.encoder_3d(_obs)
            mu, pi, _, _ = self.actor(self.encoder_rl(_obs), compute_log_pi=False)
        return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, L=None, writer=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (self.discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        if L is not None:
            L.log('train_critic/loss', critic_loss, step)\

        if writer is not None:
            writer.add_scalar("Critic Loss", critic_loss, step)

        self.critic_optimizer.zero_grad(set_to_none=True)

        if self.prop_to_3d:
            self.enc3d_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward(retain_graph=True)
            self.enc3d_optimizer.step()
            # Destroy sub-graphs not required
            Q1=None; Q2=None; target_Q=None; target_Q1=None; target_Q2=None;
        else:
            print("ERROR")
            critic_loss.backward()

        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, L=None, writer=None, step=None, update_alpha=True):
        _, pi, log_pi, log_std = self.actor(obs)
        Q1, Q2 = self.critic(obs, pi)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha.detach() * log_pi - Q).mean()
        if L is not None:
            L.log('train_actor/loss', actor_loss, step)
        if writer is not None:
            writer.add_scalar("Actor_Loss", actor_loss, step)

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            if L is not None:
                L.log('train_alpha/loss', alpha_loss, step)
                L.log('train_alpha/value', self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update_3d_recon(self, imgs, latent_3d, L=None, writer=None, step=None):
        b, t, c, h, w = imgs.size()
        latent_3d = self.encoder_3d(imgs[:, 0])
        _, C, H, W, D = latent_3d.size()

        # Duplicate the representation for each view
        object_code_t = latent_3d.unsqueeze(1).repeat(1, t, 1, 1, 1, 1).view(b * t, C, H, W, D)

        # Get Poses
        imgs_ref = imgs[:, 0:1].repeat(1, t - 1, 1, 1, 1)
        imgs_pair = torch.cat([imgs_ref, imgs[:, 1:]], dim=2)  # b x t-1 x 6 x h x w
        pair_tensor = imgs_pair.view(b * (t - 1), c * 2, h, w)
        traj = self.pose_3d(pair_tensor)  # b*t-1 x 6
        poses = torch.cat([torch.zeros(b, 1, 6).cuda(), traj.view(b, t - 1, 6)], dim=1).view(b * t, 6)

        theta = euler2mat(poses, scaling=False, translation=True)
        rot_codes = self.rotate_3d(object_code_t, theta)

        # Decode the representation to get back image.
        output = self.decoder_3d(rot_codes)
        output = F.interpolate(output, (h, w), mode='bilinear')  # T*B x 3 x H x W
        img_tensor = imgs.view(b * t, c, h, w)

        # L2 Loss
        loss_3d = F.mse_loss(output, img_tensor)

        if L is not None:
            L.log("train_3d/loss", loss_3d, step)
        if writer is not None:
            writer.add_scalar("Loss 3D Recon", loss_3d, step)

            if step % self.log_3d_imgs == 0:
                writer.add_text(f"Poses (Training)", str(poses), step)
                writer.add_images(f'input images (Training)', imgs[0], step)
                writer.add_images(f'reconstruction images (Training)',
                                  torch.clamp(output.view(b, t, c, h, w)[0], 0, 1), step)

                writer.add_video(f'input videos (Training)', imgs, step)
                writer.add_video(f'reconstruction videos (Training)',
                                 torch.clamp(output, 0, 1).view(b, t, c, h, w),step)

        self.enc3d_optimizer.zero_grad(set_to_none=True)
        self.recon3d_optimizer.zero_grad(set_to_none=True)
        self.pose3d_optimizer.zero_grad(set_to_none=True)

        loss_3d.backward()

        self.enc3d_optimizer.step()
        self.recon3d_optimizer.step()
        self.pose3d_optimizer.step()

    def gen_interpolate(self, imgs, writer=None, step=None):
        with torch.no_grad():
            b, t, c, h, w = imgs.size()

            latent_3d = self.encoder_3d(imgs[:, 0])
            _, C, H, W, D = latent_3d.size()

            a = torch.tensor(np.arange(0., 1.1, 0.1)).to(latent_3d.device).unsqueeze(0).repeat(b, 1).view(b, -1)
            object_code_t = latent_3d.unsqueeze(1).repeat(1, a.size(1), 1, 1, 1, 1).view(b * (a.size(1)), C, H, W, D)

            imgs_ref = imgs[:, 0:1].repeat(1, t - 1, 1, 1, 1)
            imgs_pair = torch.cat([imgs_ref, imgs[:, 1:]], dim=2)  # b x t-1 x 6 x h x w
            pair_tensor = imgs_pair.view(b * (t - 1), c * 2, h, w)
            traj = self.pose_3d(pair_tensor)  # b*t-1 x 6
            poses = torch.cat([torch.zeros(b, 1, 6).cuda(), traj.view(b, t - 1, 6)], dim=1).view(b * t, 6)

            poses_for_interp = poses.clone().view(b, t, -1).unsqueeze(1).repeat(1, a.size(1), 1, 1)
            a_i = a.view(-1).unsqueeze(1).repeat(1, 6).to(torch.float32)
            poses_for_interp = poses_for_interp.view(-1, t, 6)
            interp_poses = (1 - a_i) * poses_for_interp[:, 0] + a_i * poses_for_interp[:, 1]

            new_poses = interp_poses
            theta = euler2mat(new_poses, scaling=False, translation=True)
            rot_codes = self.rotate_3d(object_code_t, theta)
            output = self.decoder_3d(rot_codes)

            output = F.interpolate(output, (h, w), mode='bilinear')  # T*B x 3 x H x W

            writer.add_text(f"First Last Poses (Testing)", str(poses), step)
            writer.add_text(f"Interpolated Poses (Testing)", str(interp_poses), step)
            writer.add_images(f'Input Images (Testing)', imgs[0], step)
            writer.add_images(f'Interpolated Images (Testing)',
                              torch.clamp(output.view(b , a.size(1), c, h, w)[0], 0, 1), step)
            writer.add_video(f'Interpolation Video (Testing)',
                             torch.clamp(output, 0, 1).view(b, a.size(1), c, h, w)[0].unsqueeze(0), step)

    def update(self, replay_buffer, L, writer, step):
        if step % self.update_freq != 0:
            return

        obs, action, reward, next_obs, done = replay_buffer.sample()

        # Augment
        b, v, c, h, w = obs.shape
        obs = self.aug(obs.view(-1, c, h, w))
        n, c, h, w = obs.shape
        obs = obs.view(b, v, c, h, w)

        imgs = copy.deepcopy(obs)

        if not self.prop_to_3d:
            print("ERROR")
            with torch.no_grad():
                obs_3d = self.encoder_3d(obs[:, 0])
        else:
            obs_3d = self.encoder_3d(obs[:, 0])
        obs = self.encoder_rl(obs_3d)

        with torch.no_grad():
            next_obs = self.aug(next_obs)
            next_obs_3d = self.encoder_3d(next_obs)
            next_obs = self.encoder_rl(next_obs_3d)

        if self.train_rl:
            self.update_critic(obs, action, reward, next_obs, L, writer, step)
            self.update_actor_and_alpha(obs.detach(), L, writer, step)
            utils.soft_update_params(self.critic, self.critic_target, self.tau)

        if self.train_3d:
            self.update_3d_recon(imgs, obs_3d, L, writer, step)
            #self.update_3d_pose()


