import wandb
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from thre3d_atom.modules.volumetric_model import VolumetricModel
from torch import Tensor
from matplotlib import cm, colors
from thre3d_atom.utils.imaging_utils import CameraPose, CameraIntrinsics, to8b
from thre3d_atom.rendering.volumetric.render_interface import Rays

import maxflow

g_neighbor_offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                      [1, 0, 0], [0, 1, 0], [0, 0, 1]]


def visualize_and_log_attention_maps(attn_maps: tuple, global_step: int, log_freq: int = 50):
    if (global_step % log_freq == 0) or (global_step == 0):
        edit_attn_map = attn_maps[0]
        object_attn_map = attn_maps[1]
        cmp = cm.get_cmap('jet')

        # vis and log edit attn map:
        norm = colors.Normalize(vmin=0, vmax=torch.max(edit_attn_map).item())
        attn_frame = cmp(norm(edit_attn_map.cpu()))[:, :, :3]
        wandb.log({"Edit Attn Map": wandb.Image(attn_frame)}, step=global_step)

        # vis and log object attn map:
        norm = colors.Normalize(vmin=0, vmax=torch.max(object_attn_map).item())
        attn_frame = cmp(norm(object_attn_map.cpu()))[:, :, :3]
        wandb.log({"Object Attn Map": wandb.Image(attn_frame)}, step=global_step)

        # vis and log diff_map:
        diff_map = edit_attn_map - object_attn_map
        norm = colors.Normalize(vmin=torch.min(diff_map).item(), vmax=torch.max(diff_map).item())
        attn_frame = cmp(norm(diff_map.cpu()))[:, :, :3]
        wandb.log({"Diff Map": wandb.Image(attn_frame)}, step=global_step)


def calc_loss_on_attn_grid(attn_render: Tensor, attn_map: Tensor, token: str, 
                           global_step: int, log_freq: int=50, log_wandb=False):
    cmp = cm.get_cmap('jet')
    attn_render = attn_render.reshape(attn_map.shape)

    # get mask where attn grid render is not negative, i.e. where there is density
    non_zero_mask = attn_render > 0.0
    mask = torch.zeros_like(attn_map)
    mask[non_zero_mask] = 1.0

    # calc loss:
    diff = torch.abs(attn_render - attn_map)

    # calc masked diff
    diff_masked = diff * mask.float()

    # visualize mask
    if ((global_step % log_freq == 0) or (global_step == 0)) and log_wandb:
        norm = colors.Normalize(vmin=0, vmax=torch.max(mask).item())
        mask_frame = cmp(norm(mask.cpu()))[:, :, :3]
        wandb.log({f"Mask {token}": wandb.Image(mask_frame)}, step=global_step)

        ## get rid of large difference between background and foreground caused by -1
        attn_vis = attn_render
        norm = colors.Normalize(vmin=0, vmax=torch.max(attn_vis).item())
        attn_vis = cmp(norm(attn_vis.cpu().detach().numpy()))[:, :, :3]
        wandb.log({f"Pred Attn Map {token}": wandb.Image(attn_vis)}, step=global_step)

        # visualize diff mask
        norm = colors.Normalize(vmin=0, vmax=torch.max(diff_masked).item())
        diff_mask_frame = cmp(norm(diff_masked.cpu().detach().numpy()))[:, :, :3]
        wandb.log({f"Diff Masked {token}": wandb.Image(diff_mask_frame)}, step=global_step)

    attn_loss = diff_masked.sum() / mask.sum()
    return attn_loss


def log_and_vis_render_diff(edit_attn_render: Tensor, object_attn_render: Tensor, step: int):
    cmp = cm.get_cmap('jet')
    diff_render = edit_attn_render - object_attn_render
    norm = colors.Normalize(vmin=diff_render.min(), vmax=torch.max(diff_render).item())
    diff_frame = cmp(norm(diff_render.cpu().detach().numpy()))[:, :, :3]
    wandb.log({f"Render Diff": wandb.Image(diff_frame)}, step=step)


def plot_scatter(locations: Tensor,
                 features: Tensor,
                 edit_attn_map: Tensor,
                 object_attn_map: Tensor,
                 cluster_ids: Tensor,
                 step: int,
                 num_samples: int = 1000):
    # sample random indices
    perm = torch.randperm(features.shape[0])
    perm = perm[:num_samples]

    # get samples from tensors
    cluster_ids_sample = cluster_ids[perm].detach().cpu().numpy()
    feature_sample = features[perm].detach().cpu().numpy()
    edit_attn_sample = edit_attn_map[perm].squeeze().detach().cpu().numpy()
    object_attn_sample = object_attn_map[perm].squeeze().detach().cpu().numpy()
    locations_sample = locations[perm].squeeze().detach().cpu().numpy()

    # get colormap according to attention values
    diff_attn = (edit_attn_sample - object_attn_sample)
    neg_diff_mask = diff_attn < 0.0
    pos_diff_mask = diff_attn >= 0.0

    ### plot with original locations and colors":
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # neg:
    xs = locations_sample[neg_diff_mask, 0]
    ys = locations_sample[neg_diff_mask, 1]
    zs = locations_sample[neg_diff_mask, 2]
    ax.scatter(xs, ys, zs, marker='o', c=feature_sample[neg_diff_mask], label='higher object attn')

    # pos:
    xs = locations_sample[pos_diff_mask, 0]
    ys = locations_sample[pos_diff_mask, 1]
    zs = locations_sample[pos_diff_mask, 2]
    ax.scatter(xs, ys, zs, marker='^', c=feature_sample[pos_diff_mask], label='higher edit attn')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    wandb.log({"Location Scatter": wandb.Image(plt)}, step=step)
    plt.savefig("scatter3d_locations.png")

    ### plot with features as axis and attn as color":

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # neg:
    xs = feature_sample[..., 0]
    ys = feature_sample[..., 1]
    zs = feature_sample[..., 2]
    ax.scatter(xs, ys, zs, marker='o', c=diff_attn, cmap='jet', label='higher object attn')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    wandb.log({"Feature Scatter": wandb.Image(plt)}, step=step)
    plt.savefig("scatter3d_features.png")

    ### plot with original locations and colors":
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    zero_id_mask = cluster_ids_sample == 0
    one_id_mask = cluster_ids_sample == 1

    # neg:
    xs = locations_sample[zero_id_mask, 0]
    ys = locations_sample[zero_id_mask, 1]
    zs = locations_sample[zero_id_mask, 2]
    ax.scatter(xs, ys, zs, marker='o', c=feature_sample[zero_id_mask], label='higher object attn')

    # pos:
    xs = locations_sample[one_id_mask, 0]
    ys = locations_sample[one_id_mask, 1]
    zs = locations_sample[one_id_mask, 2]
    ax.scatter(xs, ys, zs, marker='^', c=feature_sample[one_id_mask], label='higher edit attn')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    wandb.log({"ID Scatter": wandb.Image(plt)}, step=step)
    plt.savefig("scatter3d_Ids.png")


def gen_id(t_in, x, y):
    id = t_in[0] + t_in[1] * x + t_in[2] * x * y
    return id.item()


def build_graph(features, densities, edit_attn, obj_attn, K=0.05, sigma=0.1, edit_mask_thresh=0.992,
                num_obj_voxels_thresh=5000, min_num_edit_voxels=300, top_k_edit_thresh=300, top_k_obj_thresh=200,
                downsample_grid=False, downsample_factor=4):
    g = maxflow.Graph[float]()
    m = torch.nn.MaxPool3d(3, stride=1, padding=1)

    # calculate indexes:
    if downsample_grid:
        max_pool = torch.nn.MaxPool3d(downsample_factor, stride=downsample_factor, padding=0)
        avg_pool = torch.nn.AvgPool3d(downsample_factor, stride=downsample_factor, padding=0)
        density_grid = max_pool(densities.permute(-1, 0, 1, 2)).permute(1, 2, 3, 0)
        feature_grid = avg_pool(features.permute(-1, 0, 1, 2)).permute(1, 2, 3, 0)
        non_zero_densities = density_grid > 0.0
        edit_attn_vals = max_pool(edit_attn.permute(-1, 0, 1, 2)).permute(1, 2, 3, 0)[non_zero_densities].unsqueeze(-1)
        obj_attn_vals = max_pool(obj_attn.permute(-1, 0, 1, 2)).permute(1, 2, 3, 0)[non_zero_densities].unsqueeze(-1)
    else:
        density_grid = densities
        feature_grid = features
        non_zero_densities = m(density_grid) > 0.0
        edit_attn_vals = edit_attn[non_zero_densities].unsqueeze(-1)
        obj_attn_vals = obj_attn[non_zero_densities].unsqueeze(-1)

    x, y, z, _ = density_grid.shape
    x_idxs = torch.arange(x)
    y_idxs = torch.arange(y)
    z_idxs = torch.arange(z)
    grid_x, grid_y, grid_z = torch.meshgrid((x_idxs, y_idxs, z_idxs), indexing='ij')
    idx_grid = torch.cat((grid_x.unsqueeze(dim=-1),
                          grid_y.unsqueeze(dim=-1),
                          grid_z.unsqueeze(dim=-1)), dim=-1)
    idx_values = idx_grid[non_zero_densities.cpu().squeeze()]

    # add nodes:
    num_nodes = non_zero_densities.sum()
    nodes = g.add_nodes(num_nodes)

    # node idx dict
    idx_dict = {}
    i = range(num_nodes)
    print(f"Generating IDs:")
    for i in tqdm(range(num_nodes)):
        idx_dict[gen_id(idx_values[i], x, y)] = i
    
    # calc attn diff:
    softmax_fn = torch.nn.Softmax(dim=-1)
    probs = softmax_fn(torch.cat((edit_attn_vals, obj_attn_vals), dim=-1))

    # initialize according to max:
    top_prob_edit = torch.max(probs[..., 0])
    best_voxels_edit_mask = probs[..., 0] >= edit_mask_thresh * top_prob_edit
    top_k_best_edit_idxs = best_voxels_edit_mask.nonzero().squeeze()

    obj_mask = probs[..., 1] > probs[..., 0]
    idxs = torch.arange(probs.size(0))
    obj_idxs = idxs[obj_mask.cpu()]
    perm = torch.randperm(obj_idxs.size(0))
    idx = perm[:num_obj_voxels_thresh]
    top_k_best_obj_idxs = obj_idxs[idx]

    if best_voxels_edit_mask.sum() < min_num_edit_voxels:
        print("Not enough edit voxels, using top k edit voxels")
        edit_topk = torch.topk(edit_attn_vals.squeeze(), top_k_edit_thresh)
        top_k_best_edit_idxs = edit_topk.indices

        obj_topk = torch.topk(obj_attn_vals.squeeze(), top_k_obj_thresh)
        top_k_best_obj_idxs = obj_topk.indices

    # set inter node edges
    print("Building Graph...")
    for i in tqdm(range(num_nodes)):
        if i in top_k_best_edit_idxs:
            g.add_tedge(nodes[i], np.inf, 0)
        elif i in top_k_best_obj_idxs:
            g.add_tedge(nodes[i], 0, np.inf)
        # else:
        #    g.add_tedge(nodes[i], probs[i][0], probs[i][1])
        nidx = idx_values[i]

        # for each neighbor
        for n_offset in g_neighbor_offsets:
            n_offset = torch.tensor(n_offset)
            # check for idxs outside grid
            if (((nidx + n_offset) >= x).sum() > 0.0) or \
                (((nidx + n_offset) >= y).sum() > 0.0) or (((nidx + n_offset) >= z).sum() > 0.0):
                continue
            # check for negative idxs:
            if ((nidx + n_offset) < 0).sum() > 0.0:
                continue
            # make sure neighbor has density:
            pot_n_offset = nidx + n_offset
            if density_grid[pot_n_offset[0], pot_n_offset[1], pot_n_offset[2], 0].item() <= 0.0:
                continue

            neighbor_node_idx = idx_dict[gen_id(nidx + n_offset, x, y)]

            # calculate L2 diff:
            node_feature = feature_grid[nidx.unsqueeze(-1).tolist()].squeeze()
            neighbor_feature = feature_grid[(nidx + n_offset).unsqueeze(-1).tolist()].squeeze()
            l2_probs = torch.sqrt(((probs[i] - probs[nidx]) ** 2).sum())
            l2_colors = torch.sqrt(((node_feature - neighbor_feature) ** 2).sum())

            # calculate affinity:
            w = (K * torch.exp(-(l2_colors * 1.0 + l2_probs * 0.0) / sigma))

            # set edge:
            g.add_edge(nodes[i], nodes[neighbor_node_idx], w, w)

    print(f"Calculating Min Cut...")
    flow = g.maxflow()
    print(f"Done!")
    print(f"Labeling segments:")
    segments = [g.get_segment(nodes[i]) for i in tqdm(range(num_nodes))]
    segments = torch.tensor(segments)
    segment_idxs = idx_values
    print(f"{(segments == 0).sum()} Voxels marked as Edit")
    print(f"{(segments == 1).sum()} Voxels marked as Object")
    return segments, segment_idxs

def set_and_visualize_refined_grid(vol_mod_edit: VolumetricModel, 
                                        vol_mod_object: VolumetricModel, 
                                        vol_mod_output: VolumetricModel,
                                        rays: Rays,
                                        img_height: int,
                                        img_width: int,
                                        ids: Tensor,
                                        idxs: Tensor,
                                        step: int = 0,
                                        log_wandb: bool = False):
    
    # first visualize greater than grid:
    m = torch.nn.MaxPool3d(3, stride=1, padding=1)

    # calculate indexes:
    nonzero_densities = m(vol_mod_edit.thre3d_repr._densities)
    nonzero_densities = nonzero_densities > 0.0
    gt_grid = torch.ones_like(vol_mod_edit.thre3d_repr.attn) * -20.0
    gt_mask = vol_mod_edit.thre3d_repr.attn > vol_mod_object.thre3d_repr.attn
    gt_grid[gt_mask] = 0.0
    vol_mod_output.thre3d_repr.attn = torch.nn.Parameter(gt_grid)
    attn_rendered_batch = vol_mod_output.render_rays_attn(rays)
    attn_rendered_batch = attn_rendered_batch.attn
    attn_render = attn_rendered_batch.reshape((img_height, img_width))

    cmp = cm.get_cmap('jet')

    # vis and log greater than attn map:
    norm = colors.Normalize(vmin=0, vmax=torch.max(attn_render).item())
    attn_frame = cmp(norm(attn_render.cpu()))[:, :, :3]
    if log_wandb:
        wandb.log({"GT Attn Map": wandb.Image(attn_frame)}, step=step)

    # then visualize id based grid (graphcut output):
    gt_grid = torch.ones_like(vol_mod_edit.thre3d_repr.attn) * -20.0
    gt_grid[nonzero_densities] = -10.0
    edit_ids = (idxs[ids == 0]).tolist()
    for idx in edit_ids:
        gt_grid[idx[0], idx[1], idx[2], 0] = 0.0  # maybe unsqueeze() ?
    vol_mod_output.thre3d_repr.attn = torch.nn.Parameter(gt_grid)
    attn_rendered_batch = vol_mod_output.render_rays_attn(rays)
    attn_rendered_batch = attn_rendered_batch.attn
    attn_render = attn_rendered_batch.reshape((img_height, img_width))

    # vis and log greater than attn map:
    norm = colors.Normalize(vmin=0, vmax=torch.max(attn_render).item())
    attn_frame = cmp(norm(attn_render.cpu()))[:, :, :3]
    if log_wandb:
        wandb.log({"GraphCut result Attn Map": wandb.Image(attn_frame)}, step=step)


def get_edit_region(vol_mod_edit: VolumetricModel,
                    vol_mod_object: VolumetricModel,
                    vol_mod_output: VolumetricModel,
                    downsample_grid: bool = False,
                    downsample_factor: int = 4,
                    K: int = 5.0,
                    sigma: float = 0.1,
                    #produce_scatter_plot: bool = False, 
                    edit_mask_thresh=0.992,
                    num_obj_voxels_thresh=5000, 
                    min_num_edit_voxels=300, 
                    top_k_edit_thresh=300, 
                    top_k_obj_thresh=200):
    # first make sure the densities and features of both grids are the same
    assert (
        torch.eq(vol_mod_edit.thre3d_repr._densities, vol_mod_object.thre3d_repr._densities).all().item()
    ), f"ERROR: Density values for edit and object grids don't match"

    assert (
        torch.eq(vol_mod_edit.thre3d_repr._features, vol_mod_object.thre3d_repr._features).all().item()
    ), f"ERROR: Feature values for edit and object grids don't match"

    # Begin with calculating edit and object indexes
    with torch.no_grad():
        densities = vol_mod_edit.thre3d_repr._densities.detach()
        edit_attn = vol_mod_edit.thre3d_repr.attn.detach()
        obj_attn = vol_mod_object.thre3d_repr.attn.detach()
        features = torch.sigmoid(vol_mod_edit.thre3d_repr._features.detach())

        ids, idxs = build_graph(features, densities, edit_attn, obj_attn, K=K, sigma=sigma,
                                edit_mask_thresh=edit_mask_thresh,
                                num_obj_voxels_thresh=num_obj_voxels_thresh, 
                                min_num_edit_voxels=min_num_edit_voxels,
                                top_k_edit_thresh=top_k_edit_thresh, 
                                top_k_obj_thresh=top_k_obj_thresh,
                                downsample_grid=downsample_grid)

        # Create keep grid
        keep_grid = torch.ones_like(edit_attn) * -10
        keep_grid[densities > 0.0] = -5
        edit_ids = (idxs[ids == 0]).tolist()

        # Mark edit voxels
        if downsample_grid:
            factor = downsample_factor
        else:
            factor = 1

        for idx in edit_ids:
            keep_grid[idx[0] * factor: idx[0] * factor + factor, 
                      idx[1] * factor: idx[1] * factor + factor,
                      idx[2] * factor: idx[2] * factor + factor] = 0.0 

        # Set output attn grid
        vol_mod_output.thre3d_repr.attn = torch.nn.Parameter(keep_grid)
        print(f"Finished calculating edit / object regions!")
