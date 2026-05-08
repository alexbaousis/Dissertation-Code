import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset.cityscapesssl import Cityscapes
from model.models import get_MTL, get_MTI

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def prepare_rgb_for_display(rgb):
    rgb = rgb.astype(np.float32)

    if rgb.min() >= 0.0 and rgb.max() <= 1.0:
        return rgb

    if rgb.max() > 1.5:
        rgb = rgb / 255.0

    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def depth_to_vis(depth):
    depth = depth.astype(np.float32).copy()
    valid = np.isfinite(depth)
    if valid.sum() == 0:
        return np.zeros_like(depth, dtype=np.float32)

    dmin = np.percentile(depth[valid], 2)
    dmax = np.percentile(depth[valid], 98)
    depth = np.clip(depth, dmin, dmax)
    return (depth - dmin) / (dmax - dmin + 1e-8)


def compute_semantic_miou(pred, gt, num_classes=7, ignore_index=-1):
    pred = pred.astype(np.int64)
    gt = gt.astype(np.int64)

    valid = (gt != ignore_index)
    pred = pred[valid]
    gt = gt[valid]

    ious = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        gt_mask = (gt == cls)

        union = np.logical_or(pred_mask, gt_mask).sum()
        if union == 0:
            continue

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        ious.append(intersection / union)

    if len(ious) == 0:
        return 0.0

    return 100.0 * float(np.mean(ious))


def compute_depth_abs_error(pred, gt):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)

    valid = np.isfinite(gt) & (gt > 0)
    if valid.sum() == 0:
        return 0.0

    return float(np.mean(np.abs(pred[valid] - gt[valid])))


def build_model(model_type, tasks, device):
    if model_type == "mtl":
        model = get_MTL(tasks, 7)
    elif model_type == "mti":
        model = get_MTI(tasks, 7)
    else:
        raise ValueError("model_type must be 'mtl' or 'mti'")

    return model.to(device)


def load_checkpoint_into_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    model.eval()
    return checkpoint


def save_npy_outputs(outdir, idx, rgb, sem_gt, depth_gt, sem_pred, depth_pred):
    os.makedirs(outdir, exist_ok=True)

    np.save(os.path.join(outdir, f"{idx}_image.npy"), rgb)
    np.save(os.path.join(outdir, f"{idx}_sem_gt.npy"), sem_gt)
    np.save(os.path.join(outdir, f"{idx}_depth_gt.npy"), depth_gt)
    np.save(os.path.join(outdir, f"{idx}_sem_pred.npy"), sem_pred)
    np.save(os.path.join(outdir, f"{idx}_depth_pred.npy"), depth_pred)


def make_figure(
    save_path,
    rgb,
    sem_gt,
    depth_gt,
    sem_pred,
    depth_pred,
    model_label,
    show_gt=True,
    num_classes=7,
    title_fontsize=18,
):
    rgb_vis = prepare_rgb_for_display(rgb)
    depth_gt_vis = depth_to_vis(depth_gt)
    depth_pred_vis = depth_to_vis(depth_pred)

    sem_miou = compute_semantic_miou(
        sem_pred, sem_gt, num_classes=num_classes, ignore_index=-1
    )
    depth_aerr = compute_depth_abs_error(depth_pred, depth_gt)

    if show_gt:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        axes[0, 0].imshow(rgb_vis)
        axes[0, 0].set_title("ground-truth image", fontsize=title_fontsize)

        axes[0, 1].imshow(sem_gt, cmap="tab20")
        axes[0, 1].set_title("ground-truth semantic", fontsize=title_fontsize)

        axes[0, 2].imshow(depth_gt_vis, cmap="viridis")
        axes[0, 2].set_title("ground-truth depth", fontsize=title_fontsize)

        axes[1, 0].imshow(rgb_vis)
        axes[1, 0].set_title(f"{model_label} input", fontsize=title_fontsize)

        axes[1, 1].imshow(sem_pred, cmap="tab20")
        axes[1, 1].set_title("semantic", fontsize=title_fontsize)

        axes[1, 2].imshow(depth_pred_vis, cmap="viridis")
        axes[1, 2].set_title("depth", fontsize=title_fontsize)

    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

        axes[0].imshow(rgb_vis)
        axes[0].set_title(f"{model_label} input", fontsize=title_fontsize)

        axes[1].imshow(sem_pred, cmap="tab20")
        axes[1].set_title("semantic", fontsize=title_fontsize)

        axes[2].imshow(depth_pred_vis, cmap="viridis")
        axes[2].set_title("depth", fontsize=title_fontsize)

        axes = np.array(axes).reshape(-1)

    for ax in np.array(axes).ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return sem_miou, depth_aerr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="./data/cityscapes", type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--outdir", default="./visualisations/cityscapes", type=str)
    parser.add_argument("--index", default=0, type=int)
    parser.add_argument("--model_type", default="mtl", choices=["mtl", "mti"], type=str)
    parser.add_argument("--label", default="", type=str)
    parser.add_argument("--num_classes", default=7, type=int)
    parser.add_argument("--show_gt", action="store_true")
    parser.add_argument("--title_fontsize", default=18, type=int)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = ["semantic", "depth"]

    model = build_model(args.model_type, tasks, device)
    checkpoint = load_checkpoint_into_model(model, args.checkpoint, device)

    dataset = Cityscapes(root=args.dataroot, train=False)

    sample = dataset[args.index]

    if len(sample) == 5:
        image, semantic, depth, sam_masks, sam_edges = sample
    elif len(sample) == 3:
        image, semantic, depth = sample
        sam_masks, sam_edges = None, None
    else:
        raise ValueError(
            f"Expected Cityscapes sample to return 3 or 5 items, got {len(sample)}"
        )

    x = image.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_dict, _ = model(x)

    if semantic.ndim == 2:
        gt_size = semantic.shape
    else:
        gt_size = semantic.shape[-2:]

    pred_sem_logits = F.interpolate(
        pred_dict["semantic"],
        size=gt_size,
        mode="bilinear",
        align_corners=False,
    )

    pred_depth_map = F.interpolate(
        pred_dict["depth"],
        size=depth.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    rgb = image.cpu().numpy()
    rgb = np.moveaxis(rgb, 0, -1)

    sem_gt = semantic.cpu().numpy().astype(np.int64)
    depth_gt = depth.squeeze(0).cpu().numpy().astype(np.float32)

    sem_pred = pred_sem_logits.argmax(1).squeeze(0).cpu().numpy().astype(np.int64)
    depth_pred = pred_depth_map.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    model_label = args.label if args.label else args.model_type.upper()

    sample_dir = os.path.join(args.outdir, f"sample_{args.index}_{model_label}")
    os.makedirs(sample_dir, exist_ok=True)

    save_npy_outputs(
        sample_dir,
        args.index,
        rgb,
        sem_gt,
        depth_gt,
        sem_pred,
        depth_pred,
    )

    fig_suffix = "with_gt" if args.show_gt else "pred_only"
    fig_path = os.path.join(sample_dir, f"{args.index}_{model_label}_{fig_suffix}.png")

    sem_miou, depth_aerr = make_figure(
        fig_path,
        rgb,
        sem_gt,
        depth_gt,
        sem_pred,
        depth_pred,
        model_label,
        show_gt=args.show_gt,
        num_classes=args.num_classes,
        title_fontsize=args.title_fontsize,
    )

    print(f"Loaded checkpoint: {args.checkpoint}")
    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
    print(f"Sample index: {args.index}")
    print("Per-image metrics are computed directly from this selected sample:")
    print(f"  semantic mIoU: overlap between predicted and ground-truth semantic classes = {sem_miou:.2f}")
    print(f"  depth aErr: mean absolute difference between predicted and ground-truth depth = {depth_aerr:.4f}")
    print(f"Saved figure to: {fig_path}")
    print(f"Saved .npy outputs to: {sample_dir}")


if __name__ == "__main__":
    main()