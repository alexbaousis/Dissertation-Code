import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset.nyuv2ssl import NYUv2
from model.models import get_MTL, get_MTI

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def prepare_rgb_for_display(rgb):
    rgb = rgb.astype(np.float32)

    if rgb.min() >= 0.0 and rgb.max() <= 1.0:
        return rgb

    if rgb.max() > 1.5:
        rgb = rgb / 255.0

    return np.clip(rgb, 0.0, 1.0)


def normal_to_rgb(normal):
    normal = normal.astype(np.float32)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
    normal = normal / norm
    return (normal + 1.0) / 2.0


def depth_to_vis(depth):
    depth = depth.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)

    if valid.sum() == 0:
        return np.zeros_like(depth)

    dmin = np.percentile(depth[valid], 2)
    dmax = np.percentile(depth[valid], 98)

    depth = np.clip(depth, dmin, dmax)
    return (depth - dmin) / (dmax - dmin + 1e-8)


def semantic_to_vis(mask, invalid_value=-1):
    mask = mask.astype(np.float32)
    mask[mask == invalid_value] = np.nan
    return mask


def compute_semantic_miou(pred, gt, num_classes=13):
    valid = (gt != -1)
    pred = pred[valid]
    gt = gt[valid]

    ious = []
    for cls in range(num_classes):
        p = (pred == cls)
        g = (gt == cls)

        union = (p | g).sum()
        if union == 0:
            continue

        inter = (p & g).sum()
        ious.append(inter / union)

    return 100.0 * np.mean(ious) if len(ious) > 0 else 0.0


def compute_depth_abs_error(pred, gt):
    valid = np.isfinite(gt) & (gt > 0)
    if valid.sum() == 0:
        return 0.0
    return np.mean(np.abs(pred[valid] - gt[valid]))


def compute_normal_mean_error(pred, gt):
    valid = np.linalg.norm(gt, axis=-1) > 1e-6
    if valid.sum() == 0:
        return 0.0

    pred = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
    gt = gt / (np.linalg.norm(gt, axis=-1, keepdims=True) + 1e-8)

    dot = np.sum(pred * gt, axis=-1)
    dot = np.clip(dot, -1, 1)

    err = np.degrees(np.arccos(dot))
    return np.mean(err[valid])


def build_model(model_type, tasks, device):
    if model_type == "mtl":
        model = get_MTL(tasks)
    elif model_type == "mti":
        model = get_MTI(tasks)
    else:
        raise ValueError("model_type must be mtl or mti")

    return model.to(device)


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    model.eval()
    return ckpt


def make_figure(
    save_path,
    rgb,
    sem_gt,
    depth_gt,
    normal_gt,
    sem_pred,
    depth_pred,
    normal_pred,
    model_label,
    show_gt=True,
    num_classes=13,
):

    rgb_vis = prepare_rgb_for_display(rgb)
    gt_normal_vis = normal_to_rgb(normal_gt)
    pred_normal_vis = normal_to_rgb(normal_pred)
    depth_gt_vis = depth_to_vis(depth_gt)
    depth_pred_vis = depth_to_vis(depth_pred)

    sem_gt_vis = semantic_to_vis(sem_gt)
    sem_pred_vis = semantic_to_vis(sem_pred)

    sem_miou = compute_semantic_miou(sem_pred, sem_gt, num_classes)
    depth_err = compute_depth_abs_error(depth_pred, depth_gt)
    normal_err = compute_normal_mean_error(normal_pred, normal_gt)

    if show_gt:
        fig, ax = plt.subplots(2, 4, figsize=(18, 9))

        ax[0,0].imshow(rgb_vis)
        ax[0,0].set_title("GT Image")

        ax[0,1].imshow(gt_normal_vis)
        ax[0,1].set_title("GT Normal")

        ax[0,2].imshow(sem_gt_vis, cmap="tab20", vmin=0, vmax=num_classes-1)
        ax[0,2].set_title("GT Semantic")

        ax[0,3].imshow(depth_gt_vis, cmap="viridis")
        ax[0,3].set_title("GT Depth")

        ax[1,0].imshow(rgb_vis)
        ax[1,0].set_title(f"{model_label} Input")

        ax[1,1].imshow(pred_normal_vis)
        ax[1,1].set_title("Pred Normal")

        ax[1,2].imshow(sem_pred_vis, cmap="tab20", vmin=0, vmax=num_classes-1)
        ax[1,2].set_title("Pred Semantic")

        ax[1,3].imshow(depth_pred_vis, cmap="viridis")
        ax[1,3].set_title("Pred Depth")

    else:
        fig, ax = plt.subplots(1, 4, figsize=(18, 4))

        ax[0].imshow(rgb_vis)
        ax[0].set_title("Input")

        ax[1].imshow(pred_normal_vis)
        ax[1].set_title("Normal")

        ax[2].imshow(sem_pred_vis, cmap="tab20", vmin=0, vmax=num_classes-1)
        ax[2].set_title("Semantic")

        ax[3].imshow(depth_pred_vis, cmap="viridis")
        ax[3].set_title("Depth")

    for a in np.array(ax).ravel():
        a.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()

    return sem_miou, depth_err, normal_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_type", default="mtl", choices=["mtl","mti"])
    parser.add_argument("--index", default=0, type=int)
    parser.add_argument("--outdir", default="./visualisations")
    parser.add_argument("--show_gt", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = ["semantic","depth","normal"]

    model = build_model(args.model_type, tasks, device)
    load_checkpoint(model, args.checkpoint, device)

    dataset = NYUv2(root=args.dataroot, train=False)
    sample = dataset[args.index]

    if len(sample) == 4:
        image, semantic, depth, normal = sample
    else:
        image, semantic, depth, normal, *_ = sample

    x = image.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_dict, _ = model(x)

    pred_sem = F.interpolate(pred_dict["semantic"], size=semantic.shape, mode="bilinear", align_corners=False)
    pred_depth = F.interpolate(pred_dict["depth"], size=depth.shape[-2:], mode="bilinear", align_corners=False)
    pred_normal = F.interpolate(pred_dict["normal"], size=normal.shape[-2:], mode="bilinear", align_corners=False)

    rgb = np.moveaxis(image.cpu().numpy(), 0, -1)

    sem_gt = semantic.numpy()
    depth_gt = depth.squeeze(0).numpy()
    normal_gt = np.moveaxis(normal.numpy(), 0, -1)

    sem_pred = pred_sem.argmax(1).squeeze().cpu().numpy()
    depth_pred = pred_depth.squeeze().cpu().numpy()
    normal_pred = np.moveaxis(pred_normal.squeeze().cpu().numpy(), 0, -1)

    os.makedirs(args.outdir, exist_ok=True)
    save_path = os.path.join(args.outdir, f"sample_{args.index}.png")

    sem_miou, depth_err, normal_err = make_figure(
        save_path,
        rgb,
        sem_gt,
        depth_gt,
        normal_gt,
        sem_pred,
        depth_pred,
        normal_pred,
        args.model_type.upper(),
        show_gt=args.show_gt
    )

    print(f"\nSaved {save_path}")
    print(f"Semantic mIoU: {sem_miou:.2f}")
    print(f"Depth abs error: {depth_err:.4f}")
    print(f"Normal mean error: {normal_err:.2f}")


if __name__ == "__main__":
    main()
