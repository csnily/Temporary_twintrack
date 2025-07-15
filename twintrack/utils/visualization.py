import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2

def gradcam_visualization(model, input_tensor, target_layer, class_idx=None):
    """
    Compute Grad-CAM heatmap for a given model and input.
    Args:
        model: nn.Module
        input_tensor: (1, C, H, W)
        target_layer: nn.Module (e.g., model.backbone.layer4)
        class_idx: int or None
    Returns:
        heatmap: (H, W) numpy array
    """
    model.eval()
    activations = []
    gradients = []
    def forward_hook(module, inp, out):
        activations.append(out.detach())
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)
    output = model(input_tensor)
    if class_idx is None:
        class_idx = output.argmax().item()
    loss = output[0, class_idx]
    loss.backward()
    act = activations[0][0]  # (C, H, W)
    grad = gradients[0][0]   # (C, H, W)
    weights = grad.mean(dim=(1, 2))
    cam = (weights[:, None, None] * act).sum(0)
    cam = F.relu(cam)
    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    handle_f.remove()
    handle_b.remove()
    return cam

def plot_tracking_results(frames, tracks, save_path=None):
    """
    Visualize tracking results on frames.
    Args:
        frames: list of np.ndarray (H, W, 3)
        tracks: list of dicts {id: (x1, y1, x2, y2)} per frame
        save_path: str or None
    """
    for i, (img, objs) in enumerate(zip(frames, tracks)):
        img_vis = img.copy()
        for tid, box in objs.items():
            x1, y1, x2, y2 = map(int, box)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_vis, str(tid), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        plt.imshow(img_vis[..., ::-1])
        plt.title(f'Frame {i}')
        if save_path:
            plt.savefig(f'{save_path}/frame_{i:04d}.png')
        plt.show()

def plot_failure_cases(frames, fail_events, save_path=None):
    """
    Visualize failure cases (e.g., ID switch, occlusion, etc.)
    Args:
        frames: list of np.ndarray
        fail_events: list of dicts {type: str, frame: int, info: ...}
        save_path: str or None
    """
    for event in fail_events:
        frame_idx = event['frame']
        img = frames[frame_idx].copy()
        plt.imshow(img[..., ::-1])
        plt.title(f"Failure: {event['type']} @ Frame {frame_idx}")
        if save_path:
            plt.savefig(f'{save_path}/fail_{frame_idx:04d}.png')
        plt.show() 