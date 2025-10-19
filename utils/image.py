def normalize_normals(n):
    mag = n.pow(2).sum(1, keepdim=True).sqrt().clamp_min(1e-6)
    return n / mag

def make_fg(mask3):
    # mask3: (B,3,H,W), foreground if any channel >= 0.25
    return (mask3 >= 0.25).any(dim=1, keepdim=True)

def safe_divide(num, den, eps=1e-3):
    return num / (den.abs().clamp_min(eps))

def compute_shading_gt(image, reflectance, mask3):
    # image: (B,3,H,W), reflectance: (B,3,H,W)
    # grayscale shading target from I/R; average across RGB
    fg = make_fg(mask3)
    S_rgb = safe_divide(image, reflectance.clamp_min(1e-3))
    S_gray = S_rgb.mean(dim=1, keepdim=True)
    # optional clamp to [0,1]
    return S_gray.clamp(0, 1), fg

def masked_l1(pred, target, mask_bool):
    # pred/target: (B,1,H,W), mask_bool: (B,1,H,W) bool
    m = mask_bool.expand_as(pred)
    num = m.sum().clamp_min(1)
    return (pred[m] - target[m]).abs().sum() / num

def mask_image(image, mask):
    return image * mask
