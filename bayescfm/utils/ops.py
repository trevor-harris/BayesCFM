def snap_to_stride(h, w, stride):
    H = max(stride, (h // stride) * stride)
    W = max(stride, (w // stride) * stride)
    return H, W
