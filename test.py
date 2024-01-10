import torch
import tri_interpolate


if __name__ == '__main__':
    N = 65536
    F = 256

    feats = torch.rand(N, 8, F, device='cuda')
    point = torch.rand(1024, 3, device='cuda')*2-1

    out = tri_interpolate.trilinear_interpolate(feats, point)
    print(out.shape)