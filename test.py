import torch
import tri_interpolate
import time


def trilinear_interpolation_py(feats, points):
    """
    Inputs:
        feats: (N, 8, F)
        points: (N, 3) local coordinates in [-1, 1]
    
    Outputs:
        feats_interp: (N, F)
    """
    u = (points[:, 0:1]+1)/2
    v = (points[:, 1:2]+1)/2
    w = (points[:, 2:3]+1)/2
    a = (1-v)*(1-w)
    b = (1-v)*w
    c = v*(1-w)
    d = 1-a-b-c

    feats_interp = (1-u)*(a*feats[:, 0] +
                          b*feats[:, 1] +
                          c*feats[:, 2] +
                          d*feats[:, 3]) + \
                       u*(a*feats[:, 4] +
                          b*feats[:, 5] +
                          c*feats[:, 6] +
                          d*feats[:, 7])
    
    return feats_interp


if __name__ == '__main__':
    N = 2**16
    F = 256

    feats = torch.rand(N, 8, F, device='cuda').requires_grad_()
    point = torch.rand(N, 3, device='cuda')*2-1

    # time test
    # t = time.time()
    # out_cuda = tri_interpolate.trilinear_interpolate(feats, point)
    # torch.cuda.synchronize()
    # print('CUDA time: ', time.time()-t)

    # t = time.time()
    # out_torch = trilinear_interpolation_py(feats, point)
    # print('Pytorch time: ', time.time()-t)

    # print(torch.allclose(out_cuda, out_torch))

    # grad test
    out_cuda = tri_interpolate.trilinear_interpolate(feats, point)
    out_torch = trilinear_interpolation_py(feats, point)
    print(out_cuda.requires_grad)
    print(out_torch.requires_grad)