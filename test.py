import torch
import tri_interpolate


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')

    feats = torch.ones(2).to(device)
    point = torch.zeros(2).to(device)

    out = tri_interpolate.trilinear_interpolate(feats, point)
    print(out)