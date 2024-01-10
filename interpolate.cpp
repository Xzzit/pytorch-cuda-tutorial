#include <torch/extension.h>
#include "utils.h"


torch::Tensor trilinear_interpolate(
    torch::Tensor feats,
    torch::Tensor point)
{
    CHECK_INPUT(feats);
    CHECK_INPUT(point);
    
    return trilinear_fw_cu(feats, point);
}

torch::Tensor trilinear_interpolation_bw(
    const torch::Tensor dL_dfeat_interp,
    const torch::Tensor feats,
    const torch::Tensor points
){
    CHECK_INPUT(dL_dfeat_interp);
    CHECK_INPUT(feats);
    CHECK_INPUT(points);

    return trilinear_bw_cu(dL_dfeat_interp, feats, points);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("trilinear_interpolate", &trilinear_interpolate, "Trilinear interpolation");
    m.def("trilinear_interpolation_bw", &trilinear_interpolation_bw, "Trilinear interpolation backward");
}