nn.Linear: y = x \dot a.T
FT: 
- y = cublasGemmEx(a, x): cublas is column major
- y.T = a.T \dot x.T
- y = x \dot a

Megatron q_scaling = 1

Leave expert_scales to next iteration

TODO: Router for the first layer: need more works
