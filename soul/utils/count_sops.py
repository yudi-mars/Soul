import os
import torch
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODULE_SOP_DICT = {}

import torch
import torch.nn.functional as F

def img2col(X, kernel_size, stride=1, pad=0):
    kh = kw = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    sh = sw = stride if isinstance(stride, int) else stride[0]

    X_pad = F.pad(X, (pad, pad, pad, pad), mode='constant', value=0)
    
    return F.unfold(X_pad, (kh, kw), stride=(sh, sw))

def conv_forward_with_sparsity(X, W, b, stride=1, pad=0):
    # 使用 unfold 实现 img2col
    cols = img2col(X, W.shape[2:], stride, pad)
    
    # 获取维度信息
    N, C, H, W_in = X.shape
    F_out, _, KH, KW = W.shape
    
    # 计算输出尺寸
    OH = (H + 2*pad - KH) // stride + 1
    OW = (W_in + 2*pad - KW) // stride + 1
    
    # 重塑卷积核
    W_reshaped = W.view(F_out, -1)  # (F, C*KH*KW)
    
    # 优化后的稀疏度检测 ------------------------------------------
    # 计算输入矩阵每列的非零计数
    cols_nonzero = (cols != 0)
    cols_nonzero_count = cols_nonzero.sum(dim=0).sum(1) # 沿批次维度求和 (C*KH*KW)
    
    # 计算权重矩阵每行的非零计数
    W_nonzero = (W_reshaped != 0)
    W_nonzero_count = W_nonzero.sum(dim=0)  # 沿滤波器维度求和 (C*KH*KW)
    
    # 计算总有效乘法次数
    total_effective = torch.dot(cols_nonzero_count.float(), W_nonzero_count.float())
    
    # 计算总乘法次数
    total_multiplies = N * OH * OW * F_out * KH * KW * C
    
    # 计算有效比例
    effective_ratio = total_effective / total_multiplies if total_multiplies != 0 else torch.tensor(0.0)
    
    return effective_ratio.item()

def fc_forward_with_sparsity(X,W):
    # out = X @ W + b
    
    # sparsity calculation --------------------------------------------------
    # generate nonzero masks
    cols_nonzero = (X != 0)        # nonzero index for input
    W_nonzero = (W != 0)     # idx of nonzero element for kernel 
    # count number of multiply operations
    effective_matrix = cols_nonzero.astype(np.int64) @ W_nonzero.astype(np.int64).T
    total_effective = effective_matrix.sum()
    

    total_multiplies = batch_matrix_mul(X.shape,W.shape)
    # calculate effective ratio
    effective_ratio = total_effective / total_multiplies if total_multiplies != 0 else 0.0
    return effective_ratio

def batch_matrix_mul(Mat_input,Mat_weight):
    _unused_f = np.cumprod(Mat_input, axis=0)[-2]  # after Batch dim(including T)
    out_f = Mat_weight[1]
    in_f = Mat_weight[0]
    layer_cnt = 1.0
    layer_cnt *= _unused_f
    layer_cnt *= in_f
    layer_cnt *= out_f
    return layer_cnt

# this function is especially prepared for lasr and ac attr
def ops_hook_conv(module_name, is_sop=True):
    def hook(m, inputs, outputs):
        inputs = inputs[0]
        max_v = torch.max(inputs)
        is_mac = max_v.dtype == torch.int64 or not torch.floor(max_v) == max_v
        ran = max_v.detach().cpu().numpy().astype(int)
        lsar = 0
        stride = m.stride[0]
        padding = m.padding[0]
        kn, kn = m.kernel_size
        hn, wn = inputs.shape[-2:]
        in_channels = m.in_channels
        out_channels = m.out_channels

        B, C, H, W = inputs.shape
        for i in range(1, ran + 1):
            lsar += len(torch.where(inputs == float(i))[0]) / inputs.numel() * i
        if lsar == 0:
            lsar = inputs.count_nonzero() / inputs.numel()
        weight = m.weight
        # inputs = inputs.reshape(B,C,H,W)
        inputs = inputs.reshape(-1, C, H, W)
        inputs = inputs.detach()
        lsar = conv_forward_with_sparsity(inputs,weight,0,stride,padding)
        pass
        if module_name not in MODULE_SOP_DICT.keys():
            MODULE_SOP_DICT[module_name] = lsar * kn * kn * hn * wn *  in_channels * out_channels * B
        else:
            MODULE_SOP_DICT[module_name] += lsar * kn * kn * hn * wn *  in_channels * out_channels * B
    return hook


def ops_hook_fc(module_name,is_sop=True):
    def hook(m,input,output):
        inputs = input[0]
        max_v = torch.max(inputs)
        is_mac = max_v.dtype == torch.int64 or not torch.floor(max_v) == max_v
        ran = max_v.detach().cpu().numpy().astype(int)
        lsar = 0
        for i in range(1, ran + 1):
            lsar += len(torch.where(inputs == float(i))[0]) / inputs.numel() * i
        if lsar == 0:
            lsar = inputs.count_nonzero() / inputs.numel()
        weight = m.weight.data
        weight = weight.detach().cpu().numpy()
        # inputs = inputs.reshape(B,C,H,W)
        inputs = inputs.detach().cpu().numpy()
        fc_forward_with_sparsity(inputs,weight)
        pass
        if module_name not in MODULE_SOP_DICT.keys():
            MODULE_SOP_DICT[module_name] = lsar * batch_matrix_mul(inputs.shape,weight.shape)
        else:
            MODULE_SOP_DICT[module_name] += lsar * batch_matrix_mul(inputs.shape,weight.shape)

    return hook
