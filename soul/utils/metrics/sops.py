"""
Filename:
    sops.py

Author:
    Wentao Tong <toldzera@zju.edu.cn>

Date Created:
    2025-05-17

Description:
    implementation of counting SOPs for SNN inference.

References:
    - Man Yao et al., "Attention Spiking Neural Networks", TPAMI'2023
    https://github.com/BICLab/Attention-SNN
    The way to calculate SOPs is described at the appendix file provided in https://github.com/BICLab/Attention-SNN/issues/3
"""
import os
import numpy as np

import torch
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODULE_SOP_DICT = {}
MODULE_FLOPS_DICT = {}
def ops_monitor(net, is_sop=False):
    m_dict = dict(net.named_modules())
    for key in m_dict.keys():
        if key == "":
            continue
        m = m_dict[key]
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_hook(ops_hook_conv2d(key + ".weight", is_sop))
        elif isinstance(m, torch.nn.Conv1d):
            m.register_forward_hook(ops_hook_conv1d(key + ".weight", is_sop))
        elif isinstance(m, torch.nn.Linear):
            m.register_forward_hook(ops_hook_fc(key + ".weight",is_sop))

def img2col(X, kernel_size, stride=1, pad=0):
    kh,kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    # kh = kw = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    sh = sw = stride if isinstance(stride, int) else stride[0]

    X_pad = F.pad(X, (pad, pad, pad, pad), mode='constant', value=0)
    
    return F.unfold(X_pad, (kh, kw), stride=(sh, sw))

def conv2d_forward_with_sparsity(X, W, b, stride=1, pad=0):
    cols = img2col(X, W.shape[2:], stride, pad)
    N, C, H, W_in = X.shape
    F_out, _, KH, KW = W.shape
    
    OH = (H + 2*pad - KH) // stride + 1
    OW = (W_in + 2*pad - KW) // stride + 1
    
    W_reshaped = W.view(F_out, -1)
    

    cols_nonzero = (cols != 0)
    cols_nonzero_count = cols_nonzero.sum(dim=0).sum(1) 
    W_nonzero = (W_reshaped != 0)
    W_nonzero_count = W_nonzero.sum(dim=0) 
    total_effective = torch.dot(cols_nonzero_count.float(), W_nonzero_count.float())
    total_multiplies = N * OH * OW * F_out * KH * KW * C
    
    effective_ratio = total_effective / total_multiplies if total_multiplies != 0 else torch.tensor(0.0)
    
    return effective_ratio.item()

import torch
import torch.nn.functional as F

def torch_img2col_grouped(X, kernel_size, stride=1, pad=0, dilation=1, groups=1):
    """
    支持分组卷积的img2col实现

    Args:
        X: 输入张量 (N, C, H, W)
        kernel_size: 卷积核大小 (int or tuple)
        stride: 步长 (int or tuple)
        pad: 填充大小
        dilation: 空洞大小
        groups: 分组数

    Returns:
        展开后的张量 (N, groups * kernel_size[0] * kernel_size[1], OH * OW * C//groups)
    """
    # 解析参数
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    sh, sw = stride if isinstance(stride, tuple) else (stride, stride)
    ph, pw = pad if isinstance(pad, tuple) else (pad, pad)
    dh, dw = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    
    N, C, H, W = X.shape
    assert C % groups == 0, "输入通道数必须能被groups整除"
    
    OH = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    OW = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    
    X_pad = F.pad(X, (pw, pw, ph, ph), mode='constant', value=0)

    channels_per_group = C // groups
    unfolded = []
    
    for g in range(groups):
        group_input = X_pad[:, g*channels_per_group:(g+1)*channels_per_group, :, :]
        group_unfold = F.unfold(
            group_input, 
            kernel_size=(kh, kw), 
            dilation=(dh, dw), 
            padding=0, 
            stride=(sh, sw)
        )
        unfolded.append(group_unfold)
    
    return torch.cat(unfolded, dim=1)

def grouped_conv2d_forward_with_sparsity(
    X, W, b, stride=1, pad=0, dilation=1, groups=1):
    """
    完全并行化的分组卷积前向传播，带向量化稀疏度检测

    Args:
        X: 输入数据 (N, C, H, W)
        W: 卷积核 (F, C//groups, KH, KW)
        b: 偏置 (F,)
        stride: 步长
        pad: 填充
        dilation: 空洞大小
        groups: 分组数

    Returns:
        output: 卷积结果 (N, F, OH, OW)
        effective_ratio: 有效计算比例 (0.0-1.0)
    """
    N, C, H, W_in = X.shape
    F_out, C_in_group, KH, KW = W.shape
    assert C % groups == 0, "输入通道数必须能被groups整除"
    assert F_out % groups == 0, "输出通道数必须能被groups整除"
    assert C_in_group == C // groups, "权重通道数必须匹配输入分组"
    
    OH = (H + 2*pad - dilation*(KH-1) - 1) // stride + 1
    OW = (W_in + 2*pad - dilation*(KW-1) - 1) // stride + 1
    
    X_pad = F.pad(X, (pad, pad, pad, pad), mode='constant', value=0)
    cols = F.unfold(
        X_pad, 
        kernel_size=(KH, KW), 
        dilation=dilation, 
        padding=0, 
        stride=stride
    ) 

    cols = cols.view(N, groups, C // groups, KH * KW, OH * OW)
    
    W_reshaped = W.view(groups, F_out // groups, C // groups, KH * KW)
    
    
    with torch.no_grad():
        cols_nonzero = (cols != 0).float()  # (N, groups, C//groups, KH*KW, OH*OW)
        
        W_nonzero = (W_reshaped != 0).float()  # (groups, F//groups, C//groups, KH*KW)
        
        elementwise_nonzero = torch.einsum(
            'gfcw, ngcwv -> ngfcv', 
            W_nonzero, 
            cols_nonzero
        )  # (N, groups, F//groups, C//groups, KH*KW, OH*OW)
        
        effective_per_position = elementwise_nonzero.sum(dim=(3, 4))  # (N, groups, F//groups, OH*OW)
        
        total_effective = effective_per_position.sum()
        
        total_multiplies = N * groups * (F_out // groups) * OH * OW * (C // groups) * KH * KW
    
    effective_ratio = total_effective / total_multiplies if total_multiplies != 0 else 0.0
    
    return effective_ratio.item()



def fc_forward_with_sparsity(X,W):
    # out = X @ W + b
    
    # sparsity calculation --------------------------------------------------
    # generate nonzero masks
    cols_nonzero = (X != 0)        # nonzero index for input
    cols_nonzero_count = cols_nonzero.sum(dim=0)
    W_nonzero = (W != 0)     # idx of nonzero element for kernel 
    # W_nonzero_count = W_nonzero.sum(dim=1) 
    # count number of multiply operations
    effective_matrix = torch.matmul(cols_nonzero_count.float(),W_nonzero.float().T)
    # effective_matrix = cols_nonzero.astype(np.int64) @ W_nonzero.astype(np.int64).T
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
def ops_hook_conv2d(module_name, is_sop=True):
    def hook(m, inputs, outputs):
        inputs = inputs[0]
        max_v = torch.max(inputs)
        is_float = not torch.floor(max_v) == max_v
        # is_mac = not torch.floor(max_v) == max_v
        lsar = 0
        stride = m.stride[0]
        padding = m.padding[0]
        kn, kn = m.kernel_size
        hn, wn = inputs.shape[-2:]
        in_channels = m.in_channels
        out_channels = m.out_channels

        B, C, H, W = inputs.shape
        # if lsar == 0:
        #     lsar = inputs.count_nonzero() / inputs.numel()
        weight = m.weight
        # inputs = inputs.reshape(B,C,H,W)
        inputs = inputs.reshape(-1, C, H, W)
        inputs = inputs.detach()
        if m.groups == 1:
            lsar = conv2d_forward_with_sparsity(inputs,weight,0,stride,padding)
        else:
            lsar = grouped_conv2d_forward_with_sparsity(inputs,weight,0,stride,padding,groups=m.groups)
        pass
        if is_float:
            if module_name not in MODULE_FLOPS_DICT.keys():
                MODULE_FLOPS_DICT[module_name] = lsar * kn * kn * hn * wn *  in_channels * out_channels * B
            else:
                MODULE_FLOPS_DICT[module_name] += lsar * kn * kn * hn * wn *  in_channels * out_channels * B
        else:
            if module_name not in MODULE_SOP_DICT.keys():
                MODULE_SOP_DICT[module_name] = lsar * kn * kn * hn * wn *  in_channels * out_channels * B
            else:
                MODULE_SOP_DICT[module_name] += lsar * kn * kn * hn * wn *  in_channels * out_channels * B
    return hook


def ops_hook_fc(module_name,is_sop=True):
    def hook(m,input,output):
        inputs = input[0]
        max_v = torch.max(inputs)
        is_mac = not torch.floor(max_v) == max_v
        # ran = max_v.detach().cpu().numpy().astype(int)
        # lsar = 0
        # for i in range(1, ran + 1):
        #     lsar += len(torch.where(inputs == float(i))[0]) / inputs.numel() * i
        # if lsar == 0:
        #     lsar = inputs.count_nonzero() / inputs.numel()
        weight = m.weight.data
        weight = weight.detach()
        inputs = inputs.detach()
        lsar = fc_forward_with_sparsity(inputs,weight)
        pass
        if is_mac:
            if module_name not in MODULE_FLOPS_DICT.keys():
                MODULE_FLOPS_DICT[module_name] = lsar * batch_matrix_mul(inputs.shape,weight.shape)
            else:
                MODULE_FLOPS_DICT[module_name] += lsar * batch_matrix_mul(inputs.shape,weight.shape)
        else:
            if module_name not in MODULE_SOP_DICT.keys():
                MODULE_SOP_DICT[module_name] = lsar * batch_matrix_mul(inputs.shape,weight.shape)
            else:
                MODULE_SOP_DICT[module_name] += lsar * batch_matrix_mul(inputs.shape,weight.shape)
        # if module_name not in MODULE_SOP_DICT.keys():
        #     if is_mac:
        #         MODULE_FLOPS_DICT[module_name] = batch_matrix_mul(inputs.shape,weight.shape)
        #     else:   
        #         MODULE_SOP_DICT[module_name] = lsar * batch_matrix_mul(inputs.shape,weight.shape)
        # else:
        #     if is_mac:
        #         MODULE_FLOPS_DICT[module_name] += batch_matrix_mul(inputs.shape,weight.shape)
        #     else:
        #         MODULE_SOP_DICT[module_name] += lsar * batch_matrix_mul(inputs.shape,weight.shape)
    return hook


# ==================== Conv1d Support ====================

def conv1d_forward_with_sparsity(X, W, b, stride=1, pad=0):
    """
    计算1D卷积的有效操作比例（考虑稀疏性）

    Args:
        X: 输入数据 (N, C, L)
        W: 卷积核 (F_out, C_in, K)
        b: 偏置
        stride: 步长
        pad: 填充

    Returns:
        effective_ratio: 有效计算比例 (0.0-1.0)
    """
    N, C, L = X.shape
    F_out, C_in, K = W.shape
    
    OL = (L + 2*pad - K) // stride + 1
    
    # 使用 unfold 展开1D卷积输入
    X_pad = F.pad(X, (pad, pad), mode='constant', value=0)
    # unfold: (N, C, L_pad) -> (N, C*K, OL)
    cols = X_pad.unfold(dimension=2, size=K, step=stride)  # (N, C, OL, K)
    cols = cols.permute(0, 1, 3, 2).contiguous()  # (N, C, K, OL)
    cols = cols.view(N, C * K, OL)  # (N, C*K, OL)
    
    W_reshaped = W.view(F_out, -1)  # (F_out, C*K)
    
    # 计算稀疏性
    cols_nonzero = (cols != 0)
    cols_nonzero_count = cols_nonzero.sum(dim=0).sum(1)  # (C*K,)
    W_nonzero = (W_reshaped != 0)
    W_nonzero_count = W_nonzero.sum(dim=0)  # (C*K,)
    total_effective = torch.dot(cols_nonzero_count.float(), W_nonzero_count.float())
    total_multiplies = N * OL * F_out * K * C
    
    effective_ratio = total_effective / total_multiplies if total_multiplies != 0 else torch.tensor(0.0)
    
    return effective_ratio.item()


def grouped_conv1d_forward_with_sparsity(
    X, W, b, stride=1, pad=0, dilation=1, groups=1):
    """
    分组1D卷积前向传播，带向量化稀疏度检测

    Args:
        X: 输入数据 (N, C, L)
        W: 卷积核 (F_out, C//groups, K)
        b: 偏置 (F_out,)
        stride: 步长
        pad: 填充
        dilation: 空洞大小
        groups: 分组数

    Returns:
        effective_ratio: 有效计算比例 (0.0-1.0)
    """
    N, C, L = X.shape
    F_out, C_in_group, K = W.shape
    assert C % groups == 0, "输入通道数必须能被groups整除"
    assert F_out % groups == 0, "输出通道数必须能被groups整除"
    assert C_in_group == C // groups, "权重通道数必须匹配输入分组"
    
    OL = (L + 2*pad - dilation*(K-1) - 1) // stride + 1
    
    X_pad = F.pad(X, (pad, pad), mode='constant', value=0)
    
    # 使用 unfold 进行1D展开，支持 dilation
    # 对于 dilation，需要手动处理
    cols_list = []
    for i in range(K):
        start_idx = i * dilation
        end_idx = X_pad.shape[2] - (K - 1 - i) * dilation
        col_slice = X_pad[:, :, start_idx:end_idx:stride]
        cols_list.append(col_slice)
    
    # cols: (N, C, K, OL)
    cols = torch.stack(cols_list, dim=2)
    cols = cols.view(N, groups, C // groups, K, OL)
    
    W_reshaped = W.view(groups, F_out // groups, C // groups, K)
    
    with torch.no_grad():
        cols_nonzero = (cols != 0).float()  # (N, groups, C//groups, K, OL)
        
        W_nonzero = (W_reshaped != 0).float()  # (groups, F//groups, C//groups, K)
        
        # einsum: 计算每个位置的有效乘法
        elementwise_nonzero = torch.einsum(
            'gfck, ngckv -> ngfcv', 
            W_nonzero, 
            cols_nonzero
        )  # (N, groups, F//groups, C//groups, OL)
        
        effective_per_position = elementwise_nonzero.sum(dim=(3, 4))  # (N, groups, F//groups)
        
        total_effective = effective_per_position.sum()
        
        total_multiplies = N * groups * (F_out // groups) * OL * (C // groups) * K
    
    effective_ratio = total_effective / total_multiplies if total_multiplies != 0 else 0.0
    
    return effective_ratio.item()


def ops_hook_conv1d(module_name, is_sop=True):
    """
    Conv1d层的前向钩子，用于统计SOPs/FLOPs
    """
    def hook(m, inputs, outputs):
        inputs = inputs[0]
        max_v = torch.max(inputs)
        is_float = not torch.floor(max_v) == max_v
        
        lsar = 0
        stride = m.stride[0]
        padding = m.padding[0]
        dilation = m.dilation[0]
        k = m.kernel_size[0]
        ln = inputs.shape[-1]
        in_channels = m.in_channels
        out_channels = m.out_channels

        B, C, L = inputs.shape
        weight = m.weight
        inputs = inputs.reshape(-1, C, L)
        inputs = inputs.detach()
        
        if m.groups == 1:
            lsar = conv1d_forward_with_sparsity(inputs, weight, 0, stride, padding)
        else:
            lsar = grouped_conv1d_forward_with_sparsity(
                inputs, weight, 0, stride, padding, dilation, groups=m.groups)
        
        ops_count = lsar * k * ln * in_channels * out_channels * B
        
        if is_float:
            if module_name not in MODULE_FLOPS_DICT.keys():
                MODULE_FLOPS_DICT[module_name] = ops_count
            else:
                MODULE_FLOPS_DICT[module_name] += ops_count
        else:
            if module_name not in MODULE_SOP_DICT.keys():
                MODULE_SOP_DICT[module_name] = ops_count
            else:
                MODULE_SOP_DICT[module_name] += ops_count
    return hook
