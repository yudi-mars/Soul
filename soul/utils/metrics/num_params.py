"""
Filename: num_params.py
Author: Di Yu <yudi2023@zju.edu.cn>
Date Created: 2025-04-28
Description:
    implementation of counting parameters for SNN models.
    
References:
    - Di Yu et al., "ECC-SNN: Cost-Effective Edge-Cloud Collaboration for Spiking Neural Networks", IJCAI'2025
    https://github.com/AmazingDD/ECC-SNN
"""
def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())