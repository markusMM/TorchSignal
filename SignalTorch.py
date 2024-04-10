# -*- coding: utf-8 -*-
"""
Signal Processing tools in torch

@author: Markus Meister
"""

import torch
import math

torch.pi = torch.tensor(math.pi)

def MoveLastToFirst(x):
    """
    Move the last axis to the first position.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The tensor with the last axis moved to the first position.
    """
    for xi in range(len(x.shape)-1):
        x = x.transpose(xi,-1)
    return x

def MoveFirstToLast(x):
    """
    Move the first axis to the last position.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The tensor with the first axis moved to the last position.
    """
    for xi in range(len(x.shape)-1):
        x = x.transpose(xi,xi+1)
    return x

def roll(x, n):
    """
    Roll the elements of the tensor along the specified axis.

    Args:
        x (torch.Tensor): The input tensor.
        n (int): The number of places by which elements are shifted.

    Returns:
        torch.Tensor: The rolled tensor.
    """
    return torch.cat((x[-n:], x[:-n]))

def shft(x, n):
    """
    Shift the elements of the tensor along the specified axis.

    Args:
        x (torch.Tensor): The input tensor.
        n (int): The number of places by which elements are shifted.

    Returns:
        torch.Tensor: The shifted tensor.
    """
    n = int(n*(1 - n // x.shape[0]))
    if n == 0:
        return x
    return torch.cat((
            0*x[-n:],
            1*x[:-n],
        ))

def roll_mat(x,order,strafe=1):
    """
    Create a matrix of rolled versions of the input tensor.

    Args:
        x (torch.Tensor): The input tensor.
        order (int): The number of times to roll the tensor.
        strafe (int, optional): The strafe value. Defaults to 1.

    Returns:
        torch.Tensor: The matrix of rolled versions of the input tensor.
    """
    Xe = torch.zeros(order,*x.shape).to(x.device)
    for m in range(order):
        Xe[m] = roll(x,m*strafe)
    return Xe

def shft_mat(x,order,strafe=1):
    """
    Create a matrix of shifted versions of the input tensor.

    Args:
        x (torch.Tensor): The input tensor.
        order (int): The number of times to shift the tensor.
        strafe (int, optional): The strafe value. Defaults to 1.

    Returns:
        torch.Tensor: The matrix of shifted versions of the input tensor.
    """
    Xe = torch.zeros(order,*x.shape).to(x.device)
    for m in range(order):
        Xe[m] = shft(x,m*strafe)
    return Xe

def ConvT(x, filter, strafe = 1):
    '''
    Inplace convolution function
    
    Does convolve two a filter tensor with a signal tensor along the signal axis.
    The output length though will be the the signal length.
    The Equation is unnormed:
    ..math:
        y[n] = sum_k{h[k]x[n-k]}$ with $k\in[0,..,K-1}$ and $n\in[0,..,N-1]
    Here, $K$ is the signal length.

    Args:
        x (torch.Tensor): The input tensor.
        filter (torch.Tensor): The convolution filter tensor.
        strafe (int, optional): The strafe value. Defaults to 1.

    Returns:
        torch.Tensor: The result of the convolution operation.
    '''
    t1_flg = 0
    
    Xe = roll_mat(x, filter.shape[-1], strafe)
    Xe = MoveFirstToLast(Xe)
    
    if Xe.shape[0] != filter.shape[0]:
        Xe = torch.ones([filter.shape[0],*Xe.shape]).to(x.device) * Xe[None]
        t1_flg = 1
    
    filter = torch.ones(Xe.shape).to(x.device)*filter[:,None,None]
    
    Xe = (filter * Xe).sum(dim=-1)
    
    if t1_flg:
        Xe = Xe.transpose(0,1)
    
    return Xe

def WinT(ten, win_type='ham', win_coef=None):
    """
    Apply a windowing function to the input tensor.

    Args:
        ten (torch.Tensor): The input tensor.
        win_type (str, optional): The type of windowing function to apply. Defaults to 'ham'.
        win_coef (float or list, optional): The windowing function coefficient. Defaults to None.

    Returns:
        torch.Tensor: The input tensor after applying the windowing function.
    """
    if type(win_type) == type(None):
        win_type = 'ham'
    
    if len(win_type) < 3:
        win_type += '   '
    
    if win_type[:3] == 'han':
        if type(win_coef) == type(None):
            win_coef = .50
        win = win_coef * (1 - torch.cos( 2*torch.pi*torch.arange(ten.shape[0]) / ten.shape[0] ))
    if win_type[:3] == 'ham':
        if type(win_coef) == type(None):
            win_coef = .54
        win = win_coef * (1 - torch.cos( 2*torch.pi*torch.arange(ten.shape[0]) / ten.shape[0] ))
    if win_type[:3] == 'bla':
        if type(win_coef) == type(None):
            win_coef = [.16, '.5*(1-a)', '.5', '.5*a']
        a = win_coef[0]
        win = eval(win_coef[1]) 
        rew = torch.arange(ten.shape[0]) / ten.shape[0]
        for c, coef in enumerate(win_coef[1:]):
            win += eval(coef) * torch.cos( 2*(c+1)*torch.pi*rew )
    if win_type[:3] == 'rec':
        if type(win_coef) == type(None):
            win_coef = 1
        win = win_coef * torch.ones(ten.shape[0])
    if win_type[:3] == 'tri':
        if type(win_coef) == type(None):
            win_coef = 1
        win = win_coef * (1 - torch.arange(ten.shape[0]) / ten.shape[0])
    
    
    for i in range(len(ten.shape)-1):
        win = win.unsqueeze(i+1)
    
    return ten * win
    
