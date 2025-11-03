from .Quantizer import Quantizer
import torch
import torch.nn as nn
import math
from torch.autograd import Function
import numpy as np

class LSQ(Function):
    @staticmethod
    def forward(ctx, tensor, scale, g, Qn, Qp):
        ctx.save_for_backward(tensor, scale)
        ctx.other = g, Qn, Qp
        
        Qtensor = torch.div(tensor, scale).round().clamp(Qn, Qp)
        
        return Qtensor, scale

    @staticmethod
    def backward(ctx, grad_tensor, grad_scale):
        tensor, scale = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = tensor / scale
        smaller = (q_w < Qn).float() 
        bigger = (q_w > Qp).float() 
        between = 1.0 - smaller -bigger 
        # grad_alpha = ((smaller * Qn + bigger * Qp + 
        #         between * Round.apply(q_w) - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0)
        # remove grad_weight:
        grad_scale = ((smaller * Qn + bigger * Qp + 
                between * (q_w.round()) - between * q_w)* grad_tensor* g).sum().unsqueeze(dim=0)
        # grad_scale = ((smaller * Qn + bigger * Qp + 
        #         between * (q_w.round()) - between * q_w)*grad_tensor * g).sum().unsqueeze(dim=0)

        grad_tensor = between * grad_tensor
        return grad_tensor, grad_scale, None, None, None, None

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

class LSQuantizer(Quantizer):
    
    def __init__(self, config=None):
        super(LSQuantizer, self).__init__()
        
        # self.weight_scale  = nn.Parameter(torch.ones(1), requires_grad=True)
        # self.input_scale   =  nn.Parameter(torch.ones(1), requires_grad=True)
        self.weight_scale  = nn.Parameter(torch.ones(1), requires_grad=True)
        self.input_scale   =  nn.Parameter(torch.ones(1), requires_grad=True)
        # self.weight_scale  = nn.Parameter(torch.tensor([float(1)]).reshape((1)), requires_grad=True)
        # self.input_scale   =  nn.Parameter(torch.tensor([float(1)]).reshape((1)), requires_grad=True)
        self.WQn, self.WQp = self.getQnQp(True,self.weight_precision)
        self.IQn, self.IQp = self.getQnQp(True,self.input_precision)
        self.init_state_i = 0
        self.init_state_w = 0
        
    
    def weight_init(self, weight, bits_W=None,factor=2.0, mode="fan_in"):
        scale = 1.0
        # if bits_W is None:
        #     bits_W = self.weight_precision
            
        # if mode != "fan_in":
        #     raise NotImplementedError("support only wage normal")
        # dimensions = weight.ndimension()
        # if dimensions < 2: raise ValueError("weight at least is 2d")
        # elif dimensions == 2: fan_in = weight.size(1)
        # elif dimensions > 2:
        #     num_input_fmaps = weight.size(1)
        #     receptive_field_size = 1
        #     if weight.dim() > 2:
        #         receptive_field_size = weight[0][0].numel()
        #     fan_in = num_input_fmaps * receptive_field_size
        # # This is a magic number, copied
        # float_limit = math.sqrt( 3 * factor / fan_in)/1
        # float_std =  math.sqrt( 2 / fan_in)/1
        # quant_limit,scale = self.scale_limit(float_std, bits_W)
        # weight.data.uniform_(-quant_limit, quant_limit)

        # print("fan_in {:6d}, float_limit {:.6f}, float std {:.6f}, quant limit {}, scale {}".format(fan_in, float_limit, float_std, quant_limit, scale))
        return scale   
    
    def update_range(self, input):
        pass
    
    def input_clamp(self, input):
        return input
    
    def getQnQp(self, signed, bits):
        if signed == False:
            Qn = 0
            Qp = 2 ** bits - 1
        else:
            Qn = - 2 ** (bits - 1) + 1
            Qp = 2 ** (bits - 1) - 1
        return Qn, Qp  
                      
    def QuantizeWeight(self, weight, bits=None, Wsigned=True,train=None):
        # # if Wsigned == False:
        # #     fixed_range = [-1+1/2**(self.weight_precision-1),1-1/2**(self.weight_precision-1)]
        # #     weight,weightscale,weightrange,weightshift = self.Q_to_unsigned_value(weight,self.weight_precision,fixed_range,None,odd_stage=True)
        # self.weight_g = 1.0/math.sqrt(weight.numel() * self.WQp)
        # if self.init_state_w == 0 and train == True:   
        #     self.weightscalevalue = torch.mean(torch.abs(weight.detach()))*2/(math.sqrt(self.WQp))
        #     self.weight_scale.data = self.weightscalevalue.unsqueeze(0)
        #     self.init_state_w += 1
        # # else:
        # #     self.weightscalevalue = 0.9*self.weightscalevalue + 0.1*torch.mean(torch.abs(weight.detach()))*2/(math.sqrt(self.WQp))
        # #     self.weight_scale.data = self.weightscalevalue.unsqueeze(0)
                
        # # weightscalevalue = torch.mean(torch.abs(weight.detach()))*2/(math.sqrt(self.WQp))
        # # self.weight_scale.data = weightscalevalue.unsqueeze(0)
        #     # self.weight_scale.data = torch.mean(torch.abs(weight.detach()))*2/(math.sqrt(self.WQp))
        # # method1:    
        # # weight, weightscale = LSQ.apply(weight, self.weight_scale, self.weight_g, self.WQn, self.WQp)
        # # method2:
        
        
        # weightscale = grad_scale(self.weight_scale, self.weight_g)
        # weight = weight / weightscale
        # weight = torch.clamp(weight, self.WQn, self.WQp)
        # weight = round_pass(weight)
        
        # weightrange = []
        # weightshift = 0.0
        
        # if Wsigned == False:
        #     weight += 2 ** (self.weight_precision - 1)
        #     weightshift  = -(2 ** (self.weight_precision - 1))

        # --- per‐layer override: check for layer specific precisions & scaling ---
        layer_bits = None
        layer_scale = 1.0
        if hasattr(self, 'layer_weight_precision') and self.layer_weight_precision is not None:
            layer_bits = int(self.layer_weight_precision)
        if hasattr(self, 'layer_scaling') and self.layer_scaling is not None:
            layer_scale = float(self.layer_scaling)
        
        # if no override, use existing class defaults
        if layer_bits is None:
            layer_bits = self.weight_precision
        
        # compute quantization bounds based on layer_bits
        WQp = 2**(int(layer_bits) - 1) - 1
        WQn = -2**(int(layer_bits) - 1)

        # apply layer_scale to weight before quantization
        weight = weight * layer_scale

        # adjust gradient scaling with layer scale
        self.weight_g = 1.0 / math.sqrt(weight.numel() * WQp)

        # existing initialization (unchanged)
        if self.init_state_w == 0 and train == True:
            self.weightscalevalue = torch.mean(torch.abs(weight.detach()))*2/(math.sqrt(self.WQp))
            self.weight_scale.data = self.weightscalevalue.unsqueeze(0)
            self.init_state_w += 1
        
        weightscale = grad_scale(self.weight_scale, self.weight_g)
        weight = weight / weightscale
        weight = torch.clamp(weight, WQn, WQp)
        weight = round_pass(weight)

        weightrange = []
        weightshift = 0.0

        # if Wsigned == False:
        #     weight += 2 ** (layer_bits - 1)
        #     weightshift = -(2 ** (layer_bits - 1))

        if Wsigned == False:
            # use layer bits if provided, else class precision
            if layer_bits is not None:
                shift_val = 2 ** (int(layer_bits) - 1)
            else:
                shift_val = 2 ** (self.weight_precision - 1)
            weight += shift_val
            weightshift = -shift_val
        return weight, weightscale, weightrange, weightshift 

    def QuantizeInput(self, input, bits=None, Isigned=True,train=None):
        
        # self.input_g = 1.0/math.sqrt(input.numel() * self.IQp)

        # --- per‐layer override for input precision & scaling ---
        layer_bits = None
        layer_scale = 1.0
        if hasattr(self, 'layer_input_precision') and self.layer_input_precision is not None:
            layer_bits = int(self.layer_input_precision)
        if hasattr(self, 'layer_scaling') and self.layer_scaling is not None:
            layer_scale = float(self.layer_scaling)
        
        if layer_bits is None:
            layer_bits = self.input_precision
        
        IQp = 2**(layer_bits - 1) - 1
        IQn = -2**(layer_bits - 1)

        # apply layer_scale to input before quantization
        input = input * layer_scale

        self.input_g = 1.0 / math.sqrt(input.numel() * IQp)

        # if self.init_state_i == 0:
        #     self.inputscalevalue = torch.mean(torch.abs(input.detach()))*2/(math.sqrt(self.IQp))
        #     self.input_scale.data = self.inputscalevalue.unsqueeze(0)
        #     self.init_state_i += 1
        # else:
        #     self.inputscalevalue = 0.9*self.inputscalevalue + 0.1*torch.mean(torch.abs(input.detach()))*2/(math.sqrt(self.IQp))
        #     self.input_scale.data = self.inputscalevalue.unsqueeze(0)
        # elif self.init_state<self.batch_init:
        # else:   
        #     inputscalevalue = torch.mean(torch.abs(input.detach()))*2/(math.sqrt(self.IQp))
        
        # self.input_scale.data = inputscalevalue.unsqueeze(0)
        # self.input_scale.data = torch.mean(torch.abs(input.detach()))*2/(math.sqrt(self.IQp)).unsqueeze(0)
        # method1:
        # input,inputscale = LSQ.apply(input, self.input_scale, self.input_g, self.IQn, self.IQp)
        # method2:
        inputscale = grad_scale(self.input_scale, self.input_g)

        input = input / inputscale
        # input = torch.clamp(input, self.IQn, self.IQp)
        input = torch.clamp(input, IQn, IQp)
        input = round_pass(input)
        
        inputrange = []
        inputshift = 0.0    
        
        return input, inputscale, inputrange, inputshift
    
    def QuantizeError(self, error, bits=None, Esigned=True):
        if bits is  None:
            bits= self.error_precision
               
        error, errorscale, errorrange,errorshift = self.Q(error,self.error_precision,signed=Esigned, 
                                                          fixed_range=range,
                                                          odd_stage=True)
        
        return error, errorscale, errorrange, errorshift 

    def quantize_grad(self, x): 
        raise NotImplementedError("use QSGD")