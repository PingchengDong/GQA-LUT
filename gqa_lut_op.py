import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class GQA_LUT(nn.Module):
    def __init__(self, pwl_type, pwl_dir, decimal_bit=6) -> None:
        """
        :param pwl_type: the required non-linear function to approximate
        :param pwl_dir: pretrained dir
        """
        super(GQA_LUT, self).__init__()
        # supported fp func, can be extended
        act_funcs = {
            'gelu': nn.GELU(),
            'hswish': nn.Hardswish(),
            'sigmoid': nn.Sigmoid(),
            'exp': torch.exp,
        }
        with open(pwl_dir, 'r') as f:
            params = json.load(f)
        self.func = act_funcs[pwl_type]
        self.pwl_params = params[pwl_type]

        self.breakpoints = None
        self.slopes = None
        self.intercepts = None
        self.decimal_bit = decimal_bit
        

    def forward(self, input, scale):
        """
        :param input: int8 input activation
        :param scale: scaling factor of the input activation
        :return: unquantized output after the GQA_LUT function
        """
        device = input.device
        # hw storage: sign|integer|decimal = 1|1|decimal_bit
        # obtain the scale factor's power bit
        power_bit = -int(torch.log2(scale))
        # fetch the params from the json file 
        # based on the power bit of the scale factor
        params = self.pwl_params[f'{power_bit}']

        # fetch the scaled breakpoints from the json file
        breakpoints = torch.tensor(params['breakpoints']).to(device)
        # scale the breakpoints with the scaling factor, eq(3)
        scaled_breakpoints = breakpoints / scale
        
        # fetch the intercepts from the json file
        intercepts = torch.tensor(params['intercept']).to(device)
        # shift the intercepts with the scale factor, eq(3)
        scaled_intercepts = intercepts / scale

        # fetch the slopes from the json file
        slopes = torch.tensor(params['slopes']).to(device)

        # recoder the scaled parameters for hardware deployment
        self.slopes = slopes
        self.intercepts = scaled_intercepts
        self.breakpoints = scaled_breakpoints

        # perform the pwl computation using torch.bucketize
        indices = torch.bucketize(input, scaled_breakpoints)
        slopes = slopes[indices]
        intercepts = scaled_intercepts[indices]
        pwl_func = slopes * input + intercepts
        
        # train in STE manner
        return (pwl_func * scale - self.func(input * scale)).detach() + self.func(input * scale)

def round_to_nearest_bits_torch(x, decimal_bits):
    """

    :param x: floating input
    :param decimal_bits: bits that the input should reserve
    :return: the formatted input with specific decimal bits
    """
    scaled_value = x * (2 ** decimal_bits)
    rounded_value = torch.round(scaled_value)  # very important
    result = rounded_value / (2 ** decimal_bits)
    y = result
    y_grad = x
    return (y - y_grad).detach() + y_grad

if __name__ == "__main__":
    fp32_test_input = torch.rand(10) # generate an arbitrary test input vector
    fp32_torchfunc_output = F.gelu(fp32_test_input) # the fp32 gelu activation function process using pytorch function
    scale = torch.tensor(2. ** -5) # dummy scaling factor, which should be obtained via QAT and transformed to power-of-two
    int8_test_input = torch.clamp(torch.round(fp32_test_input / scale), -128.0, 127.0) # quantize the fp32 input to int8
    gqa_hw_model = GQA_LUT(pwl_type='gelu', pwl_dir='pretrained/gelu_pwl_7.json') # instantiate the gqa hardware model
    fp32_gqa_output = gqa_hw_model(int8_test_input, scale) # process in gqa hw model
    print(fp32_torchfunc_output)
    print(fp32_gqa_output)