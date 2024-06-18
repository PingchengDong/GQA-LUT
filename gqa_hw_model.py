import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class GQA_HW_Model(nn.Module):
    def __init__(self, pwl_type, pwl_dir) -> None:
        """

        :param pwl_type: the required non-linear function to approximate
        :param pwl_dir: pretrained dir
        """
        super(GQA_HW_Model, self).__init__()
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
        """
        In real implementation, only one group of parameters is needed to be stored in hardware. By shifting it with
        the scale factor, we will get the corresponding parameters that meet the requirement of hardware operation precision.
        Therefore, to simulate it, we directly fetch the params from the json file with the highest scale decimal "6".
        Dividing the intercepts and the breakpoints by the scale factor will get the same results as we fetch the corresponding
        params with the decimal bit from the pretrained json.
        """
        self.breakpoints = torch.tensor(params[pwl_type]["6"]['breakpoints'])
        self.slopes = torch.tensor(params[pwl_type]["6"]['slopes'])
        self.intercepts = torch.tensor(params[pwl_type]["6"]['intercept'])

    def forward(self, input, scale):
        """

        :param input: int8 input, quantized in the last quantization layer (QAT)
        :param scale: scale factor of the input
        :return: int8 output after the non-linear function approximating computation
        """
        device = input.device
        decimal_bit = -torch.log2(scale)
        breakpoints_scaled = self.breakpoints.to(device)
        intercepts_scaled = round_to_nearest_bits_torch(self.intercepts.to(device), 6) / scale
        slopes_scaled = round_to_nearest_bits_torch(self.slopes.to(device), 6) # hw storage: sign|integer|decimal = 1|1|6
        breakpoints_scaled = round_to_nearest_bits_torch(breakpoints_scaled, decimal_bit)
        breakpoints_scaled = breakpoints_scaled / scale
        pwl_func = torch.zeros_like(input).to(device)
        mask = input.lt(breakpoints_scaled[0])
        pwl_func = torch.where(mask, intercepts_scaled[0] + slopes_scaled[0] * input, pwl_func)
        for i in range(1, len(breakpoints_scaled)):
            mask = input.ge(breakpoints_scaled[i - 1]) & input.lt(breakpoints_scaled[i])
            pwl_func = torch.where(mask, intercepts_scaled[i] + slopes_scaled[i] * input, pwl_func)
        mask = input.ge(breakpoints_scaled[-1])
        pwl_func = torch.where(mask, intercepts_scaled[-1] + slopes_scaled[-1] * input, pwl_func)
        return (pwl_func - self.func(input * scale) / scale).detach() + self.func(input * scale) / scale # train like lsq

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
    scale = torch.tensor(fp32_test_input.max() / 127.0) # get the scale by simply using the maximum of the input and the upper bound
    int8_test_input = torch.clamp(torch.round(fp32_test_input / scale), -128.0, 127.0) # quantize the fp32 input to int8
    gqa_hw_model = GQA_HW_Model(pwl_type='gelu', pwl_dir='pretrained/gelu_pwl_7.json') # instantiate the gqa hardware model
    int8_gqa_output = gqa_hw_model(int8_test_input, scale) # process in gqa hw model
    fp32_gqa_output = int8_gqa_output * scale # dequantize the output to fp32 again for further training or inference
    print(fp32_torchfunc_output)
    print(fp32_gqa_output)