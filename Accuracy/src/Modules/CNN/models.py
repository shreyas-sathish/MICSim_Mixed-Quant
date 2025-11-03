import configparser, os

config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))

def load_model(model_name,mode):
    assert model_name in ['VGG8','ResNet18'], model_name
    assert mode in ['WAGE', 'WAGEV2','DynamicFixedPoint','FloatingPoint','LSQ'], mode

    if model_name == 'VGG8':
        if mode == 'WAGE':
            from Accuracy.src.Network.VGG8.WAGE.network import vgg8_load
            model = vgg8_load()
            return model
        if mode == 'WAGEV2':
            from Accuracy.src.Network.VGG8.WAGE_v2.network import vgg8_load
            model = vgg8_load()
            return model
        if mode == 'DynamicFixedPoint':
            from Accuracy.src.Network.VGG8.DynamicFixedPoint.network import vgg8_load
            model = vgg8_load()
            return model
        if mode == 'FloatingPoint':
            from Accuracy.src.Network.VGG8.FloatingPoint.network import vgg8_load
            model = vgg8_load()
            return model
        if mode == 'LSQ':
            from Accuracy.src.Network.VGG8.LSQ.network import vgg8_load
            model = vgg8_load()
            return model

    if model_name == 'ResNet18':
        if mode == 'WAGE':
            from Accuracy.src.Network.ResNet18.WAGE.network import resnet18
            model = resnet18()
            model = apply_layerwise_quantization(model, config)
            return model
        if mode == 'WAGEV2':
            from Accuracy.src.Network.ResNet18.WAGE_v2.network import resnet18
            model = resnet18()
            return model
        if mode == 'DynamicFixedPoint':
            from Accuracy.src.Network.ResNet18.DynamicFixedPoint.network import resnet18
            model = resnet18()
            return model
        if mode == 'FloatingPoint':
            from Accuracy.src.Network.ResNet18.FloatingPoint.network import resnet18
            model = resnet18()
            return model
        if mode == 'LSQ':
            from Accuracy.src.Network.ResNet18.LSQ.network import resnet18
            model = resnet18()
            model = apply_layerwise_quantization(model, config)
            return model

from Accuracy.src.Modules.CNN.Quantizer.LSQuantizer_custom import LSQuantizer

def apply_layerwise_quantization(model, config):
    import ast
    if 'layer_quant_config' not in config['Quantization']:
        return model
    layer_quant_config = ast.literal_eval(config['Quantization']['layer_quant_config'])
    quantizer = LSQuantizer(config)

    for name, module in model.named_modules():
        for lname, wbits, abits, scale in layer_quant_config:
            if hasattr(lname, 'quantizer'):
                layer.quantizer.w_bits = wbits
                layer.quantizer.a_bits = abits
                layer.quantizer.scaling = scaling
            if lname in name and hasattr(module, 'set_precision'):
                module.set_precision(wbits, abits, scale)
                # Quantize the pretrained FP32 weights now
                with torch.no_grad():
                    if hasattr(module, 'weight'):
                        q_weight, _, _, _ = quantizer.QuantizeWeight(module.weight, bits=wbits, train=False)
                        module.weight.data.copy_(q_weight)
    return model