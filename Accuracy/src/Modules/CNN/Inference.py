import torch
from torch.autograd import Variable
import torch.nn as nn
from Accuracy.src.utils import  misc,make_path
from Accuracy.src.Modules.CNN import optimizer
import os
from torch.autograd import profiler

logger = misc.logger.info


import configparser
import os
config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))
num_epoch          = int(config['Training']['numEpoch'])
decreasing_lr      =     config['Training']['decreasing_lr']
quantization_mode  =     config['Quantization']['mode']
train_log_interval = int(config['Training']['train_log_interval'])
val_log_interval   = int(config['Training']['val_log_interval'])
DumpData = config['ADC']['dumpdata']
dump_average_value = config['Quantization']['dumpaveragevalue']

# todo
test_10 = False

class Tester():
    def __init__(self,model,train_loader,test_loader, layerwise_bits=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.layerwise_bits = layerwise_bits or []
        self.optimizer = optimizer.optimizer(self.model)
        self.criterion = optimizer.loss_func()
        self.cuda = torch.cuda.is_available()
        self.best_acc = 0
        self.old_file = None
        self.logdir =  make_path.makepath_logdir()
        self.layerwise_bits = layerwise_bits or []

        if self.layerwise_bits:
            self.apply_layerwise_config(self.model, self.layerwise_bits)
        
    def apply_layerwise_config(self, model, layerwise_bits):
        """
        layerwise_bits: list of tuples (layer_name, weight_bits, act_bits, scaling)
        Attaches scaling and per-layer quantization precision attributes.
        """
        config_map = {n: (w, a, s) for (n, w, a, s) in layerwise_bits}

        for full_name, module in model.named_modules():
            if full_name == '':
                continue
            for lname, (wbits, abits, scale) in config_map.items():
                if full_name.endswith(lname) or lname in full_name:
                    setattr(module, 'scaling', float(scale))
                    if hasattr(module, 'quantizer') and module.quantizer is not None:
                        setattr(module.quantizer, 'layer_weight_precision', int(wbits))
                        setattr(module.quantizer, 'layer_input_precision', int(abits))
                    break
                
        try:
            from Accuracy.src.utils import misc
            logger = misc.logger.info
            logger("Applied layer-wise quantization configuration:")
            for full_name, module in model.named_modules():
                if hasattr(module, 'scaling') or (
                    hasattr(module, 'quantizer') 
                    and getattr(module.quantizer, 'layer_weight_precision', None) is not None
                ):
                    logger(
                        f"\t{full_name}: scaling={getattr(module, 'scaling', 1.0)}, "
                        f"wbits={getattr(module.quantizer, 'layer_weight_precision', 'NA') if hasattr(module, 'quantizer') else 'NA'}, "
                        f"abits={getattr(module.quantizer, 'layer_input_precision', 'NA') if hasattr(module, 'quantizer') else 'NA'}"
                    )
        except Exception:
            pass




    def _val(self):
        
        self.model.eval()
        test_loss = 0
        correct = 0
        logger("===================== testing phase =====================")
        #print('only check 1/100 image to save time, accuracy need be re calculated by 31*16 in total')
        for i, (data, target) in enumerate(self.test_loader):
            # calculate test loss and accuracy for batch
            indx_target = target.clone()
            if self.cuda:
                data, target = data.cuda(), target.cuda() 
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
                output = self.model(data)
                test_loss_i = self.criterion(output, target)
                test_loss += test_loss_i.data
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.cpu().eq(indx_target).sum()
            if DumpData == 'True':
                print("data collect complete")
                return None
            if dump_average_value == 'True':
                break
            
            if test_10 == True:
                break
        # calculate total test loss and accuracy
        test_loss = test_loss / len(self.test_loader)  # average over number of mini-batch
        if dump_average_value == 'True' or test_10 == True:
             acc = 100. * correct / len(data)
             logger('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(data), acc))
        else:     
            acc = 100. * correct / len(self.test_loader.dataset)
            logger('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(self.test_loader.dataset), acc))