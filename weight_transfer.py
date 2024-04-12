import torch
import torch.nn as nn
from tse_model import *
from SpNet_ppt_arch import *


device='cuda'

mappings={'encoder.encoder.0.mconv.0.conv.weight': 'encoder_0.conv_r0_0.weight',
 'encoder.encoder.0.mconv.1.conv.weight': 'encoder_0.conv_r1_0.weight',
 'encoder.encoder.0.mconv.2.weight': 'encoder_0.b_norm_r0.weight',
 'encoder.encoder.0.mconv.2.bias': 'encoder_0.b_norm_r0.bias',
 'encoder.encoder.0.mconv.2.running_mean': 'encoder_0.b_norm_r0.running_mean',
 'encoder.encoder.0.mconv.2.running_var': 'encoder_0.b_norm_r0.running_var',
 'encoder.encoder.0.mconv.2.num_batches_tracked': 'encoder_0.b_norm_r0.num_batches_tracked',
 'encoder.encoder.0.res.0.0.conv.weight': 'encoder_0.conv_res.weight',
 'encoder.encoder.0.res.0.1.weight': 'encoder_0.b_norm_res.weight',
 'encoder.encoder.0.res.0.1.bias': 'encoder_0.b_norm_res.bias',
 'encoder.encoder.0.res.0.1.running_mean': 'encoder_0.b_norm_res.running_mean',
 'encoder.encoder.0.res.0.1.running_var': 'encoder_0.b_norm_res.running_var',
 'encoder.encoder.0.res.0.1.num_batches_tracked': 'encoder_0.b_norm_res.num_batches_tracked',
 'encoder.encoder.1.mconv.0.conv.weight': 'encoder_1.conv_r0_0.weight',
 'encoder.encoder.1.mconv.1.conv.weight': 'encoder_1.conv_r0_1.weight',
 'encoder.encoder.1.mconv.2.weight': 'encoder_1.b_norm_r0.weight',
 'encoder.encoder.1.mconv.2.bias': 'encoder_1.b_norm_r0.bias',
 'encoder.encoder.1.mconv.2.running_mean': 'encoder_1.b_norm_r0.running_mean',
 'encoder.encoder.1.mconv.2.running_var': 'encoder_1.b_norm_r0.running_var',
 'encoder.encoder.1.mconv.2.num_batches_tracked': 'encoder_1.b_norm_r0.num_batches_tracked',
 'encoder.encoder.1.mconv.5.conv.weight': 'encoder_1.conv_r1_0.weight',
 'encoder.encoder.1.mconv.6.conv.weight': 'encoder_1.conv_r1_1.weight',
 'encoder.encoder.1.mconv.7.weight': 'encoder_1.b_norm_r1.weight',
 'encoder.encoder.1.mconv.7.bias': 'encoder_1.b_norm_r1.bias',
 'encoder.encoder.1.mconv.7.running_mean': 'encoder_1.b_norm_r1.running_mean',
 'encoder.encoder.1.mconv.7.running_var': 'encoder_1.b_norm_r1.running_var',
 'encoder.encoder.1.mconv.7.num_batches_tracked': 'encoder_1.b_norm_r1.num_batches_tracked',
 'encoder.encoder.1.res.0.0.conv.weight': 'encoder_1.conv_res.weight',
 'encoder.encoder.1.res.0.1.weight': 'encoder_1.b_norm_res.weight',
 'encoder.encoder.1.res.0.1.bias': 'encoder_1.b_norm_res.bias',
 'encoder.encoder.1.res.0.1.running_mean': 'encoder_1.b_norm_res.running_mean',
 'encoder.encoder.1.res.0.1.running_var': 'encoder_1.b_norm_res.running_var',
 'encoder.encoder.1.res.0.1.num_batches_tracked': 'encoder_1.b_norm_res.num_batches_tracked',
 'encoder.encoder.2.mconv.0.conv.weight': 'encoder_2.conv_r0_0.weight',
 'encoder.encoder.2.mconv.1.conv.weight': 'encoder_2.conv_r0_1.weight',
 'encoder.encoder.2.mconv.2.weight': 'encoder_2.b_norm_r0.weight',
 'encoder.encoder.2.mconv.2.bias': 'encoder_2.b_norm_r0.bias',
 'encoder.encoder.2.mconv.2.running_mean': 'encoder_2.b_norm_r0.running_mean',
 'encoder.encoder.2.mconv.2.running_var': 'encoder_2.b_norm_r0.running_var',
 'encoder.encoder.2.mconv.2.num_batches_tracked': 'encoder_2.b_norm_r0.num_batches_tracked',
 'encoder.encoder.2.mconv.5.conv.weight': 'encoder_2.conv_r1_0.weight',
 'encoder.encoder.2.mconv.6.conv.weight': 'encoder_2.conv_r1_1.weight',
 'encoder.encoder.2.mconv.7.weight': 'encoder_2.b_norm_r1.weight',
 'encoder.encoder.2.mconv.7.bias': 'encoder_2.b_norm_r1.bias',
 'encoder.encoder.2.mconv.7.running_mean': 'encoder_2.b_norm_r1.running_mean',
 'encoder.encoder.2.mconv.7.running_var': 'encoder_2.b_norm_r1.running_var',
 'encoder.encoder.2.mconv.7.num_batches_tracked': 'encoder_2.b_norm_r1.num_batches_tracked',
 'encoder.encoder.2.res.0.0.conv.weight': 'encoder_2.conv_res.weight',
 'encoder.encoder.2.res.0.1.weight': 'encoder_2.b_norm_res.weight',
 'encoder.encoder.2.res.0.1.bias': 'encoder_2.b_norm_res.bias',
 'encoder.encoder.2.res.0.1.running_mean': 'encoder_2.b_norm_res.running_mean',
 'encoder.encoder.2.res.0.1.running_var': 'encoder_2.b_norm_res.running_var',
 'encoder.encoder.2.res.0.1.num_batches_tracked': 'encoder_2.b_norm_res.num_batches_tracked',
 'encoder.encoder.3.mconv.0.conv.weight': 'encoder_3.conv_r0_0.weight',
 'encoder.encoder.3.mconv.1.conv.weight': 'encoder_3.conv_r0_1.weight',
 'encoder.encoder.3.mconv.2.weight': 'encoder_3.b_norm_r0.weight',
 'encoder.encoder.3.mconv.2.bias': 'encoder_3.b_norm_r0.bias',
 'encoder.encoder.3.mconv.2.running_mean': 'encoder_3.b_norm_r0.running_mean',
 'encoder.encoder.3.mconv.2.running_var': 'encoder_3.b_norm_r0.running_var',
 'encoder.encoder.3.mconv.2.num_batches_tracked': 'encoder_3.b_norm_r0.num_batches_tracked',
 'encoder.encoder.3.mconv.5.conv.weight': 'encoder_3.conv_r1_0.weight',
 'encoder.encoder.3.mconv.6.conv.weight': 'encoder_3.conv_r1_1.weight',
 'encoder.encoder.3.mconv.7.weight': 'encoder_3.b_norm_r1.weight',
 'encoder.encoder.3.mconv.7.bias': 'encoder_3.b_norm_r1.bias',
 'encoder.encoder.3.mconv.7.running_mean': 'encoder_3.b_norm_r1.running_mean',
 'encoder.encoder.3.mconv.7.running_var': 'encoder_3.b_norm_r1.running_var',
 'encoder.encoder.3.mconv.7.num_batches_tracked': 'encoder_3.b_norm_r1.num_batches_tracked',
 'encoder.encoder.3.res.0.0.conv.weight': 'encoder_3.conv_res.weight',
 'encoder.encoder.3.res.0.1.weight': 'encoder_3.b_norm_res.weight',
 'encoder.encoder.3.res.0.1.bias': 'encoder_3.b_norm_res.bias',
 'encoder.encoder.3.res.0.1.running_mean': 'encoder_3.b_norm_res.running_mean',
 'encoder.encoder.3.res.0.1.running_var': 'encoder_3.b_norm_res.running_var',
 'encoder.encoder.3.res.0.1.num_batches_tracked': 'encoder_3.b_norm_res.num_batches_tracked',
 'encoder.encoder.4.mconv.0.conv.weight': 'encoder_4.conv_r0_0.weight',
 'encoder.encoder.4.mconv.1.conv.weight': 'encoder_4.conv_r1_0.weight',
 'encoder.encoder.4.mconv.2.weight': 'encoder_4.b_norm_r1.weight',
 'encoder.encoder.4.mconv.2.bias': 'encoder_4.b_norm_r1.bias',
 'encoder.encoder.4.mconv.2.running_mean': 'encoder_4.b_norm_r1.running_mean',
 'encoder.encoder.4.mconv.2.running_var': 'encoder_4.b_norm_r1.running_var',
 'encoder.encoder.4.mconv.2.num_batches_tracked': 'encoder_4.b_norm_r1.num_batches_tracked',
 'decoder.emb_layers.0.0.weight': 'L1.weight',
 'decoder.emb_layers.0.0.bias': 'L1.bias'}

model_spnet = torch.load('/data/Nivedita/Arun/TSE/logs--val_loss=5.7506-epoch=34.ckpt') #loading the ckpt which has all the information

model_checkpoint = model_spnet['state_dict'] #loading its specific state_dict ( names and weights of the model)

new_state_dict={} #creating a new dict for mapping state_dict

for ckpt_param_name,model_param_name in mappings.items():
    new_state_dict[model_param_name.strip()] = model_checkpoint[ckpt_param_name.strip()]



model_1 = SpeakerNet().to(device)  # intialize the coded architecture code
model_1.load_state_dict(new_state_dict) # load its state_dict with the new state_dict which contains the ckpt weight 


print("loaded the new state dict")

#print(model_1.state_dict().keys()) # prints all the layers names of model_1

print(model_1.state_dict()['encoder_0.conv_r0_0.weight']) # getting the weight tensor of the specific layer name

print(model_checkpoint['encoder.encoder.0.mconv.0.conv.weight'])

# checking the loaded weights and the ckpt correspondig weights are same or not.

assert torch.equal(model_1.state_dict()['encoder_0.conv_r0_0.weight'],model_checkpoint['encoder.encoder.0.mconv.0.conv.weight'])

print("both are same")

'''
model_2 = SpeakerNet() # create the class object

# way1
for name,param in model_2.named_params():
    print(name,param) # this prints the name and corresponding weights we can also print the shape.

# way 2

model_2_st_dict = model_2.state_dict()

# this contains keys and values of the network.

'''



