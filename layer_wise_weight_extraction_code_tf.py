        for i, layer in enumerate(self.model.layers):
            print(f'Layer {i + 1}: {layer.name}')
            if isinstance(layer, EncoderBlock):
                for sub_layer in layer.layers:
                    print(f'  Sub-layer: {sub_layer.name}')
                    if isinstance(sub_layer, ResidualBlock):
                        # Handle ResidualBlock layers
                        for res_sub_layer in sub_layer.layers:
                            print(f'    Residual Sub-layer: {res_sub_layer.name}')
                            weights = res_sub_layer.get_weights()
                            if weights:
                                for j, w in enumerate(weights):
                                    print(f'      Weight {j + 1} shape: {w.shape}')
                                    f.write(w)
                                    
                            else:
                                print('      No weights for this residual sub-layer')
                    else:
                        weights = sub_layer.get_weights()
                        if weights:
                            for j, w in enumerate(weights):
                                print(f'    Weight {j + 1} shape: {w.shape}')
                                f.write(w)
                                
                        else:
                            print('    No weights for this sub-layer')
            else:
                if isinstance(layer, DecoderBlock):
                    print(f'  Decoder Block:')
                    # Handle ResidualBlock within DecoderBlock
                    res_block = layer.residual_block
                    print(f'    Residual Block: {res_block.name}')
                    for res_sub_layer in res_block.layers:
                        print(f'      Residual Sub-layer: {res_sub_layer.name}')
                        weights = res_sub_layer.get_weights()
                        if weights:
                            for j, w in enumerate(weights):
                                print(f'        Weight {j + 1} shape: {w.shape}')
                                f.write(w)
                        else:
                            print('        No weights for this residual sub-layer')

                    # Handle SubpixelBlock within DecoderBlock
                    subpixel_block = layer.sub_pixel_block
                    print(f'    Subpixel Block: {subpixel_block.name}')
                    weights = subpixel_block.get_weights()
                    if weights:
                        for j, w in enumerate(weights):
                            print(f'      Weight {j + 1} shape: {w.shape}')
                            f.write(w)
                    else:
                        print('      No weights for this subpixel block')
                else:
                    weights = layer.get_weights()
                    if weights:
                        for j, w in enumerate(weights):
                            print(f'  Weight {j + 1} shape: {w.shape}')
                            f.write(w)
                    else:
                        print('  No weights for this layer')
            print('-' * 30)

if np.ndim(w) > 2:
    for idx in range(w.shape[0]):
        f.write(f'indices of the dim 0:{idx} \n')
        #print(w.shape)
        w1 = w[idx:idx+1,:,:,:]
        w1 = np.squeeze(w1)
        #print("in first for loop",w.shape)
        for idx_1 in range(w1.shape[0]):
            f.write(f'Layer {i + 1}: {layer.name} indices of the dim1:{idx_1} \n')
            #print("second loop begin",w1.shape)
            w2 = w1[idx_1:idx_1+1,:,:]
            w2 = np.squeeze(w2)
            #print("after second for loop",w2.shape)
            for row in range(w2.shape[0]):
                #print("in 3 for loop",w2.shape, row+1)
                w3 = w2[row:row+1,:]
                #print("loop 3 2",w3.shape)
                #print(w3.flatten().shape)
                f.write(f'{str(w3.flatten())} \n')
            #print("loop 3 end",row+1)
        print(f'Layer {i + 1}: {layer.name} indices of the dim 0:{idx} \n')
        

else:
    f.write(f'Weight {j + 1} shape: {w.shape} \n')
    f.write(f'{str(w.flatten())} \n')