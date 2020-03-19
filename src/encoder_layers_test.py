import torch
import encoder_layers

def test_Conv2dSubsampleV2():
    layer = encoder_layers.Conv2dSubsampleV2(80, 512, 3)
    feats = torch.rand(3, 3000, 80)
    lengths = torch.tensor([100, 2899, 3000]).long()
    outputs, output_lengths = layer(feats, lengths)
    print("outputs.shape", outputs.shape)
    print("input_lengths", lengths)
    print("output_lengths", output_lengths)     

if __name__ == "__main__":
    test_Conv2dSubsampleV2()



