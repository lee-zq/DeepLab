from lib.nn.OctConv import *

if __name__ == '__main__':
    # nn.Conv2d
    high = torch.Tensor(1, 64, 32, 32).cuda()
    low = torch.Tensor(1, 192, 16, 16).cuda()
    # test Oc conv
    OCconv = OctaveConv(kernel_size=(3,3),in_channels=256,out_channels=512,bias=False,stride=2,alpha=0.75).cuda()
    i = high,low
    x_out,y_out = OCconv(i)
    print(x_out.size())
    print(y_out.size())

    i = torch.Tensor(1, 3, 512, 512).cuda()
    FOCconv = FirstOctaveConv(kernel_size=(3, 3), in_channels=3, out_channels=128,alpha=0.75).cuda()
    x_out, y_out = FOCconv(i)
    print("First: ", x_out.size(), y_out.size())
    # test last Octave Cov
    LOCconv = LastOctaveConv(kernel_size=(3, 3), in_channels=256, out_channels=128, alpha=0.75).cuda()
    i = high, low
    out = LOCconv(i)
    print("Last: ", out.size())

    # test OCB
    ocb = OctaveCB(in_channels=256, out_channels=128, alpha=0.75).cuda()
    i = high, low
    x_out_h, y_out_l = ocb(i)
    print("OCB:",x_out_h.size(),y_out_l.size())

    # test last OCB
    ocb_last = LastOCtaveCBR(256, 128, alpha=0.75).cuda()
    i = high, low
    x_out_h = ocb_last(i)
    print("Last OCB", x_out_h.size())