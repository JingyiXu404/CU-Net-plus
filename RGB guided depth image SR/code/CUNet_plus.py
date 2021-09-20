import torch
import torch.nn as nn
import torch.nn.functional as F

class Prediction(nn.Module):
    def __init__(self, num_channels):
        super(Prediction, self).__init__()
        self.num_layers = 4
        self.in_channel = num_channels
        self.kernel_size = 9
        self.num_filters = 64
        self.padding=4

        self.layer_in = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_filters,
                                 kernel_size=self.kernel_size, padding=self.padding, stride=1, bias=False)
        nn.init.xavier_uniform_(self.layer_in.weight.data)
        self.lam_in = nn.Parameter(torch.Tensor([0.01]))
        self.lam_i = []
        self.layer_down = []
        self.layer_up = []
        for i in range(self.num_layers):
            down_conv = 'down_conv_{}'.format(i)
            up_conv = 'up_conv_{}'.format(i)
            lam_id = 'lam_{}'.format(i)
            layer_2 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.in_channel,
                                kernel_size=self.kernel_size, padding=self.padding, stride=1, bias=False)
            nn.init.xavier_uniform_(layer_2.weight.data)
            setattr(self, down_conv, layer_2)
            self.layer_down.append(getattr(self, down_conv))
            layer_3 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_filters,
                                kernel_size=self.kernel_size, padding=self.padding, stride=1, bias=False)
            nn.init.xavier_uniform_(layer_3.weight.data)
            setattr(self, up_conv, layer_3)
            self.layer_up.append(getattr(self, up_conv))

            lam_ = nn.Parameter(torch.Tensor([0.01]))
            setattr(self, lam_id, lam_)
            self.lam_i.append(getattr(self, lam_id))

    def forward(self, mod):
        p1 = self.layer_in(mod)
        tensor = torch.mul(torch.sign(p1), F.relu(torch.abs(p1) - self.lam_in))

        for i in range(self.num_layers):
            p3 = self.layer_down[i](tensor)
            p4 = self.layer_up[i](p3)
            p5 = tensor - p4
            p6 = torch.add(p1, p5)
            tensor = torch.mul(torch.sign(p6), F.relu(torch.abs(p6) - self.lam_i[i]))
        return tensor


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.channel = 3
        self.kernel_size = 9
        self.filters = 64
        self.padding=4
        self.conv_1 = nn.Conv2d(in_channels=self.filters, out_channels=self.channel, kernel_size=self.kernel_size,stride=1, padding=self.padding, bias=False)
        nn.init.xavier_uniform_(self.conv_1.weight.data)
        self.conv_2 = nn.Conv2d(in_channels=self.filters, out_channels=self.channel, kernel_size=self.kernel_size,stride=1, padding=self.padding, bias=False)
        nn.init.xavier_uniform_(self.conv_2.weight.data)

    def forward(self, u, z):
        rec_u = self.conv_1(u)
        rec_z = self.conv_2(z)
        z_rec = rec_u + rec_z
        return z_rec


class CUNet_plus(nn.Module):
    def __init__(self):
        super(CUNet_plus, self).__init__()
        self.channel = 3
        self.num_filters = 64
        self.kernel_size = 9
        self.padding=4
        self.net_u1 = Prediction(num_channels=self.channel)
        self.net_u2 = Prediction(num_channels=self.channel)
        self.conv_u1 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.channel, kernel_size=self.kernel_size,stride=1, padding=self.padding, bias=False)
        nn.init.xavier_uniform_(self.conv_u1.weight.data)
        self.conv_u2 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.channel, kernel_size=self.kernel_size,stride=1,padding=self.padding,bias=False)
        nn.init.xavier_uniform_(self.conv_u2.weight.data)
        self.net_v1 = Prediction(num_channels=self.channel)
        self.net_v2 = Prediction(num_channels=self.channel)
        self.conv_v1 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.channel, kernel_size=self.kernel_size,stride=1, padding=self.padding, bias=False)
        nn.init.xavier_uniform_(self.conv_v1.weight.data)
        self.conv_v2 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.channel, kernel_size=self.kernel_size,stride=1, padding=self.padding, bias=False)
        nn.init.xavier_uniform_(self.conv_v2.weight.data)
        self.net_z1 = Prediction(num_channels=2 * self.channel)
        self.net_z2 = Prediction(num_channels=2 * self.channel)
        self.conv_x = nn.Conv2d(in_channels=self.num_filters, out_channels=self.channel, kernel_size=self.kernel_size,stride=1, padding=self.padding, bias=False)
        nn.init.xavier_uniform_(self.conv_x.weight.data)
        self.conv_y = nn.Conv2d(in_channels=self.num_filters, out_channels=self.channel, kernel_size=self.kernel_size,stride=1, padding=self.padding, bias=False)
        nn.init.xavier_uniform_(self.conv_y.weight.data)
        self.decoder = decoder()

    def forward(self, x, y):
        u = self.net_u1(x)
        v = self.net_v1(y)

        p_x = x - self.conv_u1(u)
        p_y = y - self.conv_v1(v)
        p_xy = torch.cat((p_x, p_y), dim=1)

        z1 = self.net_z1(p_xy)
        x2 = self.conv_x(z1)
        y2 = self.conv_y(z1)
        hat_x = x - x2
        hat_y = y - y2

        u2 = self.net_u2(hat_x)
        v2 = self.net_v2(hat_y)

        p_x2 = x - self.conv_u2(u2)
        p_y2 = y - self.conv_v2(v2)
        p_xy2 = torch.cat((p_x2, p_y2), dim=1)

        z = self.net_z2(p_xy2)
        f_pred = self.decoder(u2, z)
        return f_pred
if __name__ =='__main__':
    cu = CUNet_plus()
    print(cu)
    a=torch.rand([1,1,64,64])
    b=torch.rand([1,1,64,64])
    net=CUNet_plus()
    a_out=net(a,b)
    print(a.shape)
    print(b.shape)
    print(a_out.shape)