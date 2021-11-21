import torch.nn as nn
# from .gen_resblock import GenBlock

#comment
class Generator(nn.Module):
    def __init__(self, args, ch = 512, bw=4):
        super(Generator, self).__init__()
        self.ch = ch
        self.bw = bw
        self.lin = nn.Linear(args.latent_dim, 4*4*ch)
        self.bn0 = nn.BatchNorm1d(4*4*ch)
        self.main = nn.Sequential(

            nn.ConvTranspose2d(ch, ch//2, 4, 2, 1),
            nn.BatchNorm2d(ch//2),
            nn.ReLU(True),

            # state size. (ngf*8) x 2x2
            nn.ConvTranspose2d(ch//2, ch//4, 4, 2, 1),
            nn.BatchNorm2d(ch//4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4x4
            nn.ConvTranspose2d(ch//4, ch//8, 4, 2, 1),
            nn.BatchNorm2d(ch//8),
            nn.ReLU(True),
            # state size. (ngf*2) x 8x8
            nn.ConvTranspose2d(ch//8, 3, 3, 1, 1),
            # nn.BatchNorm2d(3),
            nn.Tanh(),
            # state size. (ngf) x 16x16
            # nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            # nn.Tanh()
            # state size. (nc) x 32x32
        )

    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        output = self.bn0(self.lin(input)).view(-1,self.ch, self.bw, self.bw)
        output = self.main(output)
        return output

class Discriminator(nn.Module):
    def __init__(self, args, bw=4, ch=512):

        super(Discriminator, self).__init__()
        self.bw = bw
        self.ch = ch
        self.main = nn.Sequential(
            # input is (nc) x 32x32
            nn.Conv2d(3, ch//8, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16x16
            nn.Conv2d(ch//8, ch//4, 4, 2, 1),
            nn.BatchNorm2d(ch//4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8x8
            nn.Conv2d(ch//4, ch//4, 3, 1, 1),
            nn.BatchNorm2d(ch//4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4x4
            nn.Conv2d(ch//4, ch//2, 4, 2, 1),
            nn.BatchNorm2d(ch//2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 2x2
            nn.Conv2d(ch//2, ch//2, 3, 1, 1),
            nn.BatchNorm2d(ch//2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch//2, ch//1, 4, 2, 1),
            nn.BatchNorm2d(ch//1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch//1, ch//1, 3, 1, 1),
            nn.BatchNorm2d(ch//1),
            nn.LeakyReLU(0.2, inplace=True),

        )
        self.ln = nn.Linear(bw*bw*ch, 1)
    
    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        output = self.main(input)
        return self.ln(output.view(-1,self.bw*self.bw*self.ch))
