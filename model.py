import torch as t
import torch.nn as nn
"""
4层你卷积操作流程构成
"""
class NetGenerator(nn.Module):
    def __init__(self,opt):
        super(NetGenerator,self).__init__()
        ngf=opt.ngf
        self.main=nn.Sequential(
            #w=(input_w-1)*stride-2*pad+kernel_size
            #h=(input_h-1)*stride-2*pad+kernel_size
            nn.ConvTranspose2d(opt.nz,ngf*8,4,1,0,bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # 以上 输入 opt.nz 输出ngf*8*(4*4)  widh=(1-1)*1-2*0+4=4 heigh=(1-1)*1-2*0+4=4

            nn.ConvTranspose2d(ngf*8,ngf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # 以上 输入 opt.nz 输出ngf*4*(8*8)  widh=(4-1)*2-2*1+4=8 heigh=8

            nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # 以上 输入 opt.nz 输出ngf*2*(16*6)  widh=(8-1)*2-2*1+4=16 heigh=16

            nn.ConvTranspose2d(ngf*2,ngf,4,2,1,bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 以上 输入 opt.nz 输出ngf*(32*32)  widh=(16-1)*2-2*1+4=32 heigh=32

            nn.ConvTranspose2d(ngf,3,5,3,1,bias=True),
            nn.Tanh()
            # 以上 输入 opt.nz 输出3*(96*96)  widh=(32-1)*3-2*1+5=96 heigh=96
        )

    def forward(self, input):
        return self.main(input)

class NetDiscriminator(nn.Module):
    def __init__(self,opt):
        super(NetDiscriminator,self).__init__()
        ndf=opt.ndf

        self.main=nn.Sequential(
            #输出3*96*96
            nn.Conv2d(3,ndf,5,3,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            #输出ndf*32*32    widh=(96-5+2*1)/3+1=32 height=32

            nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,inplace=True),
            # 输出ndf*8*16*16    widh=(32-4+2*1)/2+1=16 height=16

            nn.Conv2d(ndf*2,ndf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,inplace=True),
            # 输出ndf*2*8*8    widh=(16-4+2*1)/2+1=8 height=8

            nn.Conv2d(ndf*4,ndf*8,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,inplace=True),
            # 输出ndf*8*4*4    widh=(8-4+2*1)/2+1=4 height=4

            nn.Conv2d(ndf*8,1,4,1,0,bias=True),
            nn.Sigmoid()#输出一个概率
            # 输出ndf*8*1*1    widh=(4-4+2*0)/2+1=1 height=1
        )

    def forward(self, input):
        return self.main(input).view(-1)


