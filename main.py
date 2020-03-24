import torch as t
from torchvision import  transforms
import torchvision as tv
import os
import ipdb
import tqdm
import time
from torchnet.meter import AverageValueMeter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import NetGenerator ,NetDiscriminator
class Config(object):
    def __init__(self):
        if not os.path.exists("./imgs/"):
            os.mkdir("./imgs/")
        if not os.path.exists("./checkpoints/"):
            os.mkdir("./checkpoints/")
    data_path='./data'
    num_workers=4
    iamge_szie=96
    bach_size=64
    max_epoch=200
    lr1=2e-4
    lr2=2e-4
    betal=0.5
    use_gpu=False
    nz=100
    ngf=64#生成器维度64
    ndf=64#判别器维度64
    save_path="./imgs"
    env="GAN"
    plot_every=20

    debug_file="./debug/_debug"
    d_every=1
    g_every=5
    decay_every=10
    netd_path="D:/workspace/code/pytorch/GAN/checkpoints/netd_50.pth"
    netg_path="D:/workspace/code/pytorch/GAN/checkpoints/netg_50.pth"
    #测试用的参数
    gen_img="result.jpg"
    gen_num=64
    gen_search_num=512
    gen_mean=0
    gen_std=1
opt=Config()

def train(**kwargs):
    #处理参数
    for _k,_v in kwargs.items():
        if getattr(opt,_k):
            setattr(opt,_k,_v)
        else :
            print("%s not exists!" %str(_k) )
    # 数据处理
    transform = transforms.Compose([
        tv.transforms.Scale(opt.iamge_szie),
        tv.transforms.CenterCrop(opt.iamge_szie),
        tv.transforms.ToTensor(),  # shape=(C x H x W)的像素值范围为[0.0, 1.0]的torch.FloatTensor。
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 将tensor值归一化到-1，1的范围
    ])

    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=opt.bach_size, shuffle=True, num_workers=opt.num_workers,
                            drop_last=True)
    # 实例化网络
    netg, netd = NetGenerator(opt), NetDiscriminator(opt)
    map_location = lambda storage, loc: storage
    if opt.netd_path:
        netd.load_state_dict(t.load(opt.netd_path,map_location=map_location))#使用 load_state_dict() 加载模型参数时, 要求保存的模型参数键值类型和模型完全一致, 一旦我们对模型结构做了些许修改, 就不能直接调用该函数.
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path,map_location=map_location))

    #定义优化器和宋史函数
    optimizerG=t.optim.Adam(netg.parameters(),opt.lr1,betas=(opt.betal,0.999))
    optimizerD=t.optim.Adam(netd.parameters(),opt.lr2,betas=(opt.betal,0.999))

    criterion=t.nn.BCELoss()

    true_labels=Variable(t.ones(opt.bach_size))
    fake_labels=Variable(t.zeros(opt.bach_size))
    fix_noises=Variable(t.randn(opt.bach_size,opt.nz,1,1))

    noises=Variable(t.randn(opt.bach_size,opt.nz,1,1))

    #errord_meter = AverageValueMeter()
    #errorg_meter = AverageValueMeter()

    if opt.use_gpu:
        netd.cuda()
        netg.cuda()
        criterion.cuda()
        true_labels,fake_labels=true_labels.cuda(),fake_labels.cuda()
        fix_noises,noises=fix_noises.cuda(),noises.cuda()
    #迭代训练
    epochs = range(opt.max_epoch)
    for epoch in iter(epochs):
        epoch_start=time.time()
        for ii, (img, _) in tqdm.tqdm(enumerate(dataloader)):
            batch_start=time.time()#一个batch的开始时间
            real_img = Variable(img)
            if opt.use_gpu:
                real_img = real_img.cuda()
            # 训练判别器
            if (ii + 1) % opt.d_every == 0:
                optimizerD.zero_grad()
                # 真图尽可能判别为1
                output = netd(real_img)
                error_d_real = criterion(output, true_labels)
                error_d_real.backward()

                # 假图尽可能判别为0
                noises.data.copy_(t.randn(opt.bach_size, opt.nz, 1, 1))
                fake_img = netg(noises).detach()

                fake_output = netd(fake_img)
                error_d_fake = criterion(fake_output, fake_labels)
                error_d_fake.backward()
                optimizerD.step()

                error_d = error_d_fake + error_d_real
               # errord_meter.add(error_d.item())

            # 训练生成器
            if (ii + 1) % opt.g_every == 0:
                optimizerG.zero_grad()
                noises.data.copy_(noises).detach()  # 更具噪声生成假图像
                fake_img = netg(noises)
                fake_output = netd(fake_img)
                # 尽可能是判别器把假图判断为1
                error_g = criterion(fake_output, true_labels)
                error_g.backward()
                optimizerG.step()

               # errorg_meter.add(error_g.item())

            if ii % opt.plot_every == opt.plot_every - 1:
                ## 可视化
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                fix_fake_imgs = netg(fix_noises)
            batch_end=time.time()
            print("\nTime spent on a batch is :{:.2f}s ".format(batch_end-batch_start));
            print("\n")
        if epoch % opt.decay_every == 0:
            # 保存模型、图片
            tv.utils.save_image(fix_fake_imgs.data[:64], '%s/%s.png' % (opt.save_path, epoch), normalize=True,
                                range=(-1, 1))
            t.save(netd.state_dict(), 'checkpoints/netd_%s.pth' % epoch)
            t.save(netg.state_dict(), 'checkpoints/netg_%s.pth' % epoch)
            print("\nsave result to %s.pth" % epoch, "successed!\n")
            # errord_meter.reset()
            # errorg_meter.reset()
            optimizerG = t.optim.Adam(netg.parameters(), opt.lr1, betas=(opt.betal, 0.999))
            optimizerD = t.optim.Adam(netd.parameters(), opt.lr2, betas=(opt.betal, 0.999))

        epoch_end = time.time()
    print("\nEpoch finished"+"Time spent on a epoch is :{:.2f}s".format(epoch_end - epoch_start));
    print("\n")



def generate(**kwargs):
    '''
    随机生成动漫头像，并根据netd的分数选择较好的
    '''
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    netg, netd = NetGenerator(opt).eval() ,NetDiscriminator(opt).eval()
    noises = t.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std)
    noises = Variable(noises, volatile=True)

    map_location = lambda storage, loc: storage
    netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))

    if opt.use_gpu:
        netd.cuda()
        netg.cuda()
        noises = noises.cuda()

    # 生成图片，并计算图片在判别器的分数
    fake_img = netg(noises)
    scores = netd(fake_img).data

    # 挑选最好的某几张
    indexs = scores.topk(opt.gen_num)[1]
    result = []
    for ii in indexs:
        result.append(fake_img.data[ii])
    # 保存图片
    tv.utils.save_image(t.stack(result), opt.gen_img, normalize=True, range=(-1, 1))


if __name__ == '__main__':
    #import fire
    #fire.Fire()
    #train()
    generate()








