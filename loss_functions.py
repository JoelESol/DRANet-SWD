import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_weights(task, dsets, style):
    '''
        You must set these hyperparameters to apply our method to other datasets.
        These hyperparameters may not be the optimal value for your machine.
    '''
    alpha = dict()
    alpha['style'], alpha['dis'], alpha['gen'] = dict(), dict(), dict()
    if task == 'clf':
        if style == 'gram':
            alpha['recon'], alpha['consis'], alpha['content'] = 5, 1, 1

            # MNIST <-> SVHN
            if 'M' in dsets and 'SVHN' in dsets and 'U' not in dsets:
                alpha['style']['M2SVHN'], alpha['style']['SVHN2M'] = 5e4, 1e4
                alpha['dis']['M'], alpha['dis']['SVHN'] = 0.5, 0.5
                alpha['gen']['M'], alpha['gen']['SVHN'] = 0.5, 1.0

            # MNIST <-> MNIST-M
            elif 'M' in dsets and 'MM' in dsets and 'U' not in dsets:
                alpha['style']['M2MM'], alpha['style']['MM2M'] = 5e4, 1e4
                alpha['dis']['M'], alpha['dis']['MM'] = 0.5, 0.5
                alpha['gen']['M'], alpha['gen']['MM'] = 0.5, 1.0

            # MNIST <-> USPS
            elif 'M' in dsets and 'U' in dsets and 'MM' not in dsets:
                alpha['style']['M2U'], alpha['style']['U2M'] = 5e3, 5e3
                alpha['dis']['M'], alpha['dis']['U'] = 0.5, 0.5
                alpha['gen']['M'], alpha['gen']['U'] = 0.5, 0.5

            # MNIST <-> MNIST-M <-> USPS
            elif 'M' in dsets and 'U' in dsets and 'MM' in dsets:
                alpha['style']['M2MM'], alpha['style']['MM2M'], alpha['style']['M2U'], alpha['style']['U2M'] = 5e4, 1e4, 1e4, 1e4
                alpha['dis']['M'], alpha['dis']['MM'], alpha['dis']['U'] = 0.5, 0.5, 0.5
                alpha['gen']['M'], alpha['gen']['MM'], alpha['gen']['U'] = 0.5, 1.0, 0.5

        else: #SWD
            alpha['recon'], alpha['consis'], alpha['content'] = 5, 1, 1

            # MNIST <-> SVHN
            if 'M' in dsets and 'SVHN' in dsets and 'U' not in dsets:
                alpha['style']['M2SVHN'], alpha['style']['SVHN2M'] = 0.55, 0.45
                alpha['dis']['M'], alpha['dis']['SVHN'] = 0.5, 0.5
                alpha['gen']['M'], alpha['gen']['SVHN'] = 0.5, 1.0

            # MNIST <-> MNIST-M
            elif 'M' in dsets and 'MM' in dsets and 'U' not in dsets:
                alpha['style']['M2MM'], alpha['style']['MM2M'] = 1.25, 1 #1, 1
                alpha['dis']['M'], alpha['dis']['MM'] = 0.5, 0.5
                alpha['gen']['M'], alpha['gen']['MM'] = 0.5, 1.0

            # MNIST <-> USPS
            elif 'M' in dsets and 'U' in dsets and 'MM' not in dsets:
                alpha['style']['M2U'], alpha['style']['U2M'] = 0.75, 0.75
                alpha['dis']['M'], alpha['dis']['U'] = 0.5, 0.5
                alpha['gen']['M'], alpha['gen']['U'] = 0.5, 0.5

            # MNIST <-> MNIST-M <-> USPS
            elif 'M' in dsets and 'U' in dsets and 'MM' in dsets:
                alpha['style']['M2MM'], alpha['style']['MM2M'], alpha['style']['M2U'], alpha['style'][
                    'U2M'] = 1, 0.5, 0.5, 0.5
                alpha['dis']['M'], alpha['dis']['MM'], alpha['dis']['U'] = 0.5, 0.5, 0.5
                alpha['gen']['M'], alpha['gen']['MM'], alpha['gen']['U'] = 0.5, 1.0, 0.5

    elif task == 'seg':
        if style == 'gram':
            # GTA5 <-> Cityscapes
            alpha['recon'], alpha['consis'], alpha['content'] = 10, 1, 1
            alpha['style']['G2C'], alpha['style']['C2G'] = 5e3, 5e3 #beginning style loss 10.53
            alpha['dis']['G'], alpha['dis']['C'] = 0.5, 0.5
            alpha['gen']['G'], alpha['gen']['C'] = 0.5, 0.5

        else: #SWD
            # GTA5 <-> Cityscapes
            alpha['recon'], alpha['consis'], alpha['content'] = 10, 1, 1
            alpha['style']['G2C'], alpha['style']['C2G'] = 0.75, 0.75
            alpha['dis']['G'], alpha['dis']['C'] = 0.5, 0.5
            alpha['gen']['G'], alpha['gen']['C'] = 0.5, 0.5

    return alpha


class Loss_Functions:
    def __init__(self, args):
        self.args = args
        self.alpha = loss_weights(args.task, args.datasets, args.style)

    def recon(self, imgs, recon_imgs):
        recon_loss = 0
        for dset in imgs.keys():
            recon_loss += F.l1_loss(imgs[dset], recon_imgs[dset])
        return self.alpha['recon'] * recon_loss
        
    def dis(self, real, fake):
        dis_loss = 0
        if self.args.task == 'clf':  # DCGAN loss
            for dset in real.keys():
                dis_loss += self.alpha['dis'][dset] * F.binary_cross_entropy(real[dset], torch.ones_like(real[dset]))
            for cv in fake.keys():
                source, target = cv.split('2')
                dis_loss += self.alpha['dis'][target] * F.binary_cross_entropy(fake[cv], torch.zeros_like(fake[cv]))
        elif self.args.task == 'seg':  # Hinge loss
            for dset in real.keys():
                dis_loss += self.alpha['dis'][dset] * F.relu(1. - real[dset]).mean()
            for cv in fake.keys():
                source, target = cv.split('2')
                dis_loss += self.alpha['dis'][target] * F.relu(1. + fake[cv]).mean()
        return dis_loss

    def gen(self, fake):
        gen_loss = 0
        for cv in fake.keys():
            source, target = cv.split('2')
            if self.args.task == 'clf':
                gen_loss += self.alpha['gen'][target] * F.binary_cross_entropy(fake[cv], torch.ones_like(fake[cv]))
            elif self.args.task == 'seg':
                gen_loss += -self.alpha['gen'][target] * fake[cv].mean()
        return gen_loss

    def content_perceptual(self, perceptual, perceptual_converted):
        content_perceptual_loss = 0
        for cv in perceptual_converted.keys():
            source, target = cv.split('2')
            content_perceptual_loss += F.mse_loss(perceptual[source][-1], perceptual_converted[cv][-1])
        return self.alpha['content'] * content_perceptual_loss

    def style_perceptual(self, style_gram, style_gram_converted):
        style_percptual_loss = 0
        for cv in style_gram_converted.keys():
            source, target = cv.split('2')
            for gr in range(len(style_gram[target])):
                style_percptual_loss += self.alpha['style'][cv] * F.mse_loss(style_gram[target][gr], style_gram_converted[cv][gr])
        return style_percptual_loss

    def discrepancy_slice_wasserstein_style_loss(self, perceptual, perceptual_converted, num_projections=128):
        style_loss=0
        for cv in perceptual_converted.keys():
            source, target = cv.split('2')
            swd_loss = 0
            for p1, p2 in zip(perceptual[target], perceptual_converted[cv]):
                b, c, h, w = p1.shape

                p1 = p1.view(b, c, h*w).permute(0, 2, 1)
                p2 = p2.view(b, c, h*w).permute(0, 2, 1)

                proj = torch.randn(c, num_projections, device=p1.device)
                proj /= torch.norm(proj, dim=0, keepdim=True)

                p1_proj = torch.matmul(p1, proj)
                p2_proj = torch.matmul(p2, proj)

                p1_sorted, _ = torch.sort(p1_proj, dim=1)
                p2_sorted, _ = torch.sort(p2_proj, dim=1)

                swd = torch.mean((p1_sorted - p2_sorted) ** 2)
                swd_loss += swd

            style_loss += self.alpha['style'][cv] * swd_loss
        return style_loss

    def consistency(self, contents, styles, contents_converted, styles_converted, converts):
        consistency_loss = 0
        for cv in converts:
            source, target = cv.split('2')
            consistency_loss += F.l1_loss(contents[cv], contents_converted[cv])
            consistency_loss += F.l1_loss(styles[target], styles_converted[cv])
        return self.alpha['consis'] * consistency_loss

    def task(self, pred, gt):
        task_loss = 0
        for key in pred.keys():
            if '2' in key:
                source, target = key.split('2')
            else:
                source = key
            task_loss += F.cross_entropy(pred[key], gt[source], ignore_index=-1)
        return task_loss

