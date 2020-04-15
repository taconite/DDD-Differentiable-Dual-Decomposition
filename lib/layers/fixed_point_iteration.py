import torch
import math

import torch.nn.functional as F

import dp_cuda
import dp_ne_cuda

class FixedPointIteration(torch.nn.Module):
    def __init__(self, state_size):
        super(FixedPointIteration, self).__init__()
        self.state_size = state_size
        self.gamma = torch.nn.Parameter(torch.ones(1))
        # This layer is parameter-free
        # self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1.0 / math.sqrt(self.state_size)
    #     for weight in self.parameters():
    #         weight.data.uniform_(-stdv, +stdv)

    def forward(self, *inputs):

        horizontal_unary = inputs[0]
        vertical_unary = inputs[1]
        horizontal_pairwise = inputs[2]
        vertical_pairwise = inputs[3]

        width = horizontal_unary.size(-1)
        height = horizontal_unary.size(-2)

        # horizontal_left_cache = torch.empty(horizontal_unary.shape, device=horizontal_unary.device)
        # horizontal_right_cache = torch.empty(horizontal_unary.shape, device=horizontal_unary.device)
        # vertical_top_cache = torch.empty(vertical_unary.shape, device=vertical_unary.device)
        # vertical_bottom_cache = torch.empty(vertical_unary.shape, device=vertical_unary.device)

        # # Initialize max-marginals for head and tail positions
        # horizontal_left_cache[:, :, :, 0] = horizontal_unary[:, :, :, 0]
        # horizontal_right_cache[:, :, :, -1] = horizontal_unary[:, :, :, -1]

        # vertical_top_cache[:, :, 0, :] = vertical_unary[:, :, 0, :]
        # vertical_bottom_cache[:, :, -1, :] = vertical_unary[:, :, -1, :]

        horizontal_left_cache = [horizontal_unary[:, :, :, 0]]
        horizontal_right_cache = [horizontal_unary[:, :, :, -1]]

        vertical_top_cache = [vertical_unary[:, :, 0, :]]
        vertical_bottom_cache = [vertical_unary[:, :, -1, :]]

        # argmax_h_left = []
        # argmax_h_right = []

        # Compute max-marginals along horizontal and vertical chains
        for i in range(width - 1):
            # max_margins, argmaxes = torch.max(
            #     horizontal_pairwise[:, :, :, :, i] + horizontal_left_cache[-1].unsqueeze(2),
            #     dim=1,
            #     keepdim=False
            # )
            # argmax_h_left.append(argmaxes)
            # horizontal_left_cache.append(horizontal_unary[:, :, :, i+1] + max_margins)
            # max_margins, argmaxes = torch.max(
            #     horizontal_pairwise[:, :, :, :, width-i-2] + horizontal_right_cache[-1].unsqueeze(1),
            #     dim=2,
            #     keepdim=False
            # )
            # argmax_h_right.append(argmaxes)
            # horizontal_right_cache.append(horizontal_unary[:, :, :, width-i-2] + max_margins)
            max_margins = self.gamma * torch.logsumexp(
                (horizontal_pairwise[:, :, :, :, i] + horizontal_left_cache[-1].unsqueeze(2)) / self.gamma,
                dim=1,
                keepdim=False
            )
            horizontal_left_cache.append(horizontal_unary[:, :, :, i+1] + max_margins)
            max_margins = self.gamma * torch.logsumexp(
                (horizontal_pairwise[:, :, :, :, width-i-2] + horizontal_right_cache[-1].unsqueeze(1)) / self.gamma,
                dim=2,
                keepdim=False
            )
            horizontal_right_cache.append(horizontal_unary[:, :, :, width-i-2] + max_margins)

        # argmax_h_left = torch.stack(argmax_h_left, dim=-1)
        # argmax_h_right = torch.stack(argmax_h_right[::-1], dim=-1)

        horizontal_left_cache = torch.stack(horizontal_left_cache, dim=-1)
        horizontal_right_cache = torch.stack(horizontal_right_cache[::-1], dim=-1)

        horizontal_marginals = horizontal_left_cache + horizontal_right_cache - horizontal_unary

        # argmax_v_top = []
        # argmax_v_bottom = []

        for i in range(height - 1):
            # max_margins, argmaxes = torch.max(
            #     vertical_pairwise[:, :, :, i, :] + vertical_top_cache[-1].unsqueeze(2),
            #     dim=1,
            #     keepdim=False
            # )
            # argmax_v_top.append(argmaxes)
            # vertical_top_cache.append(vertical_unary[:, :, i+1, :] + max_margins)

            # max_margins, argmaxes = torch.max(
            #     vertical_pairwise[:, :, :, height-i-2, :] + vertical_bottom_cache[-1].unsqueeze(1),
            #     dim=2,
            #     keepdim=False
            # )
            # argmax_v_bottom.append(argmaxes)
            # vertical_bottom_cache.append(vertical_unary[:, :, height-i-2, :] + max_margins)
            max_margins = self.gamma * torch.logsumexp(
                (vertical_pairwise[:, :, :, i, :] + vertical_top_cache[-1].unsqueeze(2)) / self.gamma,
                dim=1,
                keepdim=False
            )
            vertical_top_cache.append(vertical_unary[:, :, i+1, :] + max_margins)

            max_margins = self.gamma * torch.logsumexp(
                (vertical_pairwise[:, :, :, height-i-2, :] + vertical_bottom_cache[-1].unsqueeze(1)) / self.gamma,
                dim=2,
                keepdim=False
            )
            vertical_bottom_cache.append(vertical_unary[:, :, height-i-2, :] + max_margins)

        # argmax_v_top = torch.stack(argmax_v_top, dim=-2)
        # argmax_v_bottom = torch.stack(argmax_v_bottom[::-1], dim=-2)

        vertical_top_cache = torch.stack(vertical_top_cache, dim=-2)
        vertical_bottom_cache = torch.stack(vertical_bottom_cache[::-1], dim=-2)

        vertical_marginals = vertical_top_cache + vertical_bottom_cache - vertical_unary

        # Update and return new unary terms
        average_marginals = (horizontal_marginals + vertical_marginals) / 2.0

        horizontal_unary -= 1.0 / width * (horizontal_marginals - average_marginals)
        vertical_unary -= 1.0 / height * (vertical_marginals - average_marginals)

        return horizontal_unary, vertical_unary, horizontal_marginals, vertical_marginals


class DPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, unary_h, unary_v, pairwise_h, pairwise_v, gamma):
        outputs = dp_cuda.forward(unary_h, unary_v, pairwise_h, pairwise_v, gamma)

        marginals_h, marginals_v = outputs[:2]
        variables = outputs[2:]
        ctx.save_for_backward(*variables)

        return marginals_h, marginals_v

    @staticmethod
    def backward(ctx, grad_marginals_h, grad_marginals_v):
        outputs = dp_cuda.backward(
            grad_marginals_h.contiguous(), grad_marginals_v.contiguous(), *ctx.saved_variables)
        d_unary_h, d_unary_v, d_pairwise_h, d_pairwise_v = outputs
        return d_unary_h, d_unary_v, d_pairwise_h[:, :, :, :, :-1], d_pairwise_v[:, :, :, :-1, :], None


class DDPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, unary_h, unary_v, pairwise_h, pairwise_v, gamma):
        outputs = dp_ne_cuda.forward(unary_h, unary_v, pairwise_h, pairwise_v, gamma)

        marginals_h, marginals_v = outputs[:2]
        variables = outputs[2:]
        ctx.save_for_backward(*variables)

        return marginals_h, marginals_v

    @staticmethod
    def backward(ctx, grad_marginals_h, grad_marginals_v):
        outputs = dp_ne_cuda.backward(
            grad_marginals_h.contiguous(), grad_marginals_v.contiguous(), *ctx.saved_variables)
        d_unary_h, d_unary_v, d_pairwise_h, d_pairwise_v = outputs
        return d_unary_h, d_unary_v, d_pairwise_h[:, :, :, :, :-1], d_pairwise_v[:, :, :, :-1, :], None


class FixedPointIterationFast(torch.nn.Module):
    def __init__(self, state_size, learn_gamma=True):
        super(FixedPointIterationFast, self).__init__()
        self.state_size = state_size
        if learn_gamma:
            self.gamma = torch.nn.Parameter(torch.ones(1))
        else:
            self.gamma = None
        # This layer is parameter-free
        # self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1.0 / math.sqrt(self.state_size)
    #     for weight in self.parameters():
    #         weight.data.uniform_(-stdv, +stdv)

    def forward(self, *inputs):

        unary_h = inputs[0]
        unary_v = inputs[1]
        pairwise_h = inputs[2]
        pairwise_v = inputs[3]
        if self.gamma is None:
            gamma = inputs[4]
        elif self.gamma < 1e-2:
            # truncate gamma from below at 1e-2
            self.gamma.data = 1e-2 * torch.ones(1, device=self.gamma.data.device)
        # step_size = inputs[5]

        width = unary_h.size(-1)
        height = unary_v.size(-2)

        marginals_h, marginals_v = DDPFunction.apply(unary_h, unary_v, pairwise_h, pairwise_v, gamma if self.gamma is None else self.gamma)

        # Update and return new unary terms
        average_marginals = (marginals_h + marginals_v) / 2.0

        # if step_size is None:
        unary_h = unary_h - 1.0 / width * (marginals_h - average_marginals)
        unary_v = unary_v - 1.0 / height * (marginals_v - average_marginals)
        # else:
        #     unary_h -= step_size[0] * (marginals_h - average_marginals)
        #     unary_v -= step_size[1] * (marginals_v - average_marginals)

        return unary_h, unary_v, marginals_h, marginals_v


class FixedPointIterationGap2(torch.nn.Module):
    def __init__(self, state_size, learn_gamma=True):
        super(FixedPointIterationGap2, self).__init__()
        self.state_size = state_size
        if learn_gamma:
            self.gamma = torch.nn.Parameter(torch.ones(1))
        else:
            self.gamma = None

    def forward(self, *inputs):

        # unary_h = inputs[0]
        # unary_v = inputs[1]
        # pairwise_h = inputs[2]
        # pairwise_v = inputs[3]
        # if self.gamma is None:
        #     gamma = inputs[4]
        # elif self.gamma < 1e-2:
        #     # truncate gamma from below at 1e-2
        #     self.gamma.data = 1e-2 * torch.ones(1, device=self.gamma.data.device)

        # width = unary_h[0].size(-1)
        # height = unary_v[0].size(-2)

        # marginals_h1, marginals_v1 = DDPFunction.apply(unary_h[0], unary_v[0], pairwise_h[0], pairwise_v[0], gamma if self.gamma is None else self.gamma)
        # b, c, h, w = marginals_h1.shape

        # marginals_h2, marginals_v2 = DDPFunction.apply(unary_h[1], unary_v[1], pairwise_h[1], pairwise_v[1], gamma if self.gamma is None else self.gamma)
        # unpooling_indices = torch.arange(0, width * height, 2, dtype=torch.int64, device=unary_h[1].device, requires_grad=False). \
        #     view(-1, width // 2)[::2, :].view(1, 1, height // 2, width // 2).repeat(b, c, 1, 1)
        # marginals_h2_ = F.max_unpool2d(marginals_h2, unpooling_indices, 2, 2)
        # marginals_v2_ = F.max_unpool2d(marginals_v2, unpooling_indices, 2, 2)

        # marginals_h3, marginals_v3 = DDPFunction.apply(unary_h[2], unary_v[2], pairwise_h[2], pairwise_v[2], gamma if self.gamma is None else self.gamma)
        # unpooling_indices = torch.arange(1, width * height, 2, dtype=torch.int64, device=unary_h[2].device, requires_grad=False). \
        #     view(-1, width // 2)[::2, :].view(1, 1, height // 2, width // 2).repeat(b, c, 1, 1)
        # marginals_h3_ = F.max_unpool2d(marginals_h3, unpooling_indices, 2, 2)
        # marginals_v3_ = F.max_unpool2d(marginals_v3, unpooling_indices, 2, 2)

        # marginals_h4, marginals_v4 = DDPFunction.apply(unary_h[3], unary_v[3], pairwise_h[3], pairwise_v[3], gamma if self.gamma is None else self.gamma)
        # unpooling_indices = torch.arange(width, width * height, 2, dtype=torch.int64, device=unary_h[3].device, requires_grad=False). \
        #     view(-1, width // 2)[::2, :].view(1, 1, height // 2, width // 2).repeat(b, c, 1, 1)
        # marginals_h4_ = F.max_unpool2d(marginals_h4, unpooling_indices, 2, 2)
        # marginals_v4_ = F.max_unpool2d(marginals_v4, unpooling_indices, 2, 2)

        # marginals_h5, marginals_v5 = DDPFunction.apply(unary_h[4], unary_v[4], pairwise_h[4], pairwise_v[4], gamma if self.gamma is None else self.gamma)
        # unpooling_indices = torch.arange(width+1, width * height, 2, dtype=torch.int64, device=unary_h[4].device, requires_grad=False). \
        #     view(-1, width // 2)[::2, :].view(1, 1, height // 2, width // 2).repeat(b, c, 1, 1)
        # marginals_h5_ = F.max_unpool2d(marginals_h5, unpooling_indices, 2, 2)
        # marginals_v5_ = F.max_unpool2d(marginals_v5, unpooling_indices, 2, 2)

        # marginals_h = marginals_h1 + marginals_h2_ + marginals_h3_ + marginals_h4_ + marginals_h5_
        # marginals_v = marginals_v1 + marginals_v2_ + marginals_v3_ + marginals_v4_ + marginals_v5_

        # # Update and return new unary terms
        # average_marginals = (marginals_h + marginals_v) / 4.0

        # # if step_size is None:
        # unary_h[0] -= 1.0 / width * (marginals_h1 - average_marginals)
        # unary_v[0] -= 1.0 / height * (marginals_v1 - average_marginals)
        # unary_h[1] -= 1.0 / width * (marginals_h2 - average_marginals[:, :, ::2, ::2])
        # unary_v[1] -= 1.0 / height * (marginals_v2 - average_marginals[:, :, ::2, ::2])
        # unary_h[2] -= 1.0 / width * (marginals_h3 - average_marginals[:, :, ::2, 1::2])
        # unary_v[2] -= 1.0 / height * (marginals_v3 - average_marginals[:, :, ::2, 1::2])
        # unary_h[3] -= 1.0 / width * (marginals_h4 - average_marginals[:, :, 1::2, ::2])
        # unary_v[3] -= 1.0 / height * (marginals_v4 - average_marginals[:, :, 1::2, ::2])
        # unary_h[4] -= 1.0 / width * (marginals_h5 - average_marginals[:, :, 1::2, 1::2])
        # unary_v[4] -= 1.0 / height * (marginals_v5 - average_marginals[:, :, 1::2, 1::2])

        # return unary_h, unary_v, marginals_h, marginals_v

        unary_h = inputs[0]
        unary_v = inputs[1]
        pairwise_h = inputs[2]
        pairwise_v = inputs[3]
        if self.gamma is None:
            gamma = inputs[4]
        elif self.gamma < 1e-2:
            # truncate gamma from below at 1e-2
            self.gamma.data = 1e-2 * torch.ones(1, device=self.gamma.data.device)

        dp_function = DDPFunction

        width = unary_h[0].size(-1)
        height = unary_v[0].size(-2)

        marginals_hs = []
        marginals_vs = []

        marginals_h1, marginals_v1 = dp_function.apply(
                                         unary_h[0], unary_v[0],
                                         pairwise_h[0], pairwise_v[0],
                                         gamma if self.gamma is None else self.gamma)
        b, c, h, w = marginals_h1.shape

        marginals_h = marginals_h1
        marginals_v = marginals_v1

        marginals_hs.append(marginals_h1)
        marginals_vs.append(marginals_v1)

        cnt = 1

        for i in range(2):
            for j in range(2):
                marginals_h2, marginals_v2 = dp_function.apply(
                                                 unary_h[cnt],
                                                 unary_v[cnt],
                                                 pairwise_h[cnt],
                                                 pairwise_v[cnt],
                                                 gamma if self.gamma is None else self.gamma)

                marginals_hs.append(marginals_h2)
                marginals_vs.append(marginals_v2)

                _, _, height_, width_ = marginals_h2.shape

                upsample_width = width_ * 2
                upsample_height = height_ * 2

                unpooling_indices = torch.arange(
                                        i * upsample_width + j, upsample_width * upsample_height,
                                        2, dtype=torch.int64,
                                        device=unary_h[cnt].device, requires_grad=False). \
                                        view(-1, upsample_width // 2)[::2, :].view(1, 1, upsample_height // 2, upsample_width // 2).repeat(b, c, 1, 1)

                if upsample_width >= width and upsample_height >= height:
                    marginals_h = marginals_h + F.max_unpool2d(marginals_h2, unpooling_indices, 2, 2)[:, :, :height, :width]
                    marginals_v = marginals_v + F.max_unpool2d(marginals_v2, unpooling_indices, 2, 2)[:, :, :height, :width]
                elif upsample_width < width and upsample_height >= height:
                    marginals_h = marginals_h + F.pad(F.max_unpool2d(marginals_h2, unpooling_indices, 2, 2)[:, :, :height, :], (0, 1))
                    marginals_v = marginals_v + F.pad(F.max_unpool2d(marginals_v2, unpooling_indices, 2, 2)[:, :, :height, :], (0, 1))
                elif upsample_width >= width and upsample_height < height:
                    marginals_h = marginals_h + F.pad(F.max_unpool2d(marginals_h2, unpooling_indices, 2, 2)[:, :, :, :width], (0, 0, 0, 1))
                    marginals_v = marginals_v + F.pad(F.max_unpool2d(marginals_v2, unpooling_indices, 2, 2)[:, :, :, :width], (0, 0, 0, 1))
                else:
                    marginals_h = marginals_h + F.pad(F.max_unpool2d(marginals_h2, unpooling_indices, 2, 2), (0, 1, 0, 1))
                    marginals_v = marginals_v + F.pad(F.max_unpool2d(marginals_v2, unpooling_indices, 2, 2), (0, 1, 0, 1))

                cnt += 1

        # Update and return new unary terms
        average_marginals = (marginals_h + marginals_v) / 4.0

        unary_h[0] = unary_h[0] - 1.0 / width * (marginals_hs[0] - average_marginals)
        unary_v[0] = unary_v[0] - 1.0 / height * (marginals_vs[0] - average_marginals)

        cnt = 1

        for i in range(2):
            for j in range(2):
                # width = unary_h[cnt].size(-1)
                # height = unary_v[cnt].size(-2)

                unary_h[cnt] = unary_h[cnt] - 1.0 / width * (marginals_hs[cnt] - average_marginals[:, :, i::2, j::2])
                unary_v[cnt] = unary_v[cnt] - 1.0 / height * (marginals_vs[cnt] - average_marginals[:, :, i::2, j::2])
                cnt += 1

        return unary_h, unary_v, marginals_h, marginals_v


class FixedPointIterationGap4(torch.nn.Module):
    def __init__(self, state_size, learn_gamma=True):
        super(FixedPointIterationGap4, self).__init__()
        self.state_size = state_size
        if learn_gamma:
            self.gamma = torch.nn.Parameter(torch.ones(1))
        else:
            self.gamma = None
        # This layer is parameter-free
        # self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1.0 / math.sqrt(self.state_size)
    #     for weight in self.parameters():
    #         weight.data.uniform_(-stdv, +stdv)

    def forward(self, *inputs):

        unary_h = inputs[0]
        unary_v = inputs[1]
        pairwise_h = inputs[2]
        pairwise_v = inputs[3]
        if self.gamma is None:
            gamma = inputs[4]
        else:
            self.gamma.data.clamp_(min=1e-2)

        width = unary_h[0].size(-1)
        height = unary_v[0].size(-2)

        marginals_hs = []
        marginals_vs = []

        marginals_h1, marginals_v1 = DDPFunction.apply(
                                         unary_h[0], unary_v[0],
                                         pairwise_h[0], pairwise_v[0],
                                         gamma if self.gamma is None else self.gamma)
        b, c, h, w = marginals_h1.shape

        marginals_h = marginals_h1
        marginals_v = marginals_v1

        marginals_hs.append(marginals_h1)
        marginals_vs.append(marginals_v1)

        cnt = 1

        for i in range(2):
            for j in range(2):
                marginals_h2, marginals_v2 = DDPFunction.apply(
                                                 unary_h[cnt],
                                                 unary_v[cnt],
                                                 pairwise_h[cnt],
                                                 pairwise_v[cnt],
                                                 gamma if self.gamma is None else self.gamma)

                marginals_hs.append(marginals_h2)
                marginals_vs.append(marginals_v2)

                _, _, height_, width_ = marginals_h2.shape

                upsample_width = width_ * 2
                upsample_height = height_ * 2

                unpooling_indices = torch.arange(
                                        i * upsample_width + j, upsample_width * upsample_height,
                                        2, dtype=torch.int64,
                                        device=unary_h[cnt].device, requires_grad=False). \
                                        view(-1, upsample_width // 2)[::2, :].view(1, 1, upsample_height // 2, upsample_width // 2).repeat(b, c, 1, 1)

                if upsample_width >= width and upsample_height >= height:
                    marginals_h += F.max_unpool2d(marginals_h2, unpooling_indices, 2, 2)[:, :, :height, :width]
                    marginals_v += F.max_unpool2d(marginals_v2, unpooling_indices, 2, 2)[:, :, :height, :width]
                elif upsample_width < width and upsample_height >= height:
                    marginals_h += F.pad(F.max_unpool2d(marginals_h2, unpooling_indices, 2, 2)[:, :, :height, :], (0, 1))
                    marginals_v += F.pad(F.max_unpool2d(marginals_v2, unpooling_indices, 2, 2)[:, :, :height, :], (0, 1))
                elif upsample_width >= width and upsample_height < height:
                    marginals_h += F.pad(F.max_unpool2d(marginals_h2, unpooling_indices, 2, 2)[:, :, :, :width], (0, 0, 0, 1))
                    marginals_v += F.pad(F.max_unpool2d(marginals_v2, unpooling_indices, 2, 2)[:, :, :, :width], (0, 0, 0, 1))
                else:
                    marginals_h += F.pad(F.max_unpool2d(marginals_h2, unpooling_indices, 2, 2), (0, 1, 0, 1))
                    marginals_v += F.pad(F.max_unpool2d(marginals_v2, unpooling_indices, 2, 2), (0, 1, 0, 1))

                cnt += 1

        for i in range(3):
            for j in range(3):
                marginals_h3, marginals_v3 = DDPFunction.apply(
                                                 unary_h[cnt],
                                                 unary_v[cnt],
                                                 pairwise_h[cnt],
                                                 pairwise_v[cnt],
                                                 gamma if self.gamma is None else self.gamma)

                marginals_hs.append(marginals_h3)
                marginals_vs.append(marginals_v3)

                _, _, height_, width_ = marginals_h3.shape

                upsample_width = width_ * 3
                upsample_height = height_ * 3

                unpooling_indices = torch.arange(
                                        i * upsample_width + j, upsample_width * upsample_height,
                                        3, dtype=torch.int64,
                                        device=unary_h[cnt].device, requires_grad=False). \
                                        view(-1, upsample_width // 3)[::3, :].view(1, 1, upsample_height // 3, upsample_width // 3).repeat(b, c, 1, 1)

                if upsample_width >= width and upsample_height >= height:
                    marginals_h += F.max_unpool2d(marginals_h3, unpooling_indices, 3, 3)[:, :, :height, :width]
                    marginals_v += F.max_unpool2d(marginals_v3, unpooling_indices, 3, 3)[:, :, :height, :width]
                elif upsample_width < width and upsample_height >= height:
                    marginals_h += F.pad(F.max_unpool2d(marginals_h3, unpooling_indices, 3, 3)[:, :, :height, :], (0, width - upsample_width))
                    marginals_v += F.pad(F.max_unpool2d(marginals_v3, unpooling_indices, 3, 3)[:, :, :height, :], (0, width - upsample_width))
                elif upsample_width >= width and upsample_height < height:
                    marginals_h += F.pad(F.max_unpool2d(marginals_h3, unpooling_indices, 3, 3)[:, :, :, :width], (0, 0, 0, height - upsample_height))
                    marginals_v += F.pad(F.max_unpool2d(marginals_v3, unpooling_indices, 3, 3)[:, :, :, :width], (0, 0, 0, height - upsample_height))
                else:
                    marginals_h += F.pad(F.max_unpool2d(marginals_h3, unpooling_indices, 3, 3), (0, width - upsample_width, 0, height - upsample_height))
                    marginals_v += F.pad(F.max_unpool2d(marginals_v3, unpooling_indices, 3, 3), (0, width - upsample_width, 0, height - upsample_height))

                cnt += 1

        # Update and return new unary terms
        average_marginals = (marginals_h + marginals_v) / 6.0

        unary_h[0] -= 1.0 / width * (marginals_hs[0] - average_marginals)
        unary_v[0] -= 1.0 / height * (marginals_vs[0] - average_marginals)

        cnt = 1

        for i in range(2):
            for j in range(2):
                unary_h[cnt] -= 1.0 / width * (marginals_hs[cnt] - average_marginals[:, :, i::2, j::2])
                unary_v[cnt] -= 1.0 / height * (marginals_vs[cnt] - average_marginals[:, :, i::2, j::2])
                cnt += 1

        for i in range(3):
            for j in range(3):
                unary_h[cnt] -= 1.0 / width * (marginals_hs[cnt] - average_marginals[:, :, i::3, j::3])
                unary_v[cnt] -= 1.0 / height * (marginals_vs[cnt] - average_marginals[:, :, i::3, j::3])
                cnt += 1

        return unary_h, unary_v, marginals_h, marginals_v


class FixedPointIterationGapN(torch.nn.Module):
    def __init__(self, state_size, learn_gamma=True):
        super(FixedPointIterationGapN, self).__init__()
        self.state_size = state_size
        if learn_gamma:
            self.gamma = torch.nn.Parameter(torch.ones(1))
        else:
            self.gamma = None

    def forward(self, *inputs):

        unary_h = inputs[0]
        unary_v = inputs[1]
        pairwise_h = inputs[2]
        pairwise_v = inputs[3]
        if self.gamma is None:
            gamma = inputs[4]
        else:
            # truncate gamma from below at 1e-2
            self.gamma.data.clamp_(min=1e-2)

        if self.gamma is None and gamma < 0:
            dp_function = DPFunction
        else:
            dp_function = DDPFunction

        if len(inputs) > 5:
            step_size = inputs[5]
        else:
            step_size = 1

        width = unary_h[0].size(-1)
        height = unary_v[0].size(-2)

        if len(inputs) > 6:
            alpha = inputs[6]
        else:
            alpha = (1.0 / width) if width > height else (1.0 / height)

        marginals_hs = []
        marginals_vs = []

        marginals_h1, marginals_v1 = dp_function.apply(
                                         unary_h[0], unary_v[0],
                                         pairwise_h[0], pairwise_v[0],
                                         gamma if self.gamma is None else self.gamma)
        b, c, h, w = marginals_h1.shape

        marginals_h = marginals_h1
        marginals_v = marginals_v1

        marginals_hs.append(marginals_h1)
        marginals_vs.append(marginals_v1)

        cnt = 1
        num_pairwise_terms = 1
        stride = 1

        while cnt < len(unary_h):
            stride += step_size
            num_pairwise_terms += 1

            for i in range(stride):
                for j in range(stride):
                    marginals_h_, marginals_v_ = dp_function.apply(
                                                     unary_h[cnt],
                                                     unary_v[cnt],
                                                     pairwise_h[cnt],
                                                     pairwise_v[cnt],
                                                     gamma if self.gamma is None else self.gamma)

                    marginals_hs.append(marginals_h_)
                    marginals_vs.append(marginals_v_)

                    _, _, height_, width_ = marginals_h_.shape

                    upsample_width = width_ * stride
                    upsample_height = height_ * stride

                    unpooling_indices = torch.arange(
                                            i * upsample_width + j, upsample_width * upsample_height,
                                            stride, dtype=torch.int64,
                                            device=unary_h[cnt].device, requires_grad=False). \
                                            view(-1, upsample_width // stride)[::stride, :].view(1, 1, upsample_height // stride, upsample_width // stride).repeat(b, c, 1, 1)

                    if upsample_width >= width and upsample_height >= height:
                        marginals_h = marginals_h + F.max_unpool2d(marginals_h_, unpooling_indices, stride, stride)[:, :, :height, :width]
                        marginals_v = marginals_v + F.max_unpool2d(marginals_v_, unpooling_indices, stride, stride)[:, :, :height, :width]
                    elif upsample_width < width and upsample_height >= height:
                        marginals_h = marginals_h + F.pad(F.max_unpool2d(marginals_h_, unpooling_indices, stride, stride)[:, :, :height, :], (0, width - upsample_width))
                        marginals_v = marginals_v + F.pad(F.max_unpool2d(marginals_v_, unpooling_indices, stride, stride)[:, :, :height, :], (0, width - upsample_width))
                    elif upsample_width >= width and upsample_height < height:
                        marginals_h = marginals_h + F.pad(F.max_unpool2d(marginals_h_, unpooling_indices, stride, stride)[:, :, :, :width], (0, 0, 0, height - upsample_height))
                        marginals_v = marginals_v + F.pad(F.max_unpool2d(marginals_v_, unpooling_indices, stride, stride)[:, :, :, :width], (0, 0, 0, height - upsample_height))
                    else:
                        marginals_h = marginals_h + F.pad(F.max_unpool2d(marginals_h_, unpooling_indices, stride, stride), (0, width - upsample_width, 0, height - upsample_height))
                        marginals_v = marginals_v + F.pad(F.max_unpool2d(marginals_v_, unpooling_indices, stride, stride), (0, width - upsample_width, 0, height - upsample_height))


                    cnt += 1

        # Update and return new unary terms
        denom = num_pairwise_terms * 2
        average_marginals = (marginals_h + marginals_v) / denom

        new_unary_h = []
        new_unary_v = []

        new_unary_h.append(unary_h[0] - alpha * (marginals_hs[0] - average_marginals))
        new_unary_v.append(unary_v[0] - alpha * (marginals_vs[0] - average_marginals))

        cnt = 1
        stride = 1 + step_size

        while cnt < len(unary_h):
            for i in range(stride):
                for j in range(stride):
                    new_unary_h.append(unary_h[cnt] - alpha * (marginals_hs[cnt] - average_marginals[:, :, i::stride, j::stride]))
                    new_unary_v.append(unary_v[cnt] - alpha * (marginals_vs[cnt] - average_marginals[:, :, i::stride, j::stride]))
                    cnt += 1

            stride += step_size

        return new_unary_h, new_unary_v, marginals_h, marginals_v, marginals_hs, marginals_vs
