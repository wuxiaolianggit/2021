# source: https://github.com/JDAI-CV/CoTNet

class CotLayer(nn.Module):
    def __init__(self, dim, kernel_size):
        super(CotLayer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size//2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(2*dim, dim//factor, 1, bias=False),
            nn.BatchNorm2d(dim//factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm2d(dim)
        act = get_act_layer('swish')
        self.act = act(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_chs, self.radix*dim, 1)
        )

    def forward(self, x):
        # 获取static Contexts：CBL三元组 （3×3）
        k = self.key_embed(x)
        # 将查询特征 q 和键 k 拼接起来（Concat）
        qk = torch.cat([x, k], dim=1)
        # 获取qk矩阵的 B, C, H, W
        b, c, qk_hh, qk_ww = qk.size()

        # 两个1×1卷积层：CBL + CG
        # 1. 第二个卷积层没有ReLU，可以避免屏蔽过多的有效信息
        # 2. 第二个卷积层使用GroupNorm，将channel分为G组，每组分别做归一化，避免不同batch特征之间造成干扰
        w = self.embed(qk)
        # 展开为多个头部
        w = w.view(b, 1, -1, self.kernel_size*self.kernel_size, qk_hh, qk_ww)

        # 获取 value
        # Conv 1×1 + BN
        x = self.conv1x1(x)
        # 获取dynamic contexts：结合权重 w 和 值 v 作矩阵乘法，对每个空间像素进行加权
        # Conv 3×3 + BN + swish
        x = self.local_conv(x)
        x = self.bn(x)
        x = self.act(x)

        # 增强维度用于特征融合
        B, C, H, W = x.shape
        x = x.view(B, C, 1, H, W)
        k = k.view(B, C, 1, H, W)
        #将 dynamic contexts 和 static contexts 拼接起来（Concat）
        x = torch.cat([x, k], dim=2)

        # 融合 dynamic contexts 和 static contexts，并基于此再进行注意力加权
        x_gap = x.sum(dim=2)
        # 对特征图的空间维度求均值
        x_gap = x_gap.mean((2, 3), keepdim=True)
        # CBLC （1×1）特征编码
        x_attn = self.se(x_gap)
        # 特征维度变换
        x_attn = x_attn.view(B, C, self.radix)
        # 计算特征重要性
        x_attn = F.softmax(x_attn, dim=2)
        # 将注意力图重新加权回 x
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1))).sum(dim=2)
        # 使tensor在内存中连续存放
        return out.contiguous()
