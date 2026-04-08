import torch
import torch.nn as nn

class M_XA(nn.Module):
    def __init__(
        self,
        time_step,
        channels=64,
        reduction=16,
        spatial_kernel=7
    ):
        super().__init__()
        reduced_c = max(channels // reduction, 4)

        self.conv_t = nn.Conv1d(
            in_channels=time_step,
            out_channels=time_step,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.chan_mlp = nn.Sequential(
            nn.Linear(channels, reduced_c, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_c, channels, bias=False),
        )

        padding = spatial_kernel // 2
        self.spatial_attn = nn.Conv2d(
            in_channels=2, out_channels=1,
            kernel_size=spatial_kernel, padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

        self.scale_t = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.scale_c = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.scale_s = nn.Parameter(torch.tensor(1.0, dtype=torch.float))

    def forward(self, x_seq):
        b, t, c, _, _ = x_seq.shape

        pooled = x_seq.mean(dim=[3, 4])  # B, T, C

        time_out = self.conv_t(pooled)
        attn_t = 1 + self.scale_t * self.sigmoid(time_out)

        chan_in = pooled.reshape(-1, c)
        chan_out = self.chan_mlp(chan_in).view(b, t, c)
        attn_c = 1 + self.scale_c * self.sigmoid(chan_out)

        attn_t = attn_t[:, :, :, None, None]
        attn_c = attn_c[:, :, :, None, None]

        spatial_feat = x_seq.mean(dim=1)  # B, C, H, W
        avg_pool = spatial_feat.mean(dim=1, keepdim=True)
        max_pool, _ = spatial_feat.max(dim=1, keepdim=True)
        spatial_out = self.spatial_attn(torch.cat([avg_pool, max_pool], dim=1))
        attn_s = 1 + self.scale_s * self.sigmoid(spatial_out)
        attn_s = attn_s[:, None, :, :, :]

        return x_seq * attn_t * attn_c * attn_s


class TCSA(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size,
        dilation=3,
        reduction=16,
        spatial_kernel=7,
        spatial_reduction=16,
    ):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)
        self.LTCA = nn.Sequential(
            nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim),
            nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation),
            nn.Conv2d(dim, dim, 1)
        )
        self.reduction = max(dim // reduction, 4)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.MLP_block = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False))

        spatial_padding = spatial_kernel // 2
        self.LSA = nn.Sequential(
            nn.Conv2d(dim, dim, spatial_kernel, padding=spatial_padding, groups=dim),
            nn.Conv2d(dim, dim, 1),
        )

        self.spatial_reduction = max(dim // spatial_reduction, 4)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.spatial_conv1 = nn.Conv2d(dim, self.spatial_reduction, kernel_size=1, bias=False)
        self.spatial_bn1 = nn.BatchNorm2d(self.spatial_reduction)
        self.spatial_act = nn.ReLU(inplace=True)
        self.spatial_conv_h = nn.Conv2d(self.spatial_reduction, dim, kernel_size=1, bias=False)
        self.spatial_conv_w = nn.Conv2d(self.spatial_reduction, dim, kernel_size=1, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()
        self.scale_t = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.scale_c = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.scale_lsa = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.scale_gsa = nn.Parameter(torch.tensor(1.0, dtype=torch.float))


    def forward(self, x):
        LTCA_attn = self.LTCA(x)

        b, c, h, w = x.size()
        GAP_attn = self.GAP(x).view(b, c)
        GTCA_attn = self.MLP_block(GAP_attn).view(b, c, 1, 1)
        LSA_attn = self.LSA(x)

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        spatial_cat = torch.cat([x_h, x_w], dim=2)
        spatial_cat = self.spatial_conv1(spatial_cat)
        spatial_cat = self.spatial_bn1(spatial_cat)
        spatial_cat = self.spatial_act(spatial_cat)
        x_h_split, x_w_split = torch.split(spatial_cat, [h, w], dim=2)
        x_w_split = x_w_split.permute(0, 1, 3, 2)
        GSA_attn_h = self.spatial_conv_h(x_h_split)
        GSA_attn_w = self.spatial_conv_w(x_w_split)
        GSA_attn = self.spatial_sigmoid(GSA_attn_h) * self.spatial_sigmoid(GSA_attn_w)

        LTCA_attn = 1 + self.scale_t * torch.tanh(LTCA_attn)
        GTCA_attn = 1 + self.scale_c * torch.tanh(GTCA_attn)
        LSA_attn = 1 + self.scale_lsa * torch.tanh(LSA_attn)
        GSA_attn = 1 + self.scale_gsa * torch.tanh(GSA_attn)

        return x * LTCA_attn * GTCA_attn * LSA_attn * GSA_attn


class M_NA(nn.Module):
    def __init__(self, in_planes, kernel_size=21, attn_shortcut=True):
        super().__init__()

        self.encoding = nn.Sequential(nn.Conv2d(in_planes, in_planes, 1), nn.GELU() )
        self.tcsa = TCSA(in_planes, kernel_size)
        self.decoding = nn.Conv2d(in_planes, in_planes, 1)
        self.attn_shortcut = attn_shortcut

    def forward(self, x):

        if self.attn_shortcut:
            shortcut = x

        x = self.encoding(x)
        x = self.tcsa(x)
        x = self.decoding(x)

        if self.attn_shortcut:
            x = x + shortcut
        return x

class Chao_A(nn.Module):
    """
    Chaos mapping module using a Logistic map:
    1. Learnable alpha/x0 parameters per channel (or shared if num_channels=1)
    2. Iterate the map along the temporal dimension to produce x_chaos(t)
    3. Broadcast to (b, t, c, h, w) and apply as a gating mask via Hadamard product
    -----------------------------------------------------------------
    Adjustments:
    - Change alpha/x0 parameter shapes if you need per-position values
    - Return only the chaos gating when integrating with TA/SCA externally
    - Constrain alpha/x0 further if required (e.g., keep alpha in [3.57, 4])
    -----------------------------------------------------------------
    """
    def __init__(self, 
                num_channels,  # controls dimensionality of alpha/x0 (typically matches channels)
                alpha_min=3.57, 
                alpha_max=4.0):
        super().__init__()
        self.num_channels = num_channels
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        # learnable alpha/x0 parameters; shape=[num_channels] (use 1 for global sharing)
        self.alpha_param = nn.Parameter(torch.randn(num_channels))
        self.x0_param = nn.Parameter(torch.randn(num_channels))
        
        # map alpha to [alpha_min, alpha_max] and x0 to (0, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        x: (b, t, c, h, w)
        returns: tensor with same shape, gated by the chaos sequence along the time axis
        """
        b, t, c, h, w = x.shape

        if c != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, but got {c}")

        if t == 0:
            return x
        
        # alpha in [alpha_min, alpha_max]; alpha.shape = [num_channels]
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * self.sigmoid(self.alpha_param)
        alpha = alpha.to(dtype=x.dtype, device=x.device)
        
        # initial x0 mapped to (0, 1)
        x_prev = self.sigmoid(self.x0_param).to(dtype=x.dtype, device=x.device)

        # iterate logistic map along the temporal dimension
        x_chaos = torch.empty(t, c, device=x.device, dtype=x.dtype)
        for step in range(t):
            x_prev = alpha * x_prev * (1.0 - x_prev)
            x_chaos[step] = x_prev
        
        # broadcast to (b, t, c, h, w) without materializing copies
        x_chaos = x_chaos.view(1, t, c, 1, 1).expand(b, t, c, h, w)
        
        # Hadamard product gating
        out = x * x_chaos
        return out

class TCMMAC(nn.Module):
    def __init__(self, T, out_channels, enable_na=True, enable_xa=True, enable_chao=True):
        super().__init__()

        self.M_NA   = M_NA(in_planes=out_channels * T, kernel_size=7) if enable_na else None
        self.M_XA   = M_XA(time_step=T, channels=out_channels) if enable_xa else None
        self.Chao_A = Chao_A(num_channels=out_channels) if enable_chao else None

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq, spikes):

        B, T, C, H, W = x_seq.shape

        attn_components = []

        if self.M_NA is not None:
            x_seq_2 = x_seq.reshape(B, T * C, H, W)
            attn_na = self.M_NA(x_seq_2).reshape(B, T, C, H, W)
            attn_components.append(attn_na)

        if self.M_XA is not None:
            attn_components.append(self.M_XA(x_seq))
        
        if self.Chao_A is not None:
            attn_components.append(self.Chao_A(x_seq))

        if not attn_components:
            return spikes

        attn = attn_components[0]
        for comp in attn_components[1:]:
            attn = attn * comp

        out = self.sigmoid(attn)

        y_seq = out * spikes

        return y_seq
