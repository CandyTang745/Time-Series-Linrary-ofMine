import torch
import torch.nn as nn
import torch.fft


class Model(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.enc_in = args.enc_in
        self.d_model = args.d_model

        # channel independence
        self.input_proj = nn.Linear(1, self.d_model)

        # temporal encoder (Conv1D)
        self.time_encoder = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.GELU()
        )

        # frequency encoder (very light)
        self.freq_encoder = nn.Sequential(
            nn.Linear(self.seq_len//2 + 1, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )

        # fusion
        self.fusion = nn.Linear(self.d_model * 2, self.d_model)

        # prediction head
        self.head = nn.Linear(self.seq_len * self.d_model, self.pred_len)


    def forward(self, x, batch_x_mark, dec_inp, batch_y_mark):

        B, L, C = x.shape

        # channel independence
        x = x.permute(0,2,1)
        x = x.reshape(B*C, L, 1)

        # normalization
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5
        x = (x - mean) / std

        # embedding
        x = self.input_proj(x)      # [BC,L,d]

        # ----- time encoder -----
        t = x.permute(0,2,1)        # [BC,d,L]
        t = self.time_encoder(t)
        t = t.permute(0,2,1)        # [BC,L,d]

        # ----- frequency encoder -----
        xf = torch.fft.rfft(x.squeeze(-1), dim=1)
        xf = torch.abs(xf)

        f = self.freq_encoder(xf)   # [BC,d]
        f = f.unsqueeze(1).repeat(1,L,1)

        # ----- fusion -----
        feat = torch.cat([t,f], dim=-1)
        feat = self.fusion(feat)

        # ----- prediction -----
        feat = feat.reshape(B*C, -1)
        out = self.head(feat)

        # inverse norm
        out = out * std.squeeze(-1) + mean.squeeze(-1)

        out = out.reshape(B, C, self.pred_len)

        out = out[:,0,:]
        out = out.unsqueeze(-1)

        return out