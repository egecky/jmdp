import torch
import torch.nn as nn


class AtariEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 84, 84)
            n = self.conv(dummy).view(1, -1).shape[1]
        self.fc = nn.Sequential(nn.Linear(n, out_dim), nn.ReLU())

    def forward(self, obs):
        x = self.conv(obs)
        x = x.view(x.shape[0], -1)
        return self.fc(x)


class JIPE2Net(nn.Module):
    def __init__(
        self, n_actions, enc_dim=256, hid=256, act_dim=32, sigma_mode="pair", gram_dim=128
    ):
        super().__init__()
        self.nA = int(n_actions)
        if sigma_mode not in {"pair", "gram", "covgram"}:
            raise ValueError(f"sigma_mode must be 'pair', 'gram', or 'covgram', got {sigma_mode!r}")
        self.sigma_mode = str(sigma_mode)
        self.encoder = AtariEncoder(out_dim=enc_dim)
        self.mu_head = nn.Sequential(nn.Linear(enc_dim, hid), nn.ReLU(), nn.Linear(hid, self.nA))
        self.act_emb = nn.Embedding(self.nA, act_dim)
        feat_dim = 3 * enc_dim + 3 * act_dim
        self.sig_head = nn.Sequential(
            nn.Linear(feat_dim, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1)
        )
        self.gram_head = nn.Sequential(
            nn.Linear(enc_dim + act_dim, hid), nn.ReLU(), nn.Linear(hid, int(gram_dim))
        )
        self.cov_gram_head = nn.Sequential(
            nn.Linear(enc_dim + act_dim, hid), nn.ReLU(), nn.Linear(hid, int(gram_dim))
        )
        cov_last = self.cov_gram_head[-1]
        nn.init.normal_(cov_last.weight, mean=0.0, std=0.001)
        nn.init.zeros_(cov_last.bias)

    def _state_features(self, obs):
        return self.encoder(obs)

    def mu_all(self, obs):
        h = self._state_features(obs)
        return self.mu_head(h)

    def mu(self, obs, act):
        q = self.mu_all(obs)
        return q.gather(1, act.view(-1, 1)).squeeze(1)

    def _pair_features(self, h1, a1, h2, a2):
        e1 = self.act_emb(a1)
        e2 = self.act_emb(a2)
        h_sum = h1 + h2
        h_diff = torch.abs(h1 - h2)
        h_prod = h1 * h2
        e_sum = e1 + e2
        e_diff = torch.abs(e1 - e2)
        e_prod = e1 * e2
        return torch.cat([h_sum, h_diff, h_prod, e_sum, e_diff, e_prod], dim=1)

    def _gram_features(self, h, a):
        e = self.act_emb(a)
        return self.gram_head(torch.cat([h, e], dim=1))

    def _cov_gram_features(self, h, a):
        e = self.act_emb(a)
        return self.cov_gram_head(torch.cat([h, e], dim=1))

    def Sigma(self, obs1, a1, obs2, a2):
        h1 = self._state_features(obs1)
        h2 = self._state_features(obs2)
        if self.sigma_mode == "gram":
            u1 = self._gram_features(h1, a1)
            u2 = self._gram_features(h2, a2)
            return torch.sum(u1 * u2, dim=1)
        if self.sigma_mode == "covgram":
            v1 = self._cov_gram_features(h1, a1)
            v2 = self._cov_gram_features(h2, a2)
            q1 = self.mu_head(h1).gather(1, a1.view(-1, 1)).squeeze(1)
            q2 = self.mu_head(h2).gather(1, a2.view(-1, 1)).squeeze(1)
            return q1 * q2 + torch.sum(v1 * v2, dim=1)
        feat = self._pair_features(h1, a1, h2, a2)
        out = self.sig_head(feat).squeeze(1)
        return out

    def covariance(self, obs1, a1, obs2, a2):
        h1 = self._state_features(obs1)
        h2 = self._state_features(obs2)
        if self.sigma_mode == "covgram":
            v1 = self._cov_gram_features(h1, a1)
            v2 = self._cov_gram_features(h2, a2)
            return torch.sum(v1 * v2, dim=1)
        q1 = self.mu_head(h1).gather(1, a1.view(-1, 1)).squeeze(1)
        q2 = self.mu_head(h2).gather(1, a2.view(-1, 1)).squeeze(1)
        if self.sigma_mode == "gram":
            u1 = self._gram_features(h1, a1)
            u2 = self._gram_features(h2, a2)
            raw = torch.sum(u1 * u2, dim=1)
        else:
            feat = self._pair_features(h1, a1, h2, a2)
            raw = self.sig_head(feat).squeeze(1)
        return raw - q1 * q2


class JIPE2Nets:
    def __init__(self, model):
        self.model = model
