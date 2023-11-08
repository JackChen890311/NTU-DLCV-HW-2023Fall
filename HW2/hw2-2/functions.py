import torch

from utils import beta_scheduler
from constant import CONSTANT

C = CONSTANT()

def inverse_data_transform(X):
    X = (X - X.min()) / (X.max() - X.min())
    return X


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def sample_sequence(model, noise, eta):
    model.eval()

    with torch.no_grad():
        _, x = sample_image(noise, model, eta, last=False)

    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = inverse_data_transform(x[i][j])

    return x


def sample_interpolation(model, z1, z2, eta, use_slerp=True):
    def slerp(z1, z2, alpha):
        theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
        return (
            torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
            + torch.sin(alpha * theta) / torch.sin(theta) * z2
        )
    def linear(z1, z2, alpha):
        return ((1 - alpha) * z1 + alpha  * z2)

    model.eval()
    z1, z2 = z1.to(C.device), z2.to(C.device)

    alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
    z_ = []
    for i in range(alpha.size(0)):
        z_.append(slerp(z1, z2, alpha[i]) if use_slerp else linear(z1, z2, alpha[i]))

    x = torch.cat(z_, dim=0)
    xs = []

    # Hard coded here, modify to your preferences
    with torch.no_grad():
        for i in range(0, x.size(0), 8):
            xs.append(sample_image(x[i : i + 8], model, eta))
    x = torch.cat(xs, dim=0)
    for i in range(x.size(0)):
        x[i] = inverse_data_transform(x[i])

    return x


def sample_image(x, model, eta, last=True):
    
    betas = beta_scheduler(C.num_timesteps, C.beta_start, C.beta_end).float().to(C.device)

    skip = C.num_timesteps // C.timesteps
    seq = range(0, C.num_timesteps, skip)
    seq = [int(s) for s in list(seq)]
    x = generalized_steps(x, seq, model, betas, eta=eta)

    if last:
        x = x[0][-1]
    return x