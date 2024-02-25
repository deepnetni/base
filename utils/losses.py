from einops import rearrange
import torch
from torch import Tensor
from models.PMSQE.pmsqe_asteroid import SingleSrcPMSQE


def loss_pmsqe(sph_spec, est_spec, pad_mask=None):
    """Perceptally-Motivated Speech Quality
    Args:
        `sph_spec` and `est_spec`: features in frequency-domain
            with shape B,2,T,F
        pad_mask: indicate the padding frames with shape B, T, 1 where 1 for valid frame.
    """

    def power(spec, freq_dim=-1):
        # B,T,2F -> ((B,T,F), (B,T,F))(chunk) -> B,T,F,2(stack) -> B,T,F(sum)
        return (
            torch.stack(torch.chunk(spec, 2, dim=freq_dim), dim=-1).pow(2).sum(dim=-1)
        )

    if pad_mask is None:
        pad_mask = torch.ones(
            est_spec.shape[0], est_spec.shape[2], 1, device=est_spec.device
        )

    # b,2,t,f -> b,t,2f
    sph_spec = rearrange(sph_spec, "b c t f->b t (c f)")
    est_spec = rearrange(est_spec, "b c t f->b t (c f)")

    power_sph_spec = power(sph_spec)
    power_est_spec = power(est_spec)

    # wD: Mean symmetric distortion.
    # wDA: Mean asymmetric distortion.
    # pmsq = PMSQE().cuda()
    # wD, wDA = pmsq(power_est_spec, power_sph_spec, pad_mask)
    # alpha = 0.1
    # score = alpha * (wD + 0.309 * wDA)

    pmsq = SingleSrcPMSQE(sample_rate=16000).cuda()
    score = pmsq(power_est_spec, power_sph_spec, pad_mask)
    score = score.mean()

    return score


def l2_norm(s: Tensor, keepdim=False):
    """
    sum(x^ord) ^ 1/ord
    """
    return torch.linalg.norm(s, dim=-1, keepdim=keepdim)


def loss_sisnr(sph: Tensor, est, zero_mean: bool = False) -> Tensor:
    """
    Args:
        sph: float tensor with shape `(..., time)`
        est: float tensor with shape `(..., time)`

    Returns:
        Float tensor with shape `(...,)` of SDR values per sample

    Example:
        >>> a = torch.tensor([1,2,3,4]).float()
        >>> b = torch.tensor([1,2,3,4]).float()
        >>> score = loss_sisnr(a, b)

    Algo:
        s_target = <sph, est> * sph / sph^2, where <> means inner dot
        e_noise = est - s_target
        sisnr = 10 * log_10(|s_target|^2 / |e_noise|^2)
    """
    # eps = torch.finfo(torch.float32).eps
    # eps = 1e-8
    eps = torch.finfo(sph.dtype).eps

    if zero_mean is True:
        s = sph - torch.mean(sph, dim=-1, keepdim=True)
        s_hat = est - torch.mean(est, dim=-1, keepdim=True)
    else:
        s = sph
        s_hat = est

    s_target = (
        (torch.sum(s_hat * s, dim=-1, keepdim=True) + eps)
        * s
        / (l2_norm(s, keepdim=True) ** 2 + eps)
    )
    e_noise = s_hat - s_target
    # sisnr = 10 * torch.log10(
    #     (l2_norm(s_target) ** 2 + eps) / (l2_norm(e_noise) ** 2 + eps)
    # )
    sisnr = 10 * torch.log10(
        (torch.sum(s_target**2, dim=-1) + eps)
        / (torch.sum(e_noise**2, dim=-1) + eps)
    )
    return -torch.mean(sisnr)


def loss_snr(sph: Tensor, est: Tensor, zero_mean: bool = False) -> Tensor:
    r"""Calculate `Signal-to-noise ratio`_ (SNR_) meric for evaluating quality of audio.

    .. math::
        \text{SNR} = \frac{P_{signal}}{P_{noise}}

    where  :math:`P` denotes the power of each signal. The SNR metric compares the level of the desired signal to
    the level of background noise. Therefore, a high value of SNR means that the audio is clear.

    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``
        zero_mean: if to zero mean target and preds or not

    Returns:
        Float tensor with shape ``(...,)`` of SNR values per sample

    Raises:
        RuntimeError:
            If ``preds`` and ``target`` does not have the same shape

    Example:
        >>> from torchmetrics.functional.audio import signal_noise_ratio
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> signal_noise_ratio(preds, target)
        tensor(16.1805)
    """
    eps = torch.finfo(sph.dtype).eps
    if zero_mean:
        sph = sph - torch.mean(sph, dim=-1, keepdim=True)
        est = est - torch.mean(est, dim=-1, keepdim=True)

    noise = sph - est

    snr_value = 10 * torch.log10(
        (torch.sum(sph**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    )
    return -torch.mean(snr_value)


def loss_compressed_mag(sph: Tensor, est: Tensor, compress_factor=0.3):
    """
    Input:
        sph: specturm of sph, B,2,T,F
        est: specturm of sph, B,2,T,F

    Return: loss of mse_mag, mse_specs
    """
    mag_sph_2 = torch.maximum(
        torch.sum((sph * sph), dim=1, keepdim=True),
        torch.zeros_like(torch.sum((sph * sph), dim=1, keepdim=True)) + 1e-12,
    )
    mag_est_2 = torch.maximum(
        torch.sum((est * est), dim=1, keepdim=True),
        torch.zeros_like(torch.sum((est * est), dim=1, keepdim=True)) + 1e-12,
    )
    mag_sph_cpr = torch.pow(mag_sph_2, compress_factor / 2)  # B,1,T,F
    mag_est_cpr = torch.pow(mag_est_2, compress_factor / 2)  # B,1,T,F
    specs_sph_cpr = torch.pow(mag_sph_2, (compress_factor - 1) / 2) * sph  # B,2,T,F
    specs_est_cpr = torch.pow(mag_est_2, (compress_factor - 1) / 2) * est  # B,2,T,F

    mse_mag = torch.mean((mag_sph_cpr - mag_est_cpr) ** 2)
    mse_specs = torch.mean((specs_sph_cpr - specs_est_cpr) ** 2)
    return mse_mag, mse_specs
