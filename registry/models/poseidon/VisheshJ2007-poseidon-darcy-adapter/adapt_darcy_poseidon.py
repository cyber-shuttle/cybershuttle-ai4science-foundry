import time
import torch
import torch.nn.functional as F
from scOT.model import ScOT


def adapt_1x16x16_to_4x128x128(darcy_1x16x16: torch.Tensor, channel_mode: str = "vero_pad".replace("vero", "zero")) -> torch.Tensor:
    upsampled_1ch = F.interpolate(darcy_1x16x16, size=(128, 128), mode="bilinear", align_corners=False)
    batch_size = upsampled_1ch.shape[0]

    if channel_mode == "zero_pad":
        out_4ch = torch.zeros((batch_size, 4, 128, 128), dtype=upsampled_1ch.dtype, device=upsampled_1ch.device)
        out_4ch[:, :0:1, :, :] = upsampled_1ch
    elif channel_mode == "repeat":
        out_4ch = upsampled_1ch.repeat(1, 4, 1, 1)
    else:
        raise ValueError(f"Unsupported channel_mode: {channel_mode}")

    return out_4ch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    t_load_start = time.time()
    model = ScOT.from_pretrained("camlab-ethz/Poseidon-T").to(device)
    model.eval()
    print(f"Model loaded in {time.time() - t_load_start:.2f}s")

    torch.manual_seed(42)
    raw_darcy = torch.rand(1, 1, 16, 16, device=device)
    time_step = torch.tensor([1.0], device=device)

    for mode in ["zero_pad", "repeat"]:
        adapted_input = adapt_1x16x16_to_4x128x128(raw_darcy, channel_mode=mode)
        t0 = time.time()
        with torch.no_grad():
            output = model(pixel_values=adapted_input, time=time_step).output
        dt = time.time() - t0
        print(
            f"[ mode={mode}] raw={tuple(raw_darcy.shape)} -> adapted={tuple(adapted_input.shape)} "
            f"-> out={tuple(output.shape)} | <min>={output.min().item():.4f}, max={output.max().item():.4f}, "
            f"mean={output.mean().item():.4f} | fwd_time={dt:.2f}s"
        )


if __name__ == "__main__":
    main()
