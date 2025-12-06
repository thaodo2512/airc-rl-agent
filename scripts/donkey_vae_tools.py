#!/usr/bin/env python3
"""
Utility helpers to (1) record camera frames from DonkeySim into a dataset folder
and (2) train the repo's VAE on that dataset to produce vae.torch.
"""

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# Ensure local repo import works even if package not installed
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning_racer.vae.vae import VAE


class DonkeyImageDataset(Dataset):
    def __init__(self, root: Path, transform: T.Compose):
        self.paths = sorted(
            p for p in Path(root).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def collect_from_sim(args) -> None:
    import gym
    import gym_donkeycar  # noqa: F401
    import imageio

    conf = {
        "exe_path": "remote",
        "port": args.port,
        "host": args.host,
        "cam_resolution": (160, 120),
    }
    env = gym.make(args.env, conf=conf)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    obs = env.reset()
    for i in range(args.frames):
        steer = args.steering_amp * math.sin(i / max(args.steering_period, 1.0))
        steer += rng.normal(scale=args.steering_noise)
        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(args.throttle)
        obs, _, done, _ = env.step([steer, throttle])
        imageio.imwrite(out_dir / f"{i:06d}.jpg", obs)
        if done:
            obs = env.reset()
    env.close()
    print(f"Saved {args.frames} frames to {out_dir}")


def train_vae(args) -> None:
    device = torch.device(args.device)
    transform = T.Compose(
        [
            T.Resize((120, 160)),
            T.Lambda(lambda x: x.crop((0, 40, 160, 120))),
            T.ToTensor(),
        ]
    )
    dataset = DonkeyImageDataset(Path(args.data_dir), transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    vae = VAE(image_channels=3, z_dim=args.z_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

    vae.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            mu_y, sigma_y, mu, logvar = vae(batch)
            loss = vae.loss_fn(batch, mu_y, sigma_y, mu, logvar)
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / max(len(loader), 1)
        print(f"Epoch {epoch}/{args.epochs} - loss {avg:.2f}")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vae.state_dict(), out_path, _use_new_zipfile_serialization=True)
    print(f"Saved VAE weights to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DonkeySim dataset collector and VAE trainer.")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Record frames from DonkeySim into a folder.")
    collect.add_argument("--host", default="localhost", help="DonkeySim host.")
    collect.add_argument("--port", type=int, default=9091, help="DonkeySim port.")
    collect.add_argument("--frames", type=int, default=5000, help="Number of frames to capture.")
    collect.add_argument("--out", default="dataset", help="Output folder for images.")
    collect.add_argument("--throttle", type=float, default=0.35, help="Constant throttle to apply.")
    collect.add_argument(
        "--steering-amp", type=float, default=0.4, help="Steering sinusoid amplitude (clip to [-1,1])."
    )
    collect.add_argument(
        "--steering-period", type=float, default=40.0, help="Steering sinusoid period divisor."
    )
    collect.add_argument("--steering-noise", type=float, default=0.05, help="Gaussian noise on steering.")
    collect.add_argument("--seed", type=int, default=0, help="Seed for steering noise.")
    collect.add_argument(
        "--env",
        default="donkey-generated-track-v0",
        help="Gym env ID (track) to load, e.g., donkey-warehouse-v0, donkey-mountain-track-v0.",
    )

    train = sub.add_parser("train", help="Train VAE on a folder of images and write vae.torch.")
    train.add_argument("--data-dir", default="dataset", help="Folder of images (jpg/png).")
    train.add_argument("--out", default="vae.torch", help="Output path for saved VAE weights.")
    train.add_argument("--device", default="cuda", help="torch device (cuda or cpu).")
    train.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    train.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    train.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    train.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    train.add_argument("--z-dim", type=int, default=32, help="Latent dimension (match config.yml).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "collect":
        collect_from_sim(args)
    elif args.command == "train":
        train_vae(args)


if __name__ == "__main__":
    main()
