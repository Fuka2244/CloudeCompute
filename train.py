import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


def get_env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def should_use_ddp():
    """
    Kubeflow PyTorchJob 会注入 WORLD_SIZE、RANK、MASTER_ADDR 等变量。
    WORLD_SIZE > 1 时，说明是分布式训练。
    """
    world_size = get_env_int("WORLD_SIZE", 1)
    return world_size > 1


def setup_distributed(backend):
    if should_use_ddp():
        dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return True, rank, world_size

    return False, 0, 1


def cleanup_distributed(distributed_mode):
    if distributed_mode and dist.is_initialized():
        dist.destroy_process_group()


def train_one_epoch(args, model, device, train_loader, optimizer, epoch, rank, world_size):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()

        if rank == 0 and batch_idx % args.log_interval == 0:
            processed = batch_idx * len(data) * world_size
            processed = min(processed, len(train_loader.dataset))

            percent = 100.0 * batch_idx / len(train_loader)

            print(
                f"Train Epoch: {epoch} "
                f"[{processed}/{len(train_loader.dataset)} "
                f"({percent:.0f}%)]\tLoss: {loss.item():.6f}",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description="MNIST CNN Single Node and DDP Training")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--data-dir", type=str, default="/workspace/data")

    args = parser.parse_args()

    distributed_mode, rank, world_size = setup_distributed(args.backend)

    device = torch.device("cpu")

    if rank == 0:
        print("========== 运行配置 ==========", flush=True)
        print(f"MODE={'DDP' if distributed_mode else 'SINGLE'}", flush=True)
        print(f"RANK={rank}", flush=True)
        print(f"WORLD_SIZE={world_size}", flush=True)
        print(f"DEVICE={device}", flush=True)
        print(f"DATA_DIR={args.data_dir}", flush=True)
        print("==============================", flush=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset = datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    if distributed_mode:
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=2,
        )
    else:
        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
        )

    model = Net().to(device)

    if distributed_mode:
        model = DDP(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    if distributed_mode:
        dist.barrier()

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        if distributed_mode:
            train_sampler.set_epoch(epoch)

        train_one_epoch(
            args=args,
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
        )

    if distributed_mode:
        dist.barrier()

    elapsed = time.time() - start_time

    if rank == 0:
        print("\n========== 训练完成 ==========", flush=True)
        print(f"MODE={'DDP' if distributed_mode else 'SINGLE'}", flush=True)
        print(f"WORLD_SIZE={world_size}", flush=True)
        print(f"TRAINING_TIME_SECONDS={elapsed:.2f}", flush=True)
        print(f"[Rank 0] 总耗时: {elapsed:.2f} 秒", flush=True)
        print("==============================\n", flush=True)

    cleanup_distributed(distributed_mode)


if __name__ == "__main__":
    main()