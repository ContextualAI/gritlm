import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not dist.is_initialized():
        return 1
    else:
        return dist.get_world_size()

@torch.no_grad()
def varsize_all_gather(x: torch.Tensor, dim: int = 0):
    """all_gather tensors of different sizes along the specified dimension with concatenation"""
    if not dist.is_initialized():
        return x

    size = x.size(dim)
    tensor_size = torch.tensor(size, device=x.device, dtype=torch.int64)
    all_sizes = [torch.zeros_like(tensor_size) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, tensor_size)
    max_size = max([s.item() for s in all_sizes])

    padding_tuple_size = [max_size - size if k == dim else x.size(k) for k in range(x.ndim)]
    tensor_tuple_size = [max_size if k == dim else x.size(k) for k in range(x.ndim)]
    if size != max_size:
        padding = torch.empty(size=padding_tuple_size, dtype=x.dtype, device=x.device)
        x = torch.cat((x, padding), dim=dim)

    tensor_list = [torch.empty(tensor_tuple_size, device=x.device, dtype=x.dtype) for s in all_sizes]

    dist.all_gather(tensor_list=tensor_list, tensor=x)
    tensor_list = [torch.narrow(tensor, dim, start=0, length=all_sizes[k]) for k, tensor in enumerate(tensor_list)]
    output = torch.cat(tensor_list, dim=dim)
    return output

@torch.no_grad()
def varsize_gather(x: torch.Tensor, dst: int = 0, dim: int = 0):
    """gather tensors of different sizes along the specified dimension"""
    if not dist.is_initialized():
        return x

    size = x.size(dim)
    tensor_size = torch.tensor(size, device=x.device, dtype=torch.int64)
    all_sizes = [torch.zeros_like(tensor_size) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, tensor_size)
    max_size = max([s.item() for s in all_sizes])

    padding_tuple_size = [max_size - size if k == dim else x.size(k) for k in range(x.ndim)]
    tensor_tuple_size = [max_size if k == dim else x.size(k) for k in range(x.ndim)]
    if size != max_size:
        padding = torch.empty(size=padding_tuple_size, dtype=x.dtype, device=x.device)
        x = torch.cat((x, padding), dim=dim)

    if get_rank() == dst:
        tensor_list = [torch.empty(tensor_tuple_size, device=x.device, dtype=x.dtype) for s in all_sizes]
    else:
        tensor_list = None

    dist.gather(x, gather_list=tensor_list, dst=dst)
    if get_rank() == dst:
        tensor_list = [torch.narrow(tensor, dim, start=0, length=all_sizes[k]) for k, tensor in enumerate(tensor_list)]

    return tensor_list

@torch.no_grad()
def get_varsize(x: torch.Tensor, dim: int = 0):
    """gather tensors of different sizes along the first dimension"""
    if not dist.is_initialized():
        return torch.tensor([x.size(dim)])

    # determine max size
    size = torch.tensor([x.size(dim)], device=x.device, dtype=torch.int)
    allsizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(allsizes, size)
    allsizes = torch.cat(allsizes)
    return allsizes

def weighted_average(x, count):
    if not dist.is_initialized():
        if isinstance(x, torch.Tensor):
            x = x.item()
        return x, count
    t_loss = torch.tensor([x * count]).cuda()
    t_total = torch.tensor([count]).cuda()
    t_loss = sum_main(t_loss)
    t_total = sum_main(t_total)
    return (t_loss / t_total).item(), t_total.item()

def avg_dist_dict(keys, dictionary):
    avg = {}
    for m in keys:
        v = dictionary[m]
        if len(v) > 0:
            avg[m] = np.mean(v)
        else:
            avg[m] = 0.0
        avg[m] = weighted_average(avg[m], len(v))[0]
    return avg

def save_distributed_dataset(data, dataset_name, global_rank, dir_path):
    dir_path = Path(dir_path)
    write_path = dir_path / "tmp_dir"
    write_path.mkdir(exist_ok=True)
    tmp_path = write_path / f"{global_rank}.json"
    with open(tmp_path, "w") as fw:
        json.dump(data, fw)
    barrier()
    if global_rank == 0:
        final_path = dir_path / f"{dataset_name}.jsonl"
        logger.info(f"Writing dataset with scores at {final_path}")
        results_path = list(write_path.glob("*.json"))
        results_path.sort()

        alldata = []
        for path in results_path:
            with open(path, "r") as f:
                data = json.load(f)
            alldata.extend(data)
            path.unlink()
        with open(final_path, "w") as fout:
            for ex in alldata:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")
        write_path.rmdir()

def barrier():
    if dist.is_initialized():
        torch.distributed.barrier()