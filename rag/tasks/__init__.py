from . import base, qa

AVAILABLE_TASKS = {m.__name__.split(".")[-1]: m for m in [base, qa]}


def get_task(opt, tokenizer):
    if opt.task not in AVAILABLE_TASKS:
        raise ValueError(f"{opt.task} not recognised")
    task_module = AVAILABLE_TASKS[opt.task]
    return task_module.Task(opt, tokenizer)