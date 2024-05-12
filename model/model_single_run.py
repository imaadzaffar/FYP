import wandb
import os
import multiprocessing
import collections


Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("num", "sweep_id", "sweep_run_name", "train_loader", "val_loader", "test_loader", "config")
)
WorkerDoneData = collections.namedtuple("WorkerDoneData", ("mean_diff", "std_diff"))


import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from utils.custom_dataset import CustomDataset
from model_pipeline import set_seed, make, make_loader, train, test

DATA_DIR = "/cs/student/projects1/2021/izaffar/FYP/FYP-MUL/data"
# for filename in os.listdir(DATA_DIR):
#     print(filename)

SEED = 42


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]

# CUDA initialization function
def initialize_cuda():
    # Use GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(device)
    return device

def train_model(sweep_q, worker_q):
    # Initialize CUDA if available
    device = initialize_cuda()

    reset_wandb_env()
    worker_data = worker_q.get()
    run_name = "{}-{}".format(worker_data.sweep_run_name, worker_data.num)
    config = worker_data.config
    run = wandb.init(
        project="fyp-mul",
        group=worker_data.sweep_id,
        job_type=worker_data.sweep_run_name,
        name=run_name,
        config=config,
    )

    # MODEL
    set_seed(SEED)

    model, criterion, optimizer = make(config, device)
    # print(model)

    # Train the model
    train(model, worker_data.train_loader, worker_data.val_loader, criterion, optimizer, run, config, device)

    # and test its final performance
    mean_diff, std_diff = test(model, worker_data.test_loader, run, config, device)
    # run.log(dict(mean_diff=mean_diff, std_diff=std_diff))

    wandb.join()
    sweep_q.put(WorkerDoneData(mean_diff=mean_diff, std_diff=std_diff))


def main(config):
    # Spin up workers before calling wandb.init()
    # Workers will be blocked on a queue waiting to start
    sweep_q = multiprocessing.Queue()
    workers = []
    for fold_num in range(config["k_folds"]):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=train_model, kwargs=dict(sweep_q=sweep_q, worker_q=q)
        )
        p.start()
        workers.append(Worker(queue=q, process=p))

    sweep_run = wandb.init(project="fyp-mul")
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown"

    # make the model, data, and optimization problem
    dataset = CustomDataset(data_dir=DATA_DIR, model_arch=config["model_arch"], mask_type=config["mask_type"], transform=True)

    # split off test set
    train_val_indices, test_indices = train_test_split(
        list(range(len(dataset))), test_size=(config["test_size"]/100), random_state=SEED
    )
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    test_loader = make_loader(test_dataset, batch_size=config["batch_size"])

    # KFold Cross-validation
    kfold = KFold(n_splits=config["k_folds"], shuffle=True, random_state=SEED)

    metrics = {"mean_diff": [], "std_diff": []}
    for fold_num, (train_indices, val_indices) in enumerate(kfold.split(range(len(train_val_indices)))):
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        train_loader = make_loader(train_dataset, batch_size=config["batch_size"])
        val_loader = make_loader(val_dataset, batch_size=config["batch_size"])

        worker = workers[fold_num]
        # start worker
        worker.queue.put(
            WorkerInitData(
                sweep_id=sweep_id,
                num=fold_num,
                sweep_run_name=sweep_run_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config=dict(config),
            )
        )
        # get metric from worker
        result = sweep_q.get()
        # wait for worker to finish
        worker.process.join()
        # log metric to sweep_run
        metrics["mean_diff"].append(result.mean_diff)
        metrics["std_diff"].append(result.std_diff)

    sweep_run.log(dict(mean_diff=sum(metrics["mean_diff"]) / len(metrics["mean_diff"]), std_diff=sum(metrics["std_diff"]) / len(metrics["std_diff"])))
    wandb.join()

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)


if __name__ == "__main__":
    config = {
        "model_output": "length",
        "model_arch": "brain_mri",
        "mask_type": "right",
        "mask_threshold": 0.1,
        "criterion": "MSE",
        # "criterion_mask": "Dice",
        # "criterion_length": "MSE",
        # "criterion_alpha": 1,
        # "criterion_threshold_epochs": 0,
        # "criterion_threshold_length_loss": 0,
        "batch_size": 4,
        "k_folds": 5,
        "val_size": 20,
        "test_size": 20,
        "epochs": 100,
        "learning_rate": 3e-5,
    }
    main(config)
