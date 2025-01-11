import torch
import numpy as np
import argparse
import yaml
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error


def train(model_name, use_counts, model, device, loader, optimizer, mode):
    """
    Performs one training epoch, i.e. one optimization pass over the batches of a data loader.
    """

    loss_fn = torch.nn.BCELoss()  # for cut node classification

    curve = list()
    model.train()
    # for step, batch in enumerate(tqdm(loader, desc="Training iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)  # batch.cuda() if torch.cuda.is_available() else batch

        optimizer.zero_grad()

        if (
            model_name == "GINCutPred"
            or model_name == "GCNCutPred"
            or model_name == "GATCutPred"
        ):
            pred = model(
                x=batch.x,
                edge_index=batch.edge_index,
                counts=batch.counts,
                use_counts=use_counts,
                batch=batch.batch,
            )
        else:
            print("model not supported")

        targets = batch.y.to(torch.float32).view(
            pred.shape
        )  # y for now for cut node prediction

        loss = loss_fn(pred, targets)

        loss.backward()
        optimizer.step()
        curve.append(loss.detach().cpu().item())

    return curve


def eval(model_name, use_counts, model, device, loader, threshold=0.5, mode="node"):
    """
    Evaluates a model over all the batches of a data loader.
    Computes the average BCELoss and a custom accuracy measure:
      - For each graph, the 121-length prediction must match
        the 121-length label exactly (with thresholding).
    Returns:
      - accuracy: ratio of "completely correct" graphs to total graphs
      - mean_loss: average BCELoss over all graphs
    """

    # loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.BCELoss()
    model.eval()

    total_graphs = 0
    correct_graphs = 0
    losses = []

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if torch.get_default_dtype() == torch.float64:
                for dim in range(batch.dimension + 1):
                    batch.cochains[dim].x = batch.cochains[dim].x.double()
                    assert batch.cochains[dim].x.dtype == torch.float64

            batch = batch.to(device)

            if model_name in ["GINCutPred", "GCNCutPred", "GATCutPred"]:
                pred = model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    counts=batch.counts,
                    use_counts=use_counts,
                    batch=batch.batch,
                )
            else:
                raise ValueError("Model not supported")

            # pred should be of shape [B, max_edges or max_nodes]
            targets = batch.y.float().view(pred.shape)

            loss = loss_fn(pred, targets)
            losses.append(loss.detach().cpu().item())

            # Graph level accuracy computation
            pred_thresholded = (
                pred > threshold
            ).float()  # shape [B, max_nodes or max_edges]

            correct_per_entry = (
                pred_thresholded == targets
            )  # shape [B, max_nodes or max_edges]
            correct_entire_graph = correct_per_entry.all(dim=1)  # shape [B]
            correct_in_batch = correct_entire_graph.sum().item()

            batch_size = pred.shape[0]
            correct_graphs += correct_in_batch
            total_graphs += batch_size

    mean_loss = float(np.mean(losses)) if len(losses) > 0 else np.nan
    accuracy = correct_graphs / total_graphs if total_graphs > 0 else 0.0

    return accuracy, mean_loss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cut node/edge identification experiment"
    )
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-project", "--project")
    parser.add_argument("-group", "--group", help="group name on wandb", required=True)
    parser.add_argument("-seed", "--seed", help="seed", required=True)

    args, unparsed = parser.parse_known_args()

    with open(args.config, "r") as f:
        conf = yaml.safe_load(f)

    conf["project"] = args.project
    conf["group"] = args.group
    conf["seed"] = args.seed

    return conf
