import numpy as np
import torch
import torch.nn as nn


def get_x_y(arr, num_labels):
    x = arr[:, :-num_labels]
    y = arr[:, -num_labels:]
    return x, y


def save_numpy_object(arr, path, name):
    with open(f"{path}/{name}.npy", "wb") as f:
        np.save(f, arr)


def load_numpy_object(path, name):
    with open(f"{path}/{name}.npy", "rb") as f:
        arr = np.load(f)
    return arr


def normalize(arr, path, load_values=False):
    if load_values is False:
        max_v = np.max(arr, axis=0).reshape(1, arr.shape[1])
        min_v = np.min(arr, axis=0).reshape(1, arr.shape[1])

        save_numpy_object(max_v, path, "normalize_max_values")
        save_numpy_object(min_v, path, "normalize_min_values")
    else:
        max_v = load_numpy_object(path, "normalize_max_values")
        min_v = load_numpy_object(path, "normalize_min_values")

    arr = np.divide(arr - min_v, max_v - min_v)

    return arr, min_v, max_v


def get_data(data_path, out_path, name, load_values, device, num_labels,
             return_extra, drop_last=False, drop_extra=0):
    arr = np.load(f"{data_path}/{name}.npy")

    if drop_extra != 0:
        arr = np.concatenate((arr[:, :-num_labels-drop_extra],
                             arr[:, -num_labels:]), axis=1)

    _, Y = get_x_y(arr, num_labels)

    if drop_last:
        arr = arr[:, :-1]

    arr, min_v, max_v = normalize(arr, out_path, load_values)
    arr_x, arr_y = get_x_y(arr, num_labels)

    arr_x = torch.from_numpy(arr_x).to(dtype=torch.float32, device=device)
    arr_y = torch.from_numpy(arr_y).to(dtype=torch.float32, device=device)

    Y = torch.from_numpy(Y).to(dtype=torch.float32, device=device)
    min_v = torch.from_numpy(min_v).to(dtype=torch.float32, device=device)
    max_v = torch.from_numpy(max_v).to(dtype=torch.float32, device=device)

    min_v = min_v[0, -num_labels:]
    max_v = max_v[0, -num_labels:]

    if return_extra:
        return arr_x, arr_y, Y, min_v, max_v
    else:
        return arr_x, arr_y


class Model(nn.Module):
    def __init__(self, input_size, num_labels, hidden_size=1024,
                 num_layers=4, p=0.5):
        super().__init__()
        layers = []
        for i in range(num_layers-1):
            in_size = input_size if i == 0 else hidden_size
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.Dropout(p=p))
        layers.append(nn.Linear(hidden_size, num_labels))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


def train_model(model, optim, criterion, x, y, scheduler=None):
    model.train()
    optim.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optim.step()
    if scheduler is not None:
        try:
            scheduler.step()
        except TypeError:
            try:
                scheduler.step(loss)
            except TypeError:
                pass
    return loss.item()


@torch.no_grad()
def test_model(model, x, y, min_v, max_v):
    model.eval()
    out = model(x)
    out = out * (max_v - min_v) + min_v
    losses = torch.mean(torch.abs(out - y), axis=0)
    return losses
