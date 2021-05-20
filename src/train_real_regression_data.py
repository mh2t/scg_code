import torch
from utils import get_data, Model, train_model, test_model
from tqdm import tqdm
from copy import deepcopy


data_path = "../data/real_regression_data"
out_path = "../outputs/real_regression_data"

NUM_LABELS = 4
HIDDEN_SIZE = 1024
NUM_LAYERS = 6
NUM_EPOCHS = 1000
WD = 1e-7
P = 0.6

label_map = {
    0: ['h', True, 1e-4],
    1: ['r', False, 1e-2],
    2: ['s', True, 1e-4],
    3: ['d', True, 1e-4],
}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Get data
train_x, train_y = get_data(
    data_path=data_path,
    out_path=out_path,
    name="real_train_truesorted",
    load_values=False,
    device=device,
    num_labels=NUM_LABELS,
    return_extra=False,
    drop_extra=2)
test_x, test_y, test_Y, min_v, max_v = get_data(
    data_path=data_path,
    out_path=out_path,
    name="real_test_truesorted",
    load_values=True,
    device=device,
    num_labels=NUM_LABELS,
    return_extra=True,
    drop_extra=2)


def func(label_id: int):
    label = label_map[label_id][0]
    use_student_model = label_map[label_id][1]
    lr = label_map[label_id][2]

    # Get data
    y = train_y[:, label_id].view(len(train_y), 1)
    Y = test_Y[:, label_id].view(len(test_Y), 1)
    min_val = min_v[label_id]
    max_val = max_v[label_id]

    # Get model
    model = Model(
        input_size=train_x.shape[1],
        num_labels=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        p=P).to(device)

    label = label_map[label_id][0]
    student_path = f"{out_path}/student_model_{label}.pt"
    save_path = f"{out_path}/model_{label}.pt"

    if use_student_model:
        model.load_state_dict(torch.load(student_path))

    criterion = torch.nn.L1Loss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WD)

    best_metric = 1e6
    best_model = deepcopy(model)
    best_model = best_model.to('cpu')

    print(f"Started training model for \"{label}\"")
    for i in tqdm(range(NUM_EPOCHS), total=NUM_EPOCHS):
        train_loss = train_model(model, optim, criterion, train_x, y)

        if (i+1) % 100 == 0:
            test_losses = test_model(model, test_x, Y, min_val, max_val)
            test_loss = test_losses[0].item()

            if test_loss < best_metric:
                best_metric = test_loss
                best_model = deepcopy(model)
                best_model = best_model.to('cpu')

            tqdm.write(f"Epoch {i+1}/{NUM_EPOCHS}")
            tqdm.write(f"Train loss = {train_loss:.3f}")
            tqdm.write(f"MAE of {label} = {test_loss:.3f}")
            tqdm.write("")

    torch.save(best_model.state_dict(), save_path)


if __name__ == "__main__":
    for i in range(4):
        func(i)
