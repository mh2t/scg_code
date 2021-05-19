import torch
from utils import get_data, Model, train_model, test_model
from tqdm import tqdm

data_path = "../data/real_regression_data"
out_path = "../outputs/real_regression_data"

NUM_LABELS = 4
HIDDEN_SIZE = 1024
NUM_LAYERS = 6
NUM_EPOCHS = 1000
LR = 1e-4
WD = 1e-7
P = 0.6


if __name__ == "__main__":
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
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

    train_y = train_y[:, 3].view(len(train_y), 1)
    test_y = test_y[:, 3].view(len(test_y), 1)
    test_Y = test_Y[:, 3].view(len(test_Y), 1)
    min_v = min_v[3]
    max_v = max_v[3]

    model = Model(
        input_size=train_x.shape[1],
        num_labels=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        p=P).to(device)
    model.load_state_dict(torch.load(f"{out_path}/student_model_d.pt"))

    criterion = torch.nn.L1Loss()
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = None

    for i in tqdm(range(NUM_EPOCHS), total=NUM_EPOCHS):
        train_loss = train_model(model, optim, criterion, train_x, train_y,
                                 scheduler=scheduler)
        if (i+1) % 100 == 0:
            test_losses = test_model(model, test_x, test_Y, min_v, max_v)
            tqdm.write(f"Epoch {i+1}/{NUM_EPOCHS}")
            tqdm.write(f"Train loss = {train_loss:.3f}")
            tqdm.write(f"MAE of D = {test_losses[0]:.3f}")
            tqdm.write("")

    model = model.to("cpu")
    torch.save(model.state_dict(), f"{out_path}/model_d.pt")
