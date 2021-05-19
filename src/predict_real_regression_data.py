import torch
from utils import get_data, Model, test_model

data_path = "../data/real_regression_data"
out_path = "../outputs/real_regression_data"

NUM_LABELS = 4
HIDDEN_SIZE = 1024
NUM_LAYERS = 6


if __name__ == "__main__":
    device = torch.device('cpu')

    test_x, test_y, test_Y, min_v, max_v = get_data(
        data_path=data_path,
        out_path=out_path,
        name="real_test_truesorted",
        load_values=True,
        device=device,
        num_labels=NUM_LABELS,
        return_extra=True,
        drop_extra=2)

    models = [Model(
        input_size=test_x.shape[1],
        num_labels=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS) for _ in range(4)]

    models[0].load_state_dict(torch.load(f"{out_path}/model_h.pt"))
    models[1].load_state_dict(torch.load(f"{out_path}/model_r.pt"))
    models[2].load_state_dict(torch.load(f"{out_path}/model_s.pt"))
    models[3].load_state_dict(torch.load(f"{out_path}/model_d.pt"))

    models = [x.to(device).eval() for x in models]

    loss_h = test_model(models[0], test_x, test_Y[:, 0].view(len(test_Y), 1),
                        min_v[0], max_v[0])[0]
    loss_r = test_model(models[1], test_x, test_Y[:, 1].view(len(test_Y), 1),
                        min_v[1], max_v[1])[0]
    loss_s = test_model(models[2], test_x, test_Y[:, 2].view(len(test_Y), 1),
                        min_v[2], max_v[2])[0]
    loss_d = test_model(models[3], test_x, test_Y[:, 3].view(len(test_Y), 1),
                        min_v[3], max_v[3])[0]

    print(f"MAE of H = {loss_h:.3f}")
    print(f"MAE of R = {loss_r:.3f}")
    print(f"MAE of S = {loss_s:.3f}")
    print(f"MAE of D = {loss_d:.3f}")
