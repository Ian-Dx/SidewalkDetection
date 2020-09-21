""" The training step """
from nnModel.DenseASPP import DenseASPP
from processor import *
from torch.utils.data import TensorDataset, DataLoader

# Hyper parameters
DATA_PATH = "./resources/data/"
TARGET_PATH = "./resources/target/"
EPOCHS = 10


def train():
    # Get data and target
    data_tensor = data_reader(DATA_PATH)
    target_tensor = target_reader(TARGET_PATH)

    # Generate dataset and data loader
    dataset = TensorDataset(data_tensor, target_tensor)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=5,
        shuffle=True
    )

    # DenseASPP model
    model = DenseASPP()

    # Optimizer and criteria
    opt = torch.optim.Adam(model.parameters(), 0.05, (0.99, 0.9))
    criteria = torch.nn.CrossEntropyLoss()

    # Log
    print("-" * 10 + "Data loaded!" + "-" * 10)

    for epoch in range(EPOCHS):
        for idx, data, target in enumerate(data_loader):
            # Predict to get the output
            output = model(data)
            # Gradient zero
            opt.zero_grad()
            # Back propagation
            loss = criteria(output, target)
            # Gradient decent
            loss.backward()
            # Forward propagation
            opt.step()
            # Log
            print("Epoch: {}/{} | Index: {:04} | Loss: {.4}"
                  .format(epoch + 1, EPOCHS, idx, loss.detach()))


if __name__ == '__main__':
    train()
