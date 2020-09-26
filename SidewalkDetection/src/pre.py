""" Check if this is working """
from torch.utils.data import TensorDataset, DataLoader
from nnModel.DenseASPP import DenseASPP
from processor import *
from html_gene import *
from torch.utils.tensorboard import SummaryWriter


# Hyper parameters

# ** Set dataset path **
TRAIN_DATA_PATH = "./resources/data"
TRAIN_TARGET_PATH = "./resources/target"
VALIDATION_DATA_PATH = ""
VALIDATION_TARGET_PATH = ""
TEST_DATA_PATH = ""
TEST_TARGET_PATH = ""

EPOCHS = 1


def train():
    """
    Training process
    :return:
    """
    # Time
    time1 = time.time()

    # Check gpu
    is_gpu = torch.cuda.is_available()

    # Get data and target
    train_data, _ = data_reader(TRAIN_DATA_PATH)
    train_target, _ = target_reader(TRAIN_TARGET_PATH)
    val_data, val_imgs_data = data_reader(VALIDATION_DATA_PATH)
    val_target, val_imgs_target = target_reader(VALIDATION_TARGET_PATH)

    # Gpu
    if is_gpu:
        train_data = train_data.cuda()
        train_target = train_target.cuda()
        val_data = val_data.cuda()
        val_target = val_target.cuda()

    # Generate dataset and data loader
    dataset = TensorDataset(train_data, train_target)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True
    )

    # DenseASPP model
    model = DenseASPP()

    # Optimizer and criteria
    opt = torch.optim.Adam(model.parameters(), 0.05, (0.99, 0.9))
    criteria = torch.nn.CrossEntropyLoss()

    # Log
    writer = SummaryWriter(log_dir="logs")
    print("-" * 10 + "Data loaded!" + "-" * 10)

    # Gpu
    if is_gpu:
        model = model.cuda()
        criteria = criteria.cuda()

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
            # Writer
            if idx % 10 == 0:
                writer.add_scalar("Train loss", loss, idx // 10 + epoch * 38)
                print("Loss {}".format(loss))

    # Save model
    if is_gpu:
        torch.save(model.modules, "DenseASPP_Model.pth")
    else:
        torch.save(model, "DenseASPP_Model")
    # Log
    print("-" * 10 + "Model saved!" + "-" * 10)

    # Get the output with probabilities
    val_output = model(val_data)
    # Get each the most likely one to be the predicted one
    val_predicted = torch.max(val_output, 1)[1].data.numpy()

    # Loss
    loss = criteria(val_output, val_target)
    # Calculate IoU
    ious = IoU(val_predicted, val_target).data.numpy()
    miou = np.mean(ious)
    # Calculate dsc
    dscs = np.array(list(map(lambda a: 2 * a / (a + 1), ious)))
    mdsc = np.mean(dscs)

    # Log
    print("mean IoU: {} | mean dsc: {}".format(miou, mdsc))
    # Time
    time2 = time.time()

    # Generate images part for html
    html_images = []
    for idx, (img_data, img_target, iou, dsc) in enumerate(zip(val_imgs_data, val_imgs_target, ious, dscs)):
        img = image_gene(idx, img_data, img_target, iou, dsc)
        html_images.append(img)

    # Generate html
    html_gene(
        "DenseASPP",
        2975,
        500,
        5.95,
        "https://github.com/Ian-Dx/SidewalkDetection/blob/master/"
        "references/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf",
        "DenseASPP for Semantic Segmentation in Street Scenes",
        "https://github.com/Ian-Dx/SidewalkDetection/tree/master/SidewalkDetection",
        miou,
        mdsc,
        loss,
        time2 - time1,
        html_images
    )


def IoU(output, target):
    """
    Calculate IoU
    :param output:
    :param target:
    :return:
    """
    # Smooth to avoid 0/0
    SMOOTH = 1e-6
    # Will be zero if Truth=0 or Prediction=0
    intersection = (output & target).float().sum((1, 2))
    # Will be zero if both are 0
    union = (output | target).float().sum((1, 2))
    # Smooth to avoid 0/0
    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou


if __name__ == '__main__':
    train()
