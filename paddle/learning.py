import torch


def test(test_loader, model, device):
    """
    Test the model on the test data set provided by the test loader.

    :param test_loader:     The provided test data set
    :param model:           The model that should be tested
    :return: The percentage of correctly classified samples from the data set.
    """
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for test_features, test_labels in test_loader:
            test_features = test_features.to(device)
            test_labels = test_labels.to(device)

            # TODO remove flattening and change MaskedDAN-networks to work with channels
            #test_features = torch.flatten(test_features, start_dim=1)

            outputs = model(test_features)
            _, predicted = torch.max(outputs.data, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()
        return 100 * correct / total


def train(train_loader, model, optimizer, criterion, device):
    """
    Train the model on the train data set with a loss function and and optimization algorithm.

    :param train_loader:    The training data set.
    :param model:           The to be trained model.
    :param optimizer:       The used optimizer for the learning process.
    :param criterion:       The loss function of the network.
    :param percentage:      If the function should also calculate the percentage of right made decisions.
    :return: The average loss of the network in this epoch.
    """
    total_step = len(train_loader)
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for i, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)

        # TODO remove flattening and change MaskedDAN-networks to work with channels
        #features = torch.flatten(features, start_dim=1)

        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader), 100 * correct / total


def cross_validation_error(data_set, model, criterion):
    """
    Calculate the loss of the network on the cross-validation dataset.

    :param data_set:   The loader of the cross validation dataset.
    :param model:      The model on which the loss should be calculated on.
    :param criterion:  The loss function that should be used with the model.
    :return: The overall loss of the network.
    """

    loss = 0
    for (images, labels) in data_set:
        images = images.reshape(-1, 28 * 28)
        outputs = model(images)
        loss = criterion(outputs, labels)

    return loss
