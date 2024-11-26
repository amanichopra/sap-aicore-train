import torch.nn as nn
import torch
import datetime
from ai_core_sdk.models import Metric, MetricTag, MetricCustomInfo, MetricLabel
from time import perf_counter

class Classifier(nn.Module):
  def __init__(self, dropout=0.5, activation='relu'):
    super(Classifier, self).__init__()
    self.linear1 = nn.Linear(34, 128)
    self.linear2 = nn.Linear(128, 64)
    self.linear3 = nn.Linear(64, 82)
    self.dropout = nn.Dropout(dropout)
    if activation == 'relu':
      self.act = torch.relu
    elif activation == 'sigmoid':
      self.act = torch.sigmoid
    elif activation == 'tanh':
      self.act = torch.tanh
  def forward(self, x):
    x = self.act(self.linear1(x))
    x = self.dropout(x)
    x = self.act(self.linear2(x))
    x = self.dropout(x)
    x = self.linear3(x)
    return x

class PoseEmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = {'embedding': self.X[idx], 'pose': self.y[idx]}
        return sample

def get_dataloader(X, y, batch_size):
  dataset = PoseEmbeddingsDataset(X, y)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
  return dataloader

def build_opt(model, opt, lr):
  if opt == "sgd":
    optimizer = torch.optim.SGD(model.parameters(),
                              lr=lr, momentum=0.9)
  elif opt == "adam":
    optimizer = torch.optim.Adam(model.parameters(),
                               lr=lr)
  return optimizer

def train_epoch(model, loader, optimizer, device):
  model.train()
  ce_loss = torch.nn.CrossEntropyLoss()
  cumu_loss = 0
  correct, total = 0, 0
  for i, batch in enumerate(loader):
    e, labels = batch['embedding'].to(device, dtype=torch.float32), batch['pose'].to(device, dtype=torch.float32)
    optimizer.zero_grad()

    # ➡ Forward pass
    outputs = model(e).to(device)
    loss = ce_loss(outputs, labels)
    cumu_loss += loss.item()

    # ⬅ Backward pass + weight update
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(outputs.data, 1)
    _, labels = torch.max(labels, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

  return cumu_loss / len(loader), correct / total

def test_epoch(model, loader, device):
  model.eval()
  ce_loss = torch.nn.CrossEntropyLoss()
  with torch.no_grad():
    cumu_loss = 0
    correct, total = 0, 0
    for i, batch in enumerate(loader):
      e, labels = batch['embedding'].to(device, dtype=torch.float32), batch['pose'].to(device, dtype=torch.float32)

      outputs = model(e)
      loss = ce_loss(outputs, labels)
      cumu_loss += loss.item()

      _, predicted = torch.max(outputs.data, 1)
      _, labels = torch.max(labels, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

    return cumu_loss / len(loader), correct / total

def train(X_train, y_train_dummy, X_val, y_val_dummy, device, epochs, dropout, activation, opt, lr, dl_batch_size, aic_connection=None):
  train_loader = get_dataloader(X_train, y_train_dummy, dl_batch_size)
  val_loader = get_dataloader(X_val, y_val_dummy, dl_batch_size)
  model = Classifier(dropout, activation).to(device)
  optimizer = build_opt(model, opt, lr)
  
  for epoch in range(epochs):
    start_train = perf_counter()
    train_avg_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
    end_train = perf_counter()
    start_test = perf_counter()
    test_avg_loss, test_acc = test_epoch(model, val_loader, device)
    end_test = perf_counter()
    if aic_connection:
      aic_connection.log_metrics(
          metrics = [
              Metric(name="epoch", value=epoch, timestamp=datetime.datetime.now(datetime.timezone.utc), step=epoch),
              Metric(name="train_time", value=end_train - start_train, timestamp=datetime.datetime.now(datetime.timezone.utc), step=epoch),
              Metric(name="test_time", value=end_test - start_test, timestamp=datetime.datetime.now(datetime.timezone.utc), step=epoch),
              Metric(name="train_epoch_loss", value=train_avg_loss, timestamp=datetime.datetime.now(datetime.timezone.utc), step=epoch),
              Metric(name="test_epoch_loss", value=test_avg_loss, timestamp=datetime.datetime.now(datetime.timezone.utc), step=epoch),
              Metric(name="train_epoch_acc", value=train_acc, timestamp=datetime.datetime.now(datetime.timezone.utc), step=epoch),
              Metric(name="test_epoch_acc", value=test_acc, timestamp=datetime.datetime.now(datetime.timezone.utc), step=epoch),
          ]
      )
    print({"epoch": epoch, "train_time": end_train - start_train, "inference_time": end_test - start_test, "train_epoch_loss": train_avg_loss, "test_epoch_loss": test_avg_loss, "train_epoch_acc": train_acc, "test_epoch_acc": test_acc})

  return model
    