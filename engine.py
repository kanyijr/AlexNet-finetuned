import torchvision, torch, tqdm
from torchvision.datasets import ImageFolder
from torch import nn
from tqdm.auto import tqdm



def train_step(model:nn.Module,
               optimizer:torch.optim.Optimizer,
               loss_fn:nn.Module,
               dataset,
               device:torch.device):
    
    # setting the model to train mode
    model= model.to(device)
    model.train()

    train_acc, train_loss = 0, 0

    for batch,(X, y)in enumerate(dataset):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss

        y_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=-1)
        acc = (torch.eq(y_pred, y).sum().item()) / len(y_pred)
        train_acc += acc

        # setting the optimizer to zero grad to reset the gradients calculated from previous batches
        optimizer.zero_grad()

        # calculating the gradients
        loss.backward()

        # performing the step
        optimizer.step()

    
    return train_loss/len(dataset), train_acc/len(dataset) 



# defining the evaluation step
def test_step(model:nn.Module,
              loss_fn:nn.Module,
              dataset, 
              device:torch.device):
    
    model = model.to(device)
    # setting it to evaluation mode
    model.eval()
    test_acc, test_loss = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataset):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            test_loss += loss

            y_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=-1)
            acc = (torch.eq(y_pred, y).sum().item()) / len(y_pred)
            test_acc += acc
    
    return test_loss/len(dataset), test_acc/len(dataset) 




# the wrapper function



# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          device:str,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5
          ):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataset=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer, 
                                           device=device)
        test_loss, test_acc = test_step(model=model,
            dataset=test_dataloader,
            loss_fn=loss_fn,
            device=device)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results


    