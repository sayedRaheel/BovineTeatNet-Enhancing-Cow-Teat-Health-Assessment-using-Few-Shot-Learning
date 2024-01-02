import torch
from tqdm import tqdm

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    iou = 0
    dice = 0
    loss = 0
    # tqdm._instances.clear()
    bar = tqdm(val_loader, desc=f"Val Progress -  iou: {iou:.3f}, dice: {dice:.3f},loss:{loss:.5f}")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(bar):
            bar.set_description(f"Val Progress -  iou: {iou:.3f}, dice: {dice:.3f},loss:{loss:.5f}")
            # print("\r val {:2}%".format((i+1)/len(val_loader)),end="")
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            # One-hot encode the labels

            outputs = model(inputs)
            # print(torch.unique(torch.argmax(outputs, dim=1)))
            # print(torch.unique(labels))
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Compute IoU and dice scores
            predicted = torch.argmax(outputs, dim=1)
            # labels = labels.squeeze(1)
            union = torch.logical_or(predicted, labels).sum()
            intersection = torch.logical_and(predicted, labels).sum()
            if union == 0 :
                iou = 1
                dice = 1
            else:
                iou = intersection / union
                dice = iou*2/(iou+1)
            running_iou += iou
            running_dice += dice
    epoch_loss = running_loss / len(val_loader)
    epoch_iou = running_iou / len(val_loader)
    epoch_dice = running_dice / len(val_loader)
    
    # print(f'Validation Loss: {epoch_loss:.4f} | IoU: {epoch_iou:.4f} | Dice: {epoch_dice:.4f}')
    # print(1)
    return epoch_loss,epoch_dice,epoch_iou




def train(model, train_loader, valid_loader, optimizer, criterion, device, cfg,num_epochs=3,start_epoch=0):
    _, _, best_iou = validate(model, valid_loader, criterion, device)
    for epoch in range(start_epoch, start_epoch+num_epochs):
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        iou = 0
        dice = 0
        loss = 0
        bar = tqdm(train_loader, desc=f"Training Progress - Epoch: {epoch+1}/{start_epoch+num_epochs}, iou: {iou:.3f}, dice: {dice:.3f}")
        # bar = tqdm(valid_loader, desc=f"Val Progress -  iou: {iou:.3f}, dice: {dice:.3f},loss:{loss:.5f}")
        for i, (inputs, labels) in enumerate(bar):
            bar.set_description(f"Training Progress - Epoch: {epoch+1}/{start_epoch+num_epochs}, iou: {iou:.3f}, dice: {dice:.3f},loss:{loss:.5f}")
            # bar.set_description(f"Val Progress -  iou: {iou:.3f}, dice: {dice:.3f},loss:{loss:.5f}")
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            union = torch.logical_or(predicted, labels).sum()
            intersection = torch.logical_and(predicted, labels).sum()
            if union == 0 :
                iou = 1
                dice = 1
            else:
                iou = intersection / union
                dice = iou*2/(iou+1)
            # dice = (2 * intersection / (predicted_onehot.sum(dim=(1,2)) + labels_onehot.sum(dim=(1,2)))).mean().item()

            running_iou += iou
            running_dice += dice
        
        epoch_loss = running_loss / len(train_loader)
        epoch_iou = running_iou / len(train_loader)
        epoch_dice = running_dice / len(train_loader)
        val_loss,val_dice,val_iou = validate(model, valid_loader, criterion, device)
        print(f'Epoch: {epoch+1}/{num_epochs}')
        print(f'Training   Loss: {epoch_loss:.4f} | IoU: {epoch_iou:.4f} | Dice: {epoch_dice:.4f}')
        print(f'Validation Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}')
        
        if val_iou > best_iou or val_iou > 0.5:
            model_path= cfg.model_save_dir+'/epoch_'+str(epoch)+'iou_'+str(val_iou.item())+'.pth'
            save_model(model, model_path)
            print('model saved')
            best_iou = val_iou
        print()

def evaluate(test_loader,valid_loader, model, criterion, device):
    test_loss,test_dice,test_iou= validate(model, test_loader, criterion, device)
    val_loss,val_dice,val_iou= validate(model, valid_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f} | IoU: {test_iou:.4f} | Dice: {test_dice:.4f}')
    print(f'Validation Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}')