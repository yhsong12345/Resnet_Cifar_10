import torch
import matplotlib.pyplot as plt
import os
import pandas as pd


# current_path = os.getcwd()

# def make_folder(m):

#     path = current_path + '/' + m
#     if path.exist() == False:
#         os.mkdir(f'./{path}')
#         newpath = path
#     else:
#         n = (len([entry for entry in os.listdir(path) 
#                   if os.path.isfile(os.path.join(path, entry))]))
#         newpath = path + n
#         os.mkdir(f'{newpath}')
        
#     path1 = newpath + '/train'

#     os.mkdir(f'./{path1}')

#     path1_1 = path1 + '/accuracy'
#     path1_2 = path1 + '/loss'

#     os.mkdir(f'./{path1_1}')
#     os.mkdir(f'./{path1_2}')
        
#     path2 = newpath + '/val'

#     os.mkdir(f'./{path2}')

#     path2_1 = path2 + '/accuracy'
#     path2_2 = path2 + '/loss'

#     os.mkdir(f'./{path2_1}')
#     os.mkdir(f'./{path2_2}')

#     path3 = newpath + 'plot'
#     os.mkdir(f'./{path3}')


            

plt.style.use('ggplot')


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, m, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        path = f'./outputs/{m}'
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f'{path}/best_model.pt')


def save_model(m, epoch, model, optimizer, criterion):
    print(f'Saving final model...')
    path = f'./outputs/{m}'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        }, f'{path}/model{epoch}.pt')



def save_data(m, train_acc, valid_acc, train_loss, valid_loss):
    print("Saving losses and accuracies")
    path = f'./outputs/{m}'
    data = {'train_accuracy': train_acc, 'valid_accuracy':valid_acc,
            'train_loss': train_loss, 'valid_loss': valid_loss}
    df = pd.DataFrame(data=data)
    df.to_excel(f'{path}/{m}result.xlsx')

    


def save_plots(m,train_acc, valid_acc, train_loss, valid_loss):
    path = f'./outputs/{m}'
    plt.figure(figsize=(10,7))
    plt.plot(
        train_acc, color='red', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validation accuracy'
    )

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{path}/accuracy.png')


    plt.figure(figsize=(10,7))
    plt.plot(
        train_loss, color='green', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='orange', linestyle='-',
        label='validation loss'
    )

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{path}/loss.png')


# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res


# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     """
#     Save the training model
#     """
#     torch.save(state, filename)