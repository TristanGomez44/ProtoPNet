import os
import torch

def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print):
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))

def save_model_if_best(model, model_dir, model_name, accu, best_accu, log=print):
    '''
    model: this is not the multigpu model
    '''
    if accu > best_accu:
        log('\tbest {0:.2f}%'.format(accu * 100))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + 'best{}.pth'.format(accu))))

        oldPath = os.path.join(model_dir, (model_name + 'best{}.pth'.format(best_accu)))
        if os.path.exists(oldPath):
            os.remove(oldPath)
