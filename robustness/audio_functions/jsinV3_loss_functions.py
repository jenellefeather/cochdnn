import torch as ch


class jsinV3_multi_task_loss(ch.nn.Module):
    ''' 
    Creates a loss function for the jsinV3 dataset, combining the loss across tasks. 
  
    Args:
        task_loss_params (dict): dictionary of task-specific loss function parameters. 
            Keys are the task names, fields are dicts. 
                TASK_LOSS_PARAMS['stimuli/word_int'] = {
                    'loss_type': 'crossentropyloss',
                    'weight': 1.0,
                }
    
    Returns
    -------
    batch_loss (ch.tensor): loss for each element of the batch
    '''
    def __init__(self, task_loss_params, batch_size, reduction='mean'):
        super(jsinV3_multi_task_loss, self).__init__()
        # Supported loss functions ("a" refers to logits or activations, "b" refers to labels)
        loss_functions = {
            'bcewithlogitsloss': ch.nn.BCEWithLogitsLoss, 
            'crossentropyloss': ch.nn.CrossEntropyLoss
        }
        all_loss_functions = {}
        all_loss_weights = {}
        for task, params in task_loss_params.items():
            all_loss_functions[task] = loss_functions[params['loss_type']](
                                           reduction=reduction)
            all_loss_weights[task] = params['weight']
      
        self.all_loss_weights = all_loss_weights
        self.all_loss_functions = all_loss_functions
        self.tasks = list(task_loss_params.keys())
        self.reduction=reduction
        self.batch_size = batch_size
 
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def forward(self, output, target):
        """
        Computes the loss given the model output and the target

        Args: 
            output (dict): dictionary containing the model outputs. Keys should match with
                self.tasks
            target (dict): element from the torch dataset containing the target values. 
                Keys should match with self.tasks
        Returns: 
            loss (ch.tensor): loss
        """
        if self.reduction=='none':
            # We want to reduce across the task -- important for multitask classification
            not_reduced_losses = [self.all_loss_weights[task] * self.all_loss_functions[task](
                                 output[task], target[task]) for task in self.tasks]
            loss_list = [not_reduced.view(not_reduced.shape[0], -1).sum(1) for not_reduced \
                           in not_reduced_losses]
            loss = ch.stack(loss_list, dim=1).sum(dim=1)
        else: # the losses have been collapsed across the batch
            loss_list = [self.all_loss_weights[task] * self.all_loss_functions[task](
                         output[task], target[task]) for task in self.tasks]
            loss = ch.stack(loss_list).sum(dim=0)
        return loss
        
