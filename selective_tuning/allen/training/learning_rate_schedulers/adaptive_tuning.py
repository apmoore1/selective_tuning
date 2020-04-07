from overrides import overrides
import torch

from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler

@LearningRateScheduler.register("adaptive_tuning")
class AdaptiveTuning(LearningRateScheduler):
    """
    
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer) -> None:
        super().__init__(optimizer)

    @overrides
    def step(self, metric: float = None, epoch: int = None) -> None:
        pass

    def step_batch(self, batch_num_total: int = None) -> None:
        # This is where we can change the require grad values I think
        count = 0
        for i, param_group in enumerate(reversed(self.optimizer.param_groups)):
            for param in param_group["params"]:
                grads = param.grad
                var = None
                if hasattr(param, 'variance'):
                    count += 1
                    var = param.variance
                grads
                # i = 0 is the default group; we care about i > 0
                #param.requires_grad = bool(i <= num_layers_to_unfreeze)
        print(count)
    @overrides
    def get_values(self):
        return self.base_values