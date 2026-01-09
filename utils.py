class CombinedOptimizer:
    def __init__(self, optimizers):
        self.optimizers = optimizers
        self._param_groups = None

    @property
    def param_groups(self):
        # Rebuild the list each time to get current values
        all_groups = []
        for optimizer in self.optimizers:
            all_groups.extend(optimizer.param_groups)
        return all_groups

    @param_groups.setter
    def param_groups(self, value):
        # This allows code that sets param_groups directly
        # Distribute back to individual optimizers
        idx = 0
        for optimizer in self.optimizers:
            n_groups = len(optimizer.param_groups)
            optimizer.param_groups = value[idx:idx + n_groups]
            idx += n_groups

    def step(self, closure=None):
        loss = None
        for optimizer in self.optimizers:
            loss = optimizer.step(closure)
        return loss

    def zero_grad(self, set_to_none=False):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts):
        for opt, state in zip(self.optimizers, state_dicts):
            opt.load_state_dict(state)