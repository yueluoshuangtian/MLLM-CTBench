class BaseCLearner:
    def before_train(self, task_id, model, tokenizer=None, **kwargs):
        pass

    def after_train(self, task_id, model, tokenizer=None, **kwargs):
        pass

    def loss(self, loss, **kwargs):
        return loss
