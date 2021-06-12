"""Example on how to freeze certain layers while training (in pytorch lightning)
Author: Mohit Mayank
"""

# Before defining the optimizer, we need to freeze the layers
# In pytorch lightning, as optimizer is defined in configure_optimizers, we freeze layers there.
def configure_optimizers(self):
    # iterate through the layers and freeze the one with certain name (here all BERT models)
    # note: the name of layer depends on the varibale name
    for name, param in self.named_parameters():
        if 'BERTModel' in name:
            param.requires_grad = False
    # only pass the non-frozen paramters to optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    # return optimizer
    return optimizer
