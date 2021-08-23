# %%

from collections import OrderedDict
import dataloaders.base
from dataloaders.datasetGen import SplitGen, PermutedGen


train_dataset, val_dataset = dataloaders.base.__dict__["CIFAR10"]('data', False)

# %%

print(train_dataset)
# %%
train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                          first_split_sz=2,
                                                                          other_split_sz=2,
                                                                          rand_split=False,
                                                                          remap_class=not False)
# %%

for X, y , cl in train_dataset_splits["4"]:
    print(X.shape)
    print("cl")
    print(y)
    break
# %%

# %%
