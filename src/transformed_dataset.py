from torch.utils import data as data


class TransformedDataset(data.Dataset):
    """
    Transforms a dataset.

    Arguments:
        dataset (Dataset): The whole Dataset
        transformer (LambdaType): (idx, sample) -> transformed_sample
    """

    def __init__(self, dataset, *, transformer=None, vision_transformer=None):
        self.dataset = dataset
        assert not transformer or not vision_transformer
        if transformer:
            self.transformer = transformer
        else:
            try:
                # you are dealing with RSNA dataset
                self.transformer = lambda _, data_label: (vision_transformer(data_label['image']), data_label['target'])
            except:
                self.transformer = lambda _, data_label: (vision_transformer(data_label[0]), data_label[1])
            # finally:
            #     raise ValueError("Something error happened in transforming dataset!")

    def __getitem__(self, idx):
        return self.transformer(idx, self.dataset[idx])

    def __len__(self):
        return len(self.dataset)
