from dataset import get_dataset
from torchtrainer.models.medsam.medsam import get_model
from torchtrainer.train import DefaultTrainer


class MedSAMTrainer(DefaultTrainer):

    def get_dataset(
            self, 
            dataset_class, 
            dataset_path, 
            split_strategy, 
            resize_size, 
            augmentation_strategy,
            split_folder,
            **dataset_params):
        
        if dataset_class == "vessmap_color_bce":
            return get_dataset(
                dataset_path=dataset_path,
                split_strategy=split_strategy,
                resize_size=resize_size,
                split_folder=split_folder
            )

    def get_model(self, model_class, weights_strategy, num_classes, num_channels, freeze_encoder,
                  **model_params):
        if model_class == "medsam":
            model = get_model(freeze_image_encoder=freeze_encoder)

        return model

if __name__ == "__main__":
    # This is required to run the script from the command line
    MedSAMTrainer().fit()