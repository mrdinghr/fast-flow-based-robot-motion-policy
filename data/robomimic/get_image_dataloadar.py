import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.file_utils as FileUtils
from torch.utils.data import DataLoader
from robomimic.config import config_factory
import robomimic.utils.obs_utils as ObsUtils


# Robomimic dataloadar with vision observation
def get_robomimic_image_dataloadar(cfg, path=None):
    '''
    this function is used for getting train & validation dataloadar of Robomimic tasks with vision-based observation
    '''
    config = config_factory(algo_name='bc')
    config.train.data = cfg.dataset_path + cfg.task + '/ph/image_v141.hdf5'
    config.train.batch_size = cfg.optim.batch_size
    config.experiment.validate = True
    config.train.hdf5_filter_key = "train"
    config.train.hdf5_validation_filter_key = "valid"
    config.train.frame_stack = cfg.n_ref
    config.train.seq_length = cfg.n_pred
    if cfg.task == 'tool_hang':
        config.observation.modalities.obs.rgb = ["sideview_image",
                    "robot0_eye_in_hand_image"]
    else:
        config.observation.modalities.obs.rgb = [
            "agentview_image", "robot0_eye_in_hand_image"
        ]

    config.observation.modalities.obs.low_dim = ["robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos"]
    ObsUtils.initialize_obs_utils_with_config(config)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True
    )
    num_workers = min(config.train.num_data_workers, 1)
    valid_sampler = validset.get_dataset_sampler()
    valid_loader = DataLoader(
        dataset=validset,
        sampler=valid_sampler,
        batch_size=config.train.batch_size,
        shuffle=(valid_sampler is None),
        num_workers=num_workers,
        drop_last=True
    )
    return train_loader, valid_loader


if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('refcond_rfm_robomimic.yaml')
    train_loader, valid_loader = get_robomimic_image_dataloadar(cfg)
    for batch in train_loader:
        print(batch['actions'].shape)
        print(batch['obs']['robot0_gripper_qpos'].shape)
        pass
