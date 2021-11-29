"""Tune Model.

- Author: Junghoon Kim, Jongkuk Lim, Jimyeong Kim
- Contact: placidus36@gmail.com, lim.jeikei@gmail.com, wlaud1001@snu.ac.kr
- Reference
    https://github.com/j-marple-dev/model_compression
"""

import os
from typing import Any, Dict
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from importlib import import_module


def create_dataloader(config: Dict[str, Any]):
    """Simple dataloader.

    Args:
        cfg: yaml file path or dictionary type of the data.

    Returns:
        train_loader
        valid_loader
    """
    # Data Setup
    train_dataset, val_dataset = get_pipe(
        data_path=config["DATA_PATH"],
        img_size=config["IMG_SIZE"],
        batch_size=config["BATCH_SIZE"],
        transform_train=config["AUG_TRAIN"],
        transform_test=config["AUG_TEST"],
        transform_train_params=config["AUG_TRAIN_PARAMS"],
        transform_test_params=config.get("AUG_TEST_PARAMS"),
    )

    return get_dataloader(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )


def get_pipe(
    data_path: str = "./save/data",
    img_size: float = 32,
    batch_size: int = 8,
    transform_train: str = "simple_augment_train",
    transform_test: str = "simple_augment_test",
    transform_train_params: Dict[str, int] = None,
    transform_test_params: Dict[str, int] = None,
):
    """Get dataloader pipeline for training and testing."""
    if not transform_train_params:
        transform_train_params = dict()
    if not transform_test_params:
        transform_test_params = dict()

    train_pipe = getattr(import_module("src.dali_pipes"), transform_train)
    val_pipe = getattr(import_module("src.dali_pipes"), transform_test)

    train_path = os.path.join(data_path, "train")
    val_path = os.path.join(data_path, "val")

    train_pipe = train_pipe(batch_size=batch_size, num_threads=8, device_id=0,
                            image_dir=train_path, pipe_name='train', img_size=img_size, **transform_train_params)
    val_pipe = val_pipe(batch_size=batch_size, num_threads=8, device_id=0,
                        image_dir=val_path, pipe_name='val', img_size=img_size, **transform_test_params)
    train_pipe.build()
    val_pipe.build()

    return train_pipe, val_pipe


def get_dataloader(train_dataset, val_dataset):
    """Get dataloader for training and testing."""

    train_loader = DALIGenericIterator(
        [train_dataset], ['data', 'label'], reader_name="train", auto_reset=True)
    valid_loader = DALIGenericIterator(
        [val_dataset], ['data', 'label'], reader_name="val", auto_reset=True)

    return train_loader, valid_loader
