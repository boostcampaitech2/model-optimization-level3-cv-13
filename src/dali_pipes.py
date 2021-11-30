from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn

DATASET_NORMALIZE_INFO = {
    "TACO": {"MEAN": (0.485, 0.456, 0.406), "STD": (0.229, 0.224, 0.225)},
}


@pipeline_def
def simple_augment_train(image_dir, pipe_name: str='train', img_size=32):
    jpegs, labels = fn.readers.file(file_root=image_dir, name=pipe_name, random_shuffle=True)
    images = fn.decoders.image(jpegs, device='mixed') / 255
    images = fn.resize(images, size=(img_size, img_size))
    images = fn.crop_mirror_normalize(
        images, output_layout="CHW", mean=DATASET_NORMALIZE_INFO["TACO"]["MEAN"], std=DATASET_NORMALIZE_INFO["TACO"]["STD"])

    return images, labels


@pipeline_def
def simple_augment_test(image_dir, pipe_name: str='val', img_size=32):
    jpegs, labels = fn.readers.file(file_root=image_dir, name=pipe_name)
    images = fn.decoders.image(jpegs, device='mixed') / 255
    images = fn.resize(images, size=(img_size, img_size))
    images = fn.crop_mirror_normalize(
        images, output_layout="CHW", mean=DATASET_NORMALIZE_INFO["TACO"]["MEAN"], std=DATASET_NORMALIZE_INFO["TACO"]["STD"])

    return images, labels