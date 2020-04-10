from lib import dataloader as dl

### 各種transform
### Data Augmentationなし
def trainsform_none():
    """
    Data Augmentationなし

    """
    return dl.pattern_transform(resize=224, HorizontalFlip=False, VerticalFlip=False, Rotation=False, Perspective=False, Crop=False, Erasing=False)

### 全Data Augmentation
def trainsform_all():
    """
    全Data Augmentation

    """
    return dl.pattern_transform(resize=224, HorizontalFlip=True, VerticalFlip=True, Rotation=True, Perspective=True, Crop=True, Erasing=True)


### 対象のData Augmentationのみ
### HorizontalFlip
def trainsform_horizontal_flip():
    """
    datasetに対するData Augmentation(HorizontalFlipのみ実施)

    """
    return dl.pattern_transform(resize=224, HorizontalFlip=True, VerticalFlip=False, Rotation=False, Perspective=False, Crop=False, Erasing=False)

### VerticalFlip
def trainsform_vertical_flip():
    """
    datasetに対するData Augmentation(VerticalFlipのみ実施)

    """
    return dl.pattern_transform(resize=224, HorizontalFlip=False, VerticalFlip=True, Rotation=False, Perspective=False, Crop=False, Erasing=False)

### Rotation
def trainsform_rotation():
    """
    datasetに対するData Augmentation(Rotationのみ実施)

    """
    return dl.pattern_transform(resize=224, HorizontalFlip=False, VerticalFlip=False, Rotation=True, Perspective=False, Crop=False, Erasing=False)

### Perspective
def trainsform_perspective():
    """
    datasetに対するData Augmentation(Perspectiveのみ実施)

    """
    return dl.pattern_transform(resize=224, HorizontalFlip=False, VerticalFlip=False, Rotation=False, Perspective=True, Crop=False, Erasing=False)

### Crop
def trainsform_crop():
    """
    datasetに対するData Augmentation(Cropのみ実施)

    """
    return dl.pattern_transform(resize=224, HorizontalFlip=False, VerticalFlip=False, Rotation=False, Perspective=False, Crop=True, Erasing=False)

### Erasing
def trainsform_erasing():
    """
    datasetに対するData Augmentation(Erasingのみ実施)

    """
    return dl.pattern_transform(resize=224, HorizontalFlip=False, VerticalFlip=False, Rotation=False, Perspective=False, Crop=False, Erasing=True)


### 対象のData Augmentationを除く
### HorizontalFlip
def exclude_horizontal_flip():
    """
    datasetに対するData Augmentation(HorizontalFlip以外を実施)

    """
    return dl.pattern_transform(resize=224, HorizontalFlip=False, VerticalFlip=True, Rotation=True, Perspective=True, Crop=True, Erasing=True)

### VerticalFlip
def exclude_vertical_flip():
    """
    datasetに対するData Augmentation(VerticalFlip以外を実施)

    """
    return dl.pattern_transform(resize=224, HorizontalFlip=True, VerticalFlip=False, Rotation=True, Perspective=True, Crop=True, Erasing=True)

### Rotation
def exclude_rotation():
    """
    datasetに対するData Augmentation(Rotation以外を実施)

    """
    return dl.pattern_transform(resize=224, HorizontalFlip=True, VerticalFlip=True, Rotation=False, Perspective=True, Crop=True, Erasing=True)

### Perspective
def exclude_perspective():
    """
    datasetに対するData Augmentation(Perspective以外を実施)

    """
    return dl.pattern_transform(resize=224, HorizontalFlip=True, VerticalFlip=True, Rotation=True, Perspective=False, Crop=True, Erasing=True)

### Crop
def exclude_crop():
    """
    datasetに対するData Augmentation(Crop以外を実施)

    """
    return dl.pattern_transform(resize=224, HorizontalFlip=True, VerticalFlip=True, Rotation=True, Perspective=True, Crop=False, Erasing=True)

### Erasing
    """
    datasetに対するData Augmentation(Erasing以外を実施)

    """
def exclude_erasing():
    return dl.pattern_transform(resize=224, HorizontalFlip=True, VerticalFlip=True, Rotation=True, Perspective=True, Crop=True, Erasing=False)
