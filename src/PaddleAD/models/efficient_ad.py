"""EfficientAd model implementation."""

from __future__ import annotations

import logging
import random
from enum import Enum

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.vision import transforms

logger = logging.getLogger(__name__)


def imagenet_norm_batch(x: paddle.Tensor) -> paddle.Tensor:
    """Normalizes a batch of images with mean and standard deviation of ImageNet."""
    mean = paddle.to_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = paddle.to_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    x_norm = (x - mean) / std
    return x_norm


def reduce_tensor_elems(x: paddle.Tensor, m=2**24) -> paddle.Tensor:
    """Reduces the number of elements in a tensor to m."""
    x = paddle.flatten(x)
    if len(x) > m:
        perm = paddle.randperm(len(x))
        idx = perm[:m]
        x = x[idx]
    return x


class EfficientAdModelSize(str, Enum):
    """Supported EfficientAd model sizes"""

    M = "medium"
    S = "small"


class PDNS(nn.Layer):
    """Patch Description Network small"""

    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2D(3, 128, 4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2D(128, 256, 4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2D(256, 256, 3, stride=1, padding=1 * pad_mult)
        self.conv4 = nn.Conv2D(256, out_channels, 4, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2D(2, stride=2, padding=1 * pad_mult, exclusive=False)
        self.avgpool2 = nn.AvgPool2D(2, stride=2, padding=1 * pad_mult, exclusive=False)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class PDNM(nn.Layer):
    """Patch Description Network medium"""

    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2D(3, 256, 4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2D(256, 512, 4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2D(512, 512, 1, stride=1, padding=0 * pad_mult)
        self.conv4 = nn.Conv2D(512, 512, 3, stride=1, padding=1 * pad_mult)
        self.conv5 = nn.Conv2D(512, out_channels, 4, stride=1, padding=0 * pad_mult)
        self.conv6 = nn.Conv2D(
            out_channels, out_channels, 1, stride=1, padding=0 * pad_mult
        )
        self.avgpool1 = nn.AvgPool2D(2, stride=2, padding=1 * pad_mult, exclusive=False)
        self.avgpool2 = nn.AvgPool2D(2, stride=2, padding=1 * pad_mult, exclusive=False)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x


class Encoder(nn.Layer):
    """Autoencoder Encoder model."""

    def __init__(self) -> None:
        super().__init__()
        self.enconv1 = nn.Conv2D(3, 32, 4, stride=2, padding=1)
        self.enconv2 = nn.Conv2D(32, 32, 4, stride=2, padding=1)
        self.enconv3 = nn.Conv2D(32, 64, 4, stride=2, padding=1)
        self.enconv4 = nn.Conv2D(64, 64, 4, stride=2, padding=1)
        self.enconv5 = nn.Conv2D(64, 64, 4, stride=2, padding=1)
        self.enconv6 = nn.Conv2D(64, 64, 8, stride=1, padding=0)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        x = self.enconv6(x)
        return x


class Decoder(nn.Layer):
    """Autoencoder Decoder model."""

    def __init__(
        self, out_channels: int, padding: bool, img_size: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.img_size = img_size
        self.last_upsample = int(img_size / 4) if padding else int(img_size / 4) - 8
        self.deconv1 = nn.Conv2D(64, 64, 4, stride=1, padding=2)
        self.deconv2 = nn.Conv2D(64, 64, 4, stride=1, padding=2)
        self.deconv3 = nn.Conv2D(64, 64, 4, stride=1, padding=2)
        self.deconv4 = nn.Conv2D(64, 64, 4, stride=1, padding=2)
        self.deconv5 = nn.Conv2D(64, 64, 4, stride=1, padding=2)
        self.deconv6 = nn.Conv2D(64, 64, 4, stride=1, padding=2)
        self.deconv7 = nn.Conv2D(64, 64, 3, stride=1, padding=1)
        self.deconv8 = nn.Conv2D(64, out_channels, 3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = F.interpolate(x, size=int(self.img_size / 64) - 1, mode="bilinear")
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(x, size=int(self.img_size / 32), mode="bilinear")
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(x, size=int(self.img_size / 16) - 1, mode="bilinear")
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(x, size=int(self.img_size / 8), mode="bilinear")
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(x, size=int(self.img_size / 4) - 1, mode="bilinear")
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(x, size=int(self.img_size / 2) - 1, mode="bilinear")
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = F.interpolate(x, size=self.last_upsample, mode="bilinear")
        x = F.relu(self.deconv7(x))
        x = self.deconv8(x)
        return x


class AutoEncoder(nn.Layer):
    """EfficientAd Autoencoder."""

    def __init__(
        self, out_channels: int, padding: bool, img_size: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder(out_channels, padding, img_size)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = imagenet_norm_batch(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class EfficientAD(nn.Layer):
    """EfficientAd model."""

    def __init__(
        self,
        teacher_out_channels: int,
        input_size: tuple[int, int],
        model_size: EfficientAdModelSize = EfficientAdModelSize.S,
        padding=False,
        pad_maps=True,
    ) -> None:
        super().__init__()

        self.pad_maps = pad_maps
        self.teacher: PDNM | PDNS
        self.student: PDNM | PDNS

        if model_size == EfficientAdModelSize.M:
            self.teacher = PDNM(teacher_out_channels, padding=padding).eval()
            self.student = PDNM(teacher_out_channels * 2, padding=padding)
        elif model_size == EfficientAdModelSize.S:
            self.teacher = PDNS(teacher_out_channels, padding=padding).eval()
            self.student = PDNS(teacher_out_channels * 2, padding=padding)
        else:
            raise ValueError(f"Unknown model size {model_size}")

        self.ae: AutoEncoder = AutoEncoder(teacher_out_channels, padding, input_size[0])
        self.teacher_out_channels: int = teacher_out_channels
        self.input_size: tuple[int, int] = input_size

        mean = paddle.zeros((1, self.teacher_out_channels, 1, 1))
        std = paddle.zeros((1, self.teacher_out_channels, 1, 1))
        self.mean_std: dict = {
            "mean": paddle.create_parameter(
                mean.shape, mean.dtype, default_initializer=nn.initializer.Assign(mean)
            ),
            "std": paddle.create_parameter(
                std.shape, std.dtype, default_initializer=nn.initializer.Assign(std)
            ),
        }
        self.mean_std["mean"].stop_gradient = True
        self.mean_std["std"].stop_gradient = True

        qa_st = paddle.to_tensor(0.0)
        qb_st = paddle.to_tensor(0.0)
        qa_ae = paddle.to_tensor(0.0)
        qb_ae = paddle.to_tensor(0.0)
        self.quantiles: dict = {
            "qa_st": paddle.create_parameter(
                qa_st.shape,
                qa_st.dtype,
                default_initializer=nn.initializer.Assign(qa_st),
            ),
            "qb_st": paddle.create_parameter(
                qb_st.shape,
                qb_st.dtype,
                default_initializer=nn.initializer.Assign(qb_st),
            ),
            "qa_ae": paddle.create_parameter(
                qa_ae.shape,
                qa_ae.dtype,
                default_initializer=nn.initializer.Assign(qa_ae),
            ),
            "qb_ae": paddle.create_parameter(
                qb_ae.shape,
                qb_ae.dtype,
                default_initializer=nn.initializer.Assign(qb_ae),
            ),
        }
        self.quantiles["qa_st"].stop_gradient = True
        self.quantiles["qb_st"].stop_gradient = True
        self.quantiles["qa_ae"].stop_gradient = True
        self.quantiles["qb_ae"].stop_gradient = True

    def is_set(self, p_dic: dict) -> bool:
        """Check if a dictionary is set."""
        for _, value in p_dic.items():
            if value.sum() != 0:
                return True
        return False

    def choose_random_aug_image(self, image: paddle.Tensor) -> paddle.Tensor:
        """Choose a random augmentation function and apply it to the image."""
        transform_functions = [
            transforms.functional.adjust_brightness,
            transforms.functional.adjust_contrast,
            transforms.functional.adjust_saturation,
        ]
        # Sample an augmentation coefficient λ from the uniform distribution U(0.8, 1.2)
        coefficient = random.uniform(0.8, 1.2)
        transform_function = random.choice(transform_functions)
        return transform_function(image, coefficient)

    def forward(
        self, batch: paddle.Tensor, batch_imagenet: paddle.Tensor = None
    ) -> paddle.Tensor | dict:
        with paddle.no_grad():
            teacher_output = self.teacher(batch)
            if self.is_set(self.mean_std):
                teacher_output = (
                    teacher_output - self.mean_std["mean"]
                ) / self.mean_std["std"]

        student_output = self.student(batch)
        distance_st = paddle.pow(
            teacher_output - student_output[:, : self.teacher_out_channels, :, :], 2
        )

        if self.training:
            # Student loss
            distance_st = reduce_tensor_elems(distance_st)
            d_hard = paddle.quantile(distance_st, 0.999)
            loss_hard = paddle.mean(distance_st[distance_st >= d_hard])
            student_output_penalty = self.student(batch_imagenet)[
                :, : self.teacher_out_channels, :, :
            ]
            loss_penalty = paddle.mean(student_output_penalty**2)
            loss_st = loss_hard + loss_penalty

            # Autoencoder and Student AE loss
            aug_img = self.choose_random_aug_image(batch)
            ae_output_aug = self.ae(aug_img)

            with paddle.no_grad():
                teacher_output_aug = self.teacher(aug_img)
                if self.is_set(self.mean_std):
                    teacher_output_aug = (
                        teacher_output_aug - self.mean_std["mean"]
                    ) / self.mean_std["std"]

            student_output_ae_aug = self.student(aug_img)[
                :, self.teacher_out_channels :, :, :
            ]

            distance_ae = paddle.pow(teacher_output_aug - ae_output_aug, 2)
            distance_stae = paddle.pow(ae_output_aug - student_output_ae_aug, 2)

            loss_ae = paddle.mean(distance_ae)
            loss_stae = paddle.mean(distance_stae)

            return loss_st, loss_ae, loss_stae

        with paddle.no_grad():
            ae_output = self.ae(batch)

        map_st = paddle.mean(distance_st, axis=1, keepdim=True)
        map_stae = paddle.mean(
            (ae_output - student_output[:, self.teacher_out_channels :]) ** 2,
            axis=1,
            keepdim=True,
        )

        if self.pad_maps:
            map_st = F.pad(map_st, (4, 4, 4, 4))
            map_stae = F.pad(map_stae, (4, 4, 4, 4))

        map_st = F.interpolate(
            map_st, size=(self.input_size[0], self.input_size[1]), mode="bilinear"
        )
        map_stae = F.interpolate(
            map_stae, size=(self.input_size[0], self.input_size[1]), mode="bilinear"
        )

        if self.is_set(self.quantiles):
            map_st = (
                0.1
                * (map_st - self.quantiles["qa_st"])
                / (self.quantiles["qb_st"] - self.quantiles["qa_st"])
            )
            map_stae = (
                0.1
                * (map_stae - self.quantiles["qa_ae"])
                / (self.quantiles["qb_ae"] - self.quantiles["qa_ae"])
            )

        map_combined = 0.5 * map_st + 0.5 * map_stae

        return {
            "anomaly_map_combined": map_combined,
            "map_st": map_st,
            "map_ae": map_stae,
        }
