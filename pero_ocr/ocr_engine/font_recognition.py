import os
import re
import json
import torch
import einops
import numpy as np


class FontRecognitionEngine:
    def __init__(self, json_def, device, batch_size=32):
        with open(json_def, 'r', encoding='utf8') as f:
            self.config = json.load(f)

        self.device = device
        self.batch_size = batch_size

        if os.path.isabs(self.config['checkpoint']):
            self.model_path = self.config['checkpoint']
        else:
            self.model_path = os.path.realpath(os.path.join(os.path.dirname(json_def), self.config['checkpoint']))

        self.characters = list(self.config['characters']) + ['<BOUNDARY>', '<PAD>']
        self.joker_index = 0
        self.transcription_padding_index = len(self.characters) - 1
        self.transcription_start_token_index = len(self.characters) - 2
        self.transcription_end_token_index = len(self.characters) - 2

        self.font_families = list(self.config['font_families']) + ['<BOUNDARY>', '<PAD>']
        self.font_family_padding_index = len(self.font_families) - 1
        self.font_family_end_token_index = len(self.font_families) - 2

        self.font_styles = self.config['font_styles']
        self.style_threshold = self.config['font_styles_threshold']

        self.model = self.load_model(self.model_path)
        self.batch_padding_coefficient = 32

    def load_model(self, model_path: str):
        model = torch.jit.load(model_path, map_location=self.device)
        model.eval()
        return model

    def process_lines(self, lines: list[tuple[np.ndarray, str]]) -> list:
        """
        Process lines for font recognition.
        Args:
            lines (list[tuple[np.ndarray, str]]): List of tuples containing line images and their transcriptions.
        Returns:
            list: List of recognized fonts for each line.
        """
        recognized_fonts = []
        while lines:
            batch_lines = lines[:self.batch_size]
            lines = lines[self.batch_size:]

            images = [line[0] for line in batch_lines]
            transcriptions = [line[1] for line in batch_lines]

            batch_images, batch_transcriptions = self.create_batch(images, transcriptions)
            font_families, font_styles = self.process_batch(batch_images, batch_transcriptions)
            fonts = self.postprocess_batch(transcriptions, font_families, font_styles)
            recognized_fonts.extend(fonts)

        return recognized_fonts

    def create_batch(self, images: list[np.ndarray], transcriptions: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Create a batch of images and transcriptions.
        Args:
            images (list[np.ndarray]): List of line images.
            transcriptions (list[str]): List of transcriptions.
        Returns:
            tuple[np.ndarray, np.ndarray]: Batched images and transcriptions.
        """

        images = self.create_batch_images(images)
        transcriptions = self.create_batch_transcriptions(transcriptions)
        return images, transcriptions

    def create_batch_images(self, images: list[np.ndarray]) -> np.ndarray:
        """
        Create a batch of images with padding.
        Args:
            images: List of line images.
        Returns:
            np.ndarray: Batched images with padding.
        """
        max_width = max(img.shape[1] for img in images)
        target_width = int(np.ceil(
            max_width / self.batch_padding_coefficient) * self.batch_padding_coefficient) + self.batch_padding_coefficient

        batch_images = np.zeros((len(images), images[0].shape[0], target_width + 2 * self.batch_padding_coefficient, images[0].shape[2]), dtype=np.uint8)
        for batch_img, data_img in zip(batch_images, images):
            lp = self.batch_padding_coefficient
            batch_img[:, lp:lp + data_img.shape[1]] = data_img

        batch_images = einops.rearrange(batch_images, "n h w c -> n c h w")

        return batch_images

    def create_batch_transcriptions(self, transcriptions: list[str]) -> np.ndarray:
        """
        Create a batch of transcriptions.
        Args:
            transcriptions: List of transcriptions.
        Returns:
            np.ndarray: Batched transcriptions.
        """
        max_length = max(len(t) for t in transcriptions) + 2
        batch_transcriptions = np.full((len(transcriptions), max_length), fill_value=self.transcription_padding_index, dtype=np.int32)

        batch_transcriptions[:, 0] = self.transcription_start_token_index
        for i, t in enumerate(transcriptions):
            batch_transcriptions[i, 1:1+len(t)] = self.transcription_to_labels(t)
            batch_transcriptions[i, 1+len(t)] = self.transcription_end_token_index

        return batch_transcriptions

    def transcription_to_labels(self, transcription: str) -> list[int]:
        """
        Convert transcription string to label indices.
        Args:
            transcription (str): Transcription string.
        Returns:
            list[int]: List of label indices.
        """
        return [self.characters.index(c) if c in self.characters else self.joker_index for c in transcription]


    def process_batch(self, images: np.ndarray, transcriptions: np.ndarray) -> tuple[list, list]:
        """
        Process a batch of lines for font recognition.
        Args:
            images (np.ndarray): Array of line images.
            transcriptions (np.ndarray): Array of transcriptions corresponding to the images.
        Returns:
            list: List of recognized fonts for each line in the batch.
        """

        state_dict = self.model.state_dict()
        model_dtype = state_dict[list(state_dict.keys())[0]].dtype

        images = torch.from_numpy(images).to(self.device).to(model_dtype) / 255.0
        transcriptions = torch.from_numpy(transcriptions).to(self.device)

        encoded_images = self.model.encode(images)
        encoded_images = self.model.adapt(encoded_images)
        family_logits, style_logits = self.model.decode(encoded_images, transcriptions)

        family_logits = einops.rearrange(family_logits, 's b c -> b s c')
        style_logits = einops.rearrange(style_logits, 's b c -> b s c')

        family_outputs, style_outputs = self.decode_logits(transcriptions, family_logits, style_logits, style_threshold=self.style_threshold)

        return family_outputs, style_outputs

    def decode_logits(self, transcriptions, family_logits: torch.Tensor, style_logits: torch.Tensor, style_threshold: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
        '''
        Decode the logits to obtain family and style labels.
        Args:
            transcriptions: Tensor [batch_size, seq_len].
            family_logits: Tensor [batch_size, seq_len, num_family_classes].
            style_logits: Tensor [batch_size, seq_len, num_style_classes].
            style_threshold: Threshold for style classification.
        Returns:
            family_outputs: List of family labels for each line in the batch.
            style_outputs: List of style labels for each line in the batch.
        '''
        family_outputs = []
        style_outputs = []

        for line_transcription, line_family_logits, line_style_logits in zip(transcriptions, family_logits, style_logits):
            line_family_labels = torch.argmax(line_family_logits, dim=-1)

            line_style_logits = torch.sigmoid(line_style_logits)
            line_style_labels = (line_style_logits >= style_threshold).long()

            start_index = 1
            end_index = start_index
            for i, label in enumerate(line_transcription):
                if label == self.transcription_padding_index:
                    print("Warning: padding index found in family labels during decoding.")

                if label == self.transcription_end_token_index and i > start_index:
                    end_index = i - 1
                    break

            line_family_outputs = line_family_labels[start_index:end_index].cpu().numpy()
            line_style_outputs = line_style_labels[start_index:end_index].cpu().numpy()

            family_outputs.append(line_family_outputs)
            style_outputs.append(line_style_outputs)

        return family_outputs, style_outputs

    def postprocess_batch(self, transcriptions: list[str], family_outputs: list, style_outputs: list) -> list:
        '''
        Postprocess the batch outputs to obtain final font recognition results.
        Args:
            transcriptions: List of transcriptions for each line in the batch.
            family_outputs: List of family labels for each line in the batch.
            style_outputs: List of style labels for each line in the batch.
        Returns:
            fonts: List of recognized fonts for each line in the batch.
        '''
        fonts = []
        for line_transcription, line_family_labels, line_style_labels in zip(transcriptions, family_outputs, style_outputs):
            line_fonts = []

            word_indices = [(m.start(), m.end()) for m in re.finditer(r"\S+", line_transcription)]

            for start_index, end_index in word_indices:
                word_transcription = line_transcription[start_index:end_index]
                word_family_labels = line_family_labels[start_index:end_index]
                word_style_labels = line_style_labels[start_index:end_index]

                if len(word_transcription) == 0:
                    continue

                family_label = max(set(word_family_labels), key=list(word_family_labels).count)

                word_style_labels = word_style_labels[word_family_labels == family_label]
                word_style_labels = np.sum(word_style_labels, axis=0, dtype=np.float32)
                word_style_labels /= len(word_style_labels)
                style_labels = [i for i, v in enumerate(word_style_labels) if v > 0.5]

                family_name = self.font_families[family_label]
                styles = sorted([self.font_styles[i] for i in style_labels])

                font_name = family_name.replace(" ", "-").lower()
                if styles:
                    styles_name = "-".join(styles)
                    font_name += f"_{styles_name}"

                line_fonts.append({
                    "text": word_transcription,
                    "family": family_name,
                    "styles": styles,
                    "font": font_name
                })

            fonts.append(line_fonts)

        return fonts
