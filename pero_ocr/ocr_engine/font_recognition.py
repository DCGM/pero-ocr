import os
import re
import cv2
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

        self.characters = list(self.config['characters'])
        self.joker_index = 0
        self.transcription_padding_index = len(self.characters)
        self.transcription_start_token_index = len(self.characters) + 1
        self.transcription_end_token_index = len(self.characters) + 1

        self.font_families = list(self.config['font_families'])
        self.font_family_padding_index = len(self.font_families)
        self.font_family_end_token_index = len(self.font_families) + 1

        self.font_styles = self.config['font_styles']
        self.style_threshold = self.config['font_styles_threshold']

        self.preprocessing = self.config.get('preprocessing', {})
        self.postprocessing = self.config.get('postprocessing', {})

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

            images = [self.preprocess(line[0]) for line in batch_lines]
            transcriptions = [line[1] for line in batch_lines]

            batch_images, batch_transcriptions = self.create_batch(images, transcriptions)

            # import IPython; IPython.embed(); exit(1)

            with torch.no_grad():
                font_families, font_styles = self.process_batch(batch_images, batch_transcriptions)
            
            fonts = self.postprocess_batch(transcriptions, font_families, font_styles)
            fonts = self.postprocess(fonts)
            recognized_fonts.extend(fonts)

        return recognized_fonts

    @staticmethod
    def gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 0) -> np.ndarray:
        """
        Apply Gaussian blur to the image.
        Args:
            image (np.ndarray): Input image.
            kernel_size (int): Size of the Gaussian kernel.
            sigma (float): Standard deviation of the Gaussian kernel.
        Returns:
            np.ndarray: Blurred image.
        """
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)        
        return blurred_image

    @staticmethod
    def threshold_image(image: np.ndarray, threshold: int = 200) -> np.ndarray:
        """
        Apply thresholding to the image to enhance contrast.
        Args:
            image (np.ndarray): Input image.
            threshold (int): Threshold value.
        Returns:
            np.ndarray: Thresholded image.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        return binary_image

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image according to the configuration.
        Args:
            image (np.ndarray): Input image.
        Returns:
            np.ndarray: Preprocessed image.
        """
        is_active = self.preprocessing.get('active', True)
        if not is_active:
            return image

        gaussian_blurring = self.preprocessing.get('gaussian_blur', False)
        if gaussian_blurring:
            kernel_size = int(gaussian_blurring.get('kernel_size', 5))
            sigma = float(gaussian_blurring.get('sigma', 0))
            image = self.gaussian_blur(image, kernel_size, sigma)

        thresholding = self.preprocessing.get('thresholding', False)
        if thresholding:
            threshold = int(thresholding.get('threshold_value', 164))
            image = self.threshold_image(image, threshold)

        return image

    @staticmethod
    def filter_styles_by_family(fonts: list, style_families: list) -> list:
        for line_fonts in fonts:
            for word_font in line_fonts:
                family = word_font['family']
                if family not in style_families:
                    word_font['styles'] = []
                    word_font['font'] = family.replace(" ", "-").lower()

        return fonts


    def postprocess(self, fonts: list) -> list:
        """
        Postprocess the recognized fonts according to the configuration.
        Args:
            fonts (list): List of recognized fonts.
        Returns:
            list: Postprocessed recognized fonts.
        """
        is_active = self.postprocessing.get('active', True)
        if not is_active:
            return fonts

        style_families = self.postprocessing.get('styles_families', None)
        if style_families is not None:
            fonts = self.filter_styles_by_family(fonts, style_families)

        return fonts

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
                    end_index = i
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

                if family_label < len(self.font_families):
                    family_name = self.font_families[family_label]
                else:
                    family_name = "Unknown"

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
