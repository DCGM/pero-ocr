import unittest

import arabic_reshaper


class ArabicHelper:
    def __init__(self):
        self._reshaper = arabic_reshaper.ArabicReshaper()
        self._backward_mapping = self._create_backward_mapping()

    def reshape_transcription(self, text):
        text = self._reverse_transcription(text)
        text = self._reshaper.reshape(text)
        text = self._reverse_transcription(text)
        return text

    def _reverse_transcription(self, transcription):
        transcription = transcription[::-1]
        return transcription

    def reshape_transcription_back(self, text):
        text = self._map_backward(text)
        return text

    def _create_backward_mapping(self):
        forward_mapping = self._reshaper.letters
        backward_mapping = {}

        for letter in forward_mapping:
            letter_options = forward_mapping[letter]
            for letter_option in letter_options:
                if len(letter_option) > 0:
                    backward_mapping[letter_option] = letter

        return backward_mapping

    def _map_backward(self, text):
        result = ""

        for letter in text:
            if letter in self._backward_mapping:
                result += self._backward_mapping[letter]

            else:
                result += letter

        return result


class ArabicHelperTest(unittest.TestCase):
    # The transcription was taken from MADCAT Arabic dataset, specifically from text line with ID
    # 'XIA20041213.0121_1_LDC0316-r16-l001'. Strings below might not be rendered correctly in a document viewer
    # or an IDE, but should be rendered correctly when printed to the console.
    source_simple = "ةيركسعلا طباورلا نا هوق لاقو"
    target_simple = "ﺔﻳﺮﻜﺴﻌﻟﺍ ﻂﺑﺍﻭﺮﻟﺍ ﻥﺍ ﻩﻮﻗ ﻝﺎﻗﻭ"

    source_rich = "ةيركسعلا ASDF طباورلا 25.6 نا هوق لاقو!"
    target_rich = "ﺔﻳﺮﻜﺴﻌﻟﺍ ASDF ﻂﺑﺍﻭﺮﻟﺍ 25.6 ﻥﺍ ﻩﻮﻗ ﻝﺎﻗﻭ!"

    def test_reshape_transcription_simple(self):
        helper = ArabicHelper()
        reshaped = helper.reshape_transcription(self.source_simple)
        self.assertEqual(reshaped, self.target_simple)

    def test_reshape_transcription_rich(self):
        helper = ArabicHelper()
        reshaped = helper.reshape_transcription(self.source_rich)
        self.assertEqual(reshaped, self.target_rich)

    def test_reshape_transcription_back_simple(self):
        helper = ArabicHelper()
        reshaped_back = helper.reshape_transcription_back(self.target_simple)
        self.assertEqual(reshaped_back, self.source_simple)

    def test_reshape_transcription_back_rich(self):
        helper = ArabicHelper()
        reshaped_back = helper.reshape_transcription_back(self.target_rich)
        self.assertEqual(reshaped_back, self.source_rich)


if __name__ == "__main__":
    unittest.main()
