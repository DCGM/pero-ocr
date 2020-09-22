import unittest
import re
import arabic_reshaper


class ArabicHelper:
    def __init__(self):
        self._reshaper = arabic_reshaper.ArabicReshaper()
        self._backward_mapping = self._create_backward_mapping()
        self._arabic_chars_pattern = "^[\u0621-\u064A]+$"

    def label_form_to_visual_form(self, text):
        text = self._reverse_transcription(text)
        text = self._reshaper.reshape(text)
        text = self._reverse_transcription(text)
        return text

    def visual_form_to_label_form(self, text):
        text = self._map_backward(text)
        return text

    def string_to_label_form(self, text):
        words = text.split(' ')
        words = self._reverse_arabic_words(words)
        words = self._reverse_words_order(words)

        result = ' '.join(words)
        return result

    def label_form_to_string(self, text):
        result = self.string_to_label_form(text)
        return result

    def _is_arabic(self, word):
        result = False
        pattern = self._arabic_chars_pattern

        if re.match(pattern, word):
            result = True

        return result

    def _reverse_transcription(self, transcription):
        transcription = transcription[::-1]
        return transcription

    def _create_backward_mapping(self):
        forward_mapping = self._reshaper.letters
        backward_mapping = {}

        for letter in forward_mapping:
            letter_options = forward_mapping[letter]
            for letter_option in letter_options:
                if len(letter_option) > 0:
                    backward_mapping[letter_option] = letter

        self._add_ligatures(backward_mapping)

        return backward_mapping

    def _add_ligatures(self, backward_mapping):
        all_ligatures = arabic_reshaper.ligatures.LIGATURES
        
        for ligature_record in all_ligatures:
            name, forward_mapping = ligature_record
            letters, letter_options = forward_mapping            
            for letter_option in letter_options:
                if len(letter_option) > 0:
                    backward_mapping[letter_option] = self._reverse_transcription(letters)

    def _map_backward(self, text):
        result = ""

        for letter in text:
            if letter in self._backward_mapping:
                result += self._backward_mapping[letter]

            else:
                result += letter

        return result

    def _reverse_arabic_words(self, words):
        new_words = []
        for word in words:
            if self._is_arabic(word):
                new_words.append(word[::-1])
            else:
                new_words.append(word)

        return new_words

    def _reverse_words_order(self, words):
        # reverse the order of all words
        words = words[::-1]

        arabic = True
        start = 0

        # reverse back the order of subsequent non-arabic words
        for index, word in enumerate(words):
            if arabic != self._is_arabic(word):
                if not arabic:
                    words[start:index] = words[start:index][::-1]
                    arabic = True
                    start = index

                else:
                    arabic = False
                    start = index

        return words


class ArabicHelperLabelAndVisualFormsTest(unittest.TestCase):
    # The transcription was taken from MADCAT Arabic dataset, specifically from text line with ID
    # 'XIA20041213.0121_1_LDC0316-r16-l001'. Strings below might not be rendered correctly in a document viewer
    # or an IDE, but should be rendered correctly when printed to the console.
    source_simple = "ةيركسعلا طباورلا نا هوق لاقو"
    target_simple = "ﺔﻳﺮﻜﺴﻌﻟﺍ ﻂﺑﺍﻭﺮﻟﺍ ﻥﺍ ﻩﻮﻗ ﻝﺎﻗﻭ"

    source_rich = "ةيركسعلا ASDF طباورلا 25.6 نا هوق لاقو!"
    target_rich = "ﺔﻳﺮﻜﺴﻌﻟﺍ ASDF ﻂﺑﺍﻭﺮﻟﺍ 25.6 ﻥﺍ ﻩﻮﻗ ﻝﺎﻗﻭ!"

    def test_label_form_to_visual_form_simple(self):
        helper = ArabicHelper()
        visual_form = helper.label_form_to_visual_form(self.source_simple)
        self.assertEqual(visual_form, self.target_simple)

    def test_label_form_to_visual_form_rich(self):
        helper = ArabicHelper()
        visual_form = helper.label_form_to_visual_form(self.source_rich)
        self.assertEqual(visual_form, self.target_rich)

    def test_visual_form_to_label_form_simple(self):
        helper = ArabicHelper()
        label_form = helper.visual_form_to_label_form(self.target_simple)
        self.assertEqual(label_form, self.source_simple)

    def test_visual_form_to_label_form_rich(self):
        helper = ArabicHelper()
        label_form = helper.visual_form_to_label_form(self.target_rich)
        self.assertEqual(label_form, self.source_rich)


class ArabicHelperStringAndLabelFormTest(unittest.TestCase):
    source_1 = "الاستخدام في بصريات المعادن"
    target_1 = "نداعملا تايرصب يف مادختسالا"

    source_2 = "الاستخدام XYZ 12.3 QWER في بصريات ASDF JKL المعادن"
    target_2 = "نداعملا ASDF JKL تايرصب يف XYZ 12.3 QWER مادختسالا"

    source_3 = "ليس من الممكن ASDF QWER 12.3 XYZ@FIT.VUTBR.CZ تعيين معامل الانكسار في الشرائح الرقيقة بدقة، لكن في بعض الأحيان يمكن تقديره"
    target_3 = "هريدقت نكمي نايحألا ضعب يف نكل بدقة، ةقيقرلا حئارشلا يف راسكنالا لماعم نييعت ASDF QWER 12.3 XYZ@FIT.VUTBR.CZ نكمملا نم سيل"

    def test_string_to_label_form_1(self):
        helper = ArabicHelper()
        label_form = helper.string_to_label_form(self.source_1)
        self.assertEqual(label_form, self.target_1)

    def test_string_to_label_form_2(self):
        helper = ArabicHelper()
        label_form = helper.string_to_label_form(self.source_2)
        self.assertEqual(label_form, self.target_2)

    def test_string_to_label_form_3(self):
        helper = ArabicHelper()
        label_form = helper.string_to_label_form(self.source_3)
        self.assertEqual(label_form, self.target_3)

    def test_label_form_to_string_1(self):
        helper = ArabicHelper()
        string = helper.string_to_label_form(self.target_1)
        self.assertEqual(string, self.source_1)

    def test_label_form_to_string_2(self):
        helper = ArabicHelper()
        string = helper.string_to_label_form(self.target_2)
        self.assertEqual(string, self.source_2)

    def test_label_form_to_string_3(self):
        helper = ArabicHelper()
        string = helper.string_to_label_form(self.target_3)
        self.assertEqual(string, self.source_3)


if __name__ == "__main__":
    unittest.main()
