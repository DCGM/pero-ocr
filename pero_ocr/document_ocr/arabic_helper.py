import unittest
import re
import arabic_reshaper
import string


class ArabicHelper:
    def __init__(self):
        self._reshaper = arabic_reshaper.ArabicReshaper()
        self._backward_mapping = self._create_backward_mapping()
        self._arabic_chars_pattern = "^([\u0600-\u06ff]|[\u0750-\u077f]|[\ufb50-\ufbc1]|[\ufbd3-\ufd3f]|[\ufd50-\ufd8f]|\
                                     [\ufd92-\ufdc7]|[\ufe70-\ufefc]|[\uFDF0-\uFDFD])+$"

    def label_form_to_visual_form(self, text, reverse_before=True, reverse_after=True):
        if reverse_before:
            text = self.string_to_label_form(text)

        text = self._reshaper.reshape(text)

        if reverse_after:
            text = self.label_form_to_string(text)

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

    def is_arabic_word(self, word):
        result = False
        pattern = self._arabic_chars_pattern

        if re.match(pattern, word):
            result = True

        return result

    def is_arabic_line(self, text):
        result = False

        for word in text.split():
            if self.is_arabic_word(word):
                result = True
                break

        return result

    def ligatures_mapping(self, text):
        result = []
        counter = 0

        for char in text:
            if char not in self._backward_mapping:
                result.append([counter])
                counter += 1
            else:
                mapped_chars_result = []
                mapped_chars = self._backward_mapping[char]
                for mapped_char in mapped_chars:
                    mapped_chars_result.append(counter)
                    counter += 1

                result.append(mapped_chars_result)

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
            if self.is_arabic_word(word):
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
            if arabic != self.is_arabic_word(word):
                if not arabic:
                    words[start:index] = words[start:index][::-1]
                    arabic = True
                    start = index

                else:
                    arabic = False
                    start = index

        return words


class ArabicHelperTest(unittest.TestCase):
    # RECOMMENDATION: open this script in an editor which preserves the order of arabic characters from left to right 
    # (e.g. terminal, sublime text; NOT PyCharm, Visual Studio Code, gedit, github - these editors reverse the order 
    # of arabic characters).
    #
    # Test cases were generated using the HTML below. After opening this file, the browser (tested on Google Chrome 
    # and Firefox) should render the characters in the correct format. Strings in the HTML file should correspond to
    # the string_* variables and the rendered strings should correspond to the visual_* variables.
    # 
    # <html>
    # <head>
    # </head>
    # <body>
    #     <h1 style="direction: rtl">الاستخدام في بصريات المعادن</h1>
    #     <h1 style="direction: rtl">الاستخدام XYZ 12.3 QWER في بصريات ASDF JKL المعادن</h1>
    #     <h1 style="direction: rtl">ليس من الممكن تعيين معامل الانكسار في الشرائح الرقيقة بدقة، لكن في بعض الأحيان يمكن تقديره</h1>
    #     <h1 style="direction: rtl">ليس من الممكن ASDF QWER 12.3 XYZ@FIT.VUTBR.CZ تعيين معامل الانكسار في الشرائح الرقيقة بدقة، 
    #                                لكن في بعض الأحيان يمكن تقديره</h1>
    # </body>
    # </html>
    #

    string_1 = "الاستخدام في بصريات المعادن"
    labels_1 = "نداعملا تايرصب يف مادختسالا"
    visual_1 = "ﻥﺩﺎﻌﻤﻟﺍ ﺕﺎﻳﺮﺼﺑ ﻲﻓ ﻡﺍﺪﺨﺘﺳﻻﺍ"

    string_2 = "الاستخدام XYZ 12.3 QWER في بصريات ASDF JKL المعادن"
    labels_2 = "نداعملا ASDF JKL تايرصب يف XYZ 12.3 QWER مادختسالا"
    visual_2 = "ﻥﺩﺎﻌﻤﻟﺍ ASDF JKL ﺕﺎﻳﺮﺼﺑ ﻲﻓ XYZ 12.3 QWER ﻡﺍﺪﺨﺘﺳﻻﺍ"

    string_3 = "ليس من الممكن تعيين معامل الانكسار في الشرائح الرقيقة بدقة، لكن في بعض الأحيان يمكن تقديره"
    labels_3 = "هريدقت نكمي نايحألا ضعب يف نكل ،ةقدب ةقيقرلا حئارشلا يف راسكنالا لماعم نييعت نكمملا نم سيل"
    visual_3 = "ﻩﺮﻳﺪﻘﺗ ﻦﻜﻤﻳ ﻥﺎﻴﺣﻷﺍ ﺾﻌﺑ ﻲﻓ ﻦﻜﻟ ،ﺔﻗﺪﺑ ﺔﻘﻴﻗﺮﻟﺍ ﺢﺋﺍﺮﺸﻟﺍ ﻲﻓ ﺭﺎﺴﻜﻧﻻﺍ ﻞﻣﺎﻌﻣ ﻦﻴﻴﻌﺗ ﻦﻜﻤﻤﻟﺍ ﻦﻣ ﺲﻴﻟ"

    string_4 = "ليس من الممكن ASDF QWER 12.3 XYZ@FIT.VUTBR.CZ تعيين معامل الانكسار في الشرائح الرقيقة بدقة، لكن في بعض الأحيان يمكن تقديره"
    labels_4 = "هريدقت نكمي نايحألا ضعب يف نكل ،ةقدب ةقيقرلا حئارشلا يف راسكنالا لماعم نييعت ASDF QWER 12.3 XYZ@FIT.VUTBR.CZ نكمملا نم سيل"
    visual_4 = "ﻩﺮﻳﺪﻘﺗ ﻦﻜﻤﻳ ﻥﺎﻴﺣﻷﺍ ﺾﻌﺑ ﻲﻓ ﻦﻜﻟ ،ﺔﻗﺪﺑ ﺔﻘﻴﻗﺮﻟﺍ ﺢﺋﺍﺮﺸﻟﺍ ﻲﻓ ﺭﺎﺴﻜﻧﻻﺍ ﻞﻣﺎﻌﻣ ﻦﻴﻴﻌﺗ ASDF QWER 12.3 XYZ@FIT.VUTBR.CZ ﻦﻜﻤﻤﻟﺍ ﻦﻣ ﺲﻴﻟ"
    
    def test_string_to_label_form_1(self):
        helper = ArabicHelper()
        label_form = helper.string_to_label_form(self.string_1)
        self.assertEqual(label_form, self.labels_1)

    def test_string_to_label_form_2(self):
        helper = ArabicHelper()
        label_form = helper.string_to_label_form(self.string_2)
        self.assertEqual(label_form, self.labels_2)

    def test_string_to_label_form_3(self):
        helper = ArabicHelper()
        label_form = helper.string_to_label_form(self.string_3)
        self.assertEqual(label_form, self.labels_3)
    
    def test_string_to_label_form_4(self):
        helper = ArabicHelper()
        label_form = helper.string_to_label_form(self.string_4)
        self.assertEqual(label_form, self.labels_4)
    
    def test_label_form_to_string_1(self):
        helper = ArabicHelper()
        string = helper.label_form_to_string(self.labels_1)
        self.assertEqual(string, self.string_1)

    def test_label_form_to_string_2(self):
        helper = ArabicHelper()
        string = helper.label_form_to_string(self.labels_2)
        self.assertEqual(string, self.string_2)

    def test_label_form_to_string_3(self):
        helper = ArabicHelper()
        string = helper.label_form_to_string(self.labels_3)
        self.assertEqual(string, self.string_3)

    def test_label_form_to_string_4(self):
        helper = ArabicHelper()
        string = helper.label_form_to_string(self.labels_4)
        self.assertEqual(string, self.string_4)

    def test_label_form_to_visual_form_1(self):
        helper = ArabicHelper()
        visual_form = helper.label_form_to_visual_form(self.labels_1)
        self.assertEqual(visual_form, self.visual_1)

    def test_label_form_to_visual_form_2(self):
        helper = ArabicHelper()
        visual_form = helper.label_form_to_visual_form(self.labels_2)
        self.assertEqual(visual_form, self.visual_2)

    def test_label_form_to_visual_form_3(self):
        helper = ArabicHelper()
        visual_form = helper.label_form_to_visual_form(self.labels_3)
        self.assertEqual(visual_form, self.visual_3)

    def test_label_form_to_visual_form_4(self):
        helper = ArabicHelper()
        visual_form = helper.label_form_to_visual_form(self.labels_4)
        self.assertEqual(visual_form, self.visual_4)

    def test_visual_form_to_label_form_1(self):
        helper = ArabicHelper()
        label_form = helper.visual_form_to_label_form(self.visual_1)
        self.assertEqual(label_form, self.labels_1)

    def test_visual_form_to_label_form_2(self):
        helper = ArabicHelper()
        label_form = helper.visual_form_to_label_form(self.visual_2)
        self.assertEqual(label_form, self.labels_2)

    def test_visual_form_to_label_form_3(self):
        helper = ArabicHelper()
        label_form = helper.visual_form_to_label_form(self.visual_3)
        self.assertEqual(label_form, self.labels_3)

    def test_visual_form_to_label_form_4(self):
        helper = ArabicHelper()
        label_form = helper.visual_form_to_label_form(self.visual_4)
        self.assertEqual(label_form, self.labels_4)


if __name__ == "__main__":
    unittest.main()
