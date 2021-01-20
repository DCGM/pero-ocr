import unittest
import re
import arabic_reshaper
import string
import sys


class ArabicHelper:
    def __init__(self):
        self._reshaper = arabic_reshaper.ArabicReshaper()
        self._backward_mapping = self._create_backward_mapping()
        self._arabic_chars_pattern = "^([\u0600-\u06ff]|[\u0750-\u077f]|[\ufb50-\ufbc1]|[\ufbd3-\ufd3f]|[\ufd50-\ufd8f]|\
                                     [\ufd92-\ufdc7]|[\ufe70-\ufefc]|[\uFDF0-\uFDFD])+$"

        self.LETTER = 0;
        self.FORM = 1;
        self.NOT_SUPPORTED = -1;
        self.ISOLATED = 0;
        self.INITIAL = 1;
        self.MEDIAL = 2;
        self.FINAL = 3;

        self.forward_mapping = {};
        self.forward_mapping['ء'] = ['ﺀ', '', '', ''];
        self.forward_mapping['آ'] = ['ﺁ', '', '', 'ﺂ'];
        self.forward_mapping['أ'] = ['ﺃ', '', '', 'ﺄ'];
        self.forward_mapping['ؤ'] = ['ﺅ', '', '', 'ﺆ'];
        self.forward_mapping['إ'] = ['ﺇ', '', '', 'ﺈ'];
        self.forward_mapping['ئ'] = ['ﺉ', 'ﺋ', 'ﺌ', 'ﺊ'];
        self.forward_mapping['ا'] = ['ﺍ', '', '', 'ﺎ'];
        self.forward_mapping['ب'] = ['ﺏ', 'ﺑ', 'ﺒ', 'ﺐ'];
        self.forward_mapping['ة'] = ['ﺓ', '', '', 'ﺔ'];
        self.forward_mapping['ت'] = ['ﺕ', 'ﺗ', 'ﺘ', 'ﺖ'];
        self.forward_mapping['ث'] = ['ﺙ', 'ﺛ', 'ﺜ', 'ﺚ'];
        self.forward_mapping['ج'] = ['ﺝ', 'ﺟ', 'ﺠ', 'ﺞ'];
        self.forward_mapping['ح'] = ['ﺡ', 'ﺣ', 'ﺤ', 'ﺢ'];
        self.forward_mapping['خ'] = ['ﺥ', 'ﺧ', 'ﺨ', 'ﺦ'];
        self.forward_mapping['د'] = ['ﺩ', '', '', 'ﺪ'];
        self.forward_mapping['ذ'] = ['ﺫ', '', '', 'ﺬ'];
        self.forward_mapping['ر'] = ['ﺭ', '', '', 'ﺮ'];
        self.forward_mapping['ز'] = ['ﺯ', '', '', 'ﺰ'];
        self.forward_mapping['س'] = ['ﺱ', 'ﺳ', 'ﺴ', 'ﺲ'];
        self.forward_mapping['ش'] = ['ﺵ', 'ﺷ', 'ﺸ', 'ﺶ'];
        self.forward_mapping['ص'] = ['ﺹ', 'ﺻ', 'ﺼ', 'ﺺ'];
        self.forward_mapping['ض'] = ['ﺽ', 'ﺿ', 'ﻀ', 'ﺾ'];
        self.forward_mapping['ط'] = ['ﻁ', 'ﻃ', 'ﻄ', 'ﻂ'];
        self.forward_mapping['ظ'] = ['ﻅ', 'ﻇ', 'ﻈ', 'ﻆ'];
        self.forward_mapping['ع'] = ['ﻉ', 'ﻋ', 'ﻌ', 'ﻊ'];
        self.forward_mapping['غ'] = ['ﻍ', 'ﻏ', 'ﻐ', 'ﻎ'];
        self.forward_mapping['ـ'] = ['ـ', 'ـ', 'ـ', 'ـ'];
        self.forward_mapping['ف'] = ['ﻑ', 'ﻓ', 'ﻔ', 'ﻒ'];
        self.forward_mapping['ق'] = ['ﻕ', 'ﻗ', 'ﻘ', 'ﻖ'];
        self.forward_mapping['ك'] = ['ﻙ', 'ﻛ', 'ﻜ', 'ﻚ'];
        self.forward_mapping['ل'] = ['ﻝ', 'ﻟ', 'ﻠ', 'ﻞ'];
        self.forward_mapping['م'] = ['ﻡ', 'ﻣ', 'ﻤ', 'ﻢ'];
        self.forward_mapping['ن'] = ['ﻥ', 'ﻧ', 'ﻨ', 'ﻦ'];
        self.forward_mapping['ه'] = ['ﻩ', 'ﻫ', 'ﻬ', 'ﻪ'];
        self.forward_mapping['و'] = ['ﻭ', '', '', 'ﻮ'];
        self.forward_mapping['ى'] = ['ﻯ', 'ﯨ', 'ﯩ', 'ﻰ'];
        self.forward_mapping['ي'] = ['ﻱ', 'ﻳ', 'ﻴ', 'ﻲ'];
        self.forward_mapping['ٱ'] = ['ﭐ', '', '', 'ﭑ'];
        self.forward_mapping['ٷ'] = ['ﯝ', '', '', ''];
        self.forward_mapping['ٹ'] = ['ﭦ', 'ﭨ', 'ﭩ', 'ﭧ'];
        self.forward_mapping['ٺ'] = ['ﭞ', 'ﭠ', 'ﭡ', 'ﭟ'];
        self.forward_mapping['ٻ'] = ['ﭒ', 'ﭔ', 'ﭕ', 'ﭓ'];
        self.forward_mapping['پ'] = ['ﭖ', 'ﭘ', 'ﭙ', 'ﭗ'];
        self.forward_mapping['ٿ'] = ['ﭢ', 'ﭤ', 'ﭥ', 'ﭣ'];
        self.forward_mapping['ڀ'] = ['ﭚ', 'ﭜ', 'ﭝ', 'ﭛ'];
        self.forward_mapping['ڃ'] = ['ﭶ', 'ﭸ', 'ﭹ', 'ﭷ'];
        self.forward_mapping['ڄ'] = ['ﭲ', 'ﭴ', 'ﭵ', 'ﭳ'];
        self.forward_mapping['چ'] = ['ﭺ', 'ﭼ', 'ﭽ', 'ﭻ'];
        self.forward_mapping['ڇ'] = ['ﭾ', 'ﮀ', 'ﮁ', 'ﭿ'];
        self.forward_mapping['ڈ'] = ['ﮈ', '', '', 'ﮉ'];
        self.forward_mapping['ڌ'] = ['ﮄ', '', '', 'ﮅ'];
        self.forward_mapping['ڍ'] = ['ﮂ', '', '', 'ﮃ'];
        self.forward_mapping['ڎ'] = ['ﮆ', '', '', 'ﮇ'];
        self.forward_mapping['ڑ'] = ['ﮌ', '', '', 'ﮍ'];
        self.forward_mapping['ژ'] = ['ﮊ', '', '', 'ﮋ'];
        self.forward_mapping['ڤ'] = ['ﭪ', 'ﭬ', 'ﭭ', 'ﭫ'];
        self.forward_mapping['ڦ'] = ['ﭮ', 'ﭰ', 'ﭱ', 'ﭯ'];
        self.forward_mapping['ک'] = ['ﮎ', 'ﮐ', 'ﮑ', 'ﮏ'];
        self.forward_mapping['ڭ'] = ['ﯓ', 'ﯕ', 'ﯖ', 'ﯔ'];
        self.forward_mapping['گ'] = ['ﮒ', 'ﮔ', 'ﮕ', 'ﮓ'];
        self.forward_mapping['ڱ'] = ['ﮚ', 'ﮜ', 'ﮝ', 'ﮛ'];
        self.forward_mapping['ڳ'] = ['ﮖ', 'ﮘ', 'ﮙ', 'ﮗ'];
        self.forward_mapping['ں'] = ['ﮞ', '', '', 'ﮟ'];
        self.forward_mapping['ڻ'] = ['ﮠ', 'ﮢ', 'ﮣ', 'ﮡ'];
        self.forward_mapping['ھ'] = ['ﮪ', 'ﮬ', 'ﮭ', 'ﮫ'];
        self.forward_mapping['ۀ'] = ['ﮤ', '', '', 'ﮥ'];
        self.forward_mapping['ہ'] = ['ﮦ', 'ﮨ', 'ﮩ', 'ﮧ'];
        self.forward_mapping['ۅ'] = ['ﯠ', '', '', 'ﯡ'];
        self.forward_mapping['ۆ'] = ['ﯙ', '', '', 'ﯚ'];
        self.forward_mapping['ۇ'] = ['ﯗ', '', '', 'ﯘ'];
        self.forward_mapping['ۈ'] = ['ﯛ', '', '', 'ﯜ'];
        self.forward_mapping['ۉ'] = ['ﯢ', '', '', 'ﯣ'];
        self.forward_mapping['ۋ'] = ['ﯞ', '', '', 'ﯟ'];
        self.forward_mapping['ی'] = ['ﯼ', 'ﯾ', 'ﯿ', 'ﯽ'];
        self.forward_mapping['ې'] = ['ﯤ', 'ﯦ', 'ﯧ', 'ﯥ'];
        self.forward_mapping['ے'] = ['ﮮ', '', '', 'ﮯ'];
        self.forward_mapping['ۓ'] = ['ﮰ', '', '', 'ﮱ'];
        self.forward_mapping['‍'] = ['‍', '‍', '‍', '‍'];

        self.ligatures = ['لا', 'الله', 'لأ', 'لإ'];

        self.arabic_delimiters = ['،', 'ً', 'ّ', '»'];
        self.delimiters = [' ', ',', '-', '.', '"', ':'];


    def string_to_label_form(self, text):
        text = self._reverse(text)
        return text

    def label_form_to_string(self, text):
        text = self.string_to_label_form(text)
        return text

    def visual_form_to_string(self, text):
        text = self._map_backward(text)
        text = self._reverse(text)
        return text

    def string_to_visual_form(self, text):
        text = self._reshaper.reshape(text)
        text = self._reverse(text)
        return text

    def label_form_to_visual_form(self, text):
        text = self.label_form_to_string(text)
        text = self.string_to_visual_form(text)
        return text

    def visual_form_to_label_form(self, text):
        text = self.visual_form_to_string(text)
        text = self.string_to_label_form(text)
        return text



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

    def _reverse(self, text):
        class Sequence:
            def __init__(self, chars=[], arabic=True):
                self.chars = chars
                self.arabic = arabic

        sequences = []
        seq = Sequence()
        for c in text:
            if c in self.forward_mapping or c in self._backward_mapping or c in self.arabic_delimiters:
                if not seq.arabic:
                    ####
                    if len(seq.chars) > 0:
                        arabic_seq = []
                        number_of_ending_spaces = 0

                        for i in seq.chars[::-1]:
                            if i in self.delimiters:
                                arabic_seq.insert(0, i)
                                number_of_ending_spaces += 1
                            else:
                                break

                        if number_of_ending_spaces > 0:
                            seq.chars = seq.chars[:-number_of_ending_spaces]
                        sequences.append(seq)
                        seq = Sequence(chars=arabic_seq, arabic=True)
                    ####

                    seq.arabic = True

            elif c not in self.delimiters:
                if seq.arabic:
                    if len(seq.chars) > 0:
                        sequences.append(seq)
                        seq = Sequence(chars=[], arabic=False)

                    seq.arabic = False

            seq.chars.append(c)

        ####
        if len(seq.chars) > 0:
            arabic_seq = []
            number_of_ending_spaces = 0

            for i in seq.chars[::-1]:
                if i in self.delimiters:
                    arabic_seq.insert(0, i)
                    number_of_ending_spaces += 1
                else:
                    break

            if number_of_ending_spaces > 0:
                seq.chars = seq.chars[:-number_of_ending_spaces]
            sequences.append(seq)

            if len(arabic_seq):
                seq = Sequence(chars=arabic_seq, arabic=True)
                sequences.append(seq)
        ####

        for index, seq in enumerate(sequences):
            if seq.arabic:
                seq.chars = seq.chars[::-1]
            
        sequences = sequences[::-1]

        reversed_text = ""

        for seq in sequences:
            for c in seq.chars:
                reversed_text += c

        return reversed_text


def log(msg):
    print(msg)

def for_examples(*parameters):
    def tuplify(x):
        if not isinstance(x, tuple):
            return (x,)
        return x

    def decorator(method, parameters=parameters):
        for parameter in (tuplify(x) for x in parameters):
            test_id, *params = parameter
            def method_for_parameter(self, method=method, parameter=params):
                method(self, *parameter)
            
            args_for_parameter = ",".join(repr(v) for v in params)
            name_for_parameter = f"{method.__name__}_{test_id}({args_for_parameter})"
            frame = sys._getframe(1)  # pylint: disable-msg=W0212
            frame.f_locals[name_for_parameter] = method_for_parameter
        
        return None

    return decorator


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
    #     <h1 style="direction: rtl">ليس من الممكن ASDF QWER 12.3 XYZ@FIT.VUTBR.CZ تعيين معامل الانكسار في الشرائح الرقيقة بدقة، لكن في بعض الأحيان يمكن تقديره</h1>
    #     <h1 style="direction: rtl">ليس من الممكن 29/2/2021 - 31/2/2021 تعيين معامل الانكسار</h1>
    #     <h1 style="direction: rtl">ليس من الممكن 29 / 2 / 2021 - 31 / 2 / 2021 تعيين معامل الانكسار</h1>
    #     <h1 style="direction: rtl">ليس من الممكن February 31st تعيين معامل الانكسار</h1>
    #     <h1 style="direction: rtl">الاستخدام في 17% بصريات المعادن</h1>
    #     <h1 style="direction: rtl">الاستخدام في 17 % بصريات المعادن</h1>
    #     <h1 style="direction: rtl">الاستخدام (10 في بصريات) المعادن</h1>
    #     <h1 style="direction: rtl">الاستخدام ( 10 في بصريات ) المعادن</h1>
    #     <h1 style="direction: rtl">الاستخدام (في 10 بصريات) المعادن</h1>
    #     <h1 style="direction: rtl">الاستخدام ( في 10 بصريات ) المعادن</h1>
    # </body>
    # </html>

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

    string_5 = "ليس من الممكن 29/2/2021 - 31/2/2021 تعيين معامل الانكسار"
    labels_5 = "راسكنالا لماعم نييعت 31/2/2021 - 29/2/2021 نكمملا نم سيل"
    visual_5 = "ﺭﺎﺴﻜﻧﻻﺍ ﻞﻣﺎﻌﻣ ﻦﻴﻴﻌﺗ 31/2/2021 - 29/2/2021 ﻦﻜﻤﻤﻟﺍ ﻦﻣ ﺲﻴﻟ"

    string_6 = "ليس من الممكن 29 / 2 / 2021 - 31 / 2 / 2021 تعيين معامل الانكسار"
    labels_6 = "راسكنالا لماعم نييعت 2021 / 2 / 31 - 2021 / 2 / 29 نكمملا نم سيل"
    visual_6 = "ﺭﺎﺴﻜﻧﻻﺍ ﻞﻣﺎﻌﻣ ﻦﻴﻴﻌﺗ 2021 / 2 / 31 - 2021 / 2 / 29 ﻦﻜﻤﻤﻟﺍ ﻦﻣ ﺲﻴﻟ"

    string_7 = "ليس من الممكن February 31st تعيين معامل الانكسار"
    labels_7 = "راسكنالا لماعم نييعت February 31st نكمملا نم سيل"
    visual_7 = "ﺭﺎﺴﻜﻧﻻﺍ ﻞﻣﺎﻌﻣ ﻦﻴﻴﻌﺗ February 31st ﻦﻜﻤﻤﻟﺍ ﻦﻣ ﺲﻴﻟ"

    string_8 = "الاستخدام في 17% بصريات المعادن"
    labels_8 = "نداعملا تايرصب %17 يف مادختسالا"
    visual_8 = "ﻥﺩﺎﻌﻤﻟﺍ ﺕﺎﻳﺮﺼﺑ %17 ﻲﻓ ﻡﺍﺪﺨﺘﺳﻻﺍ"

    string_9 = "الاستخدام في 17 % بصريات المعادن"
    labels_9 = "نداعملا تايرصب % 17 يف مادختسالا"
    visual_9 = "ﻥﺩﺎﻌﻤﻟﺍ ﺕﺎﻳﺮﺼﺑ % 17 ﻲﻓ ﻡﺍﺪﺨﺘﺳﻻﺍ"

    string_10 = "الاستخدام (10 في بصريات) المعادن"
    labels_10 = "نداعملا (تايرصب يف 10) مادختسالا"
    visual_10 = "ﻥﺩﺎﻌﻤﻟﺍ (ﺕﺎﻳﺮﺼﺑ ﻲﻓ 10) ﻡﺍﺪﺨﺘﺳﻻﺍ"


    string_11 = "الاستخدام ( 10 في بصريات ) المعادن"
    labels_11 = "نداعملا ( تايرصب يف 10 ) مادختسالا"
    visual_11 = "ﻥﺩﺎﻌﻤﻟﺍ ( ﺕﺎﻳﺮﺼﺑ ﻲﻓ 10 ) ﻡﺍﺪﺨﺘﺳﻻﺍ"

    string_12 = "الاستخدام (في 10 بصريات) المعادن"
    labels_12 = "نداعملا (تايرصب 10 يف) مادختسالا"
    visual_12 = "ﻥﺩﺎﻌﻤﻟﺍ (ﺕﺎﻳﺮﺼﺑ 10 ﻲﻓ) ﻡﺍﺪﺨﺘﺳﻻﺍ"

    string_13 = "الاستخدام ( في 10 بصريات ) المعادن"
    labels_13 = "نداعملا ( تايرصب 10 يف ) مادختسالا"
    visual_13 = "ﻥﺩﺎﻌﻤﻟﺍ ( ﺕﺎﻳﺮﺼﺑ 10 ﻲﻓ ) ﻡﺍﺪﺨﺘﺳﻻﺍ"

    #
    #  String to Label
    #
    @for_examples((1, string_1, labels_1),
                  (2, string_2, labels_2),
                  (3, string_3, labels_3),
                  (4, string_4, labels_4),
                  (5, string_5, labels_5),
                  (6, string_6, labels_6),
                  (7, string_7, labels_7),
                  (8, string_8, labels_8),
                  (9, string_9, labels_9),
                  (10, string_10, labels_10),
                  (11, string_11, labels_11),
                  (12, string_12, labels_12),
                  (13, string_13, labels_13))
    def test_string_to_label_form(self, _string, _label_form):
        helper = ArabicHelper()
        output = helper.string_to_label_form(_string)
        self.assertEqual(output, _label_form)

    #
    #  Label to String
    #
    @for_examples((1, labels_1, string_1),
                  (2, labels_2, string_2),
                  (3, labels_3, string_3),
                  (4, labels_4, string_4),
                  (5, labels_5, string_5),
                  (6, labels_6, string_6),
                  (7, labels_7, string_7),
                  (8, labels_8, string_8),
                  (9, labels_9, string_9),
                  (10, labels_10, string_10),
                  (11, labels_11, string_11),
                  (12, labels_12, string_12),
                  (13, labels_13, string_13))
    def test_label_form_to_string(self, _label_form, _string):
        helper = ArabicHelper()
        output = helper.label_form_to_string(_label_form)
        self.assertEqual(output, _string)

    #
    #  String to Visual
    #
    @for_examples((1, string_1, visual_1),
                  (2, string_2, visual_2),
                  (3, string_3, visual_3),
                  (4, string_4, visual_4),
                  (5, string_5, visual_5),
                  (6, string_6, visual_6),
                  (7, string_7, visual_7),
                  (8, string_8, visual_8),
                  (9, string_9, visual_9),
                  (10, string_10, visual_10),
                  (11, string_11, visual_11),
                  (12, string_12, visual_12),
                  (13, string_13, visual_13))
    def test_string_to_visual_form(self, _string, _visual_form):
        helper = ArabicHelper()
        output = helper.string_to_visual_form(_string)
        self.assertEqual(output, _visual_form)

    #
    #  Visual to String
    #
    @for_examples((1, visual_1, string_1),
                  (2, visual_2, string_2),
                  (3, visual_3, string_3),
                  (4, visual_4, string_4),
                  (5, visual_5, string_5),
                  (6, visual_6, string_6),
                  (7, visual_7, string_7),
                  (8, visual_8, string_8),
                  (9, visual_9, string_9),
                  (10, visual_10, string_10),
                  (11, visual_11, string_11),
                  (12, visual_12, string_12),
                  (13, visual_13, string_13))
    def test_visual_form_to_string(self, _visual_form, _string):
        helper = ArabicHelper()
        output = helper.visual_form_to_string(_visual_form)
        self.assertEqual(output, _string)

    #
    #  Label to Visual
    #
    @for_examples((1, labels_1, visual_1),
                  (2, labels_2, visual_2),
                  (3, labels_3, visual_3),
                  (4, labels_4, visual_4),
                  (5, labels_5, visual_5),
                  (6, labels_6, visual_6),
                  (7, labels_7, visual_7),
                  (8, labels_8, visual_8),
                  (9, labels_9, visual_9),
                  (10, labels_10, visual_10),
                  (11, labels_11, visual_11),
                  (12, labels_12, visual_12),
                  (13, labels_13, visual_13))
    def test_label_form_to_visual_form(self, _label_form, _visual_form):
        helper = ArabicHelper()
        output = helper.label_form_to_visual_form(_label_form)
        self.assertEqual(output, _visual_form)

    #
    #  Visual to Label
    #
    @for_examples((1, visual_1, labels_1),
                  (2, visual_2, labels_2),
                  (3, visual_3, labels_3),
                  (4, visual_4, labels_4),
                  (5, visual_5, labels_5),
                  (6, visual_6, labels_6),
                  (7, visual_7, labels_7),
                  (8, visual_8, labels_8),
                  (9, visual_9, labels_9),
                  (10, visual_10, labels_10),
                  (11, visual_11, labels_11),
                  (12, visual_12, labels_12),
                  (13, visual_13, labels_13))
    def test_visual_form_to_label_form(self, _visual_form, _label_form):
        helper = ArabicHelper()
        output = helper.visual_form_to_label_form(_visual_form)
        self.assertEqual(output, _label_form)


if __name__ == "__main__":
    unittest.main()
