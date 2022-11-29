from transformers import GPTNeoXTokenizerFast
import logging
import re
import fasttext
from codepile.stats.src.flagged_words import english_flagged_words
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PATH_FASTTEXT = ""


def get_words(text: str) -> list:
    """custom regex to extract all the words in a string"""
    return re.findall(r'\w+', text.lower())

def load_fasttext_model(path_fasttext_model:str=PATH_FASTTEXT):
    return fasttext.load_model(path_fasttext_model)


def get_fasttext_info(line, model_lang_id):
    """The line should be in lower case and without \n in it."""
    pred = model_lang_id.predict(line)
    lang_pred_fasttext_id = pred[0][0].replace("__label__", "")
    score_pred = pred[1][0]
    return lang_pred_fasttext_id, score_pred


def get_all_fasttext_info(document, model_lang_id):
    document = document.lower()
    lang_pred_fasttext_id, score_pred = get_fasttext_info(
        document.replace("\n", " "), model_lang_id
    )
    info = {
        "lang_pred_fasttext_id": lang_pred_fasttext_id,
        "score_pred": score_pred,
        "on_lines": [
            {
                "id_line": id_line,
                "number_caracters_line": len(line),
                "lang_pred_fasttext_id_line": result_fasttext_line[0],
                "score_pred_line": result_fasttext_line[1],
            }
            for id_line, line in enumerate(document.split("\n"))
            for result_fasttext_line in [get_fasttext_info(line, model_lang_id)]
        ],
    }
    return info


class LangDetection:
    #adapted from https://github.com/bigcode-project/bigcode-analysis/blob/main/data_analysis/python_data_analysis/nl_language_identification/language_identifier.py
    def __init__(self) -> None:
        self.lang_model_path = "stats/utils/lid.176.bin"
        self.model = load_fasttext_model("src/lid.176.bin")
        self.model.add_pipe("language_detector")


    def detect(self, text: str) -> str:
        """
        Detects the language of the text
        args:
            text (str) : Text to detect the language

        returns:
            language (str) : Predicted Language of the text
            score_pred (str) : confidence of the prediction

        """
        text = text.lower()

        fasttext_pred = get_all_fasttext_info(
            text, self.model
        )
        return fasttext_pred["nl_language"], fasttext_pred["score_pred"]

    def detect_batch(self, text_list: list[str]) -> list[str]:
        return [self.detect(text) for text in text_list]


class Tokenization:
    def __init__(self, model_name="EleutherAI/gpt-neox-20b") -> None:
        self.tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_name)

        logger.info(f"Sucessfully loaded {model_name}")

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def tokenize_batch(self, text_list: list[str]) -> list[list[int]]:
        return self.tokenizer.encode(text_list)


class FlaggedWords:
    def __init__(self) -> None:
        self.flagged_words = english_flagged_words
    
    def check_flagged_words(self, text: str) -> list[str]:
        words = get_words(text)
        check_list = [word for word in words if word in self.flagged_words]
        if len(check_list) > 1:
            return True
        else:
            return False


stat_config_map: dict[str] = {
    "lang_idt": LangDetection,
    "tokenize": Tokenization(),
    "len_char": lambda doc: len(doc),
    "len_utf8bytes": lambda doc: len(doc.encode("utf-8")),
    "len_words": lambda doc: len(re.split(r"\s+", doc)),
    "flagged_words" : lambda doc: FlaggedWords().check_flagged_words,
}

