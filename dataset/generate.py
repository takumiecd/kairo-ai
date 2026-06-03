import pykakasi
from sudachipy import tokenizer
from sudachipy import dictionary

class DatasetGenerator:
    """
    既存の日本語テキスト（例：「私は昨日ピザを食べた」）から、
    タイピング時のローマ字入力（例：「watashihakinoupizawotabeta」）を自動生成するクラス。
    """
    def __init__(self):
        # 形態素解析器 (漢字 -> 読み)
        self.tokenizer_obj = dictionary.Dictionary().create()
        self.mode = tokenizer.Tokenizer.SplitMode.C
        
        # ローマ字変換器 (読み -> ローマ字)
        self.kks = pykakasi.kakasi()
        
    def generate_pair(self, text: str):
        """
        Input: "私は昨日、AppleのMacBookを買いました。"
        Output: ("watashihakinou、ApplenoMacBookwokaimashita。", "私は昨日、AppleのMacBookを買いました。")
        """
        # 1. 形態素解析で「読み」を取得
        tokens = self.tokenizer_obj.tokenize(text, self.mode)
        
        romaji_parts = []
        for token in tokens:
            surface = token.surface()
            reading = token.reading_form()
            
            # 英単語などは読みが取得できない、または元の文字を維持したいケースがある
            # 簡単なヒューリスティック：読みが存在すればよみがなを、なければ表層をそのまま使う
            target_text = reading if reading else surface
            
            # カタカナ/ひらがなをローマ字に変換
            conv_result = self.kks.convert(target_text)
            for item in conv_result:
                # pykakasiは辞書で {orig, hira, kana, hepburn} などを返す
                romaji_parts.append(item['hepburn'])
                
        input_romaji = "".join(romaji_parts)
        
        return input_romaji, text

if __name__ == "__main__":
    generator = DatasetGenerator()
    
    # テスト
    sample_texts = [
        "私は昨日、AppleのMacBookを買いました。",
        "git commit -m \"バグを修正した\"",
        "こんにちは、世界！"
    ]
    
    for txt in sample_texts:
        x, y = generator.generate_pair(txt)
        print(f"Target (Y) : {y}")
        print(f"Input  (X) : {x}")
        print("-" * 40)
