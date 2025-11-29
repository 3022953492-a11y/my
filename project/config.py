import os

class Config:
    """项目配置文件"""
    
    # 数据路径配置 - 使用根目录下的dataset
    DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    IMAGES_DIR1 = os.path.join(DATA_ROOT, "images")
    IMAGES_DIR2 = os.path.join(DATA_ROOT, "images1")
    LABELS_DIR1 = os.path.join(DATA_ROOT, "labels")
    LABELS_DIR2 = os.path.join(DATA_ROOT, "labels1")
    LABELS_FILE = os.path.join(DATA_ROOT, "labels.txt")
    
    # 数据集标注和词汇表文件
    ANNOTATION_FILE = os.path.join(DATA_ROOT, "annotations.json")
    VOCAB_FILE = os.path.join(DATA_ROOT, "vocab.json")
    
    # 数据集划分比例
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # 图像预处理参数
    IMG_SIZE = (224, 224)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # 模型参数
    VOCAB_SIZE = 84   # 词汇表大小（基于完整数据集vocab_full.json）
    EMBED_DIM = 256   # 嵌入维度
    HIDDEN_DIM = 512  # 隐藏层维度
    NUM_LAYERS = 3    # LSTM层数
    
    # 训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    PATIENCE = 10     # 早停耐心值
    
    # 字符集（化学方程式常用字符）
    CHAR_SET = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        '+', '-', '=', '(', ')', '[', ']', '{', '}',
        '.', ',', ';', ':', '!', '?',
        ' ', '\t', '\n', '\r',
        # 化学特殊符号
        '↑', '↓', '⇌', '→', '←', '↔',
        '°', '℃', '℉', 'Δ', 'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
        '·', '×', '÷', '±', '≈', '≠', '≤', '≥', '∞', '∝', '∫', '∑', '∏', '∂', '∇', '√',
        '∠', '⊥', '∥', '≅', '∼', '≡', '≢', '⊂', '⊃', '⊆', '⊇', '∪', '∩', '∈', '∉', '∋', '∌', '∅',
        '∀', '∃', '∴', '∵',
        '─', '│', '┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼',
        '═', '║', '╒', '╓', '╔', '╕', '╖', '╗', '╘', '╙', '╚', '╛', '╜', '╝', '╞', '╟',
        '╠', '╡', '╢', '╣', '╤', '╥', '╦', '╧', '╨', '╩', '╪', '╫', '╬',
        '■', '□', '▪', '▫', '▲', '△', '▶', '▷', '▼', '▽', '◆', '◇', '○', '●',
        '★', '☆', '☀', '☁', '☂', '☃', '☄',
        '♀', '♂', '♠', '♡', '♢', '♣', '♤', '♥', '♦', '♧',
        '✓', '✔', '✕', '✖', '✗', '✘',
        '❤', '➔', '➕', '➖', '➗',
        '⟶', '⟷', '⟸', '⟹', '⟺'
    ]
    
    # 特殊标记字符
    PAD_TOKEN = '<PAD>'
    SOS_TOKEN = '<SOS>'
    EOS_TOKEN = '<EOS>'
    UNK_TOKEN = '<UNK>'
    
    def __init__(self):
        # 构建完整的字符表
        self.char2idx = {}
        self.idx2char = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """构建字符表"""
        # 添加特殊标记
        special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        
        # 构建字符映射
        all_chars = special_tokens + self.CHAR_SET
        for idx, char in enumerate(all_chars):
            self.char2idx[char] = idx
            self.idx2char[idx] = char
        
        # 更新词汇表大小
        self.VOCAB_SIZE = len(all_chars)
    
    def get_char_idx(self, char):
        """获取字符索引"""
        return self.char2idx.get(char, self.char2idx[self.UNK_TOKEN])
    
    def get_idx_char(self, idx):
        """获取索引对应的字符"""
        return self.idx2char.get(idx, self.UNK_TOKEN)