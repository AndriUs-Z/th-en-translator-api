import gc
import math
import os
import re

import sentencepiece as spm
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

torch.set_num_threads(int(os.getenv("TORCH_THREADS", "1")))
torch.set_num_interop_threads(1)
device = torch.device("cpu")

# ---------------------------------------------------------------------------
# Hyperparameters: ต้องตรงกับ notebook ตอน train ทุกค่า (ไม่งั้น load weight ไม่ได้)
# ---------------------------------------------------------------------------
D_MODEL = 256
NHEAD = 8
ENC_GRU_HIDDEN = 64
DEC_GRU_HIDDEN = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
MAX_LEN = 128

# ตั้งให้ตรงกับ Cell 11 ใน notebook (ค่าชุดเดียวกับที่วัด BLEU)
# เดิม MAX_NEW_TOKENS=64 ทำให้ประโยคยาวถูกตัดกลางคัน, REPETITION_PENALTY=1.2 และไม่ได้ใช้ no-repeat n-gram
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "150"))
NO_REPEAT_NGRAM_SIZE = 3
REPETITION_PENALTY = 1.3

# ความยาว source ต่อ 1 chunk (หน่วย token) ต้องน้อยกว่า MAX_LEN - 2 (เผื่อ BOS/EOS)
SRC_CHUNK_TOKENS = min(int(os.getenv("SRC_CHUNK_TOKENS", "100")), MAX_LEN - 2)
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "5000"))

# ชื่อไฟล์ตาม output จาก notebook: sp_en_th.model และ finetuned_model_th_en.pt
SPM_TH_EN_PATH = os.getenv("SPM_TH_EN_PATH", "sp_en_th.model")
MODEL_TH_EN_PATH = os.getenv("MODEL_TH_EN_PATH", "finetuned_model_th_en.pt")
FALLBACK_MODEL_PATH = "best_model_th_en.pt"  # ใช้เมื่อไม่มี finetuned

if not os.path.exists(SPM_TH_EN_PATH):
    raise FileNotFoundError(f"ไม่พบไฟล์ tokenizer: {SPM_TH_EN_PATH}")

print("[*] กำลังโหลด SentencePiece Tokenizer (TH-EN)...")
sp_th_en = spm.SentencePieceProcessor(model_file=SPM_TH_EN_PATH)
vocab_size_th_en = sp_th_en.get_piece_size()

PAD_ID = sp_th_en.pad_id() if sp_th_en.pad_id() != -1 else 0
BOS_ID = sp_th_en.bos_id() if sp_th_en.bos_id() != -1 else 2
EOS_ID = sp_th_en.eos_id() if sp_th_en.eos_id() != -1 else 3


# ---------------------------------------------------------------------------
# Model (เหมือน notebook Cell 6)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class RNNAugmentedEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, gru_hidden, dim_feedforward, dropout):
        super().__init__()
        self.gru = nn.GRU(d_model, gru_hidden, bidirectional=True, batch_first=True)
        self.gru_proj = nn.Linear(gru_hidden * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        gru_out, _ = self.gru(x)
        gru_out = self.gru_proj(gru_out)
        x = self.norm1(x + self.dropout(gru_out))
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask, need_weights=False)
        x = self.norm2(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        return self.norm3(x + self.dropout(ff_out))


class RNNAugmentedDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, gru_hidden, dim_feedforward, dropout):
        super().__init__()
        self.gru = nn.GRU(d_model, gru_hidden, bidirectional=False, batch_first=True)
        self.gru_proj = nn.Linear(gru_hidden, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model)
        )
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        gru_out, _ = self.gru(x)
        gru_out = self.gru_proj(gru_out)
        x = self.norm1(x + self.dropout(gru_out))
        attn_out, _ = self.self_attn(
            x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False
        )
        x = self.norm2(x + self.dropout(attn_out))
        cross_out, _ = self.cross_attn(
            x, memory, memory, key_padding_mask=memory_key_padding_mask, need_weights=False
        )
        x = self.norm3(x + self.dropout(cross_out))
        ff_out = self.ff(x)
        return self.norm4(x + self.dropout(ff_out))


class RNNAugmentedTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        enc_gru_hidden=ENC_GRU_HIDDEN,
        dec_gru_hidden=DEC_GRU_HIDDEN,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_seq_len=MAX_LEN,
        pad_id=PAD_ID,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.encoder_layers = nn.ModuleList(
            [RNNAugmentedEncoderLayer(d_model, nhead, enc_gru_hidden, dim_feedforward, dropout) for _ in range(num_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [RNNAugmentedDecoderLayer(d_model, nhead, dec_gru_hidden, dim_feedforward, dropout) for _ in range(num_decoder_layers)]
        )
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.embed_scale = math.sqrt(d_model)

    def make_padding_mask(self, seq):
        return seq == self.pad_id

    def make_causal_mask(self, size, dev):
        return torch.triu(torch.ones(size, size, device=dev), diagonal=1).bool()

    def encode(self, src):
        src_key_padding_mask = self.make_padding_mask(src)
        x = self.embedding(src) * self.embed_scale
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x, src_key_padding_mask

    def decode(self, tgt, memory, memory_key_padding_mask=None):
        tgt_key_padding_mask = self.make_padding_mask(tgt)
        tgt_mask = self.make_causal_mask(tgt.size(1), tgt.device)
        x = self.embedding(tgt) * self.embed_scale
        x = self.pos_encoding(x)
        for layer in self.decoder_layers:
            x = layer(
                x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------
model_info = {"path": None, "alpha": None}


def resolve_model_path():
    for path in (MODEL_TH_EN_PATH, FALLBACK_MODEL_PATH):
        if os.path.exists(path):
            if path != MODEL_TH_EN_PATH:
                print(f"[!] ไม่พบ {MODEL_TH_EN_PATH} -> ใช้ {path} (โมเดลก่อน fine-tune) แทน")
            return path
    return None


def load_checkpoint_safe(model_path, target_vocab_size):
    if model_path is None:
        print(f"[!] ไม่พบไฟล์โมเดล ({MODEL_TH_EN_PATH} / {FALLBACK_MODEL_PATH})")
        return None

    # weights_only=True: ไม่ unpickle โค้ดใด ๆ (ปลอดภัยกว่า) checkpoint จาก notebook เป็น dict ของ tensor/ตัวเลข จึงโหลดได้
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    # ตรวจ vocab ให้ตรงก่อน: ถ้า tokenizer ไม่ใช่ตัวที่ใช้ train จะได้ error ที่อ่านรู้เรื่อง
    ckpt_vocab = state_dict["output_proj.weight"].shape[0]
    if ckpt_vocab != target_vocab_size:
        raise RuntimeError(
            f"vocab ไม่ตรงกัน: โมเดล={ckpt_vocab} แต่ tokenizer={target_vocab_size} "
            f"-> ต้องใช้ {SPM_TH_EN_PATH} ตัวเดียวกับที่ใช้ train"
        )

    model = RNNAugmentedTransformer(vocab_size=target_vocab_size).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if isinstance(ckpt, dict):
        model_info["alpha"] = ckpt.get("alpha")
        if model_info["alpha"] is not None:
            print(f"[*] fine-tune interpolation alpha = {model_info['alpha']}")
            if float(model_info["alpha"]) == 0.0:
                print("[!] alpha=0 แปลว่า fine-tune ถูกปฏิเสธ (ลืมเกินเกณฑ์) -> โมเดลนี้เท่ากับก่อน fine-tune")
    model_info["path"] = model_path

    del ckpt, state_dict
    gc.collect()
    return model


print("[*] กำลังโหลดโมเดล TH-EN...")
model_th_en = load_checkpoint_safe(resolve_model_path(), vocab_size_th_en)


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------
def encode_text(text: str, target_sp, add_bos_eos: bool = True):
    ids = target_sp.encode(text, out_type=int)
    if add_bos_eos:
        # ตัดเนื้อหาก่อนแล้วค่อยใส่ BOS/EOS (เดิมตัดทีหลังทำให้ EOS หาย เมื่อประโยคยาวเกิน)
        ids = [BOS_ID] + ids[: MAX_LEN - 2] + [EOS_ID]
    return ids[:MAX_LEN]


def decode_ids(ids, target_sp):
    ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
    return target_sp.decode(ids)


def get_banned_ngram_tokens(generated_ids, ngram_size):
    if len(generated_ids) < ngram_size - 1:
        return set()
    prefix = tuple(generated_ids[-(ngram_size - 1):])
    banned = set()
    for i in range(len(generated_ids) - ngram_size + 1):
        ngram = tuple(generated_ids[i: i + ngram_size])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])
    return banned


@torch.no_grad()
def greedy_translate(model, src_ids):
    model.eval()
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    memory, memory_key_padding_mask = model.encode(src)

    safe_max_len = min(MAX_NEW_TOKENS, MAX_LEN - 1)
    tgt_ids = [BOS_ID]

    for _ in range(safe_max_len):
        tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
        logits = model.decode(tgt, memory, memory_key_padding_mask=memory_key_padding_mask)
        next_token_logits = logits[0, -1, :].clone()

        # repetition penalty + no-repeat n-gram: เหมือน notebook Cell 11 ทุกประการ
        for tok in set(tgt_ids):
            if next_token_logits[tok] > 0:
                next_token_logits[tok] /= REPETITION_PENALTY
            else:
                next_token_logits[tok] *= REPETITION_PENALTY

        for tok in get_banned_ngram_tokens(tgt_ids, NO_REPEAT_NGRAM_SIZE):
            next_token_logits[tok] = float("-inf")

        next_token = torch.argmax(next_token_logits).item()
        tgt_ids.append(next_token)
        if next_token == EOS_ID:
            break

    return tgt_ids


# ---------------------------------------------------------------------------
# แบ่งข้อความยาวเป็น chunk ก่อนแปล
# ---------------------------------------------------------------------------
def capitalize_first(text: str) -> str:
    # str.capitalize() จะทำให้ตัวอักษรที่เหลือเป็นตัวพิมพ์เล็กหมด (API -> api, JSON -> json)
    # จึงเปลี่ยนเฉพาะตัวแรก
    return text[:1].upper() + text[1:] if text else text


def pack_by_words(text, count_tokens, limit):
    """ภาษาไทยคั่นวลี/ประโยคด้วยช่องว่าง: รวมคำให้ได้ chunk ยาวไม่เกิน limit token"""
    chunks, current = [], []
    for word in text.split():
        candidate = " ".join(current + [word])
        if current and count_tokens(candidate) > limit:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current))
    return chunks


def split_into_paragraphs(text, sp, limit=SRC_CHUNK_TOKENS):
    """
    คืนค่า list ของย่อหน้า แต่ละย่อหน้าเป็น list ของ (chunk_text, starts_sentence)
    - แบ่งย่อหน้าด้วย \\n  - แบ่งประโยคด้วย . ! ? ;  - ถ้ายังยาวเกิน limit แบ่งตามช่องว่าง
    - ถ้ายังมีคำเดียวที่ยาวเกิน (ไทยไม่เว้นวรรค) แบ่งตาม token
    ป้องกันกรณีเดิมที่ประโยคเกิน 128 token ถูกตัดท้ายทิ้งเงียบ ๆ
    """
    def count_tokens(s):
        return len(sp.encode(s, out_type=int))

    paragraphs = []
    for para in text.split("\n"):
        para = para.strip()
        chunks = []
        if para:
            for sent in re.split(r"(?<=[.!?;])\s+", para):
                sent = sent.strip()
                if not sent:
                    continue
                if count_tokens(sent) <= limit:
                    chunks.append((sent, True))
                    continue
                first = True
                for piece in pack_by_words(sent, count_tokens, limit):
                    if count_tokens(piece) <= limit:
                        chunks.append((piece, first))
                        first = False
                    else:
                        ids = sp.encode(piece, out_type=int)
                        for i in range(0, len(ids), limit):
                            chunks.append((sp.decode(ids[i: i + limit]), first))
                            first = False
        paragraphs.append(chunks)
    return paragraphs


def translate_long_text(model, text: str, target_sp) -> str:
    if not text or not text.strip():
        return ""
    out_paragraphs = []
    for chunks in split_into_paragraphs(text.strip(), target_sp):
        parts = []
        for chunk_text, starts_sentence in chunks:
            out_ids = greedy_translate(model, encode_text(chunk_text, target_sp))
            piece = decode_ids(out_ids, target_sp).strip()
            if not piece:
                continue
            parts.append(capitalize_first(piece) if starts_sentence else piece)
        out_paragraphs.append(" ".join(parts))
    return "\n".join(out_paragraphs).strip()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(title="CS Translator API (TH-EN Microservice)", version="1.3")

# ตั้ง ALLOWED_ORIGINS="https://your-site.com,https://other.com" ตอน deploy จริง (ค่าเริ่มต้น * เพื่อให้ทดสอบง่าย)
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,  # API นี้ไม่ใช้ cookie จึงไม่ต้องเปิด (และ "*" ใช้คู่กับ credentials ไม่ได้ตามมาตรฐาน)
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslationRequest(BaseModel):
    text: str
    direction: str = "th-en"


class TranslationResponse(BaseModel):
    translated_text: str
    direction: str


@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "TH-EN Translator",
        "model_loaded": model_th_en is not None,
        "model_file": model_info["path"],
        "finetune_alpha": model_info["alpha"],
    }


@app.post("/translate", response_model=TranslationResponse)
def translate(req: TranslationRequest):
    direction = req.direction.strip().lower()
    input_text = req.text.strip()

    if direction != "th-en":
        raise HTTPException(status_code=400, detail="API นี้รองรับเฉพาะ 'th-en' เท่านั้น")

    if model_th_en is None:
        raise HTTPException(status_code=503, detail="ไฟล์โมเดล TH-EN ยังไม่พร้อมใช้งาน")

    if len(input_text) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=413, detail=f"ข้อความยาวเกิน {MAX_INPUT_CHARS} ตัวอักษร")

    result = translate_long_text(model_th_en, input_text, sp_th_en)
    return TranslationResponse(translated_text=result, direction="th-en")
