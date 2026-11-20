"""
preprocess.py - เตรียมข้อความภาษาไทยก่อนส่งเข้าโมเดล TH->EN

โมเดลเห็นตัวย่อเดือน/วันแบบไทย (ม.ค., จ.-ศ.) และปี พ.ศ. น้อยมากในข้อมูลเทรน จึงแปลพัง
เช่น "5 ม.ค. 2569" -> "Semi-Supervised 209" แต่ถ้าเขียนเต็ม "5 มกราคม 2026" จะแปลถูก
ไฟล์นี้จึงแปลงรูปแบบที่โมเดลไม่คุ้นให้เป็นรูปแบบที่โมเดลแปลได้ ก่อนส่งเข้าโมเดล
ทุกกฎตรวจบริบทก่อนแปลง (เช่น ต้องอยู่ติดตัวเลข) เพื่อไม่ให้ไปแก้คำทั่วไป
"""
import re

THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

MONTH_ABBR = {
    "ม.ค.": "มกราคม", "ก.พ.": "กุมภาพันธ์", "มี.ค.": "มีนาคม", "เม.ย.": "เมษายน",
    "พ.ค.": "พฤษภาคม", "มิ.ย.": "มิถุนายน", "ก.ค.": "กรกฎาคม", "ส.ค.": "สิงหาคม",
    "ก.ย.": "กันยายน", "ต.ค.": "ตุลาคม", "พ.ย.": "พฤศจิกายน", "ธ.ค.": "ธันวาคม",
}
MONTH_FULL = "|".join(MONTH_ABBR.values())

DAY_ABBR = {"อา.": "วันอาทิตย์", "จ.": "วันจันทร์", "อ.": "วันอังคาร", "พ.": "วันพุธ",
            "พฤ.": "วันพฤหัสบดี", "ศ.": "วันศุกร์", "ส.": "วันเสาร์"}

WORD_ABBR = {"กทม.": "กรุงเทพมหานคร", "ผอ.": "ผู้อำนวยการ", "ฯลฯ": "และอื่น ๆ"}

_NOT_THAI_BEFORE = r"(?<![ก-๎])"


def _abbr_regex(abbr):
    # "ม.ค." -> ม\s*\.\s*ค\s*\.   (รับเว้นวรรคแทรก เช่น "ม. ค.")
    parts = [re.escape(p) for p in abbr.split(".") if p]
    return r"\s*\.\s*".join(parts) + r"\s*\."


_MONTH_RE = [(re.compile(_NOT_THAI_BEFORE + _abbr_regex(a)), full) for a, full in
             sorted(MONTH_ABBR.items(), key=lambda kv: -len(kv[0]))]
_DAY_TOKEN = r"(?:อา|พฤ|จ|อ|พ|ศ|ส)\s*\."
# ใช้เฉพาะเมื่อเป็นชุดวันย่อ 2 ตัวขึ้นไป (เช่น "จ.-ศ.", "จ. พ. ศ.") เพราะ "จ." ตัวเดียวอาจหมายถึง "จังหวัด"
_DAY_RUN = re.compile(_NOT_THAI_BEFORE + rf"{_DAY_TOKEN}(?:\s*[-–,/]?\s*{_DAY_TOKEN})+")
_BE_YEAR = re.compile(r"(?:พ\s*\.\s*ศ\s*\.\s*)(\d{4})")
_CE_YEAR = re.compile(r"ค\s*\.\s*ศ\s*\.\s*(\d{4})")
_YEAR_AFTER_MONTH = re.compile(rf"({MONTH_FULL})\s*(\d{{4}})")
_TIME_TH = re.compile(r"(\d{1,2})[.:](\d{2})\s*น\s*\.")


def _be_to_ce(year: str) -> str:
    y = int(year)
    return str(y - 543) if 2400 <= y <= 2700 else year


def preprocess_th(text: str) -> str:
    t = text.translate(THAI_DIGITS)
    t = _BE_YEAR.sub(lambda m: _be_to_ce(m.group(1)), t)              # พ.ศ. 2569 -> 2026
    t = _CE_YEAR.sub(r"\1", t)                                         # ค.ศ. 2026 -> 2026
    for pat, full in _MONTH_RE:                                        # ม.ค. -> มกราคม
        t = pat.sub(full + " ", t)
    t = _YEAR_AFTER_MONTH.sub(lambda m: f"{m.group(1)} {_be_to_ce(m.group(2))}", t)  # มกราคม 2569 -> มกราคม 2026
    t = _DAY_RUN.sub(lambda m: re.sub(_DAY_TOKEN, lambda d: DAY_ABBR[re.sub(r"\s", "", d.group(0))] + " ", m.group(0)), t)
    t = _TIME_TH.sub(lambda m: f"{int(m.group(1))}:{m.group(2)}", t)  # 23.45 น. -> 23:45
    for a, full in WORD_ABBR.items():
        t = t.replace(a, full)
    return re.sub(r"[ \t]+", " ", t).strip()


# ---------------------------------------------------------------------------
# ใส่อักขระที่ tokenizer ไม่รู้จักคืนหลังแปล
# tokenizer ถูกเทรนด้วย character_coverage=0.9995 อักขระที่พบน้อย (@ # อีโมจิ ★) จึงกลายเป็น <unk>
# และโมเดลแปลออกมาเป็น " ⁇ " ฟังก์ชันนี้จำอักขระเหล่านั้นจากต้นฉบับ แล้วใส่คืนตามลำดับ
# ---------------------------------------------------------------------------
UNK_MARK = "⁇"   # ⁇ ที่ SentencePiece ใช้แทน <unk> ตอน decode


def unknown_spans(text, sp):
    unk = sp.unk_id()
    spans, cur = [], ""
    for ch in text:
        is_unk = (not ch.isspace()) and unk in sp.encode(ch, out_type=int)
        if is_unk:
            cur += ch
        elif cur:
            spans.append(cur); cur = ""
    if cur:
        spans.append(cur)
    return spans


def restore_unknown(translated, source, sp):
    if UNK_MARK not in translated:
        return translated
    spans = iter(unknown_spans(source, sp))
    def put_back(m):
        sym = next(spans, "")
        if not sym:
            return " "
        # เคารพการเว้นวรรครอบอักขระตามต้นฉบับ เช่น "test@example.com" ไม่มีช่องว่างรอบ @
        i = source.find(sym)
        before = " " if i > 0 and source[i - 1].isspace() else ""
        after = " " if i + len(sym) < len(source) and source[i + len(sym)].isspace() else ""
        return before + sym + after
    out = re.sub(r"\s*" + UNK_MARK + r"\s*", put_back, translated)
    return re.sub(r"[ \t]{2,}", " ", out).strip()
