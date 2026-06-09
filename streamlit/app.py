import io
import html
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn


# =============================
# Model definition
# =============================

class ConstrainedCRF(nn.Module):
    """Minimal CRF decoder compatible with the saved SimpleRNN-CRF checkpoint.

    The checkpoint contains learned transitions and BILOU constraint penalties.
    This class focuses on Viterbi decoding for inference in Streamlit.
    """

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))

        # These are loaded from the checkpoint. Invalid BILOU transitions normally
        # contain a large negative value such as -10000.
        self.register_buffer("start_penalties", torch.zeros(num_tags))
        self.register_buffer("transition_penalties", torch.zeros(num_tags, num_tags))
        self.register_buffer("end_penalties", torch.zeros(num_tags))

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """Viterbi decode.

        Args:
            emissions: Tensor with shape [batch_size, seq_len, num_tags].
            mask: Boolean tensor with shape [batch_size, seq_len].

        Returns:
            A list of predicted tag-id sequences, one sequence per item.
        """
        batch_paths: List[List[int]] = []
        transition_scores = self.transitions + self.transition_penalties
        start_scores = self.start_transitions + self.start_penalties
        end_scores = self.end_transitions + self.end_penalties

        batch_size = emissions.size(0)
        for b in range(batch_size):
            seq_len = int(mask[b].sum().item())
            if seq_len <= 0:
                batch_paths.append([])
                continue

            emit = emissions[b, :seq_len]  # [seq_len, num_tags]
            score = start_scores + emit[0]
            history: List[torch.Tensor] = []

            for t in range(1, seq_len):
                # score(prev_tag) + transition(prev_tag -> next_tag) + emission(next_tag)
                next_score = score.unsqueeze(1) + transition_scores + emit[t].unsqueeze(0)
                best_score, best_prev_tag = next_score.max(dim=0)
                score = best_score
                history.append(best_prev_tag)

            score = score + end_scores
            best_last_tag = int(score.argmax().item())

            best_path = [best_last_tag]
            for best_prev_tag in reversed(history):
                best_last_tag = int(best_prev_tag[best_last_tag].item())
                best_path.append(best_last_tag)
            best_path.reverse()
            batch_paths.append(best_path)

        return batch_paths


class SimpleRNNCRFTagger(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0,
        bidirectional: bool = False,
        rnn_type: str = "RNN",
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_cls = nn.RNN if rnn_type.upper() == "RNN" else nn.GRU
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.emission_layer = nn.Linear(output_dim, num_tags)
        self.crf = ConstrainedCRF(num_tags)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        rnn_out, _ = self.rnn(embedded)
        emissions = self.emission_layer(self.dropout(rnn_out))
        return emissions


# =============================
# Text processing and decoding
# =============================

TOKEN_PATTERN = re.compile(
    r"https?://\S+|[\wÀ-ÖØ-öø-ÿ]+(?:[-./][\wÀ-ÖØ-öø-ÿ]+)*|[^\w\s]",
    flags=re.UNICODE,
)


def tokenize_with_offsets(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokens: List[str] = []
    offsets: List[Tuple[int, int]] = []
    for match in TOKEN_PATTERN.finditer(text):
        tokens.append(match.group(0))
        offsets.append((match.start(), match.end()))
    return tokens, offsets


def safe_torch_load(source):
    """Load a user-provided PyTorch checkpoint across PyTorch versions."""
    try:
        return torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(source, map_location="cpu")


@st.cache_resource(show_spinner="Loading model...")
def load_checkpoint_model(model_bytes: Optional[bytes], default_model_path: str):
    if model_bytes is not None:
        checkpoint = safe_torch_load(io.BytesIO(model_bytes))
    else:
        checkpoint = safe_torch_load(default_model_path)

    token2idx: Dict[str, int] = checkpoint["token2idx"]
    tag2idx: Dict[str, int] = checkpoint["tag2idx"]
    idx2tag_raw = checkpoint["idx2tag"]
    idx2tag: Dict[int, str] = {int(k): v for k, v in idx2tag_raw.items()}
    config = checkpoint.get("config", {})

    pad_idx = token2idx.get("<PAD>", 0)
    model = SimpleRNNCRFTagger(
        vocab_size=len(token2idx),
        num_tags=len(tag2idx),
        embedding_dim=int(config.get("embedding_dim", 128)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_layers=int(config.get("num_layers", 1)),
        dropout=float(config.get("dropout", 0.3)),
        pad_idx=pad_idx,
        bidirectional=bool(config.get("bidirectional", False)),
        rnn_type=str(config.get("rnn_type", "RNN")),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, token2idx, idx2tag, config


def encode_tokens(tokens: List[str], token2idx: Dict[str, int]) -> torch.Tensor:
    unk_idx = token2idx.get("<UNK>", 1)
    ids = []
    for token in tokens:
        if token in token2idx:
            ids.append(token2idx[token])
        elif token.lower() in token2idx:
            ids.append(token2idx[token.lower()])
        else:
            ids.append(unk_idx)
    return torch.tensor([ids], dtype=torch.long)


@torch.no_grad()
def predict_tags(model, token2idx: Dict[str, int], idx2tag: Dict[int, str], tokens: List[str]) -> List[str]:
    if not tokens:
        return []
    input_ids = encode_tokens(tokens, token2idx)
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    emissions = model(input_ids)
    pred_ids = model.crf.decode(emissions, mask)[0]
    return [idx2tag.get(int(tag_id), "O") for tag_id in pred_ids]


def parse_bilou_tag(tag: str) -> Tuple[str, Optional[str]]:
    if tag in {"O", "<PAD>", ""} or "-" not in tag:
        return "O", None
    prefix, ent_type = tag.split("-", 1)
    return prefix, ent_type


def extract_spans(
    tokens: List[str],
    offsets: List[Tuple[int, int]],
    tags: List[str],
    original_text: str,
) -> List[Dict[str, object]]:
    spans: List[Dict[str, object]] = []
    i = 0
    n = len(tags)

    while i < n:
        prefix, ent_type = parse_bilou_tag(tags[i])
        if prefix == "O" or ent_type is None:
            i += 1
            continue

        if prefix == "U":
            start_i = end_i = i
            i += 1
        elif prefix == "B":
            start_i = i
            end_i = i
            i += 1
            while i < n:
                p, t = parse_bilou_tag(tags[i])
                if t != ent_type:
                    break
                end_i = i
                i += 1
                if p == "L":
                    break
        else:
            # Fallback for rare invalid predictions: treat I/L as a short span.
            start_i = end_i = i
            i += 1

        start_char = offsets[start_i][0]
        end_char = offsets[end_i][1]
        spans.append(
            {
                "type": ent_type,
                "start_token": start_i,
                "end_token": end_i,
                "start_char": start_char,
                "end_char": end_char,
                "text": original_text[start_char:end_char],
            }
        )
    return spans


# =============================
# Statement-speaker association
# =============================

SPEAKER_TYPES = {"PERSON", "PERSONCOREF"}
CUE_TYPES = {"CUE", "CUECOREF"}


def char_distance(a: Dict[str, object], b: Dict[str, object]) -> int:
    a_start, a_end = int(a["start_char"]), int(a["end_char"])
    b_start, b_end = int(b["start_char"]), int(b["end_char"])
    if a_end < b_start:
        return b_start - a_end
    if b_end < a_start:
        return a_start - b_end
    return 0


def nearest_span(
    anchor: Dict[str, object],
    candidates: List[Dict[str, object]],
    max_distance: int = 400,
    prefer_after: bool = False,
) -> Optional[Dict[str, object]]:
    if not candidates:
        return None

    def sort_key(candidate):
        distance = char_distance(anchor, candidate)
        after_bonus = 0
        if prefer_after and int(candidate["start_char"]) >= int(anchor["end_char"]):
            after_bonus = -40
        return distance + after_bonus

    ranked = sorted(candidates, key=sort_key)
    if char_distance(anchor, ranked[0]) <= max_distance:
        return ranked[0]
    return None


def extract_statement_records(spans: List[Dict[str, object]]) -> pd.DataFrame:
    statements = [s for s in spans if s["type"] == "STATEMENT"]
    speakers = [s for s in spans if s["type"] in SPEAKER_TYPES]
    cues = [s for s in spans if s["type"] in CUE_TYPES]
    roles = [s for s in spans if s["type"] == "ROLE"]
    affiliations = [s for s in spans if s["type"] == "AFFILIATION"]

    records = []
    for st_span in statements:
        speaker = nearest_span(st_span, speakers, max_distance=500, prefer_after=True)
        cue = nearest_span(st_span, cues, max_distance=180, prefer_after=True)

        role_anchor = speaker if speaker is not None else st_span
        affiliation_anchor = speaker if speaker is not None else st_span
        role = nearest_span(role_anchor, roles, max_distance=180)
        affiliation = nearest_span(affiliation_anchor, affiliations, max_distance=220)

        records.append(
            {
                "Pernyataan": st_span["text"],
                "Tokoh": speaker["text"] if speaker else "-",
                "Jenis Tokoh": speaker["type"] if speaker else "-",
                "Jabatan/Peran": role["text"] if role else "-",
                "Afiliasi": affiliation["text"] if affiliation else "-",
                "Cue": cue["text"] if cue else "-",
                "Karakter Mulai": st_span["start_char"],
                "Karakter Selesai": st_span["end_char"],
            }
        )

    return pd.DataFrame(records)


# =============================
# Visualization
# =============================

ENTITY_COLORS = {
    "STATEMENT": "#f6a609",
    "CUE": "#ff6b57",
    "CUECOREF": "#22c7f2",
    "PERSON": "#fff000",
    "PERSONCOREF": "#9b51e0",
    "ROLE": "#ff00d4",
    "AFFILIATION": "#21e881",
    "LOCATION": "#95ff2d",
    "DATETIME": "#f6e5d3",
    "EVENT": "#d8c2a3",
    "ISSUE": "#7fffd4",
}


def render_annotated_html(text: str, spans: List[Dict[str, object]]) -> str:
    spans = sorted(spans, key=lambda x: (int(x["start_char"]), int(x["end_char"])))
    parts = []
    cursor = 0

    for span in spans:
        start = int(span["start_char"])
        end = int(span["end_char"])
        ent_type = str(span["type"])
        if start < cursor:
            continue

        parts.append(html.escape(text[cursor:start]))
        color = ENTITY_COLORS.get(ent_type, "#dddddd")
        fg = "#000000"
        if ent_type in {"PERSONCOREF"}:
            fg = "#000000"
        marked_text = html.escape(text[start:end])
        label = html.escape(ent_type)
        parts.append(
            f"<mark class='entity' style='background:{color}; color:{fg};'>"
            f"{marked_text}<span class='entity-label'>{label}</span></mark>"
        )
        cursor = end

    parts.append(html.escape(text[cursor:]))
    return "".join(parts)


def spans_to_dataframe(spans: List[Dict[str, object]]) -> pd.DataFrame:
    if not spans:
        return pd.DataFrame(columns=["Text", "Type", "Start Token", "End Token", "Start Char", "End Char"])
    return pd.DataFrame(
        [
            {
                "Text": s["text"],
                "Type": s["type"],
                "Start Token": s["start_token"],
                "End Token": s["end_token"],
                "Start Char": s["start_char"],
                "End Char": s["end_char"],
            }
            for s in spans
        ]
    )


# =============================
# Streamlit interface
# =============================

st.set_page_config(
    page_title="Ekstraksi Pernyataan Tokoh Publik",
    page_icon="📰",
    layout="wide",
)

st.markdown(
    """
    <style>
    .annotated-box {
        border: 1px solid #e6e6e6;
        border-radius: 12px;
        padding: 18px;
        background: #ffffff;
        color: #111111;
        line-height: 2.25;
        font-size: 16px;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }
    mark.entity {
        border-radius: 5px;
        padding: 3px 5px 4px 5px;
        margin: 0 2px;
        box-decoration-break: clone;
        -webkit-box-decoration-break: clone;
    }
    .entity-label {
        font-size: 10px;
        font-weight: 800;
        margin-left: 7px;
        letter-spacing: 0.3px;
    }
    .legend-chip {
        display: inline-block;
        border-radius: 5px;
        padding: 4px 8px;
        margin: 3px 4px 3px 0;
        font-size: 12px;
        font-weight: 700;
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Ekstraksi Pernyataan Tokoh Publik dari Artikel Berita Bahasa Indonesia")
st.caption("Model: SimpleRNN-CRF dengan skema BILOU. Visualisasi dibuat menyerupai gaya displaCy pada contoh gambar.")

with st.sidebar:
    st.header("Pengaturan")
    default_model = Path(__file__).with_name("simplernn_crf_model_state.pt")
    uploaded_model = st.file_uploader("Upload model .pt", type=["pt"])
    model_bytes = uploaded_model.getvalue() if uploaded_model is not None else None

    if uploaded_model is None:
        st.info(f"Menggunakan model default: `{default_model.name}`")
    else:
        st.success(f"Model diupload: `{uploaded_model.name}`")

    show_token_table = st.checkbox("Tampilkan tabel token dan label", value=False)
    show_all_spans = st.checkbox("Tampilkan semua span entitas", value=True)

if uploaded_model is None and not default_model.exists():
    st.error(
        "File model default tidak ditemukan. Letakkan `simplernn_crf_model_state.pt` "
        "di folder yang sama dengan `app.py` atau upload file model melalui sidebar."
    )
    st.stop()

model, token2idx, idx2tag, config = load_checkpoint_model(model_bytes, str(default_model))

with st.expander("Informasi model", expanded=False):
    col1, col2, col3 = st.columns(3)
    col1.metric("Algoritma", config.get("algorithm", "SimpleRNN-CRF"))
    col2.metric("Entity F1 Dev", f"{float(config.get('best_dev_entity_f1', 0)):.4f}")
    col3.metric("Jumlah Tag", len(idx2tag))
    st.json(config)

legend_html = "".join(
    f"<span class='legend-chip' style='background:{color}'>{name}</span>"
    for name, color in ENTITY_COLORS.items()
)
st.markdown(legend_html, unsafe_allow_html=True)

sample_text = """JAKARTA, KOMPAS — Pemerintah serius mewujudkan transformasi energi menuju energi baru dan terbarukan, termasuk penggunaan kendaraan listrik. Pemerintah menargetkan pada 2025 sebanyak 2 juta kendaraan listrik dapat digunakan oleh masyarakat Indonesia.

\"Pemerintah sangat serius untuk masuk pada energi baru terbarukan, termasuk di dalamnya adalah menuju pada kendaraan listrik,\" ujar Presiden Joko Widodo saat memberikan sambutan di Jakarta, Selasa (22/2/2022).

\"Dan, selanjutnya kita akan menuju ke pasar-pasar ekspor,\" tuturnya."""

article_text = st.text_area(
    "Masukkan artikel berita",
    value=sample_text,
    height=260,
    help="Tempel artikel berita bahasa Indonesia. Sistem akan memprediksi tag BILOU, mengekstrak span pernyataan, lalu mengasosiasikannya dengan tokoh terdekat.",
)

run_button = st.button("Ekstrak Pernyataan", type="primary", use_container_width=True)

if run_button:
    text = article_text.strip()
    if not text:
        st.warning("Artikel masih kosong.")
        st.stop()

    tokens, offsets = tokenize_with_offsets(text)
    if not tokens:
        st.warning("Tidak ada token yang dapat diproses.")
        st.stop()

    with st.spinner("Melakukan inferensi model..."):
        tags = predict_tags(model, token2idx, idx2tag, tokens)
        spans = extract_spans(tokens, offsets, tags, text)
        statement_df = extract_statement_records(spans)
        spans_df = spans_to_dataframe(spans)

    st.subheader("Ringkasan")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Token", len(tokens))
    c2.metric("Span Entitas", len(spans))
    c3.metric("Pernyataan", int((spans_df["Type"] == "STATEMENT").sum()) if not spans_df.empty else 0)
    c4.metric("Tokoh/Referensi Tokoh", int(spans_df["Type"].isin(["PERSON", "PERSONCOREF"]).sum()) if not spans_df.empty else 0)

    st.subheader("Visualisasi Artikel")
    annotated_html = render_annotated_html(text, spans)
    st.markdown(f"<div class='annotated-box'>{annotated_html}</div>", unsafe_allow_html=True)

    st.subheader("Hasil Ekstraksi Pernyataan")
    if statement_df.empty:
        st.info("Belum ada span STATEMENT yang terdeteksi pada artikel ini.")
    else:
        st.dataframe(statement_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download hasil pernyataan sebagai CSV",
            data=statement_df.to_csv(index=False).encode("utf-8"),
            file_name="hasil_ekstraksi_pernyataan.csv",
            mime="text/csv",
        )

    if show_all_spans:
        st.subheader("Semua Span Entitas")
        st.dataframe(spans_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download semua span sebagai CSV",
            data=spans_df.to_csv(index=False).encode("utf-8"),
            file_name="semua_span_entitas.csv",
            mime="text/csv",
        )

    if show_token_table:
        st.subheader("Token dan Label Prediksi")
        token_df = pd.DataFrame({"Token": tokens, "Predicted Tag": tags})
        st.dataframe(token_df, use_container_width=True, hide_index=True)
