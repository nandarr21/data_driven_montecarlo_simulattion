"""
Simulasi Monte Carlo untuk Mengestimasi Risiko Kerugian Bersih Kredit
Berdasarkan Tingkat Pemulihan Dana  (TA-03 Data-Driven Monte Carlo Simulation)

Dataset : Lending Club Loan Data (Kaggle, ethon0426/lending-club-20072020q1)
Jalankan: streamlit run app.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------
KANDIDAT_KOLOM = {
    "loan_amnt": ["loan_amnt", "funded_amnt", "loan_amount", "funded_amount"],
    "recoveries": ["recoveries", "recovery", "total_recoveries"],
    "loan_status": ["loan_status", "status"],
}
LABEL_KOLOM = {
    "loan_amnt": "Kolom jumlah pinjaman",
    "recoveries": "Kolom dana pemulihan",
    "loan_status": "Kolom status pinjaman",
}
UKURAN_CHUNK = 500_000
MIN_ITERASI = 5_000
UKURAN_BATCH = 20_000
SEED_SAMPLING = 42
DATASET_PATH = Path(__file__).resolve().parent / "data" / "lending_club_sample.csv.gz"
OPSI_KECUALIKAN = "Kecualikan pinjaman gagal bayar yang recoveries-nya kosong (disarankan)"
OPSI_IMPUTASI = "Imputasi recoveries kosong dengan median rasio pemulihan (recoveries / loan_amnt)"


# --------------------------------------------------------------------------
# Fungsi bantu format (gaya angka Indonesia)
# --------------------------------------------------------------------------
def fmt_usd(x: float) -> str:
    return "US$ " + f"{x:,.0f}".replace(",", ".")


def fmt_int(x: int) -> str:
    return f"{int(x):,}".replace(",", ".")


def fmt_pct(p: float, desimal: int = 2) -> str:
    return f"{p * 100:.{desimal}f}".replace(".", ",") + "%"


# --------------------------------------------------------------------------
# Pembacaan & pembersihan data
# --------------------------------------------------------------------------
_KOMPRESI = {"gzip": "gzip", "zip": "zip"}


def deteksi_format(file) -> str:
    """Deteksi format dari isi file (bukan ekstensi): csv, gzip, zip, parquet, atau xlsx."""
    file.seek(0)
    kepala = file.read(4)
    file.seek(0)
    if kepala[:2] == b"\x1f\x8b":
        return "gzip"
    if kepala == b"PAR1":
        return "parquet"
    if kepala[:2] == b"PK":
        try:
            with zipfile.ZipFile(file) as z:
                nama = z.namelist()
        except zipfile.BadZipFile:
            nama = []
        file.seek(0)
        return "xlsx" if any(n.startswith("xl/") for n in nama) else "zip"
    return "csv"


def periksa_bukan_kamus(kolom) -> None:
    k = {str(c).strip().lower() for c in kolom}
    if {"loanstatnew", "description"} <= k:
        raise ValueError(
            "File ini adalah KAMUS DATA (LCDataDictionary), bukan data pinjaman. Gunakan file data pinjaman "
            "yang berisi kolom loan_amnt, recoveries, dan loan_status."
        )


def baca_pratinjau(file, sep: str, n: int = 100):
    """Baca n baris pertama dari CSV / CSV terkompresi (.gz/.gzip/.zip) / Parquet / Excel."""
    fmt = deteksi_format(file)
    if fmt == "parquet":
        import pyarrow.parquet as pq
        df = next(pq.ParquetFile(file).iter_batches(batch_size=n)).to_pandas()
    elif fmt == "xlsx":
        df = pd.read_excel(file, nrows=n)
    else:
        df = pd.read_csv(file, sep=sep, nrows=n, low_memory=False,
                         compression=_KOMPRESI.get(fmt), encoding_errors="replace")
    file.seek(0)
    periksa_bukan_kamus(df.columns)
    return df, fmt


def iter_potongan(file, sep: str, kolom: list[str]):
    """Hasilkan potongan DataFrame yang hanya berisi kolom yang diperlukan."""
    fmt = deteksi_format(file)
    if fmt == "parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(file).iter_batches(batch_size=UKURAN_CHUNK, columns=kolom):
            yield batch.to_pandas()
    elif fmt == "xlsx":
        yield pd.read_excel(file, usecols=kolom)
    else:
        yield from pd.read_csv(file, sep=sep, usecols=kolom, chunksize=UKURAN_CHUNK, low_memory=False,
                               compression=_KOMPRESI.get(fmt), encoding_errors="replace")


def tebak_kolom(daftar_kolom: list[str], kandidat: list[str]) -> str | None:
    peta = {str(k).strip().lower(): k for k in daftar_kolom}
    for c in kandidat:
        if c in peta:
            return peta[c]
    return None


def baca_data(file, sep: str, peta: dict, persen: int, seed: int):
    """Baca hanya 3 kolom yang diperlukan, per potongan (hemat memori).

    Nilai non-numerik pada kolom angka diubah menjadi NaN (BUKAN 0).
    """
    file.seek(0)
    potongan, total, gagal_jumlah, gagal_rec = [], 0, 0, 0
    pembaca = iter_potongan(file, sep, list(peta.values()))
    for i, chunk in enumerate(pembaca):
        total += len(chunk)
        if persen < 100:
            chunk = chunk.sample(frac=persen / 100, random_state=seed + i)
        chunk = chunk.rename(columns={v: k for k, v in peta.items()})
        for kol in ("loan_amnt", "recoveries"):
            sebelum = int(chunk[kol].notna().sum())
            chunk[kol] = pd.to_numeric(chunk[kol], errors="coerce")
            hilang = sebelum - int(chunk[kol].notna().sum())
            if kol == "loan_amnt":
                gagal_jumlah += hilang
            else:
                gagal_rec += hilang
        chunk["loan_status"] = chunk["loan_status"].astype("string").str.strip()
        potongan.append(chunk[["loan_amnt", "recoveries", "loan_status"]])
    if not potongan:
        raise ValueError("File CSV tidak berisi baris data.")
    df = pd.concat(potongan, ignore_index=True)
    df["loan_status"] = df["loan_status"].astype("category")
    info = {
        "baris_file": total,
        "baris_dimuat": len(df),
        "gagal_konversi_jumlah": gagal_jumlah,
        "gagal_konversi_rec": gagal_rec,
    }
    return df, info


def bersihkan(df: pd.DataFrame, status_lunas: list[str], status_gagal: list[str], perlakuan: str):
    """Bangun 'kumpulan pinjaman final' beserta kerugian aktualnya.

    - Lunas        : kerugian = 0 (pinjaman selesai dibayar penuh).
    - Gagal bayar  : kerugian = loan_amnt - recoveries (recoveries harus valid).
    - Lainnya (Current, Late, dst.) : dikecualikan, hasil akhirnya belum diketahui.
    """
    r = {"baris_dianalisis": len(df)}

    jumlah_valid = df["loan_amnt"].notna() & (df["loan_amnt"] > 0)
    status_ada = df["loan_status"].notna()
    r["jumlah_tidak_valid"] = int((~jumlah_valid).sum())
    r["status_kosong"] = int((jumlah_valid & ~status_ada).sum())

    d = df[jumlah_valid & status_ada]
    teks = d["loan_status"].astype(str)
    m_lunas = teks.isin(status_lunas)
    m_gagal = teks.isin(status_gagal) & ~m_lunas
    r["belum_final"] = int((~(m_lunas | m_gagal)).sum())

    lunas = d.loc[m_lunas, ["loan_amnt", "recoveries"]].copy()
    gagal = d.loc[m_gagal, ["loan_amnt", "recoveries"]].copy()

    rec = gagal["recoveries"]
    kosong = rec.isna()
    negatif = rec < 0
    berlebih = rec > gagal["loan_amnt"]
    valid = ~kosong & ~negatif & ~berlebih
    r["gagal_total"] = len(gagal)
    r["rec_kosong"] = int(kosong.sum())
    r["rec_negatif"] = int(negatif.sum())
    r["rec_berlebih"] = int(berlebih.sum())
    r["rec_imputasi"] = 0
    r["rasio_imputasi"] = np.nan

    pakai = valid
    if perlakuan == OPSI_IMPUTASI and valid.any() and kosong.any():
        rasio = float((gagal.loc[valid, "recoveries"] / gagal.loc[valid, "loan_amnt"]).median())
        gagal.loc[kosong, "recoveries"] = rasio * gagal.loc[kosong, "loan_amnt"]
        r["rec_imputasi"] = int(kosong.sum())
        r["rasio_imputasi"] = rasio
        pakai = valid | kosong
    gagal = gagal.loc[pakai].copy()

    gagal["kerugian"] = gagal["loan_amnt"] - gagal["recoveries"]
    gagal["gagal_bayar"] = True
    lunas["kerugian"] = 0.0          # recoveries pada pinjaman lunas tetap NaN (tidak diisi 0)
    lunas["gagal_bayar"] = False
    kumpulan = pd.concat([lunas, gagal], ignore_index=True)

    r["n_lunas"] = len(lunas)
    r["n_gagal"] = len(gagal)
    r["n_kumpulan"] = len(kumpulan)
    r["tingkat_gagal"] = len(gagal) / len(kumpulan) if len(kumpulan) else float("nan")
    return kumpulan, r


# --------------------------------------------------------------------------
# Simulasi Monte Carlo
# --------------------------------------------------------------------------
def simulasi(kerugian: np.ndarray, n_iter: int, n_pinjaman: int, seed: int) -> np.ndarray:
    """Bootstrap: tiap iterasi menarik n_pinjaman pinjaman (dengan pengembalian)
    dari kumpulan pinjaman final historis, lalu menjumlahkan kerugiannya.
    Hasil identik untuk seed, parameter, dan data yang sama."""
    rng = np.random.default_rng(seed)
    hasil = np.empty(n_iter, dtype=float)
    for mulai in range(0, n_iter, UKURAN_BATCH):
        m = min(UKURAN_BATCH, n_iter - mulai)
        idx = rng.integers(0, len(kerugian), size=(m, n_pinjaman))
        hasil[mulai:mulai + m] = kerugian[idx].sum(axis=1)
    return hasil


def ringkas_hasil(hasil: np.ndarray, anggaran: float) -> dict:
    p = float((hasil > anggaran).mean())
    return {
        "rata_rata": float(hasil.mean()),
        "median": float(np.median(hasil)),
        "p95": float(np.percentile(hasil, 95)),
        "std": float(hasil.std(ddof=1)) if len(hasil) > 1 else 0.0,
        "maks": float(hasil.max()),
        "prob": p,
        "se_prob": float(np.sqrt(p * (1 - p) / len(hasil))),
    }


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def histogram(data, judul, xlabel, warna="#4C78A8", bins=40):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(data, bins=bins, color=warna, edgecolor="white")
    ax.set_title(judul, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frekuensi")
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Tab 1: unggah & validasi
# --------------------------------------------------------------------------
def muat_dataset_bawaan():
    """Muat dataset Lending Club bawaan dari folder data/."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset bawaan tidak ditemukan: {DATASET_PATH}. "
            "Pastikan folder data/lending_club_sample.csv.gz ikut disertakan."
        )

    df = pd.read_csv(
        DATASET_PATH,
        compression="gzip",
        usecols=["loan_amnt", "recoveries", "loan_status"],
        low_memory=False,
    )

    gagal_jumlah = int(pd.to_numeric(df["loan_amnt"], errors="coerce").isna().sum())
    gagal_rec = int(pd.to_numeric(df["recoveries"], errors="coerce").isna().sum())
    df["loan_amnt"] = pd.to_numeric(df["loan_amnt"], errors="coerce")
    df["recoveries"] = pd.to_numeric(df["recoveries"], errors="coerce")
    df["loan_status"] = df["loan_status"].astype("string").str.strip().astype("category")

    info = {
        "baris_file": len(df),
        "baris_dimuat": len(df),
        "gagal_konversi_jumlah": gagal_jumlah,
        "gagal_konversi_rec": gagal_rec,
        "sumber": DATASET_PATH.name,
    }
    return df, info


@st.cache_data(show_spinner="Memuat dataset Lending Club...")
def baca_dataset_bawaan():
    """Cache pembacaan dataset agar tidak dibaca ulang pada setiap interaksi."""
    return muat_dataset_bawaan()


# --------------------------------------------------------------------------
# Tab 1: dataset bawaan
# --------------------------------------------------------------------------
def tab_data():
    st.subheader("1. Dataset Lending Club (dimuat otomatis)")
    st.caption(
        "Aplikasi menggunakan dataset bawaan. Pengguna tidak perlu mengunggah file "
        "setiap kali membuka website."
    )

    try:
        if "raw" not in st.session_state:
            df, info = baca_dataset_bawaan()
            st.session_state["raw"] = df
            st.session_state["raw_info"] = info
            st.session_state["raw_id"] = st.session_state.get("raw_id", 0) + 1
            for k in ("pool", "pool_key", "pool_r", "hasil"):
                st.session_state.pop(k, None)
        else:
            df = st.session_state["raw"]
            info = st.session_state["raw_info"]
    except Exception as e:  # noqa: BLE001
        st.error(f"Gagal memuat dataset bawaan: {e}")
        st.info(
            "Pastikan file `lending_club_sample.csv.gz` berada di dalam folder `data` "
            "yang sejajar dengan `app.py`."
        )
        return

    st.success(
        f"Dataset berhasil dimuat: **{fmt_int(info['baris_dimuat'])} baris**. "
        f"File: `{info['sumber']}`."
    )
    st.dataframe(df.head(100), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah baris", fmt_int(len(df)))
    c2.metric("Kolom digunakan", "3")
    c3.metric("Status dataset", "Siap digunakan")
    st.caption(
        "Kolom yang digunakan: `loan_amnt`, `recoveries`, dan `loan_status`. "
        "Lanjutkan ke tab **Pembersihan & Statistik** untuk memeriksa status pinjaman."
    )


# --------------------------------------------------------------------------
# Tab 2: pembersihan + statistik
# --------------------------------------------------------------------------
def tab_pembersihan():
    raw = st.session_state.get("raw")
    if raw is None:
        st.info("Muat data terlebih dahulu di tab **Data**.")
        return
    raw_id = st.session_state["raw_id"]

    st.subheader("2. Pembersihan data")
    st.markdown(
        "Tentukan status pinjaman mana yang dianggap **lunas** dan **gagal bayar**. "
        "Status lain (mis. *Current*, *Late*) **dikecualikan** karena hasil akhirnya belum diketahui."
    )
    hitung = raw["loan_status"].value_counts()
    hitung = hitung[hitung > 0]
    st.dataframe(hitung.rename_axis("loan_status").reset_index(name="jumlah"))
    opsi = [str(s) for s in hitung.index]
    default_lunas = [s for s in opsi if "fully paid" in s.lower()]
    default_gagal = [s for s in opsi if "charged off" in s.lower() or "default" in s.lower()]

    c1, c2 = st.columns(2)
    status_lunas = c1.multiselect("Status dianggap LUNAS", opsi, default=default_lunas, key=f"ms_lunas_{raw_id}")
    status_gagal = c2.multiselect("Status dianggap GAGAL BAYAR", opsi, default=default_gagal, key=f"ms_gagal_{raw_id}")
    perlakuan = st.radio("Perlakuan recoveries kosong pada pinjaman gagal bayar", [OPSI_KECUALIKAN, OPSI_IMPUTASI])

    if set(status_lunas) & set(status_gagal):
        st.error("Satu status tidak boleh berada di kedua kelompok.")
        return
    if not status_lunas and not status_gagal:
        st.warning("Pilih minimal satu status lunas/gagal bayar.")
        return

    kunci = (raw_id, tuple(sorted(status_lunas)), tuple(sorted(status_gagal)), perlakuan)
    if st.session_state.get("pool_key") != kunci:
        with st.spinner("Membersihkan data..."):
            kumpulan, r = bersihkan(raw, status_lunas, status_gagal, perlakuan)
        st.session_state.update(pool=kumpulan, pool_r=r, pool_key=kunci)
    kumpulan, r = st.session_state["pool"], st.session_state["pool_r"]

    if kumpulan.empty:
        st.error("Tidak ada pinjaman final yang valid. Periksa pemilihan status dan kolom.")
        return

    baris = [
        ("Baris dianalisis", r["baris_dianalisis"]),
        ("Dibuang: loan_amnt kosong / non-numerik / ≤ 0", r["jumlah_tidak_valid"]),
        ("Dibuang: status pinjaman kosong", r["status_kosong"]),
        ("Dikecualikan: status belum final (bukan lunas/gagal bayar)", r["belum_final"]),
        ("Pinjaman gagal bayar (sebelum validasi recoveries)", r["gagal_total"]),
        ("   recoveries kosong " + ("(diimputasi)" if r["rec_imputasi"] else "(dikecualikan, BUKAN dianggap 0)"), r["rec_kosong"]),
        ("   dikecualikan: recoveries negatif", r["rec_negatif"]),
        ("   dikecualikan: recoveries > loan_amnt", r["rec_berlebih"]),
        ("Kumpulan final — lunas", r["n_lunas"]),
        ("Kumpulan final — gagal bayar valid", r["n_gagal"]),
        ("Kumpulan final — total", r["n_kumpulan"]),
    ]
    st.dataframe(pd.DataFrame(baris, columns=["Tahap", "Jumlah baris"]).astype({"Jumlah baris": int}))
    if r["rec_imputasi"]:
        st.caption(f"Median rasio pemulihan yang dipakai untuk imputasi: {fmt_pct(r['rasio_imputasi'])}.")

    gagal = kumpulan[kumpulan["gagal_bayar"]]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pinjaman final", fmt_int(r["n_kumpulan"]))
    m2.metric("Tingkat gagal bayar", fmt_pct(r["tingkat_gagal"]))
    m3.metric("Rata-rata kerugian / pinjaman gagal", fmt_usd(gagal["kerugian"].mean()) if len(gagal) else "-")
    m4.metric("Rata-rata kerugian / pinjaman (semua)", fmt_usd(kumpulan["kerugian"].mean()))

    st.subheader("3. Statistik deskriptif data asli")
    tabel = {
        "loan_amnt (semua pinjaman final)": kumpulan["loan_amnt"],
        "recoveries (pinjaman gagal bayar)": gagal["recoveries"],
        "kerugian bersih (pinjaman gagal bayar)": gagal["kerugian"],
        "kerugian bersih (semua pinjaman final)": kumpulan["kerugian"],
    }
    stat = pd.DataFrame({k: v.describe(percentiles=[0.25, 0.5, 0.75, 0.95]) for k, v in tabel.items()}).T
    stat = stat.rename(columns={"count": "n", "mean": "rata-rata", "std": "simpangan baku", "min": "min",
                                "25%": "P25", "50%": "median", "75%": "P75", "95%": "P95", "max": "maks"})
    st.dataframe(stat.round(2))

    st.markdown("**Histogram data asli**")
    h1, h2, h3 = st.columns(3)
    with h1:
        fig = histogram(kumpulan["loan_amnt"], "Jumlah pinjaman", "loan_amnt (US$)")
        st.pyplot(fig)
        plt.close(fig)
    if len(gagal):
        with h2:
            fig = histogram(gagal["recoveries"], "Dana pemulihan (gagal bayar)", "recoveries (US$)", "#59A14F")
            st.pyplot(fig)
            plt.close(fig)
        with h3:
            fig = histogram(gagal["kerugian"], "Kerugian bersih (gagal bayar)", "loan_amnt − recoveries (US$)", "#E15759")
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.warning("Tidak ada pinjaman gagal bayar valid, histogram recoveries dan kerugian tidak dapat dibuat.")


# --------------------------------------------------------------------------
# Tab 3: simulasi
# --------------------------------------------------------------------------
def tab_simulasi():
    kumpulan = st.session_state.get("pool")
    if kumpulan is None or kumpulan.empty:
        st.info("Muat dan bersihkan data terlebih dahulu (tab **Data** dan **Pembersihan & Statistik**).")
        return

    st.subheader("4. Simulasi Monte Carlo")
    c1, c2, c3, c4 = st.columns(4)
    n_iter = c1.number_input("Jumlah iterasi", MIN_ITERASI, 200_000, 10_000, step=1_000,
                             help=f"Minimal {fmt_int(MIN_ITERASI)} iterasi.")
    n_pinjaman = c2.number_input("Pinjaman per iterasi", 10, 1_000, 100, step=10)
    seed = c3.number_input("Seed acak (reprodusibilitas)", 0, 2**31 - 1, 42, step=1)
    anggaran = c4.number_input("Anggaran kerugian (US$)", 0.0, 1e12, 250_000.0, step=10_000.0, format="%.0f")

    harapan = float(kumpulan["kerugian"].mean()) * int(n_pinjaman)
    st.caption(f"Acuan: nilai harapan analitis kerugian {int(n_pinjaman)} pinjaman = {fmt_usd(harapan)} "
               f"(rata-rata kerugian per pinjaman final × jumlah pinjaman).")

    if st.button("Jalankan simulasi", type="primary"):
        with st.spinner("Menjalankan simulasi..."):
            hasil = simulasi(kumpulan["kerugian"].to_numpy(dtype=float), int(n_iter), int(n_pinjaman), int(seed))
        st.session_state["hasil"] = {
            "nilai": hasil, "n_iter": int(n_iter), "n_pinjaman": int(n_pinjaman),
            "seed": int(seed), "pool_key": st.session_state["pool_key"],
        }

    h = st.session_state.get("hasil")
    if h is None:
        st.info("Atur parameter lalu klik **Jalankan simulasi**.")
        return
    if h["pool_key"] != st.session_state["pool_key"]:
        st.warning("Data/pengaturan pembersihan berubah sejak simulasi ini dijalankan. Jalankan ulang simulasi.")

    hasil = h["nilai"]
    s = ringkas_hasil(hasil, anggaran)
    st.markdown(f"**Hasil** — {fmt_int(h['n_iter'])} iterasi × {h['n_pinjaman']} pinjaman, seed {h['seed']}")
    a, b, c, d = st.columns(4)
    a.metric("Rata-rata kerugian", fmt_usd(s["rata_rata"]))
    b.metric("Median", fmt_usd(s["median"]))
    c.metric("Persentil ke-95", fmt_usd(s["p95"]))
    d.metric("P(kerugian > anggaran)", fmt_pct(s["prob"]))
    e, f, g = st.columns(3)
    e.metric("Simpangan baku", fmt_usd(s["std"]))
    f.metric("Kerugian maksimum", fmt_usd(s["maks"]))
    g.metric("Galat baku probabilitas", "± " + fmt_pct(s["se_prob"]))

    st.info(
        f"Dengan anggaran {fmt_usd(anggaran)}, estimasi probabilitas total kerugian melampaui anggaran adalah "
        f"**{fmt_pct(s['prob'])}**. Dalam 95% skenario, kerugian tidak melebihi {fmt_usd(s['p95'])}."
    )

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(hasil, bins=60, color="#4C78A8", edgecolor="white")
    ax.axvline(anggaran, color="#D62728", linestyle="--", linewidth=2, label=f"Anggaran: {fmt_usd(anggaran)}")
    ax.axvline(s["rata_rata"], color="#2CA02C", linestyle="-", linewidth=1.5, label=f"Rata-rata: {fmt_usd(s['rata_rata'])}")
    ax.axvline(s["p95"], color="#FF7F0E", linestyle=":", linewidth=2, label=f"Persentil 95: {fmt_usd(s['p95'])}")
    ax.set_title("Distribusi total kerugian bersih portofolio hasil simulasi")
    ax.set_xlabel("Total kerugian bersih per iterasi (US$)")
    ax.set_ylabel("Frekuensi")
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    keluaran = pd.DataFrame({
        "iterasi": np.arange(1, len(hasil) + 1),
        "total_kerugian_usd": np.round(hasil, 2),
        "melampaui_anggaran": hasil > anggaran,
    })
    st.download_button(
        "⬇️ Unduh hasil simulasi (CSV)",
        data=keluaran.to_csv(index=False).encode("utf-8"),
        file_name=f"hasil_simulasi_seed{h['seed']}.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------
# Tab 4: asumsi
# --------------------------------------------------------------------------
def tab_asumsi():
    st.subheader("Asumsi model dan keterbatasan")
    st.markdown(
        """
**Definisi kerugian bersih** (per pinjaman): `loan_amnt − recoveries`, dihitung hanya untuk pinjaman **gagal bayar**
(*Charged Off*/*Default*). Pinjaman **lunas** (*Fully Paid*) dianggap berkerugian **0**.

**Perlakuan data**
- Pinjaman yang belum final (*Current*, *In Grace Period*, *Late*, dst.) **dikecualikan**: kerugian akhirnya belum diketahui,
  sehingga tidak boleh dianggap 0.
- `recoveries` kosong pada pinjaman gagal bayar **tidak dianggap 0**. Default: pinjaman itu dikecualikan; opsi lain: imputasi
  dengan median rasio pemulihan dari pinjaman gagal bayar yang datanya lengkap.
- `loan_amnt` kosong/non-numerik/≤ 0, `recoveries` negatif, dan `recoveries > loan_amnt` dianggap tidak valid dan dibuang.
- `recoveries` pada pinjaman lunas tidak dipakai (kerugian sudah 0 karena dibayar penuh).

**Metode simulasi (bootstrap data historis)**: setiap iterasi menarik sejumlah pinjaman secara acak **dengan pengembalian**
dari kumpulan pinjaman final, lalu menjumlahkan kerugiannya. Dengan cara ini peluang gagal bayar dan distribusi kerugian
per pinjaman gagal bayar (termasuk tingkat pemulihan) diambil langsung dari data, tanpa asumsi distribusi parametrik.
Hasil dapat direproduksi dengan seed, data, dan parameter yang sama.

**Keterbatasan**
1. `loan_amnt − recoveries` adalah ukuran kerugian *kasar*. Pembayaran pokok sebelum pinjaman *charge-off* tidak
   dikurangkan, sehingga kerugian ekonomi sebenarnya cenderung lebih kecil. Kolom seperti `total_rec_prncp`
   dapat dipakai untuk perhitungan yang lebih akurat.
2. Pengecualian pinjaman belum final dapat menimbulkan bias (*censoring*): tingkat gagal bayar historis dapat berbeda dari
   yang akan terjadi pada pinjaman yang masih berjalan.
3. Pinjaman diasumsikan saling bebas dan identik secara statistik; korelasi (mis. resesi) dan perbedaan grade/tahun
   tidak dimodelkan.
4. Hasil bergantung pada dataset Lending Club bawaan dan pada pemilihan status; tidak ada data buatan yang digunakan aplikasi ini.
5. Satuan mengikuti dataset (US$).
"""
    )


# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Simulasi Monte Carlo Risiko Kredit", page_icon="📊", layout="wide")
    st.title("📊 Simulasi Monte Carlo Risiko Kerugian Bersih Kredit")
    st.caption("Estimasi risiko kerugian bersih kredit berdasarkan tingkat pemulihan dana — data historis Lending Club.")

    with st.sidebar:
        st.header("Alur penggunaan")
        st.markdown(
            "1. Dataset Lending Club dimuat otomatis\n"
            "2. Pilih status lunas / gagal bayar\n"
            "3. Lihat statistik & histogram data asli\n"
            "4. Jalankan simulasi, atur anggaran\n"
            "5. Unduh hasil CSV"
        )
        st.caption("Seed tetap menjamin hasil yang sama pada data dan parameter yang sama.")

    t1, t2, t3, t4 = st.tabs(["📁 Data", "🧹 Pembersihan & Statistik", "🎲 Simulasi", "📘 Asumsi"])
    with t1:
        tab_data()
    with t2:
        tab_pembersihan()
    with t3:
        tab_simulasi()
    with t4:
        tab_asumsi()


if __name__ == "__main__":
    main()
