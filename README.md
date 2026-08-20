# Persiapan Interview — OrgPulse

## 1. "Ceritakan tentang project ini"
Masalah: keputusan retensi HR berdasarkan intuisi, bukan data.
Pendekatan: gabungkan 5 sumber data → ONA + NLP + ML → insight actionable.
Hasil: CV AUC 0.875, ROI Rp 16.3M, scenario simulator untuk C-level.

## 2. "Kenapa pakai ONA?"
Karyawan yang influential secara jaringan tapi flight risk tinggi adalah
titik risiko paling kritis — tidak terdeteksi oleh model konvensional.
Knowledge Risk Matrix = influence × flight risk.

## 3. "Bagian paling menantang?"
Data leakage: avg_hours_worked terlalu berkorelasi dengan label.
Diagnosis: feature importance 0.9999 untuk satu fitur.
Fix: drop fitur yang merupakan konsekuensi langsung label, bukan penyebab.

## 4. "Bagaimana kamu tahu model bekerja?"
- CV ROC-AUC 5-fold stratified
- Fairness audit DIR > 0.80
- SHAP waterfall per individu untuk validasi logika
- Sanity check: top features masuk akal secara domain (sentiment, absensi)

## 5. "Keterbatasan?"
- Data simulasi, bukan produksi
- 1000 karyawan vs enterprise skala ribuan
- Survival analysis butuh data longitudinal resign aktual

## 6. "Scale ke produksi?"
- Spark untuk 262K+ baris attendance
- Feature store untuk real-time scoring
- Retraining otomatis setiap 90 hari atau AUC < 0.75
- Human-in-the-loop wajib sebelum intervensi HR