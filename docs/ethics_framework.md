# Ethics Framework — OrgPulse People Analytics

## 1. TRANSPARENCY (Transparansi)
Karyawan harus tahu jenis data apa yang dikumpulkan dan untuk tujuan apa.
- Metadata komunikasi (bukan isi pesan) digunakan untuk analisis jaringan
- Kebijakan tertulis tersedia untuk seluruh karyawan
- Laporan agregat dibagikan ke seluruh organisasi secara berkala

## 2. PURPOSE LIMITATION (Batasan Tujuan)
Data hanya digunakan untuk tujuan yang dinyatakan saat pengumpulan.
- ✅ DIIZINKAN: program retensi, pengembangan karier, succession planning
- ❌ DILARANG: skor flight risk TIDAK BOLEH digunakan sebagai alasan pemecatan
- ❌ DILARANG: sharing data individual tanpa persetujuan karyawan

## 3. FAIRNESS & NON-DISCRIMINATION
- Model tidak menggunakan variabel proxy diskriminasi (etnis, agama, status perkawinan)
- Audit DIR (Disparate Impact Ratio) dijalankan setiap 6 bulan
- Threshold: DIR < 0.80 memerlukan investigasi dan perbaikan model

## 4. HUMAN-IN-THE-LOOP
- Skor model adalah INPUT untuk HRBP, BUKAN keputusan otomatis
- HRBP wajib melakukan conversation sebelum intervensi apapun
- Karyawan berhak menyanggah skor yang dianggap tidak akurat
- Semua keputusan final tetap di tangan manusia

## 5. DATA MINIMIZATION
- Kumpulkan hanya data yang benar-benar dibutuhkan
- Retensi data komunikasi: maksimum 2 tahun
- Data individual dihapus 90 hari setelah karyawan keluar

## 6. KETERBATASAN MODEL
- Model dilatih pada data simulasi, bukan data produksi nyata
- CV ROC-AUC 0.875 — ada ~12.5% ketidakpastian
- Model tidak mempertimbangkan faktor eksternal (kondisi pasar kerja, keluarga)
- Skor tinggi ≠ pasti resign; skor rendah ≠ pasti bertahan

## Referensi
- GDPR Article 22 (automated decision-making)
- EEOC Uniform Guidelines on Employee Selection Procedures
- IEEE Ethically Aligned Design v2