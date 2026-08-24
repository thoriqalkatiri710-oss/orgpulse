import xlsxwriter
import pandas as pd
import numpy as np


# ── 10.1.1 Annual Workforce Report ───────────────────────────────────────────

def generate_annual_workforce_report(
    df_employees: pd.DataFrame,
    df_flight_risk: pd.DataFrame,
    df_centrality: pd.DataFrame,
    scenario_results: pd.DataFrame,
    df_training: pd.DataFrame,
    output_path: str = "excel_deliverables/OrgPulse_Annual_Report.xlsx"
):
    """
    Generate board-ready Excel report dengan 5 sheet:
    1. Executive Summary
    2. Flight Risk Register
    3. Scenario Analysis
    4. Compensation Equity
    5. L&D Effectiveness
    """
    wb = xlsxwriter.Workbook(output_path)

    # ── Format palette ────────────────────────────────────────────────────────
    title_fmt = wb.add_format({"bold": True, "font_size": 18, "font_color": "#1D3557"})
    hdr_fmt   = wb.add_format({"bold": True, "bg_color": "#1D3557", "font_color": "white",
                                "border": 1, "align": "center"})
    red_fmt   = wb.add_format({"bg_color": "#FEE2E2", "border": 1})
    green_fmt = wb.add_format({"bg_color": "#D1FAE5", "border": 1})
    cell_fmt  = wb.add_format({"border": 1})
    bold_cell = wb.add_format({"bold": True, "border": 1})
    idr_fmt   = wb.add_format({"num_format": "Rp #,##0", "border": 1})
    pct_fmt   = wb.add_format({"num_format": "0.0%", "border": 1})

    # ── Sheet 1: Executive Summary ────────────────────────────────────────────
    ws1 = wb.add_worksheet("Executive Summary")
    ws1.set_column("A:A", 40)
    ws1.set_column("B:B", 20)
    ws1.set_column("C:C", 15)

    ws1.write("A1", "OrgPulse — Laporan Tahunan Workforce Intelligence 2024", title_fmt)
    ws1.write("A2", "Disusun oleh: People Analytics Team | Bersifat Rahasia")

    high_risk_count = (df_flight_risk["flight_risk_score"] >= 0.6).sum()
    replacement_cost = high_risk_count * df_employees["monthly_salary"].mean() * 6

    kpis = [
        ("Total Karyawan Aktif",              len(df_employees),                                                    "orang"),
        ("Flight Risk HIGH (score ≥ 0.6)",    f"{(df_flight_risk['flight_risk_score'] >= 0.6).mean():.1%}",        "%"),
        ("Rata-rata Engagement Score",         f"{df_employees['engagement_score'].mean():.2f}",                    "/ 5.0"),
        ("Karyawan di Bawah Gaji Median",     f"{(df_employees['monthly_salary'] < df_employees['monthly_salary'].median()).mean():.1%}", "%"),
        ("Estimated Replacement Cost at Risk", f"Rp {replacement_cost:,.0f}",                                       ""),
    ]

    ws1.write(3, 0, "INDIKATOR UTAMA WORKFORCE", hdr_fmt)
    ws1.write(3, 1, "NILAI",                      hdr_fmt)
    ws1.write(3, 2, "SATUAN",                     hdr_fmt)

    for row_idx, (label, value, unit) in enumerate(kpis, start=4):
        ws1.write(row_idx, 0, label, bold_cell)
        ws1.write(row_idx, 1, str(value), cell_fmt)
        ws1.write(row_idx, 2, unit, cell_fmt)

    # Dept summary
    ws1.write(10, 0, "HEADCOUNT PER DEPARTEMEN", hdr_fmt)
    dept_summary = df_employees.groupby("department").agg(
        headcount=("employee_id", "count"),
        avg_engagement=("engagement_score", "mean"),
        avg_salary=("monthly_salary", "mean"),
    ).reset_index()

    ws1.write(11, 0, "Departemen",     hdr_fmt)
    ws1.write(11, 1, "Headcount",      hdr_fmt)
    ws1.write(11, 2, "Avg Engagement", hdr_fmt)
    ws1.write(11, 3, "Avg Salary",     hdr_fmt)

    for i, (_, row) in enumerate(dept_summary.iterrows(), start=12):
        ws1.write(i, 0, row["department"],              cell_fmt)
        ws1.write(i, 1, row["headcount"],               cell_fmt)
        ws1.write(i, 2, round(row["avg_engagement"], 2), cell_fmt)
        ws1.write(i, 3, int(row["avg_salary"]),         idr_fmt)

    # ── Sheet 2: Flight Risk Register ─────────────────────────────────────────
    ws2 = wb.add_worksheet("Flight Risk Register")
    ws2.set_column("A:I", 18)

    risk_df = df_employees.merge(
        df_flight_risk[["employee_id", "flight_risk_score"]], on="employee_id"
    ).merge(
        df_centrality[["employee_id", "influence_score", "network_role"]], on="employee_id"
    ).sort_values("flight_risk_score", ascending=False)

    risk_df["risk_tier"] = risk_df["flight_risk_score"].apply(
        lambda x: "CRITICAL" if x >= 0.8 else "HIGH" if x >= 0.6 else "MEDIUM" if x >= 0.4 else "LOW"
    )

    cols = ["full_name", "department", "job_level_name", "tenure_years",
            "engagement_score", "flight_risk_score", "influence_score",
            "network_role", "risk_tier"]

    for col_idx, col in enumerate(cols):
        ws2.write(0, col_idx, col.replace("_", " ").title(), hdr_fmt)

    for row_idx, (_, row) in enumerate(risk_df[cols].iterrows(), start=1):
        for col_idx, val in enumerate(row):
            fmt = red_fmt if col_idx == cols.index("flight_risk_score") and float(row["flight_risk_score"]) >= 0.6 else cell_fmt
            ws2.write(row_idx, col_idx, val, fmt)

    # ── Sheet 3: Scenario Analysis ────────────────────────────────────────────
    ws3 = wb.add_worksheet("Scenario Analysis")
    ws3.set_column("A:F", 25)
    ws3.write("A1", "Analisis Skenario: Dampak Kehilangan Karyawan", title_fmt)

    scen_cols = scenario_results.columns.tolist()
    for col_idx, col in enumerate(scen_cols):
        ws3.write(1, col_idx, col.replace("_", " ").title(), hdr_fmt)

    for row_idx, (_, row) in enumerate(scenario_results.iterrows(), start=2):
        for col_idx, val in enumerate(row):
            ws3.write(row_idx, col_idx, str(val), cell_fmt)

    # ── Sheet 4: Compensation Equity ──────────────────────────────────────────
    ws4 = wb.add_worksheet("Compensation Equity")
    ws4.set_column("A:B", 35)
    ws4.write("A1", "Analisis Kesetaraan Kompensasi 2024", title_fmt)

    salary_by_gender = df_employees.groupby("gender")["monthly_salary"].mean()
    salary_by_dept   = df_employees.groupby("department")["monthly_salary"].mean().reset_index()

    ws4.write(2, 0, "Gender", hdr_fmt)
    ws4.write(2, 1, "Avg Salary", hdr_fmt)
    for i, (gender, sal) in enumerate(salary_by_gender.items(), start=3):
        ws4.write(i, 0, gender, cell_fmt)
        ws4.write(i, 1, int(sal), idr_fmt)

    ws4.write(7, 0, "Departemen", hdr_fmt)
    ws4.write(7, 1, "Avg Salary", hdr_fmt)
    for i, (_, row) in enumerate(salary_by_dept.iterrows(), start=8):
        ws4.write(i, 0, row["department"], cell_fmt)
        ws4.write(i, 1, int(row["monthly_salary"]), idr_fmt)

    # ── Sheet 5: L&D Effectiveness ────────────────────────────────────────────
    ws5 = wb.add_worksheet("L&D Effectiveness")
    ws5.set_column("A:E", 20)
    ws5.write("A1", "Efektivitas Learning & Development", title_fmt)

    ld_summary = df_training.groupby("category").agg(
        n_completions    =("completion_rate", "count"),
        avg_completion   =("completion_rate", "mean"),
        avg_assessment   =("assessment_score", "mean"),
    ).reset_index()

    headers = ["Category", "N Completions", "Avg Completion Rate", "Avg Assessment Score"]
    for col_idx, h in enumerate(headers):
        ws5.write(2, col_idx, h, hdr_fmt)

    for row_idx, (_, row) in enumerate(ld_summary.iterrows(), start=3):
        ws5.write(row_idx, 0, row["category"],                   cell_fmt)
        ws5.write(row_idx, 1, int(row["n_completions"]),         cell_fmt)
        ws5.write(row_idx, 2, round(row["avg_completion"], 3),   pct_fmt)
        ws5.write(row_idx, 3, round(row["avg_assessment"], 1),   cell_fmt)

    wb.close()
    print(f"✅ Annual report saved: {output_path}")
    return output_path
# ── 10.2.1 ROI Calculator Sheet ───────────────────────────────────────────────

def add_roi_calculator_sheet(wb, df_flight_risk: pd.DataFrame,
                              df_employees: pd.DataFrame):
    """
    Sheet interaktif formula-driven untuk CFO.
    Sel kuning = input yang bisa diubah langsung di Excel.
    Formula Excel terhubung ke FlightRiskRegister sheet.
    """
    ws = wb.add_worksheet("ROI Retention Calculator")
    ws.set_column("A:A", 50)
    ws.set_column("B:B", 20)

    title_fmt = wb.add_format({"bold": True, "font_size": 14, "font_color": "#1D3557"})
    input_fmt = wb.add_format({"bg_color": "#FEF3C7", "border": 1})
    hdr_fmt   = wb.add_format({"bold": True, "bg_color": "#1D3557",
                                "font_color": "white", "border": 1})
    cell_fmt  = wb.add_format({"border": 1})
    bold_fmt  = wb.add_format({"bold": True, "border": 1})
    idr_fmt   = wb.add_format({"num_format": "Rp #,##0", "border": 1})
    pct_fmt   = wb.add_format({"num_format": "0.0%", "border": 1})
    green_fmt = wb.add_format({"bold": True, "font_color": "#16A34A",
                                "num_format": "Rp #,##0", "border": 1})

    ws.write("A1", "ROI Calculator: Investasi Retensi vs Biaya Penggantian", title_fmt)
    ws.write("A2", "Sel berwarna kuning dapat diubah sesuai asumsi bisnis")

    ws.write("A3", "ASUMSI BIAYA (ubah sel kuning)", hdr_fmt)
    ws.write("B3", "NILAI", hdr_fmt)

    ws.write("A4", "Biaya rekrutmen (% gaji tahunan):",              bold_fmt)
    ws.write("B4", 0.20,                                              input_fmt)
    ws.write("A5", "Biaya onboarding & produktivitas hilang (bulan gaji):", bold_fmt)
    ws.write("B5", 3,                                                 input_fmt)
    ws.write("A6", "Kenaikan gaji untuk retensi (%):",               bold_fmt)
    ws.write("B6", 0.10,                                              input_fmt)
    ws.write("A7", "Bonus retensi (% gaji tahunan):",                bold_fmt)
    ws.write("B7", 0.15,                                              input_fmt)

    # Hitung dari data aktual
    critical_count   = int((df_flight_risk["flight_risk_score"] >= 0.6).sum())
    merged           = df_employees.merge(df_flight_risk, on="employee_id")
    avg_salary_risk  = int(merged[merged["flight_risk_score"] >= 0.6]["monthly_salary"].mean())

    ws.write("A9",  "HASIL KALKULASI",                                hdr_fmt)
    ws.write("B9",  "",                                               hdr_fmt)
    ws.write("A10", "Total karyawan HIGH risk (score ≥ 0.6):",        bold_fmt)
    ws.write("B10", critical_count,                                    cell_fmt)
    ws.write("A11", "Rata-rata gaji karyawan HIGH risk (Rp):",        bold_fmt)
    ws.write("B11", avg_salary_risk,                                   idr_fmt)

    ws.write("A13", "Total biaya penggantian jika SEMUA resign (Rp):", bold_fmt)
    ws.write_formula("B13", f"=B10*(B11*12*B4+B11*B5)",               idr_fmt)

    ws.write("A14", "Total biaya program retensi (Rp):",              bold_fmt)
    ws.write_formula("B14", f"=B10*(B11*12*B6+B11*12*B7)",           idr_fmt)

    ws.write("A15", "Net penghematan jika program retensi berhasil (Rp):", bold_fmt)
    ws.write_formula("B15", "=B13-B14",                               green_fmt)

    ws.write("A16", "ROI Program Retensi:",                           bold_fmt)
    ws.write_formula("B16", "=(B13-B14)/B14",                         pct_fmt)

    ws.write("A18", "INTERPRETASI",                                   hdr_fmt)
    ws.write("A19", "ROI > 100%: Program retensi sangat layak diinvestasikan", cell_fmt)
    ws.write("A20", "ROI > 200%: Setiap Rp 1 investasi menghasilkan Rp 3 penghematan", cell_fmt)

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.planning.scenario_simulator import run_scenario_comparison
    from src.network.ona_builder import build_employee_graph

    print("Loading data...")
    df_emp        = pd.read_csv("data/simulated/employees.csv")
    df_risk       = pd.read_csv("data/processed/flight_risk_scores.csv")
    df_centrality = pd.read_csv("data/processed/centrality_scores.csv")
    df_training   = pd.read_csv("data/simulated/training_data.csv")
    df_comm       = pd.read_csv("data/simulated/communications.csv")

    print("Building graph for scenarios...")
    G = build_employee_graph(df_comm, df_emp, min_interactions=3)
    scenario_results = run_scenario_comparison(G, df_emp, df_centrality, df_risk)

    print("Generating annual report...")
    generate_annual_workforce_report(
        df_employees     = df_emp,
        df_flight_risk   = df_risk,
        df_centrality    = df_centrality,
        scenario_results = scenario_results,
        df_training      = df_training,
    )