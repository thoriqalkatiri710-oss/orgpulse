import networkx as nx
import pandas as pd
import numpy as np


# ── 7.2.1 Scenario Simulation Engine ─────────────────────────────────────────

def simulate_departure_impact(G: nx.DiGraph,
                               df_employees: pd.DataFrame,
                               departing_employee_ids: list,
                               df_centrality: pd.DataFrame) -> dict:
    """
    Simulasi dampak kehilangan karyawan terhadap kapasitas jaringan.
    7 dimensi dampak: koneksi, diameter, fragmentasi, influence,
    produktivitas, departemen terdampak, waktu pemulihan.
    """
    G_after = G.copy()
    G_after.remove_nodes_from(departing_employee_ids)
    results = {}

    # 1. Koneksi langsung yang hilang
    lost_connections = 0
    for emp_id in departing_employee_ids:
        if emp_id in G:
            lost_connections += G.degree(emp_id)
    results["lost_direct_connections"] = lost_connections

    # 2. Perubahan diameter jaringan
    try:
        if nx.is_weakly_connected(G) and nx.is_weakly_connected(G_after):
            diam_before = nx.diameter(G.to_undirected())
            diam_after  = nx.diameter(G_after.to_undirected())
            results["network_diameter_change"] = diam_after - diam_before
        else:
            results["network_diameter_change"] = "Network fragmented"
    except Exception:
        results["network_diameter_change"] = "N/A"

    # 3. Komponen yang terfragmentasi
    before_cc = nx.number_weakly_connected_components(G)
    after_cc  = nx.number_weakly_connected_components(G_after)
    results["new_isolated_components"] = after_cc - before_cc

    # 4. Kehilangan influence score kumulatif
    total_influence_lost = sum(
        df_centrality[df_centrality["employee_id"] == eid]["influence_score"].values[0]
        for eid in departing_employee_ids
        if eid in df_centrality["employee_id"].values
    )
    results["total_influence_lost"] = round(total_influence_lost, 4)

    # 5. Estimasi dampak produktivitas (proxy)
    est_productivity_impact_pct = min(total_influence_lost * 100 * 2.5, 40)
    results["est_productivity_impact_pct"] = round(est_productivity_impact_pct, 1)

    # 6. Departemen paling terdampak
    affected_depts = df_employees[
        df_employees["employee_id"].isin(departing_employee_ids)
    ]["department"].value_counts().to_dict()
    results["most_affected_departments"] = affected_depts

    # 7. Estimasi waktu pemulihan
    avg_tenure_lost = df_employees[
        df_employees["employee_id"].isin(departing_employee_ids)
    ]["tenure_years"].mean()
    results["est_recovery_months"] = round(avg_tenure_lost * 0.6, 1)

    return results


def run_scenario_comparison(G: nx.DiGraph,
                             df_employees: pd.DataFrame,
                             df_centrality: pd.DataFrame,
                             df_flight_risk: pd.DataFrame) -> pd.DataFrame:
    """
    Bandingkan 3 skenario kehilangan karyawan:
    A: Top-5 flight risk resign
    B: Top-5 influencer resign (poaching)
    C: 5 karyawan acak (baseline)
    """
    top_flight    = df_flight_risk.nlargest(5, "flight_risk_score")["employee_id"].tolist()
    top_influence = df_centrality.nlargest(5, "influence_score")["employee_id"].tolist()
    random_emp    = df_employees.sample(5, random_state=42)["employee_id"].tolist()

    scenarios = {
        "Skenario A: Top-5 Flight Risk": top_flight,
        "Skenario B: Top-5 Influencer":  top_influence,
        "Skenario C: 5 Karyawan Acak":   random_emp,
    }

    records = []
    for name, ids in scenarios.items():
        result = simulate_departure_impact(G, df_employees, ids, df_centrality)
        result["scenario"] = name
        records.append(result)

    return pd.DataFrame(records)[[
        "scenario", "lost_direct_connections", "total_influence_lost",
        "est_productivity_impact_pct", "est_recovery_months", "new_isolated_components"
    ]]


if __name__ == "__main__":
    from src.network.ona_builder import build_employee_graph

    print("Loading data...")
    df_emp        = pd.read_csv("data/simulated/employees.csv")
    df_comm       = pd.read_csv("data/simulated/communications.csv")
    df_centrality = pd.read_csv("data/processed/centrality_scores.csv")
    df_risk       = pd.read_csv("data/processed/flight_risk_scores.csv")

    print("Building graph...")
    G = build_employee_graph(df_comm, df_emp, min_interactions=3)

    print("Running scenario comparison...")
    df_scenarios = run_scenario_comparison(G, df_emp, df_centrality, df_risk)

    print("\n── Scenario Comparison ──")
    print(df_scenarios.to_string(index=False))

    # Narasi C-level
    print("\n── Narasi untuk C-Level ──")
    for _, row in df_scenarios.iterrows():
        print(f"\n{row['scenario']}:")
        print(f"  • Koneksi hilang       : {row['lost_direct_connections']}")
        print(f"  • Influence hilang     : {row['total_influence_lost']:.4f}")
        print(f"  • Dampak produktivitas : {row['est_productivity_impact_pct']}%")
        print(f"  • Estimasi pemulihan   : {row['est_recovery_months']} bulan")
        print(f"  • Komponen terfragmen  : {row['new_isolated_components']}")

    df_scenarios.to_csv("results/reports/scenario_comparison.csv", index=False)
    print("\n✅ Saved: results/reports/scenario_comparison.csv")