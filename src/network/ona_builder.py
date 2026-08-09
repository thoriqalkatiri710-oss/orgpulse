import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ── 4.1.1 Build Employee Graph ────────────────────────────────────────────────

def build_employee_graph(df_interactions: pd.DataFrame,
                          df_employees: pd.DataFrame,
                          min_interactions: int = 3) -> nx.DiGraph:
    """
    Bangun directed weighted graph dari data interaksi.
    Node = karyawan, Edge = interaksi dengan bobot frekuensi.
    min_interactions: filter noise — hanya edge dengan ≥N interaksi.
    """
    G = nx.DiGraph()

    # Tambahkan node dengan atribut
    for _, emp in df_employees.iterrows():
        G.add_node(
            emp["employee_id"],
            name=emp["full_name"],
            department=emp["department"],
            job_level=emp["job_level"],
            engagement=emp["engagement_score"],
        )

    # Agregasi interaksi menjadi edge berbobot
    edge_agg = df_interactions.groupby(["sender_id", "receiver_id"]).agg(
        weight=("interaction_id", "count"),
        avg_sentiment=("sentiment_raw", "mean")
    ).reset_index()

    # Filter noise — hanya edge dengan minimal N interaksi
    for _, row in edge_agg[edge_agg["weight"] >= min_interactions].iterrows():
        G.add_edge(
            row["sender_id"], row["receiver_id"],
            weight=row["weight"],
            avg_sentiment=row["avg_sentiment"]
        )

    return G
# ── 4.2.1 Centrality Metrics ──────────────────────────────────────────────────

def compute_all_centrality(G: nx.DiGraph) -> pd.DataFrame:
    """
    Hitung semua centrality metrics untuk setiap node.
    Composite influence score = weighted average dari 4 metrik utama.
    """
    n = G.number_of_nodes()
    in_degree  = dict(G.in_degree(weight="weight"))
    out_degree = dict(G.out_degree(weight="weight"))

    # Betweenness: approximation untuk graph besar (k=200 sample)
    betweenness = nx.betweenness_centrality(
        G, weight="weight",
        k=min(200, n) if n > 200 else None,
        normalized=True
    )

    # Eigenvector centrality
    try:
        eigenvector = nx.eigenvector_centrality(
            G, weight="weight", max_iter=1000, tol=1e-6
        )
    except nx.PowerIterationFailedConvergence:
        eigenvector = {node: 0 for node in G.nodes()}

    # PageRank
    pagerank = nx.pagerank(G, weight="weight", alpha=0.85)

    # Clustering coefficient (undirected projection)
    G_undir   = G.to_undirected()
    clustering = nx.clustering(G_undir, weight="weight")

    # Gabungkan semua metrik
    records = []
    for node in G.nodes():
        records.append({
            "employee_id":          node,
            "in_degree_weighted":   in_degree.get(node, 0),
            "out_degree_weighted":  out_degree.get(node, 0),
            "betweenness":          betweenness.get(node, 0),
            "eigenvector":          eigenvector.get(node, 0),
            "pagerank":             pagerank.get(node, 0),
            "clustering_coeff":     clustering.get(node, 0),
        })

    df = pd.DataFrame(records)

    # Normalisasi 0-1
    for col in ["in_degree_weighted", "out_degree_weighted",
                "betweenness", "eigenvector", "pagerank"]:
        max_val = df[col].max()
        df[f"{col}_norm"] = df[col] / max_val if max_val > 0 else 0

    # Composite ONA Influence Score
    df["influence_score"] = (
        df["betweenness_norm"]          * 0.30 +
        df["pagerank_norm"]             * 0.25 +
        df["eigenvector_norm"]          * 0.25 +
        df["in_degree_weighted_norm"]   * 0.20
    ).round(4)

    return df.sort_values("influence_score", ascending=False)
# ── 4.3.1 Network Role Classification ────────────────────────────────────────

def classify_network_roles(df_centrality: pd.DataFrame) -> pd.DataFrame:
    """
    Klasifikasi otomatis peran jaringan berdasarkan threshold top-20%.
    Peran: Central Connector, Information Broker, Energizer/Hub,
           Peripheral Specialist, Peripheral/Isolated, Regular Member.
    """
    df = df_centrality.copy()

    # Threshold top-20% di tiap dimensi
    bc_thresh  = df["betweenness_norm"].quantile(0.80)
    in_thresh  = df["in_degree_weighted_norm"].quantile(0.80)
    out_thresh = df["out_degree_weighted_norm"].quantile(0.80)
    cc_thresh  = df["clustering_coeff"].quantile(0.80)

    def assign_role(row):
        bc  = row["betweenness_norm"]        >= bc_thresh
        ind = row["in_degree_weighted_norm"] >= in_thresh
        oud = row["out_degree_weighted_norm"] >= out_thresh
        cc  = row["clustering_coeff"]        >= cc_thresh

        if bc and (ind or oud):
            return "Central Connector"
        elif bc and not ind and not oud:
            return "Information Broker"
        elif ind and not bc:
            return "Energizer/Hub"
        elif cc and not bc and not ind:
            return "Peripheral Specialist"
        elif row["influence_score"] < df["influence_score"].quantile(0.20):
            return "Peripheral/Isolated"
        else:
            return "Regular Member"

    df["network_role"] = df.apply(assign_role, axis=1)
    return df


# ── 4.4.1 ONA Network Visualization ──────────────────────────────────────────

DEPT_COLORS = {
    "Engineering":      "#3B82F6",
    "Product":          "#8B5CF6",
    "Marketing":        "#EC4899",
    "Sales":            "#EF4444",
    "Finance":          "#10B981",
    "HR":               "#F59E0B",
    "Operations":       "#6B7280",
    "Data & Analytics": "#14B8A6",
    "Legal":            "#F97316",
    "Customer Success": "#06B6D4",
    "IT Infrastructure":"#84CC16",
    "Research":         "#A855F7",
}


def plot_ona_network(G: nx.DiGraph, df_employees: pd.DataFrame,
                     df_centrality: pd.DataFrame,
                     save_path: str = None):
    """
    Force-directed network plot.
    Node size = influence score, Node color = department.
    Label ditampilkan untuk top-100 influencer.
    """
    dept_map  = df_employees.set_index("employee_id")["department"].to_dict()
    score_map = df_centrality.set_index("employee_id")["influence_score"].to_dict()

    pos = nx.spring_layout(G, k=2.5, seed=42, iterations=100)

    fig, ax = plt.subplots(figsize=(20, 16))

    node_colors = [DEPT_COLORS.get(dept_map.get(n, ""), "#cccccc") for n in G.nodes()]
    node_sizes  = [300 + score_map.get(n, 0) * 2000 for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5,
                           edge_color="gray", arrows=True, arrowsize=8, ax=ax)

    # Label untuk top-100 influencer
    top100 = df_centrality.head(100)["employee_id"].tolist()
    labels = {
        n: df_employees[df_employees["employee_id"] == n]["full_name"].values[0].split()[0]
        for n in top100 if n in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, ax=ax)

    legend_handles = [Patch(color=c, label=d) for d, c in DEPT_COLORS.items()]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    ax.set_title(
        "OrgPulse — Organizational Network Analysis\n"
        "Node size = Influence Score | Node color = Department | Labels = Top 100 Influencer",
        fontsize=14
    )
    ax.axis("off")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")

    return fig

# ── 4.5.1 Community Detection ─────────────────────────────────────────────────

def detect_communities_hr(G: nx.DiGraph,
                           df_employees: pd.DataFrame) -> tuple:
    """
    Louvain community detection untuk menemukan klaster tersembunyi.
    Identifikasi 'silo communities' — komunitas yang hanya terdiri
    dari satu departemen, berisiko untuk inovasi cross-functional.
    """
    import community as community_louvain

    G_undir   = G.to_undirected()
    partition = community_louvain.best_partition(G_undir, weight="weight", random_state=42)

    community_df = pd.DataFrame([
        {"employee_id": node, "community_id": cid}
        for node, cid in partition.items()
    ])
    community_df = community_df.merge(
        df_employees[["employee_id", "department", "job_level", "engagement_score"]],
        how="left"
    )

    # Analisis setiap komunitas
    community_summary = community_df.groupby("community_id").agg(
        n_members        =("employee_id",      "count"),
        departments      =("department",       lambda x: ", ".join(sorted(x.unique()))),
        avg_engagement   =("engagement_score", "mean"),
        avg_level        =("job_level",        "mean"),
        dept_diversity   =("department",       "nunique"),
    ).reset_index()

    # Flag silo communities
    community_summary["is_silo"]    = community_summary["dept_diversity"] == 1
    community_summary["silo_risk"]  = (
        community_summary["is_silo"] & (community_summary["n_members"] > 10)
    )

    return community_df, community_summary



if __name__ == "__main__":
    df_emp  = pd.read_csv("data/simulated/employees.csv")
    df_comm = pd.read_csv("data/simulated/communications.csv")

    print("Building employee graph...")
    G = build_employee_graph(df_comm, df_emp, min_interactions=3)

    print(f"\n── Graph Stats ──")
    print(f"Nodes             : {G.number_of_nodes()}")
    print(f"Edges             : {G.number_of_edges()}")
    print(f"Density           : {nx.density(G):.4f}")

    print("\nComputing centrality metrics...")
    df_centrality = compute_all_centrality(G)

    # Merge dengan info karyawan
    df_centrality = df_centrality.merge(
        df_emp[["employee_id", "full_name", "department", "job_level_name"]],
        on="employee_id"
    )

    print(f"\n── Top 10 Most Influential Employees ──")
    print(df_centrality[["full_name", "department", "job_level_name",
                          "influence_score", "betweenness", "pagerank"]].head(10).to_string(index=False))

    df_centrality.to_csv("data/processed/centrality_scores.csv", index=False)
    print("\n✅ Saved: data/processed/centrality_scores.csv")

    print("\nClassifying network roles...")
    df_centrality = classify_network_roles(df_centrality)

    print(f"\n── Network Role Distribution ──")
    print(df_centrality["network_role"].value_counts().to_string())

    print(f"\n── Role per Department (Top 5) ──")
    role_dept = df_centrality.groupby(["department", "network_role"]).size().reset_index(name="count")
    print(role_dept.sort_values("count", ascending=False).head(10).to_string(index=False))

    df_centrality.to_csv("data/processed/centrality_scores.csv", index=False)
    print("\n✅ Saved: data/processed/centrality_scores.csv")

    print("\nPlotting ONA network...")
    fig = plot_ona_network(G, df_emp, df_centrality,
                           save_path="results/figures/ona_network.png")
    plt.close()
    print("✅ Plot saved: results/figures/ona_network.png")

    print("\nDetecting communities...")
    community_df, community_summary = detect_communities_hr(G, df_emp)

    print(f"\n── Community Summary ──")
    print(f"Total communities: {len(community_summary)}")
    print(f"Silo communities : {community_summary['is_silo'].sum()}")
    print(f"High-risk silos  : {community_summary['silo_risk'].sum()}")
    print(f"\nTop 10 largest communities:")
    print(community_summary.sort_values("n_members", ascending=False)
          [["community_id", "n_members", "dept_diversity", "avg_engagement", "is_silo", "silo_risk"]]
          .head(10).to_string(index=False))

    community_df.to_csv("data/processed/community_assignments.csv", index=False)
    community_summary.to_csv("data/processed/community_summary.csv", index=False)
    print("\n✅ Saved: community_assignments.csv & community_summary.csv")