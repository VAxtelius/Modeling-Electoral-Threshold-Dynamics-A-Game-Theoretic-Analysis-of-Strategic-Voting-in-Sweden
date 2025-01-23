import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr

# Basic config
SEED = 42
np.random.seed(SEED)
N_VOTERS = 3000
ELECTORAL_THRESHOLD = 0.04
COALITION_BONUS = 2.0

# Party coordinates
parties_3d = {
    "Moderate Party":          np.array([7.0,  6.5,  4.0]),
    "People's Party Liberals": np.array([6.5,  5.0,  5.5]),
    "Centre Party":            np.array([6.0,  7.0,  7.5]),
    "Christian Democrats":     np.array([5.8,  8.5,  3.0]),
    "Social Democrats":        np.array([3.5,  3.5,  5.0]),
    "Left Party":              np.array([2.5,  2.5,  5.5]),
    "Green Party":             np.array([4.0,  4.5,  8.0]),
    "Sweden Democrats":        np.array([6.0,  8.0,  2.5])
}

center_right_alliance = ["Moderate Party", "People's Party Liberals", "Centre Party", "Christian Democrats"]
red_greens = ["Social Democrats", "Left Party", "Green Party"]
coalition_labels = {}
for p in center_right_alliance:
    coalition_labels[p] = "center_right"
for p in red_greens:
    coalition_labels[p] = "red_greens"
coalition_labels["Sweden Democrats"] = "outside"

poll_based_phi = {
    "Moderate Party":          0.999,
    "People's Party Liberals": 0.95,
    "Centre Party":            0.90,
    "Christian Democrats":     0.75,
    "Social Democrats":        0.999,
    "Left Party":              0.85,
    "Green Party":             0.95,
    "Sweden Democrats":        0.70
}

coefficients_data = {
    "Variable": [
        "First party preference (0–1)",
        "Sympathy score (0–10)",
        "Coalition preference (0–1)",
        "Left-right distance (0–10)",
        "Party identification",
        "Party leader evaluation (0–10)",
        "Threshold",
        "Insurance (0–1)",
        "Government evaluation (0–10)",
        "Age (18–84)",
        "Education (0–1)",
        "Sex (0–1)",
        "Constant"
    ],
    "Moderate Party":          [1.20, 0.50, 1.26, -0.42, 1.36, 0.03, 3.05, 1.12, -0.64, -0.05, 0.59, 0.15, 5.37],
    "Social Democrats":        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.46, 0.91, -0.46, -0.01, -0.85, -1.27, 4.86],
    "Centre Party":            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.66, 0.71, -0.21, 0.01, 0.58, -0.54, -0.14],
    "People's Party Liberals": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.04, 0.71, -0.13, 0.02, 1.04, 0.57, -2.24],
    "Christian Democrats":     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 2.57, 0.65, -0.57, 0.03, 1.26, -0.29, 1.65],
    "Green Party":             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.36, 0.96, -0.47, -0.04, 1.11, -0.02, 3.19],
    "Left Party":              [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 2.09, 1.00, -0.31, 0.06, 0.24, 0.44, -0.21],
    "Sweden Democrats":        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, -0.31, 0.00, 0.00, 0.00, -0.21]
}
df_file3 = pd.DataFrame(coefficients_data)
df_file3.set_index("Variable", inplace=True)

partial_derivatives = {
    "Christian Democrats": {"dy/dx": 0.24, "Std. error": 0.11},
    "Centre Party":         {"dy/dx": 0.04, "Std. error": 0.03},
    "Left Party":           {"dy/dx": 0.01, "Std. error": 0.02},
    "Sweden Democrats":     {"dy/dx": 0.02, "Std. error": 0.03}
}
for party, metrics in partial_derivatives.items():
    dy_dx = metrics["dy/dx"]
    df_file3.loc["Party identification", party] += dy_dx

def generate_voters(n: int) -> pd.DataFrame:
    # Make clusters
    cluster_sizes = [int(n * 0.45), int(n * 0.45), n - int(n * 0.45)*2]
    means = [(6.0, 6.5, 4.0), (3.5, 3.5, 6.0), (6.2, 8.0, 3.0)]
    stds = [(1.2, 1.2, 1.2), (1.2, 1.2, 1.2), (1.0, 1.0, 1.0)]
    all_voter_arrays = []
    for size, (mx, my, mz), (sx, sy, sz) in zip(cluster_sizes, means, stds):
        x_vals = np.random.normal(mx, sx, size=size).clip(0, 10)
        y_vals = np.random.normal(my, sy, size=size).clip(0, 10)
        z_vals = np.random.normal(mz, sz, size=size).clip(0, 10)
        cluster_array = np.column_stack([x_vals, y_vals, z_vals])
        all_voter_arrays.append(cluster_array)
    final_voters = np.vstack(all_voter_arrays)
    np.random.shuffle(final_voters)
    df_voters = pd.DataFrame({
        "voter_id": range(n),
        "x_pos": final_voters[:, 0],
        "y_pos": final_voters[:, 1],
        "z_pos": final_voters[:, 2]
    })
    return df_voters

def spatial_utility(voter_coord: np.ndarray, party_coord: np.ndarray) -> float:
    dist_sq = np.sum((voter_coord - party_coord) ** 2)
    return -dist_sq

def fallback_utility(voter_coord: np.ndarray, parties_dict: dict, excluded_party: str) -> float:
    best_alt = float("-inf")
    for party, coords in parties_dict.items():
        if party != excluded_party:
            alt_util = spatial_utility(voter_coord, coords)
            if alt_util > best_alt:
                best_alt = alt_util
    return best_alt

def expected_utility(voter_coord: np.ndarray,
                     party: str,
                     phi_j: float,
                     parties_dict: dict) -> float:
    base_util = spatial_utility(voter_coord, parties_dict[party])
    fb_util = fallback_utility(voter_coord, parties_dict, party)
    return phi_j * base_util + (1.0 - phi_j) * fb_util

def coalition_utility_bonus(party: str,
                            desired_coal: str,
                            coalition_map: dict,
                            bonus_value: float = COALITION_BONUS) -> float:
    chosen_coal = coalition_map.get(party, "outside")
    return bonus_value if chosen_coal == desired_coal else 0.0

def extended_expected_utility(voter_coord: np.ndarray,
                              party: str,
                              phi_j: float,
                              parties_dict: dict,
                              coalition_map: dict,
                              desired_coal: str) -> float:
    base_eu = expected_utility(voter_coord, party, phi_j, parties_dict)
    c_bonus = coalition_utility_bonus(party, desired_coal, coalition_map)
    return base_eu + c_bonus

def simulate_votes(voters_df: pd.DataFrame,
                   parties_dict: dict,
                   phi_dict: dict,
                   coalition_map: dict) -> pd.DataFrame:
    records = []
    for _, row in voters_df.iterrows():
        v_id = row["voter_id"]
        v_coord = np.array([row["x_pos"], row["y_pos"], row["z_pos"]])
        desired_coal = row["desired_coalition"]
        best_strat_party = None
        best_strat_util = float("-inf")
        for p in parties_dict:
            phi_j = phi_dict[p]
            eu_p = extended_expected_utility(v_coord, p, phi_j, parties_dict, coalition_map, desired_coal)
            if eu_p > best_strat_util:
                best_strat_util = eu_p
                best_strat_party = p
        switched_flag = int(row["sincere_pref"] != best_strat_party)
        records.append({
            "voter_id": v_id,
            "strategic_vote": best_strat_party,
            "strategic_util": best_strat_util,
            "switched": switched_flag
        })
    return pd.DataFrame(records)

def assign_desired_coalition(voters_df: pd.DataFrame) -> pd.DataFrame:
    voters_df = voters_df.copy()
    voters_df["desired_coalition"] = np.where(np.random.rand(len(voters_df)) < 0.5, "center_right", "red_greens")
    return voters_df

def assign_sincere_preferences(voters_df: pd.DataFrame) -> pd.DataFrame:
    voters_df = voters_df.copy()
    sincere_prefs = []
    for _, row in voters_df.iterrows():
        voter_coord = np.array([row["x_pos"], row["y_pos"], row["z_pos"]])
        best_party = None
        best_util = float("-inf")
        for p in parties_3d:
            util = spatial_utility(voter_coord, parties_3d[p])
            if util > best_util:
                best_util = util
                best_party = p
        sincere_prefs.append(best_party)
    voters_df["sincere_pref"] = sincere_prefs
    return voters_df

def assign_voter_attributes(voters_df: pd.DataFrame, file2_df: pd.DataFrame, full_to_abbrev: dict) -> pd.DataFrame:
    def assign_attributes(row):
        party = row["sincere_pref"]
        abbrev = full_to_abbrev.get(party, "")
        if abbrev:
            prefs = file2_df[file2_df["Vote"] == f"{party} ({abbrev})"].iloc[0]
            attributes = {
                "V": np.random.binomial(1, prefs["V"]/100),
                "S": np.random.binomial(1, prefs["S"]/100),
                "C": np.random.binomial(1, prefs["C"]/100),
                "FP": np.random.binomial(1, prefs["FP"]/100),
                "M": np.random.binomial(1, prefs["M"]/100),
                "KD": np.random.binomial(1, prefs["KD"]/100),
                "MP": np.random.binomial(1, prefs["MP"]/100),
                "SD": np.random.binomial(1, prefs["SD"]/100)
            }
            return pd.Series(attributes)
        return pd.Series({"V":0, "S":0, "C":0, "FP":0, "M":0, "KD":0, "MP":0, "SD":0})
    attributes_df = voters_df.apply(assign_attributes, axis=1)
    return pd.concat([voters_df, attributes_df], axis=1)

def plot_3d_equilibrium(voters_df: pd.DataFrame, parties_dict: dict):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    color_map = {
        "Moderate Party":          "navy",
        "People's Party Liberals": "teal",
        "Centre Party":            "darkgreen",
        "Christian Democrats":     "sienna",
        "Social Democrats":        "crimson",
        "Left Party":              "indigo",
        "Green Party":             "limegreen",
        "Sweden Democrats":        "gold"
    }
    for p in parties_dict.keys():
        subset = voters_df[voters_df["strategic_vote"] == p]
        ax.scatter(subset["x_pos"], subset["y_pos"], subset["z_pos"],
                   s=10, alpha=0.2, c=color_map.get(p, "gray"), label=None)
    for p, coords in parties_dict.items():
        ax.scatter(coords[0], coords[1], coords[2],
                   marker="X", s=400,
                   c=color_map.get(p, "black"),
                   edgecolors="white",
                   linewidths=3,
                   label=f"{p} (Party)")
        ax.text(coords[0], coords[1], coords[2] + 0.5,
                f" {p}", fontsize=14, fontweight='bold',
                color='black', ha='center', va='bottom')
    ax.set_xlabel("X (Economic)", fontsize=16, labelpad=15, fontweight='bold')
    ax.set_ylabel("Y (Social)", fontsize=16, labelpad=15, fontweight='bold', rotation=90)
    ax.set_zlabel("Z (Other)", fontsize=16, labelpad=15, fontweight='bold')
    ax.set_title("3D Policy Space", fontsize=20, pad=20, fontweight='bold')
    ax.view_init(elev=15, azim=60)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(),
              loc="upper left", fontsize='large', frameon=True)
    plt.tight_layout()
    plt.show()

def plot_vote_shares(sincere_df: pd.Series, strategic_df: pd.Series):
    parties_union = sorted(set(sincere_df.index).union(strategic_df.index))
    comp_df = pd.DataFrame({
        "Party": parties_union,
        "Sincere": [sincere_df.get(p, 0) for p in parties_union],
        "Strategic": [strategic_df.get(p, 0) for p in parties_union]
    }).set_index("Party").sort_index()
    ax = comp_df.plot(kind="bar", figsize=(10, 6), color=["skyblue", "salmon"])
    ax.set_ylabel("Vote Share (%)", fontsize=12)
    ax.set_ylim(0, comp_df.values.max() * 1.1)
    ax.set_title("Sincere vs. Strategic", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Voting", fontsize='medium', title_fontsize='13')
    plt.tight_layout()
    plt.show()

def additional_statistics(merged_df: pd.DataFrame):
    print("=== Stats ===")
    avg_sincere_util = merged_df["sincere_util"].mean()
    avg_strategic_util = merged_df["strategic_util"].mean()
    print(f"Sincere Util: {avg_sincere_util:.3f}, Strategic Util: {avg_strategic_util:.3f}")
    print(f"Gain: {(avg_strategic_util - avg_sincere_util):.3f}")
    total_switchers = merged_df["switched"].sum()
    switch_pct = (total_switchers / len(merged_df)) * 100
    print(f"Switchers: {total_switchers}, {switch_pct:.2f}%")
    switchers_to_party = merged_df[merged_df["switched"] == 1].groupby("strategic_vote")["voter_id"].count()
    print("\nTo Party:\n", switchers_to_party.sort_values(ascending=False))
    switchers_from_party = merged_df[merged_df["switched"] == 1].groupby("sincere_pref")["voter_id"].count()
    print("\nFrom Party:\n", switchers_from_party.sort_values(ascending=False))
    util_metrics = merged_df[["sincere_util", "strategic_util"]].agg(['mean', 'median', 'std', 'min', 'max'])
    print("\nUtil Dist:\n", util_metrics)
    switchers = merged_df["switched"] == 1
    if switchers.sum() > 1:
        corr, pv = pearsonr(merged_df.loc[switchers, "sincere_util"],
                            merged_df.loc[switchers, "strategic_util"])
        print(f"\nCorr (switchers): {corr:.3f}, p={pv:.3f}")
    aligned_coalition = merged_df.apply(
        lambda row: coalition_labels.get(row["strategic_vote"], "outside") == row["desired_coalition"],
        axis=1
    )
    aligned_count = aligned_coalition.sum()
    aligned_pct = (aligned_count / len(merged_df)) * 100
    print(f"Aligned coalition: {int(aligned_count)}, {aligned_pct:.2f}% (+{COALITION_BONUS} bonus)")
    party_util = merged_df.groupby("strategic_vote")[["sincere_util", "strategic_util"]].agg(['mean', 'median'])
    print("\nParty Utility:\n", party_util)
    merged_df["sincere_distance"] = merged_df.apply(
        lambda row: np.linalg.norm(np.array([row["x_pos"], row["y_pos"], row["z_pos"]]) - parties_3d[row["sincere_pref"]]),
        axis=1
    )
    merged_df["strategic_distance"] = merged_df.apply(
        lambda row: np.linalg.norm(np.array([row["x_pos"], row["y_pos"], row["z_pos"]]) - parties_3d[row["strategic_vote"]]),
        axis=1
    )
    avg_sincere_dist = merged_df["sincere_distance"].mean()
    avg_strat_dist = merged_df["strategic_distance"].mean()
    print(f"\nDist sincere: {avg_sincere_dist:.3f}, Dist strategic: {avg_strat_dist:.3f}, Diff: {(avg_strat_dist - avg_sincere_dist):.3f}")
    if merged_df["sincere_distance"].nunique() > 1 and merged_df["strategic_distance"].nunique() > 1:
        c_dist, p_dist = pearsonr(merged_df["sincere_distance"], merged_df["strategic_distance"])
        print(f"Corr Dist: {c_dist:.3f}, p={p_dist:.3f}")

def main():
    voters_df = generate_voters(N_VOTERS)
    voters_with_coal = assign_desired_coalition(voters_df)
    voters_with_prefs = assign_sincere_preferences(voters_with_coal)
    data_file2 = {
        "Vote": [
            "Left Party (V)", "Social Democrats (S)", "Centre Party (C)",
            "People's Party Liberals (FP)", "Moderate Party (M)",
            "Christian Democrats (KD)", "Green Party (MP)", "Sweden Democrats (SD)"
        ],
        "V": [72.3, 31.9, 6.4, 10.6, 8.5, 2.1, 46.8, 2.1],
        "S": [9.8, 83.3, 3.3, 5.3, 9.0, 4.5, 30.2, 2.9],
        "C": [1.7, 8.3, 65.0, 20.0, 43.3, 8.3, 16.7, 0.0],
        "FP": [2.9, 11.4, 7.1, 72.9, 47.1, 7.1, 10.0, 2.9],
        "M": [1.1, 4.7, 5.0, 10.0, 93.6, 5.4, 6.1, 1.8],
        "KD": [1.9, 3.8, 18.9, 13.2, 50.9, 49.1, 9.4, 0.0],
        "MP": [7.8, 15.6, 3.9, 3.9, 7.8, 1.3, 92.2, 1.3],
        "SD": [5.6, 13.9, 5.6, 13.9, 27.8, 2.8, 13.9, 61.1]
    }
    file2_df = pd.DataFrame(data_file2)
    full_to_abbrev = {
        "Left Party": "V",
        "Social Democrats": "S",
        "Centre Party": "C",
        "People's Party Liberals": "FP",
        "Moderate Party": "M",
        "Christian Democrats": "KD",
        "Green Party": "MP",
        "Sweden Democrats": "SD"
    }
    voters_with_attrs = assign_voter_attributes(voters_with_prefs, file2_df, full_to_abbrev)
    voters_with_attrs["sincere_util"] = voters_with_attrs.apply(
        lambda row: spatial_utility(np.array([row["x_pos"], row["y_pos"], row["z_pos"]]),
                                    parties_3d[row["sincere_pref"]]),
        axis=1
    )
    results_df = simulate_votes(voters_with_attrs, parties_3d, poll_based_phi, coalition_labels)
    merged_df = pd.merge(voters_with_attrs, results_df, on="voter_id")
    if 'sincere_pref' not in merged_df.columns:
        raise KeyError("'sincere_pref' missing.")
    if 'sincere_util' not in merged_df.columns:
        raise KeyError("'sincere_util' missing.")
    switch_rate = merged_df["switched"].mean() * 100
    print(f"Non-sincere votes: {switch_rate:.2f}%\n")
    strat_shares = merged_df.groupby("strategic_vote")["voter_id"].count() / len(merged_df) * 100
    sincere_shares = merged_df.groupby("sincere_pref")["voter_id"].count() / len(merged_df) * 100
    print("Strategic Votes (%):")
    print(strat_shares.sort_values(ascending=False).round(2).to_string())
    print("\nSincere Votes (%):")
    print(sincere_shares.sort_values(ascending=False).round(2).to_string())
    kd_strat_from_others = merged_df.loc[
        (merged_df["strategic_vote"] == "Christian Democrats") &
        (merged_df["sincere_pref"] != "Christian Democrats")
    ]
    kd_share_pct = 100 * len(kd_strat_from_others) / len(merged_df)
    print(f"\nKD from others: {len(kd_strat_from_others)} ({kd_share_pct:.2f}%)\n")
    switched_df = merged_df[merged_df["switched"] == 1]
    from_to_counts = (switched_df.groupby(["sincere_pref", "strategic_vote"])["voter_id"]
                      .count()
                      .reset_index())
    from_to_counts.columns = ["from_party", "to_party", "num_voters"]
    from_to_counts["pct_of_sample"] = from_to_counts["num_voters"] / len(merged_df) * 100
    print("Switches:")
    if not from_to_counts.empty:
        print(from_to_counts.sort_values("num_voters", ascending=False).to_string(index=False))
    else:
        print("No switches.\n")
    additional_statistics(merged_df)
    plot_3d_equilibrium(merged_df, parties_3d)
    plot_vote_shares(sincere_shares, strat_shares)

if __name__ == "__main__":
    main()
