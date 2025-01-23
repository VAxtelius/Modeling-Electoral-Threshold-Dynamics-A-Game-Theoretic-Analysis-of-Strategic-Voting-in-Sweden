"""
Preliminary Tests for Logical Soundness (Synthetic)
==================================================

Tests model components with synthetic data:
  - Part A: Utility calculation + 2D plot
  - Part B: Sincere preference selection
  - Part C: Threshold crossing logic
  - Part D: Simple best-response demo in 1D

No real parties or data used here, just basic checks to ensure each part works.
"""

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Part A: Utility Calculation + Plot
###############################################################################
def test_utility_calculation():
    """
    Confirm utility function correctness with a few voters and parties in 2D.
    Plots their positions for a quick visual.
    """
    print("=== Part A: Utility Calculation Test ===")

    # Synthetic party coords (2D)
    party_positions = {
        "PartyA": np.array([2.0, 2.0]),
        "PartyB": np.array([5.0, 1.0])
    }

    # Synthetic voters (2D)
    voters = {
        "Voter1": np.array([2.2, 1.8]),
        "Voter2": np.array([5.1, 1.2]),
        "Voter3": np.array([3.5, 1.5])
    }

    def spatial_utility(voter_coord, party_coord):
        dist_sq = np.sum((voter_coord - party_coord) ** 2)
        return -dist_sq

    # Print utility values
    for voter, v_pos in voters.items():
        for party, p_pos in party_positions.items():
            util_val = spatial_utility(v_pos, p_pos)
            print(f"{voter} utility for {party}: {util_val:.3f}")

    # Plot: voters and parties in 2D
    fig, ax = plt.subplots(figsize=(6, 5))
    for party, coords in party_positions.items():
        ax.scatter(coords[0], coords[1], c='red', s=100, marker='X')
        ax.text(coords[0]+0.1, coords[1], party, fontsize=10, color='red')
    for voter, coords in voters.items():
        ax.scatter(coords[0], coords[1], c='blue', s=50, marker='o')
        ax.text(coords[0]+0.1, coords[1], voter, fontsize=9, color='blue')

    ax.set_title("Part A: 2D Utility Test")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    plt.show()
    print()

###############################################################################
# Part B: Sincere Preference + Plot
###############################################################################
def test_sincere_preferences():
    """
    Each voter chooses the party with highest (negative distance) utility.
    Shows which party they pick via color in a 2D scatter plot.
    """
    print("=== Part B: Sincere Preference Test ===")

    party_positions = {
        "PartyX": np.array([0.0, 0.0]),
        "PartyY": np.array([10.0, 10.0])
    }
    voter_coords = {
        "VoterA": np.array([1.0, 1.0]),
        "VoterB": np.array([9.0, 9.0]),
        "VoterC": np.array([5.0, 5.0]),
    }

    def spatial_utility(voter_coord, party_coord):
        dist_sq = np.sum((voter_coord - party_coord) ** 2)
        return -dist_sq

    voter_preferences = {}
    for voter, v_pos in voter_coords.items():
        best_party = None
        best_util = float("-inf")
        for party, p_pos in party_positions.items():
            u_val = spatial_utility(v_pos, p_pos)
            if u_val > best_util:
                best_util = u_val
                best_party = party
        voter_preferences[voter] = best_party
        print(f"{voter} -> {best_party} (util={best_util:.3f})")

    fig, ax = plt.subplots(figsize=(6, 6))
    for party, coords in party_positions.items():
        ax.scatter(coords[0], coords[1], c='red', s=120, marker='X')
        ax.text(coords[0]+0.2, coords[1], party, fontsize=10, color='red')
    colors = {"PartyX": "blue", "PartyY": "green"}
    for voter, coords in voter_coords.items():
        chosen_party = voter_preferences[voter]
        ax.scatter(coords[0], coords[1], c=colors[chosen_party], s=60, marker='o')
        ax.text(coords[0]+0.2, coords[1], voter, fontsize=9, color=colors[chosen_party])

    ax.set_title("Part B: Sincere Preferences in 2D")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    plt.show()
    print()

###############################################################################
# Part C: Threshold Crossing Logic
###############################################################################
def test_threshold_crossing():
    """
    Parties below threshold are 'failed'. Simple bar chart to show shares.
    """
    print("=== Part C: Threshold Crossing Test ===")

    party_vote_shares = {
        "Party1": 0.12,
        "Party2": 0.04,
        "Party3": 0.039,
        "Party4": 0.25
    }
    threshold = 0.04

    for party, share in party_vote_shares.items():
        crosses = (share >= threshold)
        print(f"{party} {share*100:.2f}% -> crosses? {crosses}")

    print("\nFallback example for Party3 if it fails.\n")

    parties = list(party_vote_shares.keys())
    shares = list(party_vote_shares.values())
    colors = ["green" if s >= threshold else "gray" for s in shares]

    fig, ax = plt.subplots()
    ax.bar(parties, [s*100 for s in shares], color=colors)
    ax.axhline(y=threshold*100, color='r', linestyle='--', label="Threshold")
    ax.set_title("Part C: Threshold Crossing (Vote Shares)")
    ax.set_ylabel("Vote Share (%)")
    ax.legend()
    plt.show()

###############################################################################
# Part D: Simple Best-Response (1D)
###############################################################################
def test_best_response():
    """
    Minimal best-response check for 1 voter and 3 parties in 1D.
    Plots if the voter should switch to a better party.
    """
    print("=== Part D: Simple Best-Response Logic Test ===")

    party_positions = {
        "P_A": 2.0,
        "P_B": 5.0,
        "P_C": 8.0
    }
    voter_coord = 6.0
    current_choice = "P_A"

    def spatial_utility(voter_pos, party_pos):
        return -(voter_pos - party_pos)**2

    current_utility = spatial_utility(voter_coord, party_positions[current_choice])
    print(f"Current choice: {current_choice} => {current_utility:.3f}")

    best_alt_party = current_choice
    best_alt_util = current_utility
    for party, position in party_positions.items():
        alt_util = spatial_utility(voter_coord, position)
        if alt_util > best_alt_util:
            best_alt_util = alt_util
            best_alt_party = party

    if best_alt_party != current_choice:
        print(f"Voter switches to {best_alt_party} (util={best_alt_util:.3f})")
    else:
        print("No better choice found.")

    # Optional plot
    all_positions = sorted(party_positions.values())
    fig, ax = plt.subplots()
    ax.axhline(y=0, color='black', linewidth=0.8)
    for party, x_val in party_positions.items():
        ax.plot(x_val, 0, 'rX', markersize=12)
        ax.text(x_val, 0.05, party, ha='center', color='red', fontsize=9)
    ax.plot(voter_coord, 0, 'bo', markersize=8, label="Voter")
    ax.text(voter_coord, -0.05, "Voter", ha='center', color='blue', fontsize=9)
    ax.set_ylim([-0.2, 0.2])
    ax.set_xlim([min(all_positions)-1, max(all_positions)+1])
    ax.set_title("Part D: Best-Response in 1D")
    ax.get_yaxis().set_visible(False)
    plt.legend()
    plt.show()
    print()

###############################################################################
# Main: Run Tests
###############################################################################
def main():
    print("=======================================================")
    print("Preliminary Logical Soundness Tests (Synthetic)")
    print("=======================================================\n")
    test_utility_calculation()
    test_sincere_preferences()
    test_threshold_crossing()
    test_best_response()

if __name__ == "__main__":
    main()
