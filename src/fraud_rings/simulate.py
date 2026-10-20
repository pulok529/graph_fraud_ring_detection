"""Synthetic transaction network simulator with realistic fraud rings and confounders."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate(
    n_accounts: int = 3000,
    n_rings: int = 12,
    seed: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic financial transaction dataset with embedded fraud rings.

    Simulates realistic legitimate activity (preferential attachment hub payments,
    random peer-to-peer transfers, and benign device-sharing confounders like families)
    alongside sophisticated fraud rings (tight cycle transfers with near-equal amounts,
    shared device footprints, recent creation dates, and peripheral members).

    Parameters
    ----------
    n_accounts : int, default=3000
        Total number of accounts to simulate.
    n_rings : int, default=12
        Number of fraud rings to embed.
    seed : int, default=5
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        accounts_df : DataFrame with columns [account_id, created_day, kyc_level,
                      device_id, is_ring_member, ring_id]
        transfers_df : DataFrame with columns [src, dst, amount, day]
    """
    rng = np.random.default_rng(seed)

    # 1. Determine ring sizes: 5-15 accounts per ring, ~3-5% total ring accounts
    target_ring_accounts = max(5 * n_rings, min(15 * n_rings, int(n_accounts * 0.04)))
    ring_sizes: list[int] = []
    remaining = target_ring_accounts
    for i in range(n_rings):
        rings_left = n_rings - i
        if rings_left == 1:
            size = int(np.clip(remaining, 5, 15))
        else:
            avg_left = remaining / rings_left
            low = max(5, int(avg_left) - 2)
            high = min(15, int(avg_left) + 2)
            size = int(rng.integers(low, high + 1)) if low <= high else 8
            # Ensure bounds are respected for remainder
            size = max(5, min(size, remaining - 5 * (rings_left - 1)))
            size = min(15, max(size, remaining - 15 * (rings_left - 1)))
        ring_sizes.append(size)
        remaining -= size

    total_ring_accounts = sum(ring_sizes)
    n_legit = n_accounts - total_ring_accounts

    # 2. Build Account Profiles
    account_records: list[dict[str, object]] = []

    # 2a. Confounder: Legitimate small communities (families/roommates)
    # They share devices and transact densely, but are older and have diverse amounts
    n_family_groups = max(2, int(n_legit * 0.015))
    family_members_assigned = 0
    family_groups: list[list[str]] = []
    current_acc_idx = 0

    for g in range(n_family_groups):
        fsize = int(rng.integers(2, 5))
        dev_id = f"dev_family_{g:03d}"
        group_accs = []
        for _ in range(fsize):
            acc_id = f"acc_{current_acc_idx:05d}"
            group_accs.append(acc_id)
            account_records.append(
                {
                    "account_id": acc_id,
                    "created_day": int(rng.integers(10, 95)),  # older accounts
                    "kyc_level": int(rng.choice([2, 3], p=[0.4, 0.6])),  # established kyc
                    "device_id": dev_id,  # shared device
                    "is_ring_member": 0,
                    "ring_id": -1,
                }
            )
            current_acc_idx += 1
            family_members_assigned += 1
        family_groups.append(group_accs)

    # 2b. Hub merchants (preferential attachment sinks)
    n_merchants = max(5, int(n_legit * 0.03))
    merchant_ids: list[str] = []
    for _ in range(n_merchants):
        acc_id = f"acc_{current_acc_idx:05d}"
        merchant_ids.append(acc_id)
        account_records.append(
            {
                "account_id": acc_id,
                "created_day": int(rng.integers(0, 30)),
                "kyc_level": 3,
                "device_id": f"dev_merch_{current_acc_idx:05d}",
                "is_ring_member": 0,
                "ring_id": -1,
            }
        )
        current_acc_idx += 1

    # 2c. General legitimate accounts
    remaining_legit = n_legit - family_members_assigned - n_merchants
    general_legit_ids: list[str] = []
    for _ in range(remaining_legit):
        acc_id = f"acc_{current_acc_idx:05d}"
        general_legit_ids.append(acc_id)
        account_records.append(
            {
                "account_id": acc_id,
                "created_day": int(rng.integers(0, 160)),
                "kyc_level": int(rng.choice([1, 2, 3], p=[0.15, 0.45, 0.40])),
                "device_id": f"dev_usr_{current_acc_idx:05d}",
                "is_ring_member": 0,
                "ring_id": -1,
            }
        )
        current_acc_idx += 1

    # 2d. Fraud ring accounts
    ring_account_groups: list[list[str]] = []
    for r_idx, r_size in enumerate(ring_sizes):
        # Rings share a small pool of devices (1-3 devices per ring)
        n_ring_devices = max(1, min(3, r_size // 3))
        ring_devices = [f"dev_ring_{r_idx:02d}_{d}" for d in range(n_ring_devices)]

        r_members: list[str] = []
        for _ in range(r_size):
            acc_id = f"acc_{current_acc_idx:05d}"
            r_members.append(acc_id)
            account_records.append(
                {
                    "account_id": acc_id,
                    # Mostly new accounts, but ~40% are aged "mule" accounts bought or recruited
                    "created_day": int(
                        rng.integers(120, 181) if rng.random() < 0.6 else rng.integers(10, 160)
                    ),
                    "kyc_level": int(rng.choice([1, 2, 3], p=[0.45, 0.35, 0.20])),
                    # Only about half the members operate from a shared ring device
                    "device_id": str(rng.choice(ring_devices))
                    if rng.random() < 0.5
                    else f"dev_usr_{current_acc_idx:05d}",
                    "is_ring_member": 1,
                    "ring_id": r_idx,
                }
            )
            current_acc_idx += 1
        ring_account_groups.append(r_members)

    accounts_df = pd.DataFrame(account_records)
    created_map = dict(zip(accounts_df["account_id"], accounts_df["created_day"], strict=False))

    # 3. Simulate Transfers
    transfer_records: list[dict[str, object]] = []

    # 3a. Legitimate merchant payments (preferential attachment)
    all_normal_legit = [r["account_id"] for r in account_records if r["is_ring_member"] == 0]
    merchant_weights = rng.exponential(scale=1.0, size=len(merchant_ids))
    merchant_probs = merchant_weights / merchant_weights.sum()

    for acc_id in all_normal_legit:
        if acc_id in merchant_ids:
            continue
        c_day = created_map[acc_id]
        n_tx = int(rng.integers(2, 7))
        for _ in range(n_tx):
            m_id = str(rng.choice(merchant_ids, p=merchant_probs))
            tx_day = int(rng.integers(max(c_day, created_map[m_id]), 181))
            amount = float(np.round(rng.lognormal(mean=3.8, sigma=0.8), 2))
            transfer_records.append(
                {
                    "src": acc_id,
                    "dst": m_id,
                    "amount": max(5.0, amount),
                    "day": tx_day,
                }
            )

    # 3b. Legitimate peer-to-peer transfers
    p2p_candidates = np.array(all_normal_legit)
    for acc_id in all_normal_legit:
        if rng.random() < 0.65:
            n_p2p = int(rng.integers(1, 4))
            peers = rng.choice(p2p_candidates, size=n_p2p, replace=False)
            c_day = created_map[acc_id]
            for p in peers:
                if p == acc_id:
                    continue
                tx_day = int(rng.integers(max(c_day, created_map[p]), 181))
                amount = float(np.round(rng.lognormal(mean=4.2, sigma=1.0), 2))
                transfer_records.append(
                    {
                        "src": acc_id,
                        "dst": str(p),
                        "amount": max(10.0, amount),
                        "day": tx_day,
                    }
                )

    # 3c. Confounder activity: Family/roommate transfers
    # High edge density, but varied amounts (high CV) and no tight cycle amounts
    for f_group in family_groups:
        if len(f_group) < 2:
            continue
        for i in range(len(f_group)):
            for j in range(len(f_group)):
                if i != j:
                    src_id = f_group[i]
                    dst_id = f_group[j]
                    n_tx = int(rng.integers(2, 5))
                    for _ in range(n_tx):
                        min_day = max(created_map[src_id], created_map[dst_id])
                        tx_day = int(rng.integers(min_day, 181))
                        # Varied amounts: from small reimbursement to rent share
                        amount = float(np.round(rng.choice([18.5, 42.0, 115.0, 320.0, 750.0]), 2))
                        transfer_records.append(
                            {
                                "src": src_id,
                                "dst": dst_id,
                                "amount": amount,
                                "day": tx_day,
                            }
                        )

    # 3d. Fraud ring transfers
    # Core cycles (A->B->C->A) with similar amounts, dense internal edges, cashouts
    all_outside_sinks = merchant_ids + general_legit_ids[:50]

    for members in ring_account_groups:
        m_len = len(members)
        # Confounder: 1-2 peripheral accounts with only 1-2 edges
        n_peripheral = 1 if m_len <= 6 else 2
        core_members = members[:-n_peripheral]
        peripheral_members = members[-n_peripheral:]

        # Base amount for the ring cycle (near-equal amounts)
        base_amount = float(rng.choice([60.0, 120.0, 250.0, 500.0, 900.0]))
        cycle_day = int(rng.integers(160, 181))

        # Build primary directed 3-cycle or 4-cycle among core members
        cycle_len = min(len(core_members), 3 if rng.random() < 0.6 else 4)
        cycle_nodes = core_members[:cycle_len]
        for k in range(cycle_len):
            u = cycle_nodes[k]
            v = cycle_nodes[(k + 1) % cycle_len]
            # Near-identical amount (low CV)
            amt = float(np.round(base_amount * (1.0 + rng.uniform(-0.12, 0.12)), 2))
            transfer_records.append(
                {
                    "src": u,
                    "dst": v,
                    "amount": amt,
                    "day": cycle_day,
                }
            )

        # Build secondary cycle or reciprocal edges among remaining core members
        if len(core_members) >= 4:
            sub_len = min(len(core_members), 3)
            sub_nodes = core_members[-sub_len:]
            for k in range(sub_len):
                u = sub_nodes[k]
                v = sub_nodes[(k + 1) % sub_len]
                amt = float(np.round(base_amount * (1.0 + rng.uniform(-0.15, 0.15)), 2))
                transfer_records.append(
                    {
                        "src": u,
                        "dst": v,
                        "amount": amt,
                        "day": cycle_day,
                    }
                )

        # Dense internal transfers between core members
        for _ in range(len(core_members) * 2):
            u, v = rng.choice(core_members, size=2, replace=False)
            amt = float(np.round(base_amount * (1.0 + rng.uniform(-0.2, 0.2)), 2))
            transfer_records.append(
                {
                    "src": str(u),
                    "dst": str(v),
                    "amount": amt,
                    "day": int(rng.integers(155, 181)),
                }
            )

        # Peripheral members transact lightly with one core member
        for p_node in peripheral_members:
            core_partner = str(rng.choice(core_members))
            amt = float(np.round(base_amount * (1.0 + rng.uniform(-0.25, 0.25)), 2))
            transfer_records.append(
                {
                    "src": p_node,
                    "dst": core_partner,
                    "amount": amt,
                    "day": int(rng.integers(155, 181)),
                }
            )

        # Cash-out edges to outside legitimate accounts / merchants
        n_cashouts = int(rng.integers(1, 3))
        for _ in range(n_cashouts):
            cashout_src = str(rng.choice(core_members))
            outside_dst = str(rng.choice(all_outside_sinks))
            transfer_records.append(
                {
                    "src": cashout_src,
                    "dst": outside_dst,
                    "amount": float(np.round(base_amount * rng.uniform(0.8, 1.2), 2)),
                    "day": int(rng.integers(170, 181)),
                }
            )

    # Camouflage: ring members also make ordinary merchant purchases
    for members in ring_account_groups:
        for acc_id in members:
            for _ in range(int(rng.integers(1, 6))):
                m_id = str(rng.choice(merchant_ids, p=merchant_probs))
                transfer_records.append(
                    {
                        "src": acc_id,
                        "dst": m_id,
                        "amount": max(5.0, float(np.round(rng.lognormal(mean=3.8, sigma=0.8), 2))),
                        "day": int(rng.integers(max(created_map[acc_id], created_map[m_id]), 181)),
                    }
                )

    transfers_df = pd.DataFrame(transfer_records)

    return accounts_df, transfers_df
