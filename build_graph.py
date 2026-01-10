import networkx as nx
import numpy as np, pandas as pd, os

def main(seed=1, n_accounts=800, n_merchants=120, n_edges=9000, out_csv="data/edges.csv"):
    rng=np.random.default_rng(seed)
    accounts=[f"a{i}" for i in range(n_accounts)]
    merchants=[f"m{i}" for i in range(n_merchants)]

    # normal transactions
    src = rng.choice(accounts, size=n_edges)
    dst = rng.choice(merchants, size=n_edges)
    amount = rng.lognormal(mean=4.0, sigma=0.6, size=n_edges).clip(1, 20000)

    # inject a small fraud ring: a subset of accounts transact repeatedly with a subset of merchants
    ring_accounts = rng.choice(accounts, size=20, replace=False)
    ring_merchants = rng.choice(merchants, size=6, replace=False)
    ring_edges = 1200
    src2 = rng.choice(ring_accounts, size=ring_edges)
    dst2 = rng.choice(ring_merchants, size=ring_edges)
    amount2 = rng.lognormal(mean=6.0, sigma=0.4, size=ring_edges).clip(50, 60000)

    src = np.concatenate([src, src2])
    dst = np.concatenate([dst, dst2])
    amount = np.concatenate([amount, amount2])

    df=pd.DataFrame({"src":src,"dst":dst,"amount":amount})
    os.makedirs("data", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} edges. Ring accounts={len(ring_accounts)} ring merchants={len(ring_merchants)}")

if __name__=="__main__":
    main()
