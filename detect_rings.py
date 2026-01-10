import pandas as pd, networkx as nx
from collections import Counter

def main(edges_csv="data/edges.csv"):
    df=pd.read_csv(edges_csv)
    G=nx.Graph()
    for _,r in df.iterrows():
        G.add_edge(r["src"], r["dst"], amount=float(r["amount"]))

    # Heuristic 1: nodes with very high degree (hub-like)
    deg = dict(G.degree())
    top = sorted(deg.items(), key=lambda x:x[1], reverse=True)[:25]
    print("Top degree nodes (possible ring hubs):")
    for n,d in top:
        print(f"  {n}: degree={d}")

    # Heuristic 2: community detection (greedy modularity)
    comms = list(nx.algorithms.community.greedy_modularity_communities(G))
    comms = sorted(comms, key=len, reverse=True)[:10]
    print("\nTop communities by size:")
    for i,c in enumerate(comms,1):
        # score by average internal degree
        sub = G.subgraph(c)
        avg_deg = sum(dict(sub.degree()).values())/max(1,len(sub))
        print(f"  C{i}: size={len(c)}  avg_deg={avg_deg:.2f}")

    # Heuristic 3: bipartite density proxy (accounts vs merchants)
    # classify by prefix
    suspicious=[]
    for c in comms:
        a=[n for n in c if str(n).startswith("a")]
        m=[n for n in c if str(n).startswith("m")]
        if len(a)>=10 and len(m)>=3:
            # compute density between a and m
            possible=len(a)*len(m)
            actual=sum(1 for u,v in G.subgraph(c).edges() if (str(u).startswith("a") and str(v).startswith("m")) or (str(u).startswith("m") and str(v).startswith("a")))
            density=actual/possible if possible else 0
            suspicious.append((density,len(a),len(m),len(c)))
    suspicious=sorted(suspicious, reverse=True)
    print("\nSuspicious communities (high account-merchant density):")
    for d,na,nm,sz in suspicious[:10]:
        print(f"  density={d:.3f}  accounts={na}  merchants={nm}  total={sz}")

if __name__=="__main__":
    main()
