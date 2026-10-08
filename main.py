"""Tiny embedding similarity search utility."""
import sys, math

def load_embeddings(path):
    d={}
    with open(path) as f:
        for line in f:
            parts=line.split()
            if not parts: continue
            id=parts[0]; vec=list(map(float,parts[1:]))
            d[id]=vec
    return d

def cosine(a,b):
    dot=sum(x*y for x,y in zip(a,b))
    norm_a=math.sqrt(sum(x*x for x in a))
    norm_b=math.sqrt(sum(y*y for y in b))
    return dot/(norm_a*norm_b) if norm_a and norm_b else 0.0

def search(data, target_vec, top=5):
    sims=[(id,cosine(vec,target_vec)) for id,vec in data.items()]
    sims.sort(key=lambda x:x[1], reverse=True)
    return sims[:top]

if __name__=="__main__":
    if len(sys.argv)<3:
        print("Usage: {} <embedding_file> <target_id> [top_n]".format(sys.argv[0]))
        sys.exit(1)
    file=sys.argv[1]
    target_id=sys.argv[2]
    top=int(sys.argv[3]) if len(sys.argv)>3 else 5
    data=load_embeddings(file)
    if target_id not in data:
        print("Target id not found")
        sys.exit(1)
    target=data[target_id]
    results=search(data,target,top)
    for id,sim in results:
        print(f"{id}\t{sim:.4f}")