NOM = "MENDES GOUVEIA DA SILVA"
PRENOM = "Ana Julia"

import random
import itertools

# Question 4 (c) (2 points) Evaluating a link predictor:
def evaluate_link_prediction(graph, predictor, f=0.1, k_values=[50,100,200,300,400]):
    
    # 1. copy graph and get edges
    G = graph.copy()
    edges = list(G.edges())
    
    # 2. remove a fraction f of edges
    num_remove = int(f * len(edges))
    removed_edges = random.sample(edges, num_remove)
    
    G.remove_edges_from(removed_edges)
    
    # Sorting them with min/max avoids the need to check both (A,B) and (B,A) later
    removed_set = set() 
    for u, v in removed_edges:
        removed_set.add((min(u, v), max(u, v)))

   # 3. generate all possible node pairs (not already connected)
    node_pairs = []
    
    # here, itertools is much faster than nested 'for i / for j' loops for big graphs
    for u, v in itertools.combinations(G.nodes(), 2):
        if not G.has_edge(u, v):
            node_pairs.append((u, v))
            
    model = predictor(G)
    scores = model.fit(node_pairs)
    
    # 4. sort by decreasing score
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    results = {}
    
    # 5. compute precision and recall for each k
    for k in k_values:
        top_k = scores_sorted[:k]
        predicted_edges = [pair for pair, score in top_k]
        
        tp = 0
        for u, v in predicted_edges:
            # check if the prediction is in our removed set
            if (min(u, v), max(u, v)) in removed_set:
                tp += 1
                
        # the formula TP + FP is literally just k (the number of predictions we made)
        # the formula TP + FN is just the total number of removed edges
        precision = tp / k
        recall = tp / len(removed_set)
        
        results[k] = {
            "top@k": precision, # top@k is the same as precision in this context
            "precision": precision,
            "recall": recall
        }
        
        print(f"k={k:3d} | TP: {tp:3d} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        
    return results
