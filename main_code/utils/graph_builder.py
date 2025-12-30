# /root/hhq/main_code/utils/graph_builder.py
import torch
import torch.nn.functional as F

def compute_similarity_matrix(global_feats, local_feats, tau=30, event_times=None, threshold=0.65):
    """
    严格复现论文公式 (3) 和 (4)
    Args:
        global_feats: (N, D) [CLS] tokens
        local_feats: (N, L, D) Patch tokens, L is number of patches
        tau: temporal distance threshold (seconds)
        event_times: list of (start_sec, end_sec)
        threshold: delta in Eq. 4
    """
    N, L, D = local_feats.shape
    device = global_feats.device
    
    # --- 1. Global Similarity (Cosine) ---
    # x_g normalized
    g_norm = F.normalize(global_feats, p=2, dim=-1)
    sim_global = torch.mm(g_norm, g_norm.t()) # (N, N)
    
    # --- 2. Fine-grained Similarity (Eq. 3) ---
    # Formula: s_ij = 0.5 * ( (1/L * sum_k max_m cos(p_ik, p_jm)) + cos(g_i, g_j) )
    # Calculation of Patch Sim: (N, L, D) x (N, L, D)^T -> Huge!
    # We compute it in chunks to save GPU memory.
    
    sim_local = torch.zeros((N, N), device=device)
    local_feats = F.normalize(local_feats, p=2, dim=-1) # Normalize first
    
    # 分块计算 (防止 OOM)
    chunk_size = 10  # 每次算 10 个 Event 对所有 Events 的相似度
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        # Query chunk: (B, L, D)
        q_chunk = local_feats[i:end_i] 
        
        # (B, L, D) @ (N, L, D).permute -> (B, N, L, L)
        # 注意: torch.einsum 此时可能显存也大，用 matmul 拆解
        # Let's verify each pair (u, v)
        # Optimized: calculate max over last dim
        for j in range(N):
            # q_chunk: (B, L, D), target: (1, L, D)
            # pairwise cos: (B, L, L)
            tgt = local_feats[j].unsqueeze(0) # (1, L, D)
            
            # (B, L, D) @ (D, L) -> (B, L, L)
            pair_sim = torch.matmul(q_chunk, tgt.transpose(1, 2))
            
            # max over m (target patches): (B, L)
            max_sim_values, _ = pair_sim.max(dim=2)
            
            # avg over k (source patches): (B,)
            avg_max_sim = max_sim_values.mean(dim=1)
            
            sim_local[i:end_i, j] = avg_max_sim

    # Combined Similarity
    s_ij = 0.5 * (sim_local + sim_global)
    
    # --- 3. Apply Constraints (Eq. 4) ---
    # Filter 1: Similarity > delta
    mask_sim = (s_ij > threshold).float()
    
    # Filter 2: Temporal Distance > tau
    mask_time = torch.ones((N, N), device=device)
    if event_times is not None:
        centers = [(s+e)/2 for s,e in event_times]
        # 向量化计算时间差
        t_tensor = torch.tensor(centers, device=device).unsqueeze(1)
        dist_matrix = torch.abs(t_tensor - t_tensor.t())
        mask_time = (dist_matrix > tau).float()
    
    # Diagonal is always 0 (no self-loops for semantic edges)
    mask_diag = 1 - torch.eye(N, device=device)
    
    # Final Semantic Adjacency
    adj_semantic = s_ij * mask_sim * mask_time * mask_diag
    
    return adj_semantic

def compute_pagerank_matrix(adj_semantic, alpha=0.15):
    """
    论文公式 (6): Reachability Matrix
    adj: Directed Weighted Graph (includes Temporal + Semantic)
    """
    N = adj_semantic.shape[0]
    device = adj_semantic.device
    
    # 构建完整图：Semantic + Temporal (i -> i+1)
    # Temporal edges have weight 1.0 (Eq. 2)
    adj_total = adj_semantic.clone()
    
    # Add Temporal Edges (i -> i+1)
    # 使用 torch.diag_embed 构建对角偏移矩阵会更快
    indices = torch.arange(N-1, device=device)
    adj_total[indices, indices+1] = 1.0 
    
    # Normalize rows (Transition Matrix P)
    # P = D^-1 A
    row_sum = torch.sum(adj_total, dim=1, keepdim=True)
    # Avoid division by zero (nodes with no outgoing edges stay put)
    row_sum[row_sum == 0] = 1.0
    P = adj_total / row_sum
    
    # Solve Closed Form: Pi = alpha * (I - (1-alpha)P)^-1
    eye = torch.eye(N, device=device)
    try:
        mat = eye - (1 - alpha) * P
        inv_mat = torch.inverse(mat)
    except:
        # Fallback for singular matrix (add epsilon)
        mat = eye - (1 - alpha) * P + 1e-6 * eye
        inv_mat = torch.inverse(mat)
        
    Pi = alpha * inv_mat
    return Pi