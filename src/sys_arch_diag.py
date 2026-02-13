from graphviz import Digraph


def make_node(dot, name, label, shape='box', style='filled', color='lightgrey', fontcolor='black'):
    dot.node(name, label=label, shape=shape, style=style,
             fillcolor=color, fontcolor=fontcolor, fontname='Arial', penwidth='0')


# Initialize Graph
# rankdir='LR': Left-to-Right flow
# splines='ortho': Right-angle lines
dot = Digraph('NeuralSearchPrime_Architecture', format='png')
dot.attr(dpi='300', rankdir='LR', splines='ortho',
         nodesep='0.6', ranksep='1.0')
dot.attr(bgcolor='white')

# --- CLUSTER 1: OFFLINE TRAINING PIPELINE ---
# "Rationale-Guided Distillation" [cite: 60, 62]
with dot.subgraph(name='cluster_0_offline') as c:
    c.attr(label='Offline Training & Distillation',
           style='dashed', color='#aaaaaa', fontname='Arial')

    make_node(c, 'ESCI', 'Amazon ESCI\nDataset',
              shape='cylinder', color='#FFF3E0')
    make_node(c, 'Teacher',
              'Teacher LLM\n(Llama-3)\nGenerates Rationales', color='#FFE0B2')
    make_node(c, 'Student', 'Student\nCross-Encoder\n(Distilled)', color='#FFCC80')
    make_node(c, 'Registry', 'Model\nRegistry',
              shape='folder', color='#FFB74D')

    # Internal Flow
    c.edge('ESCI', 'Teacher', penwidth='2', color='#aaaaaa',
           arrowhead='dot', arrowtail='none', dir='forward')
    c.edge('Teacher', 'Student', penwidth='2', color='#aaaaaa',
           arrowhead='dot', arrowtail='none', dir='forward')
    c.edge('Student', 'Registry', penwidth='2', color='#aaaaaa',
           arrowhead='dot', arrowtail='none', dir='forward')

# --- CLUSTER 2: CLIENT LAYER ---
# "External Interface" [cite: 67]
with dot.subgraph(name='cluster_1_client') as c:
    c.attr(label='Client Layer', style='invis')
    make_node(c, 'UI', 'Streamlit\nDashboard', shape='rect', color='#E1F5FE')
    make_node(c, 'Client', 'End User', shape='circle', color='#B3E5FC')

# --- CLUSTER 3: MICROSERVICE ORCHESTRATION ---
# "Kubernetes (EKS)" [cite: 35]
with dot.subgraph(name='cluster_2_runtime') as c:
    c.attr(label='Online Runtime (Kubernetes/EKS)',
           style='rounded', color='#0277BD', bgcolor='#F1F8E9')

    # API Gateway
    make_node(c, 'API', 'Search API\n(FastAPI)',
              shape='component', color='#C8E6C9')

    # --- SUB-CLUSTER: STAGE 1 RETRIEVAL ---
    # "OpenSearch 2.11+" [cite: 33]
    with dot.subgraph(name='cluster_3_retrieval') as r:
        r.attr(label='Stage 1: Hybrid Retrieval',
               style='solid', color='#2E7D32', bgcolor='white')

        make_node(r, 'Lexical', 'Lexical Search\n(BM25)', color='#A5D6A7')
        make_node(r, 'Semantic', 'Semantic Search\n(k-NN Vectors)',
                  color='#A5D6A7')
        make_node(r, 'RRF', 'Fusion Layer\n(RRF Algorithm)',
                  shape='diamond', color='#81C784')

    # --- SUB-CLUSTER: STAGE 2 RERANKING ---
    # "SageMaker / TEI" [cite: 34]
    with dot.subgraph(name='cluster_4_rerank') as rr:
        rr.attr(label='Stage 2: Reranking', style='solid',
                color='#1565C0', bgcolor='white')
        make_node(rr, 'Inference',
                  'Cross-Encoder\nInference Service\n(SageMaker/TEI)', color='#90CAF9')

# --- EDGES (CONNECTIONS) ---
# "Dot" style connectors requested by user
edge_style = {
    'arrowhead': 'dot',
    'arrowtail': 'none',
    'dir': 'forward',
    'penwidth': '2.0',
    'color': '#455A64',
    'minlen': '2'
}

# 1. User -> UI -> API
dot.edge('Client', 'UI', **edge_style)
dot.edge('UI', 'API', xlabel='JSON Query', **edge_style)

# 2. API -> Hybrid Retrieval (Split)
# Note: Using lhead/ltail requires compound=True, but simple node links work better for Ortho
dot.edge('API', 'Lexical', **edge_style)
dot.edge('API', 'Semantic', **edge_style)

# 3. Retrieval -> Fusion
dot.edge('Lexical', 'RRF', **edge_style)
dot.edge('Semantic', 'RRF', **edge_style)

# 4. Fusion -> Reranking (Top 50 Candidates)
dot.edge('RRF', 'Inference', xlabel='Top 50 Candidates', **edge_style)

# 5. Reranking -> API (Response)
dot.edge('Inference', 'API', xlabel='Sorted Results',
         style='dashed', **edge_style)

# 6. Deployment: Registry -> Inference (Model Load)
dot.edge('Registry', 'Inference', style='dotted',
         color='#FFB74D', label='Deploy', arrowhead='dot')

# Render
dot.render('neuralsearch_architecture', cleanup=True)
print("System Architecture diagram generated successfully.")
