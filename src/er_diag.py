from graphviz import Digraph


def make_entity(dot, name, attrs, bg_color="lightgrey", title_color="black"):
    rows = ""
    for a in attrs:
        port_name = a.split(":")[0].strip()
        # Bold PK, Italicize FK
        if a.startswith("PK"):
            fmt_a = f'<B>{a}</B>'
        elif a.startswith("FK"):
            fmt_a = f'<I>{a}</I>'
        else:
            fmt_a = a
        rows += f'<TR><TD ALIGN="LEFT" PORT="{port_name}">{fmt_a}</TD></TR>'

    label = f'''<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD BGCOLOR="{bg_color}"><B><FONT COLOR="{title_color}" POINT-SIZE="12">{name}</FONT></B></TD></TR>
        {rows}
      </TABLE>
    >'''
    dot.node(name, label=label, shape='none', fontname="Arial")


# 1. DPI=300 for crisp text
# 2. splines='ortho' for clean right-angle lines
# 3. nodesep/ranksep increased significantly to prevent overlapping lines
dot = Digraph('NeuralSearchPrime_ERD', format='png')
dot.attr(dpi='300')
dot.attr(rankdir='LR', splines='ortho')
dot.attr(nodesep='1.0', ranksep='1.5')
dot.attr('node', fontname='Arial', fontsize='10')
dot.attr('edge', fontname='Arial', fontsize='10')

# --- CLUSTER 1: CORE INVENTORY ---
with dot.subgraph(name='cluster_0_core') as c:
    c.attr(label='Core Inventory', style='dashed',
           color='grey', fontname='Arial')

    make_entity(c, 'Product', [
        'PK: product_id',
        'title',
        'brand',
        'description',
        'price',
        'embedding_vector_ref',
        'created_at'
    ], bg_color="#E1F5FE")

    make_entity(c, 'Query', [
        'PK: query_id',
        'query_text',
        'embedding_vector_ref',
        'created_at'
    ], bg_color="#E1F5FE")

# --- CLUSTER 2: TRAINING & FEEDBACK ---
with dot.subgraph(name='cluster_1_training') as c:
    c.attr(label='Training & Ground Truth',
           style='dashed', color='grey', fontname='Arial')

    make_entity(c, 'Query_Product_Relevance', [
        'PK: id',
        'FK: query_id',
        'FK: product_id',
        'esc_label',
        'numeric_score'
    ], bg_color="#FFF9C4")

    make_entity(c, 'Teacher_Annotation', [
        'PK: annotation_id',
        'FK: relevance_id',
        'teacher_score',
        'rationale_text',
        'reviewer_id'
    ], bg_color="#FFF9C4")

# --- CLUSTER 3: RUNTIME TELEMETRY ---
with dot.subgraph(name='cluster_2_runtime') as c:
    c.attr(label='Runtime Telemetry', style='dashed',
           color='grey', fontname='Arial')

    make_entity(c, 'Search_Request', [
        'PK: search_id',
        'FK: query_id',
        'top_k',
        'mode',
        'latency_ms',
        'timestamp'
    ], bg_color="#E8F5E9")

    make_entity(c, 'Retrieval_Result', [
        'PK: result_id',
        'FK: search_id',
        'FK: product_id',
        'bm25_score',
        'knn_score',
        'rrf_score',
        'initial_rank'
    ], bg_color="#E8F5E9")

    make_entity(c, 'Rerank_Result', [
        'PK: rerank_id',
        'FK: search_id',
        'FK: product_id',
        'cross_encoder_score',
        'final_rank',
        'rank_change'
    ], bg_color="#E8F5E9")

# --- RELATIONSHIPS (DOT STYLE) ---

# Styling:
# arrowhead='dot' : Ends the line with a filled circle
# arrowtail='none': Starts with nothing
# penwidth='1.5'  : Slightly thicker lines for visibility
# minlen='2'      : Forces nodes to be at least 2 "slots" apart, fixing cramped lines

edge_attr = {
    'arrowhead': 'dot',
    'arrowtail': 'none',
    'dir': 'forward',
    'penwidth': '1.5',
    'color': '#555555',
    'minlen': '2'
}

# Core -> Runtime
dot.edge('Query', 'Search_Request', xlabel='generates',
         tailport='e', headport='w', **edge_attr)
dot.edge('Search_Request', 'Retrieval_Result', xlabel='yields',
         tailport='e', headport='w', **edge_attr)

# Linking Product to Result
# We increase minlen here to ensure the line jumps over the gap cleanly
dot.edge('Product', 'Retrieval_Result', xlabel='referenced in',
         tailport='e', headport='w', **edge_attr)

# Pipeline Flow (Retrieval -> Rerank)
dot.edge('Retrieval_Result', 'Rerank_Result', xlabel='refined by',
         tailport='e', headport='w', **edge_attr)

dot.edge('Rerank_Result', 'Query_Product_Relevance', xlabel='Used for retrain \n when \n clicked on the \n relevant query',
         tailport='e', headport='w', **edge_attr)


# Training Logic
dot.edge('Query_Product_Relevance', 'Teacher_Annotation',
         xlabel='audited by', tailport='e', headport='w', **edge_attr)

dot.render('neuralsearch_prime_dots', cleanup=True)
print("Diagram generated with dot-style connectors.")
