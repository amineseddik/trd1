import streamlit as st
import numpy as np
import pandas as pd

# Méthode du Nord-Ouest
def nord_ouest(D, O):
    n = len(D)
    m = len(O)
    x = [[0 for _ in range(n)] for _ in range(m)]
    D = D.copy()
    O = O.copy()
    j = 0
    for i in range(n):
        while D[i] != 0:
            if D[i] <= O[j]:
                x[j][i] = D[i]
                O[j] -= D[i]
                D[i] = 0
            else:
                x[j][i] = O[j]
                D[i] -= O[j]
                j += 1
    return x

# Fonction de calcul du coût total
def calcul_cout(P, x):
    cout = 0
    for i in range(len(P)):
        for j in range(len(P[0])):
            cout += x[i][j] * P[i][j]
    return cout

# Fonction pour trouver le minimum dans la matrice des coûts
def trouve_min(P):
    min_val = P[0][0]
    min_i = min_j = 0
    for i in range(len(P)):
        for j in range(len(P[0])):
            if P[i][j] < min_val:
                min_val = P[i][j]
                min_i = i
                min_j = j
    return min_i, min_j

# Méthode du moindre coût
def moindre_cout(P, D, O):
    n = len(D)
    m = len(O)
    x = [[0 for _ in range(n)] for _ in range(m)]
    P = [row[:] for row in P]
    D = D.copy()
    O = O.copy()
    fini = False
    while not fini:
        k, w = trouve_min(P)
        if D[w] <= O[k]:
            x[k][w] = D[w]
            O[k] -= D[w]
            D[w] = 0
            for j in range(m):
                P[j][w] = 100
        else:
            x[k][w] = O[k]
            D[w] -= O[k]
            O[k] = 0
            for j in range(n):
                P[k][j] = 100
        fini = True
        for i in range(n):
            if D[i] > 0:
                fini = False
                break
    return x

# Méthode de Vogel
def vogel_method(costs, sources, destinations):
    n, m = len(costs), len(costs[0])
    solution = []
    total_cost = 0
    remaining_sources = sources.copy()
    remaining_destinations = destinations.copy()
    available_rows = set(range(n))
    available_cols = set(range(m))
    
    while available_rows and available_cols:
        penalties_rows = []
        penalties_cols = []
        
        for i in available_rows:
            row_costs = [costs[i][j] for j in available_cols]
            if len(row_costs) >= 2:
                sorted_costs = sorted(row_costs)
                penalties_rows.append((sorted_costs[1] - sorted_costs[0], i))
            elif len(row_costs) == 1:
                penalties_rows.append((float('inf'), i))
                
        for j in available_cols:
            col_costs = [costs[i][j] for i in available_rows]
            if len(col_costs) >= 2:
                sorted_costs = sorted(col_costs)
                penalties_cols.append((sorted_costs[1] - sorted_costs[0], j))
            elif len(col_costs) == 1:
                penalties_cols.append((float('inf'), j))
        
        max_penalty_row = max(penalties_rows) if penalties_rows else (float('-inf'), -1)
        max_penalty_col = max(penalties_cols) if penalties_cols else (float('-inf'), -1)
        
        if max_penalty_row[0] >= max_penalty_col[0]:
            selected_row = max_penalty_row[1]
            min_cost = float('inf')
            selected_col = -1
            for j in available_cols:
                if costs[selected_row][j] < min_cost:
                    min_cost = costs[selected_row][j]
                    selected_col = j
        else:
            selected_col = max_penalty_col[1]
            min_cost = float('inf')
            selected_row = -1
            for i in available_rows:
                if costs[i][selected_col] < min_cost:
                    min_cost = costs[i][selected_col]
                    selected_row = i
        
        quantity = min(remaining_sources[selected_row], remaining_destinations[selected_col])
        remaining_sources[selected_row] -= quantity
        remaining_destinations[selected_col] -= quantity
        solution.append((selected_row, selected_col, quantity))
        total_cost += costs[selected_row][selected_col] * quantity
        
        if remaining_sources[selected_row] == 0:
            available_rows.remove(selected_row)
        if remaining_destinations[selected_col] == 0:
            available_cols.remove(selected_col)
    
    result_matrix = [[0 for _ in range(m)] for _ in range(n)]
    for i, j, q in solution:
        result_matrix[i][j] = q
    
    return result_matrix

# Fonctions pour la solution optimale
def calculate_dual_variables(cost_matrix, solution_matrix):
    m, n = len(cost_matrix), len(cost_matrix[0])
    u = [None] * m
    v = [None] * n
    u[0] = 0
    basic_vars = [(i, j) for i in range(m) for j in range(n) if solution_matrix[i][j] > 0]
    
    while None in u or None in v:
        changed = False
        for i, j in basic_vars:
            if u[i] is not None and v[j] is None:
                v[j] = cost_matrix[i][j] - u[i]
                changed = True
            elif u[i] is None and v[j] is not None:
                u[i] = cost_matrix[i][j] - v[j]
                changed = True
        
        if not changed:
            if None in u:
                row_counts = [sum(1 for j in range(n) if solution_matrix[i][j] > 0) for i in range(m)]
                max_count = max((count for i, count in enumerate(row_counts) if u[i] is None), default=0)
                if max_count > 0:
                    i = next(i for i in range(m) if u[i] is None and row_counts[i] == max_count)
                    u[i] = 0
            elif None in v:
                j = v.index(None)
                for i in range(m):
                    if solution_matrix[i][j] > 0 and u[i] is not None:
                        v[j] = cost_matrix[i][j] - u[i]
                        break
    return u, v

def find_cycle(solution_matrix, entering_i, entering_j):
    if entering_i == -1 or entering_j == -1:
        return None
        
    m, n = len(solution_matrix), len(solution_matrix[0])
    basic_positions = set((i, j) for i in range(m) for j in range(n) 
                     if solution_matrix[i][j] > 0)
    
    def get_next_positions(current_pos, visited):
        i, j = current_pos
        candidates = []
        row_positions = [(i, y) for y in range(n)]
        col_positions = [(x, j) for x in range(m)]
        
        for pos in row_positions + col_positions:
            if pos != current_pos and pos not in visited:
                if pos in basic_positions or pos == (entering_i, entering_j):
                    if len(visited) >= 2:
                        prev_pos = visited[-1]
                        prev_prev_pos = visited[-2]
                        if pos[0] == prev_pos[0] == prev_prev_pos[0]:
                            continue
                        if pos[1] == prev_pos[1] == prev_prev_pos[1]:
                            continue
                    candidates.append(pos)
        return candidates
    
    def find_path(path):
        if len(path) > 3:
            if (path[0][0] == path[-1][0] or path[0][1] == path[-1][1]):
                return path
                
        current_pos = path[-1]
        for next_pos in get_next_positions(current_pos, path):
            new_path = path + [next_pos]
            result = find_path(new_path)
            if result is not None:
                return result
        return None
    
    initial_path = [(entering_i, entering_j)]
    return find_path(initial_path)

def optimize_solution(cost_matrix, demands, supplies, initial_solution):
    solution = [row[:] for row in initial_solution]
    total_iterations = 0
    
    while True:
        u, v = calculate_dual_variables(cost_matrix, solution)
        
        marginal_costs = [[cost_matrix[i][j] - u[i] - v[j] 
                          for j in range(len(cost_matrix[0]))]
                         for i in range(len(cost_matrix))]
        
        min_cost = 0
        entering_i = entering_j = -1
        
        for i in range(len(cost_matrix)):
            for j in range(len(cost_matrix[0])):
                if solution[i][j] == 0 and marginal_costs[i][j] < min_cost:
                    min_cost = marginal_costs[i][j]
                    entering_i, entering_j = i, j
        
        if min_cost >= 0:
            break
            
        cycle = find_cycle(solution, entering_i, entering_j)
        if not cycle:
            break
            
        delta = float('inf')
        for idx in range(len(cycle)):
            if idx % 2 == 1:
                i, j = cycle[idx]
                if solution[i][j] > 0:
                    delta = min(delta, solution[i][j])
        
        for idx, (i, j) in enumerate(cycle):
            if idx % 2 == 0:
                solution[i][j] += delta
            else:
                solution[i][j] -= delta
        
        total_iterations += 1
        if total_iterations >= 100:
            break
    
    total_cost = sum(solution[i][j] * cost_matrix[i][j] 
                    for i in range(len(cost_matrix)) 
                    for j in range(len(cost_matrix[0])))
    
    return solution, total_cost, total_iterations

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Problème de Transport - Heuristiques",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main {
        background-color: #f5f5dc;
    }
    .stTitle, .stHeader {
        color: #000000 !important;
    }
    .stButton button {
        background-color: #aaaadc;
        color: white;
    }
    .matrix-input {
        background-color: #fff8dc;
    }
    h1 {
        color: red;
    }
    h2 {
        color: green;
    }
    h3 {
        color: blue;
    }
    p {
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre principal
st.title("🚛 Résolution du Problème de Transport")

# Création des colonnes pour l'interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Configuration du problème")
    
    # Dimensions du problème
    n_sources = st.number_input("Nombre de sources", min_value=1, max_value=10, value=2)
    n_destinations = st.number_input("Nombre de destinations", min_value=1, max_value=10, value=3)
    
    # Saisie des offres
    st.subheader("Offres (Sources)")
    offres = []
    for i in range(n_sources):
        offre = st.number_input(f"Offre source {i+1}", min_value=0, value=100)
        offres.append(offre)
    
    # Saisie des demandes
    st.subheader("Demandes (Destinations)")
    demandes = []
    for i in range(n_destinations):
        demande = st.number_input(f"Demande destination {i+1}", min_value=0, value=80)
        demandes.append(demande)

with col2:
    st.subheader("Matrice des coûts")
    couts = []
    for i in range(n_sources):
        row = []
        cols = st.columns(n_destinations)
        for j in range(n_destinations):
            with cols[j]:
                val = st.number_input(f"Coût {i+1},{j+1}", value=3)
                row.append(val)
        couts.append(row)

# Sélection de la méthode
method = st.selectbox(
    "Choisir la méthode de résolution",
    ["Méthode de Vogel", "Méthode du Nord-Ouest", "Méthode du Moindre Coût", "Solution Optimale"]
)

# Bouton de calcul
if st.button("Résoudre"):
    # Vérification de l'équilibre offre-demande
    if sum(offres) != sum(demandes):
        st.error("Attention : Le problème n'est pas équilibré (somme des offres ≠ somme des demandes)")
    
    # Résolution selon la méthode choisie
    if method == "Solution Optimale":
        initial_solution = vogel_method(couts, offres, demandes)
        solution, cout_total, iterations = optimize_solution(couts, demandes, offres, initial_solution)
    elif method == "Méthode de Vogel":
        solution = vogel_method(couts, offres, demandes)
        cout_total = calcul_cout(couts, solution)
    elif method == "Méthode du Nord-Ouest":
        solution = nord_ouest(demandes, offres)
        cout_total = calcul_cout(couts, solution)
    else:  # Méthode du Moindre Coût
        solution = moindre_cout(couts, demandes, offres)
        cout_total = calcul_cout(couts, solution)
    
    # Affichage des résultats
    st.subheader("Solution")
    df_solution = pd.DataFrame(
        solution,
        index=[f"Source {i+1}" for i in range(n_sources)],
        columns=[f"Dest {i+1}" for i in range(n_destinations)]
    )
    st.dataframe(df_solution)
    st.success(f"Coût total de la solution : {cout_total}")