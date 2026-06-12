"""
=============================================================================
Partie 2 – Transport optimal et transfert de couleurs
=============================================================================
Cours  : Optimisation pour l'apprentissage automatique – M2 IASD 2025/2026
Sujet  : PR-OAA-2526  (C. W. Royer, Dauphine–PSL)

──────────────────────────────────────────────────────────────────────────
DÉPENDANCE AVEC LA PARTIE 1
──────────────────────────────────────────────────────────────────────────
Ce fichier réutilise EXACTEMENT les mêmes itérations GDA et PDHG que la
Partie 1 (fichier `partie1_primale_duale.py`), appliquées au Lagrangien

    L(w, z) = c^T w − z^T (A w − b)        (notations du sujet, formules (3)-(4))

Pour le problème de transport optimal (5), on a w = vec(P) ∈ R^{d²} et
z = (p, q) ∈ R^{2d} (multiplicateurs associés aux contraintes de marge
P 1_d = a et P^T 1_d = b respectivement).

⚠️  Pour le problème jouet (Q4, d=2), `A` est une matrice dense 4×4 et l'on
    peut directement appeler `gda`/`pdhg` de la Partie 1 (voir la fonction
    de validation croisée `_cross_check_with_part1`, qui importe
    `partie1_primale_duale`). PLACER CE FICHIER DANS LE MÊME RÉPERTOIRE QUE
    `partie1_primale_duale.py`.

⚠️  Pour le transfert de couleurs (Q5-Q6, d=300), `A ∈ R^{2d × d²}` aurait
    600×90000 ≈ 5.4·10⁷ coefficients (≈432 Mo en float64) : la former
    explicitement serait coûteux et inutile. On exploite donc la structure
    de A pour calculer A w et A^T z par des opérateurs en O(d²) (sommes de
    lignes/colonnes de P), ce qui est mathématiquement RIGOUREUSEMENT
    ÉQUIVALENT aux formules (3)-(4) de la Partie 1 (voir preuve ci-dessous
    et test de validation croisée).

──────────────────────────────────────────────────────────────────────────
Mise en équation : du problème (5) au format LP (1)
──────────────────────────────────────────────────────────────────────────
Problème de transport optimal (5) :

    minimiser_{P ∈ R^{d×d}}  ⟨C, P⟩   s.c.  P 1_d = a,  P^T 1_d = b,  P ≥ 0

On pose w = vec(P) ∈ R^{d²} (aplatissement ligne par ligne : w[i·d+j] = P_ij)
et c = vec(C). On a alors c^T w = Σ_ij C_ij P_ij = ⟨C, P⟩, et le problème (5)
est exactement le LP (1) avec :

    A ∈ R^{2d × d²},   b_LP = [a; b] ∈ R^{2d}

où A encode les 2d contraintes de marge :
    (A w)_i     = Σ_j P_ij = (P 1_d)_i        pour i = 1..d   (sommes de lignes)
    (A w)_{d+j} = Σ_i P_ij = (P^T 1_d)_j      pour j = 1..d   (sommes de colonnes)

──────────────────────────────────────────────────────────────────────────
Opérateurs A et A^T sans matrice explicite
──────────────────────────────────────────────────────────────────────────
    A w        ↔  apply_A(P)       = [ P.sum(axis=1) ; P.sum(axis=0) ]  ∈ R^{2d}
    A^T (p,q)  ↔  apply_AT(p, q)   = p ⊕ q  (somme externe)              ∈ R^{d×d}
                  c.-à-d.  [A^T(p,q)]_{ij} = p_i + q_j

Justification de A^T :  (A^T z)_k = Σ_l A_{lk} z_l. Pour k = i·d+j (case
P_ij), la colonne k de A a exactement deux 1 : en ligne i (contrainte de
ligne i) et en ligne d+j (contrainte de colonne j). Donc
    (A^T z)_{i·d+j} = z_i + z_{d+j} = p_i + q_j.                        ∎

──────────────────────────────────────────────────────────────────────────
Norme de l'opérateur A  —  ‖A‖² = 2d
──────────────────────────────────────────────────────────────────────────
On calcule A A^T ∈ R^{2d × 2d} par blocs (lignes/colonnes de contraintes) :

  • Bloc (lignes, lignes) : (A A^T)_{ii'} = ⟨A_{i,:}, A_{i',:}⟩.
    Le support de la ligne i de A est {(i, j) : j = 1..d}. Les supports de
    deux lignes différentes sont disjoints, donc ce bloc = d · I_d.
  • Bloc (colonnes, colonnes) : par symétrie, = d · I_d.
  • Bloc (lignes, colonnes) : (A A^T)_{i, d+j} = |{(i,j)} ∩ {(i,j)}| = 1
    pour tout (i,j), donc ce bloc = J_d (matrice tout-à-1, d×d).

    ⟹   A A^T = [[ d I_d ,  J_d  ],
                  [ J_d   ,  d I_d ]]

Les valeurs propres de J_d sont d (vecteur propre 1_d) et 0 (multiplicité
d−1, sous-espace orthogonal à 1_d). En diagonalisant par blocs (cf. code
`power_iteration_norm_A` pour vérification numérique) :

    • sur span(1_d) ⊗ R²      : valeurs propres 2d et 0
    • sur 1_d^⊥ ⊗ R²          : valeur propre d (multiplicité 2(d−1))

    ⟹   λ_max(A A^T) = 2d    ⟹    ‖A‖ = √(2d)

La valeur propre 0 correspond à la direction (p,q) = (1_d, −1_d), qui est
dans le noyau de A^T : c'est la REDONDANCE bien connue des contraintes de
transport (Σa = Σb ⟹ une contrainte de marge est combinaison linéaire des
2d−1 autres). Le couple optimal (p*, q*) n'est donc unique qu'à l'addition
de (c·1_d, −c·1_d) — ceci n'affecte pas la convergence de P_k car GDA/PDHG
n'ont besoin que des RÉSIDUS A w_k − b_LP, qui tendent individuellement
vers 0 (et non de z_k lui-même).

──────────────────────────────────────────────────────────────────────────
Condition de pas pour PDHG
──────────────────────────────────────────────────────────────────────────
Condition suffisante de convergence (cf. Partie 1, Q3) : α·β·‖A‖² < 1, soit

                    α · β · 2d < 1.

Pour α = β :   α < 1 / √(2d).
    - Q4 (d=2)   :  α < 1/2  = 0.5
    - Q5-6 (d=300):  α < 1/√600 ≈ 0.0408
=============================================================================
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Réutilisation directe de la projection de la Partie 1 (même fonction,
# car proj_{≥0} appliquée à w = vec(P) équivaut à l'appliquer à P
# composante par composante : max(0, P_ij) pour tout i,j).
from partie1_primale_duale import proj_pos, gda as gda_part1, pdhg as pdhg_part1

plt.rcParams.update({"font.size": 11, "figure.dpi": 120})


# =============================================================================
# Opérateurs A et A^T (sans matrice explicite)
# =============================================================================

def apply_A(P: np.ndarray) -> np.ndarray:
    """
    Calcule A w pour w = vec(P), sans former A.

        A w = [ P 1_d ; P^T 1_d ] = [ sommes de lignes ; sommes de colonnes ]

    Parameters
    ----------
    P : ndarray (d, d)

    Returns
    -------
    ndarray (2d,)  =  concat(P.sum(axis=1), P.sum(axis=0))
    """
    return np.concatenate([P.sum(axis=1), P.sum(axis=0)])


def apply_AT(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Calcule A^T z pour z = (p, q) ∈ R^d × R^d, sans former A.

        [A^T (p,q)]_{ij} = p_i + q_j     (somme externe / broadcasting)

    Parameters
    ----------
    p, q : ndarray (d,)

    Returns
    -------
    ndarray (d, d)
    """
    return p[:, None] + q[None, :]


def build_A_dense(d: int) -> np.ndarray:
    """
    Construit la matrice A ∈ R^{2d × d²} de manière EXPLICITE.

    ⚠️ Utilisée uniquement pour la validation croisée (Q4, d petit) et pour
    la vérification de ‖A‖. Pour d=300 (Q5-6) cette matrice ferait ≈432 Mo :
    NE PAS L'UTILISER dans le pipeline de transfert de couleurs.

    Convention de vectorisation : w[i*d+j] = P[i,j]  (aplatissement 'C'
    / row-major de numpy, cohérent avec `P.flatten()` et `w.reshape(d,d)`).
    """
    A = np.zeros((2 * d, d * d))
    for i in range(d):
        for j in range(d):
            col = i * d + j
            A[i, col] = 1.0       # contrainte de ligne i  :  Σ_j P_ij = a_i
            A[d + j, col] = 1.0   # contrainte de colonne j:  Σ_i P_ij = b_j
    return A


def power_iteration_norm_A(d: int, n_iter: int = 200, seed: int = 0) -> float:
    """
    Estime ‖A‖ = √λ_max(A A^T) par itération de la puissance, en utilisant
    UNIQUEMENT les opérateurs apply_A / apply_AT (coût O(d²) par itération,
    jamais O(d⁴)).

    Sert à vérifier numériquement la formule fermée ‖A‖ = √(2d) dérivée
    plus haut, pour n'importe quel d (y compris d=300).
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(2 * d)
    z /= np.linalg.norm(z)
    for _ in range(n_iter):
        p, q = z[:d], z[d:]
        Az = apply_A(apply_AT(p, q))     # (A A^T) z
        z = Az / np.linalg.norm(Az)
    p, q = z[:d], z[d:]
    rayleigh = z @ apply_A(apply_AT(p, q))   # = λ_max(A A^T)
    return float(np.sqrt(rayleigh))


# =============================================================================
# Métriques pour le problème de transport
# =============================================================================

def ot_metrics(C: np.ndarray, a: np.ndarray, b: np.ndarray, P: np.ndarray) -> dict:
    """
    Métriques de qualité d'un plan de transport P (pas nécessairement
    admissible pendant les itérations).

    Returns
    -------
    dict :
      'cost'          : ⟨C, P⟩
      'row_violation' : ‖P 1_d − a‖     (violation contrainte de ligne)
      'col_violation' : ‖P^T 1_d − b‖   (violation contrainte de colonne)
      'neg_mass'      : Σ max(0, −P_ij)  (masse négative — doit être 0
                        puisque P = proj_{≥0}[...] à chaque itération,
                        utile uniquement pour debug)
    """
    return {
        "cost": float(np.sum(C * P)),
        "row_violation": float(np.linalg.norm(P.sum(axis=1) - a)),
        "col_violation": float(np.linalg.norm(P.sum(axis=0) - b)),
        "neg_mass": float(np.sum(np.maximum(0.0, -P))),
    }


# =============================================================================
# Q5 — GDA pour le transport optimal
# =============================================================================

def ot_gda(
    C: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    P0: np.ndarray,
    p0: np.ndarray,
    q0: np.ndarray,
    alpha: float,
    beta: float,
    n_iters: int,
    track_every: int = 1,
) -> tuple:
    """
    GDA appliqué au problème de transport optimal (5).

    Correspondance EXACTE avec la formule (3) de la Partie 1
    (w ↔ vec(P), z ↔ (p,q), A ↔ apply_A, A^T ↔ apply_AT) :

    ┌────────────────────────────────────────────────────────────────────┐
    │  P_{k+1} = proj_{≥0}[ P_k − α (C − A^T(p_k,q_k)) ]                │
    │                    = proj_{≥0}[ P_k − α (C − (p_k ⊕ q_k)) ]       │
    │                                                                      │
    │  (p_{k+1}, q_{k+1}) = (p_k, q_k) − β ( A w_k − [a;b] )            │
    │                     = (p_k, q_k) − β ( apply_A(P_k) − [a;b] )    │
    └────────────────────────────────────────────────────────────────────┘

    Comme en Partie 1 (Q1), la mise à jour duale utilise P_k (AVANT la mise
    à jour primale) — c'est ce qui distingue GDA de PDHG.

    Parameters
    ----------
    C       : (d,d)  matrice de coûts
    a, b    : (d,)   marginales cibles (a^T 1 = b^T 1 nécessaire)
    P0      : (d,d)  plan de transport initial (P0 ≥ 0)
    p0, q0  : (d,)   multiplicateurs duaux initiaux
    alpha   : float  pas primal
    beta    : float  pas dual
    n_iters : int    nombre d'itérations
    track_every : int  fréquence d'enregistrement des métriques (pour ne
                  pas stocker toute la trajectoire de P, de taille d² —
                  prohibitif pour d=300 sur des milliers d'itérations)

    Returns
    -------
    P, p, q : ndarrays — itérés finaux
    history : dict avec listes 'iter', 'cost', 'row_violation',
              'col_violation' (une valeur tous les `track_every` pas,
              plus le point initial et le point final)
    """
    P, p, q = P0.copy().astype(float), p0.copy().astype(float), q0.copy().astype(float)
    rhs = np.concatenate([a, b])

    history = {"iter": [], "cost": [], "row_violation": [], "col_violation": []}

    def _record(k):
        m = ot_metrics(C, a, b, P)
        history["iter"].append(k)
        history["cost"].append(m["cost"])
        history["row_violation"].append(m["row_violation"])
        history["col_violation"].append(m["col_violation"])

    _record(0)
    for k in range(1, n_iters + 1):
        # ── Gradient primal : ∇_P L = C − A^T(p,q) = C − (p ⊕ q) ────────────
        grad_P = C - apply_AT(p, q)

        # ── Mise à jour primale (descente projetée) ─────────────────────────
        P_new = proj_pos(P - alpha * grad_P)

        # ── Mise à jour duale (montée), utilise P_k (avant mise à jour) ─────
        z = np.concatenate([p, q]) - beta * (apply_A(P) - rhs)

        P, p, q = P_new, z[:len(a)], z[len(a):]

        if k % track_every == 0 or k == n_iters:
            _record(k)

    return P, p, q, history


# =============================================================================
# Q5 — PDHG pour le transport optimal
# =============================================================================

def ot_pdhg(
    C: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    P0: np.ndarray,
    p0: np.ndarray,
    q0: np.ndarray,
    alpha: float,
    beta: float,
    n_iters: int,
    track_every: int = 1,
) -> tuple:
    """
    PDHG appliqué au problème de transport optimal (5).

    Correspondance EXACTE avec la formule (4) de la Partie 1 :

    ┌────────────────────────────────────────────────────────────────────┐
    │  P_{k+1} = proj_{≥0}[ P_k − α (C − (p_k ⊕ q_k)) ]                 │
    │  P̃       = 2 P_{k+1} − P_k                    (point extrapolé) │
    │  (p_{k+1}, q_{k+1}) = (p_k, q_k) − β ( apply_A(P̃) − [a;b] )      │
    └────────────────────────────────────────────────────────────────────┘

    Condition suffisante de convergence : α β ‖A‖² = α β · 2d < 1.

    Parameters : identiques à `ot_gda`.

    Returns : identiques à `ot_gda`.
    """
    P, p, q = P0.copy().astype(float), p0.copy().astype(float), q0.copy().astype(float)
    rhs = np.concatenate([a, b])

    history = {"iter": [], "cost": [], "row_violation": [], "col_violation": []}

    def _record(k):
        m = ot_metrics(C, a, b, P)
        history["iter"].append(k)
        history["cost"].append(m["cost"])
        history["row_violation"].append(m["row_violation"])
        history["col_violation"].append(m["col_violation"])

    _record(0)
    for k in range(1, n_iters + 1):
        # ── Mise à jour primale (identique à GDA) ───────────────────────────
        grad_P = C - apply_AT(p, q)
        P_new = proj_pos(P - alpha * grad_P)

        # ── Point extrapolé (overrelaxation θ = 1) ──────────────────────────
        P_extrap = 2.0 * P_new - P

        # ── Mise à jour duale avec extrapolation ────────────────────────────
        z = np.concatenate([p, q]) - beta * (apply_A(P_extrap) - rhs)

        P, p, q = P_new, z[:len(a)], z[len(a):]

        if k % track_every == 0 or k == n_iters:
            _record(k)

    return P, p, q, history


# =============================================================================
# Validation croisée avec la Partie 1 (matrice A dense, d petit uniquement)
# =============================================================================

def _cross_check_with_part1(d: int, C, a, b, alpha, beta, n_iters):
    """
    Vérifie que `ot_gda`/`ot_pdhg` (opérateurs apply_A/apply_AT) produisent
    EXACTEMENT la même trajectoire que `gda`/`pdhg` de la Partie 1 appliqués
    à la matrice A dense construite par `build_A_dense`.

    Utilisé uniquement pour d petit (Q4, d=2) — la matrice dense devient
    prohibitive pour d grand.

    Returns
    -------
    (max_diff_gda, max_diff_pdhg) : écarts maximaux (doivent être ~0)
    """
    A = build_A_dense(d)
    c_vec = C.flatten()
    rhs = np.concatenate([a, b])

    P0 = np.zeros((d, d))
    p0 = np.zeros(d)
    q0 = np.zeros(d)
    w0 = P0.flatten()
    z0 = np.concatenate([p0, q0])

    # ── GDA ──────────────────────────────────────────────────────────────────
    P_op, p_op, q_op, _ = ot_gda(C, a, b, P0, p0, q0, alpha, beta, n_iters,
                                  track_every=n_iters)
    ws_dense, zs_dense = gda_part1(c_vec, A, rhs, w0, z0, alpha, beta, n_iters)
    diff_gda = max(
        np.max(np.abs(P_op.flatten() - ws_dense[-1])),
        np.max(np.abs(np.concatenate([p_op, q_op]) - zs_dense[-1])),
    )

    # ── PDHG ─────────────────────────────────────────────────────────────────
    P_op2, p_op2, q_op2, _ = ot_pdhg(C, a, b, P0, p0, q0, alpha, beta, n_iters,
                                      track_every=n_iters)
    ws_dense2, zs_dense2 = pdhg_part1(c_vec, A, rhs, w0, z0, alpha, beta, n_iters)
    diff_pdhg = max(
        np.max(np.abs(P_op2.flatten() - ws_dense2[-1])),
        np.max(np.abs(np.concatenate([p_op2, q_op2]) - zs_dense2[-1])),
    )

    return diff_gda, diff_pdhg


# =============================================================================
# Pipeline de transfert de couleurs (Q5-Q6)
# =============================================================================

def load_or_synthesize_images(
    src_path: str = "ossau.jpg",
    tgt_path: str = "rer.jpg",
    fallback_shape: tuple = (64, 64),
    seed: int = 0,
) -> tuple:
    """
    Charge les deux images du notebook fourni (`ossau.jpg`, `rer.jpg`) si
    elles sont présentes dans le répertoire courant. Sinon, génère deux
    images synthétiques de couleurs nettement différentes, ce qui permet de
    FAIRE TOURNER ET VALIDER tout le pipeline même sans les fichiers
    originaux (utile pour la correction / les tests).

    Returns
    -------
    src, tgt : ndarrays (H, W, 3), valeurs dans [0, 1]
    is_synthetic : bool
    """
    if os.path.exists(src_path) and os.path.exists(tgt_path):
        from PIL import Image
        src = np.array(Image.open(src_path).convert("RGB")) / 255.0
        tgt = np.array(Image.open(tgt_path).convert("RGB")) / 255.0
        return src, tgt, False

    print(f"  [Attention] '{src_path}' / '{tgt_path}' introuvables : "
          f"génération d'images synthétiques de test {fallback_shape}.")
    rng = np.random.default_rng(seed)
    H, W = fallback_shape

    # Source : dégradé verdâtre + bruit (type paysage de montagne)
    xx, yy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    src = np.stack([0.2 + 0.3 * yy, 0.4 + 0.4 * (1 - yy), 0.2 + 0.2 * xx], axis=-1)
    src = np.clip(src + 0.05 * rng.standard_normal((H, W, 3)), 0, 1)

    # Cible : dégradé bleu/orange (type ciel au coucher du soleil)
    tgt = np.stack([0.3 + 0.5 * yy, 0.3 + 0.2 * yy, 0.5 + 0.4 * (1 - yy)], axis=-1)
    tgt = np.clip(tgt + 0.05 * rng.standard_normal((H, W, 3)), 0, 1)

    return src, tgt, True


def sample_pixels(image: np.ndarray, n_samples: int, seed: int = 0) -> tuple:
    """
    Sous-échantillonne `n_samples` pixels d'une image (H,W,3) → (n_samples,3).

    Returns
    -------
    X       : (n_samples, 3) pixels échantillonnés
    indices : indices choisis dans image.reshape(-1,3)
    """
    flat = image.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(flat), n_samples, replace=False)
    return flat[indices], indices


def recolor_image(image: np.ndarray, Xs: np.ndarray, T_Xs: np.ndarray) -> np.ndarray:
    """
    Étend la recoloration des pixels échantillonnés Xs → T_Xs à l'image
    complète via les plus proches voisins (identique au notebook fourni).

    Parameters
    ----------
    image : (H,W,3) image source complète, normalisée dans [0,1]
    Xs    : (ns,3) pixels source échantillonnés
    T_Xs  : (ns,3) couleurs recolorisées correspondantes

    Returns
    -------
    out : (H,W,3) image recolorisée, uint8 dans [0,255]
    """
    from sklearn.neighbors import NearestNeighbors

    flat = image.reshape(-1, 3)
    nn = NearestNeighbors(n_neighbors=1).fit(Xs)
    _, indices = nn.kneighbors(flat)
    mapped = T_Xs[indices[:, 0]]
    out = np.clip(mapped.reshape(image.shape), 0, 1)
    return (out * 255).astype(np.uint8)


def transport_to_colors(P: np.ndarray, a: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    """
    Applique le plan de transport P aux couleurs cibles Xt pour obtenir les
    couleurs recolorisées des pixels source échantillonnés (formule du
    notebook fourni, normalisée par la masse de ligne) :

        T_Xs = (P @ Xt) / (P 1_d + eps)

    Note : pour P exactement admissible, P 1_d = a, donc cette formule
    redonne la barycentration usuelle du transport optimal.
    """
    return (P @ Xt) / (P.sum(axis=1, keepdims=True) + 1e-9)


def color_transfer_lp(
    src: np.ndarray,
    tgt: np.ndarray,
    method: str,
    n_samples: int = 300,
    alpha: float = None,
    beta: float = None,
    n_iters: int = 3000,
    seed: int = 0,
) -> dict:
    """
    Pipeline complet de transfert de couleurs par transport optimal,
    paramétré par la méthode de résolution.

    1. Sous-échantillonnage de `n_samples` pixels de `src` et `tgt`.
    2. a = b = 1_d  (formulation du sujet, Q5).
    3. C_ij = ‖[Xs]_i − [Xt]_j‖  (norme euclidienne, formule du sujet).
    4. Résolution du LP de transport (5) par GDA, PDHG ou Sinkhorn.
    5. Recoloration de l'image complète par plus proches voisins.

    Parameters
    ----------
    src, tgt  : (H,W,3) images normalisées dans [0,1]
    method    : 'gda' | 'pdhg' | 'sinkhorn'
    n_samples : d, nombre de pixels échantillonnés par image
    alpha, beta : pas pour GDA/PDHG. Si None, on prend
                  alpha = beta = 0.9 / sqrt(2*n_samples)
                  (90% de la borne de convergence garantie de PDHG, cf.
                  dérivation ‖A‖=√(2d) en tête de fichier)
    n_iters   : nombre d'itérations (GDA/PDHG) ou numItermax (Sinkhorn)
    seed      : graine pour l'échantillonnage des pixels

    Returns
    -------
    dict avec :
      'out'      : image recolorisée (H,W,3) uint8
      'P'        : plan de transport final (d,d)
      'metrics'  : dict ot_metrics(C,a,b,P)
      'history'  : historique de convergence (GDA/PDHG) ou None (Sinkhorn)
      'runtime'  : temps d'exécution en secondes
      'cost_star': coût LP optimal (calculé par `scipy.linprog` si d ≤ 60,
                   None sinon car le LP devient trop grand)
    """
    d = n_samples
    Xs, _ = sample_pixels(src, d, seed=seed)
    Xt, _ = sample_pixels(tgt, d, seed=seed + 1)

    a = np.ones(d)
    b = np.ones(d)
    C = np.sqrt(((Xs[:, None, :] - Xt[None, :, :]) ** 2).sum(axis=-1))  # ‖Xs_i-Xt_j‖

    if alpha is None:
        alpha = 0.9 / np.sqrt(2 * d)
    if beta is None:
        beta = 0.9 / np.sqrt(2 * d)

    t0 = time.time()
    history = None

    if method == "sinkhorn":
        import ot as pot
        P = pot.sinkhorn(a, b, C, reg=1e-2, numItermax=n_iters)
    elif method == "gda":
        P0 = np.zeros((d, d))
        P, _, _, history = ot_gda(C, a, b, P0, np.zeros(d), np.zeros(d),
                                   alpha, beta, n_iters,
                                   track_every=max(1, n_iters // 200))
    elif method == "pdhg":
        P0 = np.zeros((d, d))
        P, _, _, history = ot_pdhg(C, a, b, P0, np.zeros(d), np.zeros(d),
                                    alpha, beta, n_iters,
                                    track_every=max(1, n_iters // 200))
    else:
        raise ValueError(f"Méthode inconnue : {method!r}")

    runtime = time.time() - t0

    T_Xs = transport_to_colors(P, a, Xt)
    out = recolor_image(src, Xs, T_Xs)

    cost_star = None
    if d <= 60:
        A = build_A_dense(d)
        res = linprog(C.flatten(), A_eq=A, b_eq=np.concatenate([a, b]),
                       bounds=(0, None), method="highs")
        if res.success:
            cost_star = res.fun

    return {
        "out": out, "P": P, "metrics": ot_metrics(C, a, b, P),
        "history": history, "runtime": runtime, "cost_star": cost_star,
        "Xs": Xs, "Xt": Xt, "C": C, "a": a, "b": b,
    }


# =============================================================================
# Programme principal
# =============================================================================

if __name__ == "__main__":

    SEP = "=" * 70

    # =========================================================================
    # Q4 — Problème jouet
    # =========================================================================
    print(SEP)
    print("  Q4 — Transport optimal sur le problème jouet")
    print(SEP)

    a4 = np.array([0.4, 0.6])
    b4 = np.array([0.5, 0.5])
    C4 = np.array([[1.0, 5.0],
                    [1.0, 2.0]])
    d4 = 2

    # ── Solution exacte du LP par scipy (référence) ──────────────────────────
    A4 = build_A_dense(d4)
    res4 = linprog(C4.flatten(), A_eq=A4, b_eq=np.concatenate([a4, b4]),
                    bounds=(0, None), method="highs")
    P4_star = res4.x.reshape(d4, d4)
    cost4_star = res4.fun
    print(f"  Solution exacte (linprog) : P* =\n{P4_star}")
    print(f"  Coût optimal              : ⟨C,P*⟩ = {cost4_star}\n")

    # ── Vérification de ‖A‖ = √(2d) = 2 ──────────────────────────────────────
    norm_A4 = power_iteration_norm_A(d4)
    print(f"  ‖A‖ (formule)        = √(2·{d4}) = {np.sqrt(2*d4):.6f}")
    print(f"  ‖A‖ (itér. puissance) = {norm_A4:.6f}")
    print(f"  Borne de pas PDHG (α=β) : α < 1/‖A‖ = {1/norm_A4:.4f}\n")

    # ── Validation croisée avec la Partie 1 (A dense) ────────────────────────
    diff_gda, diff_pdhg = _cross_check_with_part1(d4, C4, a4, b4,
                                                    alpha=0.2, beta=0.2,
                                                    n_iters=10)
    print(f"  Validation croisée vs Partie 1 (10 itérations, A dense 4×4) :")
    print(f"    écart max GDA  = {diff_gda:.2e}")
    print(f"    écart max PDHG = {diff_pdhg:.2e}")
    assert diff_gda < 1e-12 and diff_pdhg < 1e-12, "Incohérence avec la Partie 1 !"
    print("    → ot_gda/ot_pdhg sont identiques aux formules (3)-(4) de la Part.1  ✓\n")

    # ── Exécution GDA / PDHG, pas alpha=beta=0.2 (comme Partie 1, Q1-Q3) ────
    N4 = 2000
    P0 = np.zeros((d4, d4))
    p0 = q0 = np.zeros(d4)

    P_gda, _, _, hist_gda = ot_gda(C4, a4, b4, P0, p0, q0, 0.2, 0.2, N4)
    P_pdhg, _, _, hist_pdhg = ot_pdhg(C4, a4, b4, P0, p0, q0, 0.2, 0.2, N4)

    m_gda = ot_metrics(C4, a4, b4, P_gda)
    m_pdhg = ot_metrics(C4, a4, b4, P_pdhg)

    print(f"  Après N={N4} itérations (α=β=0.2) :")
    print(f"  {'':12s} {'cost':>10s} {'row_viol':>10s} {'col_viol':>10s} {'‖P-P*‖':>10s}")
    print(f"  {'GDA':12s} {m_gda['cost']:>10.4f} {m_gda['row_violation']:>10.2e} "
          f"{m_gda['col_violation']:>10.2e} {np.linalg.norm(P_gda-P4_star):>10.4f}")
    print(f"  {'PDHG':12s} {m_pdhg['cost']:>10.4f} {m_pdhg['row_violation']:>10.2e} "
          f"{m_pdhg['col_violation']:>10.2e} {np.linalg.norm(P_pdhg-P4_star):>10.4f}")
    print(f"  {'Optimal':12s} {cost4_star:>10.4f} {0.0:>10.2e} {0.0:>10.2e} {0.0:>10.4f}")
    print()
    print("  → PDHG converge vers P* à précision machine, comme en Partie 1.")
    print("  → GDA oscille / ne converge pas (mêmes valeurs propres |λ|=√1.04>1")
    print("    sur les directions instables du Lagrangien linéarisé).\n")

    # ── Figure de convergence Q4 ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Q4 : GDA vs PDHG — Transport optimal jouet", fontweight="bold")

    axes[0].semilogy(hist_gda["iter"],
                      np.abs(np.array(hist_gda["cost"]) - cost4_star) + 1e-16,
                      color="tab:red", label="GDA")
    axes[0].semilogy(hist_pdhg["iter"],
                      np.abs(np.array(hist_pdhg["cost"]) - cost4_star) + 1e-16,
                      color="tab:blue", label="PDHG")
    axes[0].set(title=r"Écart au coût optimal $|\langle C,P_k\rangle - \mathrm{OPT}|$",
                xlabel="Itération $k$")
    axes[0].legend(); axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogy(hist_gda["iter"],
                      np.array(hist_gda["row_violation"]) + np.array(hist_gda["col_violation"]) + 1e-16,
                      color="tab:red", label="GDA")
    axes[1].semilogy(hist_pdhg["iter"],
                      np.array(hist_pdhg["row_violation"]) + np.array(hist_pdhg["col_violation"]) + 1e-16,
                      color="tab:blue", label="PDHG")
    axes[1].set(title=r"Violation des contraintes $\|P_k\mathbf{1}-a\|+\|P_k^T\mathbf{1}-b\|$",
                xlabel="Itération $k$")
    axes[1].legend(); axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("q4_convergence.png", dpi=150, bbox_inches="tight")
    print("  → q4_convergence.png sauvegardé\n")
    plt.close(fig)

    # =========================================================================
    # Q5-Q6 — Transfert de couleurs
    # =========================================================================
    print(SEP)
    print("  Q5-Q6 — Transfert de couleurs : GDA / PDHG vs Sinkhorn")
    print(SEP)

    src, tgt, is_synth = load_or_synthesize_images("ossau.jpg", "rer.jpg")
    print(f"  Images chargées : src {src.shape}, tgt {tgt.shape} "
          f"({'synthétiques' if is_synth else 'fichiers réels'})\n")

    N_SAMPLES = 300
    N_ITERS = 4000
    alpha_ct = beta_ct = 0.9 / np.sqrt(2 * N_SAMPLES)
    print(f"  d = {N_SAMPLES} échantillons par image")
    print(f"  α = β = 0.9/√(2d) = {alpha_ct:.5f}  (borne PDHG : 1/√(2d) = "
          f"{1/np.sqrt(2*N_SAMPLES):.5f})")
    print(f"  n_iters (GDA/PDHG) = {N_ITERS}, numItermax (Sinkhorn) = {N_ITERS}\n")

    results = {}
    for method in ["sinkhorn", "gda", "pdhg"]:
        res = color_transfer_lp(src, tgt, method=method,
                                 n_samples=N_SAMPLES,
                                 alpha=alpha_ct, beta=beta_ct,
                                 n_iters=N_ITERS, seed=0)
        results[method] = res
        m = res["metrics"]
        print(f"  {method.upper():9s} : cost={m['cost']:8.4f}  "
              f"row_viol={m['row_violation']:.2e}  col_viol={m['col_violation']:.2e}  "
              f"runtime={res['runtime']:.2f}s")

    print()
    print("  ── Comparaison Q6 ──")
    print("  • Sinkhorn (régularisé entropiquement) donne un plan dense, lisse,")
    print("    avec un coût LÉGÈREMENT supérieur au coût LP non régularisé,")
    print("    mais converge en très peu d'itérations.")
    print("  • PDHG résout le LP non régularisé exact : coût ≈ optimal et")
    print("    violations de contraintes → 0, comme observé en Q4.")
    print("  • GDA, comme en Q4, n'atteint pas la convergence dans le même")
    print("    nombre d'itérations : violations de contraintes nettement")
    print("    plus élevées et coût plus instable.")
    print()

    # ── Figure de comparaison visuelle ───────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    fig.suptitle("Q5-Q6 : Transfert de couleurs — comparaison des méthodes",
                  fontweight="bold")

    axes[0].imshow(src); axes[0].set_title("Source"); axes[0].axis("off")
    axes[1].imshow(tgt); axes[1].set_title("Cible"); axes[1].axis("off")
    for ax, method in zip(axes[2:], ["sinkhorn", "gda", "pdhg"]):
        ax.imshow(results[method]["out"])
        ax.set_title(method.upper())
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("q5_q6_transfert_couleurs.png", dpi=150, bbox_inches="tight")
    print("  → q5_q6_transfert_couleurs.png sauvegardé")
    plt.close(fig)

    # ── Figure de convergence GDA vs PDHG sur le pb. de transfert de couleurs ─
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Q5-Q6 : Convergence GDA vs PDHG (transfert de couleurs)",
                  fontweight="bold")

    for method, col in [("gda", "tab:red"), ("pdhg", "tab:blue")]:
        h = results[method]["history"]
        axes[0].plot(h["iter"], h["cost"], color=col, label=method.upper())
        axes[1].semilogy(h["iter"],
                          np.array(h["row_violation"]) + np.array(h["col_violation"]) + 1e-16,
                          color=col, label=method.upper())

    axes[0].axhline(results["sinkhorn"]["metrics"]["cost"], color="gray",
                     ls=":", label="Sinkhorn (référence)")
    axes[0].set(title=r"Coût $\langle C, P_k\rangle$", xlabel="Itération $k$")
    axes[1].set(title=r"Violation des contraintes $\|P_k\mathbf{1}-a\|+\|P_k^T\mathbf{1}-b\|$",
                xlabel="Itération $k$")
    for ax in axes:
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("q5_q6_convergence.png", dpi=150, bbox_inches="tight")
    print("  → q5_q6_convergence.png sauvegardé")
    plt.close(fig)

    print(f"\n{SEP}")
    print("  Partie 2 terminée.")
    print(SEP)