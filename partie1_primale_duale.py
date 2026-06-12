"""
=============================================================================
Partie 1 – Méthodes primales-duales de base
=============================================================================
Cours  : Optimisation pour l'apprentissage automatique – M2 IASD 2025/2026
Sujet  : PR-OAA-2526  (C. W. Royer, Dauphine–PSL)

On résout le problème de point-selle associé au programme linéaire (LP) :

    (LP)   minimiser  c^T w    s.c.  A w = b,  w ≥ 0
           w ∈ R^d

via le Lagrangien augmenté de la contrainte d'égalité :

    (Lag)  min_{w ≥ 0}  max_{z ∈ R^r}  L(w, z) = c^T w − z^T (A w − b)

Trois algorithmes sont implémentés et comparés :

    Q1.  GDA     – Gradient Descent-Ascent (formule (3) du sujet)
    Q2.  Alt-GDA – GDA à pas alternants    (variante récente)
    Q3.  PDHG    – Primal-Dual Hybrid Gradient / Chambolle–Pock (formule (4))

Application au problème jouet :

    min_{w ≥ 0}  max_{z ∈ R}  (w − 3) z
    Forme LP :  c = [0],  A = [[-1]],  b = [-3]
    Saddle-point exact :  w* = 3,  z* = 0,  L(w*, z*) = 0.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.dpi": 120})


# =============================================================================
# Utilitaire
# =============================================================================

def proj_pos(v: np.ndarray) -> np.ndarray:
    """
    Projection euclidienne sur l'orthant positif R^d_{≥0} :

        [proj_{≥0}(v)]_i = max(0, v_i)

    Parameters
    ----------
    v : ndarray (d,)
    Returns
    -------
    ndarray (d,)  — même forme, toutes composantes ≥ 0
    """
    return np.maximum(0.0, v)


# =============================================================================
# Question 1 – Gradient Descent-Ascent (GDA)
# =============================================================================

def gda(
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    z0: np.ndarray,
    alpha: float,
    beta: float,
    n_iters: int = 500,
) -> tuple:
    """
    Gradient Descent-Ascent appliqué au Lagrangien L(w,z) = c^T w − z^T(Aw−b).

    Gradients partiels :
        ∇_w L(w, z) =  c − A^T z          (gradient primal, direction descente)
        ∇_z L(w, z) = −(A w − b)          (gradient dual,   direction montée)

    Itération (formule (3) du sujet) :
    ┌──────────────────────────────────────────────────────────────────────┐
    │  w_{k+1} = proj_{≥0}[ w_k − α_k (c − A^T z_k) ]   descente projetée│
    │  z_{k+1} = z_k − β_k (A w_k − b)                   montée gradient  │
    └──────────────────────────────────────────────────────────────────────┘

    Note sur la mise à jour de z :
        z_{k+1} = z_k + β_k ∇_z L(w_k, z_k)
                = z_k + β_k (−(A w_k − b))
                = z_k − β_k (A w_k − b)          ← forme utilisée ici

    IMPORTANT : la mise à jour de z utilise w_k (AVANT la mise à jour primale),
    pas w_{k+1}. C'est ce qui différencie GDA de PDHG.

    ──────────────────────────────────────────────────────────────────────────
    Comportement sur le problème jouet (bilinéaire)
    ──────────────────────────────────────────────────────────────────────────
    La linéarisation au voisinage du saddle-point (w*, z*) = (3, 0) donne le
    jacobien  J_GDA = [[1, −α], [β, 1]]  avec valeurs propres λ = 1 ± i√(αβ).

    Pour α = β = 0.2 :  |λ| = √(1 + αβ) = √1.04 ≈ 1.02 > 1
    → Les itérés divergent (spirale vers l'extérieur).

    Parameters
    ----------
    c       : (d,)    vecteur de coût primal
    A       : (r, d)  matrice de contraintes
    b       : (r,)    second membre
    w0      : (d,)    itéré primal initial  (doit vérifier w0 ≥ 0)
    z0      : (r,)    itéré dual  initial
    alpha   : float   pas primal  α > 0
    beta    : float   pas dual    β > 0
    n_iters : int     nombre d'itérations

    Returns
    -------
    ws : ndarray (n_iters+1, d)  — itérés primaux  [w_0, w_1, …, w_K]
    zs : ndarray (n_iters+1, r)  — itérés duaux    [z_0, z_1, …, z_K]
    """
    w = w0.copy().astype(float)
    z = z0.copy().astype(float)
    ws = [w.copy()]
    zs = [z.copy()]

    for _ in range(n_iters):
        # ── Gradient primal ─────────────────────────────────────────────────
        grad_w = c - A.T @ z                   # ∇_w L = c − A^T z

        # ── Mise à jour primale : descente de gradient projetée ─────────────
        w_new = proj_pos(w - alpha * grad_w)

        # ── Mise à jour duale : montée de gradient ──────────────────────────
        # On utilise w_k (AVANT la mise à jour primale)
        # z_{k+1} = z_k + β ∇_z L(w_k, z_k) = z_k − β (A w_k − b)
        z_new = z - beta * (A @ w - b)

        w, z = w_new, z_new
        ws.append(w.copy())
        zs.append(z.copy())

    return np.array(ws), np.array(zs)


# =============================================================================
# Question 2 – GDA à pas alternants (Alt-GDA)
# =============================================================================

def alt_gda(
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    z0: np.ndarray,
    step: float = 0.2,
    n_iters: int = 500,
) -> tuple:
    """
    GDA à pas alternants – variante proposée récemment dans la littérature.

    Schéma de pas (tel que défini dans le sujet) :
        α_{2k}   = −β_{2k}   = −α_{2k+1} = β_{2k+1} = step > 0

    Ce qui donne, pour chaque itération k :
        k pair   →  (α_k, β_k) = (+step, −step)
        k impair →  (α_k, β_k) = (−step, +step)

    ──────────────────────────────────────────────────────────────────────────
    Pourquoi est-ce surprenant ?
    ──────────────────────────────────────────────────────────────────────────
    • Aux itérations impaires, α_k < 0 implique :
          w_{k+1} = proj_{≥0}[w_k − (−step)·∇_w L] = proj_{≥0}[w_k + step·∇_w L]
      On effectue une MONTÉE de gradient sur w, à l'encontre de la minimisation.

    • Aux itérations paires, β_k < 0 implique une DESCENTE de gradient sur z,
      à l'encontre de la maximisation.

    Malgré ces inversions périodiques de rôle, la méthode converge. La clé est
    la structure par paires d'itérations.

    ──────────────────────────────────────────────────────────────────────────
    Preuve de convergence sur le problème jouet (sans projection active)
    ──────────────────────────────────────────────────────────────────────────
    Posons u_k = w_k − w*. En déroulant deux itérations consécutives :

    Étape 1 — k pair   (α = +step, β = −step) :
        u_{k+1} = u_k  −  step · z_k
        z_{k+1} = z_k  −  step · u_k      ← β < 0 donne une addition

    Étape 2 — k impair (α = −step, β = +step) :
        u_{k+2} = u_{k+1} + step · z_{k+1}
                = (u_k − step·z_k) + step·(z_k − step·u_k)
                = u_k − step²·u_k
                = (1 − step²) · u_k

        z_{k+2} = z_{k+1} + step · u_{k+1}
                = (z_k − step·u_k) + step·(u_k − step·z_k)
                = z_k − step²·z_k
                = (1 − step²) · z_k

    ⟹  Après deux itérations :  (u, z) ↦ (1 − step²) · (u, z)

    Pour step = 0.2 :  facteur de contraction = 1 − 0.04 = 0.96 < 1  ✓
    Convergence linéaire avec taux O(0.96^{k/2}).

    Parameters
    ----------
    step    : float  amplitude commune des pas  (doit vérifier 0 < step < 1)
    (autres paramètres identiques à gda())
    """
    w = w0.copy().astype(float)
    z = z0.copy().astype(float)
    ws = [w.copy()]
    zs = [z.copy()]

    for k in range(n_iters):
        # Schéma de pas : α et β ont des signes opposés, qui s'inversent
        if k % 2 == 0:                       # itération paire
            alpha_k, beta_k = +step, -step
        else:                                # itération impaire
            alpha_k, beta_k = -step, +step

        grad_w = c - A.T @ z
        w_new  = proj_pos(w - alpha_k * grad_w)
        z_new  = z - beta_k * (A @ w - b)   # utilise toujours w_k (avant mise à jour)

        w, z = w_new, z_new
        ws.append(w.copy())
        zs.append(z.copy())

    return np.array(ws), np.array(zs)


# =============================================================================
# Question 3 – Primal-Dual Hybrid Gradient (PDHG)
# =============================================================================

def pdhg(
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    z0: np.ndarray,
    alpha: float,
    beta: float,
    n_iters: int = 500,
) -> tuple:
    """
    PDHG – Primal-Dual Hybrid Gradient (Chambolle–Pock, 2011).

    Itération (formule (4) du sujet) :
    ┌──────────────────────────────────────────────────────────────────────┐
    │  w_{k+1} = proj_{≥0}[ w_k − α_k (c − A^T z_k) ]                    │
    │  z_{k+1} = z_k − β_k ( A (2 w_{k+1} − w_k) − b )                   │
    └──────────────────────────────────────────────────────────────────────┘

    ──────────────────────────────────────────────────────────────────────────
    Différence clé avec GDA
    ──────────────────────────────────────────────────────────────────────────
    La mise à jour duale utilise le point extrapolé (overrelaxation, θ = 1) :
        w̃_{k+1} := 2 w_{k+1} − w_k   =   w_{k+1} + (w_{k+1} − w_k)

    Intuition : on « anticipe » la prochaine étape primale en prolongeant le
    déplacement w_{k+1} − w_k. Cela compense l'effet oscillant du GDA.

    ──────────────────────────────────────────────────────────────────────────
    Analyse spectrale au voisinage du saddle-point (cas scalaire)
    ──────────────────────────────────────────────────────────────────────────
    En posant u_k = w_k − w*, le jacobien linéarisé est :
        J_PDHG = [[1, −α], [β, 1 − 2αβ]]

    Pour α = β = 0.2 :
        J = [[1, −0.2], [0.2, 0.92]]
        Polynôme caractéristique : λ² − 1.92λ + 0.96 = 0
        Discriminant = 1.92² − 4·0.96 = −0.1536 < 0  (valeurs propres complexes)
        λ = 0.96 ± 0.196i
        |λ|² = det(J) = 0.96  ⟹  |λ| = √0.96 ≈ 0.98 < 1   ✓  (convergence)

    Comparaison directe :
        GDA  : |λ(J_GDA)|  = √1.04 ≈ 1.02 > 1  →  diverge
        PDHG : |λ(J_PDHG)| = √0.96 ≈ 0.98 < 1  →  converge

    Condition suffisante de convergence garantie : α · β · ‖A‖² < 1
    Ici : 0.2 × 0.2 × ‖−1‖² = 0.04 < 1  ✓

    Parameters (identiques à gda)
    """
    w = w0.copy().astype(float)
    z = z0.copy().astype(float)
    ws = [w.copy()]
    zs = [z.copy()]

    for _ in range(n_iters):
        # ── Mise à jour primale (identique à GDA) ───────────────────────────
        grad_w = c - A.T @ z
        w_new  = proj_pos(w - alpha * grad_w)

        # ── Point extrapolé (overrelaxation θ = 1) ──────────────────────────
        #    w̃ = 2 w_{k+1} − w_k  =  w_{k+1} + (w_{k+1} − w_k)
        w_extrap = 2.0 * w_new - w

        # ── Mise à jour duale avec extrapolation ────────────────────────────
        #    Utilise w̃ au lieu de w_k  ← différence fondamentale avec GDA
        z_new = z - beta * (A @ w_extrap - b)

        w, z = w_new, z_new
        ws.append(w.copy())
        zs.append(z.copy())

    return np.array(ws), np.array(zs)


# =============================================================================
# Métriques de convergence
# =============================================================================

def compute_metrics(
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    ws: np.ndarray,
    zs: np.ndarray,
    w_star: np.ndarray | None = None,
) -> dict:
    """
    Calcule les métriques de convergence pour une trajectoire (ws, zs).

    Returns
    -------
    dict avec les clés :
      'lagrangian'      : L(w_k, z_k) = c^T w_k − z_k^T(A w_k − b), pour k=0…K
      'primal_residual' : ‖A w_k − b‖  (violation des contraintes d'égalité)
      'dist_to_opt'     : ‖w_k − w*‖   si w_star fourni, sinon None
    """
    lagr = np.array([c @ w - z @ (A @ w - b) for w, z in zip(ws, zs)])
    pres = np.array([np.linalg.norm(A @ w - b) for w in ws])
    dist = (
        np.array([np.linalg.norm(w - w_star) for w in ws])
        if w_star is not None
        else None
    )
    return {"lagrangian": lagr, "primal_residual": pres, "dist_to_opt": dist}


# =============================================================================
# Tracé des figures de comparaison
# =============================================================================

def plot_all(algos, iters, save=True):
    """
    Produit trois figures de comparaison :
        Fig. 1 – Trajectoires w_k et z_k
        Fig. 2 – Convergence en échelle log
        Fig. 3 – Portraits de phase (w_k, z_k)

    Parameters
    ----------
    algos : list de tuples (nom, ws, zs, metrics, couleur, linestyle)
    iters : ndarray des indices d'itération
    save  : bool  — sauvegarde les figures en PNG si True
    """
    N = len(iters) - 1

    # ── Figure 1 : Trajectoires ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        r"Q1–Q3 : Trajectoires — $\min_{w \geq 0}\max_z\,(w-3)z$",
        fontweight="bold",
    )

    for name, ws, zs, _, col, ls in algos:
        axes[0].plot(iters, ws[:, 0], label=name, color=col, ls=ls, lw=1.8)
        axes[1].plot(iters, zs[:, 0], label=name, color=col, ls=ls, lw=1.8)

    axes[0].axhline(3.0, color="k", lw=0.9, ls=":", label=r"$w^*=3$")
    axes[1].axhline(0.0, color="k", lw=0.9, ls=":", label=r"$z^*=0$")

    axes[0].set(title="Variable primale $w_k$",
                xlabel="Itération $k$", ylabel=r"$w_k$")
    axes[1].set(title="Variable duale $z_k$",
                xlabel="Itération $k$", ylabel=r"$z_k$")
    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig("q1_trajectoires.png", dpi=150, bbox_inches="tight")
        print("  → q1_trajectoires.png sauvegardé")
    plt.show()

    # ── Figure 2 : Convergence (échelle log) ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Q1–Q3 : Convergence (échelle logarithmique)", fontweight="bold")

    for name, _, _, m, col, ls in algos:
        # +1e-16 pour éviter log(0) si la méthode converge exactement
        axes[0].semilogy(iters, m["dist_to_opt"] + 1e-16,
                         label=name, color=col, ls=ls, lw=1.8)
        axes[1].semilogy(iters, m["primal_residual"] + 1e-16,
                         label=name, color=col, ls=ls, lw=1.8)

    # Taux théoriques : Alt-GDA et PDHG ont tous les deux un taux de 0.96 par 2 iter
    rate_conv = 0.96 ** (iters / 2)
    axes[0].semilogy(iters, rate_conv * 1.0, color="gray", ls=":", lw=1.2,
                     label=r"Taux $0.96^{k/2}$")

    axes[0].set(title=r"Distance à la solution  $\|w_k - w^*\|$",
                xlabel="Itération $k$")
    axes[1].set(title=r"Résidu primal  $\|A w_k - b\|$",
                xlabel="Itération $k$")
    for ax in axes:
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig("q1_convergence.png", dpi=150, bbox_inches="tight")
        print("  → q1_convergence.png sauvegardé")
    plt.show()

    # ── Figure 3 : Portrait de phase ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(r"Q1–Q3 : Portraits de phase $(w_k,\,z_k)$", fontweight="bold")

    for ax, (name, ws, zs, _, col, _) in zip(axes, algos):
        sc = ax.scatter(
            ws[:, 0], zs[:, 0],
            c=np.arange(N + 1), cmap="viridis", s=7, zorder=3,
        )
        ax.plot(ws[:, 0], zs[:, 0], lw=0.5, alpha=0.5, color=col)
        ax.scatter([3.0], [0.0], marker="*", s=250, color="red",
                   zorder=5, label=r"$(w^*, z^*)=(3,0)$")
        ax.set(title=name, xlabel=r"$w_k$", ylabel=r"$z_k$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label="Itération $k$")

    plt.tight_layout()
    if save:
        plt.savefig("q1_phase.png", dpi=150, bbox_inches="tight")
        print("  → q1_phase.png sauvegardé")
    plt.show()


# =============================================================================
# Programme principal – Application au problème jouet
# =============================================================================

if __name__ == "__main__":

    # ── Mise en forme du problème jouet ──────────────────────────────────────
    #
    #  Objectif : min_{w ≥ 0}  max_{z ∈ R}  (w − 3) z
    #
    #  On vérifie la correspondance avec L(w,z) = c^T w − z^T (A w − b) :
    #
    #    c = 0, A = [[-1]], b = [-3]  →  L = 0 − z·(−w − (−3))
    #                                        = −z(3 − w)
    #                                        = z(w − 3) = (w−3)z  ✓
    #
    #  Solution exacte (saddle-point unique) :  w* = 3,  z* = 0
    #
    #    • Pour w ≠ 3 :  sup_z (w−3)z = +∞  (en prenant z → ±∞)
    #    • Pour w  = 3 :  sup_z 0     = 0
    #    → min_{w≥0} vaut 0, atteint en w* = 3.
    #
    #  Vérification KKT :  condition de stationnarité primale z* = 0
    #    ∇_w L(w*, z*) = z* = 0 ≥ 0  et  z*·(w* − 0) = 0·3 = 0  ✓
    # ─────────────────────────────────────────────────────────────────────────

    c_toy  = np.array([0.0])
    A_toy  = np.array([[-1.0]])    # 1×1
    b_toy  = np.array([-3.0])
    w_star = np.array([3.0])       # solution primale exacte
    z_star = np.array([0.0])       # solution duale exacte

    w0 = np.array([2.0])
    z0 = np.array([2.0])

    ALPHA = 0.2    # pas primal commun
    BETA  = 0.2    # pas dual   commun
    N     = 300    # nombre d'itérations

    print("=" * 64)
    print("  Partie 1 – Méthodes primales-duales de base")
    print("  Problème jouet : min_{w≥0} max_z (w−3)z")
    print(f"  Point initial  : w0 = {w0[0]},  z0 = {z0[0]}")
    print(f"  Pas            : α = β = {ALPHA}")
    print(f"  Itérations     : {N}")
    print("=" * 64)

    # ── Vérification numérique du saddle-point ───────────────────────────────
    L_star = c_toy @ w_star - z_star @ (A_toy @ w_star - b_toy)
    assert abs(L_star) < 1e-12, "Erreur : (w*, z*) n'est pas un point-selle."
    assert abs(A_toy @ w_star - b_toy) < 1e-12, "Erreur : w* n'est pas faisable."
    print(f"  Saddle-point vérifié : L(w*, z*) = {L_star:.1f}  ✓\n")

    # ── Exécution des algorithmes ─────────────────────────────────────────────
    print("Exécution des algorithmes…")
    ws_gda,  zs_gda  = gda    (c_toy, A_toy, b_toy, w0, z0, ALPHA, BETA,    N)
    ws_alt,  zs_alt  = alt_gda(c_toy, A_toy, b_toy, w0, z0, step=0.2, n_iters=N)
    ws_pdhg, zs_pdhg = pdhg   (c_toy, A_toy, b_toy, w0, z0, ALPHA, BETA,    N)
    print("  Terminé.\n")

    # ── Calcul des métriques ──────────────────────────────────────────────────
    m_gda  = compute_metrics(c_toy, A_toy, b_toy, ws_gda,  zs_gda,  w_star)
    m_alt  = compute_metrics(c_toy, A_toy, b_toy, ws_alt,  zs_alt,  w_star)
    m_pdhg = compute_metrics(c_toy, A_toy, b_toy, ws_pdhg, zs_pdhg, w_star)

    # ── Résumé numérique ──────────────────────────────────────────────────────
    SEP = "─" * 64
    print(SEP)
    print(f"  Résumé numérique à l'itération K = {N}")
    print(SEP)
    print(f"  {'Méthode':<10}  {'w_K':>8}  {'z_K':>10}  "
          f"{'‖w_K−w*‖':>10}  {'‖Aw_K−b‖':>10}")
    print(SEP)
    for name, ws, zs, m in [
        ("GDA",    ws_gda,  zs_gda,  m_gda),
        ("Alt-GDA",ws_alt,  zs_alt,  m_alt),
        ("PDHG",   ws_pdhg, zs_pdhg, m_pdhg),
    ]:
        print(f"  {name:<10}  {ws[-1, 0]:>8.4f}  {zs[-1, 0]:>10.4f}  "
              f"{m['dist_to_opt'][-1]:>10.3e}  {m['primal_residual'][-1]:>10.3e}")
    print(SEP)

    # ── Observations ─────────────────────────────────────────────────────────
    print()
    print("  ── Observations ──")
    print()
    print("  Q1 – GDA (α = β = 0.2) :")
    print("    Les itérés DIVERGENT. Les variables (w_k, z_k) oscillent avec")
    print("    une amplitude croissante. Le portrait de phase montre une spirale")
    print("    vers l'extérieur, typique des jeux bilinéaires traités par GDA.")
    print("    Analyse spectrale : |λ(J_GDA)| = √(1 + αβ) = √1.04 ≈ 1.02 > 1.")
    print()
    print("  Q2 – Alt-GDA (step = 0.2) :")
    print("    CONVERGE malgré des pas alternants partiellement négatifs.")
    print("    Aux itérations impaires, α_k = −0.2 correspond à une montée de")
    print("    gradient sur w — ce qui semble contre-intuitif pour une minimisation.")
    print("    La convergence s'explique par une contraction exacte sur deux pas :")
    print("        (u_k, z_k) → (1 − step²)·(u_k, z_k) = 0.96·(u_k, z_k)")
    print("    Taux de convergence : O(0.96^{k/2}) par itération.")
    print()
    print("  Q3 – PDHG (α = β = 0.2) :")
    print("    CONVERGE grâce au point extrapolé w̃ = 2w_{k+1} − w_k dans")
    print("    la mise à jour duale. L'extrapolation compense les oscillations")
    print("    du GDA en anticipant le prochain déplacement primal.")
    print("    Analyse spectrale : |λ(J_PDHG)| = √det(J) = √0.96 ≈ 0.98 < 1.")
    print("    Condition de pas : α·β·‖A‖² = 0.04 < 1  ✓")
    print()
    print("  Remarque : Alt-GDA et PDHG ont le même taux asymptotique 0.96^{k/2}")
    print("  sur ce problème, ce qui est une coïncidence remarquable.")
    print(SEP)

    # ── Tracé des figures ─────────────────────────────────────────────────────
    iters = np.arange(N + 1)
    algos = [
        ("GDA",     ws_gda,  zs_gda,  m_gda,  "tab:red",    "--"),
        ("Alt-GDA", ws_alt,  zs_alt,  m_alt,  "tab:orange", "-."),
        ("PDHG",    ws_pdhg, zs_pdhg, m_pdhg, "tab:blue",   "-"),
    ]

    print("\nGénération des figures…")
    plot_all(algos, iters, save=True)
    print("Terminé.")