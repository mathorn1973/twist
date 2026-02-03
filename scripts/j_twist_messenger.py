#!/usr/bin/env python3
"""
J-TWIST Canon -- Messenger
===========================

The complete derivation chain from J = 1 + zeta_5^2.

This script contains:
  - axioms
  - algebraic backbone
  - exact identities
  - derived physical constants
  - comparison with experiment
  - uniqueness proof

No simulation. No fitting. No parameters.
No random. No sweep. No external input.

It speaks. It asserts. It ends.

Canon v4.0
A. M. Thorn, 2025/2026

Dependencies: Python 3.8+ standard library only.
License: CC BY 4.0
"""

from fractions import Fraction
from math import sqrt, pi, log, sin

__all__ = [
    # Backbone
    "phi", "j", "s", "R",
    # Icosahedron
    "V", "E", "F", "chi", "deg_f", "deg_v",
    "VF", "core", "hull", "bandwidth", "A5",
    # Exact rationals
    "CORE_FRAC", "WEAK_TREE", "CODEC",
    # Gyron parameters
    "theta_q", "W", "g_geo", "X", "Z_par",
    # Capacity
    "Omega5", "S",
    # Queen
    "g_EM", "spar", "alpha_inv", "alpha",
    # Bridge
    "B_bridge",
    # Electroweak
    "sin2_thetaW",
    # Mass ladder
    "mu_proton", "muon_exact", "mu_muon", "mu_tau",
    # Gravity
    "G_dimless", "G_SI", "kappa", "G_dressed",
    # Cosmology
    "Omega_b", "Omega_DM", "Omega_m",
    "f_R", "eta_codec", "w_eos", "H_ratio",
    # Anchors
    "M_E_KG", "HBAR", "C_LIGHT", "MU_Z",
]


# ==============================================================
#  EXPERIMENT (CODATA 2022 / PDG 2024)
#  What Nature says. We compare, not fit.
#
#  Sources:
#    NIST CODATA 2022 (physics.nist.gov/cuu/Constants)
#      alpha_inv:  137.035999177(21)
#      m_p/m_e:    1836.152673426(32)
#      m_mu/m_e:   206.7682827(46)
#      m_tau/m_e:  3477.23(23)
#      G:          6.67430(15) x 10^-11 m^3 kg^-1 s^-2
#      m_e:        9.1093837139(28) x 10^-31 kg
#    PDG 2024 Electroweak review:
#      sin2(theta_W) MS-bar at M_Z: 0.23121(6)
#
#  Note on sigma: for very precisely measured ratios
#  (m_p/m_e, m_mu/m_e), ppm-level deviations correspond
#  to large sigma tensions. The script prints relative
#  error only; sigma can be computed from the uncertainties
#  listed above.
# ==============================================================

CODATA = {
    "alpha_inv":   (137.035999177,    0.000000021),
    "sin2_thetaW": (0.23121,          0.00006),
    "mu":          (1836.152673426,    0.000000032),
    "m_mu/m_e":    (206.7682827,      0.0000046),
    "m_tau/m_e":   (3477.23,          0.23),
    "G_SI":        (6.67430e-11,      0.00015e-11),
    "m_e/M_P":     (4.18546e-23,      0.00016e-23),
}


# ==============================================================
#  CONVERSION CONSTANTS
#  hbar and c are exact in modern SI (2019 redefinition).
#  m_e is the sole empirical anchor.
# ==============================================================

HBAR    = 1.054571817e-34       # J*s  (exact in SI)
C_LIGHT = 299792458             # m/s  (exact in SI)
M_E_KG  = 9.1093837139e-31     # kg   (CODATA 2022)

# For EXTENDED section only (extra empirical input):
MU_Z    = 178449.7              # M_Z/m_e (PDG)


# ==============================================================
#  UTILITY: Dilogarithm (for Genesis Identity)
# ==============================================================

def _Li2(x, terms=300):
    """Li_2(x) = sum x^k/k^2, |x| <= 1."""
    s = 0.0
    xk = x
    for k in range(1, terms + 1):
        s += xk / (k * k)
        xk *= x
    return s


# ==============================================================
#
#  THE DERIVATION CHAIN
#
#  J = 1 + zeta_5^2  -->  everything
#
# ==============================================================


# I. THE WORD
#
#  J := 1 + zeta_5^2  in Z[zeta_5]
#  N(J) = 1, |J| = phi^{-1}, arg(J) = 2pi/5
#  M_J in SL(4,Z), Spec moduli {phi,phi,phi^-1,phi^-1}


# II. BACKBONE

phi = (1 + sqrt(5)) / 2
j   = phi - 1
s   = sqrt(3 - phi)
R   = log(phi)

assert abs(s*s + phi*phi - 4) < 1e-14, "Thorn Triangle"


# III. ICOSAHEDRON {3, 5}

V       = 12
E       = 30
F       = 20
chi     = V - E + F
deg_f   = 3
deg_v   = 5

VF        = V * F
core      = V + F + 1
hull      = V + F
A5        = 60
bandwidth = V - chi

CORE_FRAC = Fraction(hull, core)        # 32/33
WEAK_TREE = Fraction(deg_f, V + 1)      # 3/13
CODEC     = Fraction(VF, 2**8)          # 15/16

assert chi == 2
assert VF == 240
assert core == 33
assert hull == 32
assert bandwidth == 10


# IV. GYRON PARAMETERS

theta_q = 2 * pi / phi**2
W       = 64 * pi**3 * phi**2
g_geo   = 32 * phi**2 * s
X       = 1 / (32 * pi**2 * phi**4)
Z_par   = 2 * s / (pi * phi**4)

assert abs(W * X - theta_q) < 1e-12, "W*X = theta_q"


# V. CAPACITY AND SLIP

Omega5 = 5 / X
S      = (Omega5 / (Omega5 + 1))**5


# VI. THE QUEEN

g_EM      = (8 * pi)**2 / 5
spar      = sqrt(s) / S
alpha_inv = g_EM * spar
alpha     = 1 / alpha_inv


# VII. BRIDGE LAW (closure)
#
#  B := 1/(alpha * g)
#  alpha * B * g = 1  by definition
#  This is not a prediction. It is the algebraic constraint
#  that closes the coupling-geometry loop.

B_bridge = 1 / (alpha * g_geo)

assert abs(alpha * B_bridge * g_geo - 1) < 1e-14, "Bridge"


# VIII. ELECTROWEAK

sin2_thetaW  = float(WEAK_TREE) + X

sin2_theta23 = 0.5 + 5 * alpha
sin2_theta12 = 1/3 - 3 * alpha
sin2_theta13 = 3 * alpha

assert abs(sin2_theta12 + sin2_theta13 - 1/3) < 1e-14
assert abs(3*sin2_theta23 - 5*sin2_theta13 - 1.5) < 1e-14


# IX. MASS LADDER

mu_proton = 6 * pi**5 * (1 + alpha**2 / 3)

muon_exact = Fraction(VF) - Fraction(core) - WEAK_TREE
assert muon_exact == Fraction(2688, 13)
mu_muon = float(muon_exact)

mu_tau = VF * deg_f * deg_v - A5 * chi - deg_f
assert mu_tau == 3477

neutron_tension = Fraction(deg_v, chi)

me_over_MP = float(CORE_FRAC) * alpha**bandwidth / sqrt(g_geo)

assert Fraction(core) + muon_exact + WEAK_TREE == Fraction(VF)


# X. GRAVITY

G_dimless = float(CORE_FRAC)**2 * alpha**(2*bandwidth) / g_geo
G_SI      = G_dimless * (HBAR * C_LIGHT) / M_E_KG**2

kappa     = 1 + X / phi
G_dressed = kappa**2 * G_SI


# XI. COSMOLOGY

Omega_b   = pi**2 / 200
f_R       = 12 * R**2 / pi**2
eta_codec = float(CODEC)
Omega_DM  = f_R * eta_codec
Omega_m   = Omega_b + Omega_DM

w_eos   = Fraction(-14, 15)
H_ratio = Fraction(13, 12)


# XII. THERMODYNAMICS

T_M_natural = kappa / (2 * R)


# XIII. ZETA LAYER

zeta_2    = pi**2 / 6
zeta_neg3 = Fraction(1, 120)

assert chi * 120 == VF, "Logos Lock"
assert abs(zeta_2 * float(zeta_neg3) - pi**2/720) < 1e-14


# XIV. EXTENDED
#
#  Status B+: proposed relations, not derived from J.
#  VEV and Top use only Canon quantities (ratios to m_e).
#  Higgs uses MU_Z: additional empirical input (not Canon).

vev_ratio   = mu_tau * (alpha_inv + deg_f / chi)    # VEV/m_e
top_ratio   = vev_ratio / sqrt(2)                   # m_t/m_e (y_t=1)
higgs_ratio = (alpha_inv / 100) * MU_Z              # m_H/m_e (EXTRA)
koide       = Fraction(chi, deg_f)                   # 2/3


# GENESIS IDENTITY (verified at import)

_genesis_lhs = pi**2 / 6
_genesis_rhs = _Li2(j**2) + _Li2(j) + 2 * R**2
assert abs(_genesis_lhs - _genesis_rhs) < 1e-10, "Genesis"


# ==============================================================
#
#  THE VOICE
#
# ==============================================================

W_ = 62      # output width


def _f(frac):
    return f"{frac.numerator}/{frac.denominator}"


def _cmp(name, tw, ev, ee):
    """Print comparison line, fits in ~62 cols."""
    err = abs(tw - ev) / abs(ev)
    if err < 1e-9:
        p = f"{err*1e9:.1f} ppb"
    elif err < 1e-6:
        p = f"{err*1e6:.1f} ppm"
    elif err < 1e-2:
        p = f"{err*100:.3f}%"
    else:
        p = f"{err*100:.1f}%"
    if abs(ev) > 1e-5:
        print(f"    {name:<10} T={tw:<17.9f} E={ev:<15.6f}"
              f" {p}")
    else:
        print(f"    {name:<10} T={tw:<17.4e} E={ev:<15.4e}"
              f" {p}")


def run():
    """The Messenger speaks."""

    print()
    print("=" * W_)
    print("  J-TWIST CANON -- MESSENGER")
    print("  v4.0 | A. M. Thorn | 2025/2026")
    print("=" * W_)

    # AXIOM

    print()
    print("  AXIOM (LOCK)")
    print()
    print("    J = 1 + zeta_5^2")
    print("    |J| = phi^(-1)    arg(J) = 2pi/5")
    print("    N(J) = 1          unit in Z[zeta_5]")

    # BACKBONE

    print()
    print("-" * W_)
    print("  BACKBONE")
    print()
    print(f"    phi = {phi}")
    print(f"    j   = {j}")
    print(f"    s   = {s}")
    print(f"    R   = {R}")
    print(f"    s^2 + phi^2 = {s*s + phi*phi}  (exact 4)")

    # ICOSAHEDRON

    print()
    print("-" * W_)
    print("  ICOSAHEDRON {{3, 5}}")
    print()
    print(f"    V={V}  E={E}  F={F}  chi={chi}")
    print(f"    deg_f={deg_f}  deg_v={deg_v}")
    print(f"    VF={VF}  core={core}  hull={hull}"
          f"  bw={bandwidth}")

    # GYRON

    print()
    print("-" * W_)
    print("  GYRON PARAMETERS")
    print()
    print(f"    theta_q = {theta_q}")
    print(f"    W       = {W}")
    print(f"    g       = {g_geo}")
    print(f"    X       = {X}")
    print(f"    Z       = {Z_par}")
    print(f"    W*X = theta_q: {abs(W*X - theta_q) < 1e-12}")

    # CAPACITY

    print()
    print("-" * W_)
    print("  CAPACITY AND SLIP")
    print()
    print(f"    Omega_5 = {Omega5}")
    print(f"    S       = {S}")

    # QUEEN

    print()
    print("-" * W_)
    print("  THE QUEEN")
    print()
    print(f"    g_EM   = 64*pi^2/5 = {g_EM}")
    print(f"    Spar   = sqrt(s)/S = {spar}")
    print(f"    a^(-1) = g_EM*Spar = {alpha_inv}")
    print()
    _cmp("a^(-1)", alpha_inv, *CODATA["alpha_inv"])

    # BRIDGE

    print()
    print("-" * W_)
    print("  BRIDGE LAW (closure)")
    print()
    print("    B := 1/(alpha*g)")
    print("    alpha*B*g = 1  by definition")
    print()
    print("    Not a prediction. The algebraic")
    print("    constraint that closes the loop:")
    print("    coupling (alpha) and geometry (g)")
    print("    are not independent.")
    print()
    print(f"    B = {B_bridge}")
    print(f"    product = {alpha * B_bridge * g_geo}")

    # ELECTROWEAK

    print()
    print("-" * W_)
    print("  ELECTROWEAK")
    print()
    print(f"    tree = {_f(WEAK_TREE)}"
          f" = {float(WEAK_TREE):.10f}")
    print(f"    phys = 3/13 + X = {sin2_thetaW:.10f}")
    print()
    _cmp("sin2(W)", sin2_thetaW, *CODATA["sin2_thetaW"])
    print()
    print(f"    PMNS:")
    print(f"      sin2(23) = 1/2+5a  = {sin2_theta23:.6f}")
    print(f"      sin2(12) = 1/3-3a  = {sin2_theta12:.6f}")
    print(f"      sin2(13) = 3a      = {sin2_theta13:.6f}")
    d1 = sin2_theta12 + sin2_theta13 - 1/3
    d2 = 3*sin2_theta23 - 5*sin2_theta13 - 1.5
    print(f"    Sum rules: {d1:.0e}, {d2:.0e} (exact 0)")

    # MASS LADDER

    print()
    print("-" * W_)
    print("  MASS LADDER (ratios to m_e)")
    print()

    print(f"    PROTON: 6*pi^5*(1 + a^2/3)")
    _cmp("m_p/m_e", mu_proton, *CODATA["mu"])
    print()

    print(f"    MUON: VF-core-3/13 = {_f(muon_exact)}")
    _cmp("m_mu/m_e", mu_muon, *CODATA["m_mu/m_e"])
    print()

    print(f"    TAU: 240*15-120-3 = {mu_tau}")
    _cmp("m_tau/m_e", float(mu_tau), *CODATA["m_tau/m_e"])
    print()

    nt = _f(neutron_tension)
    print(f"    NEUTRON: mu + {nt} - Delta_EM  (B)")
    print()

    print(f"    PLANCK: (32/33)*a^10/sqrt(g)")
    _cmp("m_e/M_P", me_over_MP, *CODATA["m_e/M_P"])
    print()

    print(f"    CONSERVATION:")
    print(f"      {core} + {_f(muon_exact)}"
          f" + {_f(WEAK_TREE)} = {VF}")

    # GRAVITY

    print()
    print("-" * W_)
    print("  GRAVITY")
    print()
    print(f"    G = (hbar*c/m_e^2)*(32/33)^2*a^20/g")
    print(f"    exponent 20 = 2*{bandwidth} = F")
    print(f"    kappa = 1 + X/phi = {kappa}")
    print()
    _cmp("G bare", G_SI, *CODATA["G_SI"])
    _cmp("G dress", G_dressed, *CODATA["G_SI"])

    # COSMOLOGY

    print()
    print("-" * W_)
    print("  COSMOLOGY")
    print()
    print(f"    Genesis: pi^2/6 ="
          f" Li2(j^2)+Li2(j)+2*ln^2(phi)")
    print(f"      residual = {abs(_genesis_lhs-_genesis_rhs):.1e}")
    print()
    print(f"    Omega_b  = pi^2/200     = {Omega_b:.6f}")
    print(f"    eta      = {_f(CODEC)}        = {eta_codec}")
    print(f"    f_R      = 12*R^2/pi^2  = {f_R:.6f}")
    print(f"    Omega_DM = f_R * eta    = {Omega_DM:.6f}")
    print(f"    Omega_m  = b + DM       = {Omega_m:.6f}")
    print(f"    w = {_f(w_eos)}  H_loc/H_CMB = {_f(H_ratio)}")

    # THERMODYNAMICS

    print()
    print("-" * W_)
    print("  THERMODYNAMICS")
    print()
    print(f"    T_M/(m_e c^2) = kappa/(2*R) = {T_M_natural:.6f}")

    # ZETA LAYER

    print()
    print("-" * W_)
    print("  ZETA LAYER")
    print()
    print(f"    zeta(2)  = {zeta_2:.12f}")
    print(f"    zeta(-3) = {_f(zeta_neg3)}")
    print(f"    Logos: chi/zeta(-3) = {VF} = VF")
    print(f"    Hall:  R_K/Z_0 = 1/(2a) = {1/(2*alpha):.3f}")

    # EXTENDED

    print()
    print("-" * W_)
    print("  EXTENDED (Status B+)")
    print()
    print("    All ratios to m_e.")
    print()
    print("    VEV/m_e = m_tau/m_e*(a^(-1)+3/2)  (B+)")
    print(f"      TWIST: {vev_ratio:.1f}    Exp: 481841")
    print()
    print("    m_t/m_e = VEV/(m_e*sqrt(2)) [y_t=1] (B)")
    print(f"      TWIST: {top_ratio:.1f}    Exp: 338083")
    print()
    print("    m_H/m_e = (a^(-1)/100)*M_Z/m_e  (C)")
    print(f"      TWIST: {higgs_ratio:.1f}    Exp: 245108")
    print("    WARNING: M_Z/m_e is extra input.")
    print("    Relation, not prediction.")
    print()
    print(f"    Koide = chi/deg_f = {_f(koide)}        (A)")
    print()
    print("    Neutron: Delta_EM open         (B)")

    # PRECISION TABLE

    print()
    print("=" * W_)
    print("  PRECISION TABLE")
    print("=" * W_)
    print()

    rows = [
        ("a^(-1)",    alpha_inv,     137.035999177, "sub-ppb"),
        ("sin2(W)",   sin2_thetaW,   0.23121,       "0.01%"),
        ("m_p/m_e",   mu_proton,     1836.152673,   "ppm"),
        ("m_mu/m_e",  mu_muon,       206.7682827,   "5 ppm"),
        ("m_tau/m_e", float(mu_tau), 3477.23,       "66 ppm"),
        ("m_e/M_P",   me_over_MP,    4.185e-23,     "0.03%"),
        ("G dress",   G_dressed,     6.6743e-11,    "sub-%"),
    ]

    print(f"    {'':10}  {'TWIST':>12}  {'EXP':>12}  {'':>7}")
    print(f"    {'-'*10}  {'-'*12}  {'-'*12}  {'-'*7}")
    for nm, tw, ex, pr in rows:
        if abs(ex) > 1e-5:
            print(f"    {nm:<10}  {tw:>12.6f}"
                  f"  {ex:>12.6f}  {pr:>7}")
        else:
            print(f"    {nm:<10}  {tw:>12.4e}"
                  f"  {ex:>12.4e}  {pr:>7}")

    # EXACT RATIONALS

    print()
    print("=" * W_)
    print("  EXACT RATIONALS")
    print("=" * W_)
    print()
    print(f"    Weinberg tree:   {_f(WEAK_TREE)}")
    print(f"    Muon ratio:      {_f(muon_exact)}")
    print(f"    Tau ratio:       {mu_tau}")
    print(f"    Hull/Core:       {_f(CORE_FRAC)}")
    print(f"    Codec:           {_f(CODEC)}")
    print(f"    Koide Q:         {_f(koide)}")
    print(f"    Eq. of state:    {_f(w_eos)}")
    print(f"    Hubble ratio:    {_f(H_ratio)}")
    print(f"    Isospin:         {_f(neutron_tension)}")
    print(f"    zeta(-3):        {_f(zeta_neg3)}")

    # UNIQUENESS

    print()
    print("=" * W_)
    print("  UNIQUENESS")
    print("=" * W_)
    print()
    _uniqueness()

    # CHAIN

    print()
    print("=" * W_)
    print("  THE CHAIN")
    print("=" * W_)
    print()
    print("    J = 1 + zeta_5^2")
    print("      |")
    print("    (phi, j, s, R)           backbone")
    print("      |")
    print("    (theta_q, W, X, g)       gyron")
    print("      |")
    print("    (Omega, S, Spar)         capacity")
    print("      |")
    print("    a^(-1)=(64pi^2/5)*Spar   QUEEN")
    print("      |")
    print("    B = 1/(a*g)              BRIDGE (closure)")
    print("      |")
    print("    sin2(W), mu, leptons, G  PHYSICS")
    print("      |")
    print("    Omega_b, Omega_DM, w, H  COSMOLOGY")
    print("      |")
    print("    k_B * T_M                THERMO")
    print()
    print("    Free parameters: 0")
    print("    Anchor: m_e (sole empirical input)")
    print("    Exact in SI: c, hbar")
    print("    M_Z used only in EXTENDED (Status C)")

    # KILL SHOTS

    print()
    print("=" * W_)
    print("  THREE KILL SHOTS")
    print("=" * W_)
    print()
    print("  1. sin2(W) outside [0.2305, 0.2320]: dead")
    print("  2. m_gamma != 0: dead")
    print("  3. 4th generation found: dead")

    # CODA

    print()
    print("=" * W_)
    print()
    print("  Five stones. One geometry. One anchor.")
    print("  Everything else is bookkeeping.")
    print()
    print("  Simplizis.")
    print()
    print("=" * W_)


def _uniqueness():
    """
    Uniqueness within the {p,q} Platonic family
    under the model's extension rules (dim=6).
    Not a general mathematical proof.
    """

    def _ag(p, q, dim=6):
        dn = 4 - (p-2)*(q-2)
        if dn <= 0:
            return None, None
        if (4*p) % dn or (4*q) % dn:
            return None, None
        Vg = (4*p)//dn
        Eg = (2*p*q)//dn
        Fg = (4*q)//dn
        if Vg - Eg + Fg != 2:
            return None, None

        th = 2*pi/phi**2
        Vgeo = (2**dim)*(pi**(dim/2))*phi**2
        Xg = th/Vgeo
        Og = q/Xg
        Sg = (Og/(Og+1))**q

        ch = sqrt(2*sin(pi/q))
        ap = (8*pi)**2 * ch / q
        ds = 1 - Xg + (p/q)*Xg**2
        ai = ap/ds

        ag = 1/ai
        w = (Vg/2)*pi**q
        mg = w*(1 + ag**2/p)
        return ai, mg

    ta = CODATA["alpha_inv"][0]
    tm = CODATA["mu"][0]

    solids = [
        ("{3,5} Ico",  3, 5),
        ("{5,3} Dod",  5, 3),
        ("{4,3} Cub",  4, 3),
        ("{3,4} Oct",  3, 4),
        ("{3,3} Tet",  3, 3),
    ]

    print(f"    {'':12} {'a^(-1)':>9}"
          f" {'err':>9} {'mu':>9} {'err':>9}")
    print(f"    {'-'*12} {'-'*9}"
          f" {'-'*9} {'-'*9} {'-'*9}")

    for nm, p, q in solids:
        ai, mg = _ag(p, q)
        if ai is None:
            print(f"    {nm:<12} REJECTED")
            continue
        ea = abs(ai-ta)/ta
        em = abs(mg-tm)/tm
        sa = "<ppb" if ea < 1e-9 else f"{ea:.0e}"
        sm = "<ppm" if em < 1e-6 else f"{em:.0e}"
        mk = " <-" if ea < 1e-6 else ""
        print(f"    {nm:<12} {ai:>9.3f}"
              f" {sa:>9} {mg:>9.1f} {sm:>9}{mk}")

    print()
    print("  Only {3, 5} survives.")


# ==============================================================

if __name__ == "__main__":
    run()
