"""Independent implementation of Lim et al. 2014 (arXiv:1311.7129) Eqs. (1)-(5), written from the paper text
(main text Eqs. 1-5 + "Evaluation" section; supplementary Eq. for eps_sec) and the unprinted model choices
listed in PHYSICS_DIFF.md. It does NOT import or read src/qkd_opt/physics_lim.py.

Conventions (as printed unless noted):
  eta_ch = 10^(-alpha L/10), eta_sys = eta_Bob eta_ch
  D_k = 1 - (1 - 2 p_dc) exp(-eta_sys k), R_k = D_k (1 + p_ap)
  e_k = p_dc + e_mis [1 - exp(-eta k)] + p_ap D_k / 2, with eta = eta_sys here (preset lim; the printed text
        says eta_ch -- see PHYSICS_DIFF.md)
  unprinted (PHYSICS_DIFF.md): N = n_X / (q_x^2 sum p_k R_k); n_{B,k} = N q_B^2 p_k R_k; m_{B,k} = N q_B^2 p_k e_k
        (q_X = q_x, q_Z = 1 - q_x); e_obs = sum_k m_{X,k} / n_X; floor of Eq. (1) dropped; s_0 clamped >= 0;
        single-photon term dropped if s_{X,1} <= 0 or s_{Z,1} <= 0; phi capped at 1/2.
  Eq. (5): gamma(a,b,c,d) = sqrt( (c+d)(1-b)b / (c d log 2) * log2( (c+d)/(c d (1-b) b) * 21^2/a^2 ) ),
        "log 2" = natural log of 2.
  Eq. (1): l = s_X0 + s_X1 - s_X1 h(phi) - lambda_EC - 6 log2(21/eps_sec) - log2(2/eps_cor),
        lambda_EC = n_X f_EC h(e_obs).
"""
import math
import unittest
from decimal import Decimal, getcontext

import numpy as np

C = dict(alpha=0.2, eta_bob=0.1, p_dc=6e-7, e_mis=5e-3, p_ap=4e-2, f_ec=1.16, eps_cor=1e-15, mu3=2e-4, kappa=1e-15)


def h2(x):
    x = min(max(x, 0.0), 0.5)
    if x == 0.0:
        return 0.0
    return -(x * math.log2(x) + (1 - x) * math.log2(1 - x))


def key_length_indep(mu1, mu2, P1, P2, qx, L, nX, eps_sec, c=C, eta_in_emis="sys"):
    mu = [mu1, mu2, c["mu3"]]
    p = [P1, P2, 1.0 - P1 - P2]
    eta_ch = 10.0 ** (-c["alpha"] * L / 10.0)
    eta_sys = c["eta_bob"] * eta_ch
    eta_e = eta_sys if eta_in_emis == "sys" else eta_ch
    D = [1.0 - (1.0 - 2.0 * c["p_dc"]) * math.exp(-eta_sys * k) for k in mu]
    Rk = [d * (1.0 + c["p_ap"]) for d in D]
    e = [c["p_dc"] + c["e_mis"] * (1.0 - math.exp(-eta_e * k)) + c["p_ap"] * d / 2.0 for k, d in zip(mu, D)]
    qz = 1.0 - qx
    N = nX / (qx * qx * sum(pk * r for pk, r in zip(p, Rk)))
    nX_k = [N * qx * qx * pk * r for pk, r in zip(p, Rk)]
    nZ_k = [N * qz * qz * pk * r for pk, r in zip(p, Rk)]
    mX_k = [N * qx * qx * pk * ek for pk, ek in zip(p, e)]
    mZ_k = [N * qz * qz * pk * ek for pk, ek in zip(p, e)]
    lnf = math.log(21.0 / eps_sec)

    def bounds(counts):
        dlt = math.sqrt(sum(counts) / 2.0 * lnf)
        plus = [math.exp(k) / pk * (cnt + dlt) for k, pk, cnt in zip(mu, p, counts)]
        minus = [math.exp(k) / pk * (cnt - dlt) for k, pk, cnt in zip(mu, p, counts)]
        return plus, minus

    tau0 = sum(pk * math.exp(-k) for k, pk in zip(mu, p))
    tau1 = sum(pk * math.exp(-k) * k for k, pk in zip(mu, p))

    def s0_s1(counts):
        npl, nmi = bounds(counts)
        s0 = max(tau0 * (mu[1] * nmi[2] - mu[2] * npl[1]) / (mu[1] - mu[2]), 0.0)
        num = tau1 * mu[0] * (nmi[1] - npl[2] - (mu[1] ** 2 - mu[2] ** 2) / mu[0] ** 2 * (npl[0] - s0 / tau0))
        s1 = num / (mu[0] * (mu[1] - mu[2]) - mu[1] ** 2 + mu[2] ** 2)
        return s0, s1

    sX0, sX1 = s0_s1(nX_k)
    _, sZ1 = s0_s1(nZ_k)
    mpl, mmi = bounds(mZ_k)
    vZ1 = tau1 * (mpl[1] - mmi[2]) / (mu[1] - mu[2])
    single = 0.0
    if sX1 > 0 and sZ1 > 0:
        b = vZ1 / sZ1
        if 0 < b < 0.5:
            cc, dd = sZ1, sX1
            inner = (cc + dd) * (1 - b) * b / (cc * dd * math.log(2.0)) * \
                math.log2((cc + dd) / (cc * dd * (1 - b) * b) * 21.0 ** 2 / eps_sec ** 2)
            phi = min(b + math.sqrt(max(inner, 0.0)), 0.5)
        else:
            phi = 0.5
        single = sX1 * (1.0 - h2(phi))
    e_obs = sum(mX_k) / nX
    lam = nX * c["f_ec"] * h2(min(e_obs, 0.5))
    return sX0 + single - lam - 6.0 * math.log2(21.0 / eps_sec) - math.log2(2.0 / c["eps_cor"])


def random_interior(rng, n):
    return [(rng.uniform(0.3, 0.7), rng.uniform(0.05, 0.2), rng.uniform(0.1, 0.6), rng.uniform(0.1, 0.35),
             rng.uniform(0.5, 0.95)) for _ in range(n)]


class TestIndependentPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax.numpy as jnp
        from src.qkd_opt import LIM, evaluate, with_flags
        cls.jnp, cls.evaluate = jnp, staticmethod(evaluate)
        cls.LIM = LIM
        cls.FIXED = with_flags(LIM, "lim_fixed_eps", eps_mode="fixed")
        assert cls.FIXED.flags.gamma_mode == "printed" and cls.FIXED.flags.emis_eta == "sys"

    def test_fixed_eps_agreement(self):
        rng = np.random.default_rng(20261105)
        eps = 1e-10
        pre = self.FIXED._replace(consts=self.FIXED.consts._replace(eps_sec=eps))
        worst, n = 0.0, 0
        for prm in random_interior(rng, 30):
            for L in (0.0, 50.0, 100.0, 140.0):
                for nX in (1e4, 1e6, 1e9):
                    li = key_length_indep(*prm, L, nX, eps)
                    lj = float(self.evaluate(self.jnp.asarray(prm), L, nX, pre.consts, pre.flags)["l"])
                    rel = abs(li - lj) / abs(li)
                    worst = max(worst, rel)
                    n += 1
                    self.assertLessEqual(rel, 1e-10, (prm, L, nX, li, lj, rel))
        self.assertEqual(n, 360)
        print(f"\n[independent] fixed-eps max relative difference over {n} evaluations: {worst:.3e}")

    def test_eps_fixed_point_residual(self):
        """eps returned by the default lim preset satisfies eps = kappa * max(l_indep(eps), 1)."""
        rng = np.random.default_rng(7)
        worst = 0.0
        for prm in random_interior(rng, 20):
            for L, nX in ((0.0, 1e9), (50.0, 1e6), (100.0, 1e9), (20.0, 1e4)):
                out = self.evaluate(self.jnp.asarray(prm), L, nX, self.LIM.consts, self.LIM.flags)
                eps = float(out["eps_sec"])
                li = key_length_indep(*prm, L, nX, eps)
                res = abs(C["kappa"] * max(li, 1.0) - eps) / eps
                worst = max(worst, res)
                self.assertLessEqual(res, 1e-10, (prm, L, nX, eps, li))
        print(f"\n[independent] eps fixed-point max residual: {worst:.3e}")

    def test_error_budget_identities(self):
        """Supplementary: eps_sec = 2[2 a1 + a2 + a3] + nu_bar + 10 eps1 + 2 eps2; with every term = eps this is
        21 eps. The key-length constant -log2(2/(eps_cor beta)) with beta = (a2 a3 nu_bar)^2 then equals
        -6 log2(1/eps) - log2(2/eps_cor) = -6 log2(21/eps_sec) - log2(2/eps_cor)."""
        getcontext().prec = 80
        for eps_sec in ("1e-10", "1e-15", "3.7e-12", "2.1e-6"):
            es = Decimal(eps_sec)
            eps = es / 21
            a1 = a2 = a3 = nu_bar = eps1 = eps2 = eps
            total = 2 * (2 * a1 + a2 + a3) + nu_bar + 10 * eps1 + 2 * eps2
            self.assertEqual(total, 21 * eps)
            self.assertLess(abs(total - es) / es, Decimal("1e-70"))
            ln2 = Decimal(2).ln()
            beta = (a2 * a3 * nu_bar) ** 2
            lhs = 6 * (Decimal(21) / es).ln() / ln2
            rhs = 6 * (1 / eps).ln() / ln2
            self.assertLess(abs(lhs - rhs), Decimal("1e-60"))
            ecor = Decimal("1e-15")
            const_supp = -((2 / (ecor * beta)).ln() / ln2)
            const_eq1 = -lhs - (2 / ecor).ln() / ln2
            self.assertLess(abs(const_supp - const_eq1), Decimal("1e-60"))

    def test_float_vs_decimal_spot(self):
        """The float64 independent implementation itself vs a 50-digit Decimal evaluation of the same formulas."""
        getcontext().prec = 50
        D = Decimal

        def ldec(mu1, mu2, P1, P2, qx, L, nX, eps_sec):
            c = {k: D(repr(v)) for k, v in C.items()}
            mu = [D(repr(mu1)), D(repr(mu2)), c["mu3"]]
            p = [D(repr(P1)), D(repr(P2))]
            p.append(1 - p[0] - p[1])
            qx, L, nX, es = D(repr(qx)), D(repr(L)), D(repr(nX)), D(repr(eps_sec))
            eta_sys = c["eta_bob"] * (D(10) ** (-c["alpha"] * L / 10))
            Dk = [1 - (1 - 2 * c["p_dc"]) * (-eta_sys * k).exp() for k in mu]
            Rk = [d * (1 + c["p_ap"]) for d in Dk]
            e = [c["p_dc"] + c["e_mis"] * (1 - (-eta_sys * k).exp()) + c["p_ap"] * d / 2 for k, d in zip(mu, Dk)]
            qz = 1 - qx
            N = nX / (qx * qx * sum(a * b for a, b in zip(p, Rk)))
            nXk = [N * qx * qx * a * b for a, b in zip(p, Rk)]
            nZk = [N * qz * qz * a * b for a, b in zip(p, Rk)]
            mXk = [N * qx * qx * a * b for a, b in zip(p, e)]
            mZk = [N * qz * qz * a * b for a, b in zip(p, e)]
            lnf = (D(21) / es).ln()
            ln2 = D(2).ln()
            log2 = lambda x: x.ln() / ln2

            def bnd(cnt):
                dl = (sum(cnt) / 2 * lnf).sqrt()
                return ([k.exp() / a * (x + dl) for k, a, x in zip(mu, p, cnt)],
                        [k.exp() / a * (x - dl) for k, a, x in zip(mu, p, cnt)])
            t0 = sum(a * (-k).exp() for k, a in zip(mu, p))
            t1 = sum(a * (-k).exp() * k for k, a in zip(mu, p))

            def s01(cnt):
                pl, mi = bnd(cnt)
                s0 = max(t0 * (mu[1] * mi[2] - mu[2] * pl[1]) / (mu[1] - mu[2]), D(0))
                s1 = t1 * mu[0] * (mi[1] - pl[2] - (mu[1] ** 2 - mu[2] ** 2) / mu[0] ** 2 * (pl[0] - s0 / t0)) / \
                    (mu[0] * (mu[1] - mu[2]) - mu[1] ** 2 + mu[2] ** 2)
                return s0, s1
            sX0, sX1 = s01(nXk)
            _, sZ1 = s01(nZk)
            pl, mi = bnd(mZk)
            v = t1 * (pl[1] - mi[2]) / (mu[1] - mu[2])
            b = v / sZ1
            H = lambda x: -(x * log2(x) + (1 - x) * log2(1 - x))
            inner = (sZ1 + sX1) * (1 - b) * b / (sZ1 * sX1 * ln2) * log2((sZ1 + sX1) / (sZ1 * sX1 * (1 - b) * b) * 441 / es ** 2)
            phi = min(b + inner.sqrt(), D("0.5"))
            eo = sum(mXk) / nX
            return sX0 + sX1 * (1 - H(phi)) - nX * c["f_ec"] * H(eo) - 6 * log2(D(21) / es) - log2(2 / c["eps_cor"])

        rng = np.random.default_rng(3)
        for prm in random_interior(rng, 5):
            for L, nX in ((0.0, 1e9), (50.0, 1e6)):
                lf = key_length_indep(*prm, L, nX, 1e-10)
                ld = float(ldec(*prm, L, nX, 1e-10))
                self.assertLessEqual(abs(lf - ld) / abs(ld), 1e-10, (prm, L, nX, lf, ld))


if __name__ == "__main__":
    unittest.main()
