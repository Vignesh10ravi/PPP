"""
PyTorch pICNN ↔ NGSolve bridge — Total-Lagrange hyperelastic UMAT.

Provides:
  - stress_and_tangent(model, F_batch, context) → (P, A) via autograd
  - solve(mesh, model, bc_cf, dirichlet, order, context)
    → (gfu, residual_history)
"""
import os

import numpy as np
import torch
import ngsolve as ngs
ngs.SetNumThreads(max(1, int(os.environ.get("NGS_NUM_THREADS", "1"))))
from ngsolve import (
    VectorH1, BilinearForm, LinearForm, GridFunction,
    Mesh, CF, specialcf, dx, BND, InnerProduct, Integrate
)
from ngsolve.comp import IntegrationRuleSpace


def _stage_log(message: str) -> None:
    """Emit optional assembly-stage markers for native crash diagnosis."""
    if os.environ.get("PPP_UMAT_STAGE_LOG") == "1":
        print(f"[UMAT stage] {message}", flush=True)

# ── autograd stress + tangent ──────────────────────────────────────────────────

def _F_to_features(F: torch.Tensor) -> torch.Tensor:
    """Compute pICNN input features [I1, I2, J, -ln J] from deformation gradient F.
    F : (batch, 3, 3)
    """
    C = F.transpose(-1, -2) @ F                          # right Cauchy-Green
    I1 = torch.diagonal(C, dim1=-1, dim2=-2).sum(-1)     # tr C
    # I2 = 0.5*(tr(C)^2 - tr(C^2))
    C2 = C @ C
    I2 = 0.5 * (I1**2 - torch.diagonal(C2, dim1=-1, dim2=-2).sum(-1))
    J  = torch.linalg.det(F)
    lnJ = torch.log(J.clamp(min=1e-12))
    return torch.stack([I1, I2, J, -lnJ], dim=-1)        # (batch, 4)


def _context_batch(model, context, batch_size, dtype, device):
    """Broadcast a material context such as volume fraction over quadrature points."""
    n_context = int(getattr(model, "n_ctx", 0))
    if n_context == 0:
        if context is not None:
            raise ValueError("context supplied to a model with n_ctx=0")
        return None
    if context is None:
        raise ValueError(
            f"this model requires {n_context} context value(s); "
            "pass the material context to the FE solve"
        )

    values = torch.as_tensor(context, dtype=dtype, device=device)
    if values.ndim == 0:
        if n_context != 1:
            raise ValueError(
                f"a scalar context is valid only for n_ctx=1, got n_ctx={n_context}"
            )
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        if values.numel() == n_context:
            values = values.reshape(1, n_context)
        else:
            raise ValueError(
                f"context must contain {n_context} value(s); "
                f"got shape {tuple(values.shape)}"
            )
    elif values.ndim != 2 or values.shape != (1, n_context):
        raise ValueError(
            f"the FE bridge expects a uniform context with shape "
            f"({n_context},) or (1, {n_context}); got shape {tuple(values.shape)}"
        )
    return values.expand(batch_size, n_context)


def stress_and_tangent(
    model,
    F_batch: torch.Tensor,
    context=None,
    return_energy: bool = False,
):
    """
    Compute 1st PK stress P and material tangent dP/dF.

    Parameters
    ----------
    model    : HyperelasticNN (nn.Module), double precision
    F_batch  : (N, 3, 3) deformation gradients, requires_grad False
    context  : optional material context; a scalar volume fraction is
               broadcast over the batch for partial ICNNs

    Returns
    -------
    P : (N, 3, 3) first Piola-Kirchhoff stress
    A : (N, 9, 9) tangent  dP_iA / dF_jB  (Voigt: row = iA, col = jB,
                           both in C-order row-major flattening)
    """
    N = F_batch.shape[0]
    F = F_batch.detach().requires_grad_(True)             # (N,3,3)
    c = _context_batch(model, context, N, F.dtype, F.device)
    reference_context = None if c is None else c[:1]
    reference = model.reference(F.dtype, F.device, reference_context)
    W = model.energy(F, c, ref=reference).squeeze(-1)      # (N,) energy density

    # P = dW/dF  via autograd
    P_flat = torch.autograd.grad(
        W.sum(), F, create_graph=True
    )[0]                                                  # (N,3,3)

    # A = dP/dF  — loop over 9 components of P
    P_flat_vec = P_flat.reshape(N, 9)
    rows = []
    for idx in range(9):
        grad_row = torch.autograd.grad(
            P_flat_vec[:, idx].sum(), F,
            retain_graph=(idx < 8)
        )[0].reshape(N, 9)                               # (N,9)
        rows.append(grad_row)

    A = torch.stack(rows, dim=1)                          # (N,9,9)  A[n,i,j] = d P_i / d F_j
    if return_energy:
        return P_flat.detach(), A.detach(), W.detach()
    return P_flat.detach(), A.detach()


# ── Newton-Raphson solve ───────────────────────────────────────────────────────

def solve(mesh: Mesh, model, bc_cf: CF,
          dirichlet: str = "left|right",
          order: int = 1,
          max_iter: int = 20,
          tol: float = 1e-10,
          context=None,
          initial_cf: CF | None = None,
          diagnose_tangent: bool = False,
          diagnose_tangent_only: bool = False,
          diagnose_tangent_iteration: int = 0,
          integration_order: int | None = None,
          merit_function: str = "residual"):
    """
    Total-Lagrange Newton solver for a hyperelastic body.

    Returns
    -------
    gfu     : GridFunction with converged displacement
    history : list of residual norms

    `context` is passed to the constitutive model at every integration point.
    For the RVE-trained partial ICNN, pass its particle volume fraction, e.g.
    `context=0.05`.

    `merit_function="energy"` globalizes Newton using the discrete
    hyperelastic potential energy while convergence is still checked by the
    free residual norm.

    `initial_cf`, when supplied, initializes the displacement throughout the
    mesh before the prescribed boundary values are applied. An affine loading
    field is a useful initial guess for finite-strain Newton solves.
    """
    # ── FE space ──────────────────────────────────────────────────────────────
    fes = VectorH1(mesh, order=order, dirichlet=dirichlet)
    ndof = fes.ndof

    # Diagnostic: confirm boundary names exist
    bnd_names = set(mesh.GetBoundaries())
    requested = set(dirichlet.replace("|", " ").split())
    missing = requested - bnd_names
    if missing:
        raise ValueError(
            f"Dirichlet boundaries {missing} not found in mesh.\n"
            f"Available boundaries: {bnd_names}\n"
            f"Check OCC face naming (box.faces.Min(X).name = 'left')."
        )

    n_free = sum(fes.FreeDofs())
    if n_free == 0:
        raise ValueError(
            "All DOFs are constrained — nothing to solve. "
            "Check that only some faces are in the Dirichlet set."
        )
    if n_free == ndof:
        raise ValueError(
            "No DOFs are constrained (FreeDofs == all DOFs). "
            "Dirichlet condition not applied — check boundary name string."
        )
    print(f"[solve] ndof={ndof}, free={n_free}, "
          f"constrained={ndof - n_free}")

    # ── quadrature space for pointwise F/P/A ─────────────────────────────────
    int_order = 2 * order if integration_order is None else int(integration_order)
    if int_order < 2 * order:
        raise ValueError(
            f"integration_order={int_order} is too low for displacement order "
            f"{order}; use at least {2 * order}"
        )
    if merit_function not in {"residual", "energy"}:
        raise ValueError("merit_function must be 'residual' or 'energy'")

    # ── grid functions ────────────────────────────────────────────────────────
    gfu = GridFunction(fes)                               # displacement u
    initial_guess_source = "zero"
    if initial_cf is not None:
        if isinstance(initial_cf, GridFunction):
            source_values = initial_cf.vec.FV().NumPy()
            target_values = gfu.vec.FV().NumPy()
            if source_values.shape != target_values.shape:
                raise ValueError(
                    "initial GridFunction has an incompatible DOF layout: "
                    f"source shape={source_values.shape}, "
                    f"target shape={target_values.shape}"
                )
            # Copy coefficients directly between the matching FE spaces to
            # preserve the previous converged state across continuation steps.
            target_values[:] = source_values
            initial_guess_source = "previous_gridfunction_vector_copy"
        else:
            gfu.Set(initial_cf)
            initial_guess_source = "coefficient_function_interpolation"
    # Apply Dirichlet BC: project bc_cf onto constrained DOFs
    gfu.Set(bc_cf, definedon=mesh.Boundaries(dirichlet))
    initial_displacement_norm = float(
        np.linalg.norm(gfu.vec.FV().NumPy())
    )
    print(
        f"[solve] initial_guess={initial_guess_source}, "
        f"displacement_norm={initial_displacement_norm:.6e}",
        flush=True,
    )

    def _residual_and_stiffness(gfu_current):
        """Assemble global residual R and stiffness K on free DOFs."""

        _stage_log("begin residual/tangent evaluation")

        # ------------------------------------------------------------
        # One scalar integration-rule space.
        # All F/P/A data use exactly these same integration points.
        # ------------------------------------------------------------
        _stage_log(f"before IntegrationRuleSpace(order={int_order})")
        irs = IntegrationRuleSpace(mesh, order=int_order)
        _stage_log("after IntegrationRuleSpace")
        irs_dx = dx(intrules=irs.GetIntegrationRules())
        _stage_log("after integration measure setup")

        # ------------------------------------------------------------
        # Deformation gradient
        # ------------------------------------------------------------
        gradu = ngs.Grad(gfu_current)
        F_cf = ngs.Id(3) + gradu

        # 9 scalar values at every integration point
        fes_F = irs ** 9
        gf_F = GridFunction(fes_F)

        _stage_log("before deformation-gradient interpolation")
        gf_F.Interpolate(F_cf.Reshape((9,)))
        _stage_log("after deformation-gradient interpolation")

        # NGSolve vector spaces are safest to read component-wise.
        F_components = [
            gf_F.components[i].vec.FV().NumPy().copy()
            for i in range(9)
        ]

        # Convert component-major -> point-major
        F_np = np.stack(F_components, axis=1)
        F_np = F_np.reshape(-1, 3, 3)
        _stage_log(f"after reading F at {F_np.shape[0]} quadrature points")

        if not np.isfinite(F_np).all():
            raise FloatingPointError("non-finite deformation gradient at quadrature points")
        jacobian_values = np.linalg.det(F_np)
        if not np.isfinite(jacobian_values).all():
            raise FloatingPointError("non-finite Jacobian at quadrature points")
        minimum_jacobian = float(np.min(jacobian_values))
        if minimum_jacobian <= 1.0e-8:
            raise FloatingPointError(
                "inadmissible quadrature Jacobian: "
                f"minimum J={minimum_jacobian:.6e}"
            )

        # ------------------------------------------------------------
        # PyTorch material evaluation
        # ------------------------------------------------------------
        F_t = torch.from_numpy(F_np).to(
            dtype=torch.float64,
            device="cpu"
        )


        # Do NOT use no_grad here: tangent needs autograd
        _stage_log("before PyTorch stress/tangent evaluation")
        P_t, A_t, W_t = stress_and_tangent(
            model,
            F_t,
            context=context,
            return_energy=True,
        )
        _stage_log("after PyTorch stress/tangent evaluation")
        if not torch.isfinite(P_t).all() or not torch.isfinite(A_t).all():
            raise FloatingPointError("non-finite constitutive stress or tangent")

        P_np = P_t.detach().cpu().numpy()
        A_np = A_t.detach().cpu().numpy()
        total_energy = None
        if merit_function == "energy":
            _stage_log("before energy GridFunction/integration")
            gf_W = GridFunction(irs)
            gf_W.vec.FV().NumPy()[:] = (
                W_t.detach().cpu().numpy().reshape(-1)
            )
            total_energy = float(Integrate(gf_W * irs_dx, mesh))
            _stage_log("after energy integration")

        # ------------------------------------------------------------
        # Put P back into an IR GridFunction
        # ------------------------------------------------------------
        fes_P = irs ** 9
        gf_P = GridFunction(fes_P)
        _stage_log("created stress quadrature GridFunction")

        P_flat = P_np.reshape(-1, 9)

        for i in range(9):
            gf_P.components[i].vec.FV().NumPy()[:] = P_flat[:, i]

        # ------------------------------------------------------------
        # Put A back into an IR GridFunction
        # ------------------------------------------------------------
        fes_A = irs ** 81
        gf_A = GridFunction(fes_A)
        _stage_log("created tangent quadrature GridFunction")

        A_flat = A_np.reshape(-1, 81)

        for i in range(81):
            gf_A.components[i].vec.FV().NumPy()[:] = A_flat[:, i]

        # ------------------------------------------------------------
        # Global FE trial/test functions
        # ------------------------------------------------------------
        u = fes.TrialFunction()
        v = fes.TestFunction()

        # ------------------------------------------------------------
        # Residual
        # ------------------------------------------------------------
        P_cf = gf_P.Reshape((3, 3))

        lf = LinearForm(fes)
        lf += InnerProduct(
            P_cf,
            ngs.Grad(v)
        ) * irs_dx

        # Assemble NGSolve forms in a shared-memory task region. The thread
        # count is controlled by NGS_NUM_THREADS and should match the PBS
        # ncpus allocation for this job.
        with ngs.TaskManager():
            _stage_log("before residual LinearForm.Assemble")
            lf.Assemble()
        _stage_log("after residual LinearForm.Assemble")

        R_full = lf.vec.CreateVector()
        R_full.data = lf.vec

        # ------------------------------------------------------------
        # Consistent tangent / stiffness
        # ------------------------------------------------------------
        A_cf = gf_A.Reshape((9, 9))

        bf = BilinearForm(fes)

        gradu_trial = ngs.Grad(u).Reshape((9,))
        gradv_test = ngs.Grad(v).Reshape((9,))

        bf += InnerProduct(
            gradv_test,
            A_cf * gradu_trial
        ) * irs_dx

        with ngs.TaskManager():
            _stage_log("before tangent BilinearForm.Assemble")
            bf.Assemble()
        _stage_log("after tangent BilinearForm.Assemble")

        return R_full, bf.mat, minimum_jacobian, total_energy

    # ── Newton loop ───────────────────────────────────────────────────────────
    history = []
    fd = fes.FreeDofs()

    def _free_mask_for_vector(vector_length):
        scalar_free_mask = np.asarray(
            [bool(fd[dof]) for dof in range(len(fd))],
            dtype=bool,
        )
        if vector_length % len(scalar_free_mask) != 0:
            raise ValueError(
                "vector length is not an integer multiple of the free-DOF mask"
            )
        return np.tile(
            scalar_free_mask,
            vector_length // len(scalar_free_mask),
        )

    def _free_residual_norm(residual):
        residual_values = residual.FV().NumPy().copy()
        free_mask = _free_mask_for_vector(len(residual_values))
        return float(np.linalg.norm(residual_values[free_mask]))

    def _free_tangent_spectrum(stiffness):
        """Return eigenvalue diagnostics for the symmetric free-DOF tangent."""
        rows, cols, matrix_values = stiffness.COO()
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        matrix_values = np.asarray(matrix_values, dtype=np.float64)
        matrix_dense = np.zeros((len(fd), len(fd)), dtype=np.float64)
        np.add.at(matrix_dense, (rows, cols), matrix_values)
        free_dofs = np.asarray(
            [bool(fd[dof]) for dof in range(len(fd))], dtype=bool
        )
        matrix_free = matrix_dense[np.ix_(free_dofs, free_dofs)]
        matrix_free_sym = 0.5 * (matrix_free + matrix_free.T)
        eigenvalues = np.linalg.eigvalsh(matrix_free_sym)
        abs_eigenvalues = np.abs(eigenvalues)
        largest_abs_eigenvalue = float(np.max(abs_eigenvalues))
        smallest_abs_eigenvalue = float(np.min(abs_eigenvalues))
        zero_threshold = max(
            np.finfo(float).eps * largest_abs_eigenvalue,
            np.finfo(float).tiny,
        )
        return {
            "free_dofs": int(matrix_free.shape[0]),
            "relative_symmetry_error": float(
                np.linalg.norm(matrix_free - matrix_free.T)
                / max(np.linalg.norm(matrix_free), np.finfo(float).eps)
            ),
            "minimum_eigenvalue_symmetric_part": float(eigenvalues[0]),
            "maximum_eigenvalue_symmetric_part": float(eigenvalues[-1]),
            "negative_eigenvalue_count": int(np.count_nonzero(eigenvalues < 0.0)),
            "minimum_absolute_eigenvalue": smallest_abs_eigenvalue,
            "maximum_absolute_eigenvalue": largest_abs_eigenvalue,
            "absolute_condition_estimate": (
                largest_abs_eigenvalue / smallest_abs_eigenvalue
                if smallest_abs_eigenvalue > zero_threshold
                else None
            ),
        }

    armijo_constant = 1.0e-4

    def _merit_accepts(
        trial_residual,
        trial_energy,
        step,
        directional_slope,
        base_residual,
        base_energy,
    ):
        if not np.isfinite(trial_residual):
            return False
        if merit_function == "residual":
            return trial_residual < base_residual
        if (
            trial_energy is None
            or not np.isfinite(trial_energy)
            or directional_slope >= 0.0
        ):
            return False
        roundoff = 100.0 * np.finfo(float).eps * max(
            abs(base_energy), abs(step * directional_slope), 1.0
        )
        armijo_bound = base_energy + armijo_constant * step * directional_slope
        return trial_energy <= armijo_bound + roundoff

    def _trial_is_better(
        trial_residual,
        trial_energy,
        best_residual,
        best_energy,
    ):
        if merit_function == "residual":
            return trial_residual < best_residual
        return (
            trial_energy is not None
            and np.isfinite(trial_energy)
            and trial_energy < best_energy
        )

    for iteration in range(max_iter):
        R, K, minimum_jacobian, state_energy = _residual_and_stiffness(gfu)
        res_norm_free = _free_residual_norm(R)
        if not np.isfinite(res_norm_free):
            raise FloatingPointError("non-finite free residual in macro-FE solve")
        history.append(res_norm_free)
        print(
            f"  iter {iteration:2d}  |R_free| = {res_norm_free:.3e}"
            f"  min(J_q) = {minimum_jacobian:.6e}"
            + (
                f"  energy = {state_energy:.9e}"
                if merit_function == "energy"
                else ""
            )
        )

        if res_norm_free < tol:
            print(f"[solve] Converged in {iteration} iterations.")
            if (
                diagnose_tangent
                and diagnose_tangent_only
                and diagnose_tangent_iteration < 0
            ):
                return gfu, history, {
                    "state": "converged_equilibrium",
                    "free_residual_norm": res_norm_free,
                    "global_tangent_spectrum": _free_tangent_spectrum(K),
                }
            break

        # Solve K_ff * du = -R_f
        inv_K = K.Inverse(freedofs=fd, inverse="umfpack")
        du = gfu.vec.CreateVector()
        du.data = -inv_K * R

        if diagnose_tangent and iteration == diagnose_tangent_iteration:
            # Compare the assembled stiffness action against a centered finite
            # difference of the assembled residual along the Newton direction.
            solution_snapshot = gfu.vec.CreateVector()
            solution_snapshot.data = gfu.vec
            du_values = du.FV().NumPy().copy()
            epsilon = 1.0e-5 / max(float(np.max(np.abs(du_values))), 1.0)
            try:
                gfu.vec.data = solution_snapshot
                gfu.vec.data += epsilon * du
                R_plus, _, _, _ = _residual_and_stiffness(gfu)
                gfu.vec.data = solution_snapshot
                gfu.vec.data -= epsilon * du
                R_minus, _, _, _ = _residual_and_stiffness(gfu)
            finally:
                gfu.vec.data = solution_snapshot

            finite_difference = (
                R_plus.FV().NumPy().copy() - R_minus.FV().NumPy().copy()
            ) / (2.0 * epsilon)
            tangent_action = gfu.vec.CreateVector()
            tangent_action.data = K * du
            assembled_action = tangent_action.FV().NumPy().copy()
            free_mask = _free_mask_for_vector(len(assembled_action))
            base_residual = R.FV().NumPy().copy()
            free_residual_norm = float(
                np.linalg.norm(base_residual[free_mask])
            )
            denominator = max(
                float(np.linalg.norm(assembled_action[free_mask])),
                np.finfo(float).eps,
            )
            relative_error = float(
                np.linalg.norm(
                    finite_difference[free_mask] - assembled_action[free_mask]
                ) / denominator
            )
            print(
                "[solve tangent check] "
                f"epsilon={epsilon:.6e}, "
                f"relative_directional_error={relative_error:.6e}, "
                f"||R||_free={free_residual_norm:.6e}, "
                f"||Kdu||_free={denominator:.6e}, "
                f"linear_solve_relative_error="
                f"{np.linalg.norm((assembled_action + base_residual)[free_mask]) / max(free_residual_norm, np.finfo(float).eps):.6e}",
                flush=True,
            )
            tangent_diagnostic = {
                "epsilon": epsilon,
                "relative_directional_error": relative_error,
                "free_residual_norm": free_residual_norm,
                "free_tangent_action_norm": denominator,
                "newton_linear_solve_relative_error": float(
                    np.linalg.norm(
                        (assembled_action + base_residual)[free_mask]
                    )
                    / max(free_residual_norm, np.finfo(float).eps)
                ),
            }

            tangent_diagnostic["global_tangent_spectrum"] = (
                _free_tangent_spectrum(K)
            )
            if diagnose_tangent_only:
                return gfu, history, tangent_diagnostic

        # Backtrack the Newton update. The pICNN contains log(J), so every
        # trial must remain orientation-preserving and reduce the free residual.
        current_solution = gfu.vec.CreateVector()
        current_solution.data = gfu.vec
        step_length = 1.0
        accepted = False
        last_trial_residual = float("inf")
        last_trial_energy = float("inf")
        stagnation_candidate_solution = None
        stagnation_candidate_residual = float("inf")
        stagnation_candidate_energy = float("inf")
        minimum_newton_step = 1.0 / 1024.0
        stagnation_window = 6
        stagnation_minimum_reduction = 0.10
        stagnation_step_threshold = 1.0 / 8.0
        recent_reduction = float("inf")
        if len(history) >= stagnation_window:
            recent_reduction = 1.0 - (
                res_norm_free / history[-stagnation_window]
            )
        stagnated = (
            len(history) >= stagnation_window
            and res_norm_free > max(100.0 * tol, 1.0e-6)
            and recent_reduction < stagnation_minimum_reduction
        )
        residual_values_for_slope = R.FV().NumPy().copy()
        direction_values_for_slope = du.FV().NumPy().copy()
        free_mask_for_slope = _free_mask_for_vector(
            len(residual_values_for_slope)
        )
        newton_slope = float(
            np.dot(
                residual_values_for_slope[free_mask_for_slope],
                direction_values_for_slope[free_mask_for_slope],
            )
        )
        line_search_iterations = (
            range(24)
            if merit_function == "residual" or newton_slope < 0.0
            else range(0)
        )
        for backtrack in line_search_iterations:
            gfu.vec.data = current_solution
            gfu.vec.data += step_length * du
            try:
                (
                    R_trial,
                    _,
                    trial_minimum_jacobian,
                    trial_energy,
                ) = _residual_and_stiffness(gfu)
                last_trial_residual = _free_residual_norm(R_trial)
                last_trial_energy = (
                    float(trial_energy)
                    if trial_energy is not None
                    else float("inf")
                )
            except FloatingPointError as exc:
                print(
                    f"[solve line search] iteration={iteration}, "
                    f"step={step_length:.6e}, rejected: {exc}"
                )
                last_trial_residual = float("inf")
                last_trial_energy = float("inf")
            if _merit_accepts(
                last_trial_residual,
                last_trial_energy,
                step_length,
                newton_slope,
                res_norm_free,
                state_energy,
            ):
                if (
                    res_norm_free > max(100.0 * tol, 1.0e-6)
                    and (
                        step_length < minimum_newton_step
                        or (
                            stagnated
                            and step_length <= stagnation_step_threshold
                        )
                    )
                ):
                    stagnation_candidate_solution = gfu.vec.CreateVector()
                    stagnation_candidate_solution.data = gfu.vec
                    stagnation_candidate_residual = last_trial_residual
                    stagnation_candidate_energy = last_trial_energy
                    # Switch directions when a tiny step or persistent slow
                    # progress shows that ordinary damped Newton is stalling.
                    if stagnated and step_length <= stagnation_step_threshold:
                        print(
                            "[solve line search] "
                            f"iteration={iteration}, step={step_length:.6e}, "
                            f"residual={last_trial_residual:.6e}, "
                            f"reduction_over_{stagnation_window - 1}_updates="
                            f"{recent_reduction:.3%}; trying shifted Newton"
                            + (
                                f", energy={last_trial_energy:.9e}"
                                if merit_function == "energy"
                                else ""
                            ),
                            flush=True,
                        )
                    else:
                        print(
                            "[solve line search] "
                            f"iteration={iteration}, step={step_length:.6e}, "
                            f"residual={last_trial_residual:.6e}; "
                            "step too small, trying shifted Newton",
                            flush=True,
                        )
                    break
                accepted = True
                print(
                    f"[solve line search] iteration={iteration}, "
                    f"step={step_length:.6e}, "
                    f"residual={last_trial_residual:.6e}, "
                    f"min(J_q)={trial_minimum_jacobian:.6e}, "
                    f"backtracks={backtrack}"
                    + (
                        f", energy={last_trial_energy:.9e}"
                        if merit_function == "energy"
                        else ""
                    )
                )
                break
            step_length *= 0.5

        if not accepted:
            gfu.vec.data = current_solution
            # First try a spectrally shifted Newton matrix. The shift makes
            # the symmetric free-DOF tangent positive definite while keeping
            # the update close to Newton's direction.
            residual_values = R.FV().NumPy().copy()
            free_mask = _free_mask_for_vector(len(residual_values))
            residual_values[~free_mask] = 0.0

            rows, cols, matrix_values = K.COO()
            rows = np.asarray(rows, dtype=np.int64)
            cols = np.asarray(cols, dtype=np.int64)
            matrix_values = np.asarray(matrix_values, dtype=np.float64)
            K_dense = np.zeros((len(fd), len(fd)), dtype=np.float64)
            np.add.at(K_dense, (rows, cols), matrix_values)
            K_free = K_dense[np.ix_(free_mask, free_mask)]
            K_free_sym = 0.5 * (K_free + K_free.T)
            tangent_eigenvalues = np.linalg.eigvalsh(K_free_sym)
            tangent_scale = max(
                float(np.max(np.abs(tangent_eigenvalues))), 1.0
            )
            # Keep the spectral floor small enough to avoid turning an
            # ill-conditioned Newton step into very slow gradient descent.
            # Increase the shift adaptively only when globalization requires it.
            relative_shift_floor = 1.0e-6
            spectral_shift_floor = relative_shift_floor * tangent_scale
            base_shift = max(
                spectral_shift_floor,
                -float(tangent_eigenvalues[0]) + spectral_shift_floor,
            )
            regularized_accepted = False

            for shift_factor in (1.0, 10.0, 100.0, 1000.0):
                shift = base_shift * shift_factor
                try:
                    regularized_free_direction = np.linalg.solve(
                        K_free_sym + shift * np.eye(K_free_sym.shape[0]),
                        -residual_values[free_mask],
                    )
                except np.linalg.LinAlgError:
                    continue

                regularized_values = np.zeros_like(residual_values)
                regularized_values[free_mask] = regularized_free_direction
                regularized_slope = float(
                    np.dot(
                        residual_values[free_mask],
                        regularized_free_direction,
                    )
                )
                regularized_direction = gfu.vec.CreateVector()
                regularized_direction.FV().NumPy()[:] = regularized_values
                regularized_step = 1.0

                for backtrack in range(24):
                    gfu.vec.data = current_solution
                    gfu.vec.data += regularized_step * regularized_direction
                    try:
                        (
                            R_trial,
                            _,
                            trial_minimum_jacobian,
                            trial_energy,
                        ) = (
                            _residual_and_stiffness(gfu)
                        )
                        regularized_residual = _free_residual_norm(R_trial)
                        regularized_energy = (
                            float(trial_energy)
                            if trial_energy is not None
                            else float("inf")
                        )
                    except FloatingPointError as exc:
                        print(
                            "[solve shifted-Newton line search] "
                            f"iteration={iteration}, shift={shift:.6e}, "
                            f"step={regularized_step:.6e}, rejected: {exc}",
                            flush=True,
                        )
                        regularized_residual = float("inf")
                        regularized_energy = float("inf")

                    if _merit_accepts(
                        regularized_residual,
                        regularized_energy,
                        regularized_step,
                        regularized_slope,
                        res_norm_free,
                        state_energy,
                    ):
                        accepted = True
                        regularized_accepted = True
                        print(
                            "[solve shifted-Newton fallback] "
                            f"iteration={iteration}, shift={shift:.6e}, "
                            f"step={regularized_step:.6e}, "
                            f"residual={regularized_residual:.6e}, "
                            f"min(J_q)={trial_minimum_jacobian:.6e}, "
                            f"backtracks={backtrack}"
                            + (
                                f", energy={regularized_energy:.9e}"
                                if merit_function == "energy"
                                else ""
                            ),
                            flush=True,
                        )
                        break
                    regularized_step *= 0.5

                if regularized_accepted:
                    break

            if regularized_accepted:
                if (
                    stagnation_candidate_solution is not None
                    and not _trial_is_better(
                        regularized_residual,
                        regularized_energy,
                        stagnation_candidate_residual,
                        stagnation_candidate_energy,
                    )
                ):
                    gfu.vec.data = stagnation_candidate_solution
                    print(
                        "[solve fallback] retaining the damped-Newton trial "
                        f"with residual={stagnation_candidate_residual:.6e}; "
                        f"shifted-Newton residual={regularized_residual:.6e}"
                        + (
                            f", energy={stagnation_candidate_energy:.9e}"
                            f" vs {regularized_energy:.9e}"
                            if merit_function == "energy"
                            else ""
                        ),
                        flush=True,
                    )
                continue

            gfu.vec.data = current_solution
            # A Newton direction may fail globalization away from equilibrium.
            # For energy globalization, use the exact energy-gradient direction
            # -R. Otherwise retain the residual-norm gradient direction -K R.
            residual_free = gfu.vec.CreateVector()
            residual_free.FV().NumPy()[:] = residual_values

            if merit_function == "energy":
                fallback_direction_values = -residual_values
                fallback_slope = -float(
                    np.dot(
                        residual_values[free_mask],
                        residual_values[free_mask],
                    )
                )
                fallback_step = 1.0 / tangent_scale
            else:
                gradient = gfu.vec.CreateVector()
                gradient.data = K * residual_free
                gradient_values = gradient.FV().NumPy().copy()
                gradient_values[~free_mask] = 0.0

                tangent_gradient = gfu.vec.CreateVector()
                tangent_gradient.data = K * gradient
                tangent_gradient_values = tangent_gradient.FV().NumPy().copy()
                gradient_norm_sq = float(
                    np.dot(gradient_values[free_mask], gradient_values[free_mask])
                )
                tangent_gradient_norm_sq = float(
                    np.dot(
                        tangent_gradient_values[free_mask],
                        tangent_gradient_values[free_mask],
                    )
                )
                fallback_direction_values = -gradient_values
                fallback_slope = float("nan")
                fallback_step = (
                    gradient_norm_sq / tangent_gradient_norm_sq
                    if tangent_gradient_norm_sq > np.finfo(float).eps
                    else float("nan")
                )
            fallback_accepted = False
            fallback_last_residual = float("inf")
            fallback_last_energy = float("inf")

            fallback_is_defined = (
                np.isfinite(fallback_slope) and fallback_slope < 0.0
                if merit_function == "energy"
                else (
                    np.isfinite(gradient_norm_sq)
                    and np.isfinite(tangent_gradient_norm_sq)
                    and gradient_norm_sq > np.finfo(float).eps
                    and tangent_gradient_norm_sq > np.finfo(float).eps
                )
            )
            if fallback_is_defined:
                fallback_direction = gfu.vec.CreateVector()
                fallback_direction.FV().NumPy()[:] = fallback_direction_values

                for backtrack in range(32):
                    gfu.vec.data = current_solution
                    gfu.vec.data += fallback_step * fallback_direction
                    try:
                        (
                            R_trial,
                            _,
                            trial_minimum_jacobian,
                            trial_energy,
                        ) = (
                            _residual_and_stiffness(gfu)
                        )
                        fallback_last_residual = _free_residual_norm(R_trial)
                        fallback_last_energy = (
                            float(trial_energy)
                            if trial_energy is not None
                            else float("inf")
                        )
                    except FloatingPointError as exc:
                        print(
                            f"[solve {merit_function}-gradient line search] "
                            f"iteration={iteration}, step={fallback_step:.6e}, "
                            f"rejected: {exc}", flush=True
                        )
                        fallback_last_residual = float("inf")
                        fallback_last_energy = float("inf")

                    if _merit_accepts(
                        fallback_last_residual,
                        fallback_last_energy,
                        fallback_step,
                        fallback_slope,
                        res_norm_free,
                        state_energy,
                    ):
                        fallback_accepted = True
                        print(
                            f"[solve {merit_function}-gradient fallback] "
                            f"iteration={iteration}, "
                            f"step={fallback_step:.6e}, "
                            f"residual={fallback_last_residual:.6e}, "
                            f"min(J_q)={trial_minimum_jacobian:.6e}, "
                            f"backtracks={backtrack}"
                            + (
                                f", energy={fallback_last_energy:.9e}"
                                if merit_function == "energy"
                                else ""
                            ),
                            flush=True,
                        )
                        break
                    fallback_step *= 0.5

            if not fallback_accepted:
                if stagnation_candidate_solution is not None:
                    gfu.vec.data = stagnation_candidate_solution
                    print(
                        f"[solve fallback] shifted Newton and "
                        f"{merit_function}-gradient did not improve on the "
                        "saved damped-Newton trial; "
                        f"retaining residual={stagnation_candidate_residual:.6e}",
                        flush=True,
                    )
                    continue
                gfu.vec.data = current_solution
                raise RuntimeError(
                    f"macro-FE Newton and {merit_function}-gradient "
                    "globalization could not find a finite, "
                    "orientation-preserving acceptable step at "
                    f"iteration {iteration}; residual={res_norm_free:.6e}, "
                    f"last Newton trial={last_trial_residual:.6e}, "
                    f"last fallback trial={fallback_last_residual:.6e}"
                )
            if (
                stagnation_candidate_solution is not None
                and not _trial_is_better(
                    fallback_last_residual,
                    fallback_last_energy,
                    stagnation_candidate_residual,
                    stagnation_candidate_energy,
                )
            ):
                gfu.vec.data = stagnation_candidate_solution
                print(
                    "[solve fallback] retaining the damped-Newton trial "
                    f"with residual={stagnation_candidate_residual:.6e}; "
                    f"fallback residual={fallback_last_residual:.6e}"
                    + (
                        f", energy={stagnation_candidate_energy:.9e}"
                        f" vs {fallback_last_energy:.9e}"
                        if merit_function == "energy"
                        else ""
                    ),
                    flush=True,
                )
    else:
        # The final line-search update is applied at the end of the last loop
        # pass. Evaluate that updated state before reporting its residual.
        R_final, _, minimum_jacobian, _ = _residual_and_stiffness(gfu)
        final_residual = _free_residual_norm(R_final)
        history.append(final_residual)
        print(
            f"[solve] final state after {max_iter} updates: "
            f"|R_free|={final_residual:.3e}, "
            f"min(J_q)={minimum_jacobian:.6e}"
        )
        if final_residual < tol:
            print(f"[solve] Converged after {max_iter} updates.")
        else:
            print(f"[solve] Warning: did not converge in {max_iter} iterations.")

    return gfu, history
