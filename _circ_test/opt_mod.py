# Stand-in for spline_optimize_3. The warm-start import sits mid-body, exactly
# like the real 'if USE_DOF_GUESS: from unpack_dofs import load_dof_guess'.
USE_DOF_GUESS = True   # flip to test both cases

print("opt_mod body start")

if USE_DOF_GUESS:
    from unpack_dofs import load_dof_guess   # <-- the circular reach
    print("opt_mod got:", load_dof_guess("dummy", 5))

print("opt_mod body end")

if __name__ == "__main__":
    print("opt_mod entry ran")
