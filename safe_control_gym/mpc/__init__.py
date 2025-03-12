from safe_control_gym.utils.registration import register

register(
    idx="mpc_acados",
    entry_point="safe_control_gym.mpc.mpc_acados:MPC_ACADOS",
    config_entry_point="safe_control_gym.mpc:mpc_acados.yaml",
)

register(
    idx="gpmpc_acados_TRP",
    entry_point="safe_control_gym.mpc.gpmpc_acados_TRP:GPMPC_ACADOS_TRP",
    config_entry_point="safe_control_gym.mpc:gpmpc_acados_TRP.yaml",
)
