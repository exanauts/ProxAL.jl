# Formulation
`ProxAL` is designed to solve AC Optimal Power Flow (ACOPF) formulations over multiple time periods. 


## Time coupling

Each time period ``t \in T`` involves the solution of an ACOPF with active ``p_{dt}`` and reactive ``q_{dt}`` load forecasts, which may differ from one time period to the next. In each time period ``t \in T``, we must determine the 'base-case' active power generation level of generator ``g \in G``, denoted by ``p^0_{gt}``. The active power generations in consecutive time periods are constrained by the generator's ramping capacity, which can be modeled as follows:
```math
-r_g \leq p^0_{g,t-1} - p^0_{g,t} \leq r_g \qquad \forall g \in G, \; \forall t \in T \setminus \{1\}.
```

Here, ``r_g`` denotes the ramping capacity of generator ``g`` (per unit of time in which ``T`` is defined).

For numerical convergence reasons, `ProxAL` implements the ramping constraint by introducing additional continuous variables ``s_{g,t}`` and ``z_{g,t}`` along with the following constraints. Note that a penalty term ``θ_t \|z\|^2`` is also added to the objective function, where the parameter ``θ_t`` is controlled within `ProxAL`, see [Algorithm parameters](@ref).
```math
\left.\begin{aligned}
    0 \leq s_{g,t} \leq 2r_g  \\
    p^0_{g,t-1} - p^0_{g,t} + s_{g,t} + z_{g,t} = r_g
\end{aligned}\right\} \qquad \forall g \in G, \; \forall t \in T \setminus \{1\}.
```

For convenience, `ProxAL` also provides the functionality to solve the full/"non-decomposed" model using `JuMP`/`Ipopt`. In this case, one can switch between the `inequality` and `penalty` forms of the ramping constraint by setting the `time_link_constr_type` field of `ProxAL.ModelParams` in [Model parameters](@ref). When solving the full/"non-decomposed" model with the `penalty` form of the ramping constraints, the user must provide a value for the parameter ``θ_t`` by setting `θ_t` in [Algorithm parameters](@ref).

## Contingency constraints

Each single-period ACOPF problem may itself be constrained further by a set of transmission line contingencies, denoted by ``K``. The active and reactive power generations, and bus voltages must satisfy the following constraints in each time period and each contingency:
1. the power flow equations, 
2. bounds on active and reactive generation and voltage magnitudes, and 
3. line power flow limits.

It is possible that the problem parameters are such that (some of) the above constraints can become infeasible. To model this, `ProxAL` also allows constraint infeasibility (except on variable bounds) by penalizing them in the objective function.

The contingencies in each time period are linked together via their active power generations in one of several forms. The choice of the form can be set using the `ctgs_link_constr_type` field of `ProxAL.ModelParams` in [Model parameters](@ref).

* _Preventive mode:_ active power generation in contingency ``k`` must be equal to the base case value. This constraint has one of two forms:  

    * _Preventive equality:_ This is the original form of the constraint. For numerical convergence reasons, `ProxAL` does not allow using this form whenever the `decompCtgs` field of `ProxAL.AlgParams` is set to `true`,  see [Algorithm parameters](@ref).
  ```math
  p_{gt}^k = p_{gt}^0 \qquad \forall g \in G, \; \forall k \in K, \; \forall t \in T.
  ```   
  
    * _Preventive penalty:_ In this form, `ProxAL` introduces additional continuous variables ``z_{g,k,t}`` along with the following constraints. Note that a penalty term ``θ_c \|z_k \|^2`` is also added to the objective function, where the parameter ``θ_c`` is controlled within `ProxAL` whenever the `decompCtgs` field of `ProxAL.AlgParams` is set to `true`. Otherwise, its value can be set using the `θ_c` field of `ProxAL.AlgParams` in [Algorithm parameters](@ref).
  ```math
  p_{gt}^k = p_{gt}^0 + z_{gkt} \qquad \forall g \in G, \; \forall k \in K, \; \forall t \in T.
  ```

* _Corrective mode:_ active power generation is allowed to deviate from base case by up to ``\Delta`` fraction of its ramping capacity. The parameter ``\Delta`` can be set using the `corr_scale` field of `ProxAL.ModelParams` in [Model parameters](@ref). This constraint has one of two forms:   

    * _Corrective inequality:_ This is the original form of the constraint. For numerical convergence reasons, `ProxAL` does not allow using this form whenever the `decompCtgs` field of `ProxAL.AlgParams` is set to `true`, see [Algorithm parameters](@ref).
  ```math
  0.1\, r_g \leq p_{gt}^k - p_{gt}^0 \leq \Delta \, r_g \qquad \forall g \in G, \; \forall k \in K, \; \forall t \in T
  ```   

    * _Corrective equality:_ In this form, `ProxAL` introduces additional continuous variables ``s_{g,k,t}`` along with the following constraints. As before, `ProxAL` does not allow using this form whenever the `decompCtgs` field of `ProxAL.AlgParams` is set to `true`.
  ```math
  \left.\begin{aligned}
    0 \leq s_{gkt} \leq 2 \Delta \, r_g  \\
    p_{gt}^0 - p_{gt}^k + s_{gkt} = \Delta \, r_g 
    \end{aligned}\right\} \qquad \forall g \in G, \; \forall k \in K, \; \forall t \in T
  ```   

    * _Corrective penalty:_ In this form, `ProxAL` introduces additional continuous variables ``s_{g,k,t}`` and ``z_{g,k,t}`` along with the following constraints. A penalty term ``θ_c \|z_k  \|^2`` is also added to the objective function, where the parameter ``θ_c`` is controlled within `ProxAL` whenever the `decompCtgs` field of `ProxAL.AlgParams` is set to `true`. Otherwise, its value can be set using the `θ_c` field of `ProxAL.AlgParams` in [Algorithm parameters](@ref).
  ```math
  \left.\begin{aligned}
      0 \leq s_{gkt} \leq 2 \Delta \, r_g  \\
      p_{gt}^0 - p_{gt}^k + s_{gkt} + z_{gkt} = \Delta \, r_g
  \end{aligned}\right\} \qquad \forall g \in G, \; \forall k \in K, \; \forall t \in T
  ```

* _Frequency control mode:_ In this case, `ProxAL` defines new continuous variables ``\omega_{kt}`` which is the (deviation from nominal) system frequency in contingency ``k`` of time period ``t``, and ``\alpha_g`` is the droop control parameter of generator ``g``. The objective functions includes an additional term ``w_\omega \| \omega \|^2``, where the parameter ``w_\omega`` must be set using the `weight_freq_ctrl` field of `ProxAL.ModelParams` in [Model parameters](@ref). This constraint has one of two forms:  

    * _Frequency equality:_ This is the original form of the constraint. For numerical convergence reasons, `ProxAL` does not allow using this form whenever the `decompCtgs` field of `ProxAL.AlgParams` is set to `true`,  see [Algorithm parameters](@ref).
  ```math
  p_{gt}^k = p_{gt}^0 + \alpha_g \omega_{kt} \qquad \forall g \in G, \; \forall k \in K, \; \forall t \in T.
  ```
  
    * _Frequency penalty:_ In this form, `ProxAL` introduces additional continuous variables ``z_{g,k,t}`` along with the following constraints. Note that a penalty term ``θ_c \|z_k \|^2`` is also added to the objective function, where the parameter ``θ_c`` is controlled within `ProxAL` whenever the `decompCtgs` field of `ProxAL.AlgParams` is set to `true`. Otherwise, its value can be set using the `θ_c` field of `ProxAL.AlgParams` in [Algorithm parameters](@ref).
  ```math
  p_{gt}^k = p_{gt}^0 + \alpha_g \omega_{kt} + z_{gkt} \qquad \forall g \in G, \; \forall k \in K, \; \forall t \in T.
  ```
