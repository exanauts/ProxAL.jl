# Formulation
`ProxAL` is designed to solve AC Optimal Power Flow (ACOPF) formulations over multiple time periods. 


## Time coupling

Each time period ``t \in T`` involves the solution of an ACOPF with active ``p_{dt}`` and reactive ``q_{dt}`` load forecasts, which may differ from one time period to the next. In each time period ``t \in T``, we must determine the 'base-case' active power generation level of generator ``g \in G``, denoted by ``p^0_{gt}``. The active power generations in consecutive time periods are constrained by the generator's ramping capacity, which can be modeled as follows:
```math
-r_g \leq p^0_{g,t-1} - p^0_{g,t} \leq r_g \qquad \forall g \in G, \; \forall t \in T \setminus \{1\}.
```

Here, ``r_g`` denotes the ramping capacity of generator ``g`` (per unit of time in which ``T`` is defined).

For numerical convergence reasons, `ProxAL` can implement the ramping constraint in one of several different forms:

* _Inequality:_ This form is exactly the same as above:
```math
-r_g \leq p^0_{g,t-1} - p^0_{g,t} \leq r_g \qquad \forall g \in G, \; \forall t \in T \setminus \{1\}.
```

* _Equality:_ In this form, `ProxAL` introduces additional continuous variables ``s_{g,t}`` along with the following constraints:
```math
\left.\begin{aligned}
    0 \leq s_{g,t} \leq 2r_g  \\
    p^0_{g,t-1} - p^0_{g,t} + s_{g,t} = r_g
\end{aligned}\right\} \qquad \forall g \in G, \; \forall t \in T \setminus \{1\}.
```

* _Penalty:_ In this form, `ProxAL` introduces additional continuous variables ``s_{g,t}`` and ``z_{g,t}`` along with the following constraints. Note that a penalty term ``θ_t \|z\|^2`` is also added to the objective function, where the parameter ``θ_t`` is controlled within `ProxAL`, see [Algorithm parameters](@ref).
```math
\left.\begin{aligned}
    0 \leq s_{g,t} \leq 2r_g  \\
    p^0_{g,t-1} - p^0_{g,t} + s_{g,t} + z_{g,t} = r_g
\end{aligned}\right\} \qquad \forall g \in G, \; \forall t \in T \setminus \{1\}.
```

Which of the above the forms to use can be set using the `time_link_constr_type` field of `ProxAL.ModelParams` in [Model parameters](@ref).

## Contingency constraints

Each single-period ACOPF problem may itself be constrained further by a set of transmission line contingencies, denoted by ``K``. The active and reactive power generations, and bus voltages must satisfy the following constraints in each time period and each contingency:
1. the power flow equations, 
2. bounds on active and reactive generation and voltage magnitudes, and 
3. line power flow limits.

It is possible that the problem parameters are such that (some of) the above constraints can become infeasible. To model this, `ProxAL` also allows constraint infeasibility (except variable bounds) by penalizing them in the objective function.

The contingencies in each time period are linked together via their active power generations in one of several forms. The choice of the form can be set using the `ctgs_link_constr_type` field of `ProxAL.ModelParams` in [Model parameters](@ref).

* _Preventive mode:_ active power generation in contingency ``k`` must be equal to the base case value. This constraint can be implemented in one of two forms:  

    * _Preventive equality:_ This can be modeled as an equality:
  ```math
  p_{gt}^k = p_{gt}^0 \qquad \forall g \in G, \; \forall k \in K, \; \forall t \in T.
  ```   
  
    * _Preventive penalty:_ In this form, `ProxAL` introduces additional continuous variables ``z_{g,k,t}`` along with the following constraints. Note that a penalty term ``w_k \|z_k \|^2`` is also added to the objective function, where the parameter ``w_k`` can be set using the `weight_quadratic_penalty_ctgs` field of `ProxAL.ModelParams` in [Model parameters](@ref).
  ```math
  p_{gt}^k = p_{gt}^0 + z_{gkt} \qquad \forall g \in G, \; \forall k \in K, \; \forall t \in T.
  ```

* _Corrective mode:_ active power generation is allowed to deviate from base case by up to 10% of its ramping capacity. This constraint can be implemented in one of two forms:   

    * _Corrective inequality:_ This can be modeled as an equality:
  ```math
  0.1\, r_g \leq p_{gt}^k - p_{gt}^0 \leq 0.1 \, r_g \qquad \forall g \in G, \; \forall k \in K, \; \forall t \in T
  ```   

    * _Corrective equality:_ In this form, `ProxAL` introduces additional continuous variables ``s_{g,k,t}`` along with the following constraints:
  ```math
  \left.\begin{aligned}
    0 \leq s_{gkt} \leq 0.2\, r_g  \\
    p_{gt}^0 - p_{gt}^k + s_{gkt} = 0.1 \, r_g 
    \end{aligned}\right\} \qquad \forall g \in G, \; \forall k \in K, \; \forall t \in T
  ```   

    * _Corrective penalty:_ In this form, `ProxAL` introduces additional continuous variables ``s_{g,k,t}`` and ``z_{g,k,t}`` along with the following constraints. A penalty term ``θ_c \|z_k  \|^2`` is also added to the objective function, where the parameter ``θ_c`` is controlled within `ProxAL`, see [Algorithm parameters](@ref).
  ```math
  \left.\begin{aligned}
      0 \leq s_{gkt} \leq 0.2\, r_g  \\
      p_{gt}^0 - p_{gt}^k + s_{gkt} + z_{gkt} = 0.1 \, r_g
  \end{aligned}\right\} \qquad \forall g \in G, \; \forall k \in K, \; \forall t \in T
  ```

* _Frequency control mode:_ In this case, `ProxAL` defines new continuous variables ``\omega_{kt}`` which is the (deviation from nominal) system frequency in contingency ``k`` of time period ``t``, and ``\alpha_g`` is the droop control parameter of generator ``g``. There is only one form of representing this constraint. The objective functions includes an additional term ``w_\omega \| \omega \|^2``, where the parameter ``w_\omega`` can be set using the `weight_freq_ctrl` field of `ProxAL.ModelParams` in [Model parameters](@ref).
```math
p_{gt}^k = p_{gt}^0 + \alpha_g \omega_{kt} \qquad \forall g \in G, \; \forall k \in K, \; \forall t \in T.
```
