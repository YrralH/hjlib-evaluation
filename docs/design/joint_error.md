# Joint-error leaves

`joint_error.py` owns method-neutral unreduced per-joint error primitives. It
does not own a dataset population, joint subset, metric profile, reducer, or
unit conversion.

`compute_joint_position_errors` returns Euclidean error for equal
`(..., J, 3)` arrays. `compute_joint_height_errors` returns the absolute error
component along a normalized ground normal. The normal can be shared `(3,)` or
match the points' leading positions as `(..., 3)`; it is normalized internally,
so equivalent plane scale and sign do not affect the result. Non-finite normals
and norms at or below `1e-12` fail.

Consumers own finite-point policy and all reduction denominators. This lets an
experiment compose the leaves without moving experiment names or population
semantics into `hjlib-evaluation`.

## Extension boundary

Extend this module only for another method-neutral, unreduced joint-error leaf
whose input contract ends in `(J, 3)`. Add dataset populations, joint subsets,
unit conversion, association and reducers in their protocol owner instead. A new
sibling leaf must state shapes and invariants here, export only its public
function, and add synthetic shape/value/error-path smoke coverage.
