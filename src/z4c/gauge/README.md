# Z4c gauge policies

Gauge evolution is kept separate from the Z4c field equations, just as the
spacetime geometry interface is kept separate from GRMHD.  The input selects
the lapse and shift policies independently:

| Input value | Lapse | Shift |
|---|---:|---:|
| `legacy` | historical coefficient-based equation | historical coefficient-based equation |
| `frozen` (`geodesic`) | zero RHS | — |
| `frozen` (`zero`) | — | zero RHS |
| `harmonic` | harmonic slicing | harmonic shift |
| `one_plus_log` | moving-puncture 1+log slicing | — |
| `gamma_driver` | — | moving-puncture Gamma driver |
| `shifted_gauge_wave` | analytic AwA shifted-wave lapse | analytic AwA shifted-wave shift |

Production puncture inputs use `one_plus_log` plus `gamma_driver`.  Ordinary
AwA gauge-wave and Gowdy inputs use harmonic lapse plus frozen shift.  The AwA
shifted gauge wave uses its analytic lapse and shift pair.  `legacy` remains
the default so existing input files retain their old equations exactly.

`dispatch.hpp` is the only entry used by the Z4c RHS.  Individual formulas live
in `frozen.hpp`, `harmonic.hpp`, `moving_puncture.hpp`, and
`shifted_gauge_wave.hpp`; new gauge families should be added here rather than
embedded in `z4c/evolution/z4c_rhs.h`.
