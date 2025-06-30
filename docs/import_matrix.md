# Import Matrix for Enhanced CSP Migration

The following table maps legacy modules and entry points to their new locations
under the `enhanced_csp` package. All old APIs should import from the new
modules to ensure compatibility.

| Legacy Path | New Module |
|-------------|------------|
| `security_hardening.py` | `enhanced_csp.security_hardening` |
| `realtime_csp_visualizer.py` | `enhanced_csp.realtime_csp_visualizer` |
| `quantum_csp_engine.py` | `enhanced_csp.quantum_csp_engine` |
| `performance_optimization.py` | `enhanced_csp.performance_optimization` |
| `multimodal_ai_hub.py` | `enhanced_csp.multimodal_ai_hub` |
| `neural_csp_optimizer.py` | `enhanced_csp.neural_csp_optimizer` |
| `blockchain_csp_network.py` | `enhanced_csp.blockchain_csp_network` |
| `autonomous_system_controller.py` | `enhanced_csp.autonomous_system_controller` |
| `core_csp_implementation.py` | `enhanced_csp.core_csp_implementation` |
| `advanced_security_engine.py` | `enhanced_csp.advanced_security_engine` |
| `ai_extensions/csp_ai_extensions.py` | `enhanced_csp.ai_extensions.csp_ai_extensions` |
| `ai_integration/csp_ai_integration.py` | `enhanced_csp.ai_integration.csp_ai_integration` |
| `core/advanced_csp_core.py` | `enhanced_csp.core.advanced_csp_core` |

## Deprecations

Imports from the old package roots are deprecated. Use the fully qualified paths
above. Any missing features should be ported into the corresponding module under
`enhanced_csp`. Legacy wrappers may be kept temporarily but should emit a
`DeprecationWarning`.
