"""Opt-in startup shim; install with this directory first on PYTHONPATH."""

import os

if os.environ.get('DSV41_MARLIN_CANONICAL_ORDER') == '1':
    from marlin_order_diagnostic import install_import_hook as install_marlin_order
    install_marlin_order()

if os.environ.get('DSV41_STABLE_TOPK_DIAGNOSTIC') == '1':
    from stable_topk_diagnostic import install_import_hook as install_stable_topk
    install_stable_topk()

if os.environ.get('DSV41_PREFILL_PROBE') == '1':
    from prefill_order_probe import install_import_hook as install_prefill_probe
    install_prefill_probe()

if os.environ.get('DSV41_SPARSE_ORDER') in ('original', 'canonical'):
    from sparse_order_hooks import install_import_hook as install_sparse_order_hook
    install_sparse_order_hook()

if os.environ.get('DSV41_FAST_ENABLED') == '1':
    from runtime_hooks import install_import_hook
    install_import_hook()

if os.environ.get('DSV41_PRECISION_HOOK') == '1':
    from precision_hooks import install_import_hook as install_precision_import_hook
    install_precision_import_hook()
