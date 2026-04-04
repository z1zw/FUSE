from __future__ import annotations

import sys

from .branding import ACCELERATOR_NAME

try:
    from .ir import ContextKey, Expr, Fragment, MicroFragment, TaskGraph
except Exception:
    ContextKey = None
    Expr = None
    Fragment = None
    MicroFragment = None
    TaskGraph = None
try:
    from .benchmarks import BenchmarkSuite, run_default_suite
    from .cloud_suite import CloudExperimentSuite
    from .hardware import HardwareBenchmarkSuite
    from .scenarios import ScenarioSuite
    from .sosp import SOSPSuite, run_sosp_suite
    from .training import AlpacaTrainingSuite, CIFARTrainingSuite
    from .sim.simulator import FuseSimulator
    from .llm import FuseVerifier
except Exception:
    BenchmarkSuite = None
    run_default_suite = None
    CloudExperimentSuite = None
    HardwareBenchmarkSuite = None
    ScenarioSuite = None
    SOSPSuite = None
    run_sosp_suite = None
    AlpacaTrainingSuite = None
    CIFARTrainingSuite = None
    FuseSimulator = None
    FuseVerifier = None

FUSESimulator = FuseSimulator
FUSEVerifier = FuseVerifier
sys.modules.setdefault('FUSE', sys.modules[__name__])
__all__ = ['ContextKey', 'Expr', 'Fragment', 'MicroFragment', 'TaskGraph', 'ACCELERATOR_NAME', 'BenchmarkSuite', 'CloudExperimentSuite', 'HardwareBenchmarkSuite', 'CIFARTrainingSuite', 'AlpacaTrainingSuite', 'ScenarioSuite', 'SOSPSuite', 'FuseSimulator', 'FUSESimulator', 'FuseVerifier', 'FUSEVerifier', 'run_default_suite', 'run_sosp_suite']
